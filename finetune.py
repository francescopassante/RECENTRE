import csv
import os
import sys

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from dataset import TimeSeriesDataset, parse_task
from engine import fit
from metrics import evaluate
from models import build_model, get_device

# Dimension order is fixed: [Tx, Ty, Tz, Rx, Ry, Rz] (rotations are 3:6).
DIMS = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]


def load_task_dicts():
    """Load the per-task {patient_id: ndarray[T, 6]} dicts. Keys are strings."""
    return {
        t: np.load(f"datasets/{t}_dict.npy", allow_pickle=True).item()
        for t in ("R", "M", "L")
    }


def summarize(out, sigma):
    """Turn one evaluate() output into the scalar metrics we record per split."""
    fb, fp = out["fd_base"], out["fd_pred"]
    err = out["pred"] - out["true"]  # physical units, no ×50
    return {
        # normalized-space MSE (matches the MSE the loss is trained on)
        "mse": ((err / np.array(sigma)) ** 2).mean(),
        "fdg": ((fb - fp) / (fb + 1e-6)).mean(),
        "fd_pred": fp.mean(),
        "fd_base": fb.mean(),
        # fraction of frames where the model beats the previous-frame baseline
        "pct_improved": (fp < fb).mean() * 100,
        # per-dimension squared error in physical units
        "mse_per_dim": (err**2).mean(axis=0),
    }


def build_patient_split(
    series_norm, patient_id, sequence_length, batch_size, split_percentages, device,
    add_velocity=False, add_acceleration=False,
):
    """Chronological train/val/test split of one patient's normalized series.

    Splits raw frames into three disjoint arrays before windowing, so no
    window straddles a boundary — no leakage across splits.
    """
    total_len = series_norm.shape[0]
    train_frac, val_frac, _ = split_percentages
    train_len = int(round(train_frac * total_len))
    val_len = int(round(val_frac * total_len))

    time_span = sequence_length * 2
    split_lengths = {
        "train": train_len,
        "val": val_len,
        "test": total_len - train_len - val_len,
    }
    too_short = [n for n, l in split_lengths.items() if l < time_span]
    if too_short:
        raise ValueError(
            f"Patient {patient_id}: splits {too_short} are shorter than "
            f"time_span={time_span} (sequence_length={sequence_length}×2). "
            f"Total frames={total_len}, split lengths={split_lengths}."
        )

    raw_splits = {
        "train": (series_norm[:train_len], True),
        "val": (series_norm[train_len : train_len + val_len], False),
        "test": (series_norm[train_len + val_len :], False),
    }
    loaders, sizes = {}, {}
    vel_std = acc_std = None
    for name, (s, shuffle) in raw_splits.items():
        ds = TimeSeriesDataset(
            s[None, :, :], [patient_id], sequence_length=sequence_length, device=device,
            add_velocity=add_velocity, add_acceleration=add_acceleration,
            vel_std=vel_std, acc_std=acc_std,
        )
        # propagate scales computed on train to val/test so they match
        if name == "train":
            vel_std, acc_std = ds.vel_std, ds.acc_std
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        sizes[name] = len(ds)
    return loaders, sizes


def split_finetune_params(model):
    """Split params into (body, head) and freeze the variance head.

    Shared convention across gru/tcn/transformer/conformer/mamba: any param with
    'logvar' is the variance head; the MLP mean head is fc1/bn_fc1/fc_mean;
    everything else is the backbone. The body trains at the smaller lr_body, the
    head at the larger lr_head.
    """
    body, head = [], []
    for name, p in model.named_parameters():
        if "logvar" in name:
            p.requires_grad = False  # freeze the variance head
        elif name.startswith(("fc1", "bn_fc1", "fc_mean")):
            head.append(p)  # MLP head -> lr_head
        else:
            body.append(p)  # backbone -> lr_body
    return body, head


def build_finetune_model(pretrained_state, model_config, cfg, device):
    """Fresh pretrained model set up for fine-tuning: frozen variance head, two
    LR groups, reduced dropout, plus the L2-SP reference snapshot (theta_0)."""
    model = build_model(model_config).to(device)
    model.load_state_dict(pretrained_state)

    # Snapshot the pretrained weights as the L2-SP reference (theta_0)
    reference = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }

    body_params, head_params = split_finetune_params(model)

    # Reduce dropout during fine-tuning — the per-patient set is tiny
    model.dp.p = cfg["dropout_ft"]

    # weight_decay=0: L2-SP already regularizes toward the pretrained weights,
    # so an extra decay toward zero would pull against it.
    optimizer = torch.optim.Adam(
        [
            {"params": body_params, "lr": cfg["lr_body"]},
            {"params": head_params, "lr": cfg["lr_head"]},
        ],
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    return model, optimizer, scheduler, reference


def finetune_patient(patient_id, task, pretrained, task_dicts, cfg, device):
    """Fine-tune the pretrained model on one patient and compare before/after.

    Trains on the early part of the patient's series, selects the best model on
    the middle part, and reports metrics on the later part — for both the
    pretrained generalist ("before") and the fine-tuned model ("after").
    """
    model_config = pretrained["config"]["model"]
    data_config = pretrained["config"]["data"]
    mu, sigma = pretrained["mu"], pretrained["sigma"]

    # normalize with the pretraining-population stats (no per-patient leakage)
    series = (task_dicts[task][patient_id] - mu) / sigma
    # window length and feature flags must match what the checkpoint was trained
    # with; the checkpoint's own config wins over the finetune config
    seq_len = data_config.get("sequence_length", cfg.get("sequence_length", 10))
    add_velocity = data_config.get("add_velocity", False)
    add_acceleration = data_config.get("add_acceleration", False)
    loaders, sizes = build_patient_split(
        series, patient_id, seq_len, cfg["batch_size"], cfg["split_percentages"], device,
        add_velocity=add_velocity, add_acceleration=add_acceleration,
    )

    # BEFORE: the pretrained generalist on this patient's test split
    pre_model = build_model(model_config).to(device)
    pre_model.load_state_dict(pretrained["model_state"])
    before = summarize(evaluate(pre_model, loaders["test"], mu, sigma, device), sigma)

    # AFTER: fine-tune a fresh copy, then evaluate the best checkpoint
    model, optimizer, scheduler, reference = build_finetune_model(
        pretrained["model_state"], model_config, cfg, device
    )
    best_state, best_epoch = fit(
        model,
        loaders["train"],
        loaders["val"],
        optimizer,
        scheduler,
        device,
        epochs=cfg["epochs"],
        mu=mu,
        sigma=sigma,
        loss="mse",
        beta=cfg["beta"],
        patience=cfg["patience"],
        reference=reference,
        lambda_l2sp=cfg["lambda_l2sp"],
        verbose=False,
    )
    model.load_state_dict(best_state)
    # keep the full per-frame arrays (not just the scalar summary): these are the
    # fine-tuned model's predictions on this patient's held-out test frames, which
    # the pooled .npz collects so plots.py can characterize the fine-tuned collection.
    after_out = evaluate(model, loaders["test"], mu, sigma, device)
    after = summarize(after_out, sigma)

    row = {
        "patient_id": str(patient_id),
        "task": task,
        "n_train": sizes["train"],
        "n_val": sizes["val"],
        "n_test": sizes["test"],
        "best_epoch": best_epoch,
        # baseline FD does not depend on the model, so it is shared before/after
        "fd_base": before["fd_base"],
        "fdg_before": before["fdg"],
        "fdg_after": after["fdg"],
        "delta_fdg": after["fdg"] - before["fdg"],
        "fd_pred_before": before["fd_pred"],
        "fd_pred_after": after["fd_pred"],
        "mse_before": before["mse"],
        "mse_after": after["mse"],
        "pct_improved_before": before["pct_improved"],
        "pct_improved_after": after["pct_improved"],
    }
    for i, d in enumerate(DIMS):
        row[f"mse_{d}_before"] = before["mse_per_dim"][i]
        row[f"mse_{d}_after"] = after["mse_per_dim"][i]
    return row, after_out


if __name__ == "__main__":
    import yaml

    # Usage: python finetune.py [configs/finetune.yaml]
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/finetune.yaml"
    cfg = yaml.safe_load(open(config_path))
    cfg["split_percentages"] = tuple(cfg["split_percentages"])

    device = get_device()
    print(f"config: {config_path}")

    pretrained = torch.load(cfg["pretrained"], map_location=device, weights_only=False)
    task_dicts = load_task_dicts()

    # Held-out patients (never seen during pretraining)
    test_ids = [str(i) for i in pretrained["test_ids"]]

    results_dir = "results/finetune"
    os.makedirs(results_dir, exist_ok=True)
    tag = cfg["tasks"]
    csv_path = os.path.join(results_dir, f"ft_{tag}_before_after.csv")
    npz_path = os.path.join(results_dir, f"ft_{tag}_arrays.npz")

    # One row per (task, patient); the row records its task, so all tasks share one CSV.
    # pooled holds every patient's fine-tuned test-frame arrays, concatenated across
    # the whole sweep (each patient scored by its own fine-tuned model) — this is the
    # frame-level dataset that represents the fine-tuned collection for plots.py.
    rows = []
    pooled = {
        k: []
        for k in (
            "pred",
            "true",
            "base",
            "std",
            "z",
            "fd_pred",
            "fd_base",
            "ids",
            "task",
        )
    }
    for task in parse_task(cfg["tasks"]):
        task_dict = task_dicts[task]
        for patient_id in tqdm.tqdm(test_ids, desc=f"Finetuning ({task})"):
            row, after_out = finetune_patient(
                patient_id, task, pretrained, task_dicts, cfg, device
            )
            rows.append(row)
            for k in ("pred", "true", "base", "std", "z", "fd_pred", "fd_base", "ids"):
                pooled[k].append(after_out[k])
            # one task label per frame, aligned with the arrays above
            pooled["task"].append(np.full(len(after_out["fd_pred"]), task))

    # column order: identifiers first, then the comparison metrics
    fieldnames = [
        "patient_id",
        "task",
        "n_train",
        "n_val",
        "n_test",
        "best_epoch",
        "fd_base",
        "fdg_before",
        "fdg_after",
        "delta_fdg",
        "fd_pred_before",
        "fd_pred_after",
        "mse_before",
        "mse_after",
        "pct_improved_before",
        "pct_improved_after",
    ]
    for d in DIMS:
        fieldnames += [f"mse_{d}_before", f"mse_{d}_after"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-patient before/after metrics -> {csv_path}")

    # pooled frame-level arrays for the fine-tuned collection (feeds finetune_plots.py)
    np.savez(npz_path, **{k: np.concatenate(v) for k, v in pooled.items()})
    print(f"Saved pooled arrays -> {npz_path}")
