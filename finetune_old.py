import csv
import os
import sys

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from dataset import parse_task
from engine import fit
from finetune import (
    DIMS,
    build_finetune_model,
    load_task_dicts,
    summarize,
)
from metrics import evaluate
from models import build_model, get_device

"""
====================================================================
Per-patient fine-tuning for LEGACY checkpoints (pre normalization unification)
====================================================================

Same fine-tuning procedure as finetune.py (per-patient chronological split,
frozen variance head, two LR groups, L2-SP toward the pretrained weights,
model selection on val FD-gain), but with the legacy feature pipeline these
old checkpoints were trained under (see evaluate_old.py for the full rationale):

  * positions are normalized with a length-6 mu/sigma (positions only),
  * velocity/acceleration channels are normalized with the per-task
    (vel_std, acc_std) stored in the checkpoint's `feat_std` dict,

instead of the single length-input_dim mu/sigma z-score of the current
dataset.py. The per-task feat_std scales are the pretraining population stats,
so reusing them (rather than recomputing per patient) keeps normalization fixed
and avoids any per-patient rescaling. Everything downstream — engine.fit
(mu[:6]/sigma[:6] slicing is a no-op on length-6 stats), metrics.evaluate, the
L2-SP setup, and the CSV/NPZ outputs — is identical to finetune.py.

Usage: python finetune_old.py [configs/finetune_old.yaml]
"""


class OldTimeSeriesDataset(Dataset):
    """Legacy feature pipeline as a torch Dataset (copied from dataset.py @ 623613c^).

    Expects positions that are ALREADY normalized (data = (raw - mu)/sigma with
    the length-6 mu/sigma). Appends step-2 velocity and second-difference
    acceleration and normalizes each by the per-task vel_std/acc_std saved in the
    checkpoint's feat_std. No augmentation (fine-tuning does not augment).
    """

    def __init__(
        self,
        data,
        ids,
        sequence_length,
        device,
        add_velocity,
        add_acceleration,
        vel_std,
        acc_std,
    ):
        self.data = torch.from_numpy(data).to(device=device, dtype=torch.float32)
        self.ids = ids
        if add_velocity or add_acceleration:
            # [N, T, 12] ( if vel) or [N, T, 18] (if vel+acc).
            # Differences are taken with step 2 (x[t]-x[t-2]),
            # so they only use frames that the model sees, not the
            # extra frames in between.
            pos = self.data  # [N, T, 6]
            extra = []
            if add_velocity:
                # Compute vel such that v[0] = x[2] - x[0]
                vel = pos[:, 2:, :] - pos[:, :-2, :]  # [N, T-2, 6]
                # Pad 2 zeros, now v[0] = 0, v[1]= 0, v[2] = x[2] - x[0]
                vel = torch.cat([torch.zeros_like(vel[:, :2, :]), vel], dim=1)
                extra.append(vel / vel_std.to(vel.device))
            if add_acceleration:
                # second difference: a[t] = (x[t] - x[t-2]) - (x[t-2] - x[t-4])
                acc = pos[:, 4:, :] - 2 * pos[:, 2:-2, :] + pos[:, :-4, :]  # [N, T-4, 6]
                acc = torch.cat([torch.zeros_like(acc[:, :4, :]), acc], dim=1)
                extra.append(acc / acc_std.to(acc.device))
            self.data = torch.cat([pos] + extra, dim=2)  # [N, T, 6/12/18]
        self.time_span = sequence_length * 2
        self.N, self.T, self.D = self.data.shape

    def __len__(self):
        return self.N * (self.T - self.time_span + 1)

    def __getitem__(self, index):
        p = index // (self.T - self.time_span + 1)  # Patient index
        t = index % (self.T - self.time_span + 1)  # Time index
        x = self.data[p, t : t + self.time_span : 2, :]  # Sub-sequence
        y = self.data[p, t + self.time_span - 1, :6]  # Next time step (6 positions only)
        return self.ids[p], x, y


def build_patient_split(
    series, patient_id, sequence_length, batch_size, split_percentages, device,
    mu, sigma, vel_std, acc_std, add_velocity=False, add_acceleration=False,
):
    """Chronological train/val/test split of one patient's raw series (legacy norm).

    Splits raw frames into three disjoint arrays before windowing, so no window
    straddles a boundary — no leakage across splits. Positions are normalized with
    the length-6 pretraining mu/sigma; velocity/acceleration with the pretraining
    per-task vel_std/acc_std.
    """
    total_len = series.shape[0]
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
        "train": (series[:train_len], True),
        "val": (series[train_len : train_len + val_len], False),
        "test": (series[train_len + val_len :], False),
    }
    loaders, sizes = {}, {}
    for name, (s, shuffle) in raw_splits.items():
        # legacy normalization: positions by length-6 mu/sigma before windowing
        s_norm = (s - mu) / sigma
        ds = OldTimeSeriesDataset(
            s_norm[None, :, :], [patient_id], sequence_length=sequence_length,
            device=device, add_velocity=add_velocity, add_acceleration=add_acceleration,
            vel_std=vel_std, acc_std=acc_std,
        )
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        sizes[name] = len(ds)
    return loaders, sizes


def finetune_patient(patient_id, task, pretrained, task_dicts, cfg, device):
    """Fine-tune the pretrained (legacy) model on one patient and compare before/after.

    Trains on the early part of the patient's series, selects the best model on
    the middle part, and reports metrics on the later part — for both the
    pretrained generalist ("before") and the fine-tuned model ("after").
    """
    model_config = pretrained["config"]["model"]
    data_config = pretrained["config"]["data"]
    mu, sigma = pretrained["mu"], pretrained["sigma"]  # length-6 (positions only)
    vel_std, acc_std = pretrained["feat_std"][task]  # per-task feature scales

    # raw positions; the split normalizes with the pretraining population stats
    # (mu/sigma for positions, feat_std for velocity/acceleration) — no per-patient leakage
    series = task_dicts[task][patient_id]
    # window length and feature flags must match what the checkpoint was trained
    # with; the checkpoint's own config wins over the finetune config
    seq_len = data_config.get("sequence_length", cfg.get("sequence_length", 10))
    add_velocity = data_config.get("add_velocity", False)
    add_acceleration = data_config.get("add_acceleration", False)
    loaders, sizes = build_patient_split(
        series, patient_id, seq_len, cfg["batch_size"], cfg["split_percentages"], device,
        mu, sigma, vel_std, acc_std,
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

    # Usage: python finetune_old.py [configs/finetune_old.yaml]
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/finetune_old.yaml"
    cfg = yaml.safe_load(open(config_path))
    cfg["split_percentages"] = tuple(cfg["split_percentages"])

    device = get_device()
    print(f"config: {config_path}")

    pretrained = torch.load(cfg["pretrained"], map_location=device, weights_only=False)
    assert "feat_std" in pretrained, (
        "finetune_old.py expects a legacy checkpoint with a `feat_std` dict; this "
        "checkpoint has none -> use finetune.py (unified normalization) instead."
    )
    task_dicts = load_task_dicts()

    # Held-out patients (never seen during pretraining)
    test_ids = [str(i) for i in pretrained["test_ids"]]

    results_dir = "results/finetune"
    os.makedirs(results_dir, exist_ok=True)
    tag = f"{cfg['tasks']}_old"
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
