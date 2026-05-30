import os

import numpy as np

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from GRU import GRUModel
from metrics import fd, fd_gain
from TimeSeriesDataset import TimeSeriesDataset

# Dimension order is fixed: [Tx, Ty, Tz, Rx, Ry, Rz] (rotations are 3:6).
DIMS = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]

# Default per-patient fine-tuning hyperparameters. Both the single-patient
# __main__ debug run and finetune_all.py build on top of this.
DEFAULT_CONFIG = {
    "model_config": {
        "input_dim": 6,
        "hidden_dim": 128,
        "output_dim": 6,
        "num_layers": 2,
        "dropout": 0.5,
    },
    "sequence_length": 10,
    "batch_size": 16384,
    # chronological (train, val, test) fractions of the patient's frames
    "split_percentages": (0.5, 0.3, 0.2),
    "beta": 0.5,
    "epochs": 30,
    "patience": 20,
    "lambda_l2sp": 1e-3,
    # small LR for the GRU recurrent layers, larger LR for the final MLP block
    "lr_gru": 1e-4,
    "lr_mlp": 3e-4,
    # reduced dropout during fine-tuning — the per-patient set is tiny
    "dropout_ft": 0.05,
}


def load_task_dicts(datasets_dir=None):
    """Load the per-task {patient_id: ndarray[T, 6]} dicts. Keys are strings."""
    if datasets_dir is None:
        datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
    return {
        t: np.load(
            os.path.join(datasets_dir, f"{t}_dict.npy"), allow_pickle=True
        ).item()
        for t in ("R", "M", "L")
    }


def l2sp_penalty(model, reference):
    # L2-SP: pull the fine-tuned weights back toward the pretrained ones (theta_0)
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        # only regularize the layers we actually fine-tune (frozen logvar excluded)
        if not param.requires_grad:
            continue
        penalty = penalty + ((param - reference[name]) ** 2).sum()
    return penalty


def finetune(
    model,
    train_loader,
    val_loader,
    train_ids,
    val_ids,
    test_ids,
    optimizer,
    criterion,
    scheduler,
    device,
    epochs,
    mu,
    sigma,
    train_task,
    test_task,
    reference,
    lambda_l2sp,
    pred_sigma,
    checkpoint_path=None,
    patience=10,
    beta=0.1,
):
    # mu/sigma as tensors on device for FD denormalization
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)

    train_loss_history = []
    val_loss_history = []

    best_val_fdg = float("-inf")
    early_counter = 0
    checkpoint_dict = {}
    pbar = tqdm.trange(epochs)
    for epoch in pbar:
        # accumulate sample-weighted MSE
        train_mse_sum = torch.zeros((), device=device)

        # train_n is the total number of samples, used to compute mean MSE.
        train_n = 0

        # store fd_baseline and fd_model for each frame
        train_fd_baselines = []
        train_fd_preds = []
        model.train()
        for _, x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred, _ = model(x)

            mse = criterion(y_pred, y)

            # get last_x to compute baseline metrics
            last_x = x[:, -1, :]
            fd_baseline = fd(last_x, y, mu_t, sigma_t)
            fd_pred = fd(y_pred, y, mu_t, sigma_t)
            gain = fd_gain(fd_baseline, fd_pred)

            loss = (
                mse - beta * gain.mean() + lambda_l2sp * l2sp_penalty(model, reference)
            )
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            train_mse_sum += mse.detach() * bs
            train_n += bs
            train_fd_baselines.append(fd_baseline.detach())
            train_fd_preds.append(fd_pred.detach())

        train_fd_baseline_cat = torch.cat(train_fd_baselines, dim=0)
        train_fd_pred_cat = torch.cat(train_fd_preds, dim=0)

        train_fdg_epoch = fd_gain(train_fd_baseline_cat, train_fd_pred_cat).mean()
        train_fd_pred_epoch = train_fd_pred_cat.mean()
        train_mse_epoch = train_mse_sum / train_n

        train_mse, train_fdg, train_fd_pred = (
            train_mse_epoch.item(),
            train_fdg_epoch.item(),
            train_fd_pred_epoch.item(),
        )
        train_loss = train_mse - beta * train_fdg

        model.eval()
        with torch.no_grad():
            val_fd_baselines = []
            val_fd_preds = []
            val_mse_sum = torch.zeros((), device=device)
            val_n = 0
            for _, x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred, _ = model(x)
                bs = y.size(0)
                val_mse_sum += criterion(y_pred, y) * bs
                val_n += bs

                last_x = x[:, -1, :]
                val_fd_baselines.append(fd(last_x, y, mu_t, sigma_t))
                val_fd_preds.append(fd(y_pred, y, mu_t, sigma_t))

            fd_baseline_cat = torch.cat(val_fd_baselines, dim=0)
            fd_model_cat = torch.cat(val_fd_preds, dim=0)
            val_mse_epoch = val_mse_sum / val_n
            val_fd_pred_epoch = fd_model_cat.mean()
            val_fdg_epoch = fd_gain(fd_baseline_cat, fd_model_cat).mean()

            # single GPU→cpu per epoch
            val_mse, val_fd_pred, val_fdg = (
                val_mse_epoch.item(),
                val_fd_pred_epoch.item(),
                val_fdg_epoch.item(),
            )
            val_loss = val_mse - beta * val_fdg

        scheduler.step(val_loss)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "train_fdg": f"{train_fdg:.4f}",
                "train_fd_pred": f"{train_fd_pred:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_fdg": f"{val_fdg:.4f}",
                "val_fd_pred": f"{val_fd_pred:.4f}",
            }
        )

        if val_fdg > best_val_fdg:
            best_val_fdg = val_fdg
            early_counter = 0
            checkpoint_dict = {
                # clone so later epochs don't mutate the stored best weights
                # (state_dict() returns references to the live parameters)
                "model_state_dict": {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                },
                "epoch": epoch + 1,
                "epochs": epochs,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
                "mu": mu,
                "sigma": sigma,
                "beta": beta,
                "lambda_l2sp": lambda_l2sp,
                "train_task": train_task,
                "test_task": test_task,
                # carry over the pretraining sigma distribution: the per-patient
                # fine-tuning set is too small to reconstruct a good one, and the
                # variance head is frozen anyway, so its sigmas are unchanged.
                "pred_sigma": pred_sigma,
            }
        else:
            early_counter += 1
            if early_counter >= patience:
                break

    if checkpoint_path is not None:
        torch.save(checkpoint_dict, checkpoint_path)
    return checkpoint_dict, train_loss_history, val_loss_history


def evaluate_loader(model, data_loader, criterion, mu, sigma, device, beta=0.1):
    """Per-split metrics for one model: FD-gain, FD, MSE, per-dim MSE, win rate."""
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        fd_baselines = []
        fd_preds = []
        mse_sum = torch.zeros((), device=device)
        # per-dimension squared error in physical (denormalized) units
        sq_err_sum = torch.zeros(6, device=device)
        n = 0
        for _, x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred, _ = model(x)
            bs = y.size(0)
            mse_sum += criterion(y_pred, y) * bs
            n += bs

            last_x = x[:, -1, :]
            fd_baselines.append(fd(last_x, y, mu_t, sigma_t))
            fd_preds.append(fd(y_pred, y, mu_t, sigma_t))

            y_pred_phys = y_pred * sigma_t + mu_t
            y_phys = y * sigma_t + mu_t
            sq_err_sum += ((y_pred_phys - y_phys) ** 2).sum(dim=0)

        fd_baseline_cat = torch.cat(fd_baselines, dim=0)
        fd_pred_cat = torch.cat(fd_preds, dim=0)
        mse_epoch = mse_sum / n
        fd_base_epoch = fd_baseline_cat.mean()
        fd_pred_epoch = fd_pred_cat.mean()
        fdg_epoch = fd_gain(fd_baseline_cat, fd_pred_cat).mean()
        # fraction of frames where the model beats the previous-frame baseline
        pct_improved = (fd_pred_cat < fd_baseline_cat).float().mean() * 100
        mse_per_dim = (sq_err_sum / n).cpu().tolist()
        loss = mse_epoch - beta * fdg_epoch

    return {
        "mse": mse_epoch.item(),
        "fdg": fdg_epoch.item(),
        "fd_pred": fd_pred_epoch.item(),
        "fd_base": fd_base_epoch.item(),
        "pct_improved": pct_improved.item(),
        "mse_per_dim": mse_per_dim,
        "loss": loss.item(),
    }


def build_patient_split(
    series_norm, patient_id, sequence_length, batch_size, split_percentages, device
):
    """Chronological train/val/test split of one patient's normalized series.

    Splits raw frames into three disjoint arrays *before* windowing, so no
    window straddles a boundary — no leakage across splits.
    """
    total_len = series_norm.shape[0]
    train_frac, val_frac, _ = split_percentages
    train_len = int(round(train_frac * total_len))
    val_len = int(round(val_frac * total_len))

    splits = {
        "train": (series_norm[:train_len], True),
        "val": (series_norm[train_len : train_len + val_len], False),
        "test": (series_norm[train_len + val_len :], False),
    }
    loaders, sizes = {}, {}
    for name, (s, shuffle) in splits.items():
        ds = TimeSeriesDataset(
            s[None, :, :], [patient_id], sequence_length=sequence_length, device=device
        )
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        sizes[name] = len(ds)
    return loaders, sizes


def build_finetune_model(pretrained, model_config, lr_gru, lr_mlp, dropout_ft, device):
    """Fresh pretrained GRU set up for fine-tuning: frozen variance head, two
    LR groups, reduced dropout, plus the L2-SP reference snapshot (theta_0)."""
    model = GRUModel(**model_config).to(device)
    model.load_state_dict(pretrained["model_state_dict"])

    # Snapshot the pretrained weights as the L2-SP reference (theta_0)
    reference = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }

    # Freeze the variance mlp (compute gradients only for other layers)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(("gru", "bn_gru", "fc1", "bn_fc1", "fc_mean"))

    # Reduce dropout during fine-tuning — the per-patient set is tiny
    model.dp.p = dropout_ft

    # Small LR for the GRU recurrent layers, larger LR for the final MLP block
    gru_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and n.startswith(("gru", "bn_gru"))
    ]
    mlp_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and n.startswith(("fc1", "bn_fc1", "fc_mean"))
    ]
    # weight_decay=0: L2-SP already regularizes toward the pretrained weights,
    # so an extra decay toward zero would pull against it.
    optimizer = torch.optim.Adam(
        [
            {"params": gru_params, "lr": lr_gru},
            {"params": mlp_params, "lr": lr_mlp},
        ],
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    return model, optimizer, scheduler, reference


def finetune_patient(
    patient_id, task, pretrained, task_dicts, device, config=None, checkpoint_path=None
):
    """Fine-tune the pretrained GRU on one patient and compare before/after.

    Trains on the early part of the patient's series, selects the best model on
    the middle part, and reports metrics on the later part — for both the
    pretrained generalist ("before") and the fine-tuned model ("after").

    Returns (row, artifacts):
      row       flat dict of scalar metrics for the summary CSV.
      artifacts loss histories and the best checkpoint, for debugging/plots.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    mu, sigma = pretrained["mu"], pretrained["sigma"]
    criterion = nn.MSELoss()

    # normalize with the pretraining-population stats (no per-patient leakage)
    series = (task_dicts[task][patient_id] - mu) / sigma
    loaders, sizes = build_patient_split(
        series,
        patient_id,
        cfg["sequence_length"],
        cfg["batch_size"],
        cfg["split_percentages"],
        device,
    )

    # BEFORE: the pretrained generalist evaluated on this patient's test split
    pre_model = GRUModel(**cfg["model_config"]).to(device)
    pre_model.load_state_dict(pretrained["model_state_dict"])
    before = evaluate_loader(
        pre_model, loaders["test"], criterion, mu, sigma, device, beta=cfg["beta"]
    )

    # AFTER: fine-tune a fresh copy, then evaluate the best checkpoint
    model, optimizer, scheduler, reference = build_finetune_model(
        pretrained,
        cfg["model_config"],
        cfg["lr_gru"],
        cfg["lr_mlp"],
        cfg["dropout_ft"],
        device,
    )
    checkpoint_dict, train_hist, val_hist = finetune(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        train_ids=[patient_id],
        val_ids=[patient_id],
        test_ids=[patient_id],
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=cfg["epochs"],
        mu=mu,
        sigma=sigma,
        train_task=task,
        test_task=task,
        reference=reference,
        lambda_l2sp=cfg["lambda_l2sp"],
        pred_sigma=pretrained["pred_sigma"],
        checkpoint_path=checkpoint_path,
        patience=cfg["patience"],
        beta=cfg["beta"],
    )
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    after = evaluate_loader(
        model, loaders["test"], criterion, mu, sigma, device, beta=cfg["beta"]
    )

    row = {
        "patient_id": str(patient_id),
        "task": task,
        "n_train": sizes["train"],
        "n_val": sizes["val"],
        "n_test": sizes["test"],
        "best_epoch": checkpoint_dict["epoch"],
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

    artifacts = {
        "train_loss_history": train_hist,
        "val_loss_history": val_hist,
        "checkpoint_dict": checkpoint_dict,
    }
    return row, artifacts


if __name__ == "__main__":
    """
    =======================================================================================
    Single-patient fine-tuning (debug entry point)

    Loads a pretrained GRU and adapts it to one patient with two learning rates
    (small for the GRU recurrent layers, larger for the final MLP block) plus an
    L2-SP penalty that keeps the weights close to the pretrained ones. The full
    sweep over all patients lives in finetune_all.py.
    =======================================================================================
    """
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TASK = "M"
    results_dir = os.path.join("results", "finetune")
    os.makedirs(results_dir, exist_ok=True)

    # Load pretrained model and checkpoint info (train/test ids, mu/sigma, sigmas)
    PRETRAINED_PATH = "checkpoints/generalist/GRU_R+M+LvR+M+L_beta0.5_ep150.pth"
    pretrained = torch.load(PRETRAINED_PATH, map_location=device, weights_only=False)
    task_dicts = load_task_dicts()

    # Fine-tune on a single held-out (test) patient — never seen during pretraining.
    # IDs are stored as strings; keep them as strings for the dict lookup.
    patient_id = str(pretrained["test_ids"][0])

    row, artifacts = finetune_patient(
        patient_id, TASK, pretrained, task_dicts, device
    )

    print(f"Windows per split: train={row['n_train']}, val={row['n_val']}, test={row['n_test']}")
    print(
        f"Patient {patient_id} ({TASK}) — test split, pretrained vs fine-tuned:\n"
        f"  FD-gain : {row['fdg_before']:+.4f} -> {row['fdg_after']:+.4f} "
        f"(Δ {row['delta_fdg']:+.4f})\n"
        f"  fd_pred : {row['fd_pred_before']:.4f} -> {row['fd_pred_after']:.4f} "
        f"(baseline {row['fd_base']:.4f})\n"
        f"  % frames improved : {row['pct_improved_before']:.1f}% -> "
        f"{row['pct_improved_after']:.1f}%"
    )

    # train and val loss history plots:
    plt.figure(figsize=(10, 5))
    plt.plot(artifacts["train_loss_history"], label="Train Loss")
    plt.plot(artifacts["val_loss_history"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Fine-tuning Train and Val Loss - GRU patient {patient_id} task {TASK} "
        f"beta {DEFAULT_CONFIG['beta']}"
    )
    plt.legend()
    plt.grid()
    plt.savefig(f"{results_dir}/GRU_ft_patient{patient_id}_{TASK}_loss_history.png")
    plt.show()
