"""routing.py — learned soft routing (stacking) over frozen experts + baseline.

This is stacking / a learned ensemble, NOT a mixture of experts: the experts are
trained independently and frozen, and only the router on top is trained. In a
real MoE the experts and gate are trained jointly so the experts specialize to
the routing; nothing here does that.

A small MLP reads per-frame features_tr (each expert's residual pred-base and its
predicted sigma) and emits softmax weights over the options;
the blended prediction is trained to MAXIMIZE FD_gain (over the previous-frame
baseline) directly. Experts stay frozen, so the router can always fall back on
the best single one.

Discipline: router trained on val_ids, evaluated on test_ids (both unseen by
every expert). 15% of val is held out for early stopping. All experts must
share the identical seeded R+M+L split and the same window length.
"""

import numpy as np
import torch
import torch.nn as nn

from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from models import build_model, get_device


def fd(pred, y):
    """Framewise displacement: sum |Δ| over translations + 50·|Δ| over rotations."""
    e = abs(pred - y)
    return e[:, :3].sum(1) + 50 * e[:, 3:].sum(1)


def expert_outputs(path, ids, task):
    """One expert's per-window (pred, std), each [N*win_per_patient, 6] in physical units, and time span"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]["data"]
    mu, sigma = ckpt["mu"], ckpt["sigma"]
    model = build_model(ckpt["config"]["model"]).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    # position-space denormalization: keep only the first 6 (position) channels
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)[:6]
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)[:6]

    d = np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()
    data = np.array([d[pid] for pid in ids])
    ds = TimeSeriesDataset(
        data,
        ids,
        sequence_length=cfg["sequence_length"],
        device=device,
        add_velocity=cfg.get("add_velocity", False),
        add_acceleration=cfg.get("add_acceleration", False),
        mu=mu,
        sigma=sigma,
    )
    loader = GPUBatchLoader(ds, batch_size=1024, shuffle=False)
    preds, stds = [], []
    with torch.no_grad():
        for _, x, _ in loader:
            mean, var = model(x)
            preds.append((mean * sigma_t + mu_t).cpu().numpy())
            stds.append((var.sqrt() * sigma_t).cpu().numpy())
    return np.concatenate(preds), np.concatenate(stds), cfg["sequence_length"] * 2


def collect(ids):
    """
    Collect features (residuals and stds) and options (expert predictions + baseline) for all tasks for the given ids
    """
    feats, options, ys, bases = [], [], [], []
    for task in TASKS:
        raw = np.array(
            [
                np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()[pid]
                for pid in ids
            ]
        )  # [N, T, 6] physical
        exp = {n: expert_outputs(p, ids, task) for n, p in EXPERTS.items()}
        ts = next(iter(exp.values()))[2]
        assert all(e[2] == ts for e in exp.values()), "experts must share window length"

        frames = np.arange(ts - 1, raw.shape[1])  # common target frames
        y = raw[:, frames].reshape(-1, 6)  # [N*wpp, 6] (wpp = windows per patient)
        base = raw[:, frames - 1].reshape(-1, 6)  # previous frame
        pred = {n: exp[n][0] for n in EXPERTS}
        std = {n: exp[n][1] for n in EXPERTS}

        options.append(
            np.stack([pred[n] for n in EXPERTS] + [base], axis=1)
        )  # [N*wpp, n_exp + 1, 6]

        feats.append(
            np.concatenate(
                [pred[n] - base for n in EXPERTS] + [std[n] for n in EXPERTS],
                axis=1,
            )
        )  # [[N*wpp, 6], [N*wpp, 6], ..., [N*wpp, 6], [N*wpp, 6], ...] -> [N*wpp, n_exp*12]
        ys.append(y)
        bases.append(base)

    return (
        np.concatenate(feats),
        np.concatenate(options),
        np.concatenate(ys),
        np.concatenate(bases),
    )


class Router(nn.Module):
    def __init__(self, in_dim, k, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, k),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)  # per-frame weights over options


def blend(router, X, E):
    """Blend the K options with the router's per-frame weights -> (pred, weights)."""
    w = router(X)
    return (w.unsqueeze(-1) * E).sum(1), w


def fd_gain(pred, y, fd_base):
    """Per-frame FD_gain against the previous-frame baseline, matching metrics.fd_gain."""
    return (fd_base - fd(pred, y)) / (fd_base + 1e-6)


def train_router(X, E, y, base, tr_idx, val_idx, epochs=200, bs=16384):
    print(X.shape, E.shape)
    # X.shape = (N*wpp, n_exp*12), E.shape = (N*wpp, n_exp + 1, 6)
    fd_base = fd(base, y)
    router = Router(X.shape[1], E.shape[1]).to(device)
    opt = torch.optim.Adam(router.parameters(), lr=1e-3, weight_decay=1e-4)
    best_gain, best_state = -np.inf, None
    for epoch in range(epochs):
        router.train()
        order = tr_idx[torch.randperm(len(tr_idx), device=device)]
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            opt.zero_grad()
            pred = blend(router, X[idx], E[idx])[0]
            gain = fd_gain(pred, y[idx], fd_base[idx])
            (-gain.mean()).backward()
            opt.step()
        router.eval()
        with torch.no_grad():
            pred_val = blend(router, X[val_idx], E[val_idx])[0]
            val_gain = fd_gain(pred_val, y[val_idx], fd_base[val_idx]).mean().item()
        if val_gain > best_gain:
            best_gain = val_gain
            best_state = {k: v.clone() for k, v in router.state_dict().items()}
        if (epoch + 1) % 25 == 0:
            print(
                f"  epoch {epoch+1:3d}  held-out FD_gain {val_gain:.4f}  (best {best_gain:.4f})"
            )
    router.load_state_dict(best_state)
    return router.eval()


if __name__ == "__main__":

    EXPERTS = {
        "mamba": "checkpoints/generalist/mamba_R+M+LvR+M+L_beta0.5_ep200.pth",
        "conformer": "checkpoints/generalist/conformer_R+M+LvR+M+L_beta0.5_ep200_3.pth",
        "gru": "checkpoints/generalist/gru_R+M+LvR+M+L_beta0.5_ep200_3.pth",
    }
    TASKS = ["R", "M", "L"]
    device = get_device()

    # Take the first expert checkpoint to get the val/test ids (they're the same for all experts)
    split = torch.load(
        next(iter(EXPERTS.values())), map_location="cpu", weights_only=False
    )
    names = list(EXPERTS) + ["baseline"]

    print("exacting router-train (val) and router-eval (test) frames...")
    # Collects "features" = residuals and stds for all experts, baseline and true target y both for val and test ids.
    features_tr, expert_preds_tr, y_tr, base_tr = collect(split["val_ids"])
    features_test, expert_preds_test, y_test, baseline_test = collect(split["test_ids"])
    print(f"  val {len(y_tr):,} frames, test {len(y_test):,} frames, options: {names}")

    # standardize features on val, move to device
    fmu, fstd = features_tr.mean(0), features_tr.std(0) + 1e-6
    features_tr_t = torch.tensor(
        (features_tr - fmu) / fstd, dtype=torch.float32, device=device
    )
    features_test_t = torch.tensor(
        (features_test - fmu) / fstd, dtype=torch.float32, device=device
    )
    expert_preds_tr_t = torch.tensor(
        expert_preds_tr, dtype=torch.float32, device=device
    )
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    base_tr_t = torch.tensor(base_tr, dtype=torch.float32, device=device)
    expert_preds_test_t = torch.tensor(
        expert_preds_test, dtype=torch.float32, device=device
    )

    # hold out 15% of val frames for actual validation (train = 85% of val, val = 15% of val, test = test)
    perm = torch.randperm(len(y_tr))
    n_val = int(0.15 * len(perm))
    val_idx, tr_idx = perm[:n_val].to(device), perm[n_val:].to(device)

    print("\ntraining soft router (objective = mean FD_gain of the blend)...")
    # Train router to maximize FD_gain on the train frames, select the best over val frames
    router = train_router(
        features_tr_t, expert_preds_tr_t, y_tr_t, base_tr_t, tr_idx, val_idx
    )

    fd_base_test = fd(baseline_test, y_test)

    def mean_fd_gain(pred):
        return fd_gain(pred, y_test, fd_base_test).mean()

    n_exp = len(EXPERTS)
    rows = [
        (f"{names[k]} (single)", mean_fd_gain(expert_preds_test[:, k]))
        for k in range(n_exp)
    ]

    # control: does per-frame weighting beat a fixed average?
    rows.append(
        (f"fixed mean of {n_exp}", mean_fd_gain(expert_preds_test[:, :n_exp].mean(1)))
    )

    with torch.no_grad():
        pred_soft_t, w = blend(router, features_test_t, expert_preds_test_t)
    rows.append(("soft router", mean_fd_gain(pred_soft_t.cpu().numpy())))

    # control: the router's average weights frozen and reused on every frame,
    # so any gap vs "soft router" is purely the per-frame routing (not the weights)
    wm = w.mean(0).cpu().numpy()  # [K] mean weight per option over test frames
    pred_fixed_w = (wm[None, :, None] * expert_preds_test).sum(1)
    rows.append(
        (
            "fixed router-mean weights",
            mean_fd_gain(pred_fixed_w),
        )
    )

    print("\n" + "=" * 56)
    print(f"{'policy':<28}{'mean FD_gain':>14}")
    for name, g in rows:
        print(f"{name:<28}{g:>14.4f}")
    print("=" * 56)

    print(
        "router weight per option: "
        + "  ".join(f"{n} {100*v:.0f}%" for n, v in zip(names, wm))
    )
