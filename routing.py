"""routing.py — learned soft routing (stacking) over frozen experts + baseline.

This is stacking / a learned ensemble, NOT a mixture of experts: the experts are
trained independently and frozen, and only the router on top is trained. In a
real MoE the experts and gate are trained jointly so the experts specialize to
the routing; nothing here does that.

A small MLP reads per-frame features_tr (each expert's residual pred-base and its
predicted sigma) and emits softmax weights over the options;
the blended prediction is trained to MINIMIZE FD directly. Experts stay frozen,
so the router can always fall back on the best single one.

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
    """One expert's per-window (pred, std), each [P*wpp, 6] in physical units,
    plus ts = the first predictable target-frame index + 1."""
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
    """Build the frame set for a group of patients: per-frame features_tr X, the K
    routable options E (experts + baseline), the target y, and the baseline."""
    feats, options, ys, bases = [], [], [], []
    for task in TASKS:
        raw = np.array(
            [
                np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()[pid]
                for pid in ids
            ]
        )  # [P, T, 6] physical
        exp = {n: expert_outputs(p, ids, task) for n, p in EXPERTS.items()}
        ts = next(iter(exp.values()))[2]
        assert all(e[2] == ts for e in exp.values()), "experts must share window length"

        frames = np.arange(ts - 1, raw.shape[1])  # common target frames
        y = raw[:, frames].reshape(-1, 6)  # [P*win_per_patient, 6]
        base = raw[:, frames - 1].reshape(-1, 6)  # previous frame
        pred = {n: exp[n][0] for n in EXPERTS}
        std = {n: exp[n][1] for n in EXPERTS}

        options.append(
            np.stack([pred[n] for n in EXPERTS] + [base], axis=1)
        )  # [N, K, 6]
        feats.append(
            np.concatenate(
                [pred[n] - base for n in EXPERTS] + [std[n] for n in EXPERTS],
                axis=1,
            )
        )
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


def train_router(X, E, y, tr, es, epochs=200, bs=16384):
    """Minimize the blend's mean FD; select on the held-out slice es."""
    router = Router(X.shape[1], E.shape[1]).to(device)
    opt = torch.optim.Adam(
        router.paramexpert_preds_testrs(), lr=1e-3, weight_decay=1e-4
    )
    best_fd, best_state = np.inf, None
    for epoch in range(epochs):
        router.train()
        order = tr[torch.randperm(len(tr), device=device)]
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            opt.zero_grad()
            fd(blend(router, X[idx], E[idx])[0], y[idx]).mean().backward()
            opt.step()
        router.eval()
        with torch.no_grad():
            val_fd = fd(blend(router, X[es], E[es])[0], y[es]).mean().item()
        if val_fd < best_fd:
            best_fd = val_fd
            best_state = {k: v.clone() for k, v in router.state_dict().items()}
        if (epoch + 1) % 25 == 0:
            print(
                f"  epoch {epoch+1:3d}  held-out FD {val_fd:.4f}  (best {best_fd:.4f})"
            )
    router.load_state_dict(best_state)
    return router.eval()


if __name__ == "__main__":

    EXPERTS = {
        "mamba": "checkpoints/generalist/mamba_R+M+LvR+M+L_beta0.5_ep100_5.pth",
        "conformer": "checkpoints/generalist/conformer_R+M+LvR+M+L_beta0.5_ep100.pth",
        "gru": "checkpoints/generalist/gru_R+M+LvR+M+L_beta0.5_ep150_5.pth",
    }
    TASKS = ["R", "M", "L"]
    device = get_device()

    split = torch.load(
        next(iter(EXPERTS.values())), map_location="cpu", weights_only=False
    )
    names = list(EXPERTS) + ["baseline"]

    print("exacting router-train (val) and router-eval (test) frames...")
    features_tr, expert_preds_tr, y_tr, _ = collect(split["val_ids"])
    features_test, expert_preds_test, y_test, baseline_test = collect(split["test_ids"])
    K = expert_preds_tr.shape[1]
    print(f"  val {len(y_tr):,} frames, test {len(y_test):,} frames, options: {names}")

    # standardize features_tr on val, then move everything to the device
    fmu, fstd = features_tr.mean(0), features_tr.std(0) + 1e-6
    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    features_tr_t, features_test_t = to_t((features_tr - fmu) / fstd), to_t(
        (features_test - fmu) / fstd
    )
    expert_preds_tr_t, y_tr_t, expert_preds_test_t = (
        to_t(expert_preds_tr),
        to_t(y_tr),
        to_t(expert_preds_test),
    )

    # hold out 15% of val frames for early stopping
    perm = torch.randperm(len(y_tr), generator=torch.Generator().manual_seed(0))
    n_es = int(0.15 * len(perm))
    es_idx, tr_idx = perm[:n_es].to(device), perm[n_es:].to(device)

    print("\ntraining soft router (objective = mean FD of the blend)...")
    router = train_router(features_tr_t, expert_preds_tr_t, y_tr_t, tr_idx, es_idx)

    # ── evaluation on test ────────────────────────────────────────────────
    fdb = fd(baseline_test, y_test)

    def gain(pred):
        fdp = fd(pred, y_test)
        return ((fdb - fdp) / (fdb + 1e-6)).mean(), (fdb.sum() - fdp.sum()) / fdb.sum()

    n_exp = len(EXPERTS)
    rows = [
        (f"{names[k]} (single)", *gain(expert_preds_test[:, k])) for k in range(n_exp)
    ]

    # oracle: per-frame lowest-FD option (peeks at the target -> upper bound)
    fde = np.stack([fd(expert_preds_test[:, k], y_test) for k in range(K)], axis=1)
    oracle = expert_preds_test[np.arange(len(y_test)), fde.argmin(1)]
    rows.append(("ORACLE best-of-all", *gain(oracle)))

    # control: does per-frame weighting beat a fixed average?
    rows.append((f"fixed mean of {n_exp}", *gain(expert_preds_test[:, :n_exp].mean(1))))

    with torch.no_grad():
        pred_soft_t, w = blend(router, features_test_t, expert_preds_test_t)
    soft = gain(pred_soft_t.cpu().numpy())
    rows.append(("SOFT ROUTER", *soft))

    print("\n" + "=" * 56)
    print(f"{'policy':<28}{'mean FD_gain':>14}{'aggregate':>12}")
    for name, g, a in rows:
        print(f"{name:<28}{g:>14.4f}{a:>12.4f}")
    print("=" * 56)

    best = max((r for r in rows if "single" in r[0]), key=lambda r: r[2])
    orc = next(r for r in rows if r[0].startswith("ORACLE"))
    captured = (soft[1] - best[2]) / (orc[2] - best[2] + 1e-9)
    print(
        f"best single {best[0].split()[0]} agg {best[2]:.4f}  |  "
        f"oracle {orc[2]:.4f}  |  router captures {100*captured:.1f}% of headroom"
    )
    wm = w.mean(0).cpu().numpy()
    print(
        "router weight per option: "
        + "  ".join(f"{n} {100*v:.0f}%" for n, v in zip(names, wm))
    )
