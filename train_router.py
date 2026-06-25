"""Differentiable soft router over the best transformer / GRU / TCN.

The three frozen experts (best of each type by mean FD_gain, see
oracle_check.py) plus the previous-frame baseline form 4 routable options.
A small MLP looks at the experts' disagreement + uncertainty + recent
motion and outputs per-frame softmax weights; the blended prediction is
trained to MINIMIZE FD directly. This optimizes the real objective
end-to-end (no classify->threshold proxy) and has a safety floor — it can
always put all weight on the best single expert.

Discipline: router trained on val_ids, evaluated on test_ids (both unseen
by every expert). A held-out slice of val is used for early stopping.
"""

import numpy as np
import torch
import torch.nn as nn

from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from models import build_model, get_device

# best of each model family (by mean FD_gain, from oracle_check.py).
# Strong families first so indices 0,1,2 are transformer/gru/tcn (used by the
# fixed-average controls below).
EXPERTS = {
    "transformer": "generalist/transformer_R+M+LvR+M+L_beta0.5_ep150_5.pth",
    "gru": "generalist/gru_R+M+LvR+M+L_beta0.5_ep150_2.pth",
    "tcn": "generalist/tcn_R+M+LvR+M+L_beta0.5_ep150.pth",
    "tsmixer": "generalist/TSMixer_R+M+LvR+M+L_beta0.5_ep150_3.pth",
    "patchtst": "generalist/patchTST_R+M+LvR+M+L_beta0.5_ep150.pth",
    "dlinear": "generalist/dlinear_R+M+LvR+M+L_beta0.5_ep150_2.pth",
}

device = get_device()


def expert_pred_std(path, ids):
    """Per-(patient, window) predicted mean + std (physical units) for one expert."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    mu, sigma = ckpt["mu"], ckpt["sigma"]
    tasks = parse_task(config["data"]["test_task"])
    seq_len = config["data"]["sequence_length"]
    add_v = config["data"].get("add_velocity", False)
    add_a = config["data"].get("add_acceleration", False)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)

    per_task = {}
    for task in tasks:
        d = np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()
        data = (np.array([d[pid] for pid in ids]) - mu) / sigma
        vstd, astd = ckpt.get("feat_std", {}).get(task, (None, None))
        ds = TimeSeriesDataset(
            data, ids, sequence_length=seq_len, device=device,
            add_velocity=add_v, add_acceleration=add_a, vel_std=vstd, acc_std=astd,
        )
        loader = GPUBatchLoader(ds, batch_size=1024, shuffle=False)
        wpp = loader.windows_per_patient
        P = len(ids)
        pred = np.empty((P * wpp, 6))
        std = np.empty((P * wpp, 6))
        i = 0
        with torch.no_grad():
            for p, x, y in loader:
                mean, var = model(x)
                b = x.shape[0]
                pred[i:i + b] = (mean * sigma_t + mu_t).cpu().numpy()
                std[i:i + b] = (var.sqrt() * sigma_t).cpu().numpy()
                i += b
        # column t -> target frame (seq_len*2 - 1 + t)
        per_task[task] = {
            "pred": pred.reshape(P, wpp, 6),
            "std": std.reshape(P, wpp, 6),
            "ts": seq_len * 2,
        }
    return per_task, tasks


def build_frames(ids):
    """Align all experts to the common frame set; return per-frame features,
    the K routable predictions, the target y, and the baseline (all physical)."""
    names = list(EXPERTS.keys())
    exp = {n: expert_pred_std(p, ids)[0] for n, p in EXPERTS.items()}
    tasks = parse_task("R+M+L")

    feats_l, experts_l, y_l, base_l = [], [], [], []
    for task in tasks:
        raw = np.array([np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()[pid]
                        for pid in ids])  # [P, T, 6] physical
        T = raw.shape[1]
        spans = {n: exp[n][task]["ts"] for n in names}
        max_ts = max(spans.values())
        frames = np.arange(max_ts - 1, T)  # common target frames

        # y and baseline are data-only (model-independent)
        y = raw[:, frames, :].reshape(-1, 6)
        base = raw[:, frames - 1, :].reshape(-1, 6)            # previous frame
        rv = (raw[:, frames - 1, :] - raw[:, frames - 2, :]).reshape(-1, 6)  # recent motion

        preds, stds = {}, {}
        for n in names:
            off = max_ts - spans[n]  # drop the frames this expert can't predict
            preds[n] = exp[n][task]["pred"][:, off:, :].reshape(-1, 6)
            stds[n] = exp[n][task]["std"][:, off:, :].reshape(-1, 6)

        # routable options: the 3 experts + the previous-frame baseline
        E = np.stack([preds[n] for n in names] + [base], axis=1)  # [N, 4, 6]
        # features: each expert's residual (pred-base) + its std + recent motion
        feats = np.concatenate(
            [preds[n] - base for n in names] + [stds[n] for n in names] + [rv], axis=1
        )
        feats_l.append(feats)
        experts_l.append(E)
        y_l.append(y)
        base_l.append(base)

    return (np.concatenate(feats_l), np.concatenate(experts_l),
            np.concatenate(y_l), np.concatenate(base_l), names + ["baseline"])


def fd_np(pred, y):
    e = np.abs(pred - y)
    return e[:, :3].sum(1) + 50 * e[:, 3:].sum(1)


def fd_t(pred, y):
    e = (pred - y).abs()
    return e[:, :3].sum(1) + 50 * e[:, 3:].sum(1)


print("extracting val (router-train) and test (router-eval) frames...")
Xtr, Etr, ytr, btr, opt_names = build_frames(torch.load(
    EXPERTS["transformer"], map_location="cpu", weights_only=False)["val_ids"])
Xte, Ete, yte, bte, _ = build_frames(torch.load(
    EXPERTS["transformer"], map_location="cpu", weights_only=False)["test_ids"])
K = Etr.shape[1]
print(f"  val {len(ytr):,} frames, test {len(yte):,} frames, {K} routable options: {opt_names}")

# standardize features on val
fmu, fstd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xtr = (Xtr - fmu) / fstd
Xte = (Xte - fmu) / fstd

# tensors
Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
Etr_t = torch.tensor(Etr, dtype=torch.float32, device=device)
ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
Ete_t = torch.tensor(Ete, dtype=torch.float32, device=device)

# hold out 15% of val frames for early stopping
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(ytr_t), generator=g)
n_es = int(0.15 * len(perm))
es_idx, tr_idx = perm[:n_es].to(device), perm[n_es:].to(device)


class Router(nn.Module):
    def __init__(self, in_dim, k, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, k),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)  # per-frame weights over options


def blend_fd(router, X, E, y):
    w = router(X)                                # [N, K]
    pred = (w.unsqueeze(-1) * E).sum(1)          # [N, 6]
    return fd_t(pred, y).mean(), w


router = Router(Xtr.shape[1], K).to(device)
opt = torch.optim.Adam(router.parameters(), lr=1e-3, weight_decay=1e-4)

print("\ntraining soft router (objective = mean FD of the blend)...")
bs = 16384
best_es, best_state, best_ep = np.inf, None, -1
for epoch in range(200):
    router.train()
    order = tr_idx[torch.randperm(len(tr_idx), device=device)]
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        opt.zero_grad()
        loss, _ = blend_fd(router, Xtr_t[idx], Etr_t[idx], ytr_t[idx])
        loss.backward()
        opt.step()
    router.eval()
    with torch.no_grad():
        es_fd, _ = blend_fd(router, Xtr_t[es_idx], Etr_t[es_idx], ytr_t[es_idx])
    if es_fd.item() < best_es:
        best_es, best_ep = es_fd.item(), epoch
        best_state = {k: v.detach().clone() for k, v in router.state_dict().items()}
    if (epoch + 1) % 25 == 0:
        print(f"  epoch {epoch+1:3d}  held-out FD {es_fd.item():.4f}  (best {best_es:.4f} @ ep{best_ep+1})")

router.load_state_dict(best_state)
router.eval()


# ── evaluation on test ───────────────────────────────────────────────────
fdb = fd_np(bte, yte)


def report(pred, label):
    fdp = fd_np(pred, yte)
    fdg = (fdb - fdp) / (fdb + 1e-6)
    agg = (fdb.sum() - fdp.sum()) / fdb.sum()
    print(f"{label:<38}{fdg.mean():>14.4f}{agg:>12.4f}")
    return fdg.mean(), agg


n_exp = len(EXPERTS)  # routable models; baseline is the extra last column of E

print("\n" + "=" * 64)
print(f"{'policy':<38}{'mean FD_gain':>14}{'aggregate':>12}")
# single experts
single = {}
for k in range(n_exp):
    single[opt_names[k]] = report(Ete[:, k, :], f"{opt_names[k]} (best single)")
print("-" * 64)
# oracles
fde = np.stack([fd_np(Ete[:, k, :], yte) for k in range(K)], axis=1)  # [N, K]
oracle_exp = Ete[np.arange(len(yte)), fde[:, :n_exp].argmin(1), :]    # best of the n_exp models
oracle_all = Ete[np.arange(len(yte)), fde.argmin(1), :]              # + baseline option
report(oracle_exp, f"ORACLE best-of-{n_exp}")
report(oracle_all, f"ORACLE best-of-{n_exp} + baseline")
print("-" * 64)
# controls: static (non-routed) ensembles — does per-frame weighting beat a fixed average?
report(0.5 * (Ete[:, 0, :] + Ete[:, 1, :]), f"fixed 50/50 {opt_names[0]}+{opt_names[1]}")
report(Ete[:, :n_exp, :].mean(1), f"fixed mean of {n_exp} experts")
print("-" * 64)
# soft router + hard (argmax) router
with torch.no_grad():
    w_te = router(Xte_t)
    pred_soft = (w_te.unsqueeze(-1) * Ete_t).sum(1).cpu().numpy()
sr = report(pred_soft, "SOFT ROUTER (blend)")
hard = Ete[np.arange(len(yte)), w_te.argmax(1).cpu().numpy(), :]
report(hard, "HARD ROUTER (argmax weight)")
print("-" * 64)

best_single_name = max(single, key=lambda n: single[n][1])
bs_agg = single[best_single_name][1]
o4_agg = (fdb.sum() - fd_np(oracle_all, yte).sum()) / fdb.sum()
captured = (sr[1] - bs_agg) / (o4_agg - bs_agg + 1e-9)
print(f"best single = {best_single_name} (aggregate {bs_agg:.4f})")
print(f"soft router aggregate {sr[1]:.4f}  |  oracle+base {o4_agg:.4f}")
print(f"fraction of oracle headroom captured: {100*captured:.1f}%")

# average weight mass per option (where does the router put its trust?)
wm = w_te.mean(0).cpu().numpy()
print("\naverage router weight per option:")
for n, w in zip(opt_names, wm):
    print(f"  {n:<12}{100*w:>6.1f}%")
print("=" * 64)
