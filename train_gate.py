import sys

import numpy as np
import torch
import torch.nn as nn

from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from metrics import fd
from models import build_model, get_device

"""
====================================================================
Gating classifier experiment
====================================================================
Train a small classifier that, per frame, predicts whether the frozen
transformer will beat the previous-frame baseline (FD_gain >= 0) or lose
to it (FD_gain < 0). At inference, frames predicted "lose" fall back to
the baseline. The question is whether this raises FD_gain over the model.

Design:
- Features = frozen transformer latent (input to fc_mean) ++ predicted
  variance ++ predicted residual. Captured with a forward hook.
- Honest split: the backbone trained on train_ids, so the gate is
  TRAINED on val_ids and EVALUATED on test_ids. Both are unseen by the
  backbone = the deployment condition.
"""

CKPT = sys.argv[1] if len(sys.argv) > 1 else "generalist/transformer_R+M+LvR+M+L_beta0.5_ep150.pth"

device = get_device()
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
config = ckpt["config"]
mu, sigma = ckpt["mu"], ckpt["sigma"]
val_ids, test_ids = ckpt["val_ids"], ckpt["test_ids"]
tasks = parse_task(config["data"]["test_task"])
seq_len = config["data"]["sequence_length"]
add_v = config["data"].get("add_velocity", False)
add_a = config["data"].get("add_acceleration", False)

model = build_model(config["model"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# capture the latent that feeds the two heads (input to fc_mean)
_latent = {}
model.fc_mean.register_forward_hook(lambda m, inp, out: _latent.__setitem__("h", inp[0].detach()))

mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)


def extract(ids):
    """Run the frozen backbone over `ids`, return per-frame features, label, FDs."""
    feats, fdp, fdb = [], [], []
    for task in tasks:
        d = np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()
        data = (np.array([d[pid] for pid in ids]) - mu) / sigma
        vstd, astd = ckpt.get("feat_std", {}).get(task, (None, None))
        ds = TimeSeriesDataset(
            data, ids, sequence_length=seq_len, device=device,
            add_velocity=add_v, add_acceleration=add_a, vel_std=vstd, acc_std=astd,
        )
        loader = GPUBatchLoader(ds, batch_size=1024, shuffle=False)
        with torch.no_grad():
            for p, x, y in loader:
                mean, var = model(x)
                h = _latent["h"]
                last = x[:, -1, :6]
                resid = mean - last  # the residual the model added on top of baseline
                feats.append(torch.cat([h, var, resid], dim=1).cpu())
                fdp.append(fd(mean, y, mu_t, sigma_t).cpu())
                fdb.append(fd(last, y, mu_t, sigma_t).cpu())
    feats = torch.cat(feats).numpy()
    fdp = torch.cat(fdp).numpy()
    fdb = torch.cat(fdb).numpy()
    fdg = (fdb - fdp) / (fdb + 1e-6)
    label = (fdg < 0).astype(np.float32)  # 1 = model loses to baseline (gate it)
    return feats, label, fdp, fdb, fdg


print("extracting features (val = gate-train, test = gate-eval)...")
Xtr, ytr, fdp_tr, fdb_tr, fdg_tr = extract(val_ids)
Xte, yte, fdp_te, fdb_te, fdg_te = extract(test_ids)
print(f"  val:  {len(ytr):,} frames, {100*ytr.mean():.1f}% 'worse'")
print(f"  test: {len(yte):,} frames, {100*yte.mean():.1f}% 'worse'")

# standardize features on the gate-train set
fmu, fstd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xtr = (Xtr - fmu) / fstd
Xte = (Xte - fmu) / fstd

Xtr_t = torch.tensor(Xtr, device=device)
ytr_t = torch.tensor(ytr, device=device)
Xte_t = torch.tensor(Xte, device=device)


class Gate(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


gate = Gate(Xtr.shape[1]).to(device)
pos_weight = torch.tensor([float((ytr == 0).sum() / (ytr == 1).sum())], dtype=torch.float32, device=device)
crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-4)

print("\ntraining gate...")
N = len(ytr_t)
bs = 8192
for epoch in range(40):
    gate.train()
    perm = torch.randperm(N, device=device)
    tot = 0.0
    for s in range(0, N, bs):
        idx = perm[s:s + bs]
        opt.zero_grad()
        loss = crit(gate(Xtr_t[idx]), ytr_t[idx])
        loss.backward()
        opt.step()
        tot += loss.item() * len(idx)
    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:2d}  bce {tot/N:.4f}")


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = label.sum()
    n0 = len(label) - n1
    return (ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


gate.eval()
with torch.no_grad():
    prob_tr = torch.sigmoid(gate(Xtr_t)).cpu().numpy()
    prob_te = torch.sigmoid(gate(Xte_t)).cpu().numpy()

print(f"\nclassifier AUC:  val {auc(prob_tr, ytr):.3f}   test {auc(prob_te, yte):.3f}")


def gated_scores(prob, thr, fdp, fdb, fdg):
    """Apply policy: prob>thr -> use baseline. Return (mean fdg, aggregate fdg, frac gated)."""
    gate_on = prob > thr
    fd_used = np.where(gate_on, fdb, fdp)
    mean_fdg = np.where(gate_on, 0.0, fdg).mean()
    agg_fdg = (fdb.sum() - fd_used.sum()) / fdb.sum()
    return mean_fdg, agg_fdg, gate_on.mean()


# pick the decision threshold that maximizes mean FD_gain ON VAL, then apply to test
grid = np.linspace(0.05, 0.95, 91)
val_means = [gated_scores(prob_tr, t, fdp_tr, fdb_tr, fdg_tr)[0] for t in grid]
best_thr = grid[int(np.argmax(val_means))]

print("\n================ RESULTS (test set) ================")
no_mean, no_agg = fdg_te.mean(), (fdb_te.sum() - fdp_te.sum()) / fdb_te.sum()
orc_mean = np.maximum(fdg_te, 0).mean()
orc_agg = (fdb_te.sum() - np.minimum(fdp_te, fdb_te).sum()) / fdb_te.sum()
g_mean, g_agg, frac = gated_scores(prob_te, best_thr, fdp_te, fdb_te, fdg_te)

# optimistic ceiling for THIS classifier: best threshold chosen on test itself
te_means = [gated_scores(prob_te, t, fdp_te, fdb_te, fdg_te)[0] for t in grid]
best_te = grid[int(np.argmax(te_means))]
gt_mean, gt_agg, gt_frac = gated_scores(prob_te, best_te, fdp_te, fdb_te, fdg_te)

print(f"{'policy':<40}{'mean FD_gain':>14}{'aggregate':>12}{'% gated':>10}")
print(f"{'no gate (model only)':<40}{no_mean:>14.4f}{no_agg:>12.4f}{0.0:>9.1f}%")
print(f"{'gate @ val-tuned thr='+format(best_thr,'.2f'):<40}{g_mean:>14.4f}{g_agg:>12.4f}{100*frac:>9.1f}%")
print(f"{'gate @ test-tuned thr='+format(best_te,'.2f')+' (optimistic)':<40}{gt_mean:>14.4f}{gt_agg:>12.4f}{100*gt_frac:>9.1f}%")
print(f"{'ORACLE (perfect gate)':<40}{orc_mean:>14.4f}{orc_agg:>12.4f}{100*yte.mean():>9.1f}%")

# confusion at the val-tuned threshold (test)
pred_pos = prob_te > best_thr
tp = int((pred_pos & (yte == 1)).sum()); fp = int((pred_pos & (yte == 0)).sum())
fn = int((~pred_pos & (yte == 1)).sum()); tn = int((~pred_pos & (yte == 0)).sum())
prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
print(f"\nconfusion @ thr={best_thr:.2f}:  precision {prec:.3f}  recall {rec:.3f}  "
      f"(TP {tp} FP {fp} FN {fn} TN {tn})")
print("====================================================")
