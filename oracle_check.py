"""Per-frame ensemble oracle check.

Picks the best transformer, best TCN and best GRU in checkpoints/generalist/ (by mean
FD_gain, same ranking analyze_checkpoints.py uses), then asks: if an
*oracle* router could pick the best of the three for every single frame,
how much would FD_gain improve over the best single model?

The three models use different sequence_length, so they predict different
valid target frames. Predictions are aligned to the common frame set
(frames present for every model) before taking the per-frame min FD.
"""

import os

import numpy as np
import torch

from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from metrics import fd
from models import build_model, get_device

CKPT_DIR = "checkpoints/generalist"
device = get_device()


def model_fd_per_frame(path):
    """Run one checkpoint over its test set; return per-(patient, frame) FD
    matrices so different-length models can be aligned afterwards."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    mu, sigma = ckpt["mu"], ckpt["sigma"]
    test_ids = ckpt["test_ids"]
    tasks = parse_task(config["data"]["test_task"])
    seq_len = config["data"]["sequence_length"]
    time_span = seq_len * 2
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
        data = (np.array([d[pid] for pid in test_ids]) - mu) / sigma
        vstd, astd = ckpt.get("feat_std", {}).get(task, (None, None))
        ds = TimeSeriesDataset(
            data, test_ids, sequence_length=seq_len, device=device,
            add_velocity=add_v, add_acceleration=add_a, vel_std=vstd, acc_std=astd,
        )
        loader = GPUBatchLoader(ds, batch_size=1024, shuffle=False)
        wpp = loader.windows_per_patient
        P = len(test_ids)
        fdp = np.empty(P * wpp, dtype=np.float64)
        fdb = np.empty(P * wpp, dtype=np.float64)
        i = 0
        # shuffle=False -> samples arrive in index order (patient-major, t-minor),
        # so a flat fill then reshape(P, wpp) gives [patient, window] directly.
        with torch.no_grad():
            for p, x, y in loader:
                mean, _ = model(x)
                last = x[:, -1, :6]
                b = x.shape[0]
                fdp[i:i + b] = fd(mean, y, mu_t, sigma_t).cpu().numpy()
                fdb[i:i + b] = fd(last, y, mu_t, sigma_t).cpu().numpy()
                i += b
        # column t corresponds to target frame (time_span - 1 + t)
        per_task[task] = {
            "fdp": fdp.reshape(P, wpp),
            "fdb": fdb.reshape(P, wpp),
            "time_span": time_span,
        }
    return test_ids, tasks, per_task


def mean_fdg_of(per_task):
    """Overall mean FD_gain for ranking (raw, no cross-model alignment)."""
    fp = np.concatenate([d["fdp"].reshape(-1) for d in per_task.values()])
    fb = np.concatenate([d["fdb"].reshape(-1) for d in per_task.values()])
    return ((fb - fp) / (fb + 1e-6)).mean()


# ── group checkpoints by model family (config type), pick best of each ───
from collections import defaultdict

by_type = defaultdict(list)
for f in sorted(os.listdir(CKPT_DIR)):
    if not f.endswith(".pth"):
        continue
    t = torch.load(os.path.join(CKPT_DIR, f), map_location="cpu",
                   weights_only=False)["config"]["model"]["type"]
    by_type[t].append(f)

best = {}  # type -> (fname, test_ids, tasks, per_task)
for mtype in sorted(by_type):
    files = by_type[mtype]
    print(f"\n=== {mtype}: ranking {len(files)} checkpoints ===")
    best_score = -np.inf
    for f in files:
        test_ids, tasks, per_task = model_fd_per_frame(os.path.join(CKPT_DIR, f))
        score = mean_fdg_of(per_task)
        print(f"  {f:<46} mean_fdg={score:.4f}")
        if score > best_score:
            best_score = score
            best[mtype] = (f, test_ids, tasks, per_task)
    print(f"  -> best {mtype}: {best[mtype][0]}  (mean_fdg={best_score:.4f})")

names = list(best.keys())

# ── sanity: same patient test set across the three ───────────────────────
ref_ids = best[names[0]][1]
for n in names[1:]:
    assert np.array_equal(best[n][1], ref_ids), f"{n} has a different test split!"
tasks = best[names[0]][2]

# ── align to the common frame set and stack per-frame FD across models ───
fd_by_model = {n: [] for n in names}
fd_base_aligned = []
for task in tasks:
    spans = {n: best[n][3][task]["time_span"] for n in names}
    max_ts = max(spans.values())
    # keep, per model, only frames f in [max_ts-1, T-1]: drop the first
    # (max_ts - span) columns so every model lines up on the same frames.
    for n in names:
        mat = best[n][3][task]["fdp"]
        offset = max_ts - spans[n]
        fd_by_model[n].append(mat[:, offset:].reshape(-1))
    # baseline FD is data-only (frame f vs f-1) -> identical across models;
    # take it from the longest-span model (offset 0) and verify the others match.
    longest = max(names, key=lambda n: spans[n])
    base = best[longest][3][task]["fdb"]
    base_off = max_ts - spans[longest]  # == 0
    fd_base_aligned.append(base[:, base_off:].reshape(-1))
    for n in names:
        chk = best[n][3][task]["fdb"][:, max_ts - spans[n]:].reshape(-1)
        assert np.allclose(chk, fd_base_aligned[-1], atol=1e-4), f"baseline mismatch {n} {task}"

fdp = {n: np.concatenate(fd_by_model[n]) for n in names}
fdb = np.concatenate(fd_base_aligned)
N = len(fdb)

stack = np.stack([fdp[n] for n in names], axis=1)       # [N, 3] model FDs
oracle_models = stack.min(axis=1)                        # best-of-3 per frame
# router + baseline fallback ceiling: also allow choosing the previous frame
oracle_all = np.minimum(oracle_models, fdb)


def summary(fd_pred):
    fdg = (fdb - fd_pred) / (fdb + 1e-6)
    agg = (fdb.sum() - fd_pred.sum()) / fdb.sum()
    return fdg.mean(), agg


print("\n" + "=" * 60)
print(f"Aligned common frames: {N:,}")
print(f"{'policy':<34}{'mean FD_gain':>14}{'aggregate':>12}")
for n in names:
    m, a = summary(fdp[n])
    print(f"{best[n][0].split('_')[0]+' (best '+n+')':<34}{m:>14.4f}{a:>12.4f}")
best_single = max(names, key=lambda n: summary(fdp[n])[0])
bm, ba = summary(fdp[best_single])
om, oa = summary(oracle_models)
fm, fa = summary(oracle_all)
print("-" * 60)
print(f"{f'ORACLE router (best of {len(names)})':<34}{om:>14.4f}{oa:>12.4f}")
print(f"{'ORACLE router + baseline fallback':<34}{fm:>14.4f}{fa:>12.4f}")
print("-" * 60)
print(f"best single model = {best_single} ({best[best_single][0]})")
print(f"headroom over best single:  mean +{om-bm:.4f}   aggregate +{oa-ba:.4f}")

# ── how often is each model the unique best? (is routing even worth it?) ──
winner = stack.argmin(axis=1)
print("\nper-frame winner share (which model has lowest FD):")
for i, n in enumerate(names):
    print(f"  {n:<12}{100*(winner==i).mean():>6.1f}%")
print("\nshare where the previous-frame baseline beats ALL models:")
print(f"  baseline    {100*(fdb[:,None] < stack).all(axis=1).mean():>6.1f}%")
print("=" * 60)
