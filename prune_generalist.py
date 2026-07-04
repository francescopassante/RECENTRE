"""Prune a checkpoint folder to one canonical vel+acc checkpoint per (family, seq_len).

For each architecture family we find the vel+acc hyperparameter signature that
covers all of {10,32,64,128}, keep one file per seq_len (best FD-gain when there
are identical-HP duplicates), and move every other .pth to <dir>/_archive/.

Usage: python prune_generalist.py [generalist]   # run from the repo root

Idempotent: re-running an already-pruned folder keeps the same files and archives
nothing. The FD-gain tie-break needs datasets/*_dict.npy; if those are missing,
duplicates fall back to the first filename alphabetically (with a warning).
"""

import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import torch

from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from metrics import evaluate
from models import build_model, get_device

D = sys.argv[1] if len(sys.argv) > 1 else "generalist"
ARCHIVE = os.path.join(D, "_archive")
TARGET = {10, 32, 64, 128}
IGNORE_MODEL = {"sequence_length", "max_len", "input_dim"}  # seq-derived, may differ
device = get_device()
print(f"device: {device}\npruning: {D}\n")


def load_cfg(f):
    return torch.load(os.path.join(D, f), map_location="cpu", weights_only=False)[
        "config"
    ]


def signature(c):
    """Architecture identity that must match across seq_len (features + arch + opt)."""
    m, da, tr = c["model"], c["data"], c["train"]
    vel, acc = da.get("add_velocity", False), da.get("add_acceleration", False)
    arch = tuple(
        sorted((k, str(v)) for k, v in m.items() if k not in IGNORE_MODEL and k != "type")
    )
    opt = (tr.get("lr"), tr.get("weight_decay"), tr.get("loss"), tr.get("beta"))
    return (vel, acc), arch, opt


# feature-aware FD-gain eval (mirrors benchmark.eval_ckpt) — used only for tie-breaks
def fdg(f):
    ckpt = torch.load(os.path.join(D, f), map_location=device, weights_only=False)
    config = ckpt["config"]
    mu, sigma, test_ids = ckpt["mu"], ckpt["sigma"], ckpt["test_ids"]
    model = build_model(config["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    seq_len = config["data"].get("sequence_length", 10)
    add_vel = config["data"].get("add_velocity", False)
    add_acc = config["data"].get("add_acceleration", False)
    fps, fbs = [], []
    for task in parse_task(config["data"]["test_task"]):
        dd = np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()
        data = (np.array([dd[pid] for pid in test_ids]) - mu) / sigma
        vs, as_ = ckpt.get("feat_std", {}).get(task, (None, None))
        ds = TimeSeriesDataset(
            data, test_ids, sequence_length=seq_len, device=device,
            add_velocity=add_vel, add_acceleration=add_acc, vel_std=vs, acc_std=as_,
        )
        out = evaluate(model, GPUBatchLoader(ds, batch_size=1024, shuffle=False),
                       mu, sigma, device)
        fps.append(out["fd_pred"]); fbs.append(out["fd_base"])
    fp, fb = np.concatenate(fps), np.concatenate(fbs)
    return float(((fb - fp) / (fb + 1e-6)).mean())


def pick(cands):
    """Choose one file among identical-HP duplicates: best FD-gain, else first."""
    if len(cands) == 1:
        return cands[0]
    try:
        return sorted(((fdg(c), c) for c in cands), reverse=True)[0][1]
    except Exception as e:  # datasets missing, etc. — fall back deterministically
        print(f"    (tie-break eval failed: {e}; keeping first alphabetically)")
        return sorted(cands)[0]


files = [f for f in sorted(os.listdir(D)) if f.endswith(".pth")]
fam = defaultdict(list)
for f in files:
    c = load_cfg(f)
    (vel, acc), arch, opt = signature(c)
    seq = c["data"].get("sequence_length", c["model"].get("sequence_length"))
    fam[c["model"]["type"]].append((f, seq, vel, acc, (vel, acc, arch, opt)))

keep = set()
for t in sorted(fam):
    by_sig = defaultdict(lambda: defaultdict(list))
    for f, seq, vel, acc, sig in fam[t]:
        if vel and acc:
            by_sig[sig][seq].append(f)
    covering = [s for s, sm in by_sig.items() if TARGET.issubset(sm)]
    print("=" * 80)
    print(f"### {t}")
    if not covering:
        print(f"  !! no vel+acc signature covers {sorted(TARGET)} — SKIPPING (nothing kept)")
        for s, sm in by_sig.items():
            print(f"     vel+acc sig covers seqs {sorted(sm)}")
        continue
    if len(covering) > 1:
        print(f"  note: {len(covering)} vel+acc signatures cover the grid; taking the first")
    seqmap = by_sig[covering[0]]
    for seq in sorted(TARGET):
        chosen = pick(seqmap[seq])
        if len(seqmap[seq]) > 1:
            print(f"  seq {seq}: {len(seqmap[seq])} dupes -> kept {chosen}")
        keep.add(chosen)
        print(f"  KEEP seq {seq:>3}: {chosen}")

os.makedirs(ARCHIVE, exist_ok=True)
moved = 0
print("\n" + "=" * 80 + "\nARCHIVING:")
for f in files:
    if f not in keep:
        shutil.move(os.path.join(D, f), os.path.join(ARCHIVE, f))
        moved += 1
        print(f"  -> _archive/ {f}")

print(f"\nkept {len(keep)} files, archived {moved}.")
