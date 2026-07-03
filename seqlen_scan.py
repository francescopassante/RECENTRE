"""Plot FD gain (and FD, %>0) vs sequence length, one line per model architecture.

Evaluates a hardcoded list of checkpoints (one per model type / sequence_length)
and groups by model type.

Usage: python seqlen_scan.py
"""

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from metrics import evaluate
from models import build_model, get_device

CHECKPOINT_DIR = "checkpoints/generalist"
CHECKPOINTS = [
    "conformer_R+M+LvR+M+L_beta0.5_ep100.pth",
    "conformer_R+M+LvR+M+L_beta0.5_ep150_2.pth",
    "conformer_R+M+LvR+M+L_beta0.5_ep100_2.pth",
    "conformer_R+M+LvR+M+L_beta0.5_ep150.pth",
    "mamba_R+M+LvR+M+L_beta0.5_ep100_5.pth",
    "mamba_R+M+LvR+M+L_beta0.5_ep100_2.pth",
    "mamba_R+M+LvR+M+L_beta0.5_ep100_3.pth",
    "mamba_R+M+LvR+M+L_beta0.5_ep100_4.pth",
    "gru_R+M+LvR+M+L_beta0.5_ep100_7.pth",
    "gru_R+M+LvR+M+L_beta0.5_ep100_6.pth",
    "gru_R+M+LvR+M+L_beta0.5_ep100.pth",
    "gru_R+M+LvR+M+L_beta0.5_ep150_2.pth",
]
RESULTS_DIR = "results/seqlen_scan"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = get_device()


def eval_checkpoint(path):
    """Evaluate one checkpoint on its test set (mirrors analyze_checkpoints.py)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    mu, sigma = ckpt["mu"], ckpt["sigma"]
    test_ids = ckpt["test_ids"]

    model = build_model(config["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_tasks = parse_task(config["data"]["test_task"])
    seq_len = config["data"]["sequence_length"]
    add_velocity = config["data"].get("add_velocity", False)
    add_acceleration = config["data"].get("add_acceleration", False)

    all_fd_pred, all_fd_base = [], []
    for task in test_tasks:
        data_dict = np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()
        data = (np.array([data_dict[pid] for pid in test_ids]) - mu) / sigma
        vel_std, acc_std = ckpt.get("feat_std", {}).get(task, (None, None))
        ds = TimeSeriesDataset(
            data,
            test_ids,
            sequence_length=seq_len,
            device=device,
            add_velocity=add_velocity,
            add_acceleration=add_acceleration,
            vel_std=vel_std,
            acc_std=acc_std,
        )
        loader = GPUBatchLoader(ds, batch_size=1024, shuffle=False)
        out = evaluate(model, loader, mu, sigma, device)
        all_fd_pred.append(out["fd_pred"])
        all_fd_base.append(out["fd_base"])

    fd_pred = np.concatenate(all_fd_pred)
    fd_base = np.concatenate(all_fd_base)
    fdg_per_sample = (fd_base - fd_pred) / (fd_base + 1e-6)
    return {
        "fd_pred": float(fd_pred.mean()),
        "fdg": float(fdg_per_sample.mean()),
        "fdg_per_sample": fdg_per_sample,
    }


# ── evaluate each checkpoint, one series per model type ──
by_model = defaultdict(list)  # mtype -> [(seq_len, result), ...]
for fname in CHECKPOINTS:
    path = os.path.join(CHECKPOINT_DIR, fname)
    config = torch.load(path, map_location="cpu", weights_only=False)["config"]
    mtype = config["model"]["type"]
    seq_len = config["data"]["sequence_length"]

    r = eval_checkpoint(path)
    print(f"{mtype:<12} seq_len={seq_len:<4} fdg={r['fdg']:.4f}  ({fname})")
    by_model[mtype].append((seq_len, r))

for entries in by_model.values():
    entries.sort(key=lambda x: x[0])

all_seq_lens = sorted({s for entries in by_model.values() for s, _ in entries})

# ── plot ──
palette = ["blue", "green", "orange", "red", "purple", "brown", "black", "teal", "magenta"]
model_color = {m: palette[i % len(palette)] for i, m in enumerate(sorted(by_model))}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for mtype, entries in sorted(by_model.items()):
    seq_lens = [s for s, _ in entries]
    fdg = [r["fdg"] for _, r in entries]
    fd_pred = [r["fd_pred"] for _, r in entries]
    pct_pos = [(r["fdg_per_sample"] > 0).mean() * 100 for _, r in entries]
    color = model_color[mtype]
    axes[0].plot(seq_lens, fdg, "o-", color=color, label=mtype)
    axes[1].plot(seq_lens, fd_pred, "o-", color=color, label=mtype)
    axes[2].plot(seq_lens, pct_pos, "o-", color=color, label=mtype)

titles = ("FD gain", "FD (predicted, mm)", "% samples with FD gain > 0")
ylabels = ("FD gain", "mean FD (mm)", "% > 0")
for ax, title, ylabel in zip(axes, titles, ylabels):
    ax.set_xscale("log")
    ax.set_xticks(all_seq_lens)
    ax.set_xticklabels([str(s) for s in all_seq_lens])
    ax.set_xlabel("sequence length")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} vs sequence length")
    ax.legend(fontsize=8)
axes[0].axhline(0, color="black", lw=0.8)

fig.suptitle("Per-architecture performance vs sequence length")
fig.tight_layout()
out_path = os.path.join(RESULTS_DIR, "fdg_vs_seqlen.png")
fig.savefig(out_path, bbox_inches="tight")
print(f"\nsaved {out_path}")
