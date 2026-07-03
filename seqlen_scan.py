"""Plot best-per-architecture FD gain (and FD, %>0) vs sequence length.

Evaluates every checkpoint in a folder, groups by (model type, sequence_length),
and keeps only the best-performing checkpoint (highest FD gain) at each length —
i.e. the best choice of hyperparameters other than sequence_length, which is the
sweep axis. Draws one line per model architecture.

Usage: python seqlen_scan.py [checkpoints/generalist]
"""

import glob
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import torch

from models import get_device
from sweep import run_eval

CHECKPOINT_DIR = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/generalist"
RESULTS_DIR = "results/seqlen_scan"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = get_device()

# ── evaluate every checkpoint, keep the best one per (model type, seq_len) ──
best = {}  # (mtype, seq_len) -> run_eval result
for path in sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))):
    config = torch.load(path, map_location="cpu", weights_only=False)["config"]
    mtype = config["model"]["type"]
    seq_len = config["data"]["sequence_length"]

    r = run_eval(path, device)
    print(
        f"{mtype:<12} seq_len={seq_len:<4} fdg={r['fdg']:.4f}  ({os.path.basename(path)})"
    )

    key = (mtype, seq_len)
    if key not in best or r["fdg"] > best[key]["fdg"]:
        best[key] = r

# ── one series per model type, sorted by sequence length ──
by_model = defaultdict(list)
for (mtype, seq_len), r in best.items():
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
    ax.set_title(f"Best {title} vs sequence length")
    ax.legend(fontsize=8)
axes[0].axhline(0, color="black", lw=0.8)

fig.suptitle("Best-per-architecture performance vs sequence length")
fig.tight_layout()
out_path = os.path.join(RESULTS_DIR, "fdg_vs_seqlen.png")
fig.savefig(out_path, bbox_inches="tight")
print(f"\nsaved {out_path}")
