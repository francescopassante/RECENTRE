"""Pareto visualization of the architecture benchmark (benchmark.py output).

Reads results/benchmark/benchmark.csv and plots the accuracy/cost trade-off:
FD-gain (higher = better) against three deployment-cost axes — single-frame
latency, FLOPs/window and float32 size — plus one accuracy-vs-robustness axis
(noise tolerance sigma). In each panel the Pareto frontier (the models no other
model beats on *both* axes) is drawn as a line; dominated models sit off it.

Every panel shares the FD-gain y-axis. The x-axis is a cost (lower = better) in
the first three panels and noise tolerance (higher = better) in the fourth, so
the "ideal" corner is top-left for costs and top-right for tolerance.

Usage: python benchmark_pareto.py [input.csv] [output.png]
Output: results/benchmark/benchmark_pareto.png (or the given output path)
"""

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark/benchmark.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/benchmark/benchmark_pareto.png"

# one stable color per architecture family (across every seq length)
COLORS = {
    "conformer": "#d62728",
    "mamba": "#9467bd",
    "gru": "#2ca02c",
    "transformer": "#1f77b4",
    "tcn": "#ff7f0e",
    "TSMixer": "#8c564b",
    "patchTST": "#e377c2",
    "dlinear": "#7f7f7f",
    "nlinear": "#bcbd22",
    "gru_distilled": "#17becf",
}
DOT_SIZE = 340  # marker area (pt^2) — big enough to hold the seq-length label
LABEL_FS = 8  # font size of the seq-length text drawn inside each dot

rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        rows.append(
            {
                "arch": r["arch"],
                "seq": int(r["seq"]),
                "fdg": float(r["fdg"]),
                "latency_ms": float(r["latency_ms"]),
                "flops_m": float(r["flops_m"]),
                "size_mb": float(r["size_mb"]),
                "degrade": float(r["degrade"]),
                "tolerance": float(r["tolerance"]),
            }
        )


def pareto_front(cost, gain):
    """Indices on the frontier: no point has lower cost AND higher gain.

    Returns them sorted by increasing cost so they can be drawn as a line. For a
    "higher-is-better" x-axis, pass its negation as `cost`.
    """
    order = sorted(range(len(cost)), key=lambda i: (cost[i], -gain[i]))
    front, best_gain = [], -np.inf
    for i in order:
        if gain[i] > best_gain:  # cheaper points already passed; keep only gain gains
            front.append(i)
            best_gain = gain[i]
    return front


def scatter_panel(ax, x_key, xlabel, logx=True, higher_better_x=False):
    x = np.array([r[x_key] for r in rows])
    gain = np.array([r["fdg"] for r in rows])
    for i, r in enumerate(rows):
        c = COLORS.get(r["arch"], "black")
        ax.scatter(
            x[i], gain[i], s=DOT_SIZE, color=c, zorder=3, edgecolor="white", lw=0.8
        )
        ax.annotate(
            str(r["seq"]),
            (x[i], gain[i]),
            fontsize=LABEL_FS,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            zorder=4,
        )
    front = pareto_front(-x if higher_better_x else x, gain)
    ax.plot(x[front], gain[front], "-", color="black", lw=1.3, alpha=0.55, zorder=2)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("FD-gain")
    ax.grid(True, which="both", alpha=0.25)


fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(
    "Architecture benchmark — accuracy vs deployment cost & robustness\n"
    "(dot label = input window length; black line = Pareto frontier)",
    fontsize=13,
)

scatter_panel(axes[0, 0], "latency_ms", "single-frame latency (ms, log)")
scatter_panel(axes[0, 1], "flops_m", "FLOPs per window (M, log)")
scatter_panel(axes[1, 0], "size_mb", "float32 size (MB, log)")
scatter_panel(
    axes[1, 1],
    "tolerance",
    "noise tolerance σ (σ at which model loses to clean baseline)",
    logx=False,
    higher_better_x=True,
)
axes[1, 1].set_title("Accuracy vs noise robustness", fontsize=10)

# single figure-level legend below the panels, so nothing covers the points
handles = [
    plt.Line2D([], [], marker="o", ls="", ms=10, color=COLORS[a], label=a)
    for a in COLORS
    if any(r["arch"] == a for r in rows)
]
handles.append(plt.Line2D([], [], color="black", lw=1.3, alpha=0.55, label="Pareto frontier"))
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=len(handles),
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
)

fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(OUT, bbox_inches="tight", dpi=130)
plt.close(fig)
print(f"saved {OUT}")

# also print the frontier members for each axis
for x_key, name, hb in [
    ("latency_ms", "latency", False),
    ("flops_m", "FLOPs", False),
    ("size_mb", "size", False),
    ("tolerance", "tolerance", True),
]:
    x = np.array([r[x_key] for r in rows])
    gain = np.array([r["fdg"] for r in rows])
    front = pareto_front(-x if hb else x, gain)
    members = ", ".join(f"{rows[i]['arch']}/{rows[i]['seq']}" for i in front)
    print(f"Pareto frontier ({name}): {members}")
