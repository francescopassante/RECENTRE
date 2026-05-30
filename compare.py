"""Compare a folder of checkpoints, grouped by one config field.

Each checkpoint carries its own config, so this works for any sweep — set
GROUP_BY to the config field that varies (e.g. "train.beta" for a β-sweep, or
"model.type" to compare architectures). Runs that share a group value are
averaged (mean ± std shown as error bars / bands).

Usage: python compare.py [checkpoints/beta_scan]
"""

import glob
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from metrics import evaluate
from models import build_model

DIM_NAMES = ["Tx (mm)", "Ty (mm)", "Tz (mm)", "Rx (mm)", "Ry (mm)", "Rz (mm)"]
CHECKPOINT_DIR = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/beta_scan"
GROUP_BY = "train.beta"  # dotted path into each checkpoint's config
RESULTS_DIR = "results/compare"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"device: {device}")


def save(fig, name):
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def config_value(config, dotted):
    """Read a nested config field given a dotted path like 'train.beta'."""
    value = config
    for part in dotted.split("."):
        value = value[part]
    return value


def run_eval(checkpoint_path):
    """Evaluate one checkpoint over all its test tasks; return per-run metrics."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    mu, sigma = ckpt["mu"], ckpt["sigma"]
    test_ids = ckpt["test_ids"]

    model = build_model(config["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_tasks = parse_task(config["data"]["test_task"])
    preds, trues, stds, fd_ps, fd_bs = [], [], [], [], []
    nll_sum = 0.0
    for task in test_tasks:
        task_dict = np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()
        data = (np.array([task_dict[pid] for pid in test_ids]) - mu) / sigma
        loader = GPUBatchLoader(TimeSeriesDataset(data, test_ids, device=device), batch_size=1024, shuffle=False)
        out = evaluate(model, loader, mu, sigma, device)
        # rotations ×50 -> mm for the value-level (MAE / σ) metrics
        p, t, s = out["pred"].copy(), out["true"].copy(), out["std"].copy()
        p[:, 3:6] *= 50
        t[:, 3:6] *= 50
        s[:, 3:6] *= 50
        preds.append(p)
        trues.append(t)
        stds.append(s)
        fd_ps.append(out["fd_pred"])
        fd_bs.append(out["fd_base"])
        nll_sum += out["nll"]

    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    std = np.concatenate(stds)
    fd_p = np.concatenate(fd_ps)
    fd_b = np.concatenate(fd_bs)
    err = pred - true

    return {
        "nll": nll_sum / len(test_tasks),
        "fd_base": float(fd_b.mean()),
        "fd_pred": float(fd_p.mean()),
        "fdg": float(((fd_b - fd_p) / fd_b).mean()),
        "fdg_per_sample": (fd_b - fd_p) / fd_b,
        "fd_pred_arr": fd_p,
        "fd_base_arr": fd_b,
        "mae_dim": np.abs(err).mean(axis=0),
        "mae_total": float(np.abs(err).mean()),
        "std_dim": std.mean(axis=0),
        "std_total": float(std.mean()),
        "cov_1sigma": float((np.abs(err) <= std).mean()),
        "cov_2sigma": float((np.abs(err) <= 2 * std).mean()),
        "epoch": ckpt["best_epoch"],
        "group": config_value(config, GROUP_BY),
    }


# group checkpoints by the chosen config field
per_run = defaultdict(list)
for f in sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))):
    r = run_eval(f)
    print(f"  {GROUP_BY}={r['group']}  ({os.path.basename(f)})")
    if r["epoch"] < 50:
        print(f"    WARNING: best epoch {r['epoch']} (under-trained?)")
    per_run[r["group"]].append(r)


def aggregate(runs):
    """Combine per-run dicts: mean+std for scalars/per-dim, concat for distributions."""
    agg = {"n_runs": len(runs), "epochs": [r["epoch"] for r in runs]}
    for k in ["nll", "fd_base", "fd_pred", "fdg", "mae_total", "std_total", "cov_1sigma", "cov_2sigma"]:
        v = np.array([r[k] for r in runs])
        agg[k] = float(v.mean())
        agg[k + "_std"] = float(v.std(ddof=0))
    for k in ["mae_dim", "std_dim"]:
        v = np.stack([r[k] for r in runs], axis=0)
        agg[k] = v.mean(axis=0)
        agg[k + "_std"] = v.std(axis=0, ddof=0)
    for k in ["fdg_per_sample", "fd_pred_arr", "fd_base_arr"]:
        agg[k] = np.concatenate([r[k] for r in runs], axis=0)
    return agg


results = {g: aggregate(rs) for g, rs in per_run.items()}
# numeric groups sort numerically, otherwise lexically
try:
    groups = sorted(results.keys(), key=float)
except (TypeError, ValueError):
    groups = sorted(results.keys(), key=str)
print(f"groups ({GROUP_BY}): {groups}")
x = np.arange(len(groups))
xlabels = [f"{g:g}" if isinstance(g, (int, float)) else str(g) for g in groups]


def scalar(key):
    return (
        np.array([results[g][key] for g in groups]),
        np.array([results[g][key + "_std"] for g in groups]),
    )


# 1) Headline metrics vs group
nll, nll_e = scalar("nll")
fd_pred, fd_pred_e = scalar("fd_pred")
fd_base, fd_base_e = scalar("fd_base")
fdg, fdg_e = scalar("fdg")
mae_total, mae_total_e = scalar("mae_total")
std_total, std_total_e = scalar("std_total")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Headline metrics vs {GROUP_BY}")

ax = axes[0, 0]
ax.errorbar(x, nll, yerr=nll_e, fmt="o-", color="blue", capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_xlabel(GROUP_BY)
ax.set_ylabel("Test NLL")
ax.set_title("NLL — lower is better")

ax = axes[0, 1]
ax.errorbar(x, fd_pred, yerr=fd_pred_e, fmt="o-", color="blue", capsize=4, label="Model")
ax.axhline(fd_base.mean(), linestyle="--", color="red", label=f"Baseline ({fd_base.mean():.3f})")
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_xlabel(GROUP_BY)
ax.set_ylabel("Mean FD (mm)")
ax.set_title("FD — lower is better")
ax.legend()

ax = axes[1, 0]
ax.errorbar(x, fdg, yerr=fdg_e, fmt="o-", color="green", capsize=4)
ax.axhline(0, color="black")
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_xlabel(GROUP_BY)
ax.set_ylabel("FD gain")
ax.set_title("FD gain — higher is better")

ax = axes[1, 1]
ax.errorbar(x, mae_total, yerr=mae_total_e, fmt="o-", color="blue", capsize=4, label="MAE")
ax.errorbar(x, std_total, yerr=std_total_e, fmt="s--", color="red", capsize=4, label="predicted σ")
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_xlabel(GROUP_BY)
ax.set_ylabel("mm")
ax.set_title("MAE vs predicted σ")
ax.legend()

fig.tight_layout()
save(fig, "01_headline_metrics")

# 2) Calibration vs group — coverage of predictive ±1σ and ±2σ intervals
cov1, cov1_e = scalar("cov_1sigma")
cov2, cov2_e = scalar("cov_2sigma")
cov1, cov1_e = cov1 * 100, cov1_e * 100
cov2, cov2_e = cov2 * 100, cov2_e * 100

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(x, cov1, yerr=cov1_e, fmt="o-", color="blue", capsize=4, label="±1σ coverage")
ax.errorbar(x, cov2, yerr=cov2_e, fmt="o-", color="purple", capsize=4, label="±2σ coverage")
ax.axhline(68.27, linestyle="--", color="blue", alpha=0.5, label="Gaussian target 68.27%")
ax.axhline(95.45, linestyle="--", color="purple", alpha=0.5, label="Gaussian target 95.45%")
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_xlabel(GROUP_BY)
ax.set_ylabel("empirical coverage (%)")
ax.set_title(f"Calibration vs {GROUP_BY}")
ax.set_ylim(0, 105)
ax.legend(loc="lower left")
fig.tight_layout()
save(fig, "02_calibration")

# 3) Per-dimension MAE & predicted σ vs group
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
dim_colors = ["red", "orange", "green", "blue", "purple", "brown"]
for ax, key, ylabel, title in zip(
    axes,
    ["mae_dim", "std_dim"],
    ["MAE (mm)", "mean predicted σ (mm)"],
    [f"Per-dim MAE vs {GROUP_BY}", f"Per-dim predicted σ vs {GROUP_BY}"],
):
    arr = np.stack([results[g][key] for g in groups], axis=0)  # (n_groups, 6)
    arr_e = np.stack([results[g][key + "_std"] for g in groups], axis=0)
    for d, name in enumerate(DIM_NAMES):
        ax.errorbar(x, arr[:, d], yerr=arr_e[:, d], fmt="o-", color=dim_colors[d], capsize=3, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(GROUP_BY)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "03_per_dim_curves")

# 4) FD distribution per group (overlaid)
palette = ["blue", "green", "orange", "red", "purple", "brown", "black"]
group_color = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# baseline FD is the same across groups (data only) — use one of them
base_fd = results[groups[0]]["fd_base_arr"]
hi = float(np.percentile(base_fd, 99.5))
bins = np.linspace(0, hi, 60)
ax = axes[0]
ax.hist(base_fd, bins=bins, color="gray", alpha=0.4, label="Baseline")
for g in groups:
    ax.hist(results[g]["fd_pred_arr"], bins=bins, histtype="step", lw=1.5, color=group_color[g], label=f"{g}")
ax.set_xlabel("framewise displacement (mm)")
ax.set_ylabel("count")
ax.set_title(f"FD distribution — model (per {GROUP_BY}) vs baseline")
ax.legend(fontsize=9)

ax = axes[1]
# per-sample FD gain — clip tails so the bulk is visible
xlim = max(np.percentile(np.abs(results[g]["fdg_per_sample"]), 99) for g in groups)
bins_g = np.linspace(-xlim, xlim, 60)
for g in groups:
    gg = np.clip(results[g]["fdg_per_sample"], -xlim, xlim)
    ax.hist(gg, bins=bins_g, histtype="step", lw=1.5, color=group_color[g], label=f"{g}")
ax.axvline(0, color="black", linestyle="--")
ax.set_xlim(-xlim, xlim)
ax.set_xlabel("FD gain (clipped to 1–99%)")
ax.set_ylabel("count")
ax.set_title(f"Per-sample FD gain distribution per {GROUP_BY}")
ax.legend(fontsize=9)
fig.tight_layout()
save(fig, "04_fd_distributions")


# 5) Summary table (all numbers in one figure)
def fmt(mean, std):
    """Format as 'mean ± std', or just 'mean' if single-run."""
    if std == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


fig, ax = plt.subplots(figsize=(14, 0.7 + 0.45 * len(groups)))
ax.axis("off")
header = [GROUP_BY, "n runs", "epochs", "NLL", "FD base", "FD model", "FDg", "MAE", "mean σ", "±1σ %", "±2σ %"]
rows = []
for g in groups:
    r = results[g]
    rows.append(
        [
            xlabels[groups.index(g)],
            f"{r['n_runs']}",
            ", ".join(str(e) for e in r["epochs"]),
            fmt(r["nll"], r["nll_std"]),
            fmt(r["fd_base"], r["fd_base_std"]),
            fmt(r["fd_pred"], r["fd_pred_std"]),
            fmt(r["fdg"], r["fdg_std"]),
            fmt(r["mae_total"], r["mae_total_std"]),
            fmt(r["std_total"], r["std_total_std"]),
            fmt(r["cov_1sigma"] * 100, r["cov_1sigma_std"] * 100),
            fmt(r["cov_2sigma"] * 100, r["cov_2sigma_std"] * 100),
        ]
    )

tbl = ax.table(cellText=rows, colLabels=header, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.4)

# highlight the best group for each metric column (green cell)
best = {
    3: int(np.argmin([results[g]["nll"] for g in groups])),  # NLL — lower is better
    5: int(np.argmin([results[g]["fd_pred"] for g in groups])),  # FD model — lower is better
    6: int(np.argmax([results[g]["fdg"] for g in groups])),  # FD gain — higher is better
    7: int(np.argmin([results[g]["mae_total"] for g in groups])),  # MAE — lower is better
    9: int(np.argmin([abs(results[g]["cov_1sigma"] * 100 - 68.27) for g in groups])),  # closest to 68.27
    10: int(np.argmin([abs(results[g]["cov_2sigma"] * 100 - 95.45) for g in groups])),  # closest to 95.45
}
for col, row in best.items():
    tbl[(row + 1, col)].set_facecolor("lightgreen")

fig.suptitle(f"{GROUP_BY} sweep summary (mean ± std across runs; green = best per column)")
fig.tight_layout()
save(fig, "05_summary_table")

print("\ndone — outputs in", RESULTS_DIR)
