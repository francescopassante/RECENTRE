"""β-sweep analysis.

Trains the same architecture with different β values weighting the FD-gain
term in the loss (`loss = nll - β * fdg`). β=0 is pure GaussianNLL; β=100
gives heavy weight to point-prediction accuracy at the expense of
likelihood calibration.

Loads every `*.pth` under `checkpoints/` (second runs are named with
"(2)" at the end of the filename), re-runs the test loop, and produces
a suite of comparison plots in `results/beta_scan/`. Per-β metrics are
averaged across the runs found, with std shown as error bars / bands.
"""

import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from GRU import GRUModel
from metrics import fd, fd_gain
from preprocessing import get_task_dict, load_data
from TimeSeriesDataset import GPUBatchLoader, TimeSeriesDataset

DIM_NAMES = ["Tx (mm)", "Ty (mm)", "Tz (mm)", "Rx (mm)", "Ry (mm)", "Rz (mm)"]
RESULTS_DIR = "results/beta_scan"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def parse_beta(path):
    m = re.search(r"beta([0-9]+(?:\.[0-9]+)?)_", path)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(checkpoint_path, device):
    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    test_dict = sd["test_dict"]  # {pid: patient_frames}
    test_ids = list(test_dict.keys())
    mu = sd["mu"]
    sigma = sd["sigma"]

    test_data = np.array([test_dict[pid] for pid in test_ids])
    test_data = (test_data - mu) / sigma
    test_ds = TimeSeriesDataset(test_data, test_ids, device=device)
    loader = GPUBatchLoader(test_ds, batch_size=1024, shuffle=False)

    model = GRUModel(
        input_dim=6, hidden_dim=128, output_dim=6, num_layers=2, dropout=0.5
    ).to(device)
    model.load_state_dict(sd["model_state_dict"])
    model.eval()

    criterion = torch.nn.GaussianNLLLoss()
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)

    preds, trues, bases, stds = [], [], [], []
    fd_b_list, fd_p_list = [], []
    nll_sum, n_batches = 0.0, 0
    with torch.no_grad():
        for _, x, y in loader:
            x, y = x.to(device), y.to(device)
            # GRU.forward returns (mean, variance) — already exp'd inside the model.
            y_pred, y_var = model(x)
            nll_sum += criterion(y_pred, y, y_var).item()
            n_batches += 1

            last_x = x[:, -1, :]
            fd_b_list.append(fd(last_x, y, mu_t, sigma_t).cpu().numpy())
            fd_p_list.append(fd(y_pred, y, mu_t, sigma_t).cpu().numpy())

            preds.append((y_pred * sigma_t + mu_t).cpu().numpy())
            trues.append((y * sigma_t + mu_t).cpu().numpy())
            bases.append((last_x * sigma_t + mu_t).cpu().numpy())
            stds.append((torch.sqrt(y_var) * sigma_t).cpu().numpy())

    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    base = np.concatenate(bases)
    std = np.concatenate(stds)
    fd_b = np.concatenate(fd_b_list)
    fd_p = np.concatenate(fd_p_list)

    # rotations -> mm
    pred[:, 3:] *= 50
    true[:, 3:] *= 50
    base[:, 3:] *= 50
    std[:, 3:] *= 50
    err = pred - true

    fdg = (fd_b - fd_p) / fd_b

    return {
        "nll": nll_sum / n_batches,
        "fd_base": float(fd_b.mean()),
        "fd_pred": float(fd_p.mean()),
        "fdg": float(fdg.mean()),
        "fdg_per_sample": fdg,
        "fd_pred_arr": fd_p,
        "fd_base_arr": fd_b,
        "mae_dim": np.abs(err).mean(axis=0),
        "mae_total": float(np.abs(err).mean()),
        "std_dim": std.mean(axis=0),
        "std_total": float(std.mean()),
        "cov_1sigma": float((np.abs(err) <= std).mean()),
        "cov_2sigma": float((np.abs(err) <= 2 * std).mean()),
        "epoch": sd["epoch"],
    }


# ---------------------------------------------------------------------------
# Main: load data once, sweep over checkpoints
# ---------------------------------------------------------------------------
device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"device: {device}")


CHECKPOINT_DIR = "checkpoints"

# group checkpoints by β; files ending in "(2).pth" are second runs
by_beta = defaultdict(list)
for f in glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")):
    b = parse_beta(f)
    if b is not None:
        by_beta[b].append(f)

per_run = defaultdict(list)  # β -> [run_result_dict, ...]
for beta in sorted(by_beta.keys()):
    for path in by_beta[beta]:
        print(f"  β={beta:<6}  ({os.path.basename(path)})")
        r = evaluate(path, device)
        per_run[beta].append(r)
        if r["epoch"] < 50:
            print(f"    WARNING: epoch {r['epoch']} (under-trained?)")


def aggregate(runs):
    """Combine per-run dicts: mean+std for scalars/per-dim, concat for distributions."""
    agg = {"n_runs": len(runs), "epochs": [r["epoch"] for r in runs]}
    scalars = [
        "nll",
        "fd_base",
        "fd_pred",
        "fdg",
        "mae_total",
        "std_total",
        "cov_1sigma",
        "cov_2sigma",
    ]
    for k in scalars:
        v = np.array([r[k] for r in runs])
        agg[k] = float(v.mean())
        agg[k + "_std"] = float(v.std(ddof=0))
    per_dim = ["mae_dim", "std_dim"]
    for k in per_dim:
        v = np.stack([r[k] for r in runs], axis=0)
        agg[k] = v.mean(axis=0)
        agg[k + "_std"] = v.std(axis=0, ddof=0)
    for k in ["fdg_per_sample", "fd_pred_arr", "fd_base_arr"]:
        agg[k] = np.concatenate([r[k] for r in runs], axis=0)
    return agg


results = {b: aggregate(rs) for b, rs in per_run.items()}
betas = sorted(results.keys())
print(f"betas: {betas}")
n_betas = len(betas)
x = np.arange(n_betas)
xlabels = [f"{b:g}" for b in betas]


def scalar(key):
    return (
        np.array([results[b][key] for b in betas]),
        np.array([results[b][key + "_std"] for b in betas]),
    )


# ---------------------------------------------------------------------------
# 1) Headline metrics vs β
# ---------------------------------------------------------------------------
nll, nll_e = scalar("nll")
fd_pred, fd_pred_e = scalar("fd_pred")
fd_base, fd_base_e = scalar("fd_base")
fdg, fdg_e = scalar("fdg")
mae_total, mae_total_e = scalar("mae_total")
std_total, std_total_e = scalar("std_total")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Headline metrics vs β")

ax = axes[0, 0]
ax.errorbar(x, nll, yerr=nll_e, fmt="o-", color="blue", capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_xlabel("β")
ax.set_ylabel("Test NLL")
ax.set_title("NLL — lower is better")

ax = axes[0, 1]
ax.errorbar(
    x, fd_pred, yerr=fd_pred_e, fmt="o-", color="blue", capsize=4, label="Model"
)
ax.axhline(
    fd_base.mean(),
    linestyle="--",
    color="red",
    label=f"Baseline ({fd_base.mean():.3f})",
)
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_xlabel("β")
ax.set_ylabel("Mean FD (mm)")
ax.set_title("FD — lower is better")
ax.legend()

ax = axes[1, 0]
ax.errorbar(x, fdg, yerr=fdg_e, fmt="o-", color="green", capsize=4)
ax.axhline(0, color="black")
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_xlabel("β")
ax.set_ylabel("FD gain")
ax.set_title("FD gain — higher is better")

ax = axes[1, 1]
ax.errorbar(
    x, mae_total, yerr=mae_total_e, fmt="o-", color="blue", capsize=4, label="MAE"
)
ax.errorbar(
    x,
    std_total,
    yerr=std_total_e,
    fmt="s--",
    color="red",
    capsize=4,
    label="predicted σ",
)
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_xlabel("β")
ax.set_ylabel("mm")
ax.set_title("MAE vs predicted σ")
ax.legend()

fig.tight_layout()
save(fig, "01_headline_metrics")

# ---------------------------------------------------------------------------
# 2) Calibration vs β — coverage of predictive ±1σ and ±2σ intervals
# ---------------------------------------------------------------------------
cov1, cov1_e = scalar("cov_1sigma")
cov2, cov2_e = scalar("cov_2sigma")
cov1, cov1_e = cov1 * 100, cov1_e * 100
cov2, cov2_e = cov2 * 100, cov2_e * 100

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(
    x, cov1, yerr=cov1_e, fmt="o-", color="blue", capsize=4, label="±1σ coverage"
)
ax.errorbar(
    x, cov2, yerr=cov2_e, fmt="o-", color="purple", capsize=4, label="±2σ coverage"
)
ax.axhline(
    68.27, linestyle="--", color="blue", alpha=0.5, label="Gaussian target 68.27%"
)
ax.axhline(
    95.45, linestyle="--", color="purple", alpha=0.5, label="Gaussian target 95.45%"
)
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_xlabel("β")
ax.set_ylabel("empirical coverage (%)")
ax.set_title("Calibration vs β")
ax.set_ylim(0, 105)
ax.legend(loc="lower left")
fig.tight_layout()
save(fig, "02_calibration_vs_beta")

# ---------------------------------------------------------------------------
# 5) Per-dimension MAE & predicted σ vs β
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
dim_colors = ["red", "orange", "green", "blue", "purple", "brown"]
for ax, key, ylabel, title in zip(
    axes,
    ["mae_dim", "std_dim"],
    ["MAE (mm)", "mean predicted σ (mm)"],
    ["Per-dim MAE vs β", "Per-dim predicted σ vs β"],
):
    arr = np.stack([results[b][key] for b in betas], axis=0)  # (n_betas, 6)
    arr_e = np.stack([results[b][key + "_std"] for b in betas], axis=0)
    for d, name in enumerate(DIM_NAMES):
        ax.errorbar(
            x,
            arr[:, d],
            yerr=arr_e[:, d],
            fmt="o-",
            color=dim_colors[d],
            capsize=3,
            label=name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("β")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "05_per_dim_curves")

# ---------------------------------------------------------------------------
# 7) FD distribution per β (overlaid)
# ---------------------------------------------------------------------------
beta_colors = ["blue", "green", "orange", "red", "purple", "brown", "black"]
beta_color = {b: beta_colors[i % len(beta_colors)] for i, b in enumerate(betas)}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# baseline FD is the same across betas (data only) — use one of them
base_fd = results[betas[0]]["fd_base_arr"]
hi = float(np.percentile(base_fd, 99.5))
bins = np.linspace(0, hi, 60)
ax = axes[0]
ax.hist(base_fd, bins=bins, color="gray", alpha=0.4, label="Baseline")
for b in betas:
    fdp = results[b]["fd_pred_arr"]
    ax.hist(
        fdp, bins=bins, histtype="step", lw=1.5, color=beta_color[b], label=f"β={b:g}"
    )
ax.set_xlabel("framewise displacement (mm)")
ax.set_ylabel("count")
ax.set_title("FD distribution — model (per β) vs baseline")
ax.legend(fontsize=9)

ax = axes[1]
# per-sample FD gain — clip tails so the bulk is visible
xlim = max(np.percentile(np.abs(results[b]["fdg_per_sample"]), 99) for b in betas)
bins_g = np.linspace(-xlim, xlim, 60)
for b in betas:
    g = np.clip(results[b]["fdg_per_sample"], -xlim, xlim)
    ax.hist(
        g, bins=bins_g, histtype="step", lw=1.5, color=beta_color[b], label=f"β={b:g}"
    )
ax.axvline(0, color="black", linestyle="--")
ax.set_xlim(-xlim, xlim)
ax.set_xlabel("FD gain (clipped to 1–99%)")
ax.set_ylabel("count")
ax.set_title("Per-sample FD gain distribution per β")
ax.legend(fontsize=9)
fig.tight_layout()
save(fig, "07_fd_distributions")


# ---------------------------------------------------------------------------
# 8) Summary table (all numbers in one figure)
# ---------------------------------------------------------------------------
def fmt(mean, std):
    """Format as 'mean ± std', or just 'mean' if single-run."""
    if std == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


fig, ax = plt.subplots(figsize=(14, 0.7 + 0.45 * len(betas)))
ax.axis("off")
header = [
    "β",
    "n runs",
    "epochs",
    "NLL",
    "FD base",
    "FD model",
    "FDg",
    "MAE",
    "mean σ",
    "±1σ %",
    "±2σ %",
]
rows = []
for b in betas:
    r = results[b]
    rows.append(
        [
            f"{b:g}",
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

# highlight the best β for each metric column (green cell)
nll_vals = [results[b]["nll"] for b in betas]
fd_pred_vals = [results[b]["fd_pred"] for b in betas]
fdg_vals = [results[b]["fdg"] for b in betas]
mae_vals = [results[b]["mae_total"] for b in betas]
cov1_vals = [abs(results[b]["cov_1sigma"] * 100 - 68.27) for b in betas]
cov2_vals = [abs(results[b]["cov_2sigma"] * 100 - 95.45) for b in betas]

best = {
    3: np.argmin(nll_vals),  # NLL — lower is better
    5: np.argmin(fd_pred_vals),  # FD model — lower is better
    6: np.argmax(fdg_vals),  # FD gain — higher is better
    7: np.argmin(mae_vals),  # MAE — lower is better
    9: np.argmin(cov1_vals),  # ±1σ % — closest to 68.27
    10: np.argmin(cov2_vals),  # ±2σ % — closest to 95.45
}
for col, row in best.items():
    tbl[(row + 1, col)].set_facecolor("lightgreen")

fig.suptitle("β-sweep summary (mean ± std across runs; green = best per column)")
fig.tight_layout()
save(fig, "08_summary_table")

print("\ndone — outputs in", RESULTS_DIR)
