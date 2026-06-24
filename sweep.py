import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from metrics import evaluate
from models import build_model

DIM_NAMES = ["Tx (mm)", "Ty (mm)", "Tz (mm)", "Rx (mm)", "Ry (mm)", "Rz (mm)"]
FIG_NAMES = (
    "01_headline_metrics",
    "02_calibration",
    "03_per_dim_curves",
    "04_fd_distributions",
    "05_summary_table",
)


def config_value(config, dotted):
    value = config
    for part in dotted.split("."):
        value = value[part]
    return value


def run_eval(checkpoint_path, device, group_field=None, noise=None, progress=False):

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    mu, sigma = ckpt["mu"], ckpt["sigma"]
    test_ids = ckpt["test_ids"]

    model = build_model(config["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_tasks = parse_task(config["data"]["test_task"])
    n_tasks = len(test_tasks)
    if progress:
        test_tasks = tqdm(
            test_tasks,
            desc=f"Eval {os.path.basename(checkpoint_path)[:15]} (noise={noise})",
            leave=False,
        )
    preds, trues, stds, fd_ps, fd_bs = [], [], [], [], []
    nll_sum = 0.0
    for task in test_tasks:
        task_dict = np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()
        data = (np.array([task_dict[pid] for pid in test_ids]) - mu) / sigma
        seq_len = config["data"].get("sequence_length", 10)
        loader = GPUBatchLoader(
            TimeSeriesDataset(data, test_ids, sequence_length=seq_len, device=device),
            batch_size=1024,
            shuffle=False,
        )
        out = evaluate(model, loader, mu, sigma, device, noise=noise)
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

    group = config_value(config, group_field) if group_field is not None else noise
    return {
        "nll": nll_sum / n_tasks,
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
        "group": group,
    }


def make_figures(
    results,
    groups,
    xlabels,
    group_by,
    results_dir,
    title_prefix="",
    fig_names=FIG_NAMES,
    show_epoch_col=True,
    summary_title=None,
):
    """Draw the five sweep figures into results_dir.

    `results` maps each group value to a run_eval() dict; `groups`/`xlabels` are
    the (already sorted) group keys and their display labels. The few strings
    that differ between sweeps are parameters: `title_prefix` (figure suptitles),
    `fig_names` (output filenames), `show_epoch_col` (best-epoch column in the
    summary table) and `summary_title`.
    """
    os.makedirs(results_dir, exist_ok=True)

    def save(fig, name):
        path = os.path.join(results_dir, f"{name}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")

    x = np.arange(len(groups))

    def col(key):
        return np.array([results[g][key] for g in groups])

    # 1) Headline metrics vs group
    nll = col("nll")
    fd_pred = col("fd_pred")
    fd_base = col("fd_base")
    fdg = col("fdg")
    mae_total = col("mae_total")
    std_total = col("std_total")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{title_prefix}Headline metrics vs {group_by}")

    ax = axes[0, 0]
    ax.plot(x, nll, "o-", color="blue")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(group_by)
    ax.set_ylabel("Test NLL")
    ax.set_title("NLL")

    ax = axes[0, 1]
    ax.plot(x, fd_pred, "o-", color="blue", label="Model")
    ax.axhline(
        fd_base.mean(),
        linestyle="--",
        color="red",
        label=f"Baseline ({fd_base.mean():.3f})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(group_by)
    ax.set_ylabel("Mean FD (mm)")
    ax.set_title("FD")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(x, fdg, "o-", color="green")
    ax.axhline(0, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(group_by)
    ax.set_ylabel("FD gain")
    ax.set_title("FD gain")

    ax = axes[1, 1]
    ax.plot(x, mae_total, "o-", color="blue", label="MAE")
    ax.plot(x, std_total, "s--", color="red", label="predicted σ")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(group_by)
    ax.set_ylabel("mm")
    ax.set_title("MAE vs predicted σ")
    ax.legend()

    fig.tight_layout()
    save(fig, fig_names[0])

    # 2) Calibration vs group — coverage of predictive ±1σ and ±2σ intervals
    cov1 = col("cov_1sigma") * 100
    cov2 = col("cov_2sigma") * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, cov1, "o-", color="blue", label="±1σ coverage")
    ax.plot(x, cov2, "o-", color="purple", label="±2σ coverage")
    ax.axhline(
        68.27, linestyle="--", color="blue", alpha=0.5, label="Gaussian target 68.27%"
    )
    ax.axhline(
        95.45, linestyle="--", color="purple", alpha=0.5, label="Gaussian target 95.45%"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(group_by)
    ax.set_ylabel("empirical coverage (%)")
    ax.set_title(f"Calibration vs {group_by}")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, fig_names[1])

    # 3) Per-dimension MAE & predicted σ vs group
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    dim_colors = ["red", "orange", "green", "blue", "purple", "brown"]
    for ax, key, ylabel, title in zip(
        axes,
        ["mae_dim", "std_dim"],
        ["MAE (mm)", "mean predicted σ (mm)"],
        [f"Per-dim MAE vs {group_by}", f"Per-dim predicted σ vs {group_by}"],
    ):
        arr = np.stack([results[g][key] for g in groups], axis=0)  # (n_groups, 6)
        for d, name in enumerate(DIM_NAMES):
            ax.plot(x, arr[:, d], "o-", color=dim_colors[d], label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel(group_by)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, fig_names[2])

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
        ax.hist(
            results[g]["fd_pred_arr"],
            bins=bins,
            histtype="step",
            lw=1.5,
            color=group_color[g],
            label=f"{g}",
        )
    ax.set_xlabel("framewise displacement (mm)")
    ax.set_ylabel("count")
    ax.set_title(f"FD distribution — model (per {group_by}) vs baseline")
    ax.legend(fontsize=9)

    ax = axes[1]
    # per-sample FD gain — clip tails so the bulk is visible
    xlim = max(np.percentile(np.abs(results[g]["fdg_per_sample"]), 99) for g in groups)
    bins_g = np.linspace(-xlim, xlim, 60)
    for g in groups:
        gg = np.clip(results[g]["fdg_per_sample"], -xlim, xlim)
        ax.hist(
            gg, bins=bins_g, histtype="step", lw=1.5, color=group_color[g], label=f"{g}"
        )
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlim(-xlim, xlim)
    ax.set_xlabel("FD gain (clipped to 1–99%)")
    ax.set_ylabel("count")
    ax.set_title(f"Per-sample FD gain distribution per {group_by}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, fig_names[3])

    # 5) Summary table (all numbers in one figure)
    fig, ax = plt.subplots(figsize=(14, 0.7 + 0.45 * len(groups)))
    ax.axis("off")
    header = [group_by]
    if show_epoch_col:
        header += ["epoch"]
    header += ["NLL", "FD base", "FD model", "FDg", "MAE", "mean σ", "±1σ %", "±2σ %"]
    rows = []
    for i, g in enumerate(groups):
        r = results[g]
        row = [xlabels[i]]
        if show_epoch_col:
            row += [str(r["epoch"])]
        row += [
            f"{r['nll']:.3f}",
            f"{r['fd_base']:.3f}",
            f"{r['fd_pred']:.3f}",
            f"{r['fdg']:.3f}",
            f"{r['mae_total']:.3f}",
            f"{r['std_total']:.3f}",
            f"{r['cov_1sigma'] * 100:.2f}",
            f"{r['cov_2sigma'] * 100:.2f}",
        ]
        rows.append(row)

    tbl = ax.table(cellText=rows, colLabels=header, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    # highlight the best group for each metric column (green cell)
    offset = 1 if show_epoch_col else 0  # the epoch column shifts the metrics right
    best = {
        offset
        + 1: int(
            np.argmin([results[g]["nll"] for g in groups])
        ),  # NLL — lower is better
        offset
        + 3: int(
            np.argmin([results[g]["fd_pred"] for g in groups])
        ),  # FD model — lower is better
        offset
        + 4: int(
            np.argmax([results[g]["fdg"] for g in groups])
        ),  # FD gain — higher is better
        offset
        + 5: int(
            np.argmin([results[g]["mae_total"] for g in groups])
        ),  # MAE — lower is better
        offset
        + 7: int(
            np.argmin([abs(results[g]["cov_1sigma"] * 100 - 68.27) for g in groups])
        ),  # closest to 68.27
        offset
        + 8: int(
            np.argmin([abs(results[g]["cov_2sigma"] * 100 - 95.45) for g in groups])
        ),  # closest to 95.45
    }
    for col_idx, row in best.items():
        tbl[(row + 1, col_idx)].set_facecolor("lightgreen")

    if summary_title is None:
        summary_title = f"{group_by} sweep summary (green = best per column)"
    fig.suptitle(summary_title)
    fig.tight_layout()
    save(fig, fig_names[4])
