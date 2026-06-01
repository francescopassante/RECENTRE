import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import plots
from finetune import DIMS

"""
=======================================================================================
Visual comparison of pretrained vs fine-tuned models from the summary CSV written
by finetune.py. Reads the CSV only — no model loading — so it is cheap to
re-run while iterating on plots.
=======================================================================================
"""


def load_rows(csv_path):
    """Read the summary CSV into a dict of float/str numpy arrays (column-major)."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    cols = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        if key in ("patient_id", "task"):
            cols[key] = np.array(vals)
        else:
            cols[key] = np.array([float(v) for v in vals])
    return cols


# fixed color per task, matching plots.py
TASK_COLORS = {"R": "tab:blue", "M": "tab:orange", "L": "tab:green"}


if __name__ == "__main__":
    TASKS = ["R", "M", "L"]
    results_dir = os.path.join("results", "finetune")
    tag = "+".join(TASKS)
    csv_path = os.path.join(results_dir, f"ft_{tag}_before_after.csv")
    cols = load_rows(csv_path)

    # tasks actually present in the CSV, in canonical R/M/L order
    present = [t for t in ("R", "M", "L") if np.any(cols["task"] == t)]

    fdg_before = cols["fdg_before"]
    fdg_after = cols["fdg_after"]
    fd_pred_before = cols["fd_pred_before"]
    fd_pred_after = cols["fd_pred_after"]
    delta = cols["delta_fdg"]
    n = len(delta)
    win_rate = 100 * np.mean(delta > 0)

    def task_mask(t):
        return cols["task"] == t

    # 1) paired before/after FD-gain scatter, per task (points above y=x improved)
    plt.figure(figsize=(6, 6))
    for t in present:
        m = task_mask(t)
        plt.scatter(
            fdg_before[m], fdg_after[m], s=18, alpha=0.6, color=TASK_COLORS[t], label=t
        )
    lo = float(min(fdg_before.min(), fdg_after.min()))
    hi = float(max(fdg_before.max(), fdg_after.max()))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="no change (y=x)")
    plt.xlabel("FD-gain pretrained")
    plt.ylabel("FD-gain fine-tuned")
    plt.title(f"FD-gain before vs after (n={n}, win rate {win_rate:.0f}%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"ft_{tag}_fdg_scatter.png"), dpi=150)

    # 2) paired before/after predicted-FD scatter, per task (below y=x improved)
    plt.figure(figsize=(6, 6))
    for t in present:
        m = task_mask(t)
        plt.scatter(
            fd_pred_before[m],
            fd_pred_after[m],
            s=18,
            alpha=0.6,
            color=TASK_COLORS[t],
            label=t,
        )
    lo = float(min(fd_pred_before.min(), fd_pred_after.min()))
    hi = float(max(fd_pred_before.max(), fd_pred_after.max()))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="no change (y=x)")
    plt.xlabel("FD pretrained")
    plt.ylabel("FD fine-tuned")
    plt.title("Predicted FD before vs after (below y=x = improved)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"ft_{tag}_fd_pred_scatter.png"), dpi=150)

    # 3) ΔFD-gain vs pretrained FD-gain, per task (does fine-tuning help most
    #    where the generalist was weakest?)
    plt.figure(figsize=(6, 6))
    for t in present:
        m = task_mask(t)
        plt.scatter(
            fdg_before[m], delta[m], s=18, alpha=0.6, color=TASK_COLORS[t], label=t
        )
    # overall moving-average trend across all tasks: sort by pretrained FD-gain,
    # then average ΔFD-gain in a centered window at every point. Windows are
    # truncated at the ends so the line spans the full x-range.
    order = np.argsort(fdg_before)
    xs, ys = fdg_before[order], delta[order]
    w = max(5, len(xs) // 20)
    if len(xs) >= 2:
        h = w // 2
        ys_ma = np.array([ys[max(0, i - h) : i + h + 1].mean() for i in range(len(ys))])
        plt.plot(xs, ys_ma, color="black", lw=2, label=f"moving avg (w={w})")
    plt.axhline(0, color="k", lw=1)
    plt.xlabel("FD-gain pretrained")
    plt.ylabel("ΔFD-gain (fine-tuned − pretrained)")
    plt.title("ΔFD-gain vs pretrained FD-gain")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"ft_{tag}_delta_vs_pretrained.png"), dpi=150)

    # 4) distribution of ΔFD-gain across patients, smoothed per task with a
    #    Gaussian KDE — overlaid histograms were too jagged to tell apart.
    plt.figure(figsize=(8, 5))
    pad = 0.05 * (delta.max() - delta.min())
    grid = np.linspace(delta.min() - pad, delta.max() + pad, 300)
    for t in present:
        d = delta[task_mask(t)]
        # scipy's gaussian_kde needs ≥2 points and non-zero variance
        if d.size < 2 or np.ptp(d) == 0:
            continue
        density = gaussian_kde(d)(grid)
        plt.plot(
            grid,
            density,
            color=TASK_COLORS[t],
            lw=2,
            label=f"{t} (mean {d.mean():+.3f})",
        )
        plt.fill_between(grid, density, color=TASK_COLORS[t], alpha=0.2)
    plt.axvline(0, color="k", lw=1)
    plt.axvline(
        delta.mean(),
        color="tab:red",
        lw=2,
        ls="--",
        label=f"overall mean {delta.mean():+.3f}",
    )
    plt.xlabel("ΔFD-gain (fine-tuned − pretrained)")
    plt.ylabel("density")
    plt.title("ΔFD-gain distribution (Gaussian KDE per task)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"ft_{tag}_delta_kde.png"), dpi=150)

    # 5) per-dimension mean MSE, pretrained vs fine-tuned (aggregated over tasks).
    # Rotations are stored in rad²; scale by (50 mm)² (head radius) so they are
    # comparable to the translation dims in mm² instead of vanishing near zero.
    rot_scale_sq = 50.0**2
    mse_before = np.array([cols[f"mse_{d}_before"].mean() for d in DIMS])
    mse_after = np.array([cols[f"mse_{d}_after"].mean() for d in DIMS])
    mse_before[3:6] *= rot_scale_sq
    mse_after[3:6] *= rot_scale_sq
    xpos = np.arange(len(DIMS))
    width = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(xpos - width / 2, mse_before, width, label="pretrained")
    plt.bar(xpos + width / 2, mse_after, width, label="fine-tuned")
    plt.xticks(xpos, DIMS)
    plt.ylabel("mean MSE (mm²; rotations ×50²)")
    plt.title("Per-dimension MSE — pretrained vs fine-tuned")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"ft_{tag}_mse_per_dim.png"), dpi=150)

    # 6) frame-level win rate before vs after, per task (above y=x = fine-tuning
    #    improved the fraction of frames that beat the previous-frame baseline)
    pct_before = cols["pct_improved_before"]
    pct_after = cols["pct_improved_after"]
    plt.figure(figsize=(6, 6))
    for t in present:
        m = task_mask(t)
        plt.scatter(
            pct_before[m], pct_after[m], s=18, alpha=0.6, color=TASK_COLORS[t], label=t
        )
    lo = float(min(pct_before.min(), pct_after.min()))
    hi = float(max(pct_before.max(), pct_after.max()))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="no change (y=x)")
    plt.xlabel("% frames improved — pretrained")
    plt.ylabel("% frames improved — fine-tuned")
    plt.title("Frame-level win rate before vs after")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"ft_{tag}_pct_improved_scatter.png"), dpi=150
    )

    # 7) per-patient FD gain for the fine-tuned collection — plots.py figure 04,
    #    built straight from the CSV (one scalar per patient, no model needed).
    #    Each patient is scored by its own fine-tuned model on its own test frames.
    fd_pp_pred = {t: fd_pred_after[task_mask(t)] for t in present}
    fd_pp_base = {t: cols["fd_base"][task_mask(t)] for t in present}
    fdg_pp = {t: fdg_after[task_mask(t)] for t in present}
    fig = plots.per_patient_fdg(fd_pp_pred, fd_pp_base, fdg_pp, present)
    fig.savefig(os.path.join(results_dir, f"ft_{tag}_04_per_patient_fdg.png"), dpi=150)
    plt.close(fig)

    # 8+) frame-level figures for the fine-tuned collection, reusing plots.py on the
    #     pooled .npz written by finetune.py. Only available after re-running the sweep.
    npz_path = os.path.join(results_dir, f"ft_{tag}_arrays.npz")
    n_frame_figs = 0
    if os.path.exists(npz_path):
        arr = np.load(npz_path, allow_pickle=True)
        # rotations ×50 -> mm for display, matching evaluate.py's convention
        fpred, ftrue, fbase = arr["pred"].copy(), arr["true"].copy(), arr["base"].copy()
        for a in (fpred, ftrue, fbase):
            a[:, 3:6] *= 50
        flabels = arr["task"]
        fpresent = [t for t in ("R", "M", "L") if np.any(flabels == t)]

        frame_figs = [
            ("01_error_per_dimension", plots.error_per_dimension(fpred, ftrue, fbase, flabels, fpresent)),
            ("02_true_vs_predicted", plots.true_vs_predicted(fpred, ftrue, flabels, fpresent)),
            ("03_fd_distribution", plots.fd_distribution(arr["fd_pred"], arr["fd_base"], flabels, fpresent)),
            ("06_sigma_calibration", plots.sigma_calibration(arr["z"], flabels, fpresent)),
            ("08_fdgain_vs_motion", plots.fdgain_vs_motion(arr["fd_pred"], arr["fd_base"], flabels, fpresent)),
        ]
        for name, fig in frame_figs:
            fig.savefig(os.path.join(results_dir, f"ft_{tag}_{name}.png"), dpi=150)
            plt.close(fig)
        n_frame_figs = len(frame_figs)
    else:
        print(f"(no {npz_path} yet — re-run finetune.py to get frame-level figures)")

    print(
        f"Wrote {7 + n_frame_figs} figures to {results_dir}/ (tasks {present}, n={n})"
    )
    plt.show()
