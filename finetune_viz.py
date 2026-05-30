import csv
import os

import numpy as np
import matplotlib.pyplot as plt

from finetuning import DIMS

"""
=======================================================================================
Visual comparison of pretrained vs fine-tuned models from the summary CSV written
by finetune_all.py. Reads the CSV only — no model loading — so it is cheap to
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


if __name__ == "__main__":
    TASK = "M"
    results_dir = os.path.join("results", "finetune")
    csv_path = os.path.join(results_dir, f"ft_{TASK}_before_after.csv")
    cols = load_rows(csv_path)

    fdg_before = cols["fdg_before"]
    fdg_after = cols["fdg_after"]
    delta = cols["delta_fdg"]
    n = len(delta)
    win_rate = 100 * np.mean(delta > 0)

    # 1) paired before/after FD-gain scatter (points above y=x improved)
    plt.figure(figsize=(6, 6))
    plt.scatter(fdg_before, fdg_after, s=18, alpha=0.6)
    lo = float(min(fdg_before.min(), fdg_after.min()))
    hi = float(max(fdg_before.max(), fdg_after.max()))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="no change (y=x)")
    plt.xlabel("FD-gain pretrained")
    plt.ylabel("FD-gain fine-tuned")
    plt.title(f"FD-gain before vs after — {TASK} (n={n}, win rate {win_rate:.0f}%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"ft_{TASK}_fdg_scatter.png"), dpi=150)

    # 2) distribution of ΔFD-gain across patients
    plt.figure(figsize=(8, 5))
    plt.hist(delta, bins=30, color="tab:blue", alpha=0.8)
    plt.axvline(0, color="k", lw=1)
    plt.axvline(delta.mean(), color="tab:red", lw=2, label=f"mean {delta.mean():+.3f}")
    plt.xlabel("ΔFD-gain (fine-tuned − pretrained)")
    plt.ylabel("# patients")
    plt.title(f"ΔFD-gain distribution — {TASK}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"ft_{TASK}_delta_hist.png"), dpi=150)

    # 3) per-dimension mean MSE (physical units), pretrained vs fine-tuned
    mse_before = np.array([cols[f"mse_{d}_before"].mean() for d in DIMS])
    mse_after = np.array([cols[f"mse_{d}_after"].mean() for d in DIMS])
    xpos = np.arange(len(DIMS))
    width = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(xpos - width / 2, mse_before, width, label="pretrained")
    plt.bar(xpos + width / 2, mse_after, width, label="fine-tuned")
    plt.xticks(xpos, DIMS)
    plt.ylabel("mean MSE (physical units)")
    plt.title(f"Per-dimension MSE — {TASK}")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"ft_{TASK}_mse_per_dim.png"), dpi=150)

    # 4) fd_pred before vs after (paired box) against the shared baseline
    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [cols["fd_pred_before"], cols["fd_pred_after"], cols["fd_base"]],
        labels=["pretrained", "fine-tuned", "baseline"],
        showmeans=True,
    )
    plt.ylabel("mean FD per patient")
    plt.title(f"Predicted FD vs previous-frame baseline — {TASK}")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"ft_{TASK}_fd_pred_box.png"), dpi=150)

    print(f"Wrote 4 figures to {results_dir}/ (task {TASK}, n={n})")
    plt.show()
