"""Evaluate model robustness by adding noise to the test-set input.

Noise is added to the model input only; the target stays clean (it is the true
next motion we want to predict). The previous-frame baseline is the model's
yardstick, but adding noise also corrupts that baseline — it *is* the last input
frame — turning FD-gain under noise into a ratio of two moving targets. So we pin
the baseline to its CLEAN value and score every noisy model against it: the FD
curve becomes a true absolute-degradation curve and FD-gain measures how much of
the clean advantage survives input degradation.

Usage: python robustness.py checkpoints/generalist/gru_...pth
"""

import os
import sys

from models import get_device
from sweep import make_figures, run_eval

CHECKPOINT_PATH = sys.argv[1]
NOISE_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
GROUP_BY = "noise level"

tag = os.path.basename(CHECKPOINT_PATH).removesuffix(".pth")
RESULTS_DIR = f"results/robustness/{tag}"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = get_device()

# evaluate the checkpoint at each noise level (one checkpoint per group).
# first get the clean baseline once — it is the fixed reference every noisy model
# is scored against (same samples/order across runs since shuffle=False).
print(f"Processing {os.path.basename(CHECKPOINT_PATH)}...")
clean = run_eval(CHECKPOINT_PATH, device, progress=True)
base_arr, base_mean = clean["fd_base_arr"], clean["fd_base"]
print(f"  clean baseline FD = {base_mean:.3f}")

results = {}
for noise in NOISE_LEVELS:
    r = run_eval(CHECKPOINT_PATH, device, noise=noise, progress=True)
    # repin the baseline to its clean value and recompute the gain against it
    fdg_ps = (base_arr - r["fd_pred_arr"]) / (base_arr + 1e-6)
    r["fd_base"] = base_mean
    r["fd_base_arr"] = base_arr
    r["fdg"] = float(fdg_ps.mean())
    r["fdg_per_sample"] = fdg_ps
    print(
        f"  noise={noise:g}: FD={r['fd_pred']:.3f} (clean base {base_mean:.3f}), "
        f"FDg={r['fdg']:.3f}, NLL={r['nll']:.3f}"
    )
    results[noise] = r

groups = sorted(results.keys())
print(f"\nnoise levels: {groups}")
xlabels = [f"{g:g}" for g in groups]

make_figures(
    results,
    groups,
    xlabels,
    GROUP_BY,
    RESULTS_DIR,
    title_prefix="Robustness: ",
    fig_names=(
        "01_robustness_headline",
        "02_robustness_calibration",
        "03_robustness_per_dim",
        "04_robustness_fd_dist",
        "05_robustness_summary",
    ),
    show_epoch_col=False,
    summary_title="Robustness sweep summary (green = best per column)",
)

print("\ndone — outputs in", RESULTS_DIR)
