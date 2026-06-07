"""Evaluate model robustness by adding noise to the test set.

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

# evaluate the checkpoint at each noise level (one checkpoint per group)
print(f"Processing {os.path.basename(CHECKPOINT_PATH)}...")
results = {}
for noise in NOISE_LEVELS:
    r = run_eval(CHECKPOINT_PATH, device, noise=noise, progress=True)
    print(f"  noise={noise:g}: FDg={r['fdg']:.3f}, NLL={r['nll']:.3f}")
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
