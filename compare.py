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

from models import get_device
from sweep import make_figures, run_eval

CHECKPOINT_DIR = sys.argv[1]
GROUP_BY = sys.argv[2]  # e.g. "train.beta" or "model.type"
RESULTS_DIR = "results/compare"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = get_device()

# one checkpoint per group, keyed by the chosen config field
results = {}
# look for all .pth files in the folder, evaluate each of them
for f in sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))):
    r = run_eval(f, device, group_field=GROUP_BY)
    print(f"{GROUP_BY}={r['group']}  ({os.path.basename(f)})")
    results[r["group"]] = r

# numeric groups sort numerically, otherwise lexically
try:
    groups = sorted(results.keys(), key=float)
except (TypeError, ValueError):
    groups = sorted(results.keys(), key=str)
print(f"groups ({GROUP_BY}): {groups}")
xlabels = [f"{g:g}" if isinstance(g, (int, float)) else str(g) for g in groups]

make_figures(
    results,
    groups,
    xlabels,
    GROUP_BY,
    RESULTS_DIR,
    show_epoch_col=True,
    summary_title=f"{GROUP_BY} sweep summary (green = best per column)",
)

print("\ndone — outputs in", RESULTS_DIR)
