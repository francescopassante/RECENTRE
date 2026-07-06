"""Scan a folder of checkpoints and report calibration alongside FD-gain.

Motivation: a GRU trained with GaussianNLL can still be badly *under-confident*
if model selection / early stopping picks the epoch with the best val FD-gain
(the mean head) before the variance head has tightened. Such a checkpoint shows
a near-zero NLL and a reduced chi-square far below 1 (predicted sigma too large),
even though its FD-gain is fine. This script surfaces that: for every checkpoint
it prints FD-gain, mean NLL, and the standardized-residual diagnostics used in
plots.sigma_calibration (reduced chi^2 = mean(z^2), and |z|<=1 / |z|<=2 coverage).

Usage:
  python check_calibration.py                      # default checkpoints/generalist
  python check_calibration.py checkpoints/generalist
  python check_calibration.py checkpoints/generalist gru   # only files starting 'gru'
"""

import os
import sys

import numpy as np
import torch

from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from metrics import evaluate
from models import build_model, get_device

CKPT_DIR = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/generalist"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else ""  # optional filename prefix filter

device = get_device()

ckpt_files = sorted(
    f for f in os.listdir(CKPT_DIR) if f.endswith(".pth") and f.startswith(PREFIX)
)

# cache the per-task dicts so we load each .npy once, not once per checkpoint
_task_cache = {}


def load_task(task):
    if task not in _task_cache:
        _task_cache[task] = np.load(
            f"datasets/{task}_dict.npy", allow_pickle=True
        ).item()
    return _task_cache[task]


def analyze(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    mu, sigma = ckpt["mu"], ckpt["sigma"]
    test_ids = ckpt["test_ids"]

    model = build_model(config["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_tasks = parse_task(config["data"]["test_task"])
    seq_len = config["data"]["sequence_length"]
    add_velocity = config["data"].get("add_velocity", False)
    add_acceleration = config["data"].get("add_acceleration", False)

    zs, fd_preds, fd_bases = [], [], []
    nll_sum, n = 0.0, 0
    for task in test_tasks:
        data_dict = load_task(task)
        data = np.array([data_dict[pid] for pid in test_ids])
        data = (data - mu) / sigma
        vel_std, acc_std = ckpt.get("feat_std", {}).get(task, (None, None))
        ds = TimeSeriesDataset(
            data,
            test_ids,
            sequence_length=seq_len,
            device=device,
            add_velocity=add_velocity,
            add_acceleration=add_acceleration,
            vel_std=vel_std,
            acc_std=acc_std,
        )
        loader = GPUBatchLoader(ds, batch_size=1024, shuffle=False)
        out = evaluate(model, loader, mu, sigma, device)
        zs.append(out["z"])
        fd_preds.append(out["fd_pred"])
        fd_bases.append(out["fd_base"])
        nll_sum += out["nll"] * len(out["z"])
        n += len(out["z"])

    z = np.concatenate(zs)  # standardized residuals in normalized space
    fd_pred = np.concatenate(fd_preds)
    fd_base = np.concatenate(fd_bases)

    return {
        "type": config["model"]["type"],
        "best_epoch": ckpt.get("best_epoch", "?"),
        "epochs": config["train"].get("epochs", "?"),
        "fdg": ((fd_base - fd_pred) / (fd_base + 1e-6)).mean(),
        "nll": nll_sum / n,
        "chi2": (z**2).mean(),  # reduced chi-square; want ~1.0
        "cov1": (np.abs(z) <= 1).mean() * 100,  # want 68.3
        "cov2": (np.abs(z) <= 2).mean() * 100,  # want 95.4
    }


rows = []
for fname in ckpt_files:
    r = analyze(os.path.join(CKPT_DIR, fname))
    r["fname"] = fname
    rows.append(r)
    print(f"  done: {fname}", flush=True)

# well-calibrated is chi^2 closest to 1; sort by that so outliers surface
rows.sort(key=lambda r: abs(r["chi2"] - 1.0))

header = (
    f"| {'Checkpoint':<50} | {'best/ep':>8} | {'FD_gain':>7} | {'NLL':>8} "
    f"| {'chi2':>6} | {'|z|<=1':>7} | {'|z|<=2':>7} |"
)
sep = (
    f"|{'-' * 52}|{'-' * 9}:|{'-' * 8}:|{'-' * 9}:"
    f"|{'-' * 7}:|{'-' * 8}:|{'-' * 8}:|"
)
print(f"\n{header}")
print(sep)
for r in rows:
    print(
        f"| {r['fname']:<50} | {str(r['best_epoch']) + '/' + str(r['epochs']):>8} "
        f"| {r['fdg']:>7.4f} | {r['nll']:>8.3f} | {r['chi2']:>6.3f} "
        f"| {r['cov1']:>6.1f}% | {r['cov2']:>6.1f}% |"
    )

print(
    "\nWant: chi2~1.0, |z|<=1 ~68.3%, |z|<=2 ~95.4%. "
    "chi2 << 1 => predicted sigma too large (under-confident); "
    "chi2 >> 1 => over-confident."
)
