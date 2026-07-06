"""Re-train miscalibrated checkpoints with the corrected engine.

Background: a variable-shadowing bug in engine.fit (fixed in d8b2963) reassigned
the `loss` string parameter to the loss tensor, so from the second batch onward
`loss == "gaussian_nll"` was False and training silently fell back to MSE. The
variance head therefore never trained -> predicted sigma stuck near its init ->
reduced chi^2 ~ 0 (badly under-confident) while FD-gain (mean only) stayed fine.
Any checkpoint trained after 2026-06-30 is suspect.

This script scans a folder of checkpoints, measures sigma calibration (reduced
chi^2 = mean(z^2) via the standard evaluate() path), flags the ones whose chi^2
is more than --tol away from 1.0, reads the config embedded *inside* each broken
checkpoint, and re-trains an identical run by driving train.py -- the single
training entry point -- so nothing about the recipe is re-guessed here. Retrained
checkpoints are written to a separate --out dir; the originals are left untouched.

Usage:
  python retrain_broken.py                                   # scan checkpoints/generalist
  python retrain_broken.py checkpoints/generalist --tol 0.3
  python retrain_broken.py checkpoints/generalist --dry-run  # list + configs, no training
  python retrain_broken.py checkpoints/generalist --out checkpoints/generalist_fixed
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import yaml

from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from metrics import evaluate
from models import build_model, get_device

device = get_device()

# cache the per-task dicts so each .npy loads once, not once per checkpoint
_task_cache = {}


def load_task(task):
    if task not in _task_cache:
        _task_cache[task] = np.load(
            f"datasets/{task}_dict.npy", allow_pickle=True
        ).item()
    return _task_cache[task]


def calibration_chi2(path):
    """Reduced chi^2 = mean(z^2) on the checkpoint's own held-out test split.
    ~1.0 is calibrated; << 1 under-confident (the MSE bug); >> 1 over-confident.
    Returns (chi2, config) or (None, config) if the model can't be built."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    try:
        model = build_model(config["model"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
    except Exception as e:
        print(f"  ! could not build {os.path.basename(path)}: {e}")
        return None, config

    mu, sigma = ckpt["mu"], ckpt["sigma"]
    test_ids = ckpt["test_ids"]
    seq_len = config["data"]["sequence_length"]
    add_velocity = config["data"].get("add_velocity", False)
    add_acceleration = config["data"].get("add_acceleration", False)

    zs = []
    for task in parse_task(config["data"]["test_task"]):
        data = np.array([load_task(task)[pid] for pid in test_ids])
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
        zs.append(evaluate(model, loader, mu, sigma, device)["z"])
    return float((np.concatenate(zs) ** 2).mean()), config


def retrain(config, out_dir):
    """Dump the embedded config to a temp yaml (only output_dir overridden) and
    run it through train.py. Returns the path of the checkpoint train.py wrote."""
    run_config = dict(config)
    run_config["output_dir"] = out_dir
    os.makedirs(out_dir, exist_ok=True)

    before = set(os.listdir(out_dir))
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, dir=out_dir
    ) as f:
        yaml.safe_dump(run_config, f, default_flow_style=False, sort_keys=False)
        yaml_path = f.name
    try:
        # reuse the single training driver; stream its output (tqdm, "saved ...")
        subprocess.run([sys.executable, "train.py", yaml_path], check=True)
    finally:
        os.remove(yaml_path)

    new = [
        f for f in os.listdir(out_dir) if f.endswith(".pth") and f not in before
    ]
    return os.path.join(out_dir, new[0]) if new else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dir", nargs="?", default="checkpoints/generalist")
    ap.add_argument("--out", default=None, help="output dir (default: <ckpt_dir>_fixed)")
    ap.add_argument(
        "--tol",
        type=float,
        default=0.3,
        help="a checkpoint is 'broken' if |chi2 - 1| > tol (default 0.3)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="list broken + configs, don't train"
    )
    args = ap.parse_args()
    out_dir = args.out or args.ckpt_dir.rstrip("/") + "_fixed"

    files = sorted(f for f in os.listdir(args.ckpt_dir) if f.endswith(".pth"))
    print(f"scanning {len(files)} checkpoints in {args.ckpt_dir} (tol={args.tol})\n")

    broken = []  # (fname, chi2, config)
    for fname in files:
        chi2, config = calibration_chi2(os.path.join(args.ckpt_dir, fname))
        if chi2 is None:
            continue
        status = "ok" if abs(chi2 - 1.0) <= args.tol else "BROKEN"
        print(f"  chi2={chi2:6.3f}  {status:6}  {fname}")
        if status == "BROKEN":
            broken.append((fname, chi2, config))

    if not broken:
        print("\nnothing to retrain -- all checkpoints are within tolerance.")
        return

    print(f"\n{len(broken)} broken checkpoint(s) to retrain -> {out_dir}\n")
    for fname, chi2, config in broken:
        m, t = config["model"]["type"], config["train"]
        d = config["data"]
        print(
            f"  {fname}\n    type={m} seq_len={d['sequence_length']} "
            f"input_dim={config['model']['input_dim']} loss={t['loss']} "
            f"beta={t['beta']} epochs={t['epochs']} lr={t['lr']} (chi2={chi2:.3f})"
        )

    if args.dry_run:
        print("\n--dry-run: not training.")
        return

    # retrain one by one, re-checking calibration of each fresh checkpoint
    results = []
    for i, (fname, _, config) in enumerate(broken, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(broken)}] retraining from {fname}\n{'=' * 70}")
        new_path = retrain(config, out_dir)
        if new_path is None:
            print(f"  ! train.py produced no new checkpoint for {fname}")
            results.append((fname, None, None))
            continue
        new_chi2, _ = calibration_chi2(new_path)
        verdict = "FIXED" if abs(new_chi2 - 1.0) <= args.tol else "still off"
        print(f"  -> {os.path.basename(new_path)}  chi2={new_chi2:.3f}  [{verdict}]")
        results.append((fname, os.path.basename(new_path), new_chi2))

    print(f"\n{'=' * 70}\nsummary\n{'=' * 70}")
    for old, new, chi2 in results:
        if new is None:
            print(f"  {old:52s} -> FAILED")
        else:
            tag = "FIXED" if abs(chi2 - 1.0) <= args.tol else "still off"
            print(f"  {old:52s} -> {new:52s} chi2={chi2:.3f} [{tag}]")


if __name__ == "__main__":
    main()
