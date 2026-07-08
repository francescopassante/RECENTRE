import copy
import os
import subprocess
import sys

import numpy as np
import torch
import yaml

from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from metrics import evaluate
from models import build_model, get_device

"""
Gamma sweep: same hyperparameters as configs/gru_generalist.yaml but with
sequence_length 32, sweeping the jerk-penalty weight gamma. Trains one model
per gamma with train.py, then evaluates each checkpoint and prints mean FD,
mean FD gain, and % of frames with FD gain > 0.

Usage: python gamma_sweep.py
"""

GAMMAS = [0, 0.001, 0.01, 0.1, 1]
BASE_CONFIG = "configs/gru_generalist.yaml"
CONFIG_DIR = "configs/gamma_scan"
OUTPUT_DIR = "checkpoints/gamma_scan"

os.makedirs(CONFIG_DIR, exist_ok=True)

base_config = yaml.safe_load(open(BASE_CONFIG))

checkpoint_paths = {}
for gamma in GAMMAS:
    config = copy.deepcopy(base_config)
    config["output_dir"] = OUTPUT_DIR
    config["data"]["sequence_length"] = 32
    config["train"]["gamma"] = gamma

    config_path = os.path.join(CONFIG_DIR, f"gru_seq32_gamma{gamma}.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    print(f"\n=== training gamma={gamma} ===")
    result = subprocess.run(
        [sys.executable, "train.py", config_path], capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"train.py failed for gamma={gamma}")

    saved_line = [l for l in result.stdout.splitlines() if l.startswith("saved ")][-1]
    checkpoint_paths[gamma] = saved_line.split("saved ")[1].split("  (best epoch")[0]

"""
=========
Evaluation
=========
"""

device = get_device()
rows = []
for gamma, ckpt_path in checkpoint_paths.items():
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    mu, sigma = ckpt["mu"], ckpt["sigma"]
    test_ids = ckpt["test_ids"]
    test_task = config["data"]["test_task"]
    test_tasks = parse_task(test_task)

    model = build_model(config["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    fd_preds, fd_bases = [], []
    for task in test_tasks:
        test_dict = np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()
        data = np.array([test_dict[pid] for pid in test_ids])
        ds = TimeSeriesDataset(
            data,
            test_ids,
            sequence_length=config["data"]["sequence_length"],
            device=device,
            add_velocity=config["data"].get("add_velocity", False),
            add_acceleration=config["data"].get("add_acceleration", False),
            mu=mu,
            sigma=sigma,
        )
        loader = GPUBatchLoader(ds, batch_size=1024, shuffle=False)
        out = evaluate(model, loader, mu, sigma, device)
        fd_preds.append(out["fd_pred"])
        fd_bases.append(out["fd_base"])

    fd_pred = np.concatenate(fd_preds)
    fd_base = np.concatenate(fd_bases)
    fd_gain = (fd_base - fd_pred) / (fd_base + 1e-6)

    rows.append(
        (gamma, fd_pred.mean(), fd_gain.mean(), 100 * (fd_gain > 0).mean())
    )

print(f"\n{'gamma':>8}  {'mean FD':>10}  {'mean FD gain':>13}  {'% frames gain>0':>16}")
for gamma, mean_fd, mean_gain, pct_positive in rows:
    print(f"{gamma:>8}  {mean_fd:>10.4f}  {mean_gain:>13.4f}  {pct_positive:>15.1f}%")
