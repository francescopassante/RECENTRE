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
Kinematic-feature ablation: same hyperparameters as configs/gru_generalist.yaml
but with sequence_length 32, sweeping which kinematic channels are appended to
the raw positions. Three variants:
  - none    : positions only            (input_dim 6)
  - vel     : positions + velocity       (input_dim 12)
  - vel+acc : positions + velocity + acc (input_dim 18)
Trains one model per variant with train.py, then evaluates each checkpoint and
prints mean FD, mean FD gain, and % of frames with FD gain > 0.

Usage: python feature_ablation.py
"""

# (label, add_velocity, add_acceleration, input_dim)
VARIANTS = [
    ("none", False, False, 6),
    ("vel", True, False, 12),
    ("vel+acc", True, True, 18),
]
BASE_CONFIG = "configs/gru_generalist.yaml"
CONFIG_DIR = "configs/feature_ablation"
OUTPUT_DIR = "checkpoints/feature_ablation"

os.makedirs(CONFIG_DIR, exist_ok=True)

base_config = yaml.safe_load(open(BASE_CONFIG))

checkpoint_paths = {}
for label, add_vel, add_acc, input_dim in VARIANTS:
    config = copy.deepcopy(base_config)
    # each variant writes to its own dir so the checkpoint filenames (identical
    # across variants: same type/task/beta/epochs) don't collide, which lets us
    # derive the path deterministically instead of parsing train.py's stdout
    out_dir = os.path.join(OUTPUT_DIR, label)
    config["output_dir"] = out_dir
    config["data"]["sequence_length"] = 32
    config["data"]["add_velocity"] = add_vel
    config["data"]["add_acceleration"] = add_acc
    config["model"]["input_dim"] = input_dim

    config_path = os.path.join(CONFIG_DIR, f"gru_seq32_{label}.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    d, t = config["data"], config["train"]
    name = f"{config['model']['type']}_{d['train_task']}v{d['test_task']}_beta{t['beta']}_ep{t['epochs']}"
    checkpoint_paths[label] = os.path.join(out_dir, f"{name}.pth")

    print(f"\n=== training {label} (input_dim={input_dim}) ===")
    # let train.py inherit our stdout/stderr so its tqdm progress bar renders
    # live in the terminal (piping it would swallow the carriage-return updates)
    subprocess.run([sys.executable, "train.py", config_path], check=True)

"""
=========
Evaluation
=========
"""

device = get_device()
rows = []
for label, ckpt_path in checkpoint_paths.items():
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
        (label, fd_pred.mean(), fd_gain.mean(), 100 * (fd_gain > 0).mean())
    )

print(f"\n{'variant':>10}  {'mean FD':>10}  {'mean FD gain':>13}  {'% frames gain>0':>16}")
for label, mean_fd, mean_gain, pct_positive in rows:
    print(f"{label:>10}  {mean_fd:>10.4f}  {mean_gain:>13.4f}  {pct_positive:>15.1f}%")
