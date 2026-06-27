"""Temporary script: load all checkpoints in checkpoints/generalist/ and print a comparison table."""

import os
import numpy as np
import torch
from collections import defaultdict

from dataset import TimeSeriesDataset, GPUBatchLoader, parse_task
from metrics import evaluate
from models import build_model, get_device

CKPT_DIR = "checkpoints/generalist"
# device = get_device()
device = "cpu"

# ── collect checkpoints ──────────────────────────────────────────────
ckpt_files = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith(".pth"))

# ── figure out which config keys vary across checkpoints of each model type ──
configs_by_type = defaultdict(list)
for f in ckpt_files:
    ckpt = torch.load(os.path.join(CKPT_DIR, f), map_location="cpu", weights_only=False)
    configs_by_type[ckpt["config"]["model"]["type"]].append((f, ckpt["config"]))


def flat_dict(d, prefix=""):
    """Flatten a nested dict into dotted keys."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flat_dict(v, key + "."))
        else:
            out[key] = repr(v)
    return out


varying_keys_by_type = {}
for mtype, entries in configs_by_type.items():
    if len(entries) < 2:
        varying_keys_by_type[mtype] = []
        continue
    flats = [flat_dict(cfg) for _, cfg in entries]
    all_keys = sorted(set().union(*flats))
    varying = [k for k in all_keys if len(set(f.get(k) for f in flats)) > 1]
    varying_keys_by_type[mtype] = varying


def short_label(config, mtype):
    """Build a label showing model type + only the parameters that vary."""
    varying = varying_keys_by_type.get(mtype, [])
    if not varying:
        return mtype
    flat = flat_dict(config)
    seen = {}
    parts = []
    for k in varying:
        short_k = k.split(".")[-1]
        val = flat.get(k, "?")
        # deduplicate keys with the same short name and same value
        if short_k in seen and seen[short_k] == val:
            continue
        seen[short_k] = val
        parts.append(f"{short_k}={val}")
    return f"{mtype} ({', '.join(parts)})"


# ── evaluate each checkpoint ─────────────────────────────────────────
rows = []
for fname in ckpt_files:
    path = os.path.join(CKPT_DIR, fname)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    mu, sigma = ckpt["mu"], ckpt["sigma"]
    test_ids = ckpt["test_ids"]
    mtype = config["model"]["type"]

    model = build_model(config["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_tasks = parse_task(config["data"]["test_task"])
    seq_len = config["data"]["sequence_length"]
    add_velocity = config["data"].get("add_velocity", False)
    add_acceleration = config["data"].get("add_acceleration", False)

    all_fd_pred, all_fd_base = [], []
    for task in test_tasks:
        data_dict = np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()
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
        all_fd_pred.append(out["fd_pred"])
        all_fd_base.append(out["fd_base"])

    fd_pred = np.concatenate(all_fd_pred)
    fd_base = np.concatenate(all_fd_base)
    fd_gain = (fd_base - fd_pred) / (fd_base + 1e-6)

    label = short_label(config, mtype)
    n_params = sum(p.numel() for p in model.parameters())
    rows.append(
        {
            "label": label,
            "fname": fname,
            "seq_len": seq_len,
            "mean_fd_pred": fd_pred.mean(),
            "mean_fd_base": fd_base.mean(),
            "mean_fdg": fd_gain.mean(),
            "pct_positive": (fd_gain > 0).mean() * 100,
            "n_params": n_params,
            "best_epoch": ckpt.get("best_epoch", "?"),
        }
    )
    print(f"  done: {fname}")

# ── print table ──────────────────────────────────────────────────────
rows.sort(key=lambda r: -r["mean_fdg"])

header = (
    f"| {'Model':<15} | {'Key Config':<45} | {'SeqLen':>6} | {'Params':>7} | {'Epoch':>5} "
    f"| {'FD_pred':>7} | {'FD_base':>7} | {'FD_gain':>7} | {'%>0':>5} | {'Checkpoint':<50} |"
)
sep = (
    f"|{'-' * 17}|{'-' * 47}|{'-' * 8}:|{'-' * 9}:|{'-' * 7}:"
    f"|{'-' * 9}:|{'-' * 9}:|{'-' * 9}:|{'-' * 7}:|{'-' * 52}|"
)
print(f"\n{header}")
print(sep)
for r in rows:
    mtype, key_cfg = r["label"], "—"
    if " (" in r["label"] and r["label"].endswith(")"):
        mtype = r["label"].split(" (")[0]
        key_cfg = r["label"].split(" (", 1)[1][:-1]
    print(
        f"| {mtype:<15} | {key_cfg:<45} | {r['seq_len']:>6} | {r['n_params']:>7,} | {r['best_epoch']:>5} "
        f"| {r['mean_fd_pred']:>7.4f} | {r['mean_fd_base']:>7.4f} | {r['mean_fdg']:>7.4f} | {r['pct_positive']:>5.1f}% | {r['fname']:<50} |"
    )
