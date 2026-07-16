"""
Usage: python benchmark.py [checkpoints/generalist]

Outputs (results/benchmark/): benchmark.csv
"""

import os
import sys

import numpy as np
import torch

from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from metrics import evaluate
from models import build_model, count_flops, get_device, time_batch

CKPT_DIR = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/generalist"
RESULTS_DIR = "results/benchmark"
NOISE_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]  # robustness sweep (σ, normalized space)
DEGRADE_AT = 0.3  # the noise level whose model-FD inflation becomes a table column
BATCH_SIZE = 1024  # for the throughput measurement

os.makedirs(RESULTS_DIR, exist_ok=True)
device = get_device()
print(f"device: {device}\nscanning: {CKPT_DIR}\n")


def profile(model, config):
    """Parameter count, float32 size, FLOPs/window and latency for one model."""
    seq_len = config["data"]["sequence_length"]
    in_dim = config["model"]["input_dim"]
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    flops = count_flops(model, seq_len, in_dim)

    latency_single = time_batch(model, device, 1, seq_len, in_dim)
    latency_batch = time_batch(model, device, BATCH_SIZE, seq_len, in_dim)
    return {
        "n_params": n_params,
        "size_mb": size_mb,
        "flops": flops,
        "latency_ms": latency_single * 1e3,
        "throughput": BATCH_SIZE / latency_batch,
    }


# ── accuracy eval
# The generalist checkpoints append velocity/acceleration channels, so the loader
# rebuilds those features from raw positions and z-scores them with the stored
# per-channel mu/sigma. `noise` is passed through to evaluate(), which perturbs the
# (normalized) model input by noise·N(0,1) — the robustness noise model.
def eval_model(model, ckpt, noise=None):
    config = ckpt["config"]
    mu, sigma = ckpt["mu"], ckpt["sigma"]
    test_ids = ckpt["test_ids"]

    seq_len = config["data"].get("sequence_length", 10)
    add_velocity = config["data"].get("add_velocity", False)
    add_acceleration = config["data"].get("add_acceleration", False)

    fd_ps, fd_bs, nll_sum = [], [], 0.0
    test_tasks = parse_task(config["data"]["test_task"])
    for task in test_tasks:
        data_dict = np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()
        data = np.array([data_dict[pid] for pid in test_ids])
        ds = TimeSeriesDataset(
            data,
            test_ids,
            sequence_length=seq_len,
            device=device,
            add_velocity=add_velocity,
            add_acceleration=add_acceleration,
            mu=mu,
            sigma=sigma,
        )
        loader = GPUBatchLoader(ds, batch_size=1024, shuffle=False)
        out = evaluate(model, loader, mu, sigma, device, noise=noise)
        fd_ps.append(out["fd_pred"])
        fd_bs.append(out["fd_base"])
        nll_sum += out["nll"]

    fd_p, fd_b = np.concatenate(fd_ps), np.concatenate(fd_bs)
    fdg_per_sample = (fd_b - fd_p) / (fd_b + 1e-6)
    return {
        "fdg": float(fdg_per_sample.mean()),
        "fd_pred": float(fd_p.mean()),
        "fd_base": float(fd_b.mean()),
        "fdg_per_sample": fdg_per_sample,
        "nll": nll_sum / len(test_tasks),
    }


# ── robustness: model-FD inflation at 0.3σ, and the noise tolerance σ ──
def tolerance_sigma(levels, fd_preds, fd_base_clean):
    """First noise level where the model's FD rises above the *clean* baseline FD.
    linearly interpolated"""
    if fd_preds[0] >= fd_base_clean:
        return 0.0
    for (x0, y0), (x1, y1) in zip(zip(levels, fd_preds), zip(levels[1:], fd_preds[1:])):
        if y1 >= fd_base_clean:  # crossed between x0 and x1
            return x0 + (x1 - x0) * (fd_base_clean - y0) / (y1 - y0)
    return None


# ── one row per checkpoint: clean accuracy + robustness sweep + deployment profile ──
rows = []
for fname in sorted(f for f in os.listdir(CKPT_DIR) if f.endswith(".pth")):
    path = os.path.join(CKPT_DIR, fname)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model(config["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"\n{config['model']['type']} ({fname})")

    clean = eval_model(model, ckpt)
    print(f"  clean: FDg={clean['fdg']:.4f} FD={clean['fd_pred']:.4f}")

    # robustness sweep: track the model's FD as input noise grows, against
    # the fixed clean-baseline reference
    # clean point is reused as level 0.
    fd_pred_at = {0.0: clean["fd_pred"]}
    for noise in NOISE_LEVELS:
        r = eval_model(model, ckpt, noise=noise)
        fd_pred_at[noise] = r["fd_pred"]
        print(
            f"  noise={noise:g}: FD={r['fd_pred']:.4f} "
            f"(clean baseline {clean['fd_base']:.4f})"
        )
    levels = [0.0] + NOISE_LEVELS
    fd_preds = [fd_pred_at[l] for l in levels]
    degrade = 100.0 * (fd_pred_at[DEGRADE_AT] - clean["fd_pred"]) / clean["fd_pred"]
    tol = tolerance_sigma(levels, fd_preds, clean["fd_base"])

    prof = profile(model, config)
    rows.append(
        {
            "arch": config["model"]["type"],
            "seq": config["data"].get("sequence_length", 10),
            "ckpt": fname,
            "n_params": prof["n_params"],
            "size_mb": prof["size_mb"],
            "flops_m": prof["flops"] / 1e6,
            "latency_ms": prof["latency_ms"],
            "throughput": prof["throughput"],
            "mean_fd": clean["fd_pred"],
            "fdg": clean["fdg"],
            "pct_pos": 100.0 * (clean["fdg_per_sample"] > 0).mean(),
            "nll": clean["nll"],
            "degrade": degrade,
            "tolerance": tol,
        }
    )

if not rows:
    sys.exit(f"no checkpoints found in {CKPT_DIR}")
rows.sort(key=lambda r: -r["fdg"])  # best architecture first

# ── render: CSV (one row per checkpoint, sorted by mean FD-gain) ──
COLS = [
    "arch",
    "seq",
    "n_params",
    "size_mb",
    "flops_m",
    "latency_ms",
    "throughput",
    "mean_fd",
    "fdg",
    "pct_pos",
    "nll",
    "degrade",
    "tolerance",
]

csv_path = os.path.join(RESULTS_DIR, "benchmark.csv")
with open(csv_path, "w") as fh:
    fh.write(",".join(COLS) + ",ckpt\n")
    for r in rows:
        fh.write(
            ",".join(("" if r[k] is None else str(r[k])) for k in COLS)
            + f",{r['ckpt']}\n"
        )
print(f"saved {csv_path}")
