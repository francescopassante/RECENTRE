"""Master benchmark: one row per architecture, comparing predictive accuracy
against real-time deployment cost across every trained model.

A "family" is a (model type, sequence length) pair, so the same architecture at
two input-window lengths gives two rows. For each family we pick its best
checkpoint (highest mean FD-gain) in the given directory, then report:

  accuracy    — mean FD (model), mean FD-gain, % of frames with FD-gain > 0, NLL
  robustness  — model FD inflation at 0.3σ input noise (% rise over clean FD),
                and the noise σ (tolerance) at which the noisy model stops
                beating a *clean* previous-frame baseline
  deployment  — parameter count, float32 size, FLOPs / window, single-frame
                latency and batched throughput

The robustness columns add the same additive-Gaussian input noise as
robustness.py (x += noise · N(0,1) in normalized space) but track the model's
ABSOLUTE FD against the fixed clean-baseline reference: adding noise also
corrupts the previous-frame baseline (it *is* the last input frame), so FD-gain
under noise is a ratio of two moving targets and is not summarized here. The
deployment columns reuse the profiling block from evaluate.py. Accuracy is the
single evaluate() path, so nothing is re-derived here.

Usage: python benchmark.py [checkpoints/generalist]

Outputs (results/benchmark/): benchmark_table.md, benchmark.csv, benchmark_table.png
"""

import os
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from metrics import evaluate
from models import build_model, get_device

CKPT_DIR = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/generalist"
RESULTS_DIR = "results/benchmark"
NOISE_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]  # robustness sweep (σ, normalized space)
DEGRADE_AT = 0.3  # the noise level whose model-FD inflation becomes a table column
BATCH_SIZE = 1024  # for the throughput measurement

os.makedirs(RESULTS_DIR, exist_ok=True)
device = get_device()
print(f"device: {device}\nscanning: {CKPT_DIR}\n")


# ── deployment profiling (params / size / FLOPs / latency), lifted from evaluate.py ──
def sync():
    # GPU kernels run asynchronously: model(x) only enqueues work and returns
    # before the GPU finishes. Block until the queue drains so perf_counter
    # brackets real compute, not just the CPU-side launch loop.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_batch(model, batch_size, seq_len, in_dim, warmup=3, runs=20):
    xb = torch.randn(batch_size, seq_len, in_dim, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(xb)
        sync()  # ensure warmup work is done before starting the clock
        t = time.perf_counter()
        for _ in range(runs):
            model(xb)
        sync()  # wait for all runs to finish before stopping the clock
        return (time.perf_counter() - t) / runs


def profile(ckpt_path):
    """Parameter count, float32 size, FLOPs/window and latency for one checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model(config["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    seq_len = config["data"]["sequence_length"]
    in_dim = config["model"]["input_dim"]
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    one = torch.randn(1, seq_len, in_dim, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(one)
    flops = fc.get_total_flops()

    latency_single = time_batch(model, 1, seq_len, in_dim)
    latency_batch = time_batch(model, BATCH_SIZE, seq_len, in_dim)
    return {
        "n_params": n_params,
        "size_mb": size_mb,
        "flops": flops,
        "latency_ms": latency_single * 1e3,
        "throughput": BATCH_SIZE / latency_batch,
    }


# ── accuracy eval (feature-aware, mirrors analyze_checkpoints.py + evaluate()) ──
# The generalist checkpoints append velocity/acceleration channels, so the loader
# rebuilds those features from raw positions and z-scores them with the stored
# per-channel mu/sigma. `noise` is passed through to evaluate(), which perturbs the
# (normalized) model input by noise·N(0,1) — the same noise model as robustness.py.
def eval_ckpt(ckpt, noise=None):
    config = ckpt["config"]
    mu, sigma = ckpt["mu"], ckpt["sigma"]
    test_ids = ckpt["test_ids"]
    model = build_model(config["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])

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

    Fixed-reference robustness: `fd_preds` is the model's mean FD at each noise
    level (model fed noisy input, scored against the clean target); `levels` are
    the aligned noise σ starting at the clean point (level 0). We report the σ at
    which the noisy model stops beating a clean previous-frame predictor
    (`fd_base_clean`), linearly interpolated. Returns 0.0 if the model never beat
    the baseline even clean, or None if it still beats it across the whole swept
    range (report as '>max'). Using the clean baseline as a fixed yardstick
    avoids FD-gain's moving-denominator problem (the noisy baseline degrades too).
    """
    if fd_preds[0] >= fd_base_clean:
        return 0.0
    for (x0, y0), (x1, y1) in zip(
        zip(levels, fd_preds), zip(levels[1:], fd_preds[1:])
    ):
        if y1 >= fd_base_clean:  # crossed between x0 and x1
            return x0 + (x1 - x0) * (fd_base_clean - y0) / (y1 - y0)
    return None


# ── pass 1: evaluate every checkpoint clean, keep the best per family ──
# family = (model type, sequence length); the same architecture at two window
# lengths therefore yields two separate rows.
ckpt_files = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith(".pth"))
best = {}  # (type, seq_len) -> (fdg, fname, clean run_eval result)
for fname in ckpt_files:
    path = os.path.join(CKPT_DIR, fname)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    data_cfg = ckpt["config"]["data"]
    key = (ckpt["config"]["model"]["type"], data_cfg.get("sequence_length", 10))
    res = eval_ckpt(ckpt)
    print(f"  {fname:<52} FDg={res['fdg']:.4f}")
    if key not in best or res["fdg"] > best[key][0]:
        best[key] = (res["fdg"], fname, res)

# ── pass 2: robustness sweep + deployment profile for each winner ──
rows = []
for (mtype, seq_len), (clean_fdg, fname, clean) in best.items():
    path = os.path.join(CKPT_DIR, fname)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n{mtype} (seq={seq_len}): best = {fname}")

    # robustness sweep: track the model's ABSOLUTE FD as input noise grows, against
    # the fixed clean-baseline reference (the noisy baseline is not a stable yardstick).
    # clean point reused from pass 1 as level 0.
    fd_base_clean = clean["fd_base"]
    levels, fd_preds = [0.0], [clean["fd_pred"]]
    fd_pred_at = {}
    for noise in NOISE_LEVELS:
        r = eval_ckpt(ckpt, noise=noise)
        levels.append(noise)
        fd_preds.append(r["fd_pred"])
        fd_pred_at[noise] = r["fd_pred"]
        print(
            f"  noise={noise:g}: FD={r['fd_pred']:.4f} "
            f"(clean baseline {fd_base_clean:.4f})"
        )
    degrade = 100.0 * (fd_pred_at[DEGRADE_AT] - clean["fd_pred"]) / clean["fd_pred"]
    tol = tolerance_sigma(levels, fd_preds, fd_base_clean)

    prof = profile(path)
    rows.append(
        {
            "arch": mtype,
            "seq": seq_len,
            "ckpt": fname,
            "n_params": prof["n_params"],
            "size_mb": prof["size_mb"],
            "flops_m": prof["flops"] / 1e6,
            "latency_ms": prof["latency_ms"],
            "throughput": prof["throughput"],
            "mean_fd": clean["fd_pred"],
            "fdg": clean_fdg,
            "pct_pos": 100.0 * (clean["fdg_per_sample"] > 0).mean(),
            "nll": clean["nll"],
            "degrade": degrade,
            "tolerance": tol,
        }
    )

if not rows:
    sys.exit(f"no checkpoints found in {CKPT_DIR}")
rows.sort(key=lambda r: -r["fdg"])  # best architecture first

# ── render: markdown (stdout + file), CSV, and a PNG table ──
COLS = [
    ("Architecture", "arch", "{}", "<13"),
    ("Seq", "seq", "{}", ">4"),
    ("Params", "n_params", "{:,}", ">9"),
    ("Size (MB)", "size_mb", "{:.3f}", ">9"),
    ("FLOPs (M)", "flops_m", "{:.2f}", ">9"),
    ("Lat 1f (ms)", "latency_ms", "{:.3f}", ">11"),
    ("Thrpt (k/s)", "throughput", "{:.0f}", ">11"),
    ("Mean FD", "mean_fd", "{:.4f}", ">8"),
    ("FD-gain", "fdg", "{:.4f}", ">8"),
    ("%>0", "pct_pos", "{:.1f}", ">5"),
    ("NLL", "nll", "{:.3f}", ">7"),
    (f"ΔFD@{DEGRADE_AT}σ %", "degrade", "{:+.0f}", ">10"),
    ("Tol σ", "tolerance", "{:.2f}", ">6"),
]


def cell(r, key, fmt):
    v = r[key]
    if key == "throughput":
        v = v / 1e3  # samples/s -> k/s
    if key == "tolerance" and v is None:
        return f">{NOISE_LEVELS[-1]:g}"
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    return fmt.format(v)


header = "| " + " | ".join(f"{h:{w}}" for h, _, _, w in COLS) + " |"
sep = "|" + "|".join("-" * (len(f"{h:{w}}") + 2) for h, _, _, w in COLS) + "|"
lines = [header, sep]
for r in rows:
    lines.append(
        "| " + " | ".join(f"{cell(r, k, f):{w}}" for _, k, f, w in COLS) + " |"
    )
table_md = "\n".join(lines)
print("\n" + table_md + "\n")

md_path = os.path.join(RESULTS_DIR, "benchmark_table.md")
with open(md_path, "w") as fh:
    fh.write(f"# Architecture benchmark ({CKPT_DIR}, device={device})\n\n")
    fh.write("Best checkpoint per architecture, sorted by mean FD-gain.\n\n")
    fh.write(table_md + "\n")
print(f"saved {md_path}")

csv_path = os.path.join(RESULTS_DIR, "benchmark.csv")
with open(csv_path, "w") as fh:
    fh.write(",".join(k for _, k, _, _ in COLS) + ",ckpt\n")
    for r in rows:
        fh.write(
            ",".join(
                ("" if r[k] is None else str(r[k])) for _, k, _, _ in COLS
            )
            + f",{r['ckpt']}\n"
        )
print(f"saved {csv_path}")

# PNG table (report-ready, repo convention of saving summary tables as figures)
fig, ax = plt.subplots(figsize=(0.95 * len(COLS), 0.7 + 0.42 * len(rows)))
ax.axis("off")
cell_text = [[cell(r, k, f) for _, k, f, _ in COLS] for r in rows]
tbl = ax.table(
    cellText=cell_text,
    colLabels=[h for h, _, _, _ in COLS],
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.4)
# green = best per column (FD-gain / %>0 / tolerance / throughput high; the rest low,
# including ΔFD inflation where a smaller rise under noise is better)
higher_better = {"fdg", "pct_pos", "tolerance", "throughput"}
for c, (_, key, _, _) in enumerate(COLS):
    if key in ("arch", "seq", "ckpt"):
        continue
    vals = [r[key] for r in rows]
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
        continue
    best_r = int(np.argmax(vals) if key in higher_better else np.argmin(vals))
    tbl[(best_r + 1, c)].set_facecolor("lightgreen")
fig.suptitle("Architecture benchmark (green = best per column)")
fig.tight_layout()
png_path = os.path.join(RESULTS_DIR, "benchmark_table.png")
fig.savefig(png_path, bbox_inches="tight")
plt.close(fig)
print(f"saved {png_path}")
