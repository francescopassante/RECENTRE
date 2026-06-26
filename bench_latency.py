"""Single-sample inference latency (real-time use): batch 1, eval mode, no_grad.
Reports median / p95 / p99 ms per prediction -- tail latency is what matters for
real-time. Run it on the actual DEPLOYMENT hardware, not just the training box.

Usage: python bench_latency.py [config.yaml] [compile] [cpu]
  compile -> torch.compile the model first
  cpu     -> force CPU (e.g. to test a CPU deployment target)
"""
import sys
import time

import torch
import yaml

from models import build_model, get_device

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/mamba_generalist_vel.yaml"
do_compile = "compile" in sys.argv[1:]
force_cpu = "cpu" in sys.argv[1:]

cfg = yaml.safe_load(open(cfg_path))
mcfg, dcfg = cfg["model"], cfg["data"]

device = torch.device("cpu") if force_cpu else get_device()
model = build_model(mcfg).to(device).eval()  # eval -> no dropout, no checkpointing
if do_compile:
    model.compile()

L, D = dcfg["sequence_length"], mcfg["input_dim"]
x = torch.randn(1, L, D, device=device)  # ONE window -> one prediction
print(f"device: {device.type} | compiled: {do_compile} | input: {tuple(x.shape)}")


def predict():
    with torch.no_grad():
        mean, var = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return mean


for _ in range(20):  # warmup (compile/autotune/allocator)
    predict()

lat = []
for _ in range(300):
    t0 = time.perf_counter()
    predict()
    lat.append((time.perf_counter() - t0) * 1e3)
lat.sort()
p = lambda q: lat[int(q * len(lat)) - 1]
print(f"latency ms  median {p(0.50):.1f} | p95 {p(0.95):.1f} | p99 {p(0.99):.1f} | max {lat[-1]:.1f}")
print(f"target: <=200 ms  ->  {'PASS' if p(0.99) <= 200 else 'FAIL'} (at p99)")
