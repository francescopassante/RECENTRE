"""Isolate the model from data loading: time one forward+backward on a synthetic
batch of the real shape, and print the device + projected epoch time. If this is
fast but `python train.py ...` still hangs, the bottleneck is split_data / loading,
not the model.

Usage: python bench_mamba.py [config.yaml] [compile]
  add the word "compile" as a 2nd arg to time the torch.compile'd model.
"""
import math
import sys
import time

import torch
import yaml

from models import build_model, get_device

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/mamba_generalist_vel.yaml"
cfg = yaml.safe_load(open(cfg_path))
mcfg, dcfg = cfg["model"], cfg["data"]

print("cuda available:", torch.cuda.is_available())
device = get_device()  # prints the device actually selected (cuda / mps / cpu)

model = build_model(mcfg).to(device)
model.train()
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
do_compile = len(sys.argv) > 2 and sys.argv[2] == "compile"
if do_compile:
    model.compile()
print(f"params: {n:,}  | config: {cfg_path}  | compiled: {do_compile}")

B, L, D = dcfg["batch_size"], dcfg["sequence_length"], mcfg["input_dim"]
x = torch.randn(B, L, D, device=device)
y = torch.randn(B, 6, device=device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
nll = torch.nn.GaussianNLLLoss()


def step():
    opt.zero_grad()
    mean, var = model(x)
    nll(mean, y, var).backward()
    opt.step()


for _ in range(3 if do_compile else 2):  # warmup (compile happens on 1st call)
    step()
if device.type == "cuda":
    torch.cuda.synchronize()

N = 10
t0 = time.perf_counter()
for _ in range(N):
    step()
if device.type == "cuda":
    torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / N * 1e3

# project epoch time: ~718 train patients * windows-per-patient (neg-aug off)
span = 2 * L
windows = sum(max(T - span + 1, 0) for T in (1200, 405, 316))
n_batches = math.ceil(718 * windows / B)
print(f"{ms:.1f} ms / batch  (B={B}, L={L}, D={D})")
print(f"~{n_batches} train batches/epoch -> ~{n_batches * ms / 1000:.0f} s/epoch (train only)")
if device.type == "cuda":
    print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
