"""
Benchmark: GPUBatchLoader vs standard DataLoader (GPU dataset, CPU dataset).
Run on a V100 and paste the output back.

Usage:
    python bench_loader.py
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import TimeSeriesDataset, GPUBatchLoader

# ------------------------------------------------------------------
# Synthetic data that matches a real R+M+L multi-task setup:
#   ~500 patients, T=1200 frames, 6 DOF
# ------------------------------------------------------------------
N_PATIENTS   = 500
T_FRAMES     = 1200
N_DIMS       = 6
SEQUENCE_LEN = 10
BATCH_SIZE   = 1024
N_EPOCHS     = 3      # average over a few epochs to reduce noise

rng = np.random.default_rng(0)
data_np = rng.standard_normal((N_PATIENTS, T_FRAMES, N_DIMS)).astype(np.float32)
ids     = np.arange(N_PATIENTS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

def time_full_epoch(loader, label, warmup=1):
    # warmup
    for _ in range(warmup):
        for batch in loader:
            pass
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_EPOCHS):
        for batch in loader:
            pass
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / N_EPOCHS
    n_batches = len(loader)
    print(f"{label:45s}  {elapsed:.3f}s/epoch  ({n_batches} batches)")
    return elapsed


# ------------------------------------------------------------------
# 1. GPUBatchLoader  (dataset lives on GPU)
# ------------------------------------------------------------------
ds_gpu = TimeSeriesDataset(data_np, ids, sequence_length=SEQUENCE_LEN, device=device)
gpu_loader = GPUBatchLoader(ds_gpu, batch_size=BATCH_SIZE, shuffle=True)
t_gpu = time_full_epoch(gpu_loader, "GPUBatchLoader (data on GPU, vectorized gather)")

# ------------------------------------------------------------------
# 2. Standard DataLoader, dataset on GPU  (the bad case the docstring warns about)
# ------------------------------------------------------------------
ds_gpu2 = TimeSeriesDataset(data_np, ids, sequence_length=SEQUENCE_LEN, device=device)
std_gpu_loader = DataLoader(ds_gpu2, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
t_std_gpu = time_full_epoch(std_gpu_loader, "DataLoader      (data on GPU, per-sample __getitem__)")

# ------------------------------------------------------------------
# 3. Standard DataLoader, dataset on CPU  (the intended use case for DataLoader)
# ------------------------------------------------------------------
ds_cpu = TimeSeriesDataset(data_np, ids, sequence_length=SEQUENCE_LEN, device=torch.device("cpu"))
std_cpu_loader = DataLoader(ds_cpu, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=(device.type == "cuda"))
t_std_cpu = time_full_epoch(std_cpu_loader, "DataLoader      (data on CPU, pin_memory, H2D per batch)")

print()
print(f"GPUBatchLoader vs DataLoader-on-GPU:  {t_std_gpu/t_gpu:.1f}x faster")
print(f"GPUBatchLoader vs DataLoader-on-CPU:  {t_std_cpu/t_gpu:.1f}x faster")
