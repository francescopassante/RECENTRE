"""Evaluate temporal stability (autoregressive multi-step ahead prediction).

The model was trained with a window of 10 frames sampled at stride=2,
meaning it spans 20 frames into the past (e.g., uses frames 0, 2, 4... 18 
to predict frame 19).

This script seeds a buffer with the first 19 ground-truth frames, then 
autoregressively predicts N steps into the future, feeding its own 
predictions back into the buffer and respecting the stride=2 sampling.

Usage: python temporal_stability.py checkpoints/generalist/tcn_...pth
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset import parse_task
from models import build_model

DIM_NAMES = ["Tx (mm)", "Ty (mm)", "Tz (mm)", "Rx (mm)", "Ry (mm)", "Rz (mm)"]
CHECKPOINT_PATH = sys.argv[1] if len(sys.argv) > 1 else None

if not CHECKPOINT_PATH or not os.path.isfile(CHECKPOINT_PATH):
    print("Please provide a valid path to a specific .pth checkpoint file.")
    print("Usage: python temporal_stability.py checkpoints/.../model.pth")
    sys.exit(1)

tag = os.path.basename(CHECKPOINT_PATH).removesuffix(".pth")
RESULTS_DIR = f"results/temporal_stability/{tag}"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"device: {device}")

# 1. Load checkpoint and model
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
config = ckpt["config"]
mu, sigma = ckpt["mu"], ckpt["sigma"]
test_ids = ckpt["test_ids"]

model = build_model(config["model"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

test_tasks = parse_task(config["data"]["test_task"])
seq_len = config["data"].get("sequence_length", 10)
time_span = seq_len * 2  # Spans 20 frames

mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)

# Take just the first task and the first 3 patients to visualize
task = test_tasks[0]
task_dict = np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()
patients_to_plot = test_ids[:3]

fig, axes = plt.subplots(len(patients_to_plot), 6, figsize=(24, 3.5 * len(patients_to_plot)), sharex=True)
if len(patients_to_plot) == 1:
    axes = np.expand_dims(axes, 0)

fig.suptitle(f"Temporal Stability (Full Trajectory Autoregression) - {tag}\nTask: {task}")

for row_idx, pid in enumerate(patients_to_plot):
    print(f"Generating autoregressive trajectory for patient {pid}...")
    
    # Get raw data and normalize it
    raw_data = task_dict[pid]
    norm_data = (raw_data - mu) / sigma
    
    # Setup the initial buffer with the first `time_span - 1` real frames
    buffer = list(norm_data[:time_span - 1]) 
    
    # Calculate how many frames are left to predict
    horizon = len(raw_data) - (time_span - 1)
    
    autoreg_preds = []
    
    # Autoregressive loop
    with torch.no_grad():
        for step in tqdm(range(horizon), leave=False):
            # Extract the correct frames from the buffer using stride=2
            input_frames = buffer[-(time_span - 1)::2]
            
            current_window = torch.tensor(np.array(input_frames), dtype=torch.float32, device=device).unsqueeze(0)
            
            mean, var = model(current_window)
            pred_frame = mean.squeeze(0).cpu().numpy()
            
            autoreg_preds.append(pred_frame)
            buffer.append(pred_frame)
            
    # Denormalize predictions
    autoreg_preds = np.array(autoreg_preds) * sigma + mu
    
    start_idx = time_span - 1
    true_trajectory = raw_data[start_idx:]
    
    # Plotting
    for dim in range(6):
        ax = axes[row_idx, dim]
        scale = 50 if dim >= 3 else 1
        
        t_axis = np.arange(start_idx, len(raw_data))
        
        ax.plot(t_axis, true_trajectory[:, dim] * scale, label="True Trajectory", color="black", alpha=0.5, linewidth=2)
        ax.plot(t_axis, autoreg_preds[:, dim] * scale, label="Autoregressive Pred", color="red", linestyle="-", alpha=0.7, linewidth=1.5)
        
        if row_idx == 0:
            ax.set_title(DIM_NAMES[dim])
        if row_idx == len(patients_to_plot) - 1:
            ax.set_xlabel("Frame")
        if dim == 0:
            ax.set_ylabel(f"Patient {pid}")

# Add a single legend for the figure
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.tight_layout()
fig.subplots_adjust(top=0.92) # Leave space for suptitle
out_path = os.path.join(RESULTS_DIR, "autoregressive_trajectory.png")
fig.savefig(out_path, dpi=150)
print(f"\nDone! Plot saved to: {out_path}")
