import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import plots
from dataset import GPUBatchLoader, TimeSeriesDataset, parse_task
from metrics import evaluate
from models import build_model

# Usage: python evaluate.py [checkpoints/.../some_checkpoint.pth]
CHECKPOINT_PATH = sys.argv[1]
# set e.g. 95 to enable the "use baseline when the model is uncertain" experiment
THRESHOLD_PERCENTILE = None

device = "cpu"

ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
config = ckpt["config"]
mu, sigma = ckpt["mu"], ckpt["sigma"]
test_ids = ckpt["test_ids"]
test_task = config["data"]["test_task"]
test_tasks = parse_task(test_task)

# the model is rebuilt straight from the config stored in the checkpoint
model = build_model(config["model"]).to(device)
model.load_state_dict(ckpt["model_state"])

# optional uncertainty threshold, picked as a percentile of the stored val σ distribution
sigma_threshold = None
if THRESHOLD_PERCENTILE is not None:
    sigma_threshold = np.percentile(ckpt["pred_sigma"], THRESHOLD_PERCENTILE, axis=0)

tag = os.path.basename(CHECKPOINT_PATH).removesuffix(".pth")
tag = f"{tag}_on{test_task}_threshold{THRESHOLD_PERCENTILE}"
results_dir = f"results/{tag}"
os.makedirs(results_dir, exist_ok=True)


def save(fig, name):
    path = os.path.join(results_dir, f"{name}_{tag}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# Evaluate one task at a time and keep each task's per-sample arrays.
test_dict = {
    t: np.load(f"datasets/{t}_dict.npy", allow_pickle=True).item() for t in test_tasks
}
out_by_task = {}
for task in test_tasks:
    data = np.array([test_dict[task][pid] for pid in test_ids])  # [patients, frames, 6]
    data = (data - mu) / sigma
    # use the window length the checkpoint was trained with, not the default
    seq_len = config["data"].get("sequence_length", 10)
    ds = TimeSeriesDataset(data, test_ids, sequence_length=seq_len, device=device)
    loader = GPUBatchLoader(ds, batch_size=1024, shuffle=False)
    out_by_task[task] = evaluate(
        model, loader, mu, sigma, device, sigma_threshold=sigma_threshold
    )

# Frame-level arrays (rotations ×50 -> mm) and matching per-sample task labels.
pred_list, true_list, base_list, labels = [], [], [], []
fd_pred_list, fd_base_list, z_list = [], [], []
for task in test_tasks:
    out = out_by_task[task]
    p, tr, b = out["pred"].copy(), out["true"].copy(), out["base"].copy()
    p[:, 3:6] *= 50
    tr[:, 3:6] *= 50
    b[:, 3:6] *= 50
    pred_list.append(p)
    true_list.append(tr)
    base_list.append(b)
    fd_pred_list.append(out["fd_pred"])
    fd_base_list.append(out["fd_base"])
    z_list.append(out["z"])
    labels.append(np.full(len(p), task))

pred = np.concatenate(pred_list)
true = np.concatenate(true_list)
base = np.concatenate(base_list)
fd_pred = np.concatenate(fd_pred_list)
fd_base = np.concatenate(fd_base_list)
z = np.concatenate(z_list)
task_labels = np.concatenate(labels)  # one label per sample/frame (they align)

# Per-patient FD aggregates, kept separate per task.
fd_per_patient_pred, fd_per_patient_base, fdg_per_patient = {}, {}, {}
for task in test_tasks:
    out = out_by_task[task]
    pred_p, base_p, fdg_p = [], [], []
    for pid in test_ids:
        # sel is the mask to select frames of the current patient
        sel = out["ids"] == pid
        if not sel.any():
            continue
        fb, fp = out["fd_base"][sel], out["fd_pred"][sel]
        pred_p.append(fp.mean())
        base_p.append(fb.mean())
        fdg_p.append(((fb - fp) / fb).mean())
    fd_per_patient_pred[task] = np.array(pred_p)
    fd_per_patient_base[task] = np.array(base_p)
    fdg_per_patient[task] = np.array(fdg_p)

# Headline scalars per task for the summary card.
task_scalars = {}
for task in test_tasks:
    out = out_by_task[task]
    fb, fp = out["fd_base"], out["fd_pred"]
    task_scalars[task] = {
        "nll": out["nll"],
        "fd_base": fb.mean(),
        "fd_pred": fp.mean(),
        "fdg": ((fb - fp) / (fb + 1e-6)).mean(),
    }

# Build and save every figure.
save(
    plots.error_per_dimension(pred, true, base, task_labels, test_tasks),
    "01_error_per_dimension",
)
save(
    plots.true_vs_predicted(pred, true, task_labels, test_tasks), "02_true_vs_predicted"
)
save(
    plots.fd_distribution(fd_pred, fd_base, task_labels, test_tasks),
    "03_fd_distribution",
)
save(
    plots.per_patient_fdg(
        fd_per_patient_pred, fd_per_patient_base, fdg_per_patient, test_tasks
    ),
    "04_per_patient_fdg",
)
save(
    plots.metrics_summary(pred, true, base, task_scalars, test_tasks, tag),
    "05_metrics_summary",
)
save(plots.sigma_calibration(z, task_labels, test_tasks), "06_sigma_calibration")

# Random held-out patient on the first task, its frames already in time order.
viz_task = test_tasks[0]
patient_id = test_ids[0]
out = out_by_task[viz_task]
sel = out["ids"] == patient_id
tp, pp, bp, sp = (
    out["true"][sel].copy(),
    out["pred"][sel].copy(),
    out["base"][sel].copy(),
    out["std"][sel].copy(),
)
for a in (tp, pp, bp, sp):
    a[:, 3:6] *= 50
save(
    plots.patient_timeseries(tp, pp, bp, sp, patient_id, viz_task),
    "07_patient_timeseries",
)

save(
    plots.fdgain_vs_motion(fd_pred, fd_base, task_labels, test_tasks),
    "08_fdgain_vs_motion",
)
