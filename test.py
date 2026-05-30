import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from GRU import GRUModel
from metrics import fd, fd_gain
from TimeSeriesDataset import GPUBatchLoader, TimeSeriesDataset

# Rotations are scaled by 50 mm (avg head radius) so every dim ends up in mm.
DIM_NAMES = ["Tx (mm)", "Ty (mm)", "Tz (mm)", "Rx (mm)", "Ry (mm)", "Rz (mm)"]
# one marker per dimension, so per-dim points stay distinguishable without inline labels
DIM_MARKERS = ["o", "s", "^", "D", "v", "P"]


def save(fig, name):
    path = os.path.join(RESULTS_DIR, f"{name}_{TAG}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def test(
    model,
    test_loader,
    criterion,
    device,
    mu,
    sigma,
    baseline_if_uncertain=False,
    percentile_threshold=None,
    sigma_dist=None,
):
    model.eval()
    test_fd_baselines = []
    test_fd_preds = []
    test_nll = 0
    z_all = []
    patient_pred_true_base = {
        p: {"pred": [], "true": [], "baseline": []} for p in test_loader.dataset.ids
    }
    mu = torch.tensor(mu, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
    if percentile_threshold is not None:
        sigma_threshold = torch.tensor(
            np.percentile(sigma_dist.numpy(), percentile_threshold, axis=0),
            device=device,
        )
    else:
        sigma_threshold = None
    with torch.no_grad():
        for p, x, y in test_loader:
            x, y = x.to(device), y.to(device)

            y_pred, y_var = model(x)

            if baseline_if_uncertain:
                pred_sigma = y_var.sqrt()
                # If the model is uncertain (predicted variance above threshold in any dimension), use the baseline instead of the prediction.
                uncertain_mask = (pred_sigma > sigma_threshold).any(
                    dim=1
                )  # [batch_size]
                y_pred[uncertain_mask] = x[
                    uncertain_mask, -1, :
                ]  # replace with baseline for uncertain samples

            test_nll += criterion(y_pred, y, y_var).item()

            # standardized residual in normalized space (scale-invariant)
            z_all.append(((y - y_pred) / torch.sqrt(y_var)).cpu().numpy())

            last_x = x[:, -1, :]
            test_fd_baselines.append(fd(last_x, y, mu, sigma).cpu())
            test_fd_preds.append(fd(y_pred, y, mu, sigma).cpu())
            for i in range(len(p)):
                denormalized_pred = y_pred[i] * sigma + mu
                denormalized_true = y[i] * sigma + mu
                denormalized_baseline = last_x[i] * sigma + mu
                patient_pred_true_base[p[i]]["pred"].append(
                    denormalized_pred.cpu().numpy()
                )
                patient_pred_true_base[p[i]]["true"].append(
                    denormalized_true.cpu().numpy()
                )
                patient_pred_true_base[p[i]]["baseline"].append(
                    denormalized_baseline.cpu().numpy()
                )

    test_nll /= len(test_loader)
    fd_baseline_cat = torch.cat(test_fd_baselines, dim=0)
    fd_model_cat = torch.cat(test_fd_preds, dim=0)

    metrics = {
        "nll": test_nll,
        "fd_pred": fd_model_cat.mean().item(),
        "fd_base": fd_baseline_cat.mean().item(),
        "fdg": fd_gain(fd_baseline_cat, fd_model_cat).mean().item(),
        "fd_pred_per_sample": fd_model_cat.numpy(),
        "fd_base_per_sample": fd_baseline_cat.numpy(),
        "z_per_sample": np.concatenate(z_all, axis=0),
    }
    return patient_pred_true_base, metrics


"""
==================================================
        Evaluation on the test set
==================================================
"""

# Pick which checkpoint to evaluate
MODEL_TAG = "GRU"
FILENAME = "GRU_R+M+LvR+M+L_beta0.5_ep150.pth"
CHECKPOINT_PATH = f"checkpoints/generalist/{FILENAME}"

device = "cpu"

# Load checkpoint + data
saved_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)
model_state_dict = saved_dict["model_state_dict"]
test_ids = saved_dict["test_ids"]
beta = saved_dict["beta"]
epochs = saved_dict["epochs"]
mu = saved_dict["mu"]
sigma = saved_dict["sigma"]
train_task = saved_dict["train_task"]
test_task = saved_dict["test_task"]
sigma_dist = saved_dict["pred_sigma"]

baseline_if_uncertain = False
threshold_percentile = None


TAG = FILENAME.removesuffix(".pth")
assert (
    FILENAME == f"{MODEL_TAG}_{train_task}v{test_task}_beta{beta}_ep{epochs}.pth"
), "Filename should follow the format: {MODEL_TAG}_{train_task}v{test_task}_beta{beta}_ep{epochs}.pth"

# Explicit what task the testing has been done on
TAG = f"{TAG}_on{test_task}_threshold{threshold_percentile}"

RESULTS_DIR = f"results/{TAG}"
os.makedirs(RESULTS_DIR, exist_ok=True)

if "+" in test_task:
    test_tasks = test_task.split("+")
else:
    test_tasks = [test_task]

test_dict = {
    task: np.load(f"datasets/{task}_dict.npy", allow_pickle=True).item()
    for task in test_tasks
}

# fixed color per task, reused by every task-aware plot below
TASK_COLORS = {
    "R": "tab:blue",
    "M": "tab:orange",
    "L": "tab:green",
}

model = GRUModel(
    input_dim=6, hidden_dim=128, output_dim=6, num_layers=2, dropout=0.5
).to(device)
model.load_state_dict(model_state_dict)
criterion = torch.nn.GaussianNLLLoss()


#  Evaluate one task at a time and keep each task's results.
results_by_task = {}
for task in test_tasks:
    task_data = np.array(
        [test_dict[task][pid] for pid in test_ids]
    )  # [num_patients, patient_frames, 6]
    task_data = (task_data - mu) / sigma

    task_dataset = TimeSeriesDataset(task_data, test_ids, device=device)
    task_loader = GPUBatchLoader(task_dataset, batch_size=1024, shuffle=False)

    patient_pred_true_base, metrics = test(
        model,
        task_loader,
        criterion,
        device,
        mu=mu,
        sigma=sigma,
        baseline_if_uncertain=baseline_if_uncertain,
        percentile_threshold=threshold_percentile,
        sigma_dist=sigma_dist,
    )
    results_by_task[task] = (patient_pred_true_base, metrics)


"""
==================================================
        Metrics and visualizations
==================================================
"""

# Aggregate per-patient arrays into flat matrices and per-(task, patient) FD values.
true_values = []
pred_values = []
baseline_values = []
frame_task_labels = []  # contains the task label for each frame. Used for coloring

# patient-level FD, kept separate per task for the 04 visualizations
fd_per_patient_pred = {task: [] for task in test_tasks}
fd_per_patient_base = {task: [] for task in test_tasks}
fdg_per_patient = {task: [] for task in test_tasks}

for task in test_tasks:
    patient_pred_true_base, _ = results_by_task[task]
    for patient_id in patient_pred_true_base.keys():
        # shape [patient_frames, 6]
        p_pred = np.array(patient_pred_true_base[patient_id]["pred"])
        p_true = np.array(patient_pred_true_base[patient_id]["true"])
        p_base = np.array(patient_pred_true_base[patient_id]["baseline"])

        # multiply the rotation parameters by 50mm (head radius) to translate to comparable values wrt translations
        p_pred[:, 3:6] *= 50
        p_true[:, 3:6] *= 50
        p_base[:, 3:6] *= 50
        pred_values.append(p_pred)
        true_values.append(p_true)
        baseline_values.append(p_base)
        # append the task label for each frame of the patient
        frame_task_labels.append(np.full(len(p_pred), task))

        # compute per patient fd both for the model and the baseline by summing over dimensions and averaging over frames
        scaled_pred = p_pred.copy()
        scaled_true = p_true.copy()
        scaled_base = p_base.copy()

        # scaled_pred: [patient_frames, 6]

        fd_pred = np.abs(scaled_pred - scaled_true).sum(axis=1)
        fd_base = np.abs(scaled_base - scaled_true).sum(axis=1)

        fd_per_patient_pred[task].append(fd_pred.mean())
        fd_per_patient_base[task].append(fd_base.mean())
        fdg_per_patient[task].append(((fd_base - fd_pred) / fd_base).mean())


# transform to np arrays
true_values = np.concatenate(true_values, axis=0)  # [total_frames, 6]
pred_values = np.concatenate(pred_values, axis=0)
baseline_values = np.concatenate(baseline_values, axis=0)
frame_task_labels = np.concatenate(frame_task_labels, axis=0)  # [total_frames]
fd_per_patient_pred = {task: np.array(v) for task, v in fd_per_patient_pred.items()}
fd_per_patient_base = {task: np.array(v) for task, v in fd_per_patient_base.items()}
fdg_per_patient = {task: np.array(v) for task, v in fdg_per_patient.items()}

# pooled per-sample arrays from each task's metrics, with matching task labels
fd_pred_per_sample = np.concatenate(
    [results_by_task[t][1]["fd_pred_per_sample"] for t in test_tasks]
)
fd_base_per_sample = np.concatenate(
    [results_by_task[t][1]["fd_base_per_sample"] for t in test_tasks]
)
z_per_sample = np.concatenate(
    [results_by_task[t][1]["z_per_sample"] for t in test_tasks], axis=0
)
sample_task_labels = np.concatenate(
    [np.full(len(results_by_task[t][1]["fd_pred_per_sample"]), t) for t in test_tasks]
)

# compute model and baseline errors for MAE and RMSE (pooled across tasks)
err_pred = pred_values - true_values
err_base = baseline_values - true_values

mae_pred_dim = np.abs(err_pred).mean(axis=0)
mae_base_dim = np.abs(err_base).mean(axis=0)
rmse_pred_dim = np.sqrt((err_pred**2).mean(axis=0))
rmse_base_dim = np.sqrt((err_base**2).mean(axis=0))

# ==========================================================================================
# 1) Per-dimension error: scatter of baseline vs model error (MAE and RMSE)
# ==========================================================================================
# dimension identity is encoded by marker shape and task by color, so the two
# legends below replace the inline dim labels (which overlapped near the origin).
task_handles = [
    Line2D([0], [0], marker="o", linestyle="", color=TASK_COLORS[t], label=t)
    for t in test_tasks
] + [Line2D([0], [0], linestyle="--", color="black", label="y = x")]
dim_handles = [
    Line2D([0], [0], marker=DIM_MARKERS[d], linestyle="", color="gray", label=name)
    for d, name in enumerate(DIM_NAMES)
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, label in zip(axes, ["MAE", "RMSE"]):
    hi = 0
    for task in test_tasks:
        m = frame_task_labels == task
        ep = pred_values[m] - true_values[m]
        eb = baseline_values[m] - true_values[m]
        if label == "MAE":
            metric_base = np.abs(eb).mean(axis=0)
            metric_model = np.abs(ep).mean(axis=0)
        else:
            metric_base = np.sqrt((eb**2).mean(axis=0))
            metric_model = np.sqrt((ep**2).mean(axis=0))
        for d in range(len(DIM_NAMES)):
            ax.scatter(
                metric_base[d],
                metric_model[d],
                color=TASK_COLORS[task],
                marker=DIM_MARKERS[d],
                s=60,
            )
        hi = max(hi, metric_base.max(), metric_model.max())
    hi *= 1.1
    ax.plot([0, hi], [0, hi], "k--")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(f"Baseline {label} (mm)")
    ax.set_ylabel(f"Model {label} (mm)")
    ax.set_title(f"{label} per dimension")
    task_legend = ax.legend(handles=task_handles, loc="upper left", fontsize=8)
    ax.add_artist(task_legend)
    ax.legend(handles=dim_handles, loc="lower right", fontsize=8)
fig.tight_layout()
save(fig, "01_error_per_dimension")


# ==========================================================================================
# 2) True vs Predicted scatter per dimension
# ==========================================================================================

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("True vs Predicted per dimension")
for d, ax in enumerate(axes.flat):
    for task in test_tasks:
        m = frame_task_labels == task
        ax.scatter(
            true_values[m, d],
            pred_values[m, d],
            s=2,
            alpha=0.3,
            color=TASK_COLORS[task],
            label=task,
        )
    x = true_values[:, d]
    y = pred_values[:, d]
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", label="y = x")
    ss_res = ((x - y) ** 2).sum()
    ss_tot = ((x - x.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{DIM_NAMES[d]}   R²={r2:.3f}")
    ax.legend(fontsize=8, markerscale=4)
fig.tight_layout()
save(fig, "02_true_vs_predicted")

# ==========================================================================================
# 3) Framewise-displacement (FD) distribution — model vs baseline
# ==========================================================================================

fd_pred_arr = fd_pred_per_sample
fd_base_arr = fd_base_per_sample

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
hi = max(np.percentile(fd_pred_arr, 99.5), np.percentile(fd_base_arr, 99.5))
bins = np.linspace(0, hi, 80)
# pooled baseline as a common reference, model FD overlaid per task.
# density=True so tasks with more samples (e.g. Resting) don't dominate.
ax.hist(
    fd_base_arr,
    bins=bins,
    color="gray",
    alpha=0.4,
    density=True,
    label=f"Baseline (mean={fd_base_arr.mean():.3f})",
)
for task in test_tasks:
    m = sample_task_labels == task
    ax.hist(
        fd_pred_arr[m],
        bins=bins,
        color=TASK_COLORS[task],
        histtype="step",
        linewidth=1.5,
        density=True,
        label=f"Model {task} (mean={fd_pred_arr[m].mean():.3f})",
    )
ax.set_xlabel("Framewise displacement (mm)")
ax.set_ylabel("density")
ax.set_title("FD distribution — model (per task) vs baseline")
ax.legend()

ax = axes[1]
fdg_per_sample = (fd_base_arr - fd_pred_arr) / fd_base_arr
# clip long tails so the central mass is visible
gain_lim = np.percentile(np.abs(fdg_per_sample), 99)
gain_bins = np.linspace(-gain_lim, gain_lim, 80)
for task in test_tasks:
    m = sample_task_labels == task
    g = np.clip(fdg_per_sample[m], -gain_lim, gain_lim)
    ax.hist(
        g,
        bins=gain_bins,
        color=TASK_COLORS[task],
        histtype="step",
        linewidth=1.5,
        density=True,
        label=f"{task} (mean={fdg_per_sample[m].mean():.3f})",
    )
ax.axvline(0, color="black")
pos = (fdg_per_sample > 0).mean() * 100
ax.set_xlabel("FD gain  (baseline − model) / baseline")
ax.set_ylabel("density")
ax.set_title(f"Per-sample FD gain — {pos:.1f}% of samples improved")
ax.legend()
fig.tight_layout()
save(fig, "03_fd_distribution")

# ==========================================================================================
# 4) Per-patient FD gain — sorted bar chart + baseline vs model FD scatter
# ==========================================================================================

# left: one sorted-bar small multiple per task; right: combined FD scatter.
n_tasks = len(test_tasks)
fig, axes = plt.subplots(1, n_tasks + 1, figsize=(6 * (n_tasks + 1), 5))

# shared y-range across the bar small multiples (scatter keeps its own scale)
all_fdg = np.concatenate([fdg_per_patient[t] for t in test_tasks])
pad = 0.05 * (all_fdg.max() - all_fdg.min())
y_lo, y_hi = all_fdg.min() - pad, all_fdg.max() + pad

for ax, task in zip(axes[:n_tasks], test_tasks):
    fdg = fdg_per_patient[task]
    order = np.argsort(fdg)
    sorted_fdg = fdg[order]
    colors = ["blue" if v >= 0 else "red" for v in sorted_fdg]
    ax.bar(np.arange(len(sorted_fdg)), sorted_fdg, color=colors)
    ax.axhline(0, color="black")
    ax.axhline(
        fdg.mean(),
        color="black",
        linestyle="--",
        label=f"mean = {fdg.mean():.3f}",
    )
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Patient (sorted by FD gain)")
    ax.set_ylabel("FD gain")
    ax.set_title(f"{task} — {(fdg > 0).mean() * 100:.1f}% of patients improved")
    ax.legend()

ax = axes[-1]
hi = 0
for task in test_tasks:
    ax.scatter(
        fd_per_patient_base[task],
        fd_per_patient_pred[task],
        color=TASK_COLORS[task],
        s=10,
        alpha=0.5,
        label=task,
    )
    hi = max(hi, fd_per_patient_base[task].max(), fd_per_patient_pred[task].max())
hi *= 1.05
ax.plot([0, hi], [0, hi], "k--", label="y = x")
ax.set_xlim(0, hi)
ax.set_ylim(0, hi)
ax.set_xlabel("Baseline FD (per patient)")
ax.set_ylabel("Model FD (per patient)")
ax.set_title("Per-patient baseline vs model FD")
ax.legend()
fig.tight_layout()
save(fig, "04_per_patient_fdg")


# ==========================================================================================
# 5) Metrics summary card
# ==========================================================================================

total_mae_pred = np.abs(err_pred).mean()
total_mae_base = np.abs(err_base).mean()
total_rmse_pred = np.sqrt((err_pred**2).mean())
total_rmse_base = np.sqrt((err_base**2).mean())

fig, ax = plt.subplots(figsize=(11, 6))
ax.axis("off")
fig.suptitle(f"Test summary — {TAG}")

# headline metrics averaged across tasks (identical to the single-task values
# when only one task is evaluated)
overall_nll = np.mean([results_by_task[t][1]["nll"] for t in test_tasks])
overall_fd_base = np.mean([results_by_task[t][1]["fd_base"] for t in test_tasks])
overall_fd_pred = np.mean([results_by_task[t][1]["fd_pred"] for t in test_tasks])
overall_fdg = np.mean([results_by_task[t][1]["fdg"] for t in test_tasks])

headline = (
    f"NLL = {overall_nll:.4f}     "
    f"FD baseline = {overall_fd_base:.4f}     "
    f"FD model = {overall_fd_pred:.4f}     "
    f"FD gain = {overall_fdg:.4f}"
)
ax.text(0.5, 0.93, headline, ha="center", va="top", fontsize=12, weight="bold")

# per-task breakdown under the headline
task_line = "     ".join(
    f"{t}: NLL={results_by_task[t][1]['nll']:.3f} "
    f"FDg={results_by_task[t][1]['fdg']:.3f}"
    for t in test_tasks
)
ax.text(0.5, 0.87, task_line, ha="center", va="top", fontsize=9)

# table of per-dim metrics
table_data = [["Dim", "MAE base", "MAE model", "RMSE base", "RMSE model"]]
for d, name in enumerate(DIM_NAMES):
    table_data.append(
        [
            name,
            f"{mae_base_dim[d]:.3f}",
            f"{mae_pred_dim[d]:.3f}",
            f"{rmse_base_dim[d]:.3f}",
            f"{rmse_pred_dim[d]:.3f}",
        ]
    )
table_data.append(
    [
        "TOTAL",
        f"{total_mae_base:.3f}",
        f"{total_mae_pred:.3f}",
        f"{total_rmse_base:.3f}",
        f"{total_rmse_pred:.3f}",
    ]
)

tbl = ax.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center",
    loc="center",
    bbox=[0.05, 0.05, 0.9, 0.78],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
save(fig, "05_metrics_summary")


# ==========================================================================================
# 6) Predicted-sigma calibration — standardized residuals z = (y − μ_pred) / σ_pred
# ==========================================================================================
# If σ_pred is well-calibrated, z should be ~ N(0, 1) per dimension.
#   empirical std(z) > 1  ⇒  model is overconfident (σ_pred too small)
#   empirical std(z) < 1  ⇒  model is underconfident (σ_pred too large)
#   reduced χ² = mean(z²) should be ≈ 1
z = z_per_sample  # [N, 6], in normalized space (scale-invariant)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(
    "Predicted-σ calibration — standardized residuals z = (y − μ_pred) / σ_pred"
)
zz = np.linspace(-5, 5, 400)
gauss_pdf = np.exp(-0.5 * zz**2) / np.sqrt(2 * np.pi)
for d, ax in enumerate(axes.flat):
    for task in test_tasks:
        m = sample_task_labels == task
        # clip extreme tails for the histogram view
        zd_clip = np.clip(z[m, d], -5, 5)
        ax.hist(
            zd_clip,
            bins=80,
            density=True,
            color=TASK_COLORS[task],
            histtype="step",
            linewidth=1.5,
            label=task,
        )
    ax.plot(zz, gauss_pdf, "k--", label="N(0, 1)")
    # stats pooled across tasks
    zd = z[:, d]
    mean_z = zd.mean()
    std_z = zd.std()
    chi2_red = (zd**2).mean()
    cov68 = (np.abs(zd) <= 1.0).mean() * 100  # should be ~68.3% if calibrated
    cov95 = (np.abs(zd) <= 2.0).mean() * 100  # should be ~95.4%
    ax.set_title(
        f"{DIM_NAMES[d]}   mean={mean_z:.2f}  std={std_z:.2f}  χ²ᵣ={chi2_red:.2f}\n|z|≤1: {cov68:.1f}%   |z|≤2: {cov95:.1f}%",
        fontsize=9,
    )
    ax.set_xlabel("z")
    ax.set_ylabel("density")
    ax.set_xlim(-5, 5)
    if d == 0:
        ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "06_sigma_calibration")


# ==========================================================================================
# 7) Random patient — time series with predictive uncertainty band
# ==========================================================================================

random_patient_id = test_ids[0]
# test_dict is keyed by task → patient; visualize this patient on the first task
viz_task = test_tasks[0]
random_patient_data = test_dict[viz_task][random_patient_id]
normalized_random_patient_data = (random_patient_data - mu) / sigma

random_patient_dataset = TimeSeriesDataset(
    np.expand_dims(normalized_random_patient_data, 0),
    [random_patient_id],
    device=device,
)
random_patient_loader = torch.utils.data.DataLoader(
    random_patient_dataset, batch_size=16, shuffle=False
)
predicted_positions = []
true_positions = []
baseline_positions = []
pred_vars = []
with torch.no_grad():
    for _, x, y in random_patient_loader:
        x, y = x.to(device), y.to(device)
        # GRU.forward returns (mean, variance) — already exp'd inside the model.
        y_pred, y_var = model(x)
        y_pred = y_pred.cpu().numpy() * sigma + mu
        true_positions.append(y.cpu().numpy() * sigma + mu)
        predicted_positions.append(y_pred)
        baseline_positions.append(x[:, -1, :].cpu().numpy() * sigma + mu)
        pred_vars.append(y_var.cpu().numpy())

true_positions = np.concatenate(true_positions, axis=0)
predicted_positions = np.concatenate(predicted_positions, axis=0)
baseline_positions = np.concatenate(baseline_positions, axis=0)
pred_vars = np.concatenate(pred_vars, axis=0)

# uncertainty band in denormalized units
pred_std = np.sqrt(pred_vars) * sigma

# multiply the last 3 columns by 50:
true_positions[:, 3:6] *= 50
predicted_positions[:, 3:6] *= 50
baseline_positions[:, 3:6] *= 50
pred_std[:, 3:6] *= 50


fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(
    f"Predicted vs True per dimension — patient {random_patient_id} ({viz_task})"
)
t_axis = np.arange(true_positions.shape[0])
for d, ax in enumerate(axes.flat):
    ax.fill_between(
        t_axis,
        predicted_positions[:, d] - pred_std[:, d],
        predicted_positions[:, d] + pred_std[:, d],
        color="blue",
        alpha=0.3,
        label="±1σ",
    )
    ax.plot(t_axis, true_positions[:, d], color="black", label="True", alpha=0.5)
    ax.plot(
        t_axis, predicted_positions[:, d], color="blue", label="Predicted", alpha=0.1
    )
    ax.plot(t_axis, baseline_positions[:, d], color="red", label="Baseline", alpha=0.1)
    ax.set_xlabel("Time step")
    ax.set_ylabel(DIM_NAMES[d])
    ax.set_title(DIM_NAMES[d])
    if d == 0:
        ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "07_patient_timeseries")


# ==========================================================================================
# 8) FD-gain vs motion magnitude — does the model still help on high-motion frames?
# ==========================================================================================
# Bin frames by baseline FD (the true previous→next motion). A next-frame
# predictor can "win" on calm frames yet fail on the large motions that
# actually corrupt the scan, so we check whether the gain holds as motion grows.
# Quantile bins give roughly equal counts per bin despite the heavy FD tail.
n_bins = 10
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
hi = 0
for task in test_tasks:
    m = sample_task_labels == task
    b = fd_base_per_sample[m]
    p = fd_pred_per_sample[m]
    edges = np.unique(np.quantile(b, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(b, edges[1:-1]), 0, len(edges) - 2)
    centers, model_fd, gain = [], [], []
    for k in range(len(edges) - 1):
        sel = idx == k
        if not sel.any():
            continue
        mean_base = b[sel].mean()
        mean_pred = p[sel].mean()
        centers.append(mean_base)
        model_fd.append(mean_pred)
        # aggregate gain per bin (robust to per-frame fd_base≈0)
        gain.append((mean_base - mean_pred) / mean_base)
    centers = np.array(centers)
    model_fd = np.array(model_fd)
    gain = np.array(gain)

    axes[0].plot(centers, model_fd, "o-", color=TASK_COLORS[task], label=task)
    axes[1].plot(centers, gain, "o-", color=TASK_COLORS[task], label=task)
    hi = max(hi, centers.max(), model_fd.max())

# left: model FD vs motion magnitude, with the previous-frame baseline (y = x);
# points below the line mean the model beats the baseline at that motion level.
hi *= 1.05
axes[0].plot([0, hi], [0, hi], "k--", label="baseline (y = x)")
axes[0].set_xlim(0, hi)
axes[0].set_ylim(0, hi)
axes[0].set_xlabel("Baseline FD = motion magnitude (mm)")
axes[0].set_ylabel("Mean model FD (mm)")
axes[0].set_title("Model FD vs motion magnitude")
axes[0].legend()

# right: FD-gain per motion bin — does the gain survive at high motion?
axes[1].axhline(0, color="black")
axes[1].set_xlabel("Baseline FD = motion magnitude (mm)")
axes[1].set_ylabel("FD-gain  (baseline − model) / baseline")
axes[1].set_title("FD-gain vs motion magnitude")
axes[1].legend()
fig.tight_layout()
save(fig, "08_fdgain_vs_motion")
