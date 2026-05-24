import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from GRU import GRUModel
from metrics import fd, fd_gain
from TimeSeriesDataset import GPUBatchLoader, TimeSeriesDataset

# Rotations are scaled by 50 mm (avg head radius) so every dim ends up in mm.
DIM_NAMES = ["Tx (mm)", "Ty (mm)", "Tz (mm)", "Rx (mm)", "Ry (mm)", "Rz (mm)"]


def save(fig, name):
    path = os.path.join(RESULTS_DIR, f"{name}_{TAG}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def test(model, test_loader, criterion, device, mu, sigma):
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
    with torch.no_grad():
        for p, x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # GRU.forward returns (mean, variance) — already exp'd inside the model.
            y_pred, y_var = model(x)
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

# Pick which checkpoint to evaluate. The tag fields (train_task, test_task,
# beta, epochs) are read from inside the checkpoint, so this path is the only
# thing to change between runs.
FILENAME = "GRU_RvM_beta0.5_ep100.pth"
CHECKPOINT_PATH = f"checkpoints/{FILENAME}"

device = "cpu"

# Load checkpoint + data
saved_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)
model_state_dict = saved_dict["model_state_dict"]
test_ids = saved_dict["test_ids"]
train_task = saved_dict["train_task"]
test_task = saved_dict["test_task"]
beta = saved_dict["beta"]
epochs = saved_dict["epochs"]
mu = saved_dict["mu"]
sigma = saved_dict["sigma"]

assert FILENAME == f"GRU_{train_task}v{test_task}_beta{beta}_ep{epochs}.pth", (
    "FILENAME does not match checkpoint contents. Please update the filename variable to match the checkpoint you want to evaluate."
)


TAG = FILENAME.removesuffix(".pth")
RESULTS_DIR = f"results/{TAG}"
os.makedirs(RESULTS_DIR, exist_ok=True)


test_dict = np.load(f"datasets/{test_task}_dict.npy", allow_pickle=True).item()

test_data = np.array(
    [test_dict[pid] for pid in test_ids]
)  # [num_patients, patient_frames, 6]
test_data = (test_data - mu) / sigma


test_dataset = TimeSeriesDataset(test_data, test_ids, device=device)
test_loader = GPUBatchLoader(test_dataset, batch_size=1024, shuffle=False)

model = GRUModel(
    input_dim=6, hidden_dim=128, output_dim=6, num_layers=2, dropout=0.5
).to(device)
model.load_state_dict(model_state_dict)
criterion = torch.nn.GaussianNLLLoss()

patient_pred_true_base, metrics = test(
    model, test_loader, criterion, device, mu=mu, sigma=sigma
)


"""
==================================================
        Metrics and visualizations
==================================================
"""

# Aggregate per-patient arrays into flat matrices and per-patient FD values.
true_values = []
pred_values = []
baseline_values = []
fd_per_patient_pred = []
fd_per_patient_base = []
fdg_per_patient = []

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

    # compute per patient fd both for the model and the baseline by summing over dimensions and averaging over frames
    scaled_pred = p_pred.copy()
    scaled_true = p_true.copy()
    scaled_base = p_base.copy()

    # scaled_pred: [patient_frames, 6]

    fd_pred = np.abs(scaled_pred - scaled_true).sum(axis=1)
    fd_base = np.abs(scaled_base - scaled_true).sum(axis=1)

    fd_per_patient_pred.append(fd_pred.mean())
    fd_per_patient_base.append(fd_base.mean())
    fdg_per_patient.append(((fd_base - fd_pred) / fd_base).mean())


# transform to np arrays
true_values = np.concatenate(true_values, axis=0)  # [num_patients*patient_frames, 6]
pred_values = np.concatenate(pred_values, axis=0)
baseline_values = np.concatenate(baseline_values, axis=0)
fd_per_patient_pred = np.array(fd_per_patient_pred)  # [num_patients]
fd_per_patient_base = np.array(fd_per_patient_base)
fdg_per_patient = np.array(fdg_per_patient)

# compute model and baseline errors for MAE and RMSE
err_pred = pred_values - true_values
err_base = baseline_values - true_values

mae_pred_dim = np.abs(err_pred).mean(axis=0)
mae_base_dim = np.abs(err_base).mean(axis=0)
rmse_pred_dim = np.sqrt((err_pred**2).mean(axis=0))
rmse_base_dim = np.sqrt((err_base**2).mean(axis=0))

# ==========================================================================================
# 1) Per-dimension error: scatter of baseline vs model error (MAE and RMSE)
# ==========================================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric_base, metric_model, label in zip(
    axes,
    [mae_base_dim, rmse_base_dim],
    [mae_pred_dim, rmse_pred_dim],
    ["MAE", "RMSE"],
):
    ax.scatter(metric_base, metric_model, color="blue", s=60)
    for d, name in enumerate(DIM_NAMES):
        ax.annotate(
            name,
            (metric_base[d], metric_model[d]),
            fontsize=9,
            xytext=(5, 5),
            textcoords="offset points",
        )
    hi = max(metric_base.max(), metric_model.max()) * 1.1
    ax.plot([0, hi], [0, hi], "k--", label="y = x")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(f"Baseline {label} (mm)")
    ax.set_ylabel(f"Model {label} (mm)")
    ax.set_title(f"{label} per dimension (below y=x means model is better)")
    ax.legend()
fig.tight_layout()
save(fig, "01_error_per_dimension")


# ==========================================================================================
# 2) True vs Predicted scatter per dimension
# ==========================================================================================

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("True vs Predicted per dimension")
for d, ax in enumerate(axes.flat):
    x = true_values[:, d]
    y = pred_values[:, d]
    ax.scatter(x, y, s=2, alpha=0.3, color="blue")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", label="y = x")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(DIM_NAMES[d])
    ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "02_true_vs_predicted")

# ==========================================================================================
# 3) Framewise-displacement (FD) distribution — model vs baseline
# ==========================================================================================

fd_pred_arr = metrics["fd_pred_per_sample"]
fd_base_arr = metrics["fd_base_per_sample"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
hi = max(np.percentile(fd_pred_arr, 99.5), np.percentile(fd_base_arr, 99.5))
bins = np.linspace(0, hi, 80)
ax.hist(
    fd_base_arr,
    bins=bins,
    color="red",
    alpha=0.5,
    label=f"Baseline (mean={fd_base_arr.mean():.3f})",
)
ax.hist(
    fd_pred_arr,
    bins=bins,
    color="blue",
    alpha=0.5,
    label=f"Model (mean={fd_pred_arr.mean():.3f})",
)
ax.set_xlabel("Framewise displacement (mm)")
ax.set_ylabel("count")
ax.set_title("FD distribution — model vs baseline")
ax.legend()

ax = axes[1]
fdg_per_sample = (fd_base_arr - fd_pred_arr) / fd_base_arr
# clip long tails so the central mass is visible
gain_lim = np.percentile(np.abs(fdg_per_sample), 99)
ax.hist(
    np.clip(fdg_per_sample, -gain_lim, gain_lim),
    bins=np.linspace(-gain_lim, gain_lim, 80),
    color="purple",
    alpha=0.7,
)
ax.axvline(0, color="black")
ax.axvline(
    fdg_per_sample.mean(),
    color="purple",
    linestyle="--",
    label=f"mean = {fdg_per_sample.mean():.3f}",
)
pos = (fdg_per_sample > 0).mean() * 100
ax.set_xlabel("FD gain  (baseline − model) / baseline")
ax.set_ylabel("count")
ax.set_title(f"Per-sample FD gain — {pos:.1f}% of samples improved")
ax.legend()
fig.tight_layout()
save(fig, "03_fd_distribution")

# ==========================================================================================
# 4) Per-patient FD gain — sorted bar chart + baseline vs model FD scatter
# ==========================================================================================

order = np.argsort(fdg_per_patient)
sorted_fdg = fdg_per_patient[order]
colors = ["blue" if v >= 0 else "red" for v in sorted_fdg]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.bar(np.arange(len(sorted_fdg)), sorted_fdg, color=colors)
ax.axhline(0, color="black")
ax.axhline(
    fdg_per_patient.mean(),
    color="black",
    linestyle="--",
    label=f"mean = {fdg_per_patient.mean():.3f}",
)
ax.set_xlabel("Patient (sorted by FD gain)")
ax.set_ylabel("FD gain")
ax.set_title(
    f"Per-patient FD gain — {(fdg_per_patient > 0).mean() * 100:.1f}% of patients improved"
)
ax.legend()

ax = axes[1]
ax.scatter(fd_per_patient_base, fd_per_patient_pred, color="blue", s=30)
hi = max(fd_per_patient_base.max(), fd_per_patient_pred.max()) * 1.05
ax.plot([0, hi], [0, hi], "k--", label="y = x")
ax.set_xlim(0, hi)
ax.set_ylim(0, hi)
ax.set_xlabel("Baseline FD (per patient)")
ax.set_ylabel("Model FD (per patient)")
ax.set_title("Per-patient baseline vs model FD (below y=x ⇒ model improves)")
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

headline = (
    f"NLL = {metrics['nll']:.4f}     "
    f"FD baseline = {metrics['fd_base']:.4f}     "
    f"FD model = {metrics['fd_pred']:.4f}     "
    f"FD gain = {metrics['fdg']:.4f}"
)
ax.text(0.5, 0.93, headline, ha="center", va="top", fontsize=12, weight="bold")

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
z = metrics["z_per_sample"]  # [N, 6], in normalized space (scale-invariant)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(
    "Predicted-σ calibration — standardized residuals z = (y − μ_pred) / σ_pred"
)
zz = np.linspace(-5, 5, 400)
gauss_pdf = np.exp(-0.5 * zz**2) / np.sqrt(2 * np.pi)
for d, ax in enumerate(axes.flat):
    zd = z[:, d]
    # clip extreme tails for the histogram view
    zd_clip = np.clip(zd, -5, 5)
    ax.hist(zd_clip, bins=80, density=True, color="blue", alpha=0.5, label="empirical")
    ax.plot(zz, gauss_pdf, "k--", label="N(0, 1)")
    mean_z = zd.mean()
    std_z = zd.std()
    chi2_red = (zd**2).mean()
    cov68 = (np.abs(zd) <= 1.0).mean() * 100  # should be ~68.3% if calibrated
    cov95 = (np.abs(zd) <= 2.0).mean() * 100  # should be ~95.4%
    ax.set_title(
        f"{DIM_NAMES[d]}   mean={mean_z:.2f}  std={std_z:.2f}  χ²ᵣ={chi2_red:.2f}\n"
        f"|z|≤1: {cov68:.1f}%   |z|≤2: {cov95:.1f}%",
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
random_patient_data = test_dict[random_patient_id]
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
fig.suptitle(f"Predicted vs True per dimension — patient {random_patient_id}")
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
