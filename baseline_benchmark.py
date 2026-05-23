import matplotlib.pyplot as plt
import numpy as np
from preprocessing import get_task_dict, load_data

data_paths = {
    "Resting": "RECENTRE-main/HCP/RestingStateLR_dataset",
    "Memory": "RECENTRE-main/HCP/MemoryTaskLR_dataset",
    "Language": "RECENTRE-main/HCP/LanguageTaskLR_dataset",
}


patient_dict = load_data(data_paths)
resting_dict = get_task_dict(patient_dict, "Resting")


# Take the first patient, plot lag1 vs true for each dimension, and then scatter plot lag1 vs true for each dimension
random_resting_patient_data = resting_dict[list(resting_dict.keys())[0]].copy()
random_resting_patient_data[:, 3:6] *= 50
time_steps = random_resting_patient_data.shape[0]
lag1_data = np.roll(random_resting_patient_data, shift=1, axis=0)[1:]
true_data = random_resting_patient_data[1:]


# time series lag1 vs true for each dimension
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()
for i in range(6):
    axes[i].plot(true_data[:, i], label="True")
    axes[i].plot(lag1_data[:, i], label="Lag1", alpha=0.8)
    axes[i].set_title(f"Dimension {i + 1}")
    if i >= 3:
        axes[i].set_xlabel("Time Steps")
    if i % 3 == 0:
        axes[i].set_ylabel("Value")

axes[0].legend()
plt.suptitle(
    "Baseline Benchmark: Lag1 vs True, patient: " + list(resting_dict.keys())[0]
)
plt.tight_layout()
plt.show()

# scatter plot lag1 vs true for each dimension
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()
for i in range(6):
    axes[i].scatter(true_data[:, i], lag1_data[:, i], alpha=0.3)
    axes[i].set_title(f"Dimension {i + 1}")
    if i >= 3:
        axes[i].set_xlabel("True values")
    if i % 3 == 0:
        axes[i].set_ylabel("Lag1 values")

axes[0].legend()
plt.suptitle(
    "Baseline Benchmark: Lag1 vs True scatterplot, patient: "
    + list(resting_dict.keys())[0]
)
plt.tight_layout()
plt.show()


# scatter plot lag1 vs true for each dimension for all patients (overlapping scatter plots)
true_data = np.stack(list(resting_dict.values()), axis=0)
true_data[:, :, 3:6] *= 50
lag1_data = np.roll(true_data, shift=1, axis=1)[::5, 1:]
true_data = true_data[::5, 1:]
print(true_data.shape, lag1_data.shape)

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

for i in range(6):
    axes[i].scatter(true_data[:, :, i], lag1_data[:, :, i], alpha=0.3)
    axes[i].set_title(f"Dimension {i + 1}")
    axes[i].set_aspect("equal", adjustable="box")
    if i >= 3:
        axes[i].set_xlabel("True Values")
    if i % 3 == 0:
        axes[i].set_ylabel("Lag1 Values")

# Ensure same range for x and y
xlims = axes[0].get_xlim()
ylims = axes[0].get_ylim()
vmin = min(xlims[0], ylims[0])
vmax = max(xlims[1], ylims[1])
plt.setp(axes, xlim=(vmin, vmax), ylim=(vmin, vmax))

plt.suptitle(
    "Baseline Benchmark: Lag1 vs True Scatter Plot, all patients (20% subsample)"
)
plt.tight_layout()
plt.show()
