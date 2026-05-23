import matplotlib.pyplot as plt
import numpy as np
import torch
from GRU import GRUModel
from metrics import fd, fd_gain
from preprocessing import get_task_dict, load_data
from TimeSeriesDataset import GPUBatchLoader, TimeSeriesDataset


def test(model, test_loader, criterion, device, mu, sigma):
    model.eval()
    test_fd_baselines = []
    test_fd_preds = []
    test_nll = 0
    patient_pred_vs_true = {
        p: {"pred": [], "true": [], "baseline": []} for p in test_loader.dataset.ids
    }
    mu = torch.tensor(mu, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
    with torch.no_grad():
        for p, x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred, y_logvar = model(x)
            test_nll += criterion(y_pred, y, y_logvar).item()

            last_x = x[:, -1, :]
            test_fd_baselines.append(fd(last_x, y, mu, sigma).cpu())
            test_fd_preds.append(fd(y_pred, y, mu, sigma).cpu())
            for i in range(len(p)):
                denormalized_pred = y_pred[i] * sigma + mu
                denormalized_true = y[i] * sigma + mu
                denormalized_baseline = last_x[i] * sigma + mu
                patient_pred_vs_true[p[i]]["pred"].append(
                    denormalized_pred.cpu().numpy()
                )
                patient_pred_vs_true[p[i]]["true"].append(
                    denormalized_true.cpu().numpy()
                )
                patient_pred_vs_true[p[i]]["baseline"].append(
                    denormalized_baseline.cpu().numpy()
                )

        test_nll /= len(test_loader)
        fd_baseline_cat = torch.cat(test_fd_baselines, dim=0)
        fd_model_cat = torch.cat(test_fd_preds, dim=0)

        test_fd_pred = fd_model_cat.mean().item()
        test_fdg = fd_gain(fd_baseline_cat, fd_model_cat).mean().item()

    print(
        f"Test NLL: {test_nll:.4f}, Test FDg: {test_fdg:.4f}, Test FD pred: {test_fd_pred:.4f}"
    )
    return patient_pred_vs_true


train_task = "Resting"
test_task = "Memory"
beta = 0.1

saved_dict = torch.load(
    f"../checkpoints/GRU_train{train_task}_test{test_task}_beta{beta}.pth"
)

model_state_dict = saved_dict["model_state_dict"]
test_patients_ids = saved_dict["test_ids"]
mu = saved_dict["mu"]
sigma = saved_dict["sigma"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set True to keep the whole dataset resident on GPU and use the vectorized
# GPUBatchLoader; set False for the classic CPU-dataset + DataLoader path.
use_gpu_loader = True

dataset_device = device if use_gpu_loader else "cpu"
data_paths = {
    "Resting": "../datasets/HCP/RestingStateLR_dataset",
    "Memory": "../datasets/HCP/MemoryTaskLR_dataset",
    "Language": "../datasets/HCP/LanguageTaskLR_dataset",
}
patient_dict = load_data(data_paths)
test_dict = get_task_dict(patient_dict, test_task)
test_data = np.array([test_dict[pid] for pid in test_patients_ids])
test_data = (test_data - mu) / sigma

test_dataset = TimeSeriesDataset(test_data, test_patients_ids, device=dataset_device)

batch_size = 8192
use_gpu_loader = True
if use_gpu_loader:
    test_loader = GPUBatchLoader(test_dataset, batch_size=batch_size, shuffle=False)
else:
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GRUModel(
    input_dim=6, hidden_dim=128, output_dim=6, num_layers=2, dropout=0.5
).to(device)


model.load_state_dict(model_state_dict)
criterion = torch.nn.GaussianNLLLoss()


patient_pred_true_base = test(
    model,
    test_loader,
    criterion,
    device,
    mu=mu,
    sigma=sigma,
)


# with open("patient_pred_vs_true_GRU_trainRESTING_testMEMORY.pkl", "wb") as f:
#     pickle.dump(patient_pred_vs_true, f)

# scatter plot of true vs predicted values
true_values = []
pred_values = []
baseline_values = []
for patient_id in patient_pred_true_base.keys():
    for true, pred, base in zip(
        patient_pred_true_base[patient_id]["true"],
        patient_pred_true_base[patient_id]["pred"],
        patient_pred_true_base[patient_id]["baseline"],
    ):
        true_values.append(true)
        pred_values.append(pred)
        baseline_values.append(base)

true_values = np.array(true_values)
pred_values = np.array(pred_values)
baseline_values = np.array(baseline_values)

plt.figure(figsize=(8, 8))
plt.scatter(true_values, pred_values, alpha=0.5, label="Predicted vs True")
plt.scatter(true_values, baseline_values, alpha=0.5, label="Baseline vs True")
plt.plot(
    [true_values.min(), true_values.max()],
    [true_values.min(), true_values.max()],
    "k--",
    label="Ideal",
)
plt.xlabel("True Values")
plt.ylabel("Predicted / Baseline Values")
plt.title("Predicted vs True Values and Baseline vs True Values")
plt.legend()
plt.grid()
plt.show()

plt.scatter(pred_values, baseline_values, alpha=0.5)
plt.xlabel("Predicted Values")
plt.ylabel("Baseline Values")
plt.title("Predicted Values vs Baseline Values")
plt.grid()
plt.show()

# Get all the windows from the first patient in the test set
random_patient_id = list(patient_pred_true_base.keys())[0]
random_patient_data = test_dict[random_patient_id]


normalized_random_patient_data = (random_patient_data - mu) / sigma

random_patient_dataset = TimeSeriesDataset(
    np.expand_dims(normalized_random_patient_data, 0),
    [random_patient_id],
    device=device,
)
random_patient_loader = torch.utils.data.DataLoader(
    random_patient_dataset, batch_size=1, shuffle=False
)
predicted_positions = []
true_positions = []
with torch.no_grad():
    for _, x, y in random_patient_loader:
        x, y = x.to(device), y.to(device)
        y_pred, _ = model(x)
        y_pred = y_pred.cpu().numpy() * sigma + mu
        true_positions.append(y.cpu().numpy() * sigma + mu)
        predicted_positions.append(y_pred)

true_positions = np.concatenate(true_positions, axis=0)
predicted_positions = np.concatenate(predicted_positions, axis=0)

dim_names = ["Tx (mm)", "Ty (mm)", "Tz (mm)", "Rx (rad)", "Ry (rad)", "Rz (rad)"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
fig.suptitle(f"Predicted vs True per dimension — patient {random_patient_id}")
for d, ax in enumerate(axes.flat):
    ax.plot(true_positions[:, d], label="True")
    ax.plot(predicted_positions[:, d], label="Predicted", alpha=0.8)
    ax.set_title(dim_names[d])
    ax.set_xlabel("Time step")
    ax.set_ylabel(dim_names[d])
    ax.grid(True)
    ax.legend()
fig.tight_layout()
plt.show()
