import os

import numpy as np
import torch
import torch.nn as nn
import tqdm
from GRU import GRUModel
from preprocessing import get_task_dict, load_data
from TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader


def fd(pred_frame, true_frame, mu, sig):
    # denormalize before computing FD. frame shape: [batch_size, D]
    pred_frame = pred_frame * sig + mu
    true_frame = true_frame * sig + mu
    translation_error = torch.abs(pred_frame[:, :3] - true_frame[:, :3]).sum(dim=1)
    rotation_error = 50 * torch.abs(pred_frame[:, 3:] - true_frame[:, 3:]).sum(dim=1)
    return translation_error + rotation_error


def fd_gain(fd_baseline, fd_pred):
    # fd_baseline and fd_pred shape: [batch_size]
    gain = (fd_baseline - fd_pred) / (fd_baseline + 1e-6)
    return gain


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    mu,
    sig,
    checkpoint_path,
    patience=10,
    beta=0.1,
):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    # mu/sig as tensors on device for FD denormalization
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sig, dtype=torch.float32, device=device)

    best_val_fdg = -1e9
    early_counter = 0
    pbar = tqdm.trange(epochs)
    for epoch in pbar:
        train_loss = 0
        train_fd_baselines = []
        train_fd_preds = []
        model.train()
        for _, x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred, y_var = model(x)

            nll = criterion(y_pred, y, y_var)

            last_x = x[:, -1, :]
            fd_baseline = fd(last_x, y, mu_t, sigma_t)
            fd_pred = fd(y_pred, y, mu_t, sigma_t)
            gain = fd_gain(fd_baseline, fd_pred)
            loss = nll - beta * gain.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_fd_baselines.append(fd_baseline.detach().cpu())
            train_fd_preds.append(fd_pred.detach().cpu())

        train_loss /= len(train_loader)
        train_fd_baseline_cat = torch.cat(train_fd_baselines, dim=0)
        train_fd_pred_cat = torch.cat(train_fd_preds, dim=0)
        train_fdg = fd_gain(train_fd_baseline_cat, train_fd_pred_cat).mean().item()
        train_fd_pred = train_fd_pred_cat.mean().item()

        model.eval()
        with torch.no_grad():
            val_fd_baselines = []
            val_fd_preds = []
            val_nll = 0
            for _, x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred, y_logvar = model(x)
                val_nll += criterion(y_pred, y, y_logvar).item()

                last_x = x[:, -1, :]
                val_fd_baselines.append(fd(last_x, y, mu_t, sigma_t).cpu())
                val_fd_preds.append(fd(y_pred, y, mu_t, sigma_t).cpu())

            val_nll /= len(val_loader)
            fd_baseline_cat = torch.cat(val_fd_baselines, dim=0)
            fd_model_cat = torch.cat(val_fd_preds, dim=0)

            val_fd_pred = fd_model_cat.mean().item()
            val_fdg = fd_gain(fd_baseline_cat, fd_model_cat).mean().item()
            val_loss = val_nll - beta * val_fdg

        scheduler.step(val_loss)

        pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "train_fdg": f"{train_fdg:.4f}",
                "train_fd_pred": f"{train_fd_pred:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_fdg": f"{val_fdg:.4f}",
                "val_fd_pred": f"{val_fd_pred:.4f}",
            }
        )

        if val_fdg > best_val_fdg:
            best_val_fdg = val_fdg
            early_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            early_counter += 1
            if early_counter >= patience:
                break

    return train_loss


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


if __name__ == "__main__":
    # dictionary with the paths to the three datasets
    data_paths = {
        "Resting": "../datasets/HCP/RestingStateLR_dataset",
        "Memory": "../datasets/HCP/MemoryTaskLR_dataset",
        "Language": "../datasets/HCP/LanguageTaskLR_dataset",
    }

    patient_dict = load_data(data_paths)
    resting_dict = get_task_dict(patient_dict, "Resting")
    memory_dict = get_task_dict(patient_dict, "Memory")

    resting_data = np.stack(list(resting_dict.values()), axis=0)
    resting_patient_ids = list(resting_dict.keys())
    memory_data = np.stack(list(memory_dict.values()), axis=0)
    num_patients = len(resting_data)

    val_percent = 0.3

    # seed so that i can compare the same split across many models
    rng = np.random.default_rng(42)

    val_patients = rng.choice(
        num_patients, size=int(val_percent * num_patients), replace=False
    )
    val_patients_ids = [np.array(list(memory_dict.keys()))[i] for i in val_patients]
    test_patients = np.setdiff1d(np.arange(num_patients), val_patients)
    test_patients_ids = [np.array(list(memory_dict.keys()))[i] for i in test_patients]

    # normalize the datasets using the mean and std of the resting dataset (per dimension)
    mean = resting_data.mean(axis=(0, 1))  # shape [6]
    std = resting_data.std(axis=(0, 1))  # shape [6]
    resting_data = (resting_data - mean) / std
    memory_data = (memory_data - mean) / std

    print("Train mean: ", mean, "\t Train std: ", std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resting_dataset = TimeSeriesDataset(
        resting_data, resting_patient_ids, device=device
    )
    val_dataset = TimeSeriesDataset(
        memory_data[val_patients], val_patients_ids, device=device
    )
    test_dataset = TimeSeriesDataset(
        memory_data[test_patients], test_patients_ids, device=device
    )

    # data lives entirely on GPU -> num_workers=0 (CUDA tensors can't cross worker
    # processes) and pin_memory=False (no H2D copy to pin for).
    train_loader = DataLoader(
        resting_dataset, batch_size=8192, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8192, shuffle=False, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8192, shuffle=False, num_workers=0, pin_memory=False
    )

    model = GRUModel(
        input_dim=6, hidden_dim=128, output_dim=6, num_layers=2, dropout=0.5
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.GaussianNLLLoss()

    os.makedirs("checkpoints", exist_ok=True)
    epochs = 100
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs,
        mu=mean,
        sig=std,
        checkpoint_path="checkpoints/GRU_trainRESTING_testMEMORY.pth",
        patience=10,
    )

    model.load_state_dict(torch.load("checkpoints/GRU_trainRESTING_testMEMORY.pth"))
    patient_pred_vs_true = test(
        model,
        test_loader,
        criterion,
        device,
        mu=mean,
        sigma=std,
    )

    # save to file
    import pickle

    with open("patient_pred_vs_true_GRU_trainRESTING_testMEMORY.pkl", "wb") as f:
        pickle.dump(patient_pred_vs_true, f)
