import os

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from GRU import GRUModel
from metrics import fd, fd_gain
from TimeSeriesDataset import GPUBatchLoader, TimeSeriesDataset


def train(
    model,
    train_loader,
    val_loader,
    train_ids,
    val_ids,
    test_ids,
    optimizer,
    criterion,
    device,
    epochs,
    mu,
    sigma,
    checkpoint_path,
    patience=10,
    beta=0.1,
):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    # mu/sigma as tensors on device for FD denormalization
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)

    train_loss_history = []
    val_loss_history = []

    best_val_fdg = -1e9
    early_counter = 0
    save_dict = {}
    pbar = tqdm.trange(epochs)
    for epoch in pbar:
        # accumulate sample-weighted NLL so the reported per-sample mean is
        # correct even when the last batch is smaller than the rest, and
        # consistent with the sample-weighted FDg computed below. Keep
        # accumulators on the GPU and sync once at epoch end — per-batch
        # .item()/.cpu() calls were forcing a GPU→host stall every step.
        train_nll_sum = torch.zeros((), device=device)
        train_n = 0
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
            bs = y.size(0)
            train_nll_sum += nll.detach() * bs
            train_n += bs
            train_fd_baselines.append(fd_baseline.detach())
            train_fd_preds.append(fd_pred.detach())

        train_fd_baseline_cat = torch.cat(train_fd_baselines, dim=0)
        train_fd_pred_cat = torch.cat(train_fd_preds, dim=0)
        train_fdg_t = fd_gain(train_fd_baseline_cat, train_fd_pred_cat).mean()
        train_fd_pred_t = train_fd_pred_cat.mean()
        train_nll_t = train_nll_sum / train_n
        # single GPU→host sync per epoch
        train_nll, train_fdg, train_fd_pred = (
            train_nll_t.item(),
            train_fdg_t.item(),
            train_fd_pred_t.item(),
        )
        train_loss = train_nll - beta * train_fdg

        model.eval()
        with torch.no_grad():
            val_fd_baselines = []
            val_fd_preds = []
            val_nll_sum = torch.zeros((), device=device)
            val_n = 0
            for _, x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred, y_logvar = model(x)
                bs = y.size(0)
                val_nll_sum += criterion(y_pred, y, y_logvar) * bs
                val_n += bs

                last_x = x[:, -1, :]
                val_fd_baselines.append(fd(last_x, y, mu_t, sigma_t))
                val_fd_preds.append(fd(y_pred, y, mu_t, sigma_t))

            fd_baseline_cat = torch.cat(val_fd_baselines, dim=0)
            fd_model_cat = torch.cat(val_fd_preds, dim=0)
            val_nll_t = val_nll_sum / val_n
            val_fd_pred_t = fd_model_cat.mean()
            val_fdg_t = fd_gain(fd_baseline_cat, fd_model_cat).mean()
            # single GPU→host sync per epoch
            val_nll, val_fd_pred, val_fdg = (
                val_nll_t.item(),
                val_fd_pred_t.item(),
                val_fdg_t.item(),
            )
            val_loss = val_nll - beta * val_fdg

        scheduler.step(val_loss)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

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
            save_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
                "mu": mu,
                "sigma": sigma,
                "beta": beta,
                "train_task": train_task,
                "test_task": test_task,
                "epochs": epochs,
            }
        else:
            early_counter += 1
            if early_counter >= patience:
                break

    torch.save(save_dict, checkpoint_path)
    return train_loss_history, val_loss_history


if __name__ == "__main__":
    train_task = "R"
    test_task = "M"
    base_dir = "../datasets"

    train_dict = np.load(f"{base_dir}/{train_task}_dict.npy", allow_pickle=True).item()
    val_test_dict = np.load(
        f"{base_dir}/{test_task}_dict.npy", allow_pickle=True
    ).item()

    train_data = np.stack(list(train_dict.values()), axis=0)
    train_data_ids = list(train_dict.keys())
    test_data = np.stack(list(val_test_dict.values()), axis=0)
    num_patients = len(train_data)

    val_percent = 0.5

    # seed so that i can compare the same split across many models
    rng = np.random.default_rng(42)

    # Split val and test patients with non-overlapping sets of patients
    val_patients = rng.choice(
        num_patients, size=int(val_percent * num_patients), replace=False
    )
    test_patients = np.setdiff1d(np.arange(num_patients), val_patients)

    # Split val and test patient ids
    val_patients_ids = [list(val_test_dict.keys())[i] for i in val_patients]
    test_patients_ids = [list(val_test_dict.keys())[i] for i in test_patients]

    # save the dictionary of patient_id : data for the test set, to be saved in checkpoints
    # and used for testing after training
    test_dict = {pid: val_test_dict[pid] for pid in test_patients_ids}

    # normalize the datasets using the mu and sigma of the training dataset (per dimension)
    mu = train_data.mean(axis=(0, 1))  # shape [6]
    sigma = train_data.std(axis=(0, 1))  # shape [6]
    train_data = (train_data - mu) / sigma
    test_data = (test_data - mu) / sigma

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set True to keep the whole dataset on GPU and use the faster
    # GPUBatchLoader; set False for the classic CPU-dataset + DataLoader path.
    use_gpu_loader = True

    dataset_device = device if use_gpu_loader else "cpu"
    train_dataset = TimeSeriesDataset(train_data, train_data_ids, device=dataset_device)
    val_dataset = TimeSeriesDataset(
        test_data[val_patients], val_patients_ids, device=dataset_device
    )
    test_dataset = TimeSeriesDataset(
        test_data[test_patients], test_patients_ids, device=dataset_device
    )

    batch_size = 8192

    if use_gpu_loader:
        train_loader = GPUBatchLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = GPUBatchLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = GPUBatchLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    model = GRUModel(
        input_dim=6, hidden_dim=128, output_dim=6, num_layers=2, dropout=0.5
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.GaussianNLLLoss()
    beta = 0.5

    os.makedirs("../checkpoints", exist_ok=True)
    epochs = 100
    RUN_TAG = f"GRU_{train_task}v{test_task}_beta{beta}_ep{epochs}"
    train_loss_history, val_loss_history = train(
        model,
        train_loader,
        val_loader,
        train_ids=train_data_ids,
        val_ids=val_patients_ids,
        test_ids=test_patients_ids,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
        mu=mu,
        sigma=sigma,
        checkpoint_path=f"../checkpoints/{RUN_TAG}.pth",
        patience=100,
        beta=beta,
    )

    # train and val loss history plots:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Train and Val Loss - GRU train {train_task} test {test_task} beta {beta}"
    )
    plt.legend()
    plt.grid()
    plt.savefig(f"../checkpoints/{RUN_TAG}_loss_history.png")
    plt.show()
