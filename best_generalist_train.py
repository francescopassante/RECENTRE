import os

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from GRU import GRUModel
from metrics import fd, fd_gain
from TimeSeriesDataset import GPUBatchLoader, MultiTaskLoader, TimeSeriesDataset


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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
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
                "tasks": ["R", "M", "L"],
                "epochs": epochs,
            }
        else:
            early_counter += 1
            if early_counter >= patience:
                break

    torch.save(save_dict, checkpoint_path)
    return train_loss_history, val_loss_history


"""
=============================================================================================
The best generalist model is a model that is trained on a set of patients for all the tasks.
The evaluation is performed on a disjoint set of patients, across all three tasks.
This can then be used as a basis for fine tuning on the patient and task of interest
=============================================================================================
"""

if __name__ == "__main__":
    base_dir = "datasets"

    # seed so that i can compare the same split across many models
    rng = np.random.default_rng(42)

    task_dicts = {
        "R": np.load(f"{base_dir}/R_dict.npy", allow_pickle=True).item(),
        "M": np.load(f"{base_dir}/M_dict.npy", allow_pickle=True).item(),
        "L": np.load(f"{base_dir}/L_dict.npy", allow_pickle=True).item(),
    }

    train_percent = 0.75
    test_percent = 0.10
    # val gets the remainder (0.15)

    patient_ids = np.array(sorted(task_dicts["R"].keys()))

    train_ids = rng.choice(patient_ids, size=int(train_percent * len(patient_ids)), replace=False)
    test_val_ids = np.setdiff1d(patient_ids, train_ids)
    test_ids = rng.choice(test_val_ids, size=int(test_percent * len(patient_ids)), replace=False)
    val_ids = np.setdiff1d(test_val_ids, test_ids)

    def stack(task_dict, ids):
        # [N, T_task, 6] — T is constant within a task but differs across tasks
        return np.stack([task_dict[pid] for pid in ids], axis=0)

    # {"train": {"R": [N_tr, T_R, 6], "M": [N_tr, T_M, 6], "L": [N_tr, T_L, 6]}, "val": {...}, "test": {...}}
    splits = {
        split_name: {task: stack(task_dict, split_ids) for task, task_dict in task_dicts.items()}
        for split_name, split_ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))
    }

    # normalize the datasets using mu/sigma pooled across all training data
    # (all 3 tasks) — one (mu, sigma) per dimension stored in the checkpoint
    # so eval applies the same normalization regardless of which task it scores.
    # probably rescaling with the task-specific mu/sigma would yield better performance, but
    # this way it's more task-agnostic.

    # To compute the total mean, collapse [N_tr, T_task, 6] -> [N_tr * T_task, 6] and concatenate tasks
    # -> [N_tr*T_R + N_tr*T_M + N_tr*T_L, 6] -> mean -> [6]
    train_frames = np.concatenate([arr.reshape(-1, 6) for arr in splits["train"].values()], axis=0)

    mu = train_frames.mean(axis=0)  # shape [6]
    sigma = train_frames.std(axis=0)  # shape [6]
    for split in splits.values():
        for task in split:
            split[task] = (split[task] - mu) / sigma

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set True to keep the whole dataset on GPU and use the faster
    # GPUBatchLoader; set False for the classic CPU-dataset + DataLoader path.
    use_gpu_loader = True

    dataset_device = device if use_gpu_loader else "cpu"

    def build_loader(split_name, split_ids, shuffle):
        # one TimeSeriesDataset per (split, task) because TimeSeriesDataset
        # requires a fixed T across its N dim — tasks have different T
        per_task_loaders = []
        for task, arr in splits[split_name].items():
            ds = TimeSeriesDataset(arr, split_ids, device=dataset_device)
            if use_gpu_loader:
                per_task_loaders.append(GPUBatchLoader(ds, batch_size=batch_size, shuffle=shuffle))
            else:
                per_task_loaders.append(
                    DataLoader(
                        ds,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=4,
                        pin_memory=True,
                    )
                )
        return MultiTaskLoader(per_task_loaders, shuffle=shuffle, seed=42)

    batch_size = 32768

    train_loader = build_loader("train", train_ids, shuffle=True)
    val_loader = build_loader("val", val_ids, shuffle=False)
    test_loader = build_loader("test", test_ids, shuffle=False)

    model = GRUModel(input_dim=6, hidden_dim=128, output_dim=6, num_layers=2, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.GaussianNLLLoss()
    beta = 0.5

    os.makedirs("checkpoints", exist_ok=True)
    epochs = 100
    RUN_TAG = f"GRU_generalist_beta{beta}_ep{epochs}"
    train_loss_history, val_loss_history = train(
        model,
        train_loader,
        val_loader,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
        mu=mu,
        sigma=sigma,
        checkpoint_path=f"checkpoints/{RUN_TAG}.pth",
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
    plt.title(f"Train and Val Loss - GRU generalist (R+M+L) beta {beta}")
    plt.legend()
    plt.grid()
    plt.savefig(f"checkpoints/{RUN_TAG}_loss_history.png")
    plt.show()
