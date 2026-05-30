import os

import numpy as np
import torch
import torch.nn as nn
import tqdm

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
    scheduler,
    device,
    epochs,
    mu,
    sigma,
    checkpoint_path,
    train_task,
    test_task,
    patience=10,
    beta=0.1,
):
    # mu/sigma as tensors on device for FD denormalization
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)

    train_loss_history = []
    val_loss_history = []

    best_val_fdg = float("-inf")
    early_counter = 0
    checkpoint_dict = {}
    pbar = tqdm.trange(epochs)
    for epoch in pbar:
        # accumulate sample-weighted NLL so the reported per-sample mean is
        # correct even when the last batch is smaller than the rest
        train_nll_sum = torch.zeros((), device=device)

        # train_n is the total number of samples, used to compute mean NLL.
        train_n = 0

        # arrays that store the fd_baseline and fd_model for each frame
        train_fd_baselines = []
        train_fd_preds = []
        model.train()
        for _, x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred, y_var = model(x)

            nll = criterion(y_pred, y, y_var)

            # get last_x to compute baseline metrics
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

        train_fdg_epoch = fd_gain(train_fd_baseline_cat, train_fd_pred_cat).mean()
        train_fd_pred_epoch = train_fd_pred_cat.mean()
        train_nll_epoch = train_nll_sum / train_n

        train_nll, train_fdg, train_fd_pred = (
            train_nll_epoch.item(),
            train_fdg_epoch.item(),
            train_fd_pred_epoch.item(),
        )
        train_loss = train_nll - beta * train_fdg

        model.eval()
        with torch.no_grad():
            val_fd_baselines = []
            val_fd_preds = []
            val_nll_sum = torch.zeros((), device=device)
            val_n = 0
            # Array to store predicted sigmas to reconstruct predicted sigma distribution
            pred_sigmas = []
            for _, x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred, y_var = model(x)
                bs = y.size(0)
                val_nll_sum += criterion(y_pred, y, y_var) * bs
                val_n += bs

                last_x = x[:, -1, :]
                val_fd_baselines.append(fd(last_x, y, mu_t, sigma_t))
                val_fd_preds.append(fd(y_pred, y, mu_t, sigma_t))
                pred_sigmas.append(y_var.sqrt())

            pred_sigmas_cat = torch.cat(pred_sigmas, dim=0)
            fd_baseline_cat = torch.cat(val_fd_baselines, dim=0)
            fd_model_cat = torch.cat(val_fd_preds, dim=0)
            val_nll_epoch = val_nll_sum / val_n
            val_fd_pred_epoch = fd_model_cat.mean()
            val_fdg_epoch = fd_gain(fd_baseline_cat, fd_model_cat).mean()

            # single GPU→cpu per epoch
            val_nll, val_fd_pred, val_fdg, pred_sigmas = (
                val_nll_epoch.item(),
                val_fd_pred_epoch.item(),
                val_fdg_epoch.item(),
                pred_sigmas_cat.cpu(),
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
            checkpoint_dict = {
                # clone so later epochs don't mutate the stored best weights
                # (state_dict() returns references to the live parameters)
                "model_state_dict": {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "epochs": epochs,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
                "mu": mu,
                "sigma": sigma,
                "beta": beta,
                "train_task": train_task,
                "test_task": test_task,
                "pred_sigma": pred_sigmas,
            }
        else:
            early_counter += 1
            if early_counter >= patience:
                break

    torch.save(checkpoint_dict, checkpoint_path)
    return train_loss_history, val_loss_history


def split_data(train_task, test_task, split_percentages, batch_size, cross_patients=False, device="cpu"):
    """
    given train_task and test_task strings like "R+M" or "L", load the corresponding datasets,
    split patients according to split_percentages, and return train/val/test loaders.
    If cross_patients is True, ensure that train/val/test sets contain disjoint patients;
    if False, allow them to overlap (but still split val/test). If there's an overlap in train_task and test_task,
    automatically set cross_patients to True to avoid data leakage.
    """

    def parse_task(task_string):
        if "+" in task_string:
            return task_string.split("+")
        else:
            return [task_string]

    train_tasks = parse_task(train_task)
    test_tasks = parse_task(test_task)
    if set(train_tasks) & set(test_tasks):
        # if there's an overlap in tasks, we must do cross-patient to avoid leakage
        print("Overlap in train and test tasks detected, enabling cross-patient splitting to avoid data leakage.")
        cross_patients = True

    base_dir = "datasets"
    task_dicts = {
        "R": np.load(f"{base_dir}/R_dict.npy", allow_pickle=True).item(),
        "M": np.load(f"{base_dir}/M_dict.npy", allow_pickle=True).item(),
        "L": np.load(f"{base_dir}/L_dict.npy", allow_pickle=True).item(),
    }

    train_dicts = {task: task_dicts[task] for task in train_tasks}
    val_dicts = {task: task_dicts[task] for task in test_tasks}  # val and test use the same tasks
    test_dicts = {task: task_dicts[task] for task in test_tasks}

    patient_ids = np.array(sorted(task_dicts["R"].keys()))
    rng = np.random.default_rng(42)

    if cross_patients:
        # We must have patients set A for training, patients set B for validation and patients set C for testing
        assert sum(split_percentages) == 1.0, "Split percentages must sum to 1.0"

        train_percent, val_percent, test_percent = split_percentages
        train_ids = rng.choice(patient_ids, size=int(train_percent * len(patient_ids)), replace=False)
        test_val_ids = np.setdiff1d(patient_ids, train_ids)
        test_ids = rng.choice(test_val_ids, size=int(test_percent * len(patient_ids)), replace=False)
        val_ids = np.setdiff1d(test_val_ids, test_ids)
    else:
        assert sum(split_percentages[1:]) == 1.0, (
            "Split percentages for val and test must sum to 1.0 when cross_patients is False"
        )
        # If train and task do not intersect and cross_patients = False,
        # we can train on all the patients, while valid and test must still be splitted
        train_ids = patient_ids
        test_ids = rng.choice(patient_ids, size=int(split_percentages[2] * len(patient_ids)), replace=False)
        val_ids = np.setdiff1d(patient_ids, test_ids)

    def stack(task_dict, ids):
        # [N, T_task, 6] — T is constant within a task but differs across tasks
        return np.stack([task_dict[pid] for pid in ids], axis=0)

    print("Train patients: ", len(train_ids), "Val patients: ", len(val_ids), "Test patients: ", len(test_ids))
    print("R_shape = ", task_dicts["R"][train_ids[0]].shape[0])
    print("M_shape = ", task_dicts["M"][train_ids[0]].shape[0])
    print("L_shape = ", task_dicts["L"][train_ids[0]].shape[0])

    # e.g. {"train": {"R": [N_tr, T_R, 6], "M": [N_tr, T_M, 6]}, "val": {"M": .., "L": ..}, "test": {"M":.., "L":..}}
    splits = {}
    for split_name, split_ids, task_dict in zip(
        ["train", "val", "test"], [train_ids, val_ids, test_ids], [train_dicts, val_dicts, test_dicts]
    ):
        splits[split_name] = {task: stack(task_dict[task], split_ids) for task in task_dict}

    # To compute the total mean, collapse [N_tr, T_task, 6] -> [N_tr * T_task, 6] and concatenate tasks
    # -> [N_tr*T_R + N_tr*T_M + N_tr*T_L, 6] -> mean -> [6]
    train_frames = np.concatenate([arr.reshape(-1, 6) for arr in splits["train"].values()], axis=0)

    mu = train_frames.mean(axis=0)  # shape [6]
    sigma = train_frames.std(axis=0)  # shape [6]
    for split in splits.values():
        for task in split:
            split[task] = (split[task] - mu) / sigma

    # set True to keep the whole dataset on GPU and use the faster
    # GPUBatchLoader; set False for the classic CPU-dataset + DataLoader path.
    use_gpu_loader = True

    dataset_device = device if use_gpu_loader else "cpu"

    train_datasets = [TimeSeriesDataset(splits["train"][task], train_ids, device=dataset_device) for task in splits["train"]]
    val_datasets = [TimeSeriesDataset(splits["val"][task], val_ids, device=dataset_device) for task in splits["val"]]
    test_datasets = [TimeSeriesDataset(splits["test"][task], test_ids, device=dataset_device) for task in splits["test"]]

    train_loaders = [GPUBatchLoader(ds, batch_size=batch_size, shuffle=True) for ds in train_datasets]
    val_loaders = [GPUBatchLoader(ds, batch_size=batch_size, shuffle=False) for ds in val_datasets]
    test_loaders = [GPUBatchLoader(ds, batch_size=batch_size, shuffle=False) for ds in test_datasets]

    train_loader = MultiTaskLoader(train_loaders)
    val_loader = MultiTaskLoader(val_loaders, shuffle=False)
    test_loader = MultiTaskLoader(test_loaders, shuffle=False)

    return train_loader, val_loader, test_loader, mu, sigma, train_ids, val_ids, test_ids


if __name__ == "__main__":
    """
    =======================================================================================
    Training configuration
    =======================================================================================
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = {
        "input_dim": 6,
        "hidden_dim": 128,
        "output_dim": 6,
        "num_layers": 2,
        "dropout": 0.5,
    }

    dataset_config = {
        "train_task": "R+M+L",
        "test_task": "L",
        "batch_size": 16384,
        # when cross_patients = False, train set is automatically the whole dataset,
        # only the val and test percentages matter and must sum to 1.0;
        "split_percentages": (0.7, 0.15, 0.15),
        "cross_patients": False,
        "device": device,
    }

    train_loader, val_loader, test_loader, mu, sigma, train_ids, val_ids, test_ids = split_data(**dataset_config)
    model = GRUModel(**model_config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.GaussianNLLLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_config = {
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
        "patience": 100,
        "beta": 0.5,
        "epochs": 150,
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "device": device,
        "mu": mu,
        "sigma": sigma,
        "train_task": dataset_config["train_task"],
        "test_task": dataset_config["test_task"],
    }

    RUN_TAG = (
        f"GRU_{dataset_config['train_task']}v{dataset_config['test_task']}_beta{train_config['beta']}_ep{train_config['epochs']}"
    )
    train_config["checkpoint_path"] = f"{checkpoint_dir}/{RUN_TAG}.pth"

    train_loss_history, val_loss_history = train(
        **train_config,
    )

    # train and val loss history plots:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Train and Val Loss - GRU train {dataset_config['train_task']} test {dataset_config['test_task']} beta {train_config['beta']}"
    )
    plt.legend()
    plt.grid()
    plt.savefig(f"{checkpoint_dir}/{RUN_TAG}_loss_history.png")
    plt.show()
