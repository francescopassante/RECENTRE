import numpy as np
import torch
from torch.utils.data import Dataset


def parse_task(task_string):
    """Turn a task string like "R+M" into ["R", "M"], or "L" into ["L"]."""
    return task_string.split("+") if "+" in task_string else [task_string]


def build_features(pos, add_velocity=False, add_acceleration=False):
    """Concatenate velocity/acceleration channels onto raw position frames.

    `pos` is [N, T, 6]. Differences are taken with step 2 (x[t]-x[t-2]) so they
    only use frames the model actually sees, not the extra frames in between.
    Velocity is padded with 2 leading zeros and acceleration with 4, so every
    channel has length T. Returns [N, T, C] with C in {6, 12, 18}. The output is
    un-normalized: the caller z-scores every channel with a single per-channel
    mu/sigma (positions, velocity and acceleration are all treated the same way).
    """
    extra = []
    if add_velocity:
        # Compute vel such that v[0] = x[2] - x[0]
        vel = pos[:, 2:, :] - pos[:, :-2, :]  # [N, T-2, 6]
        # Pad 2 zeros, now v[0] = 0, v[1]= 0, v[2] = x[2] - x[0]
        vel = torch.cat([torch.zeros_like(vel[:, :2, :]), vel], dim=1)  # [N, T, 6]
        extra.append(vel)
    if add_acceleration:
        # second difference: a[t] = (x[t] - x[t-2]) - (x[t-2] - x[t-4])
        acc = pos[:, 4:, :] - 2 * pos[:, 2:-2, :] + pos[:, :-4, :]  # [N, T-4, 6]
        acc = torch.cat([torch.zeros_like(acc[:, :4, :]), acc], dim=1)  # [N, T, 6]
        extra.append(acc)
    return torch.cat([pos] + extra, dim=2)  # [N, T, 6/12/18]


class TimeSeriesDataset(Dataset):

    def __init__(
        self,
        data,
        ids,
        sequence_length=10,
        device="cpu",
        time_augmentation=False,
        neg_augmentation=False,
        add_velocity=False,
        add_acceleration=False,
        mu=None,
        sigma=None,
    ):
        # `data` is RAW positions [N, T, 6]. Augmentation happens on positions
        # first, then velocity/acceleration are derived from the (possibly
        # flipped/negated) positions, so the derived channels stay consistent.
        self.data = torch.from_numpy(data).to(device=device, dtype=torch.float32)
        if time_augmentation == True:
            data_rev = self.data.flip(1)
            self.data = torch.cat([self.data, data_rev], dim=0)
            self.ids = np.concatenate([ids, ids])
        else:
            self.ids = ids
        if neg_augmentation == True:
            data_neg = -self.data
            self.data = torch.cat([self.data, data_neg], dim=0)
            self.ids = np.concatenate([self.ids, self.ids])

        feats = build_features(self.data, add_velocity, add_acceleration)  # [N,T,C]

        # Single per-channel z-score for every channel. `mu`/`sigma` (length C)
        # normally come from split_data (computed once on the training set,
        # pooled across tasks). If not supplied, compute them from this dataset
        # and expose them (used by per-patient fine-tuning: the train split
        # computes them and propagates to its own val/test splits).
        if mu is None or sigma is None:
            mu = feats.mean(dim=(0, 1), keepdim=True)
            sigma = feats.std(dim=(0, 1), keepdim=True, correction=0) + 1e-6
        else:
            mu = torch.as_tensor(mu, dtype=torch.float32, device=feats.device)
            sigma = torch.as_tensor(sigma, dtype=torch.float32, device=feats.device)
        self.mu, self.sigma = mu, sigma
        self.data = (feats - mu) / sigma

        self.time_span = sequence_length * 2
        self.N, self.T, self.D = self.data.shape

    def __len__(self):
        return self.N * (self.T - self.time_span + 1)

    def __getitem__(self, index):
        p = index // (self.T - self.time_span + 1)  # Patient index
        t = index % (self.T - self.time_span + 1)  # Time index

        x = self.data[p, t : t + self.time_span : 2, :]  # Sub-sequence
        y = self.data[
            p, t + (self.time_span) - 1, :6
        ]  # Next time step (6 positions only)
        return self.ids[p], x, y


class GPUBatchLoader:
    """
    HCP dataset is ~2M samples -> 2M python calls. Since the whole dataset
    fits on GPU, we avoid python loops with a single vectorized call for the whole batch.
    """

    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.windows_per_patient = dataset.T - dataset.time_span + 1
        self.n_samples = dataset.N * self.windows_per_patient
        # offsets that pick out the sub-sequence frames: [0, 2, ..., time_span-2]
        self.x_offsets = torch.arange(
            0, dataset.time_span, 2, device=dataset.data.device
        )

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        ds = self.dataset
        device = ds.data.device
        if self.shuffle:
            order = torch.randperm(self.n_samples, device=device)
        else:
            order = torch.arange(self.n_samples, device=device)
        wpp = self.windows_per_patient
        for start in range(0, self.n_samples, self.batch_size):
            idx = order[start : start + self.batch_size]
            p = idx // wpp
            t = idx % wpp
            x = ds.data[p[:, None], t[:, None] + self.x_offsets[None, :], :]
            y = ds.data[p, t + ds.time_span - 1, :6]  # target = 6 positions only
            ids_b = [ds.ids[i] for i in p.tolist()]
            yield ids_b, x, y


class MultiTaskLoader:
    """Mixes batches from several per-task loaders into one epoch.
    Each loader handles its own shuffling and batching;
    this wrapper chooses which loader to use for the next batch.
    """

    def __init__(self, loaders, shuffle=True, seed=None):
        self.loaders = list(loaders)
        self.shuffle = shuffle
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(seed)

    def __len__(self):
        return sum(len(l) for l in self.loaders)

    def __iter__(self):
        # schedule is the list: [R, R, R, ..., R, M, M, ..., M, L, L, ..., L] with R as many times as there are batches in the resting data loader and so on.
        # essentially it is a schedule of which task to draw the next batch from.
        schedule = torch.cat(
            [
                torch.full((len(l),), i, dtype=torch.long)
                for i, l in enumerate(self.loaders)
            ]
        )

        # permute schedule so that each batch is from a random task.
        if self.shuffle:
            perm = torch.randperm(schedule.numel(), generator=self.gen)
            schedule = schedule[perm]
        iters = [iter(l) for l in self.loaders]
        for i in schedule.tolist():
            yield next(iters[i])


def split_data(
    train_task,
    test_task,
    split_percentages,
    batch_size,
    cross_patients=False,
    sequence_length=10,
    device="cpu",
    time_augmentation=False,
    neg_augmentation=False,
    add_velocity=False,
    add_acceleration=False,
    ids=None,
):
    """
    given train_task and test_task strings like "R+M" or "L", load the corresponding datasets,
    split patients according to split_percentages, and return train/val/test loaders.
    If cross_patients is True, ensure that train/val/test sets contain disjoint patients;
    if False, allow them to overlap (but still split val/test). If there's an overlap in train_task and test_task,
    automatically set cross_patients to True to avoid data leakage.
    Pass ids=(train_ids, val_ids, test_ids) to reuse an exact saved split (e.g. when
    resuming a checkpoint).
    """

    train_tasks = parse_task(train_task)
    test_tasks = parse_task(test_task)
    if set(train_tasks) & set(test_tasks):
        # if there's an overlap in tasks, we must do cross-patient to avoid leakage
        print(
            "Overlap in train and test tasks detected, enabling cross-patient splitting to avoid data leakage."
        )
        cross_patients = True

    base_dir = "datasets"
    task_dicts = {
        "R": np.load(f"{base_dir}/R_dict.npy", allow_pickle=True).item(),
        "M": np.load(f"{base_dir}/M_dict.npy", allow_pickle=True).item(),
        "L": np.load(f"{base_dir}/L_dict.npy", allow_pickle=True).item(),
    }

    train_dicts = {task: task_dicts[task] for task in train_tasks}
    val_dicts = {
        task: task_dicts[task] for task in test_tasks
    }  # val and test use the same tasks
    test_dicts = {task: task_dicts[task] for task in test_tasks}

    patient_ids = np.array(sorted(task_dicts["R"].keys()))

    # default seed
    rng = np.random.default_rng(42)

    if ids is not None:
        # reuse an exact saved split, bypassing the random choice
        train_ids, val_ids, test_ids = ids
    elif cross_patients:
        # We must have patients set A for training, patients set B for validation and patients set C for testing
        assert sum(split_percentages) == 1.0, "Split percentages must sum to 1.0"

        train_percent, val_percent, test_percent = split_percentages
        train_ids = rng.choice(
            patient_ids, size=int(train_percent * len(patient_ids)), replace=False
        )
        test_val_ids = np.setdiff1d(patient_ids, train_ids)
        test_ids = rng.choice(
            test_val_ids, size=int(test_percent * len(patient_ids)), replace=False
        )
        val_ids = np.setdiff1d(test_val_ids, test_ids)
    else:
        # If train and task do not intersect and cross_patients = False,
        # we can train on all the patients, while valid and test must still be splitted
        assert (
            sum(split_percentages[1:]) == 1.0
        ), "Split percentages for val and test must sum to 1.0 when cross_patients is False"

        train_ids = patient_ids
        test_ids = rng.choice(
            patient_ids,
            size=int(split_percentages[2] * len(patient_ids)),
            replace=False,
        )
        val_ids = np.setdiff1d(patient_ids, test_ids)

    def stack(task_dict, ids):
        # [N, T_task, 6] — T is constant within a task but differs across tasks
        return np.stack([task_dict[pid] for pid in ids], axis=0)

    print(
        "Train patients: ",
        len(train_ids),
        "Val patients: ",
        len(val_ids),
        "Test patients: ",
        len(test_ids),
    )

    # splits = e.g. {"train": {"R": [N_tr, T_R, 6], "M": [N_tr, T_M, 6]}, "val": {"M": .., "L": ..}, "test": {"M":.., "L":..}}
    splits = {}
    for split_name, split_ids, task_dict in zip(
        ["train", "val", "test"],
        [train_ids, val_ids, test_ids],
        [train_dicts, val_dicts, test_dicts],
    ):
        splits[split_name] = {
            task: stack(task_dict[task], split_ids) for task in task_dict
        }

    # Single per-channel mu/sigma over the training set, pooled across tasks and
    # shared by positions, velocity and acceleration alike (no separate feat_std).
    # Build the full feature tensor per task, collapse [N_tr, T_task, C] ->
    # [N_tr * T_task, C], concatenate tasks, then mean/std -> [C] (C in {6,12,18}).
    train_feat_frames = []
    for arr in splits["train"].values():
        pos = torch.from_numpy(arr).to(dtype=torch.float32)  # [N_tr, T_task, 6]
        f = build_features(pos, add_velocity, add_acceleration)  # [N_tr, T_task, C]
        train_feat_frames.append(f.reshape(-1, f.shape[-1]))
    all_feats = torch.cat(train_feat_frames, dim=0)
    mu = all_feats.mean(dim=0).numpy()  # shape [C]
    # correction=0 -> population std, matching the previous numpy .std() behaviour
    sigma = (all_feats.std(dim=0, correction=0) + 1e-6).numpy()  # shape [C]

    # keep the whole dataset on GPU so GPUBatchLoader can gather
    # each batch there without per-batch host->device transfers. Datasets receive
    # RAW positions and the shared mu/sigma, and normalize their features internally.
    train_datasets = [
        TimeSeriesDataset(
            splits["train"][task],
            train_ids,
            sequence_length,
            device,
            time_augmentation=time_augmentation,
            neg_augmentation=neg_augmentation,
            add_velocity=add_velocity,
            add_acceleration=add_acceleration,
            mu=mu,
            sigma=sigma,
        )
        for task in splits["train"]
    ]

    val_datasets = [
        TimeSeriesDataset(
            splits["val"][task],
            val_ids,
            sequence_length,
            device,
            time_augmentation=False,
            neg_augmentation=False,
            add_velocity=add_velocity,
            add_acceleration=add_acceleration,
            mu=mu,
            sigma=sigma,
        )
        for task in splits["val"]
    ]
    test_datasets = [
        TimeSeriesDataset(
            splits["test"][task],
            test_ids,
            sequence_length,
            device,
            time_augmentation=False,
            neg_augmentation=False,
            add_velocity=add_velocity,
            add_acceleration=add_acceleration,
            mu=mu,
            sigma=sigma,
        )
        for task in splits["test"]
    ]

    train_loaders = [
        GPUBatchLoader(ds, batch_size=batch_size, shuffle=True) for ds in train_datasets
    ]
    val_loaders = [
        GPUBatchLoader(ds, batch_size=batch_size, shuffle=False) for ds in val_datasets
    ]
    test_loaders = [
        GPUBatchLoader(ds, batch_size=batch_size, shuffle=False) for ds in test_datasets
    ]

    train_loader = MultiTaskLoader(train_loaders)
    val_loader = MultiTaskLoader(val_loaders, shuffle=False)
    test_loader = MultiTaskLoader(test_loaders, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        mu,
        sigma,
        train_ids,
        val_ids,
        test_ids,
    )
