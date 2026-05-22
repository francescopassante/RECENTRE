import torch
from torch.utils.data import Dataset

"""
==========================================================================================================================================
 BUG FOUND IN PREVIOUS VERSION: it was using (self.T - self.time_span) instead of (self.T - self.time_span + 1)
 Previous version also had a TRANSFORM parameter to inject random noise into the data. Removed it now, but it can be added back if needed.
==========================================================================================================================================
"""


class TimeSeriesDataset(Dataset):
    """A dataset for time series data."""

    def __init__(self, data, ids, sequence_length=10, device="cpu"):
        """
        Args:
            data (numpy array or tensor): Shape [N, T, D]
            sequence_length (int): Length of sub-sequences
            device: where to keep self.data. Pass 'cuda' to keep the whole
                dataset resident on GPU and skip per-batch H2D transfers.
        """
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)
        self.data = data.to(device=device, dtype=torch.float32)
        self.time_span = sequence_length * 2
        self.N, self.T, self.D = self.data.shape
        self.ids = ids

    def __len__(self):
        return self.N * (self.T - self.time_span + 1)

    def __getitem__(self, index):
        p = index // (self.T - self.time_span + 1)  # Patient index
        t = index % (self.T - self.time_span + 1)  # Time index

        x = self.data[p, t : t + self.time_span : 2, :]  # Sub-sequence
        y = self.data[p, t + (self.time_span) - 1, :]  # Next time step
        return self.ids[p], x, y


class GPUBatchLoader:
    """Drop-in DataLoader replacement for a GPU-resident TimeSeriesDataset.

    Builds each batch with one vectorized gather on the GPU instead of
    per-sample __getitem__ + collate. The standard DataLoader is fine when
    samples live on CPU and the worker copies them to GPU in bulk, but when
    every sample is already a tiny CUDA tensor the per-sample Python loop and
    torch.stack dominate runtime.
    """

    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.windows_per_patient = dataset.T - dataset.time_span + 1
        self.n_samples = dataset.N * self.windows_per_patient
        # offsets that pick out the sub-sequence frames: [0, 2, ..., time_span-2]
        self._x_offsets = torch.arange(
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
            x = ds.data[p[:, None], t[:, None] + self._x_offsets[None, :], :]
            y = ds.data[p, t + ds.time_span - 1, :]
            ids_b = [ds.ids[i] for i in p.tolist()]
            yield ids_b, x, y
