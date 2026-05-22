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

    def __init__(self, data, ids, sequence_length=10, TRANSFORM=False):
        """
        Args:
            data (numpy array): Shape [N, T, D]
            sequence_length (int): Length of sub-sequences
        """
        self.data = data
        self.time_span = sequence_length * 2
        self.N, self.T, self.D = data.shape
        self.ids = ids

    def __len__(self):
        return self.N * (self.T - self.time_span + 1)

    def __getitem__(self, index):
        p = index // (self.T - self.time_span + 1)  # Patient index
        t = index % (self.T - self.time_span + 1)  # Time index

        x = self.data[p, t : t + self.time_span : 2, :]  # Sub-sequence
        y = self.data[p, t + (self.time_span) - 1, :]  # Next time step
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return self.ids[p], x, y
