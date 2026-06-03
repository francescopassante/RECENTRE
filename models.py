import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.1):
        """
        Args:
            input_dim (int): Number of input features (D)
            hidden_dim (int): Number of hidden units in GRU
            output_dim (int): Number of output features (D)
            num_layers (int): Number of GRU layers
            dropout (float): Dropout rate for regularization
        """
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dp = nn.Dropout(p=dropout)
        # GRU layer
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.bn_gru = nn.LayerNorm(hidden_dim)

        ## Fully connected layer to map the hidden state to outputù
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_fc1 = nn.LayerNorm(hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (tensor): Shape [batch_size, sequence_length, input_dim]

        Returns:
            y_pred (tensor): Shape [batch_size, output_dim]
        """
        # GRU forward pass
        _, h_n = self.gru(
            x
        )  # h_n is the final hidden state, shape [num_layers, batch_size, hidden_dim]

        # Use the last layer's hidden state for prediction

        h_n = h_n[-1]  # [batch_size, hidden_dim]

        h_n = self.bn_gru(h_n)

        h_n = self.relu(h_n)
        h_n = self.fc1(h_n)
        h_n = self.bn_fc1(h_n)
        h_n = self.relu(h_n)
        h_n = self.dp(h_n)

        # Fully connected layer for output prediction
        y_mean = self.fc_mean(h_n)  # Shape [batch_size, output_dim]
        y_logvar = self.fc_logvar(h_n)

        # the mean head predicts a residual added to the last input frame;
        # variance is returned already exponentiated (do not exp again in callers)
        return (x[:, -1, :] + y_mean), y_logvar.exp()


class CausalConv1d(nn.Module):
    """Dilated causal 1-D convolution — pad on the left, crop the right so the
    output at time t never sees future frames."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # left padding the dilated kernel needs to keep the sequence length
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=self.padding, dilation=dilation,
            )
        )

    def forward(self, x):
        # drop the trailing `padding` samples, which would peek into the future
        return self.conv(x)[:, :, : -self.padding]


class TCNResidualBlock(nn.Module):
    """Two causal convs + GroupNorm + GELU with a residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.act = nn.GELU()

    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))  # conv -> norm -> gelu
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))  # conv -> norm
        out = self.dropout(out)
        res = self.downsample(x) if self.downsample is not None else x
        return self.act(out + res)  # gelu after the residual sum


class TCNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_channels=(64, 128, 128, 64),
        kernel_size=2,
        dropout=0.2,
    ):
        """
        Args:
            input_dim (int): Number of input features (D)
            output_dim (int): Number of output features (D)
            num_channels (list[int]): channels of each residual block; dilation grows 2**i
            kernel_size (int): conv kernel size
            dropout (float): Dropout rate for regularization

        Same prediction contract as the GRU/Transformer: a causal TCN backbone
        feeds the shared two-head MLP (fc1 → LayerNorm → ReLU → Dropout → fc_mean /
        fc_logvar). The mean head predicts a residual added to the last input frame.
        """
        super(TCNModel, self).__init__()
        num_channels = list(num_channels)
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            layers.append(
                TCNResidualBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout)
            )
        self.tcn = nn.Sequential(*layers)

        # same head as the other architectures (names kept identical on purpose:
        # fine-tuning freezes fc_logvar and groups fc1/bn_fc1/fc_mean as the head)
        hidden = num_channels[-1]
        self.fc1 = nn.Linear(hidden, hidden)
        self.bn_fc1 = nn.LayerNorm(hidden)
        self.fc_mean = nn.Linear(hidden, output_dim)
        self.fc_logvar = nn.Linear(hidden, output_dim)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x (tensor): Shape [batch_size, sequence_length, input_dim]

        Returns:
            (mean, variance), each [batch_size, output_dim]
        """
        h = x.transpose(1, 2)  # [B, T, D] -> [B, D, T]
        h = self.tcn(h)
        h = h[:, :, -1]  # last timestep -> [B, hidden]

        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = self.relu(h)
        h = self.dp(h)

        y_mean = self.fc_mean(h)
        y_logvar = self.fc_logvar(h)

        # the mean head predicts a residual added to the last input frame;
        # variance is returned already exponentiated (do not exp again in callers)
        return (x[:, -1, :] + y_mean), y_logvar.exp()


# Add a new architecture by writing its class above and giving it a name here.
MODELS = {
    "gru": GRUModel,
    "tcn": TCNModel,
}


def build_model(model_config):
    """Build a model from a yaml config dict like {"type": "gru", "hidden_dim": 128, ...}."""
    config = dict(model_config)
    kind = config.pop("type")  # pop returns the value and removes it from the dict
    return MODELS[kind](**config)
