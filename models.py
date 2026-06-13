import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


def get_device():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}")
    return device


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dp = nn.Dropout(p=dropout)
        # GRU layer
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.bn_gru = nn.LayerNorm(hidden_dim)

        ## Fully connected layer to map the hidden state to output
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
    # Part of the TCN model
    """Dilated causal 1-D convolution — pad on the left, crop the right so the
    output at time t never sees future frames."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # left padding the dilated kernel needs to keep the sequence length
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                dilation=dilation,
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
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
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
        super().__init__()
        num_channels = list(num_channels)
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            layers.append(
                TCNResidualBlock(
                    in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout
                )
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


class PatchTSTEncoderLayer(nn.Module):

    def __init__(self, d, num_heads, fc_hidden, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.BatchNorm1d(d)
        self.feed_forward = nn.Sequential(
            nn.Linear(d, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, d),
        )
        self.norm2 = nn.BatchNorm1d(d)

    def norm(self, norm, x):
        # BatchNorm1d normalizes the channel axis (dim 1); bring d there and back.
        # x: [B*D, num_patches, d] -> [B*D, d, num_patches] -> [B*D, num_patches, d]
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        # x shape [B*D, num_patches, d]
        v, _ = self.attention(x, x, x)
        x = self.norm(self.norm1, x + v)  # residual connection + batch norm
        ff = self.feed_forward(x)  # [B*D, num_patches, d]
        x = self.norm(self.norm2, x + ff)  # residual connection + batch norm
        return x


class PatchTST(nn.Module):

    def __init__(
        self,
        sequence_length,
        input_dim,
        patch_size,
        d,
        num_heads,
        fc_hidden,
        transformer_layers,
        dropout=0.1,
    ):
        super().__init__()
        # parameters
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        assert (
            sequence_length % self.patch_size == 0
        ), f"Sequence length must be divisible by patch size. Recieved sequence_length={self.sequence_length}, patch_size={self.patch_size}"
        self.num_patches = sequence_length // patch_size
        self.input_dim = input_dim
        self.d = d
        self.num_heads = num_heads
        self.fc_hidden = fc_hidden
        self.transformer_layers = transformer_layers

        # layers
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d))
        self.patch_proj = nn.Linear(patch_size, d)
        # one independent block per layer (no weight tying across layers)
        self.layers = nn.ModuleList(
            [
                PatchTSTEncoderLayer(d, num_heads, fc_hidden, dropout)
                for _ in range(transformer_layers)
            ]
        )

        # per-channel heads: flatten the patches of one channel -> scalar mean/logvar.
        # names kept identical to the other models so fine-tuning can freeze fc_logvar.
        self.fc_mean = nn.Linear(self.num_patches * d, 1)
        self.fc_logvar = nn.Linear(self.num_patches * d, 1)

    def to_patches(self, x):
        # x shape [B * D, T]
        # reshape to [B*D, num_patches, patch_size]
        patches = x.view(-1, self.num_patches, self.patch_size)
        return patches

    def patch_embedding(self, patches):
        # patches shape [B * D, num_patches, patch_size]
        # embed with linear projection and add learnable positional encoding
        emb_patches = self.patch_proj(patches) + self.pos_embedding

        # output shape [B * D, num_patches, d]
        return emb_patches

    def forward(self, x):
        # x shape [B, T, D]
        B, T, D = x.shape
        # transpose to [B, D, T] and flatten batch and dims
        x_fl = x.permute(0, 2, 1).reshape(B * D, T)  # [B*D, T]

        patches = self.to_patches(x_fl)  # [B*D, num_patches, patch_size]
        # embed to latent dimension d and add positional encoding
        out = self.patch_embedding(patches)  # [B*D, num_patches, d]

        for layer in self.layers:
            out = layer(out)  # [B*D, num_patches, d]

        # flatten patches and project to a per-channel prediction
        out = out.flatten(1, 2)  # [B*D, num_patches * d]
        # two fc heads for mean and logvar
        y_mean = self.fc_mean(out).view(B, D)  # [B, D]
        y_logvar = self.fc_logvar(out).view(B, D)  # [B, D]

        return (x[:, -1, :] + y_mean), y_logvar.exp()


# Add a new architecture by writing its class above and giving it a name here.
MODELS = {
    "gru": GRUModel,
    "tcn": TCNModel,
    "patchTST": PatchTST,
}


def build_model(model_config):
    """Build a model from a yaml config dict like {"type": "gru", "hidden_dim": 128, ...}."""
    config = dict(model_config)
    kind = config.pop("type")  # pop returns the value and removes it from the dict
    return MODELS[kind](**config)
