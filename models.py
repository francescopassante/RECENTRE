import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import math


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





class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (fixed, not learnable)."""
 
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1): #d_model is the lenght of the embedding vector, max_len is the mx number of the temp positions for which PE in calculated (notice at each input of the trasformer the PE is added to the window), cioe io gli do una frase, lui trasforma ogni parola in un vettore e gli aggiunge un info sulla posizione e poi qualche componente del vettore la mette a zero

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
 
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model], ogni parola un vettore riga
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (1/10000)^(2i/d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model] per il batch
        self.register_buffer("pe", pe)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model]"""
        x = x + self.pe[:, : x.size(1)] # tagli max_len coerente con x e somma in base al numero della parola (posizione) e le componenti pari e dispari aggiungo un valore seno/coseno
        return self.dropout(x)
 
 
class TransformerModel(nn.Module):
    """
    Causal Transformer encoder that mirrors the prediction contract of
    GRUModel and TCNModel:
 
      - Takes a window of `seq_len` input frames  →  [B, T, input_dim]
      - Predicts a *residual* delta over the last input frame
      - Returns absolute position  =  last_frame + predicted_delta
      - Also returns per-feature variance (already exponentiated, never exp again)
 
    Architecture
    ------------
    input_proj  →  PositionalEncoding  →  N × TransformerEncoderLayer (causal mask)
    →  take last-timestep token  →  fc1 / bn_fc1 / relu / dropout
    →  fc_mean  (residual delta)
    →  fc_logvar (log-variance, then exponentiated)
 
    The causal mask ensures that position t can only attend to positions ≤ t,
    so the model is strictly non-look-ahead (same guarantee as TCN / GRU).
 
    Args
    ----
    input_dim   : number of input features D
    output_dim  : number of output features D (usually == input_dim)
    d_model     : internal transformer width  (default 128)
    nhead       : number of attention heads    (default 4, must divide d_model)
    num_layers  : stacked encoder layers       (default 2)
    dim_feedforward : FFN hidden size          (default 256)
    dropout     : dropout applied everywhere   (default 0.1)
    max_len     : maximum sequence length for positional encoding (default 512)
    """
 
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
 
        # Project raw features into the transformer dimension
        self.input_proj = nn.Linear(input_dim, d_model)
 
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # expect [B, T, d_model]
            norm_first=True,    # Pre-LN (more stable than Post-LN)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # avoids a PyTorch warning with causal mask
        )
 
        # ── Prediction head (names kept identical to GRU / TCN on purpose) ──
        self.fc1 = nn.Linear(d_model, d_model)
        self.bn_fc1 = nn.LayerNorm(d_model)
        self.fc_mean = nn.Linear(d_model, output_dim)
        self.fc_logvar = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropout)
 
    # ------------------------------------------------------------------
    # Helper: build a square causal (upper-triangular) mask
    # ------------------------------------------------------------------
    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Returns a [T, T] boolean mask where True = position is masked (future)."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
 
    def forward(self, x: torch.Tensor):
        """
        Args
        ----
        x : Tensor  [batch_size, sequence_length, input_dim]
            A window of `sequence_length` consecutive frames.
 
        Returns
        -------
        (mean, variance) : each Tensor  [batch_size, output_dim]
            mean     = x[:, -1, :] + residual_delta  (absolute predicted position)
            variance = exp(logvar)  — do NOT exponentiate again in callers
        """
        B, T, _ = x.shape
 
        # 1. Project + positional encoding
        h = self.input_proj(x)      # [B, T, d_model]
        h = self.pos_enc(h)         # [B, T, d_model]
 
        # 2. Causal transformer encoder
        causal_mask = self._causal_mask(T, x.device)   # [T, T]
        h = self.transformer_encoder(h, mask=causal_mask, is_causal=True)  # [B, T, d_model]
 
        # 3. Take only the last token (contains full causal context)
        h = h[:, -1, :]             # [B, d_model]
 
        # 4. Shared prediction head (identical to GRU / TCN)
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = self.relu(h)
        h = self.dp(h)
 
        y_mean   = self.fc_mean(h)      # [B, output_dim]  — residual delta
        y_logvar = self.fc_logvar(h)    # [B, output_dim]
 
        # Return absolute position + variance (already exponentiated)
        return (x[:, -1, :] + y_mean), y_logvar.exp()
 





MODELS = {
    "gru": GRUModel,
    "tcn": TCNModel,
    "transformer": TransformerModel,
}
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

        # clamp logvar before exp so the variance stays in a sane range; without
        # this the GaussianNLL variance term can blow up and the loss goes NaN
        y_logvar = y_logvar.clamp(-10.0, 10.0)

        return (x[:, -1, :] + y_mean), y_logvar.exp()


class TSMixerLayer(nn.Module):
    def __init__(self, sequence_length, input_dim, hidden_dim, dropout):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(sequence_length, sequence_length),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.BatchNorm2d(1)
        self.norm2 = nn.BatchNorm2d(1)

    def forward(self, x):
        # x shape [B, T, D]
        # 1) normalization
        x = x.unsqueeze(1)  # [B, 1, T, D]

        # 2) time mixing
        out = self.norm1(x)
        out = out.transpose(2, 3)  # [B, 1, D, T]
        out = self.time_mlp(out)  # [B, 1, D, T]

        out = out.transpose(2, 3)  # [B, 1, T, D]
        x = x + out  # residual connection

        # 3) channel mixing
        out = self.norm2(x)
        out = self.channel_mlp(out)

        x = x + out  # residual connection

        return x.squeeze(1)  # [B, T, D]


class TSMixer(nn.Module):
    def __init__(self, num_layers, sequence_length, input_dim, hidden_dim, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.mixer_layers = nn.ModuleList(
            [
                TSMixerLayer(sequence_length, input_dim, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.time_projection_mean = nn.Linear(sequence_length, 1)
        self.time_projection_logvar = nn.Linear(sequence_length, 1)

        nn.init.zeros_(self.time_projection_mean.weight)
        nn.init.zeros_(self.time_projection_mean.bias)

    def forward(self, x):
        # x shape [B, T, D]
        last_frame = x[:, -1, :]  # [B, D]
        # time and channel mixing layers
        for layer in self.mixer_layers:
            x = layer(x)

        # temporal projection to horizon frames (1 in our case)
        x = x.transpose(1, 2)  # [B, D, T]
        y_mean = self.time_projection_mean(x).squeeze(-1)  # [B, D]
        y_logvar = self.time_projection_logvar(x).squeeze(-1)  # [B, D]

        y_logvar = y_logvar.clamp(-10.0, 10.0)

        return last_frame + y_mean, y_logvar.exp()


# Add a new architecture by writing its class above and giving it a name here.
MODELS = {"gru": GRUModel, "tcn": TCNModel, "patchtst": PatchTST, "tsmixer": TSMixer}


def build_model(model_config):
    """Build a model from a yaml config dict like {"type": "gru", "hidden_dim": 128, ...}."""
    config = dict(model_config)
    kind = config.pop(
        "type"
    ).lower()  # pop returns the value and removes it from the dict
    return MODELS[kind](**config)
