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


def build_model(model_config):
    """Build a model from a yaml config dict like {"type": "gru", "hidden_dim": 128, ...}."""
    config = dict(model_config)
    kind = config.pop("type")  # pop returns the value and removes it from the dict
    return MODELS[kind](**config)
