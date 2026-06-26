import torch
import torch.nn as nn
import torch.nn.functional as F
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
        return (x[:, -1, :6] + y_mean), y_logvar.exp()


class CausalConv1d(nn.Module):
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
        return (x[:, -1, :6] + y_mean), y_logvar.exp()


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to the input embeddings."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, d_model] — add the first T positional vectors
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=512,
        causal=True,
    ):
        super().__init__()
        assert (
            d_model % nhead == 0
        ), f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.causal = causal

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN, more stable than Post-LN
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # avoids a PyTorch warning with the causal mask
        )

        # same head as the other architectures (names kept identical on purpose:
        # fine-tuning freezes fc_logvar and groups fc1/bn_fc1/fc_mean as the head)
        self.fc1 = nn.Linear(d_model, d_model)
        self.bn_fc1 = nn.LayerNorm(d_model)
        self.fc_mean = nn.Linear(d_model, output_dim)
        self.fc_logvar = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropout)

    def forward(self, x):
        # x shape [B, T, D]
        B, T, _ = x.shape
        h = self.input_proj(x)  # [B, T, d_model]
        h = self.pos_enc(h)

        # causal mask: True = masked (future); position t attends only to <= t.
        # bidirectional (causal=False) attends to all positions.
        if self.causal:
            # torch.trius returns the upper triangular part
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            h = self.transformer_encoder(h, mask=mask, is_causal=True)
        else:
            h = self.transformer_encoder(h)

        h = h[:, -1, :]  # last timestep -> [B, d_model]
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = self.relu(h)
        h = self.dp(h)

        y_mean = self.fc_mean(h)
        y_logvar = self.fc_logvar(h)

        # the mean head predicts a residual added to the last input frame;
        # variance is returned already exponentiated (do not exp again in callers)
        return (x[:, -1, :6] + y_mean), y_logvar.exp()


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

        # output one prediction per channel; keep only the 6 position channels
        # (extra channels exist when velocity features are appended to the input)
        return (x[:, -1, :6] + y_mean[:, :6]), y_logvar[:, :6].exp()


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
        last_frame = x[:, -1, :6]  # [B, 6] positions only (extra channels with velocity)
        # time and channel mixing layers
        for layer in self.mixer_layers:
            x = layer(x)

        # temporal projection to horizon frames (1 in our case)
        x = x.transpose(1, 2)  # [B, D, T]
        y_mean = self.time_projection_mean(x).squeeze(-1)  # [B, D]
        y_logvar = self.time_projection_logvar(x).squeeze(-1)  # [B, D]

        y_logvar = y_logvar.clamp(-10.0, 10.0)

        return last_frame + y_mean[:, :6], y_logvar[:, :6].exp()




class DLinear(nn.Module):
    def __init__(self, sequence_length, input_dim, output_dim, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1

        self.L = sequence_length
        self.C = input_dim
        self.pad = kernel_size // 2
        self.kernel_size = kernel_size

        self.trend_linears = nn.ModuleList([nn.Linear(self.L, 1) for _ in range(self.C)])
        self.season_linears = nn.ModuleList([nn.Linear(self.L, 1) for _ in range(self.C)])
        self.logvar_linears = nn.ModuleList([nn.Linear(self.L, 1) for _ in range(self.C)])

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.L and C == self.C

        x_t = x.permute(0, 2, 1)
        
        front = x_t[:, :, 0:1].repeat(1, 1, self.pad) #copy the first element
        end = x_t[:, :, -1:].repeat(1, 1, self.pad) #copy the last element
        x_padded = torch.cat([front, x_t, end], dim=-1)
        
        trend = F.avg_pool1d(x_padded, kernel_size=self.kernel_size, stride=1) # for each three elements takes the mean, with stride=1 the output has the same lenght and is the global (trend) movement
        season = x_t - trend # oscillations wrt trend, small scale 

        pred_trend = torch.zeros([B, self.C], device=x.device)
        pred_season = torch.zeros([B, self.C], device=x.device)
        pred_logvar = torch.zeros([B, self.C], device=x.device)

        for c in range(self.C):
            pred_trend[:, c] = self.trend_linears[c](trend[:, c, :]).squeeze(-1)
            pred_season[:, c] = self.season_linears[c](season[:, c, :]).squeeze(-1)
            pred_logvar[:, c] = self.logvar_linears[c](x_t[:, c, :]).squeeze(-1)

        mean_res = pred_trend + pred_season
        logvar = torch.clamp(pred_logvar, min=-10.0, max=10.0)

        # keep only the 6 position channels (extra channels exist with velocity input)
        return x[:, -1, :6] + mean_res[:, :6], logvar[:, :6].exp()





class NLinear(nn.Module):
    """Normalized-Linear (Zeng et al. 2022), channel-independent.

    Subtracts the last frame before the linear map, one Linear(L→1) per channel.
    """

    def __init__(self, sequence_length, input_dim, output_dim):
        super().__init__()
        self.L = sequence_length
        self.C = input_dim
        self.linears = nn.ModuleList([nn.Linear(self.L, 1) for _ in range(self.C)])
        self.logvar_linears = nn.ModuleList([nn.Linear(self.L, 1) for _ in range(self.C)])

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.L and C == self.C

        x_t = x.permute(0, 2, 1)       # [B, C, L]
        x_norm = x_t - x_t[:, :, -1:]  # subtract last frame per channel

        pred_mean = torch.zeros([B, self.C], device=x.device)
        pred_logvar = torch.zeros([B, self.C], device=x.device)

        for c in range(self.C):
            pred_mean[:, c] = self.linears[c](x_norm[:, c, :]).squeeze(-1)
            pred_logvar[:, c] = self.logvar_linears[c](x_t[:, c, :]).squeeze(-1)

        logvar = torch.clamp(pred_logvar, min=-10.0, max=10.0)
        # keep only the 6 position channels (extra channels exist with velocity input)
        return x[:, -1, :6] + pred_mean[:, :6], logvar[:, :6].exp()


# ---- Conformer: CNN-Transformer hybrid (Gulati et al., Interspeech 2020,
# arXiv 2005.08100). Each block is the "macaron" structure: half-weight
# feed-forward, multi-head self-attention, a convolution module, a second
# half-weight feed-forward, then LayerNorm. The conv module models local motion
# (spikes); attention models long-range context -- the natural fusion of this
# repo's TCN and Transformer. Length-agnostic (reads the last timestep), so it
# needs no sequence_length, like the GRU/TCN/Transformer.


class ConformerFeedForward(nn.Module):
    """Pre-LN feed-forward module; added to the residual with a 0.5 weight."""

    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, expansion * d_model),
            nn.SiLU(),  # Swish, as in the paper
            nn.Dropout(dropout),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    """Pointwise conv + GLU -> depthwise conv -> norm -> Swish -> pointwise conv.

    The depthwise conv uses symmetric ('same') padding. The whole input window is
    past history relative to the out-of-window target and only the last timestep
    is read out for the prediction, so this never leaks the target. GroupNorm
    (not BatchNorm as in the paper) keeps it batch-independent, matching the TCN.
    """

    def __init__(self, d_model, kernel_size=7, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "conv kernel must be odd for 'same' padding"
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)  # GLU halves back to d_model
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model,
        )
        self.gnorm = nn.GroupNorm(1, d_model)
        self.act = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, d_model]
        h = self.norm(x).transpose(1, 2)        # [B, d_model, T]
        h = F.glu(self.pointwise1(h), dim=1)    # [B, d_model, T]
        h = self.act(self.gnorm(self.depthwise(h)))
        h = self.dp(self.pointwise2(h))
        return h.transpose(1, 2)                # [B, T, d_model]


class ConformerAttention(nn.Module):
    """Pre-LN multi-head self-attention. Bidirectional: the input window is all
    past history relative to the (out-of-window) target, so no causal mask."""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        return self.dp(h)


class ConformerBlock(nn.Module):
    """One Conformer block (macaron FFN sandwich around attention + conv)."""

    def __init__(self, d_model, nhead, ff_expansion, conv_kernel, dropout):
        super().__init__()
        self.ff1 = ConformerFeedForward(d_model, ff_expansion, dropout)
        self.attn = ConformerAttention(d_model, nhead, dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ff2 = ConformerFeedForward(d_model, ff_expansion, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class ConformerModel(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        ff_expansion=4,
        conv_kernel=7,
        dropout=0.1,
        max_len=512,
    ):
        """CNN-Transformer hybrid. Same prediction contract as the other models:
        input [B, T, input_dim] -> (mean, variance); the mean head predicts a
        residual added to the last input frame, variance is returned exponentiated.
        """
        super().__init__()
        assert (
            d_model % nhead == 0
        ), f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(d_model, nhead, ff_expansion, conv_kernel, dropout)
                for _ in range(num_layers)
            ]
        )

        # same head as the other architectures (names kept identical on purpose:
        # fine-tuning freezes fc_logvar and groups fc1/bn_fc1/fc_mean as the head)
        self.fc1 = nn.Linear(d_model, d_model)
        self.bn_fc1 = nn.LayerNorm(d_model)
        self.fc_mean = nn.Linear(d_model, output_dim)
        self.fc_logvar = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [B, T, D]
        h = self.input_proj(x)  # [B, T, d_model]
        h = self.pos_enc(h)
        for block in self.blocks:
            h = block(h)

        h = h[:, -1, :]  # last timestep -> [B, d_model]
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = self.relu(h)
        h = self.dp(h)

        y_mean = self.fc_mean(h)
        y_logvar = self.fc_logvar(h)

        # the mean head predicts a residual added to the last input frame;
        # variance is returned already exponentiated (do not exp again in callers)
        return (x[:, -1, :6] + y_mean), y_logvar.exp()


# ---- Mamba: selective state-space model (Gu & Dao 2023, arXiv 2312.00752).
# A linear recurrence (diagonal SSM) whose parameters (Delta, B, C) are produced
# from the input -- "selective", so the state can keep long-range context yet
# focus on the predictive recent dynamics. A short depthwise causal conv inside
# each block captures local motion spikes (like the TCN/Conformer conv) before the
# scan. Recurrence's inductive bias (which wins here) + linear-time long context.
# Length-agnostic (reads the last timestep), so no sequence_length argument, like
# the GRU/TCN/Transformer/Conformer.
#
# The selective scan is a chunked parallel scan (see MambaBlock._scan): the time
# axis is cut into ~sqrt(L) chunks, each chunk is scanned in parallel across all
# chunks, then the chunk-end states are carried sequentially. This drops the
# sequential depth from L to ~2*sqrt(L) Python steps (e.g. 64 -> ~16) and is
# numerically exact vs. the naive timestep loop. For even more speed swap in the
# fused mamba-ssm CUDA kernel. The materialized [B, L, d_inner, d_state] tensors
# dominate memory -- scale batch_size (and optionally d_state) to the GPU.


class MambaBlock(nn.Module):
    """One Mamba mixer: input projection + gate, depthwise causal conv, then an
    input-dependent diagonal SSM scan, gated by SiLU(z) and projected back."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)

        # project to the SSM branch x and the gate branch z (concatenated)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        # depthwise causal conv over time (groups=d_inner -> one filter per channel)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1
        )
        # produce the input-dependent (selective) Delta, B, C from x
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)  # Delta = softplus(dt_proj(.))
        # diagonal state matrix A (negative), stored in log space; D is the skip term
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # [d_inner, d_state]
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    @staticmethod
    def _scan(a, b):
        """Chunked parallel scan of the diagonal recurrence h_t = a_t*h_{t-1}+b_t.

        a, b: [B, L, d_inner, d_state]; returns h with the same shape.
        Splits L into ~sqrt(L) chunks: each chunk is scanned in parallel across
        all chunks (the within-chunk loop runs once per chunk, not once per frame),
        then the chunk-end states are propagated sequentially and injected back.
        Sequential depth ~ chunk + n_chunks instead of L; exact (no division, so
        a_t -> 0 from strong selective forgetting is safe).
        """
        B, L, Di, Ds = a.shape
        chunk = max(1, int(round(L**0.5)))
        pad = (-L) % chunk  # pad time at the end so L is a multiple of chunk
        if pad:
            a = F.pad(a, (0, 0, 0, 0, 0, pad), value=1.0)  # a=1 -> identity decay
            b = F.pad(b, (0, 0, 0, 0, 0, pad), value=0.0)  # b=0 -> no input
        Lp = L + pad
        nc = Lp // chunk
        a = a.view(B, nc, chunk, Di, Ds)
        b = b.view(B, nc, chunk, Di, Ds)

        # 1) within-chunk from-zero scan, parallel across the nc chunks
        s = torch.zeros(B, nc, Di, Ds, device=a.device, dtype=a.dtype)
        hs = []
        for j in range(chunk):
            s = a[:, :, j] * s + b[:, :, j]  # [B, nc, Di, Ds]
            hs.append(s)
        hloc = torch.stack(hs, dim=2)  # [B, nc, chunk, Di, Ds]

        # 2) inclusive cumprod of a within each chunk -> carry-injection coeffs
        A_incl = torch.cumprod(a, dim=2)  # [B, nc, chunk, Di, Ds]

        # 3) propagate the chunk-end state across chunks (sequential over nc)
        A_chunk = A_incl[:, :, -1]  # [B, nc, Di, Ds] total decay per chunk
        hloc_end = hloc[:, :, -1]  # [B, nc, Di, Ds] chunk-end from-zero state
        s_ins, carry = [], torch.zeros(B, Di, Ds, device=a.device, dtype=a.dtype)
        for c in range(nc):
            s_ins.append(carry)  # state entering chunk c
            carry = A_chunk[:, c] * carry + hloc_end[:, c]
        s_in = torch.stack(s_ins, dim=1)  # [B, nc, Di, Ds]

        # 4) inject the incoming carry into every position of each chunk
        h = A_incl * s_in.unsqueeze(2) + hloc  # [B, nc, chunk, Di, Ds]
        return h.reshape(B, Lp, Di, Ds)[:, :L]

    def ssm(self, x):
        # x: [B, L, d_inner]
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        deltaBC = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]
        delta, Bm, Cm = torch.split(
            deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))  # [B, L, d_inner], > 0

        # zero-order-hold discretization of the diagonal SSM
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [B, L, d_inner, d_state]
        deltaBx = (
            delta.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)
        )  # [B, L, d_inner, d_state]

        # chunked parallel scan, then contract the state with C in one vectorized op
        h = self._scan(deltaA, deltaBx)  # [B, L, d_inner, d_state]
        y = torch.einsum("blds,bls->bld", h, Cm)  # y_t = <h_t, C_t>
        return y + x * self.D  # skip connection through D

    def forward(self, x):
        # x: [B, L, d_model]
        L = x.size(1)
        x_in, z = self.in_proj(x).chunk(2, dim=-1)  # each [B, L, d_inner]

        # depthwise causal conv: drop the right padding so step t sees only <= t
        x_in = self.conv1d(x_in.transpose(1, 2))[..., :L].transpose(1, 2)
        x_in = F.silu(x_in)

        y = self.ssm(x_in)  # [B, L, d_inner]
        y = y * F.silu(z)  # gated output
        return self.out_proj(y)  # [B, L, d_model]


class MambaModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=64,
        num_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        """Selective-SSM backbone with the repo's shared two-head MLP. Same
        prediction contract as the other models: input [B, T, input_dim] ->
        (mean, variance); the mean head predicts a residual added to the last input
        frame, variance is returned exponentiated.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # pre-norm residual blocks (norm -> mixer -> add)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(d_model),
                    "mixer": MambaBlock(d_model, d_state, d_conv, expand),
                }
            )
            for _ in range(num_layers)
        )
        self.norm_f = nn.LayerNorm(d_model)

        # same head as the other architectures (names kept identical on purpose:
        # fine-tuning freezes fc_logvar and groups fc1/bn_fc1/fc_mean as the head)
        self.fc1 = nn.Linear(d_model, d_model)
        self.bn_fc1 = nn.LayerNorm(d_model)
        self.fc_mean = nn.Linear(d_model, output_dim)
        self.fc_logvar = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [B, T, input_dim]
        h = self.input_proj(x)
        for blk in self.blocks:
            h = h + blk["mixer"](blk["norm"](h))  # pre-norm residual

        h = self.norm_f(h)
        h = h[:, -1, :]  # last timestep -> [B, d_model]
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = self.relu(h)
        h = self.dp(h)

        y_mean = self.fc_mean(h)
        # clamp logvar before exp for stability (same guard as patchtst/tsmixer)
        y_logvar = self.fc_logvar(h).clamp(-10.0, 10.0)

        # the mean head predicts a residual added to the last input frame;
        # variance is returned already exponentiated (do not exp again in callers)
        return (x[:, -1, :6] + y_mean), y_logvar.exp()


# Add a new architecture by writing its class above and giving it a name here.
MODELS = {
    "gru": GRUModel,
    "tcn": TCNModel,
    "transformer": TransformerModel,
    "patchtst": PatchTST,
    "tsmixer": TSMixer,
    "dlinear": DLinear,
    "nlinear": NLinear,
    "conformer": ConformerModel,
    "mamba": MambaModel,
}


def build_model(model_config):
    """Build a model from a yaml config dict like {"type": "gru", "hidden_dim": 128, ...}."""
    config = dict(model_config)
    kind = config.pop(
        "type"
    ).lower()  # pop returns the value and removes it from the dict
    return MODELS[kind](**config)
