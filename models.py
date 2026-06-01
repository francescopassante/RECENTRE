import torch
import torch.nn as nn


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


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        output_dim,
        nhead=4,
        num_layers=3,
        dim_feedforward=192,
        dropout=0.1,
        max_len=64,
    ):
        """
        Args:
            input_dim (int): Number of input features (D)
            d_model (int): Transformer embedding width
            output_dim (int): Number of output features (D)
            nhead (int): Number of attention heads
            num_layers (int): Number of encoder layers
            dim_feedforward (int): Width of the per-layer feedforward block
            dropout (float): Dropout rate for regularization
            max_len (int): Longest sequence the learned positional table supports
        """
        super(TransformerModel, self).__init__()
        # Project the 6-DOF input frames up to the embedding width
        self.input_proj = nn.Linear(input_dim, d_model)
        # Learned positional embedding, sliced to the actual sequence length in forward
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        ## Fully connected layer to map the pooled state to output
        self.fc1 = nn.Linear(d_model, d_model)
        self.bn_fc1 = nn.LayerNorm(d_model)
        self.fc_mean = nn.Linear(d_model, output_dim)
        self.fc_logvar = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x (tensor): Shape [batch_size, sequence_length, input_dim]

        Returns:
            y_pred (tensor): Shape [batch_size, output_dim]
        """
        seq_len = x.size(1)
        # Embed inputs and add the learned positional codes
        h = self.input_proj(x) + self.pos_emb[:, :seq_len, :]

        # Encoder forward pass, then keep the last timestep for prediction
        h = self.encoder(h)  # [batch_size, sequence_length, d_model]
        h = h[:, -1, :]  # [batch_size, d_model]

        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = self.relu(h)
        h = self.dp(h)

        # Fully connected layer for output prediction
        y_mean = self.fc_mean(h)  # Shape [batch_size, output_dim]
        y_logvar = self.fc_logvar(h)

        # the mean head predicts a residual added to the last input frame;
        # variance is returned already exponentiated (do not exp again in callers)
        return (x[:, -1, :] + y_mean), y_logvar.exp()


# Add a new architecture by writing its class above and giving it a name here.
MODELS = {"gru": GRUModel, "transformer": TransformerModel}


def build_model(model_config):
    """Build a model from a yaml config dict like {"type": "gru", "hidden_dim": 128, ...}."""
    config = dict(model_config)
    kind = config.pop("type")  # pop returns the value and removes it from the dict
    return MODELS[kind](**config)
