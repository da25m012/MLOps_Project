"""
LSTM Autoencoder model architecture.
Encoder compresses input windows; decoder reconstructs them.
High reconstruction error => anomaly.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encodes a multivariate time-series window into a latent vector."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, input_size)
        _, (hidden, _) = self.lstm(x)
        # Take last layer's hidden state
        return hidden[-1]  # (batch, hidden_size)


class Decoder(nn.Module):
    """Reconstructs the original sequence from the latent vector."""

    def __init__(self, hidden_size: int, output_size: int, seq_len: int, num_layers: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, latent: torch.Tensor):
        # Repeat latent across time steps
        repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_size)
        out, _ = self.lstm(repeated)
        return self.output_layer(out)  # (batch, seq_len, output_size)


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder.
    Reconstruction error (MSE per sample) is the anomaly score.
    """

    def __init__(
        self,
        input_size: int = 4,      # CPU, memory, req_rate, error_rate
        hidden_size: int = 64,
        seq_len: int = 60,         # 60 x 5min = 5 hours
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size, input_size, seq_len, num_layers, dropout)

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-sample mean squared error (anomaly score)."""
        reconstructed = self.forward(x)
        # Mean over seq_len and features; shape: (batch,)
        return ((x - reconstructed) ** 2).mean(dim=(1, 2))
