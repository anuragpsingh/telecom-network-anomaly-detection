# models/mlp_autoencoder.py — MLP (feedforward) Autoencoder for time-series windows

import torch
import torch.nn as nn


class MLPAutoencoder(nn.Module):
    """
    Simple feedforward autoencoder for multivariate time-series windows.

    The input window (seq_len, n_features) is flattened into a 1-D vector,
    compressed through a bottleneck, and reconstructed back to the original shape.

    Anomaly score = per-sample mean squared reconstruction error (same interface
    as TransformerAutoencoder so both can be trained with the same Trainer).
    """

    def __init__(
        self,
        input_dim: int,
        seq_len:   int,
        latent_dim: int  = 64,
        dropout:   float = 0.2,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.input_dim = input_dim
        flat_dim       = seq_len * input_dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, flat_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x     : (batch, seq_len, input_dim)
        return: (batch, seq_len, input_dim) — reconstructed
        """
        B    = x.size(0)
        flat = x.reshape(B, -1)                        # (B, seq_len * input_dim)
        z    = self.encoder(flat)                      # (B, latent_dim)
        out  = self.decoder(z)                         # (B, seq_len * input_dim)
        return out.view(B, self.seq_len, self.input_dim)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE score. Shape: (batch,)"""
        with torch.no_grad():
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=(1, 2))
