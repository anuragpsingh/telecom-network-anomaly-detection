# models/transformer_autoencoder.py — Transformer Autoencoder for time-series anomaly detection

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerAutoencoder(nn.Module):
    """
    Transformer Autoencoder for multivariate time-series.

    Architecture:
      Input projection -> Positional Encoding
      -> Transformer Encoder (multi-head self-attention + FFN)
      -> Bottleneck (linear compress -> linear expand)
      -> Transformer Decoder (masked self-attention + cross-attention + FFN)
      -> Output projection -> reconstructed sequence

    Anomaly score = per-sample mean squared reconstruction error.
    """

    def __init__(
        self,
        input_dim: int,
        d_model:   int  = 64,
        nhead:     int  = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        latent_dim: int = 32,
        seq_len:   int  = 24,
        dropout:   float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.seq_len   = seq_len
        self.d_model   = d_model

        # -- Input / Output projections -----------------------
        self.input_proj  = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, input_dim)

        # -- Positional encoding -------------------------------
        self.pos_enc = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        # -- Encoder -------------------------------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # -- Bottleneck ----------------------------------------
        self.compress = nn.Linear(d_model, latent_dim)
        self.expand   = nn.Linear(latent_dim, d_model)

        # -- Decoder -------------------------------------------
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # Explicit dropout after encoder and decoder stacks to prevent memorisation
        self.enc_dropout = nn.Dropout(0.2)
        self.dec_dropout = nn.Dropout(0.2)

        # Learnable query tokens for decoder input (one per time step)
        self.decoder_queries = nn.Parameter(torch.randn(1, seq_len, d_model))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x     : (batch, seq_len, input_dim)
        return: (batch, seq_len, input_dim) — reconstructed
        """
        B = x.size(0)

        # Encode
        enc_in  = self.pos_enc(self.input_proj(x))          # (B, T, d_model)
        memory  = self.enc_dropout(self.encoder(enc_in))     # (B, T, d_model)

        # Bottleneck — compress to latent then back
        latent  = self.compress(memory)                      # (B, T, latent_dim)
        memory2 = self.expand(latent)                        # (B, T, d_model)

        # Decode using learnable query tokens
        queries = self.pos_enc(
            self.decoder_queries.expand(B, -1, -1)
        )                                                    # (B, T, d_model)
        dec_out = self.dec_dropout(self.decoder(queries, memory2))  # (B, T, d_model)

        recon   = self.output_proj(dec_out)                  # (B, T, input_dim)
        return recon

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE score. Shape: (batch,)"""
        with torch.no_grad():
            recon = self.forward(x)
            err   = ((x - recon) ** 2).mean(dim=(1, 2))
        return err
