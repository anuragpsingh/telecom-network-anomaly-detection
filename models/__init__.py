from .transformer_autoencoder import TransformerAutoencoder
from .mlp_autoencoder import MLPAutoencoder
from .isolation_forest import IsolationForestScorer


def build_model(cfg, input_dim: int) -> TransformerAutoencoder:
    return TransformerAutoencoder(
        input_dim=input_dim,
        d_model=cfg.HIDDEN_DIM,
        nhead=cfg.NUM_HEADS,
        num_encoder_layers=cfg.NUM_LAYERS,
        num_decoder_layers=cfg.NUM_LAYERS,
        dim_feedforward=cfg.HIDDEN_DIM * 2,
        latent_dim=cfg.LATENT_DIM,
        seq_len=cfg.WINDOW_SIZE,
        dropout=cfg.DROPOUT,
    )


def build_mlp_model(cfg, input_dim: int) -> MLPAutoencoder:
    return MLPAutoencoder(
        input_dim=input_dim,
        seq_len=cfg.WINDOW_SIZE,
        latent_dim=cfg.LATENT_DIM,
        dropout=0.2,
    )
