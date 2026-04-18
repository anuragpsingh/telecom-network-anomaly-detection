from .transformer_autoencoder import TransformerAutoencoder


def build_model(cfg, input_dim: int):
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
