# main.py — End-to-end orchestrator:
#   generate → preprocess → train (Transformer AE + MLP AE + Isolation Forest)
#            → build weighted ensemble → evaluate → save artefacts

import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from config import Config, get_device
from data_generator import generate_dataset
from preprocessor import TelemetryPreprocessor
from models import build_model, build_mlp_model, IsolationForestScorer
from trainer import Trainer
from ensemble import EnsembleDetector
from evaluate import full_evaluation, align_labels


def main():
    cfg    = Config()
    device = get_device()
    print(f"Device: {device}")

    os.makedirs("data",        exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(cfg.RESULTS_PATH, exist_ok=True)

    # ── Step 1: Generate / load data ─────────────────────────
    def _data_columns_valid():
        if not os.path.exists(cfg.DATA_PATH):
            return False
        try:
            cols = set(pd.read_csv(cfg.DATA_PATH, nrows=0).columns)
            return all(c in cols for c in cfg.KPI_COLUMNS)
        except Exception:
            return False

    if not _data_columns_valid():
        print("\n[1/7] Generating synthetic telemetry data...")
        generate_dataset(save=True)
        for stale in [cfg.MODEL_PATH, cfg.MLP_MODEL_PATH, cfg.IF_PATH,
                      cfg.ENSEMBLE_PATH, "checkpoints/scaler.pkl",
                      "checkpoints/threshold.npy", "checkpoints/error_stats.npy"]:
            if os.path.exists(stale):
                os.remove(stale)
                print(f"  Removed stale artefact: {stale}")
    else:
        print(f"\n[1/7] Data valid ({len(cfg.KPI_COLUMNS)} features): {cfg.DATA_PATH}")

    # ── Step 2: Preprocess ───────────────────────────────────
    print("\n[2/7] Preprocessing...")
    prep = TelemetryPreprocessor(cfg)
    train_loader, val_loader, test_scaled, df_test, split_info = \
        prep.make_loaders(cfg.DATA_PATH, cfg.ANOMALY_PATH)
    prep.save_scaler("checkpoints/scaler.pkl")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Test  rows    : {len(test_scaled)}")

    # Raw training rows for Isolation Forest (individual timesteps, scaled)
    df_raw     = pd.read_csv(cfg.DATA_PATH)
    train_end  = split_info["train_end"]
    labels_all = pd.read_csv(cfg.ANOMALY_PATH)
    normal_mask = labels_all["is_anomaly"].values[:train_end] == 0
    train_rows  = prep.scaler.transform(
        df_raw.iloc[:train_end][cfg.KPI_COLUMNS].values[normal_mask]
    )

    n_features = len(cfg.KPI_COLUMNS)

    # ── Step 3: Train Transformer Autoencoder ────────────────
    print(f"\n[3/7] Training Transformer Autoencoder...")
    transformer = build_model(cfg, input_dim=n_features)
    print(f"  Parameters: {sum(p.numel() for p in transformer.parameters() if p.requires_grad):,}")
    t_trainer = Trainer(transformer, cfg, device=device, model_path=cfg.MODEL_PATH)
    t_trainer.fit(train_loader, val_loader)
    transformer.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))
    transformer.eval()
    t_threshold, t_mean, t_std = t_trainer.compute_threshold(train_loader)
    np.save("checkpoints/threshold.npy",    t_threshold)
    np.save("checkpoints/error_stats.npy",  [t_mean, t_std])

    _plot_training_curve(t_trainer.train_losses, t_trainer.val_losses,
                         "Transformer AE", cfg.RESULTS_PATH)

    # ── Step 4: Train MLP Autoencoder ───────────────────────
    print(f"\n[4/7] Training MLP Autoencoder...")
    mlp_ae = build_mlp_model(cfg, input_dim=n_features)
    print(f"  Parameters: {sum(p.numel() for p in mlp_ae.parameters() if p.requires_grad):,}")
    m_trainer = Trainer(mlp_ae, cfg, device=device, model_path=cfg.MLP_MODEL_PATH)
    m_trainer.fit(train_loader, val_loader)
    mlp_ae.load_state_dict(torch.load(cfg.MLP_MODEL_PATH, map_location=device))
    mlp_ae.eval()

    _plot_training_curve(m_trainer.train_losses, m_trainer.val_losses,
                         "MLP AE", cfg.RESULTS_PATH)

    # ── Step 5: Train Isolation Forest ──────────────────────
    print(f"\n[5/7] Training Isolation Forest on {len(train_rows):,} normal rows...")
    if_scorer = IsolationForestScorer(n_estimators=200, random_state=cfg.RANDOM_SEED)
    if_scorer.fit(train_rows)
    if_scorer.save(cfg.IF_PATH)
    print(f"  Saved -> {cfg.IF_PATH}")

    # ── Step 6: Build & calibrate Ensemble ──────────────────
    print(f"\n[6/7] Building weighted ensemble  "
          f"(Transformer×{cfg.ENSEMBLE_WEIGHTS['transformer']}  "
          f"MLP×{cfg.ENSEMBLE_WEIGHTS['mlp_autoencoder']}  "
          f"IF×{cfg.ENSEMBLE_WEIGHTS['isolation_forest']})")
    ensemble = EnsembleDetector(transformer, mlp_ae, if_scorer, cfg, device)
    ensemble.calibrate(train_loader, train_rows)
    ensemble.save_calibration(cfg.ENSEMBLE_PATH)

    # ── Step 7: Evaluate ensemble on test set ───────────────
    print(f"\n[7/7] Evaluating ensemble on test set...")

    ensemble_scores = ensemble.score_array(test_scaled)

    labels_df  = pd.read_csv(cfg.ANOMALY_PATH, parse_dates=["timestamp"])
    test_start = split_info["val_end"]

    scores_v, labels_v, types_v, timestamps_v = align_labels(
        labels_df, ensemble_scores, test_start, cfg.WINDOW_SIZE
    )

    metrics = full_evaluation(
        scores_v, labels_v, types_v, timestamps_v,
        df_test.reset_index(drop=True),
        threshold=cfg.ENSEMBLE_THRESHOLD,
        cfg=cfg,
    )

    print(f"\nAll artefacts saved in '{cfg.RESULTS_PATH}' and 'checkpoints/'")
    print("Run  python pipeline.py  to start the live simulation.")
    print("Run  python predict.py --input <file.csv>  for batch inference.")


# ── Helpers ────────────────────────────────────────────────────

def _plot_training_curve(train_losses, val_losses, title: str, save_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train loss", color="steelblue")
    ax.plot(val_losses,   label="Val loss",   color="darkorange")
    ax.set_title(f"Training Curve — {title}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(); fig.tight_layout()
    fname = title.lower().replace(" ", "_")
    path  = f"{save_dir}/training_curve_{fname}.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  Saved -> {path}")


if __name__ == "__main__":
    main()
