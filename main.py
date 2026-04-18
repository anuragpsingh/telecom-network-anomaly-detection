# main.py — End-to-end orchestrator: generate -> preprocess -> train -> evaluate

import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# -- Make local imports work regardless of working directory --
sys.path.insert(0, os.path.dirname(__file__))

from config import Config, get_device
from data_generator import generate_dataset
from preprocessor import TelemetryPreprocessor
from models import build_model
from trainer import Trainer
from detector import AnomalyDetector
from evaluate import full_evaluation, align_labels


def main():
    cfg = Config()
    device = get_device()
    print(f"Device: {device}")

    # -- Step 1: Generate / load data ------------------------
    os.makedirs("data", exist_ok=True)

    def _data_columns_valid():
        if not os.path.exists(cfg.DATA_PATH):
            return False
        try:
            cols = set(pd.read_csv(cfg.DATA_PATH, nrows=0).columns)
            return all(c in cols for c in cfg.KPI_COLUMNS)
        except Exception:
            return False

    if not _data_columns_valid():
        print("\n[1/5] Generating synthetic telemetry data...")
        generate_dataset(save=True)
        # Remove stale checkpoints — they were trained on a different feature set
        for stale in ["checkpoints/best_model.pt", "checkpoints/scaler.pkl",
                      "checkpoints/threshold.npy", "checkpoints/error_stats.npy"]:
            if os.path.exists(stale):
                os.remove(stale)
                print(f"  Removed stale checkpoint: {stale}")
    else:
        print(f"\n[1/5] Data valid ({len(cfg.KPI_COLUMNS)} features): {cfg.DATA_PATH}")

    # -- Step 2: Preprocess -----------------------------------
    print("\n[2/5] Preprocessing...")
    prep = TelemetryPreprocessor(cfg)
    train_loader, val_loader, test_scaled, df_test, split_info = prep.make_loaders(cfg.DATA_PATH, cfg.ANOMALY_PATH)
    prep.save_scaler("checkpoints/scaler.pkl")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Test  rows    : {len(test_scaled)}")

    # -- Step 3: Build & train model --------------------------
    print(f"\n[3/5] Building {cfg.MODEL_TYPE.upper()} model...")
    n_features = len(cfg.KPI_COLUMNS)
    model = build_model(cfg, input_dim=n_features)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    trainer = Trainer(model, cfg, device=device)
    train_losses, val_losses = trainer.fit(train_loader, val_loader)

    # Plot training curves
    os.makedirs(cfg.RESULTS_PATH, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses,   label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{cfg.RESULTS_PATH}/training_curve.png", dpi=130)
    plt.close()

    # Reload best checkpoint
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))
    model.eval()

    # -- Step 4: Compute threshold ----------------------------
    print("\n[4/5] Computing anomaly threshold from training data...")
    threshold, mean_err, std_err = trainer.compute_threshold(train_loader)

    os.makedirs("checkpoints", exist_ok=True)
    np.save("checkpoints/threshold.npy", threshold)
    np.save("checkpoints/error_stats.npy", [mean_err, std_err])

    # -- Step 5: Evaluate on test set -------------------------
    print("\n[5/5] Evaluating on test set...")

    detector = AnomalyDetector(model, threshold, cfg, device=device)
    scores   = detector.score_array(test_scaled)

    labels_df = pd.read_csv(cfg.ANOMALY_PATH, parse_dates=["timestamp"])
    test_start = split_info["val_end"]

    scores_v, labels_v, types_v, timestamps_v = align_labels(
        labels_df, scores, test_start, cfg.WINDOW_SIZE
    )

    metrics = full_evaluation(
        scores_v, labels_v, types_v, timestamps_v,
        df_test.reset_index(drop=True), threshold, cfg
    )

    print(f"\nAll artefacts saved in '{cfg.RESULTS_PATH}' and 'checkpoints/'")
    print("Run  python pipeline.py  to start the live simulation.")


if __name__ == "__main__":
    main()
