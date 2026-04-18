# predict.py — Run inference on any telemetry CSV using the pre-trained model
#
# Usage:
#   python predict.py --input my_telemetry.csv
#   python predict.py --input my_telemetry.csv --output results.csv
#   python predict.py --input my_telemetry.csv --threshold 0.08
#
# Input CSV must contain the 37 KPI columns defined in config.py.
# A 'timestamp' column is optional but recommended.
#
# Output CSV columns:
#   timestamp | anomaly_score | is_anomaly
#
# Pre-trained artefacts loaded from checkpoints/ (included in the repo):
#   best_model.pt   — trained Transformer Autoencoder weights
#   scaler.pkl      — StandardScaler fitted on training data
#   threshold.npy   — anomaly threshold (mean + 5σ of training errors)

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import Config, get_device
from models import build_model
from detector import AnomalyDetector
import joblib


# ─────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────

def load_artefacts(cfg: Config, device: str):
    """Load model weights, scaler, and threshold from checkpoints/."""
    model_path     = cfg.MODEL_PATH
    scaler_path    = "checkpoints/scaler.pkl"
    threshold_path = "checkpoints/threshold.npy"

    for path in [model_path, scaler_path, threshold_path]:
        if not os.path.exists(path):
            print(f"[ERROR] Required artefact not found: {path}")
            print("        Run  python main.py  first to train the model.")
            sys.exit(1)

    model = build_model(cfg, input_dim=len(cfg.KPI_COLUMNS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scaler    = joblib.load(scaler_path)
    threshold = float(np.load(threshold_path))

    return model, scaler, threshold


# ─────────────────────────────────────────────────────────────
# Validate input CSV
# ─────────────────────────────────────────────────────────────

def validate_input(df: pd.DataFrame, cfg: Config):
    missing = [c for c in cfg.KPI_COLUMNS if c not in df.columns]
    if missing:
        print(f"[ERROR] Input CSV is missing {len(missing)} required KPI column(s):")
        for col in missing:
            print(f"        - {col}")
        print("\n  Required columns:")
        for col in cfg.KPI_COLUMNS:
            print(f"        {col}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def run_inference(df: pd.DataFrame, model, scaler, threshold: float,
                  cfg: Config, device: str) -> pd.DataFrame:
    """
    Scale the input, score every sliding window, return results DataFrame.
    The first (WINDOW_SIZE - 1) rows cannot form a complete window and get
    score = 0.0 / is_anomaly = False.
    """
    scaled = scaler.transform(df[cfg.KPI_COLUMNS].values)

    detector = AnomalyDetector(model, threshold, cfg, device=device)
    scores   = detector.score_array(scaled)          # shape (n,)
    flags    = (scores > threshold).astype(int)

    results = pd.DataFrame({
        "anomaly_score": np.round(scores, 6),
        "is_anomaly":    flags,
    })

    if "timestamp" in df.columns:
        results.insert(0, "timestamp", df["timestamp"].values)

    return results


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

def print_summary(results: pd.DataFrame, threshold: float, cfg: Config):
    n_total   = len(results)
    scoreable = results["anomaly_score"] > 0          # exclude buffering rows
    n_scored  = scoreable.sum()
    n_anomaly = results["is_anomaly"].sum()

    print("\n" + "=" * 55)
    print("  INFERENCE SUMMARY")
    print("=" * 55)
    print(f"  Total rows        : {n_total:,}")
    print(f"  Scoreable rows    : {n_scored:,}  "
          f"(first {cfg.WINDOW_SIZE - 1} rows need buffer warm-up)")
    print(f"  Anomaly threshold : {threshold:.6f}")
    print(f"  Anomalies flagged : {n_anomaly:,}  "
          f"({100 * n_anomaly / max(n_scored, 1):.2f}% of scored rows)")

    if n_anomaly > 0:
        top = (results[results["is_anomaly"] == 1]
               .nlargest(5, "anomaly_score")
               [["timestamp", "anomaly_score"] if "timestamp" in results.columns
                else ["anomaly_score"]])
        print(f"\n  Top anomalous rows (highest score):")
        print(top.to_string(index=False))
    print("=" * 55)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run anomaly detection inference on a telemetry CSV."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input telemetry CSV (must contain all 37 KPI columns)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save results CSV. Defaults to <input>_predictions.csv"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override the saved anomaly threshold (optional)."
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = Config()
    device = get_device()

    print(f"Device    : {device}")
    print(f"Input     : {args.input}")

    # -- Load artefacts ----------------------------------------
    model, scaler, threshold = load_artefacts(cfg, device)
    if args.threshold is not None:
        print(f"Threshold : {args.threshold:.6f}  (overridden, saved={threshold:.6f})")
        threshold = args.threshold
    else:
        print(f"Threshold : {threshold:.6f}  (from checkpoints/threshold.npy)")

    # -- Load & validate input ---------------------------------
    df = pd.read_csv(args.input)
    print(f"Rows      : {len(df):,}  |  Columns: {len(df.columns)}")
    validate_input(df, cfg)

    # -- Run inference -----------------------------------------
    print("\nRunning inference...")
    results = run_inference(df, model, scaler, threshold, cfg, device)

    # -- Save results ------------------------------------------
    out_path = args.output or args.input.replace(".csv", "_predictions.csv")
    results.to_csv(out_path, index=False)
    print(f"Results saved -> {out_path}")

    # -- Print summary -----------------------------------------
    print_summary(results, threshold, cfg)


if __name__ == "__main__":
    main()
