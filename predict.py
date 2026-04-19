# predict.py — Batch inference using the weighted ensemble detector
#
# Usage:
#   python predict.py --input my_telemetry.csv
#   python predict.py --input my_telemetry.csv --output results.csv
#   python predict.py --input my_telemetry.csv --threshold 0.6
#
# Input CSV must contain the 37 KPI columns defined in config.py.
# A 'timestamp' column is optional but recommended.
#
# Output CSV columns:
#   timestamp | ensemble_score | is_anomaly | transformer_score | mlp_score | if_score
#
# Pre-trained artefacts loaded from checkpoints/ (included in the repo):
#   best_model.pt          — Transformer Autoencoder weights
#   mlp_model.pt           — MLP Autoencoder weights
#   isolation_forest.pkl   — Isolation Forest model + calibration
#   ensemble_params.pkl    — per-model normalisation bounds
#   scaler.pkl             — StandardScaler fitted on training data

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import Config, get_device
from models import build_model, build_mlp_model, IsolationForestScorer
from ensemble import EnsembleDetector
import joblib


# ─────────────────────────────────────────────────────────────
# Load all artefacts
# ─────────────────────────────────────────────────────────────

def load_ensemble(cfg: Config, device: str) -> EnsembleDetector:
    required = {
        "Transformer weights":  cfg.MODEL_PATH,
        "MLP weights":          cfg.MLP_MODEL_PATH,
        "Isolation Forest":     cfg.IF_PATH,
        "Ensemble calibration": cfg.ENSEMBLE_PATH,
        "Scaler":               "checkpoints/scaler.pkl",
    }
    missing = [name for name, path in required.items() if not os.path.exists(path)]
    if missing:
        print("[ERROR] Missing pre-trained artefacts:")
        for m in missing:
            print(f"        - {m}")
        print("\n  Run  python main.py  first to train all models.")
        sys.exit(1)

    n_features = len(cfg.KPI_COLUMNS)

    transformer = build_model(cfg, input_dim=n_features)
    transformer.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))

    mlp_ae = build_mlp_model(cfg, input_dim=n_features)
    mlp_ae.load_state_dict(torch.load(cfg.MLP_MODEL_PATH, map_location=device))

    if_scorer = IsolationForestScorer.load(cfg.IF_PATH)

    detector = EnsembleDetector(transformer, mlp_ae, if_scorer, cfg, device)
    detector.load_calibration(cfg.ENSEMBLE_PATH)

    return detector


def load_scaler(path: str = "checkpoints/scaler.pkl"):
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────
# Validate input CSV
# ─────────────────────────────────────────────────────────────

def validate_input(df: pd.DataFrame, cfg: Config):
    missing = [c for c in cfg.KPI_COLUMNS if c not in df.columns]
    if missing:
        print(f"[ERROR] Input CSV is missing {len(missing)} required KPI column(s):")
        for col in missing:
            print(f"        - {col}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def run_inference(df: pd.DataFrame, detector: EnsembleDetector,
                  scaler, cfg: Config) -> pd.DataFrame:
    scaled = scaler.transform(df[cfg.KPI_COLUMNS].values)

    n      = len(scaled)
    W      = cfg.WINDOW_SIZE
    rows   = []

    for i in range(n):
        if i < W - 1:
            rows.append({
                "ensemble_score":    0.0,
                "is_anomaly":        0,
                "transformer_score": 0.0,
                "mlp_score":         0.0,
                "if_score":          0.0,
            })
            continue

        window = scaled[i - W + 1 : i + 1]
        result = detector.score_window(window)

        rows.append({
            "ensemble_score":    round(result["ensemble_score"], 6),
            "is_anomaly":        int(result["is_anomaly"]),
            "transformer_score": round(result["normalized_scores"]["transformer"], 4),
            "mlp_score":         round(result["normalized_scores"]["mlp_autoencoder"], 4),
            "if_score":          round(result["normalized_scores"]["isolation_forest"], 4),
        })

    results = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        results.insert(0, "timestamp", df["timestamp"].values)
    return results


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

def print_summary(results: pd.DataFrame, cfg: Config):
    n_total   = len(results)
    scored    = results["ensemble_score"] > 0
    n_scored  = scored.sum()
    n_anomaly = results["is_anomaly"].sum()

    print("\n" + "=" * 60)
    print("  ENSEMBLE INFERENCE SUMMARY")
    print("=" * 60)
    print(f"  Total rows          : {n_total:,}")
    print(f"  Scoreable rows      : {n_scored:,}  "
          f"(first {cfg.WINDOW_SIZE - 1} rows = buffer warm-up)")
    print(f"  Ensemble threshold  : {cfg.ENSEMBLE_THRESHOLD}")
    print(f"  Anomalies flagged   : {n_anomaly:,}  "
          f"({100 * n_anomaly / max(n_scored, 1):.2f}% of scored rows)")

    if n_anomaly > 0:
        cols = ["timestamp", "ensemble_score", "transformer_score",
                "mlp_score", "if_score"] if "timestamp" in results.columns \
               else ["ensemble_score", "transformer_score", "mlp_score", "if_score"]
        top = (results[results["is_anomaly"] == 1]
               .nlargest(5, "ensemble_score")[cols])
        print(f"\n  Top 5 anomalous rows (highest ensemble score):")
        print(top.to_string(index=False))
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ensemble anomaly detection inference on a telemetry CSV."
    )
    parser.add_argument("--input",     required=True,
                        help="Path to input telemetry CSV.")
    parser.add_argument("--output",    default=None,
                        help="Output CSV path. Defaults to <input>_predictions.csv")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override ensemble threshold (default: from config).")
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = Config()
    device = get_device()

    if args.threshold is not None:
        cfg.ENSEMBLE_THRESHOLD = args.threshold

    print(f"Device              : {device}")
    print(f"Input               : {args.input}")
    print(f"Ensemble threshold  : {cfg.ENSEMBLE_THRESHOLD}")
    print(f"Weights             : Transformer={cfg.ENSEMBLE_WEIGHTS['transformer']}  "
          f"MLP={cfg.ENSEMBLE_WEIGHTS['mlp_autoencoder']}  "
          f"IF={cfg.ENSEMBLE_WEIGHTS['isolation_forest']}")

    detector = load_ensemble(cfg, device)
    scaler   = load_scaler()

    df = pd.read_csv(args.input)
    print(f"\nRows: {len(df):,}  |  Columns: {len(df.columns)}")
    validate_input(df, cfg)

    print("Running ensemble inference...")
    results  = run_inference(df, detector, scaler, cfg)

    out_path = args.output or args.input.replace(".csv", "_predictions.csv")
    results.to_csv(out_path, index=False)
    print(f"Results saved -> {out_path}")

    print_summary(results, cfg)


if __name__ == "__main__":
    main()
