# benchmark.py — Train & evaluate the Transformer autoencoder
#
# Usage:  python benchmark.py
#
# Saves checkpoint to checkpoints/transformer_model.pt
# Saves results    to results/benchmark_*.png and results/benchmark_comparison.csv

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(__file__))

from config import Config, get_device
from preprocessor import TelemetryPreprocessor
from models import build_model
from trainer import Trainer
from detector import AnomalyDetector
from evaluate import align_labels, threshold_metrics, per_type_report


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_training_curve(train_losses, val_losses, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color="steelblue", label="Train loss")
    ax.plot(epochs, val_losses,   color="darkorange", label="Val loss")
    ax.set_title("Transformer — Training Curve")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.tight_layout()
    path = f"{save_dir}/benchmark_training_curve.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved -> {path}")


def plot_scores(scores, labels, preds, threshold, save_dir):
    t = np.arange(len(scores))
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(t, scores, lw=0.7, color="steelblue", label="Recon error")
    ax.axhline(threshold, color="red", ls="--", lw=1.1,
               label=f"Threshold={threshold:.4f}")
    ax.fill_between(t, 0, scores, where=(labels == 1),
                    alpha=0.25, color="red", label="True anomaly")
    ax.fill_between(t, 0, scores, where=(preds == 1) & (labels == 0),
                    alpha=0.2, color="orange", label="False positive")
    ax.set_title("Anomaly Score Over Time — Transformer")
    ax.set_xlabel("Test step"); ax.set_ylabel("Recon MSE")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = f"{save_dir}/benchmark_scores.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved -> {path}")


def plot_pr_and_confusion(scores, labels, preds, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Transformer — Detection Performance", fontsize=12)

    prec, rec, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    axes[0].plot(rec, prec, color="steelblue", lw=1.5)
    axes[0].fill_between(rec, prec, alpha=0.2)
    axes[0].set_title(f"Precision-Recall  (AP={ap:.3f})")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")

    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"]).plot(
        ax=axes[1], cmap="Blues", colorbar=False
    )
    axes[1].set_title("Confusion Matrix")

    fig.tight_layout()
    path = f"{save_dir}/benchmark_pr_confusion.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved -> {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    cfg    = Config()
    device = get_device()
    ckpt   = "checkpoints/transformer_model.pt"
    cfg.MODEL_PATH = ckpt
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print(f"Device : {device}  |  Model: TRANSFORMER\n")

    # -- Data --------------------------------------------------
    preprocessor = TelemetryPreprocessor(cfg)
    train_loader, val_loader, test_scaled, df_test, split_info = \
        preprocessor.make_loaders(cfg.DATA_PATH, cfg.ANOMALY_PATH)

    labels_df   = pd.read_csv(cfg.ANOMALY_PATH)
    test_labels = labels_df.iloc[split_info["val_end"]:].reset_index(drop=True)
    valid_start = cfg.WINDOW_SIZE - 1
    labels_aligned = test_labels["is_anomaly"].values[valid_start:]
    types_aligned  = test_labels["anomaly_type"].values[valid_start:]

    # -- Train -------------------------------------------------
    model  = build_model(cfg, input_dim=len(cfg.KPI_COLUMNS))
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    trainer = Trainer(model, cfg, device=device)
    train_losses, val_losses = trainer.fit(train_loader, val_loader)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    threshold, mean, std = trainer.compute_threshold(train_loader)
    print(f"Error  mean={mean:.6f}  std={std:.6f}  threshold={threshold:.6f}")

    # -- Evaluate (reuse existing detector + evaluate helpers) -
    detector       = AnomalyDetector(model, threshold, cfg, device=device)
    scores         = detector.score_array(test_scaled)
    scores_aligned = scores[valid_start:]

    m = threshold_metrics(scores_aligned, labels_aligned, threshold)
    tp, fp = int(((m["preds"]==1)&(labels_aligned==1)).sum()), int(((m["preds"]==1)&(labels_aligned==0)).sum())
    tn, fn = int(((m["preds"]==0)&(labels_aligned==0)).sum()), int(((m["preds"]==0)&(labels_aligned==1)).sum())

    print(f"\n{'='*55}\n  TRANSFORMER EVALUATION\n{'='*55}")
    print(f"  Threshold       : {threshold:.6f}")
    print(f"  Accuracy        : {m['accuracy']:.4f}")
    print(f"  Precision       : {m['precision']:.4f}")
    print(f"  Recall          : {m['recall']:.4f}")
    print(f"  F1 Score        : {m['f1']:.4f}")
    print(f"  ROC-AUC         : {m['roc_auc']:.4f}")
    print(f"  Avg Precision   : {m['avg_precision']:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"\n  Per-Anomaly-Type Recall:")
    type_df = per_type_report(scores_aligned, labels_aligned, types_aligned, threshold)
    print(type_df.to_string(index=False))

    # -- Save summary ------------------------------------------
    summary = pd.DataFrame([{
        "accuracy": m["accuracy"], "precision": m["precision"],
        "recall": m["recall"], "f1": m["f1"],
        "roc_auc": m["roc_auc"], "avg_precision": m["avg_precision"],
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "threshold": round(threshold, 6),
    }])
    summary.to_csv("results/benchmark_comparison.csv", index=False)
    print("\nSaved -> results/benchmark_comparison.csv")

    # -- Plots -------------------------------------------------
    plot_training_curve(train_losses, val_losses, "results")
    plot_scores(scores_aligned, labels_aligned, m["preds"], threshold, "results")
    plot_pr_and_confusion(scores_aligned, labels_aligned, m["preds"], "results")


if __name__ == "__main__":
    main()
