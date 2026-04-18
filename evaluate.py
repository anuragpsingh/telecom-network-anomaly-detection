# evaluate.py — Metrics, plots, and per-anomaly-type breakdown

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from config import Config


def align_labels(labels_df: pd.DataFrame, scores: np.ndarray,
                 test_start_idx: int, window_size: int) -> tuple:
    """
    Align ground-truth labels with per-step scores.
    The first (window_size-1) test steps have no score, so we trim them.
    """
    test_labels = labels_df.iloc[test_start_idx:].reset_index(drop=True)
    valid_start  = window_size - 1
    return (
        scores[valid_start:],
        test_labels["is_anomaly"].values[valid_start:],
        test_labels["anomaly_type"].values[valid_start:],
        test_labels["timestamp"].values[valid_start:],
    )


def threshold_metrics(scores: np.ndarray, labels: np.ndarray,
                      threshold: float) -> dict:
    preds  = (scores > threshold).astype(int)
    report = classification_report(labels, preds,
                                   target_names=["Normal", "Anomaly"],
                                   output_dict=True, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy":      report["accuracy"],
        "precision":     report["Anomaly"]["precision"],
        "recall":        report["Anomaly"]["recall"],
        "f1":            report["Anomaly"]["f1-score"],
        "roc_auc":       auc,
        "avg_precision": average_precision_score(labels, scores),
        "preds":         preds,
    }


def per_type_report(scores: np.ndarray, labels: np.ndarray,
                    types: np.ndarray, threshold: float) -> pd.DataFrame:
    preds        = (scores > threshold).astype(int)
    unique_types = [t for t in np.unique(types) if t != "normal"]
    rows = []
    for atype in unique_types:
        mask = types == atype
        if not mask.any():
            continue
        tp     = int(((preds[mask] == 1) & (labels[mask] == 1)).sum())
        fn     = int(((preds[mask] == 0) & (labels[mask] == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append({"anomaly_type": atype, "n_steps": int(mask.sum()),
                     "detected": tp, "missed": fn,
                     "recall": round(recall, 3)})
    return pd.DataFrame(rows)


def plot_results(scores: np.ndarray, labels: np.ndarray, timestamps,
                 threshold: float, kpi_df: pd.DataFrame, preds: np.ndarray,
                 cfg: Config, save_dir: str = "results"):

    os.makedirs(save_dir, exist_ok=True)
    n = len(scores)
    t = np.arange(n)

    # ── Figure 1: Anomaly score dashboard ─────────────────────
    n_kpis   = len(cfg.KPI_COLUMNS)
    n_cols   = 4
    n_rows   = math.ceil(n_kpis / n_cols)
    fig_rows = 1 + n_rows          # row 0 = score, rows 1..n_rows = KPIs

    fig = plt.figure(figsize=(22, 3.5 * fig_rows))
    gs  = gridspec.GridSpec(fig_rows, n_cols, figure=fig,
                            hspace=0.55, wspace=0.35)
    fig.suptitle("Multi-Layer Anomaly Detection Dashboard", fontsize=14,
                 fontweight="bold", y=1.01)

    # Row 0: anomaly score (full width)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, scores, lw=0.8, color="steelblue", label="Reconstruction error")
    ax0.axhline(threshold, color="red", ls="--", lw=1.2,
                label=f"Threshold = {threshold:.4f}")
    ax0.fill_between(t, 0, scores, where=(labels == 1),
                     alpha=0.3, color="red", label="True anomaly")
    ax0.fill_between(t, 0, scores,
                     where=(preds == 1) & (labels == 0),
                     alpha=0.2, color="orange", label="False positive")
    ax0.set_title("Anomaly Score Over Time", fontsize=10)
    ax0.set_ylabel("Reconstruction MSE", fontsize=8)
    ax0.legend(loc="upper right", fontsize=7)

    # Pre-extract all KPI arrays once to avoid repeated column lookups
    kpi_values = {col: kpi_df[col].values[:n] if col in kpi_df.columns else np.zeros(n)
                  for col in cfg.KPI_COLUMNS}

    # Rows 1..: one subplot per KPI
    for i, col in enumerate(cfg.KPI_COLUMNS):
        row_idx = (i // n_cols) + 1
        col_idx = i % n_cols
        ax = fig.add_subplot(gs[row_idx, col_idx])

        label, color, layer = cfg.KPI_META.get(col, (col, "gray", ""))
        vals = kpi_values[col]
        ax.plot(t, vals, lw=0.6, color=color)
        ax.fill_between(t, vals.min(), vals,
                        where=(labels == 1), alpha=0.2, color="red")
        ax.set_title(f"[{layer}] {label}", fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
        ax.set_ylabel(label, fontsize=6)

    fig.savefig(f"{save_dir}/anomaly_dashboard.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Layer-grouped KPI overview ──────────────────
    layers = {}
    for col in cfg.KPI_COLUMNS:
        _, _, layer = cfg.KPI_META.get(col, (col, "gray", "Other"))
        layers.setdefault(layer, []).append(col)

    n_layers = len(layers)
    fig2, axes2 = plt.subplots(n_layers, 1,
                               figsize=(20, 3.5 * n_layers), sharex=True)
    if n_layers == 1:
        axes2 = [axes2]
    fig2.suptitle("KPI Overview by Observability Layer", fontsize=12,
                  fontweight="bold")

    for ax, (layer_name, cols) in zip(axes2, layers.items()):
        for col in cols:
            label, color, _ = cfg.KPI_META.get(col, (col, "gray", ""))
            vals = kpi_values[col]
            # Normalise to [0,1] for overlay comparison
            vmin, vmax = vals.min(), vals.max()
            norm = (vals - vmin) / (vmax - vmin + 1e-9)
            ax.plot(t, norm, lw=0.7, color=color, alpha=0.8, label=label)
        ax.fill_between(t, 0, 1, where=(labels == 1),
                        alpha=0.15, color="red", label="Anomaly")
        ax.set_title(f"{layer_name} Layer (normalised)", fontsize=9)
        ax.legend(loc="upper right", fontsize=6, ncol=4)
        ax.set_ylim(-0.05, 1.1)

    axes2[-1].set_xlabel("Test step")
    fig2.tight_layout()
    fig2.savefig(f"{save_dir}/layer_overview.png", dpi=130)
    plt.close(fig2)

    # ── Figure 3: PR curve + Confusion Matrix ─────────────────
    if labels.sum() > 0:
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
        fig3.suptitle("Detection Performance", fontsize=12)

        prec, rec, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        axes3[0].plot(rec, prec, color="darkorange")
        axes3[0].fill_between(rec, prec, alpha=0.2)
        axes3[0].set_title(f"Precision-Recall  (AP={ap:.3f})")
        axes3[0].set_xlabel("Recall"); axes3[0].set_ylabel("Precision")

        cm = confusion_matrix(labels, preds)
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"]).plot(
            ax=axes3[1], cmap="Blues", colorbar=False
        )
        axes3[1].set_title("Confusion Matrix")

        fig3.tight_layout()
        fig3.savefig(f"{save_dir}/pr_confusion.png", dpi=150)
        plt.close(fig3)

    print(f"Plots saved -> {save_dir}/")


def full_evaluation(scores, labels, types, timestamps, kpi_df, threshold, cfg):
    print("\n" + "=" * 60)
    print("  ANOMALY DETECTION EVALUATION REPORT")
    print("=" * 60)

    m = threshold_metrics(scores, labels, threshold)
    print(f"  Accuracy        : {m['accuracy']:.4f}")
    print(f"  Precision       : {m['precision']:.4f}")
    print(f"  Recall          : {m['recall']:.4f}")
    print(f"  F1 Score        : {m['f1']:.4f}")
    print(f"  ROC-AUC         : {m['roc_auc']:.4f}")
    print(f"  Avg Precision   : {m['avg_precision']:.4f}")

    print("\n  Per-Anomaly-Type Recall:")
    type_df = per_type_report(scores, labels, types, threshold)
    print(type_df.to_string(index=False))

    os.makedirs(cfg.RESULTS_PATH, exist_ok=True)
    type_df.to_csv(f"{cfg.RESULTS_PATH}/per_type_recall.csv", index=False)

    plot_results(scores, labels, timestamps, threshold, kpi_df,
                 m["preds"], cfg, save_dir=cfg.RESULTS_PATH)
    print("=" * 60)
    return m
