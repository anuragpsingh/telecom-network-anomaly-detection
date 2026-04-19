# ensemble.py — Weighted ensemble detector combining Transformer AE, MLP AE, Isolation Forest
#
# Each model produces a raw anomaly score.  Scores are normalised to [0, 1]
# using percentile bounds computed on the training data (calibrate()).
# The final verdict is the weighted average against ENSEMBLE_THRESHOLD.

import os
import numpy as np
import torch
import joblib
from collections import deque
from config import Config, get_device
from noc_alert import NOCAlertClient


class EnsembleDetector:
    """
    Weighted ensemble of three anomaly detectors:
      - Transformer Autoencoder  (weight: ENSEMBLE_WEIGHTS["transformer"])
      - MLP Autoencoder          (weight: ENSEMBLE_WEIGHTS["mlp_autoencoder"])
      - Isolation Forest         (weight: ENSEMBLE_WEIGHTS["isolation_forest"])

    Calibration stores p1 / p99 of each model's training-error distribution
    so raw scores are mapped to a consistent [0, 1] scale before weighting.

    Usage:
        ensemble = EnsembleDetector(transformer, mlp_ae, if_scorer, cfg, device)
        ensemble.calibrate(train_loader, normal_rows)      # once after training
        ensemble.save_calibration("checkpoints/ensemble_params.pkl")

        result = ensemble.score_window(window_array)       # dict with all scores
        scores = ensemble.score_array(test_scaled)         # full test array
        result = ensemble.ingest(scaled_row, timestamp)    # live tick-by-tick
    """

    def __init__(self, transformer, mlp_ae, if_scorer,
                 cfg: Config = None, device: str = None):
        self.cfg       = cfg or Config()
        self.device    = device or get_device()
        self.weights   = self.cfg.ENSEMBLE_WEIGHTS
        self.threshold = self.cfg.ENSEMBLE_THRESHOLD

        self.transformer   = transformer.to(self.device).eval()
        self.mlp_ae        = mlp_ae.to(self.device).eval()
        self.if_scorer     = if_scorer

        # Normalisation bounds — populated by calibrate() or load_calibration()
        # {model_name: (p1, p99)}
        self._norm: dict = {}

        # Live tick buffer
        self.buffer: deque = deque(maxlen=self.cfg.WINDOW_SIZE)
        self.alert_log: list = []
        self.noc = NOCAlertClient(self.cfg)

    # ── Calibration ────────────────────────────────────────────

    def calibrate(self, train_loader, normal_rows: np.ndarray):
        """
        Compute per-model normalisation bounds from training data.
        normal_rows : (n_samples, n_features) — individual timestep vectors (scaled).
        """
        print("Calibrating ensemble normalisation bounds...")

        # AE models — collect per-window reconstruction errors
        for name, model in [("transformer", self.transformer),
                             ("mlp_autoencoder", self.mlp_ae)]:
            errors = []
            model.eval()
            with torch.no_grad():
                for x, _ in train_loader:
                    x   = x.to(self.device)
                    err = model.reconstruction_error(x)
                    errors.extend(err.cpu().numpy().tolist())
            errors = np.array(errors)
            self._norm[name] = (float(np.percentile(errors, 1)),
                                float(np.percentile(errors, 99)))
            print(f"  {name:20s}  p1={self._norm[name][0]:.6f}  "
                  f"p99={self._norm[name][1]:.6f}")

        # Isolation Forest — score individual rows then take window mean
        raw_if = -self.if_scorer.forest.decision_function(normal_rows)
        self._norm["isolation_forest"] = (float(np.percentile(raw_if, 1)),
                                          float(np.percentile(raw_if, 99)))
        print(f"  {'isolation_forest':20s}  p1={self._norm['isolation_forest'][0]:.6f}  "
              f"p99={self._norm['isolation_forest'][1]:.6f}")

    def save_calibration(self, path: str = "checkpoints/ensemble_params.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self._norm, path)
        print(f"Ensemble calibration saved -> {path}")

    def load_calibration(self, path: str = "checkpoints/ensemble_params.pkl"):
        self._norm = joblib.load(path)

    # ── Core scoring ───────────────────────────────────────────

    def _normalize(self, model_name: str, raw: float) -> float:
        p1, p99 = self._norm[model_name]
        return float(np.clip((raw - p1) / (p99 - p1 + 1e-9), 0.0, 1.0))

    def score_window(self, window: np.ndarray) -> dict:
        """
        Score a single window (seq_len, n_features) — already scaled.
        Returns a dict with raw scores, normalised scores, and ensemble result.
        """
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)

        raw = {
            "transformer":      self.transformer.reconstruction_error(x).item(),
            "mlp_autoencoder":  self.mlp_ae.reconstruction_error(x).item(),
            "isolation_forest": float(
                (-self.if_scorer.forest.decision_function(window)).mean()
            ),
        }

        norm = {k: self._normalize(k, v) for k, v in raw.items()}

        ensemble_score = sum(self.weights[k] * norm[k] for k in norm)
        is_anomaly     = ensemble_score > self.threshold

        return {
            "raw_scores":        raw,
            "normalized_scores": norm,
            "ensemble_score":    round(ensemble_score, 6),
            "is_anomaly":        is_anomaly,
        }

    # ── Batch scoring ──────────────────────────────────────────

    def score_array(self, scaled_array: np.ndarray) -> np.ndarray:
        """
        Score a full test array with a sliding window.
        Returns per-step ensemble scores (first WINDOW_SIZE-1 steps = 0).
        """
        n      = len(scaled_array)
        scores = np.zeros(n)
        W      = self.cfg.WINDOW_SIZE
        for i in range(W - 1, n):
            window    = scaled_array[i - W + 1 : i + 1]
            result    = self.score_window(window)
            scores[i] = result["ensemble_score"]
        return scores

    # ── Live tick-by-tick interface ────────────────────────────

    def ingest(self, scaled_row: np.ndarray, timestamp=None) -> dict:
        """
        Push one new telemetry tick into the rolling buffer.
        Once the buffer is full, scores the window, fires NOC alert if needed.
        """
        self.buffer.append(scaled_row)

        if len(self.buffer) < self.cfg.WINDOW_SIZE:
            return {"status": "buffering", "buffered": len(self.buffer)}

        window = np.stack(list(self.buffer), axis=0)
        result = self.score_window(window)
        result["timestamp"] = str(timestamp) if timestamp is not None else None
        result["status"]    = "anomaly" if result["is_anomaly"] else "normal"

        if result["is_anomaly"]:
            self.alert_log.append(result)
            self._fire_alert(result, timestamp)

        return result

    # ── Alert ──────────────────────────────────────────────────

    def _fire_alert(self, result: dict, timestamp):
        print(
            f"\n  [ENSEMBLE ALERT]  {timestamp}  "
            f"score={result['ensemble_score']:.4f} > threshold={self.threshold:.4f}"
            f"\n    transformer={result['normalized_scores']['transformer']:.3f}  "
            f"mlp_ae={result['normalized_scores']['mlp_autoencoder']:.3f}  "
            f"if={result['normalized_scores']['isolation_forest']:.3f}"
        )
        self.noc.send(
            timestamp       = timestamp,
            ensemble_score  = result["ensemble_score"],
            raw_scores      = result["raw_scores"],
            normalized_scores = result["normalized_scores"],
        )
