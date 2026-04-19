# models/isolation_forest.py — Isolation Forest wrapper with window-level scoring

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest


class IsolationForestScorer:
    """
    Wraps sklearn IsolationForest to produce per-window anomaly scores
    in the same [0, 1]-normalised style used by the autoencoder models.

    Training:
        fit(normal_rows)          — train on (n_samples, n_features) of normal data
        calibrate(normal_rows)    — store percentile bounds for normalisation

    Inference:
        score_window(window)      — returns normalised scalar score for a (W, F) window
        score_array(scaled_array) — sliding-window scores over a full test array
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.forest = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",   # unsupervised — no assumed contamination rate
            random_state=random_state,
            n_jobs=-1,
        )
        # Percentile bounds computed from training data for normalisation
        self._p1:  float = 0.0
        self._p99: float = 1.0

    # ── Training ────────────────────────────────────────────────

    def fit(self, normal_rows: np.ndarray) -> "IsolationForestScorer":
        """
        Train on (n_samples, n_features) array of normal-only rows.
        Also calibrates the normalisation bounds.
        """
        self.forest.fit(normal_rows)
        self.calibrate(normal_rows)
        return self

    def calibrate(self, normal_rows: np.ndarray):
        """
        Store 1st and 99th percentile of -decision_function() on normal training rows.
        At inference, true anomalies should exceed the 99th percentile.
        """
        raw = self._raw_scores(normal_rows)
        self._p1  = float(np.percentile(raw, 1))
        self._p99 = float(np.percentile(raw, 99))

    # ── Scoring ────────────────────────────────────────────────

    def _raw_scores(self, rows: np.ndarray) -> np.ndarray:
        """Return -decision_function() so higher value = more anomalous."""
        return -self.forest.decision_function(rows)

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        """Clip-normalize raw scores to [0, 1] using training percentile bounds."""
        return np.clip((raw - self._p1) / (self._p99 - self._p1 + 1e-9), 0.0, 1.0)

    def score_window(self, window: np.ndarray) -> float:
        """
        window : (seq_len, n_features) — already scaled
        Scores each timestep individually, returns the mean normalised score.
        (Mean is more robust than max against single-tick noise.)
        """
        raw  = self._raw_scores(window)          # (seq_len,)
        norm = self._normalize(raw)              # (seq_len,)
        return float(norm.mean())

    def score_array(self, scaled_array: np.ndarray, window_size: int) -> np.ndarray:
        """
        Score a full test array with a sliding window.
        First (window_size - 1) positions get score 0.
        """
        n      = len(scaled_array)
        scores = np.zeros(n)
        for i in range(window_size - 1, n):
            window    = scaled_array[i - window_size + 1 : i + 1]
            scores[i] = self.score_window(window)
        return scores

    # ── Persistence ────────────────────────────────────────────

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "IsolationForestScorer":
        return joblib.load(path)
