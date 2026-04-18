# detector.py — Sliding window anomaly scorer + alert generator

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from config import Config, get_device


class AnomalyDetector:
    """
    Wraps the trained autoencoder model.
    Maintains a rolling window buffer to score each new telemetry tick.
    """

    def __init__(self, model: nn.Module, threshold: float,
                 cfg: Config = None, device: str = None):
        self.model     = model
        self.threshold = threshold
        self.cfg       = cfg or Config()
        self.device    = device or get_device()
        self.model.eval()
        self.model.to(self.device)

        # Rolling buffer: holds last WINDOW_SIZE scaled feature vectors
        self.buffer: deque = deque(maxlen=cfg.WINDOW_SIZE if cfg else 24)
        self.alert_log: list = []

    # -- Core scoring -----------------------------------------

    def score_window(self, window: np.ndarray) -> float:
        """
        window : (WINDOW_SIZE, n_features) — already scaled
        Returns: scalar reconstruction error
        """
        x   = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        err = self.model.reconstruction_error(x)
        return err.item()

    def is_anomaly(self, score: float) -> bool:
        return score > self.threshold

    # -- Live tick interface ----------------------------------

    def ingest(self, scaled_row: np.ndarray, timestamp=None) -> dict:
        """
        Push one new 5-min tick into the buffer.
        Returns a result dict once the buffer is full.
        """
        self.buffer.append(scaled_row)

        if len(self.buffer) < self.cfg.WINDOW_SIZE:
            return {"status": "buffering", "buffered": len(self.buffer)}

        window = np.stack(list(self.buffer), axis=0)   # (W, F)
        score  = self.score_window(window)
        anomaly = self.is_anomaly(score)

        result = {
            "status":    "anomaly" if anomaly else "normal",
            "score":     round(score, 6),
            "threshold": round(self.threshold, 6),
            "timestamp": str(timestamp) if timestamp is not None else None,
            "anomaly":   anomaly,
        }

        if anomaly:
            self.alert_log.append(result)
            self._fire_alert(result)

        return result

    # -- Batch scoring (for evaluation) ----------------------

    def score_array(self, scaled_array: np.ndarray) -> np.ndarray:
        """
        Score an entire test array with a sliding window.
        Returns per-step scores (first WINDOW_SIZE-1 steps get score 0).
        """
        n = len(scaled_array)
        scores = np.zeros(n)
        for i in range(self.cfg.WINDOW_SIZE - 1, n):
            window = scaled_array[i - self.cfg.WINDOW_SIZE + 1 : i + 1]
            scores[i] = self.score_window(window)
        return scores

    # -- Alert (extensible hook) -------------------------------

    def _fire_alert(self, result: dict):
        """Override or extend to send to Slack, PagerDuty, Kafka, etc."""
        print(
            f"[ALERT] {result['timestamp']}  "
            f"score={result['score']:.4f} > threshold={result['threshold']:.4f}"
        )
