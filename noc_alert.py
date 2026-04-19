# noc_alert.py — NOC (Network Operations Centre) Alert API client
#
# Sends an HTTP POST to the configured NOC endpoint whenever the ensemble
# detector flags an anomaly.  Gracefully degrades if the endpoint is
# unreachable — anomaly detection continues uninterrupted.
#
# Configure via config.py or environment variables:
#   NOC_ALERT_ENDPOINT  — full URL, e.g. https://noc.example.com/api/v1/alerts
#   NOC_API_KEY         — Bearer token / API key
#   NOC_ALERTS_ENABLED  — set to False to suppress all HTTP calls (e.g. in testing)

import os
import json
import datetime
import urllib.request
import urllib.error
from config import Config


# ─────────────────────────────────────────────────────────────
# Severity classification
# ─────────────────────────────────────────────────────────────

def classify_severity(ensemble_score: float, thresholds: dict) -> str:
    """
    Map an ensemble score (0–1) to a severity label.
    thresholds keys: LOW, MEDIUM, HIGH  (values are the lower bound of that tier).
    """
    if ensemble_score >= thresholds.get("HIGH",   0.85):
        return "HIGH"
    if ensemble_score >= thresholds.get("MEDIUM", 0.70):
        return "MEDIUM"
    if ensemble_score >= thresholds.get("LOW",    0.50):
        return "LOW"
    return "INFO"


# ─────────────────────────────────────────────────────────────
# NOC Alert Client
# ─────────────────────────────────────────────────────────────

class NOCAlertClient:
    """
    Sends structured anomaly alerts to a NOC REST API.

    Alert payload schema:
    {
        "alert_id":         "uuid-style string",
        "timestamp":        "ISO-8601",
        "severity":         "HIGH | MEDIUM | LOW | INFO",
        "ensemble_score":   0.87,
        "model_scores": {
            "transformer":      {"raw": 0.21, "normalized": 0.91},
            "mlp_autoencoder":  {"raw": 0.18, "normalized": 0.85},
            "isolation_forest": {"raw": 0.74, "normalized": 0.74}
        },
        "ensemble_weights": {"transformer": 0.5, "mlp_autoencoder": 0.3, "isolation_forest": 0.2},
        "threshold":        0.50,
        "source":           "telecom-anomaly-detector"
    }
    """

    def __init__(self, cfg: Config = None):
        self.cfg      = cfg or Config()
        self.endpoint = os.environ.get("NOC_ALERT_ENDPOINT", self.cfg.NOC_ALERT_ENDPOINT)
        self.api_key  = os.environ.get("NOC_API_KEY",         self.cfg.NOC_API_KEY)
        self.enabled  = self.cfg.NOC_ALERTS_ENABLED
        self.timeout  = self.cfg.NOC_TIMEOUT_SEC
        self._alert_counter = 0

    # ── Public API ─────────────────────────────────────────────

    def send(self, timestamp, ensemble_score: float,
             raw_scores: dict, normalized_scores: dict) -> bool:
        """
        Build and POST an alert payload.
        Returns True if the alert was accepted (HTTP 2xx), False otherwise.
        Failures are logged but never raise — detection must keep running.
        """
        if not self.enabled:
            return True

        self._alert_counter += 1
        payload = self._build_payload(
            timestamp, ensemble_score, raw_scores, normalized_scores
        )

        for attempt in range(1, 3):          # 1 retry
            try:
                success = self._post(payload)
                if success:
                    return True
            except Exception as exc:
                print(f"  [NOC] Attempt {attempt} failed: {exc}")

        print(f"  [NOC] Alert #{self._alert_counter} could not be delivered to {self.endpoint}")
        return False

    # ── Internal helpers ───────────────────────────────────────

    def _build_payload(self, timestamp, ensemble_score: float,
                       raw_scores: dict, normalized_scores: dict) -> dict:
        severity = classify_severity(ensemble_score, self.cfg.NOC_SEVERITY_THRESHOLDS)

        model_scores = {
            name: {
                "raw":        round(float(raw_scores.get(name, 0)), 6),
                "normalized": round(float(normalized_scores.get(name, 0)), 4),
            }
            for name in ["transformer", "mlp_autoencoder", "isolation_forest"]
        }

        return {
            "alert_id":       f"TAD-{self._alert_counter:06d}",
            "timestamp":      str(timestamp) if timestamp else datetime.datetime.utcnow().isoformat(),
            "severity":       severity,
            "ensemble_score": round(float(ensemble_score), 4),
            "model_scores":   model_scores,
            "ensemble_weights": self.cfg.ENSEMBLE_WEIGHTS,
            "threshold":      self.cfg.ENSEMBLE_THRESHOLD,
            "source":         "telecom-anomaly-detector",
        }

    def _post(self, payload: dict) -> bool:
        body    = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Source":      "telecom-anomaly-detector",
        }
        req  = urllib.request.Request(self.endpoint, data=body,
                                      headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            ok = 200 <= resp.status < 300
            if ok:
                print(f"  [NOC] Alert delivered  status={resp.status}  "
                      f"severity={payload['severity']}  "
                      f"score={payload['ensemble_score']:.4f}")
            return ok
