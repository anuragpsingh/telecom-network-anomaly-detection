# pipeline.py — Near real-time 5-min feed simulation pipeline

import time
import pandas as pd
import numpy as np
import torch
import os
import sys
from config import Config, get_device
from preprocessor import TelemetryPreprocessor
from detector import AnomalyDetector
from models import build_model


class TelemetryPipeline:
    """
    Simulates a near real-time telemetry ingestion pipeline.

    In production this would connect to:
      - Kafka / Pub-Sub topics per telemetry source
      - Network Element Manager (NEM) REST APIs
      - OSS/BSS data feeds

    Here we replay the test CSV at configurable speed.
    """

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.preprocessor = TelemetryPreprocessor(self.cfg)
        self.detector: AnomalyDetector = None
        self.model = None

    # -- Initialise from saved artifacts ---------------------

    def load(self, model_path: str = None, scaler_path: str = "checkpoints/scaler.pkl"):
        model_path = model_path or self.cfg.MODEL_PATH
        n_features = len(self.cfg.KPI_COLUMNS)

        self.model = build_model(self.cfg, input_dim=n_features)
        self.model.load_state_dict(torch.load(model_path, map_location=get_device()))
        self.model.eval()

        self.preprocessor.load_scaler(scaler_path)

        # Threshold must be stored — load from file
        threshold_path = "checkpoints/threshold.npy"
        if os.path.exists(threshold_path):
            threshold = float(np.load(threshold_path))
        else:
            raise FileNotFoundError(
                "checkpoints/threshold.npy not found. Run main.py first to train."
            )

        self.detector = AnomalyDetector(
            self.model, threshold, self.cfg,
            device=get_device()
        )
        print(f"Pipeline ready  | threshold={threshold:.6f}")
        return self

    # -- Live simulation --------------------------------------

    def run_simulation(self, telemetry_csv: str, label_csv: str = None,
                       max_ticks: int = 500):
        """
        Replay telemetry CSV tick by tick, printing anomaly alerts.
        max_ticks : how many 5-min ticks to replay (default 500 ≈ ~41 hours)
        """
        df      = pd.read_csv(telemetry_csv, parse_dates=["timestamp"])
        labels  = None
        if label_csv and os.path.exists(label_csv):
            labels = pd.read_csv(label_csv, parse_dates=["timestamp"])

        # Use only test portion (last 15%)
        n        = len(df)
        test_start = int(n * (self.cfg.TRAIN_RATIO + self.cfg.VAL_RATIO))
        df_test  = df.iloc[test_start:].reset_index(drop=True)
        if labels is not None:
            labels_test = labels.iloc[test_start:].reset_index(drop=True)

        ticks    = min(max_ticks, len(df_test))
        interval = (self.cfg.FEED_INTERVAL_MIN * 60) / self.cfg.SIMULATE_SPEED_FACTOR

        print(f"\nStarting live simulation | {ticks} ticks | "
              f"interval={interval:.2f}s (×{self.cfg.SIMULATE_SPEED_FACTOR} speed)")
        print("-" * 65)

        alert_count = 0

        for i in range(ticks):
            row   = df_test.iloc[i]
            ts    = row["timestamp"]
            kpis  = row[self.cfg.KPI_COLUMNS].values.reshape(1, -1)

            # Scale the incoming tick
            scaled = self.preprocessor.scaler.transform(
                pd.DataFrame([row[self.cfg.KPI_COLUMNS]], columns=self.cfg.KPI_COLUMNS)
            )[0]

            result = self.detector.ingest(scaled, timestamp=ts)

            if result["status"] == "buffering":
                sys.stdout.write(f"\r  Buffering... {result['buffered']}/{self.cfg.WINDOW_SIZE}")
                sys.stdout.flush()
            else:
                true_label = ""
                if labels is not None:
                    gt = labels_test.iloc[i]["is_anomaly"]
                    true_label = f" | GT={'ANOMALY' if gt else 'normal':8s}"

                if result["anomaly"]:
                    alert_count += 1
                    print(f"\n  [!]  ANOMALY  {ts}  score={result['score']:.5f}{true_label}")
                else:
                    sys.stdout.write(
                        f"\r  OK  {ts}  score={result['score']:.5f}{true_label}  alerts={alert_count}"
                    )
                    sys.stdout.flush()

            time.sleep(interval)

        print(f"\n\nSimulation complete. Total alerts: {alert_count}/{ticks} ticks.")
        return self.detector.alert_log


if __name__ == "__main__":
    pipe = TelemetryPipeline()
    pipe.load()
    pipe.run_simulation(
        telemetry_csv=Config.DATA_PATH,
        label_csv=Config.ANOMALY_PATH,
        max_ticks=300,
    )
