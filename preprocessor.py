# preprocessor.py — Normalization, sliding window creation, DataLoader construction

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os
from config import Config


# -------------------------------------------------------------
# PyTorch Dataset — sliding window over KPI matrix
# -------------------------------------------------------------

class TelemetryWindowDataset(Dataset):
    """
    Returns windows of shape (window_size, n_features).
    For the autoencoder the target equals the input (reconstruction).
    """
    def __init__(self, data: np.ndarray, window_size: int, stride: int = 1):
        self.windows = []
        for i in range(0, len(data) - window_size + 1, stride):
            self.windows.append(data[i : i + window_size])
        self.windows = np.stack(self.windows, axis=0)   # (N, W, F)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        return x, x   # (input, target) — autoencoder


# -------------------------------------------------------------
# Main preprocessor
# -------------------------------------------------------------

class TelemetryPreprocessor:
    def __init__(self, cfg: Config = None):
        self.cfg    = cfg or Config()
        self.scaler = StandardScaler()

    # -- Fit & transform --------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit scaler on training data and return normalised array."""
        values = df[self.cfg.KPI_COLUMNS].values
        scaled = self.scaler.fit_transform(values)
        return scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply already-fitted scaler."""
        values = df[self.cfg.KPI_COLUMNS].values
        return self.scaler.transform(values)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(arr)

    # -- Build DataLoaders ------------------------------------

    def make_loaders(self, telemetry_path: str, labels_path: str = None):
        """
        Load CSV -> split -> normalise -> create DataLoaders.
        If labels_path is provided, anomalous rows are removed from the
        train and val splits so the autoencoder trains only on normal data.
        Returns: train_loader, val_loader, test_array, split_indices
        """
        df = pd.read_csv(telemetry_path, parse_dates=["timestamp"])
        n  = len(df)

        train_end = int(n * self.cfg.TRAIN_RATIO)
        val_end   = int(n * (self.cfg.TRAIN_RATIO + self.cfg.VAL_RATIO))

        df_train_raw = df.iloc[:train_end]
        df_val_raw   = df.iloc[train_end:val_end]
        df_test      = df.iloc[val_end:]

        # Filter anomalous rows from train/val so the autoencoder sees only normal data
        if labels_path and os.path.exists(labels_path):
            is_anomaly = pd.read_csv(labels_path)["is_anomaly"].values
            df_train = df_train_raw[is_anomaly[:train_end] == 0]
            df_val   = df_val_raw[is_anomaly[train_end:val_end] == 0]
            n_removed_train = len(df_train_raw) - len(df_train)
            n_removed_val   = len(df_val_raw)   - len(df_val)
            if n_removed_train + n_removed_val > 0:
                print(f"  Anomaly rows removed — train: {n_removed_train}, val: {n_removed_val}")
        else:
            df_train = df_train_raw
            df_val   = df_val_raw

        # Fit only on training data to avoid leakage
        train_scaled = self.fit_transform(df_train)
        val_scaled   = self.transform(df_val)
        test_scaled  = self.transform(df_test)

        train_ds = TelemetryWindowDataset(train_scaled, self.cfg.WINDOW_SIZE, self.cfg.STRIDE)
        val_ds   = TelemetryWindowDataset(val_scaled,   self.cfg.WINDOW_SIZE, self.cfg.STRIDE)

        train_loader = DataLoader(train_ds, batch_size=self.cfg.BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=self.cfg.BATCH_SIZE, shuffle=False, num_workers=0)

        split_info = {
            "train_end":  train_end,
            "val_end":    val_end,
            "timestamps": df["timestamp"].values,
        }
        return train_loader, val_loader, test_scaled, df_test, split_info

    # -- Persistence ------------------------------------------

    def save_scaler(self, path: str = "checkpoints/scaler.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"Scaler saved -> {path}")

    def load_scaler(self, path: str = "checkpoints/scaler.pkl"):
        self.scaler = joblib.load(path)
        print(f"Scaler loaded <- {path}")

    # -- Live feed helper -------------------------------------

    def preprocess_live_batch(self, rows: pd.DataFrame) -> np.ndarray:
        """
        Called by the pipeline every 5 min with a new row (or small batch).
        Assumes scaler is already fitted.
        """
        return self.transform(rows)
