# trainer.py — Training loop with early stopping, checkpoint saving

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import Config, get_device


class Trainer:
    def __init__(self, model: nn.Module, cfg: Config = None,
                 device: str = None, model_path: str = None):
        self.cfg        = cfg or Config()
        self.device     = device or get_device()
        self.model      = model.to(self.device)
        self.model_path = model_path or self.cfg.MODEL_PATH
        self.criterion  = nn.MSELoss()
        self.optimizer  = Adam(model.parameters(), lr=self.cfg.LEARNING_RATE)
        self.scheduler  = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.best_val_loss  = float("inf")
        self.patience_count = 0
        self.train_losses   = []
        self.val_losses     = []

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, target in loader:
                x      = x.to(self.device)
                target = target.to(self.device)
                recon  = self.model(x)
                loss   = self.criterion(recon, target)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                total_loss += loss.item() * x.size(0)
        return total_loss / len(loader.dataset)

    def fit(self, train_loader, val_loader):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model_name = os.path.basename(self.model_path).replace(".pt", "").upper()
        print(f"\nTraining on {self.device} | Model: {model_name}")
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Time':>8}")
        print("-" * 44)

        for epoch in range(1, self.cfg.EPOCHS + 1):
            t0 = time.time()
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss   = self._run_epoch(val_loader,   train=False)
            elapsed    = time.time() - t0

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            improved = val_loss < self.best_val_loss
            marker   = " (best)" if improved else ""
            print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {elapsed:>6.1f}s{marker}")

            if improved:
                self.best_val_loss  = val_loss
                self.patience_count = 0
                torch.save(self.model.state_dict(), self.model_path)
            else:
                self.patience_count += 1
                if self.patience_count >= self.cfg.PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch}.")
                    break

        print(f"\nBest val loss: {self.best_val_loss:.6f}  ->  {self.model_path}")
        return self.train_losses, self.val_losses

    def compute_threshold(self, train_loader) -> tuple:
        """
        Compute anomaly threshold from reconstruction errors on training data.
        threshold = mean + THRESHOLD_SIGMA × std
        """
        self.model.eval()
        errors = []
        with torch.no_grad():
            for x, _ in train_loader:
                x   = x.to(self.device)
                err = self.model.reconstruction_error(x)
                errors.extend(err.cpu().numpy().tolist())

        errors = np.array(errors)
        mean, std = errors.mean(), errors.std()
        threshold = mean + self.cfg.THRESHOLD_SIGMA * std
        print(f"\nReconstruction error — mean: {mean:.6f}  std: {std:.6f}")
        print(f"Anomaly threshold (mean + {self.cfg.THRESHOLD_SIGMA}*std): {threshold:.6f}")
        return threshold, mean, std
