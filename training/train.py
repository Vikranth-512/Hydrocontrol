"""
LSTM policy training pipeline with early stopping and experiment logging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.lstm_policy import LSTMPolicy
from preprocessing.normalization import FeatureNormalizer
from preprocessing.sequence_builder import SequenceDataset
from training.evaluation import compute_prediction_metrics
from training.losses import ControlAwareLoss, mse_loss


class Trainer:
    """End-to-end training with checkpointing and metrics logging."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ) -> None:
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMPolicy] = None
        self.normalizer: Optional[FeatureNormalizer] = None
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "val_mae": [],
        }

    def build_model(self, input_size: int) -> LSTMPolicy:
        mcfg = self.config.get("model", {})
        model = LSTMPolicy(
            input_size=input_size,
            hidden_size=mcfg.get("hidden_size", 128),
            num_layers=mcfg.get("num_layers", 2),
            dropout=mcfg.get("dropout", 0.2),
            output_size=mcfg.get("output_size", 2),
        )
        self.model = model.to(self.device)
        return self.model

    def _get_loss_fn(self):
        tcfg = self.config.get("training", {})
        if tcfg.get("loss_type", "control_aware") == "mse":
            return lambda p, t, prev=None: mse_loss(p, t)
        return ControlAwareLoss(weights=tcfg.get("control_loss_weights"))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        normalizer: FeatureNormalizer,
        checkpoint_dir: Path,
    ) -> LSTMPolicy:
        assert self.model is not None
        tcfg = self.config.get("training", {})
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_ds = SequenceDataset(X_train, y_train)
        val_ds = SequenceDataset(X_val, y_val)
        train_loader = DataLoader(
            train_ds, batch_size=tcfg.get("batch_size", 64), shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=tcfg.get("batch_size", 64))

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=tcfg.get("learning_rate", 1e-3),
            weight_decay=tcfg.get("weight_decay", 1e-5),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=tcfg.get("lr_factor", 0.5),
            patience=tcfg.get("lr_patience", 5),
        )

        loss_fn = self._get_loss_fn()
        best_val = float("inf")
        patience_counter = 0
        patience = tcfg.get("early_stopping_patience", 12)
        grad_clip = tcfg.get("grad_clip", 1.0)
        epochs = tcfg.get("epochs", 80)

        prev_batch_pred = None

        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                if isinstance(loss_fn, ControlAwareLoss):
                    loss = loss_fn(pred, yb, prev_batch_pred)
                else:
                    loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()
                train_losses.append(loss.item())
                prev_batch_pred = pred.detach()

            val_loss, val_metrics = self._validate(val_loader, loss_fn, normalizer)
            scheduler.step(val_loss)

            self.history["train_loss"].append(float(np.mean(train_losses)))
            self.history["val_loss"].append(val_loss)
            self.history["val_rmse"].append(val_metrics["rmse"])
            self.history["val_mae"].append(val_metrics["mae"])

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                        "config": self.config,
                    },
                    checkpoint_dir / "best_model.pt",
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        ckpt = torch.load(checkpoint_dir / "best_model.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

        with open(checkpoint_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        self.normalizer = normalizer
        return self.model

    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
        loss_fn,
        normalizer: FeatureNormalizer,
    ) -> Tuple[float, Dict[str, float]]:
        assert self.model is not None
        self.model.eval()
        losses, preds, targets = [], [], []

        for xb, yb in val_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            pred = self.model(xb)
            if isinstance(loss_fn, ControlAwareLoss):
                loss = loss_fn(pred, yb)
            else:
                loss = loss_fn(pred, yb)
            losses.append(loss.item())
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())

        preds_np = normalizer.inverse_transform_targets(np.vstack(preds))
        targets_np = normalizer.inverse_transform_targets(np.vstack(targets))
        metrics = compute_prediction_metrics(targets_np, preds_np)
        return float(np.mean(losses)), metrics

    @torch.no_grad()
    def predict(self, X: np.ndarray, normalizer: FeatureNormalizer) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        ds = SequenceDataset(X, np.zeros((len(X), 2)))
        loader = DataLoader(ds, batch_size=64)
        preds = []
        for xb, _ in loader:
            xb = xb.to(self.device)
            preds.append(self.model(xb).cpu().numpy())
        preds_np = np.vstack(preds)
        return normalizer.inverse_transform_targets(preds_np)
