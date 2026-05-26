"""
Train-only feature normalization with persisted scalers.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ScalerType = Literal["standard", "minmax"]


class FeatureNormalizer:
    """Fit on training data only; transform val/test without leakage."""

    def __init__(self, scaler_type: ScalerType = "standard") -> None:
        self.scaler_type = scaler_type
        self._feature_scaler: Optional[object] = None
        self._target_scaler: Optional[object] = None
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []

    def _make_scaler(self):
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        return StandardScaler()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_columns: List[str],
        target_columns: List[str],
    ) -> "FeatureNormalizer":
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self._feature_scaler = self._make_scaler()
        self._target_scaler = self._make_scaler()

        n_samples, seq_len, n_feat = X.shape
        X_flat = X.reshape(-1, n_feat)
        self._feature_scaler.fit(X_flat)

        self._target_scaler.fit(y)
        return self

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        assert self._feature_scaler is not None
        n, t, f = X.shape
        flat = self._feature_scaler.transform(X.reshape(-1, f))
        return flat.reshape(n, t, f)

    def transform_targets(self, y: np.ndarray) -> np.ndarray:
        assert self._target_scaler is not None
        return self._target_scaler.transform(y)

    def inverse_transform_targets(self, y: np.ndarray) -> np.ndarray:
        assert self._target_scaler is not None
        return self._target_scaler.inverse_transform(y)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "feature_scaler.pkl", "wb") as f:
            pickle.dump(self._feature_scaler, f)
        with open(directory / "target_scaler.pkl", "wb") as f:
            pickle.dump(self._target_scaler, f)
        meta = {
            "scaler_type": self.scaler_type,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
        }
        with open(directory / "normalizer_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: Path) -> "FeatureNormalizer":
        directory = Path(directory)
        with open(directory / "normalizer_meta.json") as f:
            meta = json.load(f)
        norm = cls(scaler_type=meta["scaler_type"])
        norm.feature_columns = meta["feature_columns"]
        norm.target_columns = meta["target_columns"]
        with open(directory / "feature_scaler.pkl", "rb") as f:
            norm._feature_scaler = pickle.load(f)
        with open(directory / "target_scaler.pkl", "rb") as f:
            norm._target_scaler = pickle.load(f)
        return norm
