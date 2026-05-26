"""
Wrappers for sklearn baselines and learned policy inference in closed-loop.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np
import torch

from models.lstm_policy import LSTMPolicy
from preprocessing.normalization import FeatureNormalizer


class LearnedPolicyWrapper:
    """Run LSTM policy in closed-loop with rolling observation window."""

    def __init__(
        self,
        model: LSTMPolicy,
        normalizer: FeatureNormalizer,
        feature_columns: List[str],
        sequence_length: int,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.normalizer = normalizer
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.device = device
        self._buffer: Deque[np.ndarray] = deque(maxlen=sequence_length)

    def reset(self) -> None:
        self._buffer.clear()

    def _build_feature_vector(self, obs_dict: dict, engineered: dict) -> np.ndarray:
        merged = {**obs_dict, **engineered}
        return np.array([merged[c] for c in self.feature_columns], dtype=np.float32)

    def act(
        self,
        obs_dict: dict,
        engineered_features: Optional[dict] = None,
    ) -> tuple:
        engineered_features = engineered_features or {}
        vec = self._build_feature_vector(obs_dict, engineered_features)
        self._buffer.append(vec)

        if len(self._buffer) < self.sequence_length:
            return 0.0, 0.0

        seq = np.stack(list(self._buffer))
        n_feat = seq.shape[1]
        seq_norm = self.normalizer._feature_scaler.transform(seq)  # noqa: SLF001
        x = torch.from_numpy(seq_norm).float().unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x).cpu().numpy()
        action = self.normalizer.inverse_transform_targets(out)[0]
        return float(action[0]), float(action[1])


class SklearnBaseline:
    """Placeholder for optional MLP baseline (not primary)."""

    def __init__(self) -> None:
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((len(X), 2))
