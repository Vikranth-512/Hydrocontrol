"""
Sliding-window sequence generation for LSTM policy learning.

Output shape: (samples, timesteps, features)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceBuilder:
    """Build (X, y) sequence tensors from engineered DataFrames."""

    def __init__(
        self,
        sequence_length: int = 32,
        prediction_horizon: int = 1,
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_columns = feature_columns or []
        self.target_columns = target_columns or ["optimal_flowrate", "optimal_duration"]

    def build_from_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns X (N, T, F), y (N, 2), trajectory_ids (N,).

        For each valid index t, window is [t-seq_len+1, t] and label at t.
        """
        X_list, y_list, ids_list = [], [], []

        for traj_id, group in df.groupby("trajectory_id", sort=False):
            g = group.sort_values("timestep")
            feats = g[self.feature_columns].values.astype(np.float32)
            targets = g[self.target_columns].values.astype(np.float32)
            n = len(g)

            for t in range(self.sequence_length - 1, n - self.prediction_horizon + 1):
                start = t - self.sequence_length + 1
                window = feats[start : t + 1]
                label_idx = t + self.prediction_horizon - 1
                X_list.append(window)
                y_list.append(targets[label_idx])
                ids_list.append(traj_id)

        if not X_list:
            raise ValueError("No sequences built; check data length vs sequence_length")

        return (
            np.stack(X_list),
            np.stack(y_list),
            np.array(ids_list),
        )

    @staticmethod
    def train_val_test_split(
        X: np.ndarray,
        y: np.ndarray,
        traj_ids: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """Split by trajectory ID to avoid leakage across windows."""
        rng = np.random.default_rng(seed)
        unique_ids = np.unique(traj_ids)
        rng.shuffle(unique_ids)

        n = len(unique_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_ids = set(unique_ids[:n_train])
        val_ids = set(unique_ids[n_train : n_train + n_val])
        test_ids = set(unique_ids[n_train + n_val :])

        def mask(id_set):
            return np.array([tid in id_set for tid in traj_ids])

        m_train, m_val, m_test = mask(train_ids), mask(val_ids), mask(test_ids)

        return {
            "X_train": X[m_train],
            "y_train": y[m_train],
            "X_val": X[m_val],
            "y_val": y[m_val],
            "X_test": X[m_test],
            "y_test": y[m_test],
            "train_ids": traj_ids[m_train],
            "val_ids": traj_ids[m_val],
            "test_ids": traj_ids[m_test],
        }


class SequenceDataset(Dataset):
    """PyTorch Dataset wrapper."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
