"""
Temporal feature engineering for policy learning.

Derived features capture EC dynamics, rolling statistics, and control history.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Add derived columns to labeled trajectory DataFrames."""

    BASE_FEATURES = [
        "water_temp",
        "ec",
        "turbidity",
        "prev_flowrate",
        "prev_duration",
        "time_since_last_dose",
    ]
    OPTIONAL_FEATURES = ["ph", "dissolved_oxygen", "ambient_temp"]
    TARGETS = ["optimal_flowrate", "optimal_duration"]

    def __init__(
        self,
        ec_target: float = 1.2,
        rolling_window: int = 8,
        include_optional: bool = True,
    ) -> None:
        self.ec_target = ec_target
        self.rolling_window = rolling_window
        self.include_optional = include_optional

    @property
    def feature_columns(self) -> List[str]:
        cols = list(self.BASE_FEATURES)
        if self.include_optional:
            cols.extend([c for c in self.OPTIONAL_FEATURES if c not in cols])
        derived = [
            "delta_ec",
            "delta_temp",
            "delta_turbidity",
            "rolling_avg_ec",
            "rolling_std_ec",
            "ec_error",
            "cumulative_nutrients",
            "dosing_frequency",
        ]
        return cols + derived

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features per trajectory (grouped by trajectory_id)."""
        out_parts = []
        for traj_id, group in df.groupby("trajectory_id", sort=False):
            g = group.sort_values("timestep").copy()
            out_parts.append(self._transform_single(g))
        return pd.concat(out_parts, ignore_index=True)

    def _transform_single(self, g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["delta_ec"] = g["ec"].diff().fillna(0.0)
        g["delta_temp"] = g["water_temp"].diff().fillna(0.0)
        g["delta_turbidity"] = g["turbidity"].diff().fillna(0.0)

        w = self.rolling_window
        g["rolling_avg_ec"] = g["ec"].rolling(w, min_periods=1).mean()
        g["rolling_std_ec"] = g["ec"].rolling(w, min_periods=1).std().fillna(0.0)
        g["ec_error"] = g["ec"] - self.ec_target

        # Cumulative nutrient proxy from actions
        dose = g["prev_flowrate"] * g["prev_duration"] / 60.0
        if "flowrate" in g.columns:
            dose = dose + g.get("flowrate", 0) * g.get("duration", 0) / 60.0 * 0.0
        g["cumulative_nutrients"] = dose.cumsum()

        # Dosing events in rolling window
        dosed = (g["prev_flowrate"] > 0).astype(float)
        g["dosing_frequency"] = dosed.rolling(w, min_periods=1).sum()

        return g
