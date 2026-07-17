"""
Temporal feature engineering for policy learning.

Derived features capture biological states, EC dynamics, rolling statistics, and control history.
Biological features expose variables optimized by the oracle (health, reserve, damage, biomass).
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
    # Biological state features from the ecosystem model
    BIOLOGICAL_FEATURES = [
        "health_index",
        "damage_index",
        "algae_biomass",
        "internal_reserve",
        "dissolved_nutrient_mass",
        "dead_biomass_pool",
    ]
    TARGETS = ["optimal_flowrate", "optimal_duration"]

    def __init__(
        self,
        ec_target: float = 1.2,
        rolling_window: int = 8,
        include_optional: bool = True,
        include_biological: bool = True,
        internal_capacity: float = 0.125,
    ) -> None:
        self.ec_target = ec_target
        self.rolling_window = rolling_window
        self.include_optional = include_optional
        self.include_biological = include_biological
        self.internal_capacity = internal_capacity

    @property
    def feature_columns(self) -> List[str]:
        cols = list(self.BASE_FEATURES)
        if self.include_optional:
            cols.extend([c for c in self.OPTIONAL_FEATURES if c not in cols])
        if self.include_biological:
            cols.extend([c for c in self.BIOLOGICAL_FEATURES if c not in cols])
        
        derived = [
            "delta_ec",
            "delta_temp",
            "delta_turbidity",
            "rolling_avg_ec",
            "rolling_std_ec",
            "ec_error",
            "reserve_ratio",
            "dosing_frequency",
        ]
        
        if self.include_biological:
            derived.extend([
                "delta_health",
                "delta_damage",
                "delta_biomass",
                "rolling_avg_health",
                "rolling_avg_reserve",
            ])
        
        return cols + derived
    
    def get_available_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Return only feature columns that actually exist in the dataframe.
        This handles cases where biological features are requested but not present in data.
        """
        all_cols = self.feature_columns
        available_cols = [col for col in all_cols if col in df.columns]
        return available_cols

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

        # Reserve ratio: internal_reserve / (algae_biomass * internal_capacity)
        # This is a key metric for starvation risk and replaces cumulative_nutrients
        if "internal_reserve" in g.columns and "algae_biomass" in g.columns:
            g["reserve_ratio"] = g["internal_reserve"] / (g["algae_biomass"] * self.internal_capacity + 1e-6)
        else:
            # Fallback if biological features not available
            g["reserve_ratio"] = 0.5  # Default healthy reserve ratio

        # Dosing events in rolling window
        dosed = (g["prev_flowrate"] > 0).astype(float)
        g["dosing_frequency"] = dosed.rolling(w, min_periods=1).sum()

        # Biological state features (if available in the data)
        if self.include_biological:
            # Temporal derivatives of biological states
            if "health_index" in g.columns:
                g["delta_health"] = g["health_index"].diff().fillna(0.0)
                g["rolling_avg_health"] = g["health_index"].rolling(w, min_periods=1).mean()
            
            if "damage_index" in g.columns:
                g["delta_damage"] = g["damage_index"].diff().fillna(0.0)
            
            if "algae_biomass" in g.columns:
                g["delta_biomass"] = g["algae_biomass"].diff().fillna(0.0)
            
            if "internal_reserve" in g.columns:
                g["rolling_avg_reserve"] = g["internal_reserve"].rolling(w, min_periods=1).mean()

        return g
