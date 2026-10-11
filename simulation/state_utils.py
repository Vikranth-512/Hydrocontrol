"""
Neutral state utilities for algae tank simulation and controller baselines.
Provides canonical biological state helpers decoupled from any optimization/labeling logic.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def compute_rho(state: Any, internal_capacity: float = 0.5) -> float:
    """
    Compute normalized intracellular nutrient reserve ratio rho in [0.0, 1.0].

    rho = R / (B * C_int)
    where:
      R = internal nutrient reserve
      B = algae biomass (safeguarded against near-zero division via max(B, 1e-6))
      C_int = maximum intracellular nutrient capacity per unit biomass (default 0.5)

    This is the single canonical implementation used across controllers,
    supervisory labelers, and evaluation pipelines.
    """
    B = max(float(getattr(state, "algae_biomass", 1.0)), 1e-6)
    reserve = float(getattr(state, "internal_reserve", 0.0))
    return float(np.clip(reserve / (B * internal_capacity), 0.0, 1.0))
