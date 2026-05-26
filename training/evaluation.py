"""
Prediction and control-quality metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_prediction_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """RMSE, MAE, R² per output dimension and aggregate."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    for i, name in enumerate(["flowrate", "duration"]):
        metrics[f"rmse_{name}"] = float(
            np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        )
        metrics[f"mae_{name}"] = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
    return metrics


def compute_control_metrics(
    ec_trace: np.ndarray,
    ec_target: float,
    flowrates: np.ndarray,
    durations: np.ndarray,
    dt: float = 60.0,
) -> Dict[str, float]:
    """
    Control metrics: overshoot, settling time, stability variance,
    nutrient efficiency, cumulative cost.
    """
    ec = np.asarray(ec_trace)
    errors = ec - ec_target

    overshoot = float(np.max(np.maximum(0.0, ec - ec_target)))

    # Settling: first time |error| < 5% of target and stays (simplified)
    band = 0.05 * ec_target
    settled_idx = len(ec) - 1
    for i in range(len(ec)):
        if np.all(np.abs(ec[i:] - ec_target) < band) if i < len(ec) - 1 else abs(ec[i] - ec_target) < band:
            settled_idx = i
            break
    settling_time = float(settled_idx * dt)

    stability_variance = float(np.var(ec))

    dose_per_step = flowrates * durations / 60.0
    cumulative_cost = float(np.sum(dose_per_step))
    ec_mae = float(np.mean(np.abs(errors)))

    # Efficiency: lower cost per unit EC accuracy
    nutrient_efficiency = ec_mae / (cumulative_cost + 1e-6)

    return {
        "overshoot": overshoot,
        "settling_time": settling_time,
        "stability_variance": stability_variance,
        "cumulative_dosing_cost": cumulative_cost,
        "ec_mae": ec_mae,
        "nutrient_efficiency": nutrient_efficiency,
    }


def robustness_summary(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across robustness scenarios."""
    if not results:
        return {}
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}
