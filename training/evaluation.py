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


def compute_ecosystem_metrics(
    state_trace: List["TankState"],  # type: ignore[name-defined]
    flowrates: np.ndarray,
    durations: np.ndarray,
    params: "TankDynamicsParams",  # type: ignore[name-defined]
    dt: float = 60.0,
    b_bloom_threshold: float = 300.0,
) -> Dict[str, float]:
    """
    Ecosystem regulation metrics. No EC reference needed for primary score.
    """
    if not state_trace:
        return {}

    B = np.array([max(s.algae_biomass, 1e-6) for s in state_trace])
    H = np.array([s.health_index for s in state_trace])
    D = np.array([s.damage_index for s in state_trace])
    ec = np.array([s.ec for s in state_trace])
    
    # Reserve ratio (rho)
    ic = params.internal_capacity
    rho = np.array([s.internal_reserve / (B[i] * ic) for i, s in enumerate(state_trace)])
    rho = np.clip(rho, 0.0, 1.0)

    tail_n = max(20, int(len(H) * 0.35))
    H_tail = H[-tail_n:]
    B_tail = B[-tail_n:]
    rho_all = rho

    dose = flowrates * durations / 60.0
    total_dose = float(np.sum(dose))

    return {
        # Primary ecosystem health
        "health_mean":          float(np.mean(H)),
        "health_mean_tail":     float(np.mean(H_tail)),
        "damage_mean":          float(np.mean(D)),
        "damage_cumulative":    float(np.sum(np.maximum(0.0, np.diff(D, prepend=D[0])))),

        # Reserve / starvation risk
        "reserve_mean":         float(np.mean(rho_all)),
        "reserve_floor_frac":   float(np.mean(rho_all > 0.20)),
        "starvation_fraction":  float(np.mean(rho_all < 0.10)),

        # Biomass stability
        "biomass_mean":         float(np.mean(B_tail)),
        "biomass_cv_tail":      float(np.std(B_tail) / (np.mean(B_tail) + 1e-6)),
        "bloom_fraction":       float(np.mean(B > b_bloom_threshold)),

        # Collapse risk
        "collapse_free_duration": int(np.argmax(H < 0.20)) if np.any(H < 0.20) else len(H),
        "biomass_persistence":  float(np.mean(B > 20.0)),

        # Efficiency
        "total_dose":           total_dose,
        "dose_per_health":      float(total_dose / (np.mean(H) * len(H) + 1e-6)),
        "control_smoothness":   float(np.mean(np.diff(flowrates, prepend=flowrates[0])**2)),

        # Secondary EC diagnostics (reported, not optimized)
        "ec_mean":              float(np.mean(ec)),
        "ec_std":               float(np.std(ec)),
        "ec_min":               float(np.min(ec)),
        "ec_max":               float(np.max(ec)),
        "ec_range":             float(np.max(ec) - np.min(ec)),
    }


def robustness_summary(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across robustness scenarios."""
    if not results:
        return {}
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}
