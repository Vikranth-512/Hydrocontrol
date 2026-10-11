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


def compute_intervention_metrics_from_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute intervention precision/recall from regression outputs.
    
    Intervention is defined as action != (0, 0). No classifier needed.
    
    Args:
        y_true: (N, 2) - target (flowrate, duration)
        y_pred: (N, 2) - predicted (flowrate, duration)
    
    Returns:
        Dict with precision, recall, F1, confusion matrix, etc.
    """
    # True intervention labels: 1 if either action > 0
    true_intervention = ((y_true[:, 0] > 0) | (y_true[:, 1] > 0)).astype(int)
    
    # Predicted intervention: 1 if either predicted action > 0
    pred_intervention = ((y_pred[:, 0] > 0) | (y_pred[:, 1] > 0)).astype(int)
    
    # Confusion matrix
    tp = np.sum((true_intervention == 1) & (pred_intervention == 1))
    tn = np.sum((true_intervention == 0) & (pred_intervention == 0))
    fp = np.sum((true_intervention == 0) & (pred_intervention == 1))
    fn = np.sum((true_intervention == 1) & (pred_intervention == 0))
    
    # Metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    # Regression MAE on intervention samples only
    intervention_mask = true_intervention == 1
    if intervention_mask.sum() > 0:
        mae_intervention = np.mean(np.abs(y_pred[intervention_mask] - y_true[intervention_mask]))
    else:
        mae_intervention = 0.0
    
    return {
        "intervention_precision": float(precision),
        "intervention_recall": float(recall),
        "intervention_f1": float(f1),
        "intervention_accuracy": float(accuracy),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "mae_intervention_only": float(mae_intervention),
        "intervention_rate_true": float(np.mean(true_intervention)),
        "intervention_rate_pred": float(np.mean(pred_intervention)),
    }


def compute_ecosystem_metrics(
    state_trace,
    flowrates,
    durations,
    params,
    dt: float = 60.0,
    b_bloom_threshold: float = 300.0,
    ec_target: float = 1.2,
    episode_terminated: bool = False,
    termination_step=None,
    termination_reason: str = "completed",
    nominal_horizon: int = 3500,
    flowrate_cmd=None,
    duration_cmd=None,
):
    """
    Ecosystem regulation and controller benchmarking metrics.
    Preserves all historical keys while reporting extended regulation,
    biological, nutrient behavior, control effort, and horizon integrity metrics.

    Args:
        state_trace: Sequence of TankState objects from the episode.
        flowrates: Realized flowrate actions applied to the environment each step.
        durations: Realized duration actions applied to the environment each step.
        params: TankDynamicsParams used in the episode.
        dt: Timestep in seconds (default 60 s).
        b_bloom_threshold: Biomass bloom threshold in g/L (default 300).
        ec_target: EC setpoint in mS/cm (default 1.2).
        episode_terminated: True if simulator terminated early.
        termination_step: Step at which termination occurred (None = completed).
        termination_reason: Human-readable reason for termination.
        nominal_horizon: Full intended episode length (default 3500).
        flowrate_cmd: Commanded flowrate before actuator effects (None -> use flowrates).
        duration_cmd: Commanded duration before actuator effects (None -> use durations).
    """
    import numpy as np
    from collections import Counter

    if not state_trace:
        return {}

    B = np.array([max(s.algae_biomass, 1e-6) for s in state_trace])
    H = np.array([s.health_index for s in state_trace])
    D = np.array([s.damage_index for s in state_trace])
    ec = np.array([s.ec for s in state_trace])
    dissolved_masses = np.array([s.dissolved_nutrient_mass for s in state_trace])

    # Reserve ratio (rho)
    ic = params.internal_capacity
    rho = np.array([s.internal_reserve / (B[i] * ic) for i, s in enumerate(state_trace)])
    rho = np.clip(rho, 0.0, 1.0)

    tail_n = max(20, int(len(H) * 0.35))
    H_tail = H[-tail_n:]
    B_tail = B[-tail_n:]
    rho_all = rho

    # Horizon integrity
    eff_horizon = int(len(state_trace))
    term_step = int(termination_step) if termination_step is not None else eff_horizon

    # Realized dosing (what was applied to the environment)
    dose_realized = flowrates * durations / 60.0
    total_dose = float(np.sum(dose_realized))

    # Commanded vs realized dosing (commanded = what controller requested before actuator effects)
    fr_c = flowrate_cmd if flowrate_cmd is not None else flowrates
    dur_c = duration_cmd if duration_cmd is not None else durations
    dose_cmd_arr = fr_c * dur_c / 60.0
    total_dose_cmd = float(np.sum(dose_cmd_arr))
    total_dose_realized = total_dose

    # Intervention statistics (using commanded actions to reflect controller intent)
    active_mask = (fr_c > 0.0) | (dur_c > 0.0)
    n_interventions = int(np.sum(active_mask))
    intervention_fraction = float(np.mean(active_mask))
    mean_flowrate_active = float(np.mean(fr_c[active_mask])) if n_interventions > 0 else 0.0
    mean_duration_active = float(np.mean(dur_c[active_mask])) if n_interventions > 0 else 0.0

    # Control action diversity metrics
    actions = list(zip(np.round(fr_c, 3), np.round(dur_c, 3)))
    unique_actions = len(set(actions))
    n_act = max(1, len(actions))
    action_counts = Counter(actions)
    action_probs = [c / n_act for c in action_counts.values()]
    action_entropy = -float(np.sum([p * np.log2(p) for p in action_probs if p > 0]))
    no_action_frac = float(np.mean((fr_c == 0.0) & (dur_c == 0.0)))
    max_dose_frac = float(np.mean((fr_c >= 4.0) & (dur_c >= 20.0)))

    # Numerical validity checks
    state_valid = bool(not np.any(np.isnan(B) | np.isinf(B) | np.isnan(H) | np.isinf(H)))
    action_valid = bool(not np.any(np.isnan(fr_c) | np.isinf(fr_c) | np.isnan(dur_c) | np.isinf(dur_c)))

    # Failure categorization
    if not state_valid or not action_valid:
        failure = True
        failure_type = "numerical_failure"
    elif episode_terminated:
        failure = True
        failure_type = "simulator_failure"
    else:
        failure = False
        failure_type = "none"

    # Canonical collapse threshold (B < 50.0 g/L) and valid survival
    final_b = float(B[-1]) if state_valid else float("nan")
    collapsed = bool(state_valid and final_b < 50.0)
    survival = bool(state_valid and final_b >= 50.0 and not failure)

    return {
        # Primary ecosystem health
        "health_mean":          float(np.mean(H)) if state_valid else float("nan"),
        "health_mean_tail":     float(np.mean(H_tail)) if state_valid else float("nan"),
        "damage_mean":          float(np.mean(D)) if state_valid else float("nan"),
        "damage_cumulative":    float(np.sum(np.maximum(0.0, np.diff(D, prepend=D[0])))) if state_valid else float("nan"),

        # Reserve / starvation risk
        "reserve_mean":         float(np.mean(rho_all)) if state_valid else float("nan"),
        "reserve_floor_frac":   float(np.mean(rho_all > 0.20)) if state_valid else float("nan"),
        "starvation_fraction":  float(np.mean(rho_all < 0.10)) if state_valid else float("nan"),

        # Biomass stability
        "biomass_mean":         float(np.mean(B_tail)) if state_valid else float("nan"),
        "biomass_cv_tail":      float(np.std(B_tail) / (np.mean(B_tail) + 1e-6)) if state_valid else float("nan"),
        "bloom_fraction":       float(np.mean(B > b_bloom_threshold)) if state_valid else float("nan"),

        # Collapse risk
        "collapse_free_duration": int(np.argmax(H < 0.20)) if np.any(H < 0.20) else len(H),
        "biomass_persistence":  float(np.mean(B > 20.0)) if state_valid else float("nan"),

        # Efficiency & Dosing
        "total_dose":           total_dose,
        "total_dose_commanded": total_dose_cmd,
        "total_dose_realized":  total_dose_realized,
        "dose_per_health":      float(total_dose / (np.mean(H) * len(H) + 1e-6)) if state_valid else float("nan"),
        "control_smoothness":   float(np.mean(np.diff(flowrates, prepend=flowrates[0])**2)) if action_valid else float("nan"),

        # Action diversity & effort
        "unique_actions_count": unique_actions,
        "action_entropy":       action_entropy,
        "no_action_fraction":   no_action_frac,
        "max_dose_fraction":    max_dose_frac,

        # Secondary EC diagnostics (reported, not optimized)
        "ec_mean":              float(np.mean(ec)) if state_valid else float("nan"),
        "ec_std":               float(np.std(ec)) if state_valid else float("nan"),
        "ec_min":               float(np.min(ec)) if state_valid else float("nan"),
        "ec_max":               float(np.max(ec)) if state_valid else float("nan"),
        "ec_range":             float(np.max(ec) - np.min(ec)) if state_valid else float("nan"),

        # Extended Regulation Metrics (Section 13)
        "ec_mae":               float(np.mean(np.abs(ec - ec_target))) if state_valid else float("nan"),
        "ec_rmse":              float(np.sqrt(np.mean((ec - ec_target) ** 2))) if state_valid else float("nan"),
        "final_ec":             float(ec[-1]) if state_valid else float("nan"),
        "final_ec_deviation":   float(np.abs(ec[-1] - ec_target)) if state_valid else float("nan"),

        # Extended Biological State Metrics (Section 13)
        "final_biomass":        final_b,
        "biomass_change":       float(B[-1] - B[0]) if state_valid else float("nan"),
        "reserve_min":          float(np.min(rho_all)) if state_valid else float("nan"),
        "reserve_max":          float(np.max(rho_all)) if state_valid else float("nan"),
        "reserve_std":          float(np.std(rho_all)) if state_valid else float("nan"),
        "final_reserve_ratio":  float(rho_all[-1]) if state_valid else float("nan"),
        "final_health":         float(H[-1]) if state_valid else float("nan"),

        # Extended Nutrient Behaviour Metrics (Section 13)
        "dissolved_nutrient_mean": float(np.mean(dissolved_masses)) if state_valid else float("nan"),
        "dissolved_nutrient_max":  float(np.max(dissolved_masses)) if state_valid else float("nan"),
        "final_dissolved_nutrient": float(dissolved_masses[-1]) if state_valid else float("nan"),
        "nutrient_accumulation":   float(np.sum(np.maximum(0.0, np.diff(dissolved_masses, prepend=dissolved_masses[0])))) if state_valid else float("nan"),

        # Control Effort & Intervention Metrics (Section 13)
        "n_interventions":      n_interventions,
        "intervention_fraction": intervention_fraction,
        "mean_flowrate_active": mean_flowrate_active,
        "mean_duration_active": mean_duration_active,

        # Horizon and Early-Termination Integrity
        "episode_terminated":   bool(episode_terminated),
        "termination_step":     term_step,
        "termination_reason":   str(termination_reason),
        "effective_horizon":    eff_horizon,
        "nominal_horizon":      int(nominal_horizon),

        # Failure and collapse classification
        "simulator_state_valid": state_valid,
        "controller_action_valid": action_valid,
        "failure":              failure,
        "failure_type":         failure_type,
        "collapsed":            collapsed,
        "survival":             survival,
    }

def robustness_summary(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across robustness scenarios."""
    if not results:
        return {}
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}
