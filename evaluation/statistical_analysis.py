"""
Statistical analysis routines for paired multi-seed, multi-regime controller evaluation.

Implements:
1. Episode-level paired differences (LSTM - PID, LSTM - ReserveRule, PID - ReserveRule)
2. 95% bootstrap confidence intervals with deterministic resampling
3. Pre-specified practical-equivalence testing (TOST against [-delta, +delta] margins)
4. Documented threshold exceedance rates
5. Direction-aware controller win rates and tie handling
6. Equal-weighted cross-scenario aggregation
7. Supplementary Wilcoxon signed-rank testing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Pre-registered Metric Specifications and Threshold Provenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    key: str
    display_name: str
    unit: str
    category: str
    higher_is_better: bool
    practical_threshold: float
    threshold_justification: str


METRIC_SPECS: Dict[str, MetricSpec] = {
    # Regulation metrics
    "ec_mae": MetricSpec(
        key="ec_mae",
        display_name="EC MAE",
        unit="mS/cm",
        category="regulation",
        higher_is_better=False,
        practical_threshold=0.04,
        threshold_justification=(
            "Matches controller deadband (0.04 mS/cm) and 2*sigma_ec sensor noise "
            "(sigma=0.02 mS/cm). Differences below 0.04 are within 2-sigma measurement "
            "noise and below actuator trigger threshold."
        ),
    ),
    "ec_rmse": MetricSpec(
        key="ec_rmse",
        display_name="EC RMSE",
        unit="mS/cm",
        category="regulation",
        higher_is_better=False,
        practical_threshold=0.05,
        threshold_justification=(
            "Root mean square deviation scaled to 2.5*sigma_ec noise level."
        ),
    ),
    "final_ec_deviation": MetricSpec(
        key="final_ec_deviation",
        display_name="Final EC Deviation",
        unit="mS/cm",
        category="regulation",
        higher_is_better=False,
        practical_threshold=0.04,
        threshold_justification="Deviation from ec_target=1.2 at episode termination.",
    ),
    # Biological performance
    "final_biomass": MetricSpec(
        key="final_biomass",
        display_name="Final Biomass",
        unit="g/L",
        category="biological",
        higher_is_better=True,
        practical_threshold=5.0,
        threshold_justification=(
            "Baseline operating biomass is 80-100 g/L (b_ref=120 g/L). A difference of 5.0 g/L "
            "corresponds to ~5-6% meaningful biological culture yield divergence, exceeding "
            "natural background biomass fluctuation (~2-3 g/L)."
        ),
    ),
    "biomass_change": MetricSpec(
        key="biomass_change",
        display_name="Biomass Net Change",
        unit="g/L",
        category="biological",
        higher_is_better=True,
        practical_threshold=5.0,
        threshold_justification="Net accumulation of biomass over the 3500-step horizon.",
    ),
    "final_reserve_ratio": MetricSpec(
        key="final_reserve_ratio",
        display_name="Final Reserve Ratio (rho)",
        unit="fraction",
        category="biological",
        higher_is_better=True,
        practical_threshold=0.04,
        threshold_justification=(
            "Matches regulation deadband (0.04) around rho_setpoint=0.50. "
            "Differences below 0.04 are within the unperturbed deadband."
        ),
    ),
    "reserve_min": MetricSpec(
        key="reserve_min",
        display_name="Minimum Reserve Ratio",
        unit="fraction",
        category="biological",
        higher_is_better=True,
        practical_threshold=0.04,
        threshold_justification="Safety margin above starvation threshold (rho=0.10).",
    ),
    "final_health": MetricSpec(
        key="final_health",
        display_name="Final Health Index",
        unit="index [0-1]",
        category="biological",
        higher_is_better=True,
        practical_threshold=0.05,
        threshold_justification="Biological vitality index at horizon termination.",
    ),
    # Nutrient behaviour
    "dissolved_nutrient_mean": MetricSpec(
        key="dissolved_nutrient_mean",
        display_name="Mean Dissolved Nutrient",
        unit="mass units",
        category="nutrient",
        higher_is_better=False,
        practical_threshold=0.10,
        threshold_justification="Excess dissolved nutrient reflects waste accumulation.",
    ),
    "nutrient_accumulation": MetricSpec(
        key="nutrient_accumulation",
        display_name="Nutrient Accumulation",
        unit="mass units",
        category="nutrient",
        higher_is_better=False,
        practical_threshold=0.20,
        threshold_justification="Total positive cumulative accumulation of unabsorbed nutrient.",
    ),
    # Control effort
    "total_dose": MetricSpec(
        key="total_dose",
        display_name="Total Dose",
        unit="mL",
        category="effort",
        higher_is_better=False,
        practical_threshold=10.0,
        threshold_justification=(
            "Actuator baseline: standard moderate pulse is 2.0 mL/min * 15s = 0.5 mL. "
            "A cumulative difference of 10.0 mL represents 20 moderate interventions, "
            "constituting a material chemical and actuator wear cost."
        ),
    ),
    "intervention_fraction": MetricSpec(
        key="intervention_fraction",
        display_name="Intervention Fraction",
        unit="fraction",
        category="effort",
        higher_is_better=False,
        practical_threshold=0.02,
        threshold_justification="2% of steps active (~70 steps difference over 3500-step horizon).",
    ),
    "n_interventions": MetricSpec(
        key="n_interventions",
        display_name="Number of Interventions",
        unit="count",
        category="effort",
        higher_is_better=False,
        practical_threshold=20.0,
        threshold_justification="20 discrete dosing activations over 3500 steps.",
    ),
}


# ---------------------------------------------------------------------------
# Paired Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def compute_paired_bootstrap_ci(
    differences: np.ndarray,
    n_replicates: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 12345,
) -> Dict[str, float]:
    """
    Compute paired bootstrap confidence interval over episode-level differences.
    
    Args:
        differences: 1D array of paired differences (M_A,i - M_B,i) across episodes.
        n_replicates: Number of bootstrap resamples (at least 5000).
        confidence_level: Confidence level (default 0.95).
        seed: Deterministic bootstrap RNG seed.
        
    Returns:
        Dict with mean_difference, median_difference, std_difference,
        ci95_lower, ci95_upper, bootstrap_replicates, n_episodes.
    """
    diffs = np.asarray(differences, dtype=np.float64)
    n = len(diffs)
    if n == 0:
        return {
            "mean_difference": float("nan"),
            "median_difference": float("nan"),
            "std_difference": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "bootstrap_replicates": n_replicates,
            "n_episodes": 0,
        }

    rng = np.random.default_rng(seed)
    # Resample indices with replacement: shape (n_replicates, n)
    boot_indices = rng.integers(0, n, size=(n_replicates, n))
    boot_means = np.mean(diffs[boot_indices], axis=1)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean_difference": float(np.mean(diffs)),
        "median_difference": float(np.median(diffs)),
        "std_difference": float(np.std(diffs, ddof=1) if n > 1 else 0.0),
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "bootstrap_replicates": n_replicates,
        "n_episodes": n,
    }


# ---------------------------------------------------------------------------
# Cross-Scenario Equal-Weighted Aggregation
# ---------------------------------------------------------------------------

def aggregate_cross_scenarios_bootstrap(
    scenario_diffs: Dict[str, np.ndarray],
    n_replicates: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 12345,
) -> Dict[str, float]:
    """
    Compute equal-weighted aggregate difference across primary scenarios:
        Delta_agg = 1/K * sum_{s=1}^K Delta_s
    
    Bootstraps by resampling seeds within each scenario independently and
    averaging the scenario means across replicates.
    
    Args:
        scenario_diffs: Dict mapping scenario_name -> 1D array of differences.
        n_replicates: Bootstrap replicate count.
        confidence_level: Confidence level (default 0.95).
        seed: Deterministic bootstrap seed.
        
    Returns:
        Dict with aggregate mean, median, std, ci95_lower, ci95_upper.
    """
    scenarios = sorted(scenario_diffs.keys())
    k_scenarios = len(scenarios)
    if k_scenarios == 0:
        return {}

    scenario_means = [float(np.mean(scenario_diffs[s])) for s in scenarios]
    point_aggregate_mean = float(np.mean(scenario_means))

    rng = np.random.default_rng(seed)
    boot_scenario_means = np.zeros((n_replicates, k_scenarios), dtype=np.float64)

    for i, s in enumerate(scenarios):
        vals = np.asarray(scenario_diffs[s], dtype=np.float64)
        n_s = len(vals)
        if n_s == 0:
            continue
        boot_idx = rng.integers(0, n_s, size=(n_replicates, n_s))
        boot_scenario_means[:, i] = np.mean(vals[boot_idx], axis=1)

    # Equal-weighted mean across scenarios for each bootstrap replicate
    boot_aggregates = np.mean(boot_scenario_means, axis=1)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_aggregates, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_aggregates, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean_difference": point_aggregate_mean,
        "median_difference": float(np.median(boot_aggregates)),
        "std_difference": float(np.std(boot_aggregates)),
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "bootstrap_replicates": n_replicates,
        "n_scenarios": k_scenarios,
    }


# ---------------------------------------------------------------------------
# Direction-Aware Win Rates & Threshold Exceedance
# ---------------------------------------------------------------------------

def compute_controller_win_rate(
    differences: np.ndarray,
    higher_is_better: bool,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Compute controller win/loss/tie rates based on paired differences:
        Delta_i = M_LSTM,i - M_Reference,i
    
    If higher is better: win if Delta_i > +tol
    If lower is better:  win if Delta_i < -tol
    """
    diffs = np.asarray(differences, dtype=np.float64)
    n = len(diffs)
    if n == 0:
        return {"win_count": 0, "loss_count": 0, "tie_count": 0, "win_rate": 0.0, "n_episodes": 0}

    if higher_is_better:
        wins = int(np.sum(diffs > tol))
        losses = int(np.sum(diffs < -tol))
    else:
        wins = int(np.sum(diffs < -tol))
        losses = int(np.sum(diffs > tol))

    ties = n - wins - losses
    return {
        "win_count": wins,
        "loss_count": losses,
        "tie_count": ties,
        "win_rate": float(wins / n),
        "loss_rate": float(losses / n),
        "tie_rate": float(ties / n),
        "n_episodes": n,
    }


def compute_preregistered_threshold_exceedance(
    differences: np.ndarray,
    threshold: float,
    higher_is_better: bool,
) -> Dict[str, Any]:
    """
    Calculate fraction of paired episodes exceeding the pre-registered engineering threshold.
    """
    diffs = np.asarray(differences, dtype=np.float64)
    n = len(diffs)
    if n == 0:
        return {"count": 0, "fraction": 0.0, "threshold": threshold}

    if higher_is_better:
        exceed_mask = diffs >= threshold
    else:
        exceed_mask = diffs <= -threshold

    count = int(np.sum(exceed_mask))
    return {
        "count": count,
        "fraction": float(count / n),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Equivalence Analysis for ML-Discriminability
# ---------------------------------------------------------------------------

def evaluate_practical_equivalence(
    ci_lower: float,
    ci_upper: float,
    threshold_delta: float,
    higher_is_better: bool,
) -> Dict[str, str]:
    """
    Two One-Sided Tests (TOST) style practical equivalence classification:
    Examines whether the 95% CI of Delta = M_LSTM - M_ReserveRule against [-delta, +delta]:
    1. Lies strictly beyond the positive threshold -> Evidence of LSTM advantage
    2. Lies entirely inside [-delta, +delta] -> Evidence of practical equivalence
    3. Lies strictly beyond the negative threshold -> Evidence of reserve-rule advantage
    4. Overlaps both -> Inconclusive
    """
    delta = abs(threshold_delta)
    eq_min = -delta
    eq_max = delta

    if higher_is_better:
        # LSTM advantage: Delta > +delta
        if ci_lower > eq_max:
            verdict = "LSTM advantage"
            interp = f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] lies entirely above +{delta:.4f}."
        elif ci_lower >= eq_min and ci_upper <= eq_max:
            verdict = "Practical equivalence"
            interp = f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] lies entirely within equivalence margin [-{delta:.4f}, +{delta:.4f}]."
        elif ci_upper < eq_min:
            verdict = "Reserve rule advantage"
            interp = f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] lies entirely below -{delta:.4f}."
        else:
            verdict = "Inconclusive"
            interp = f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] spans across equivalence boundary [-{delta:.4f}, +{delta:.4f}]."
    else:
        # Lower is better: LSTM advantage: Delta < -delta
        if ci_upper < eq_min:
            verdict = "LSTM advantage"
            interp = f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] lies entirely below -{delta:.4f}."
        elif ci_lower >= eq_min and ci_upper <= eq_max:
            verdict = "Practical equivalence"
            interp = f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] lies entirely within equivalence margin [-{delta:.4f}, +{delta:.4f}]."
        elif ci_lower > eq_max:
            verdict = "Reserve rule advantage"
            interp = f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] lies entirely above +{delta:.4f}."
        else:
            verdict = "Inconclusive"
            interp = f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] spans across equivalence boundary [-{delta:.4f}, +{delta:.4f}]."

    return {
        "verdict": verdict,
        "interpretation": interp,
        "equivalence_margin": f"[-{delta:.4f}, +{delta:.4f}]",
    }


# ---------------------------------------------------------------------------
# Supplementary Wilcoxon Signed-Rank Test
# ---------------------------------------------------------------------------

def compute_wilcoxon_paired(differences: np.ndarray) -> Dict[str, Any]:
    """
    Perform Wilcoxon signed-rank test on paired differences as a supplementary audit.
    """
    diffs = np.asarray(differences, dtype=np.float64)
    non_zero = diffs[np.abs(diffs) > 1e-12]
    n_pairs = len(diffs)
    n_non_zero = len(non_zero)

    if n_non_zero < 5:
        # Not enough non-zero pairs for meaningful Wilcoxon test
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_pairs": n_pairs,
            "n_non_zero_pairs": n_non_zero,
            "test_performed": False,
        }

    try:
        res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return {
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "n_pairs": n_pairs,
            "n_non_zero_pairs": n_non_zero,
            "test_performed": True,
        }
    except Exception as e:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_pairs": n_pairs,
            "n_non_zero_pairs": n_non_zero,
            "test_performed": False,
            "error": str(e),
        }
