"""
Failure and Collapse Probability Analysis for Hydroponic Nutrient Regulation Benchmark.

Audits episode termination, biomass collapse (B < 50 g/L primary, B < 20 g/L secondary),
and terminal biomass distributions with Wilson 95% confidence intervals across all
evaluated scenarios and controllers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def wilson_score_interval(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate the Wilson score interval for a binomial proportion k/n.

    Args:
        k: Number of successes / positive events (integer >= 0)
        n: Total number of trials / episodes (integer > 0)
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (ci_lower, ci_upper) bounded in [0.0, 1.0]
    """
    if n <= 0:
        return (0.0, 0.0)
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")

    p_hat = k / n
    z_dict = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    z = z_dict.get(round(confidence, 2), 1.959963984540054)

    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denominator
    margin = (z / denominator) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * (n**2))))

    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)

    if k == 0:
        ci_lower = 0.0
    if k == n:
        ci_upper = 1.0

    return (float(ci_lower), float(ci_upper))


def analyze_failures_and_collapses(
    df: pd.DataFrame,
    collapse_biomass_threshold: float = 50.0,
    secondary_biomass_threshold: float = 20.0,
    confidence: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Analyze episode failure and biomass collapse probabilities.

    Args:
        df: Episode results DataFrame containing columns:
            ['controller', 'scenario', 'seed', 'final_biomass',
             'episode_terminated', 'effective_horizon', 'nominal_horizon',
             'termination_reason']
        collapse_biomass_threshold: Primary biomass collapse threshold (g/L), default 50.0
        secondary_biomass_threshold: Secondary low biomass threshold (g/L), default 20.0
        confidence: Confidence level for Wilson CI (default 0.95)

    Returns:
        (scenario_table_df, aggregate_table_df, detailed_json_dict)
    """
    records_scenario: List[Dict[str, Any]] = []
    records_aggregate: List[Dict[str, Any]] = []
    detailed_json: Dict[str, Any] = {
        "metadata": {
            "collapse_biomass_threshold_g_per_l": collapse_biomass_threshold,
            "secondary_biomass_threshold_g_per_l": secondary_biomass_threshold,
            "confidence_level": confidence,
            "statistical_unit": "episode",
        },
        "scenarios": {},
        "aggregates": {},
    }

    controllers = ["no_action", "rule_based_reserve", "pid", "lstm"]
    scenarios = ["normal", "heatwave", "nutrient_depletion", "cold_shock"]

    available_controllers = [c for c in controllers if c in df["controller"].unique()]
    available_scenarios = [s for s in scenarios if s in df["scenario"].unique()]

    # 1. Per Scenario x Controller
    for sc in available_scenarios:
        detailed_json["scenarios"][sc] = {}
        for ctrl in available_controllers:
            sub = df[(df["scenario"] == sc) & (df["controller"] == ctrl)].copy()
            n_episodes = len(sub)
            if n_episodes == 0:
                continue

            failed_episodes = sub[sub["episode_terminated"] == True]
            n_failed = len(failed_episodes)
            p_failure = n_failed / n_episodes
            fail_ci_low, fail_ci_high = wilson_score_interval(n_failed, n_episodes, confidence)

            collapsed_episodes = sub[sub["final_biomass"] < collapse_biomass_threshold]
            n_collapsed = len(collapsed_episodes)
            p_collapse = n_collapsed / n_episodes
            col_ci_low, col_ci_high = wilson_score_interval(n_collapsed, n_episodes, confidence)

            collapsed_20 = sub[sub["final_biomass"] < secondary_biomass_threshold]
            n_collapsed_20 = len(collapsed_20)
            p_collapse_20 = n_collapsed_20 / n_episodes
            col20_ci_low, col20_ci_high = wilson_score_interval(n_collapsed_20, n_episodes, confidence)

            biomass_vals = sub["final_biomass"].values
            b_mean = float(np.mean(biomass_vals))
            b_median = float(np.median(biomass_vals))
            q25 = float(np.percentile(biomass_vals, 25))
            q75 = float(np.percentile(biomass_vals, 75))
            b_iqr = float(q75 - q25)
            b_min = float(np.min(biomass_vals))
            b_max = float(np.max(biomass_vals))

            eff_horizons = sub["effective_horizon"].values
            mean_eff_horizon = float(np.mean(eff_horizons))
            n_early_term = int(np.sum(sub["effective_horizon"] < sub["nominal_horizon"]))

            reason_counts = sub["termination_reason"].value_counts().to_dict()

            rec = {
                "scenario": sc,
                "controller": ctrl,
                "n": n_episodes,
                "failures": n_failed,
                "failure_probability": p_failure,
                "failure_ci_lower": fail_ci_low,
                "failure_ci_upper": fail_ci_high,
                "collapse_count": n_collapsed,
                "collapse_probability": p_collapse,
                "collapse_ci_lower": col_ci_low,
                "collapse_ci_upper": col_ci_high,
                "collapse_20_count": n_collapsed_20,
                "collapse_20_probability": p_collapse_20,
                "collapse_20_ci_lower": col20_ci_low,
                "collapse_20_ci_upper": col20_ci_high,
                "terminal_biomass_mean": b_mean,
                "terminal_biomass_median": b_median,
                "terminal_biomass_iqr": b_iqr,
                "terminal_biomass_min": b_min,
                "terminal_biomass_max": b_max,
                "mean_effective_horizon": mean_eff_horizon,
                "early_terminated_count": n_early_term,
                "termination_reasons": str(reason_counts),
            }
            records_scenario.append(rec)
            detailed_json["scenarios"][sc][ctrl] = rec

    # 2. Aggregates across all episodes per Controller
    for ctrl in available_controllers:
        sub = df[df["controller"] == ctrl].copy()
        n_episodes = len(sub)
        if n_episodes == 0:
            continue

        failed_episodes = sub[sub["episode_terminated"] == True]
        n_failed = len(failed_episodes)
        p_failure = n_failed / n_episodes
        fail_ci_low, fail_ci_high = wilson_score_interval(n_failed, n_episodes, confidence)

        collapsed_episodes = sub[sub["final_biomass"] < collapse_biomass_threshold]
        n_collapsed = len(collapsed_episodes)
        p_collapse = n_collapsed / n_episodes
        col_ci_low, col_ci_high = wilson_score_interval(n_collapsed, n_episodes, confidence)

        collapsed_20 = sub[sub["final_biomass"] < secondary_biomass_threshold]
        n_collapsed_20 = len(collapsed_20)
        p_collapse_20 = n_collapsed_20 / n_episodes
        col20_ci_low, col20_ci_high = wilson_score_interval(n_collapsed_20, n_episodes, confidence)

        biomass_vals = sub["final_biomass"].values
        b_mean = float(np.mean(biomass_vals))
        b_median = float(np.median(biomass_vals))
        q25 = float(np.percentile(biomass_vals, 25))
        q75 = float(np.percentile(biomass_vals, 75))
        b_iqr = float(q75 - q25)
        b_min = float(np.min(biomass_vals))
        b_max = float(np.max(biomass_vals))

        eff_horizons = sub["effective_horizon"].values
        mean_eff_horizon = float(np.mean(eff_horizons))
        n_early_term = int(np.sum(sub["effective_horizon"] < sub["nominal_horizon"]))
        reason_counts = sub["termination_reason"].value_counts().to_dict()

        agg_rec = {
            "controller": ctrl,
            "n_total": n_episodes,
            "total_failures": n_failed,
            "failure_probability": p_failure,
            "failure_ci_lower": fail_ci_low,
            "failure_ci_upper": fail_ci_high,
            "total_collapses": n_collapsed,
            "collapse_probability": p_collapse,
            "collapse_ci_lower": col_ci_low,
            "collapse_ci_upper": col_ci_high,
            "total_collapses_20": n_collapsed_20,
            "collapse_20_probability": p_collapse_20,
            "collapse_20_ci_lower": col20_ci_low,
            "collapse_20_ci_upper": col20_ci_high,
            "terminal_biomass_mean": b_mean,
            "terminal_biomass_median": b_median,
            "terminal_biomass_iqr": b_iqr,
            "terminal_biomass_min": b_min,
            "terminal_biomass_max": b_max,
            "mean_effective_horizon": mean_eff_horizon,
            "early_terminated_count": n_early_term,
            "termination_reasons": str(reason_counts),
        }
        records_aggregate.append(agg_rec)
        detailed_json["aggregates"][ctrl] = agg_rec

    scenario_df = pd.DataFrame(records_scenario)
    aggregate_df = pd.DataFrame(records_aggregate)

    return scenario_df, aggregate_df, detailed_json


def run_and_save_failure_analysis(
    input_csv: Path | str,
    output_dir: Path | str,
    collapse_threshold: float = 50.0,
    secondary_threshold: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute failure analysis and write CSV and JSON reports."""
    input_path = Path(input_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    scenario_df, aggregate_df, detailed_json = analyze_failures_and_collapses(
        df,
        collapse_biomass_threshold=collapse_threshold,
        secondary_biomass_threshold=secondary_threshold,
    )

    scenario_csv = out_dir / "failure_probability_table.csv"
    aggregate_csv = out_dir / "failure_probability_aggregate.csv"
    json_path = out_dir / "failure_probabilities.json"

    scenario_df.to_csv(scenario_csv, index=False)
    aggregate_df.to_csv(aggregate_csv, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_json, f, indent=2)

    return scenario_df, aggregate_df
