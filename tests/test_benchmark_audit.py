"""
Unit tests for benchmark audit utilities: failure analysis, reserve-rule audit, and scenario inventory.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from evaluation.failure_analysis import (
    analyze_failures_and_collapses,
    wilson_score_interval,
)
from evaluation.reserve_rule_audit import get_controller_information_table
from evaluation.scenario_inventory import generate_scenario_inventory


class TestWilsonScoreInterval:
    """Test Wilson binomial score confidence intervals."""

    def test_zero_failure_case(self) -> None:
        low, high = wilson_score_interval(0, 30, confidence=0.95)
        assert low == 0.0
        assert 0.0 < high < 0.15
        assert high <= 1.0

    def test_all_failure_case(self) -> None:
        low, high = wilson_score_interval(30, 30, confidence=0.95)
        assert 0.85 < low < 1.0
        assert math.isclose(high, 1.0, abs_tol=1e-5)

    def test_mixed_failure_case(self) -> None:
        low, high = wilson_score_interval(6, 30, confidence=0.95)
        p_hat = 6 / 30  # 0.20
        assert 0.0 <= low < p_hat
        assert p_hat < high <= 1.0
        assert math.isclose((low + high) / 2.0, 0.234, abs_tol=0.05)

    def test_bounds_guarantee(self) -> None:
        for k in range(0, 31):
            low, high = wilson_score_interval(k, 30)
            assert 0.0 <= low <= 1.0
            assert 0.0 <= high <= 1.0
            assert low <= high

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            wilson_score_interval(-1, 30)
        with pytest.raises(ValueError):
            wilson_score_interval(35, 30)


class TestFailureAnalysis:
    """Test failure and collapse probability statistical grouping and metrics."""

    @pytest.fixture
    def mock_results_df(self) -> pd.DataFrame:
        """Create a synthetic episode results DataFrame with known properties."""
        rows = []
        controllers = ["no_action", "rule_based_reserve", "pid", "lstm"]
        scenarios = ["normal", "cold_shock"]

        for sc in scenarios:
            for ctrl in controllers:
                for seed in range(30):
                    # In normal: rule_based has 6 collapses out of 30, pid has 0, no_action has 30
                    if ctrl == "no_action":
                        b = 6.0
                        term = False
                    elif ctrl == "rule_based_reserve":
                        b = 0.95 if (sc == "normal" and seed < 6) or (sc == "cold_shock" and seed < 21) else 275.0
                        term = False
                    elif ctrl == "pid":
                        b = 216.0
                        term = False
                    else:  # lstm
                        b = 140.0
                        term = False

                    rows.append({
                        "scenario": sc,
                        "controller": ctrl,
                        "seed": seed,
                        "final_biomass": b,
                        "episode_terminated": term,
                        "effective_horizon": 3500,
                        "nominal_horizon": 3500,
                        "termination_reason": "completed",
                    })
        return pd.DataFrame(rows)

    def test_statistical_unit_is_episode(self, mock_results_df: pd.DataFrame) -> None:
        sc_df, agg_df, detailed = analyze_failures_and_collapses(mock_results_df)

        # Check total rows and N
        for _, row in sc_df.iterrows():
            assert row["n"] == 30, "Scenario-controller N must strictly equal 30 episodes"

        for _, row in agg_df.iterrows():
            assert row["n_total"] == 60, "Aggregate N must strictly equal 60 episodes across 2 scenarios"

    def test_correct_collapse_counts(self, mock_results_df: pd.DataFrame) -> None:
        sc_df, agg_df, detailed = analyze_failures_and_collapses(mock_results_df, collapse_biomass_threshold=50.0)

        # Normal rule_based should have 6 collapses
        rb_norm = sc_df[(sc_df["scenario"] == "normal") & (sc_df["controller"] == "rule_based_reserve")].iloc[0]
        assert rb_norm["collapse_count"] == 6
        assert math.isclose(rb_norm["collapse_probability"], 0.20, abs_tol=1e-5)

        # Cold shock rule_based should have 21 collapses
        rb_cold = sc_df[(sc_df["scenario"] == "cold_shock") & (sc_df["controller"] == "rule_based_reserve")].iloc[0]
        assert rb_cold["collapse_count"] == 21
        assert math.isclose(rb_cold["collapse_probability"], 0.70, abs_tol=1e-5)

        # PID should have 0 collapses
        pid_norm = sc_df[(sc_df["scenario"] == "normal") & (sc_df["controller"] == "pid")].iloc[0]
        assert pid_norm["collapse_count"] == 0
        assert pid_norm["collapse_probability"] == 0.0

        # No action should have 30 collapses
        na_norm = sc_df[(sc_df["scenario"] == "normal") & (sc_df["controller"] == "no_action")].iloc[0]
        assert na_norm["collapse_count"] == 30
        assert na_norm["collapse_probability"] == 1.0


class TestScenarioInventory:
    """Test scenario discovery and evaluation coverage matrix."""

    def test_scenario_inventory_completeness(self) -> None:
        inv_df, cov_df, detailed = generate_scenario_inventory()

        assert len(inv_df) == 12, f"Expected 12 identified operational regimes, found {len(inv_df)}"
        assert "normal" in inv_df["scenario_name"].values
        assert "heatwave" in inv_df["scenario_name"].values
        assert "nutrient_depletion" in inv_df["scenario_name"].values
        assert "cold_shock" in inv_df["scenario_name"].values
        assert "noisy_sensors" in inv_df["scenario_name"].values
        assert "delayed_response" in inv_df["scenario_name"].values
        assert "actuator_failure" in inv_df["scenario_name"].values

    def test_evaluation_coverage_proportions(self) -> None:
        inv_df, cov_df, detailed = generate_scenario_inventory()

        evaluated = inv_df[inv_df["used_in_3500_benchmark"] == True]
        assert len(evaluated) == 4, "Expected exactly 4 scenarios evaluated in 3500-step benchmark"

        unevaluated = inv_df[inv_df["used_in_3500_benchmark"] == False]
        assert len(unevaluated) == 8, "Expected exactly 8 unevaluated operational regimes"


class TestControllerInformationAccess:
    """Test controller information table accessibility properties."""

    def test_information_table_entries(self) -> None:
        df = get_controller_information_table()
        assert len(df) == 4
        controllers = df["Controller"].values
        assert "no_action" in controllers
        assert "rule_based_reserve" in controllers
        assert "pid" in controllers
        assert "lstm" in controllers

        # Check information asymmetry
        rb_row = df[df["Controller"] == "rule_based_reserve"].iloc[0]
        assert "privileged" in rb_row["Information Asymmetry / Latent State Access"].lower() or "direct access" in rb_row["Information Asymmetry / Latent State Access"].lower()

        lstm_row = df[df["Controller"] == "lstm"].iloc[0]
        assert "partial observability" in lstm_row["Information Asymmetry / Latent State Access"].lower()
