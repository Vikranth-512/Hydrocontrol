"""
Automated validation tests for Supervisory Policy Sensitivity Analysis (ORC-04).

Covers:
1. Baseline equivalence: H=100 reproduces the existing labeler policy.
2. Determinism: Fixed seed produces identical representative states, actions, and metrics.
3. Output integrity: All expected CSVs, JSON, reports, and plots are produced with valid content.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from simulation.dynamics import TankDynamicsParams, TankState
from simulation.optimization_labeler import OptimizationLabeler
from simulation.validate.experiments.supervisory_sensitivity import (
    select_representative_states,
    run_supervisory_policy_sensitivity_analysis,
    compute_policy_comparison_metrics,
)
from utils import load_config


@pytest.fixture
def base_config():
    cfg_path = Path("configs/default.yaml")
    assert cfg_path.exists(), f"Configuration file {cfg_path} not found."
    return load_config(cfg_path)


@pytest.fixture
def dataset_path():
    path = Path("data/processed/all_trajectories_labeled.csv")
    assert path.exists(), f"Dataset file {path} not found."
    return path


class TestBaselineEquivalence:
    """Test 1: Baseline equivalence - H=100 reproduces existing labeler."""

    def test_reproduces_existing_labeler_action(self, base_config, dataset_path):
        # Sample 5 states
        sample_df, states = select_representative_states(dataset_path, n_states=5, seed=42)
        labeler = OptimizationLabeler(base_config)

        for i, s in enumerate(states):
            fr, dur, sc = labeler.label_action(s)
            row = sample_df.iloc[i]
            expected_fr = float(row["optimal_flowrate"])
            expected_dur = float(row["optimal_duration"])

            assert fr == expected_fr, f"State {i}: flowrate {fr} != expected {expected_fr}"
            assert dur == expected_dur, f"State {i}: duration {dur} != expected {expected_dur}"

    def test_h100_horizon_agreement_is_100_percent(self, tmp_path, base_config, dataset_path):
        # Run a quick sensitivity run on 5 states
        result = run_supervisory_policy_sensitivity_analysis(
            config=base_config,
            dataset_path=dataset_path,
            n_states=5,
            seed=42,
            output_dir=tmp_path,
            num_workers=1,
        )
        h_df = pd.read_csv(tmp_path / "horizon_sensitivity.csv")
        h100_row = h_df[h_df["horizon"] == 100].iloc[0]
        assert h100_row["label_agreement_with_H100_pct"] == 100.0


class TestDeterminism:
    """Test 2: Determinism - same seed produces identical results."""

    def test_state_selection_determinism(self, dataset_path):
        df1, _ = select_representative_states(dataset_path, n_states=20, seed=42)
        df2, _ = select_representative_states(dataset_path, n_states=20, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_sensitivity_metrics_determinism(self, tmp_path, base_config, dataset_path):
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"

        res1 = run_supervisory_policy_sensitivity_analysis(
            config=base_config,
            dataset_path=dataset_path,
            n_states=6,
            seed=123,
            output_dir=dir1,
            num_workers=1,
        )
        res2 = run_supervisory_policy_sensitivity_analysis(
            config=base_config,
            dataset_path=dataset_path,
            n_states=6,
            seed=123,
            output_dir=dir2,
            num_workers=1,
        )

        df_obj1 = pd.read_csv(dir1 / "objective_weight_sensitivity.csv")
        df_obj2 = pd.read_csv(dir2 / "objective_weight_sensitivity.csv")
        pd.testing.assert_frame_equal(df_obj1, df_obj2)

        df_h1 = pd.read_csv(dir1 / "horizon_sensitivity.csv")
        df_h2 = pd.read_csv(dir2 / "horizon_sensitivity.csv")
        pd.testing.assert_frame_equal(df_h1, df_h2)


class TestOutputIntegrity:
    """Test 3: Output integrity - expected result files are generated and valid."""

    def test_output_files_exist_and_non_empty(self, tmp_path, base_config, dataset_path):
        out_dir = tmp_path / "out"
        res = run_supervisory_policy_sensitivity_analysis(
            config=base_config,
            dataset_path=dataset_path,
            n_states=5,
            seed=42,
            output_dir=out_dir,
            num_workers=1,
        )

        assert res["status"] == "PASS"

        # Check CSVs
        obj_csv = out_dir / "objective_weight_sensitivity.csv"
        h_csv = out_dir / "horizon_sensitivity.csv"
        states_csv = out_dir / "selected_representative_states.csv"
        assert obj_csv.exists() and obj_csv.stat().st_size > 0
        assert h_csv.exists() and h_csv.stat().st_size > 0
        assert states_csv.exists() and states_csv.stat().st_size > 0

        # Check JSON
        summary_json = out_dir / "summary.json"
        assert summary_json.exists()
        with open(summary_json) as f:
            data = json.load(f)
            assert data["experiment_id"] == "ORC-04"
            assert len(data["horizons_tested"]) == 5

        # Check TXT report
        report_txt = out_dir / "supervisory_policy_sensitivity_report.txt"
        assert report_txt.exists() and report_txt.stat().st_size > 0

        # Check PNG Plots
        p1 = out_dir / "objective_weight_sensitivity.png"
        p2 = out_dir / "horizon_sensitivity.png"
        assert p1.exists() and p1.stat().st_size > 0
        assert p2.exists() and p2.stat().st_size > 0
