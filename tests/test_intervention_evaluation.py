"""
Unit and integration tests for Reviewer-Response Intervention Evaluation Extension.

Covers:
- Canonical 61-action discretization and membership
- Test 1: Always-No-Action baseline properties
- Test 2: Perfect intervention predictions
- Test 3: All-no-action labels (safe NaN/division handling)
- Test 4: No-predicted-interventions edge case
- Test 5: Perfect intervention detection with incorrect magnitude (stage independence)
- Test 6: Scenario stratification correctness
- Test 7: Real held-out test split integrity, label ordering, and no data leakage
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from evaluation.intervention_evaluator import (
    InterventionEvaluator,
    discretize_action_nearest,
    get_canonical_action_space,
    is_intervention,
    is_intervention_vec,
    load_canonical_test_split,
)
from utils import load_config


class TestCanonicalActionSpace:
    """Verifies the 61-action space and nearest-neighbour discretization."""

    def test_action_space_size_and_structure(self):
        actions = get_canonical_action_space()
        assert actions.shape == (61, 2)
        # First action must be (0.0, 0.0)
        assert np.allclose(actions[0], [0.0, 0.0])
        # Remaining 60 actions must have flow > 0 and duration > 0
        nonzeros = actions[1:]
        assert len(nonzeros) == 60
        assert np.all(nonzeros[:, 0] > 0.0)
        assert np.all(nonzeros[:, 1] > 0.0)

    def test_discretization_guarantees_valid_candidate(self):
        action_space = get_canonical_action_space()
        action_set = set(map(tuple, action_space.tolist()))
        
        # Test extreme, out-of-bound, and invalid actions
        test_inputs = np.array([
            [0.5, 0.0],    # Flow without duration (invalid)
            [0.0, 15.0],   # Duration without flow (invalid)
            [5.5, 35.0],   # Above maximum bounds
            [-1.0, -5.0],  # Negative values
            [0.01, 0.02],  # Near zero epsilon
            [2.48, 14.8],  # Near candidate (2.5, 15.0)
        ])
        
        snapped = discretize_action_nearest(test_inputs, action_space)
        assert snapped.shape == (len(test_inputs), 2)
        
        for row in snapped:
            # Rounding to 4 decimal places for floating point tuple lookup
            row_tup = (round(row[0], 4), round(row[1], 4))
            match_found = any(
                math.isclose(row[0], a[0], abs_tol=1e-5) and math.isclose(row[1], a[1], abs_tol=1e-5)
                for a in action_space
            )
            assert match_found, f"Snapped action {row} not found in canonical 61-action space!"


class TestInterventionEvaluatorSynthetic:
    """Tests evaluation behavior on synthetic cases."""

    @pytest.fixture
    def evaluator(self):
        return InterventionEvaluator()

    def test_1_always_no_action_baseline(self, evaluator):
        """
        Test 1 — Always-no-action baseline:
        Construct test labels containing both (0,0) and non-zero actions.
        Verify:
            intervention recall = 0
            intervention precision = 0
            intervention F1 = 0
            non-zero exact accuracy = 0
            overall accuracy = no-action prevalence
        """
        y_true = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, 15.0],
            [3.0, 20.0],
        ])
        res = evaluator.evaluate_baseline_always_no_action(y_true)
        
        assert res["intervention_recall"] == 0.0
        assert res["intervention_precision"] == 0.0
        assert res["intervention_f1"] == 0.0
        assert res["nonzero_exact_action_accuracy"] == 0.0
        assert res["overall_accuracy"] == pytest.approx(3 / 5)
        assert res["true_positives"] == 0
        assert res["false_positives"] == 0
        assert res["true_negatives"] == 3
        assert res["false_negatives"] == 2

    def test_2_perfect_intervention_predictions(self, evaluator):
        """
        Test 2 — Perfect intervention predictions:
        Construct predictions identical to labels.
        Verify:
            intervention precision = 1
            intervention recall = 1
            intervention F1 = 1
            macro F1 = 1
            nonzero exact accuracy = 1
            conditional flow MAE = 0
            conditional duration MAE = 0
        """
        y_true = np.array([
            [0.0, 0.0],
            [1.5, 10.0],
            [0.0, 0.0],
            [4.0, 25.0],
        ])
        y_pred = y_true.copy()
        
        res = evaluator.evaluate(y_true, y_pred)
        assert res["intervention_precision"] == pytest.approx(1.0)
        assert res["intervention_recall"] == pytest.approx(1.0)
        assert res["intervention_f1"] == pytest.approx(1.0)
        assert res["macro_f1"] == pytest.approx(1.0)
        assert res["nonzero_exact_action_accuracy"] == pytest.approx(1.0)
        assert res["overall_exact_action_accuracy"] == pytest.approx(1.0)
        assert res["conditional_flow_mae"] == pytest.approx(0.0)
        assert res["conditional_duration_mae"] == pytest.approx(0.0)

    def test_3_all_no_action_labels(self, evaluator):
        """
        Test 3 — All no-action labels:
        Verify the evaluator handles zero true interventions gracefully (returns NaN/None, no crash).
        """
        y_true = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        y_pred = np.array([
            [0.0, 0.0],
            [1.0, 10.0],
            [0.0, 0.0],
        ])
        res = evaluator.evaluate(y_true, y_pred)
        assert res["total_samples"] == 3
        assert res["true_intervention_count"] == 0
        assert np.isnan(res["conditional_flow_mae"])
        assert np.isnan(res["conditional_duration_mae"])
        assert np.isnan(res["nonzero_exact_action_accuracy"])
        assert res["overall_accuracy"] == pytest.approx(2 / 3)

    def test_4_no_predicted_interventions(self, evaluator):
        """
        Test 4 — No predicted interventions:
        Verify that predicted-intervention conditional metrics do not crash.
        """
        y_true = np.array([
            [0.0, 0.0],
            [2.0, 15.0],
        ])
        y_pred = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        res = evaluator.evaluate(y_true, y_pred)
        assert res["pred_intervention_count"] == 0
        assert np.isnan(res["predicted_intervention_flow_mae"])
        assert np.isnan(res["predicted_intervention_duration_mae"])
        assert res["intervention_recall"] == 0.0

    def test_5_perfect_detection_wrong_magnitude(self, evaluator):
        """
        Test 5 — Perfect intervention detection but wrong magnitude:
        Construct examples where the model correctly identifies non-zero samples,
        but selects a different flow/duration magnitude.
        Verify:
            intervention F1 = 1.0 (Stage 1 is perfect)
            conditional MAE > 0.0 (Stage 2 captures magnitude error)
            nonzero exact accuracy < 1.0
        """
        y_true = np.array([
            [0.0, 0.0],
            [1.0, 10.0],
            [2.0, 20.0],
        ])
        y_pred = np.array([
            [0.0, 0.0],
            [3.0, 25.0],  # Intervened, but wrong magnitude
            [4.0, 30.0],  # Intervened, but wrong magnitude
        ])
        res = evaluator.evaluate(y_true, y_pred)
        assert res["intervention_precision"] == pytest.approx(1.0)
        assert res["intervention_recall"] == pytest.approx(1.0)
        assert res["intervention_f1"] == pytest.approx(1.0)
        assert res["conditional_flow_mae"] > 0.0
        assert res["conditional_duration_mae"] > 0.0
        assert res["nonzero_exact_action_accuracy"] == 0.0

    def test_6_scenario_stratification(self, evaluator):
        """
        Test 6 — Scenario stratification:
        Verify each operating regime receives its correct subset of samples.
        """
        y_true = np.array([
            [0.0, 0.0],
            [2.0, 15.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        y_pred = y_true.copy()
        scenarios = np.array(["normal", "normal", "heatwave", "cold_shock"])
        
        df_scen = evaluator.evaluate_by_regime(
            y_true, y_pred, scenarios,
            target_regimes=["normal", "heatwave", "cold_shock", "nutrient_depletion"],
        )
        assert len(df_scen) == 4
        
        norm_row = df_scen[df_scen["Regime"] == "Normal"].iloc[0]
        assert norm_row["Samples"] == 2
        assert norm_row["Intervention %"] == pytest.approx(50.0)
        assert norm_row["Intervention F1"] == pytest.approx(1.0)
        
        empty_row = df_scen[df_scen["Regime"] == "Nutrient depletion"].iloc[0]
        assert empty_row["Samples"] == 0
        assert np.isnan(empty_row["Intervention F1"])


class TestRealHeldOutTestSplit:
    """
    Test 7 — Real test-set evaluation:
    Verifies data alignment, exact sample counts, label distribution,
    zero data leakage, and uncompromised primary benchmark integrity.
    """

    @pytest.fixture(scope="class")
    def test_split_data(self):
        project_root = Path(__file__).resolve().parent.parent
        config = load_config(project_root / "configs" / "default.yaml")
        split = load_canonical_test_split(config, project_root=project_root)
        return config, split

    def test_7_exact_test_windows_and_alignment(self, test_split_data):
        config, split = test_split_data
        
        # 1. Exact sample count
        n_samples = len(split.X_test)
        assert n_samples == 6981, f"Expected 6,981 test windows, found {n_samples}"
        assert len(split.y_test_norm) == 6981
        assert len(split.y_test_physical) == 6981
        assert len(split.scenarios) == 6981
        assert len(split.trajectory_ids) == 6981
        
        # 2. Label distribution matches naturally imbalanced supervisory policy
        is_zero = ~is_intervention_vec(split.y_test_physical)
        no_action_count = int(np.sum(is_zero))
        intervention_count = int(np.sum(~is_zero))
        
        assert no_action_count == 4927, f"Expected 4,927 no-action samples, found {no_action_count}"
        assert intervention_count == 2054, f"Expected 2,054 intervention samples, found {intervention_count}"
        assert pytest.approx(no_action_count / n_samples, rel=1e-3) == 0.7058
        
        # 3. Verify no overlap between test trajectory IDs and train/val trajectory IDs
        project_root = Path(__file__).resolve().parent.parent
        df_raw = pd.read_csv(project_root / "data" / "processed" / "all_trajectories_labeled.csv")
        from preprocessing.feature_engineering import FeatureEngineer
        from preprocessing.sequence_builder import SequenceBuilder
        
        fe = FeatureEngineer(
            ec_target=config["simulation"]["ec_target"],
            rolling_window=config["preprocessing"]["rolling_window"],
            include_biological=config["preprocessing"]["include_biological_features"],
            internal_capacity=config["dynamics"]["internal_capacity"],
        )
        df_feat = fe.transform(df_raw)
        sb = SequenceBuilder(
            sequence_length=config["preprocessing"]["sequence_length"],
            prediction_horizon=config["preprocessing"]["prediction_horizon"],
            feature_columns=split.feature_cols,
            target_columns=split.target_cols,
        )
        X_all, y_all, traj_ids_all = sb.build_from_dataframe(df_feat)
        splits = sb.train_val_test_split(
            X_all, y_all, traj_ids_all,
            train_ratio=config["preprocessing"]["train_ratio"],
            val_ratio=config["preprocessing"]["val_ratio"],
            seed=config["seed"],
        )
        
        train_traj_ids = set(np.unique(splits["train_ids"]))
        val_traj_ids = set(np.unique(splits["val_ids"]))
        test_traj_ids = set(np.unique(splits["test_ids"]))
        
        # Zero leakage guarantee
        assert len(test_traj_ids.intersection(train_traj_ids)) == 0, "Data leakage: test IDs overlap with train IDs!"
        assert len(test_traj_ids.intersection(val_traj_ids)) == 0, "Data leakage: test IDs overlap with val IDs!"
        assert len(test_traj_ids) == 30, f"Expected 30 test trajectories, found {len(test_traj_ids)}"
        
        # 4. Canonical test split target alignment
        npz_data = np.load(project_root / "data" / "processed" / "sequences.npz")
        assert np.allclose(split.y_test_norm, npz_data["y_test"]), (
            "Mismatch between test_split.y_test_norm and sequences.npz['y_test']!"
        )
        assert np.allclose(split.X_test, npz_data["X_test"]), (
            "Mismatch between test_split.X_test and sequences.npz['X_test']!"
        )

    def test_7_baseline_and_lstm_evaluator_execution(self, test_split_data):
        config, split = test_split_data
        evaluator = InterventionEvaluator()
        
        # Always-no-action baseline on real test set
        base_res = evaluator.evaluate_baseline_always_no_action(split.y_test_physical)
        assert base_res["intervention_recall"] == 0.0
        assert base_res["intervention_precision"] == 0.0
        assert base_res["intervention_f1"] == 0.0
        assert base_res["nonzero_exact_action_accuracy"] == 0.0
        assert base_res["overall_accuracy"] == pytest.approx(4927 / 6981)
        
        # Verify comparison table builds cleanly
        comp_df = evaluator.build_comparison_table(base_res, base_res)
        assert isinstance(comp_df, pd.DataFrame)
        assert len(comp_df) >= 10
