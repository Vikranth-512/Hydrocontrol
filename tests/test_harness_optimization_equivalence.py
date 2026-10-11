"""
Regression test verifying that optimized feature engineering preserves exact
numerical equivalence with the unoptimized implementation for the LSTM policy.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from models.lstm_policy import LSTMPolicy
from preprocessing.normalization import FeatureNormalizer
from simulation_runner.closed_loop_eval import ClosedLoopEvaluator
from simulation.dynamics import TankDynamicsParams, TankState
from simulation.disturbances import DisturbanceGenerator
from utils import load_config

ROOT = Path(__file__).resolve().parent.parent


class TestHarnessOptimizationEquivalence:
    """Verifies that optimized rolling-window feature engineering is identical to unoptimized."""

    @pytest.fixture
    def setup_data(self):
        config = load_config(ROOT / "configs" / "default.yaml")
        evaluator = ClosedLoopEvaluator(config, seed=42)

        feat_cols_path = ROOT / "data" / "processed" / "feature_columns.json"
        with open(feat_cols_path, "r") as f:
            feat_meta = json.load(f)
        feature_cols = feat_meta["features"]

        scalers_dir = ROOT / "data" / "processed" / "scalers"
        normalizer = FeatureNormalizer.load(scalers_dir)

        mcfg = config.get("model", {})
        model = LSTMPolicy(
            input_size=len(feature_cols),
            hidden_size=mcfg.get("hidden_size", 128),
            num_layers=mcfg.get("num_layers", 2),
            dropout=mcfg.get("dropout", 0.2),
            output_size=mcfg.get("output_size", 2),
        )
        ckpt_path = ROOT / "checkpoints" / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()

        return evaluator, model, normalizer, feature_cols

    def test_lstm_unoptimized_vs_optimized_equivalence(self, setup_data):
        evaluator, model, normalizer, feature_cols = setup_data
        test_length = 100
        seed = 42

        # 1. Generate identical disturbance schedule and initial state
        gen = DisturbanceGenerator(evaluator.disturbance_config, np.random.default_rng(seed))
        disturbances = gen.build_schedule("normal", test_length)
        s0_rng = np.random.default_rng(seed)
        s0 = TankState.create_initial(evaluator.params, rng=s0_rng)

        # 2. Run unoptimized (full historical DataFrame at every step)
        res_unoptimized = evaluator.run_learned_policy(
            model=model,
            normalizer=normalizer,
            feature_columns=feature_cols,
            scenario="normal",
            device="cpu",
            custom_length=test_length,
            custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(disturbances),
            prebuilt_initial_state=copy.deepcopy(s0),
            optimize_feature_engineering=False,
        )

        # 3. Run optimized (windowed DataFrame)
        res_optimized = evaluator.run_learned_policy(
            model=model,
            normalizer=normalizer,
            feature_columns=feature_cols,
            scenario="normal",
            device="cpu",
            custom_length=test_length,
            custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(disturbances),
            prebuilt_initial_state=copy.deepcopy(s0),
            optimize_feature_engineering=True,
        )

        traj_unopt = res_unoptimized["trajectory"]
        traj_opt = res_optimized["trajectory"]

        # 4. Strict assertions: Actions must be exactly equal
        np.testing.assert_array_equal(
            traj_unopt["flowrate"],
            traj_opt["flowrate"],
            err_msg="Flowrate actions diverged between unoptimized and optimized evaluation!",
        )
        np.testing.assert_array_equal(
            traj_unopt["duration"],
            traj_opt["duration"],
            err_msg="Duration actions diverged between unoptimized and optimized evaluation!",
        )

        # 5. Sensor traces and states must be exactly equal
        np.testing.assert_allclose(
            traj_unopt["ec"],
            traj_opt["ec"],
            atol=1e-10,
            err_msg="EC trace diverged!",
        )
        np.testing.assert_allclose(
            traj_unopt["biomass"],
            traj_opt["biomass"],
            atol=1e-10,
            err_msg="Biomass trace diverged!",
        )
        np.testing.assert_allclose(
            traj_unopt["reserve_ratio"],
            traj_opt["reserve_ratio"],
            atol=1e-10,
            err_msg="Reserve ratio trace diverged!",
        )

        # 6. All numerical metrics must match within tight tolerance
        m_unopt = res_unoptimized["metrics"]
        m_opt = res_optimized["metrics"]

        for k in m_unopt:
            if isinstance(m_unopt[k], (int, float, np.number)):
                assert np.isclose(m_unopt[k], m_opt[k], atol=1e-8), (
                    f"Metric '{k}' diverged: unopt={m_unopt[k]}, opt={m_opt[k]}"
                )
            else:
                assert m_unopt[k] == m_opt[k], f"Metric '{k}' mismatch"
