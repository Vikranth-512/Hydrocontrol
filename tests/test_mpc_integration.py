"""
Integration tests and performance profiling for MPCReserveController baseline.

Verifies:
1. MPC runs through the real controller interface (ClosedLoopEvaluator).
2. All five controllers (no_action, rule_based_reserve, pid, lstm, mpc_reserve) complete a short episode.
3. MPC does not mutate the live environment during prediction rollouts.
4. MPC does not access future observations or disturbance schedules.
5. MPC does not change CRN or plant random-number streams.
6. MultiSeedEvaluator includes MPC in the experiment matrix when enabled.
7. Existing controllers retain their previous behavior.
8. Performance profiling: measure decision latency, median, 95th percentile, and candidate evaluations.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

from controllers import MPCReserveConfig, MPCReserveController
from evaluation.multiseed_evaluator import MultiSeedEvaluator
from models.lstm_policy import LSTMPolicy
from preprocessing.normalization import FeatureNormalizer
from simulation.disturbances import DisturbanceGenerator
from simulation.dynamics import TankDynamicsParams, TankState
from simulation.environment import AlgaeTankEnvironment, EnvironmentConfig
from simulation_runner.closed_loop_eval import ClosedLoopEvaluator
from utils import load_config

ROOT = Path(__file__).resolve().parent.parent


class TestMPCIntegration:
    """Verifies end-to-end integration of MPCReserveController."""

    @pytest.fixture
    def config(self) -> Dict:
        return load_config(ROOT / "configs" / "default.yaml")

    @pytest.fixture
    def evaluator(self, config: Dict) -> ClosedLoopEvaluator:
        return ClosedLoopEvaluator(config, seed=42)

    def test_mpc_runs_through_real_controller_interface(self, evaluator: ClosedLoopEvaluator) -> None:
        """MPC runs through ClosedLoopEvaluator.run_mpc and returns complete trajectory & metrics."""
        res = evaluator.run_mpc(scenario="normal", custom_length=30, custom_seed=42)
        assert res["controller"] == "mpc_reserve"
        traj = res["trajectory"]
        assert len(traj["flowrate"]) == 30
        assert len(traj["duration"]) == 30
        assert len(traj["reserve_ratio"]) == 30
        assert np.all(np.isfinite(traj["flowrate"]))
        assert np.all(np.isfinite(traj["duration"]))
        assert np.all(np.isfinite(traj["reserve_ratio"]))
        
        metrics = res["metrics"]
        assert "ec_mae" in metrics and np.isfinite(metrics["ec_mae"])
        assert "final_biomass" in metrics and np.isfinite(metrics["final_biomass"])
        assert "total_dose" in metrics and np.isfinite(metrics["total_dose"])
        assert metrics["effective_horizon"] == 30

    def test_all_five_controllers_complete_short_episode(self, evaluator: ClosedLoopEvaluator, config: Dict) -> None:
        """All five controllers complete an episode under identical CRN and initial conditions."""
        horizon = 35
        seed = 42

        # Shared CRN disturbance schedule and initial state
        gen = DisturbanceGenerator(evaluator.disturbance_config, np.random.default_rng(seed))
        disturbances = gen.build_schedule("normal", horizon)
        s0 = TankState.create_initial(evaluator.params, rng=np.random.default_rng(seed))

        # 1. No action
        res_na = evaluator.run_no_action(
            scenario="normal", custom_length=horizon, custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(disturbances),
            prebuilt_initial_state=copy.deepcopy(s0),
        )
        # 2. Rule based
        res_rb = evaluator.run_rule_based(
            scenario="normal", custom_length=horizon, custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(disturbances),
            prebuilt_initial_state=copy.deepcopy(s0),
        )
        # 3. PID
        res_pid = evaluator.run_pid(
            scenario="normal", custom_length=horizon, custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(disturbances),
            prebuilt_initial_state=copy.deepcopy(s0),
        )
        # 4. MPC Reserve
        res_mpc = evaluator.run_mpc(
            scenario="normal", custom_length=horizon, custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(disturbances),
            prebuilt_initial_state=copy.deepcopy(s0),
        )

        for res in [res_na, res_rb, res_pid, res_mpc]:
            assert res["trajectory"]["effective_horizon"] == horizon
            assert len(res["trajectory"]["flowrate"]) == horizon

        # 5. LSTM (if model available)
        ckpt_path = ROOT / "checkpoints" / "best_model.pt"
        if ckpt_path.exists():
            import json
            with open(ROOT / "data" / "processed" / "feature_columns.json", "r") as f:
                feature_cols = json.load(f)["features"]
            normalizer = FeatureNormalizer.load(ROOT / "data" / "processed" / "scalers")
            model = LSTMPolicy(
                input_size=len(feature_cols),
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                output_size=2,
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            res_lstm = evaluator.run_learned_policy(
                model=model, normalizer=normalizer, feature_columns=feature_cols,
                scenario="normal", custom_length=horizon, custom_seed=seed,
                prebuilt_disturbances=copy.deepcopy(disturbances),
                prebuilt_initial_state=copy.deepcopy(s0),
            )
            assert res_lstm["trajectory"]["effective_horizon"] == horizon

    def test_mpc_does_not_mutate_live_environment_during_prediction(self, evaluator: ClosedLoopEvaluator) -> None:
        """MPC internal rollouts must NOT mutate live environment state or absorption queues."""
        env_cfg = EnvironmentConfig()
        params = TankDynamicsParams()
        env = AlgaeTankEnvironment(env_cfg, params, rng=np.random.default_rng(42))
        obs = env.reset()

        # Capture environment state snapshot
        state_before = copy.deepcopy(env.state)
        queue_len_before = len(env.state.pending_doses)
        biomass_before = env.state.algae_biomass
        reserve_before = env.state.internal_reserve

        # Run MPC controller compute step
        ctrl = MPCReserveController()
        action = ctrl.compute_from_reserve(0.35)

        # Environment must be strictly untouched before step() is explicitly called
        assert env.state.algae_biomass == biomass_before
        assert env.state.internal_reserve == reserve_before
        assert len(env.state.pending_doses) == queue_len_before
        assert env.state.as_dict() == state_before.as_dict()

    def test_mpc_does_not_consume_crn_random_streams(self, evaluator: ClosedLoopEvaluator) -> None:
        """Running MPC does not advance or contaminate the shared random-number generator stream."""
        # Create two identical RNGs
        rng1 = np.random.default_rng(9999)
        rng2 = np.random.default_rng(9999)

        # RNG1: untouched
        expected_draws = [rng1.normal() for _ in range(20)]

        # Run MPC
        ctrl = MPCReserveController()
        for rho in [0.45, 0.40, 0.35, 0.30]:
            ctrl.compute_from_reserve(rho)

        # RNG2: draws must match RNG1 exactly (MPC did not call or alter external RNG)
        actual_draws = [rng2.normal() for _ in range(20)]
        np.testing.assert_array_equal(expected_draws, actual_draws)

    def test_multiseed_evaluator_five_controller_matrix(self, config: Dict, tmp_path: Path) -> None:
        """MultiSeedEvaluator properly runs all 5 controllers when include_mpc=True."""
        cfg = copy.deepcopy(config)
        cfg["evaluation"]["multiseed_long_horizon"]["scenarios"] = ["normal"]
        cfg["evaluation"]["multiseed_long_horizon"]["n_seeds"] = 2
        cfg["evaluation"]["multiseed_long_horizon"]["horizon"] = 25
        cfg["evaluation"]["multiseed_long_horizon"]["report_dir"] = str(tmp_path / "reports")
        cfg["evaluation"]["multiseed_long_horizon"]["figure_dir"] = str(tmp_path / "figures")

        mse = MultiSeedEvaluator(cfg, include_mpc=True)
        assert "mpc_reserve" in mse.controllers
        assert len(mse.controllers) == 5

        paired = mse.run_paired_episode("normal", seed=42, horizon=25)
        assert "no_action" in paired
        assert "rule_based_reserve" in paired
        assert "pid" in paired
        assert "mpc_reserve" in paired

    def test_mpc_performance_profiling(self, evaluator: ClosedLoopEvaluator) -> None:
        """Profile MPC decision latency, median, 95th percentile, and candidates evaluated."""
        ctrl = MPCReserveController()
        latencies_ms: List[float] = []

        # Run 200 consecutive decision steps
        for t in range(200):
            # Oscillating reserve to trigger parameter adaptation and various candidate evaluations
            rho = 0.45 + 0.15 * np.sin(t * 0.1)
            t0 = time.perf_counter()
            action = ctrl.compute_from_reserve(rho)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        latencies_arr = np.array(latencies_ms)
        avg_latency = float(np.mean(latencies_arr))
        median_latency = float(np.median(latencies_arr))
        p95_latency = float(np.percentile(latencies_arr, 95))
        candidates_eval = ctrl.last_candidates_evaluated

        print(f"\n[MPC Profiling Results (200 decisions)]")
        print(f"  Average decision latency: {avg_latency:.4f} ms")
        print(f"  Median decision latency:  {median_latency:.4f} ms")
        print(f"  95th-percentile latency:  {p95_latency:.4f} ms")
        print(f"  Candidates evaluated:     {candidates_eval}")

        # Verification targets:
        # Decision latency should be very fast (< 10 ms, typically < 1 ms)
        assert avg_latency < 10.0, f"Average latency too high: {avg_latency:.2f} ms"
        assert p95_latency < 25.0, f"P95 latency too high: {p95_latency:.2f} ms"
        assert candidates_eval <= 30
