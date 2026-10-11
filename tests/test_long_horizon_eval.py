"""
Integration tests for long-horizon evaluation pipeline with all controller baselines.
"""

from __future__ import annotations

import copy
from pathlib import Path
import numpy as np
import pytest

from simulation_runner.closed_loop_eval import ClosedLoopEvaluator
from simulation.dynamics import TankDynamicsParams, TankState
from simulation.disturbances import DisturbanceGenerator
from utils import load_config

ROOT = Path(__file__).resolve().parent.parent


class TestLongHorizonEvaluationIntegration:
    """Verifies that all controllers run through the closed-loop evaluation framework."""

    @pytest.fixture
    def config(self):
        return load_config(ROOT / "configs" / "default.yaml")

    @pytest.fixture
    def evaluator(self, config):
        return ClosedLoopEvaluator(config, seed=42)

    def test_controllers_initialization_and_rollout(self, evaluator):
        """Verify No-action, Rule-based, and PID rollouts complete on reduced horizon."""
        test_length = 60
        
        # 1. No-action
        res_no_action = evaluator.run_no_action(scenario="normal", custom_length=test_length, custom_seed=42)
        assert res_no_action["controller"] == "no_action"
        traj_na = res_no_action["trajectory"]
        assert len(traj_na["flowrate"]) == test_length
        assert np.all(traj_na["flowrate"] == 0.0)
        assert np.all(traj_na["duration"] == 0.0)
        assert res_no_action["metrics"]["total_dose"] == 0.0
        assert res_no_action["metrics"]["n_interventions"] == 0
        assert res_no_action["metrics"]["intervention_fraction"] == 0.0

        # 2. Rule-based
        res_rule = evaluator.run_rule_based(scenario="normal", custom_length=test_length, custom_seed=42)
        assert res_rule["controller"] == "rule_based_reserve"
        traj_rb = res_rule["trajectory"]
        assert len(traj_rb["flowrate"]) == test_length
        assert np.all(np.isfinite(traj_rb["flowrate"]))
        assert np.all(np.isfinite(traj_rb["duration"]))

        # 3. PID
        res_pid = evaluator.run_pid(scenario="normal", custom_length=test_length, custom_seed=42)
        assert res_pid["controller"] == "pid"
        traj_pid = res_pid["trajectory"]
        assert len(traj_pid["flowrate"]) == test_length
        assert np.all(np.isfinite(traj_pid["flowrate"]))
        assert np.all(np.isfinite(traj_pid["duration"]))

        # 4. MPC Reserve
        res_mpc = evaluator.run_mpc(scenario="normal", custom_length=test_length, custom_seed=42)
        assert res_mpc["controller"] == "mpc_reserve"
        traj_mpc = res_mpc["trajectory"]
        assert len(traj_mpc["flowrate"]) == test_length
        assert np.all(np.isfinite(traj_mpc["flowrate"]))
        assert np.all(np.isfinite(traj_mpc["duration"]))
        assert "ec_mae" in res_mpc["metrics"]

    def test_paired_crn_disturbances_and_initial_state(self, evaluator, config):
        """Verify exact Common Random Numbers (CRN) disturbance pairing across controllers."""
        test_length = 50
        seed = 42

        # Build schedule and initial state
        gen = DisturbanceGenerator(evaluator.disturbance_config, np.random.default_rng(seed))
        disturbances = gen.build_schedule("normal", test_length)
        s0 = TankState.create_initial(evaluator.params, rng=np.random.default_rng(seed))

        # Run No-action and Rule-based with paired inputs
        res_na = evaluator.run_no_action(
            scenario="normal",
            custom_length=test_length,
            custom_seed=seed,
            prebuilt_disturbances=disturbances,
            prebuilt_initial_state=copy.deepcopy(s0),
        )
        res_rb = evaluator.run_rule_based(
            scenario="normal",
            custom_length=test_length,
            custom_seed=seed,
            prebuilt_disturbances=disturbances,
            prebuilt_initial_state=copy.deepcopy(s0),
        )

        # Initial conditions of both rollouts must be strictly identical
        t0_na = res_na["trajectory"]["state_trace"][0]
        t0_rb = res_rb["trajectory"]["state_trace"][0]
        assert t0_na.algae_biomass == t0_rb.algae_biomass
        assert t0_na.internal_reserve == t0_rb.internal_reserve
        assert t0_na.dissolved_nutrient_mass == t0_rb.dissolved_nutrient_mass
        assert t0_na.water_temp == t0_rb.water_temp

    def test_metrics_schema_completeness(self, evaluator):
        """Verify all required regulation, biological, nutrient, and horizon metrics are present."""
        res = evaluator.run_rule_based(scenario="normal", custom_length=40, custom_seed=42)
        m = res["metrics"]

        # Regulation
        assert "ec_mae" in m and np.isfinite(m["ec_mae"])
        assert "ec_rmse" in m and np.isfinite(m["ec_rmse"])
        assert "final_ec" in m and np.isfinite(m["final_ec"])
        assert "final_ec_deviation" in m and np.isfinite(m["final_ec_deviation"])

        # Biological state
        assert "final_biomass" in m and np.isfinite(m["final_biomass"])
        assert "biomass_change" in m and np.isfinite(m["biomass_change"])
        assert "final_reserve_ratio" in m and np.isfinite(m["final_reserve_ratio"])
        assert "final_health" in m and np.isfinite(m["final_health"])

        # Nutrient behavior
        assert "dissolved_nutrient_mean" in m and np.isfinite(m["dissolved_nutrient_mean"])
        assert "final_dissolved_nutrient" in m and np.isfinite(m["final_dissolved_nutrient"])
        assert "nutrient_accumulation" in m and np.isfinite(m["nutrient_accumulation"])

        # Control effort
        assert "n_interventions" in m and isinstance(m["n_interventions"], (int, np.integer))
        assert "intervention_fraction" in m and 0.0 <= m["intervention_fraction"] <= 1.0

        # Horizon integrity
        assert "episode_terminated" in m
        assert "effective_horizon" in m and m["effective_horizon"] == 40
        assert "termination_reason" in m
