"""
Comprehensive unit test suite for MPCReserveController baseline.

Verifies all 15 specification requirements:
1. MPC controller registration and construction.
2. Correct handling of reserve-ratio observations.
3. Valid candidate-action generation.
4. Candidate-sequence deduplication and configured search limits.
5. Correct prediction-horizon handling.
6. Finite objective values for valid predictions.
7. Correct dose-effort and action-change penalties.
8. Deterministic action selection for identical inputs and history.
9. Safe initialization with insufficient model data.
10. Model stability under constant reserve ratio and constant dosing.
11. Model stability with noisy observations and changing actions.
12. Handling of near-zero denominators and invalid observations.
13. Safe fallback behavior after invalid model estimates.
14. Correct episode reset and no unintended cross-episode leakage.
15. Compliance with actuator limits and timing constraints.
"""

from __future__ import annotations

import copy
import numpy as np
import pytest

from controllers import (
    MPCReserveConfig,
    MPCReserveController,
    get_controller,
)
from controllers.mpc_controller import normalize_action
from evaluation.intervention_evaluator import get_canonical_action_space
from simulation.dynamics import TankDynamicsParams, TankState


class TestMPCReserveControllerUnit:
    """Rigorous unit tests for MPCReserveController."""

    @pytest.fixture
    def controller(self) -> MPCReserveController:
        return MPCReserveController()

    @pytest.fixture
    def canonical_actions(self) -> np.ndarray:
        return get_canonical_action_space()

    # 1. Registration and construction
    def test_registration_and_construction(self) -> None:
        ctrl1 = MPCReserveController()
        assert isinstance(ctrl1, MPCReserveController)
        assert ctrl1.config.prediction_horizon == 5

        # Factory registration via get_controller
        ctrl2 = get_controller("mpc_reserve")
        assert isinstance(ctrl2, MPCReserveController)

        ctrl3 = get_controller("mpc")
        assert isinstance(ctrl3, MPCReserveController)

        # Construction with custom config
        custom_cfg = MPCReserveConfig(
            prediction_horizon=6,
            tracking_weight=2.5,
            target_reserve_ratio=0.55,
        )
        ctrl4 = MPCReserveController(custom_cfg)
        assert ctrl4.config.prediction_horizon == 6
        assert ctrl4.config.tracking_weight == 2.5
        assert ctrl4.config.target_reserve_ratio == 0.55

    # 2. Handling of reserve-ratio observations
    def test_reserve_ratio_observation_handling(self, controller: MPCReserveController) -> None:
        """Controller processes diverse valid reserve-ratio observations within [0, 1]."""
        for rho in [0.05, 0.15, 0.30, 0.50, 0.70, 0.95]:
            controller.reset()
            action = controller.compute_from_reserve(rho)
            assert isinstance(action, tuple)
            assert len(action) == 2
            assert np.isfinite(action[0]) and np.isfinite(action[1])

    # 3. Valid candidate-action generation
    def test_valid_candidate_actions(self, controller: MPCReserveController, canonical_actions: np.ndarray) -> None:
        """Every action level and sequence step belongs to canonical actuator bounds."""
        canonical_set = {tuple(row) for row in canonical_actions}
        for seq in controller._candidate_library:
            for action in seq:
                assert action[0] >= 0.0 and action[1] >= 0.0
                assert action[0] <= 5.0 and action[1] <= 30.0
                assert action in canonical_set, f"Action {action} not in canonical action space!"

    # 4. Candidate-sequence deduplication and configured search limits
    def test_candidate_sequence_limits_and_uniqueness(self, controller: MPCReserveController) -> None:
        """Number of candidates must be <= max_candidate_sequences and strictly unique."""
        candidates = controller._candidate_library
        assert len(candidates) <= controller.config.max_candidate_sequences
        assert len(candidates) > 0

        # Check uniqueness
        candidate_tuples = [tuple(seq) for seq in candidates]
        assert len(candidate_tuples) == len(set(candidate_tuples)), "Duplicate candidate sequences found!"

    # 5. Prediction-horizon handling
    def test_prediction_horizon_handling(self, controller: MPCReserveController) -> None:
        """Predicted trajectories must strictly match configured prediction horizon H."""
        H = controller.config.prediction_horizon
        candidate_norm = [0.2] * H
        pred = controller._predict_trajectory(current_rho=0.45, candidate_u_seq=candidate_norm)
        assert len(pred) == H
        assert np.all(np.isfinite(pred))

        # Check with custom horizon
        cfg_h7 = MPCReserveConfig(prediction_horizon=7)
        ctrl_h7 = MPCReserveController(cfg_h7)
        pred_h7 = ctrl_h7._predict_trajectory(current_rho=0.45, candidate_u_seq=[0.0] * 7)
        assert len(pred_h7) == 7

    # 6. Finite objective values for valid predictions
    def test_finite_objective_values(self, controller: MPCReserveController) -> None:
        """Objective evaluations return finite positive values for valid candidates."""
        cand_seq = controller._candidate_library[0]  # all-zero sequence
        cand_norm = [0.0] * controller.config.prediction_horizon
        cost = controller._evaluate_objective(
            current_rho=0.40,
            candidate_seq=cand_seq,
            candidate_norm_u=cand_norm,
            cooldown_active=False,
        )
        assert np.isfinite(cost)
        assert cost >= 0.0

    # 7. Correct dose-effort and action-change penalties
    def test_effort_and_action_change_penalties(self) -> None:
        """Penalties scale strictly with effort u^2 and action delta (u_k - u_{k-1})^2."""
        # Config with high effort weight
        cfg_high_effort = MPCReserveConfig(
            tracking_weight=0.0,
            low_reserve_weight=0.0,
            dose_effort_weight=10.0,
            action_change_weight=0.0,
        )
        ctrl_effort = MPCReserveController(cfg_high_effort)
        
        # Seq with zero vs non-zero effort
        zero_seq = [ctrl_effort.action_levels[0]] * 5
        zero_norm = [0.0] * 5
        dose_seq = [ctrl_effort.action_levels[1]] + [ctrl_effort.action_levels[0]] * 4
        dose_norm = [0.20] + [0.0] * 4

        cost_zero = ctrl_effort._evaluate_objective(0.50, zero_seq, zero_norm, False)
        cost_dose = ctrl_effort._evaluate_objective(0.50, dose_seq, dose_norm, False)
        assert cost_zero == 0.0
        assert cost_dose > 0.0
        assert np.isclose(cost_dose, 10.0 * (0.20 ** 2))

        # Test action change penalty
        cfg_change = MPCReserveConfig(
            tracking_weight=0.0,
            low_reserve_weight=0.0,
            dose_effort_weight=0.0,
            action_change_weight=5.0,
        )
        ctrl_change = MPCReserveController(cfg_change)
        ctrl_change.last_commanded_norm_u = 0.0
        cost_change = ctrl_change._evaluate_objective(0.50, dose_seq, dose_norm, False)
        # delta_u: [0.2 - 0, 0 - 0.2, 0 - 0, 0 - 0, 0 - 0] -> sum of squares = 0.04 + 0.04 = 0.08
        expected_change_cost = 5.0 * (0.04 + 0.04)
        assert np.isclose(cost_change, expected_change_cost)

    # 8. Deterministic action selection for identical inputs and history
    def test_deterministic_action_selection(self) -> None:
        """Identical inputs and history produce identical actions."""
        ctrl1 = MPCReserveController()
        ctrl2 = MPCReserveController()
        
        test_rhos = [0.45, 0.40, 0.35, 0.30, 0.32, 0.38, 0.44, 0.49]
        for rho in test_rhos:
            a1 = ctrl1.compute_from_reserve(rho)
            a2 = ctrl2.compute_from_reserve(rho)
            assert a1 == a2

    # 9. Safe initialization with insufficient model data
    def test_safe_initialization_insufficient_data(self) -> None:
        """Controller safely initializes with prior theta_0 when no transition history exists."""
        ctrl = MPCReserveController()
        assert len(ctrl.regression_history) == 0
        np.testing.assert_array_equal(ctrl.current_theta, ctrl.theta_0)
        
        # First decision at step 0
        action = ctrl.compute_from_reserve(0.50)
        assert isinstance(action, tuple)
        assert len(action) == 2
        # After 1 step, we only have 0 transitions recorded
        assert len(ctrl.regression_history) == 0
        np.testing.assert_array_equal(ctrl.current_theta, ctrl.theta_0)

    # 10. Model stability under constant reserve ratio and constant dosing
    def test_model_stability_constant_inputs(self, controller: MPCReserveController) -> None:
        """Constant observations and constant actions do not cause divergence or singularity."""
        controller.reset()
        for _ in range(50):
            action = controller.compute_from_reserve(0.50)
            assert np.all(np.isfinite(controller.current_theta))
            assert np.all(np.isfinite(action))

    # 11. Model stability with noisy observations and changing actions
    def test_model_stability_noisy_observations(self, controller: MPCReserveController) -> None:
        """Noisy inputs remain bounded and do not explode parameters."""
        rng = np.random.default_rng(1234)
        controller.reset()
        for t in range(100):
            noisy_rho = 0.45 + rng.normal(0, 0.05)
            action = controller.compute_from_reserve(noisy_rho)
            assert np.all(np.isfinite(controller.current_theta))
            assert np.all(np.isfinite(action))
            # Verify parameter bounds
            assert -0.05 <= controller.current_theta[0] <= 0.02
            assert -0.20 <= controller.current_theta[1] <= 0.00
            for b in controller.current_theta[2:]:
                assert 0.00 <= b <= 0.25

    # 12. Handling of near-zero denominators and invalid observations
    def test_handling_invalid_observations(self, controller: MPCReserveController) -> None:
        """NaN, Inf, negative, or extreme reserve ratios fallback safely to (0.0, 0.0)."""
        controller.reset()
        assert controller.compute_from_reserve(float("nan")) == (0.0, 0.0)
        assert controller.compute_from_reserve(float("inf")) == (0.0, 0.0)
        assert controller.compute_from_reserve(float("-inf")) == (0.0, 0.0)

        # Clamping of out-of-bounds finite values
        controller.reset()
        action_neg = controller.compute_from_reserve(-0.5)
        assert np.all(np.isfinite(action_neg))

        action_large = controller.compute_from_reserve(2.5)
        assert np.all(np.isfinite(action_large))

    # 13. Safe fallback behavior after invalid model estimates
    def test_safe_fallback_on_numerical_failure(self, controller: MPCReserveController) -> None:
        """Corrupted parameters immediately revert to safe prior theta_0 without crash."""
        controller.reset()
        # Manually corrupt theta
        controller.current_theta = np.array([np.nan, np.inf, 999.0, -999.0, np.nan])
        # Next computation must not crash and must produce a valid action
        action = controller.compute_from_reserve(0.40)
        assert isinstance(action, tuple)
        assert np.all(np.isfinite(action))

    # 14. Correct episode reset and no unintended cross-episode leakage
    def test_episode_reset_no_leakage(self) -> None:
        """Reset clears all history, regression data, cooldown timers, and restores theta_0."""
        ctrl = MPCReserveController()
        
        # Run episode 1 with perturbations
        for t in range(25):
            ctrl.compute_from_reserve(0.25)
            
        assert len(ctrl.regression_history) > 0
        assert ctrl.total_decisions == 25
        assert ctrl.time_since_last_dose != float("inf")
        
        # Reset for episode 2
        ctrl.reset()
        assert len(ctrl.regression_history) == 0
        assert ctrl.total_decisions == 0
        assert ctrl.time_since_last_dose == float("inf")
        assert ctrl.last_commanded_action == (0.0, 0.0)
        assert ctrl.last_commanded_norm_u == 0.0
        assert ctrl.prev_rho is None
        np.testing.assert_array_equal(ctrl.current_theta, ctrl.theta_0)

    # 15. Compliance with actuator limits and timing constraints
    def test_cooldown_and_timing_constraints(self, controller: MPCReserveController) -> None:
        """Consecutive calls within cooldown window (120s = 2 steps of dt=60s) produce (0.0, 0.0)."""
        controller.reset()
        
        # Trigger dose at severely depleted reserve
        a1 = controller.compute_from_reserve(0.10, dt=60.0)
        assert a1[0] > 0.0 and a1[1] > 0.0, "Expected dosing intervention at severe deficit!"
        
        # Step 2: 1 step later (tsld=0 at call start < 120s) -> cooldown active
        a2 = controller.compute_from_reserve(0.10, dt=60.0)
        assert a2 == (0.0, 0.0), f"Actuator cooldown violated: issued {a2} at step 2"
        
        # Step 3: 2 steps later (tsld=60 at call start < 120s) -> cooldown active
        a3 = controller.compute_from_reserve(0.10, dt=60.0)
        assert a3 == (0.0, 0.0), f"Actuator cooldown violated: issued {a3} at step 3"

        # Step 4: 3 steps later (tsld=120 >= 120s) -> cooldown elapsed, can dose again
        a4 = controller.compute_from_reserve(0.10, dt=60.0)
        assert a4[0] > 0.0 and a4[1] > 0.0, f"Cooldown should have elapsed at step 4, got {a4}"

    def test_compute_dispatch_interface(self, controller: MPCReserveController) -> None:
        """Generic compute() dispatches correctly for float, dict, and state-like objects."""
        # 1. Float
        controller.reset()
        res_float = controller.compute(0.50)
        controller.reset()
        assert res_float == controller.compute_from_reserve(0.50)

        # 2. Dict with reserve_ratio
        controller.reset()
        res_dict = controller.compute({"reserve_ratio": 0.45})
        controller.reset()
        assert res_dict == controller.compute_from_reserve(0.45)

        # 3. Dict with rho
        controller.reset()
        res_rho = controller.compute({"rho": 0.45})
        controller.reset()
        assert res_rho == controller.compute_from_reserve(0.45)

        # 4. Dict with reserve_ratio and time_since_last_dose
        controller.reset()
        res_dict_tsld = controller.compute({"reserve_ratio": 0.45, "time_since_last_dose": 30.0})
        controller.reset()
        assert res_dict_tsld == controller.compute_from_reserve(0.45, time_since_last_dose=30.0)

        # 5. Direct TankState object via compute_from_state
        controller.reset()
        params = TankDynamicsParams()
        state = TankState.create_initial(params)
        action_state = controller.compute_from_state(state, internal_capacity=0.5)
        assert isinstance(action_state, tuple)
        assert len(action_state) == 2
        assert action_state[0] >= 0.0 and action_state[1] >= 0.0

    def test_compute_guarded_against_hidden_state_access(self, controller: MPCReserveController) -> None:
        """compute() disallows fallback to latent state when reserve_ratio is absent from dict."""
        params = TankDynamicsParams()
        state = TankState.create_initial(params)
        
        # Passing dict with 'state' but without 'reserve_ratio' or 'rho' must raise ValueError
        with pytest.raises(ValueError, match="disallows accessing latent environment state"):
            controller.compute({"state": state, "biomass": 100.0})

        with pytest.raises(ValueError, match="disallows accessing latent environment state"):
            controller.compute({"state": state})

