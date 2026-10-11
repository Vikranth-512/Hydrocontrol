"""
Unit tests for NoActionController and Reserve-Ratio RuleBasedController baselines.
"""

from __future__ import annotations

import numpy as np
import pytest

from controllers.no_action_controller import NoActionController
from controllers.rule_based_controller import RuleBasedConfig, RuleBasedController
from evaluation.intervention_evaluator import get_canonical_action_space
from simulation.dynamics import TankDynamicsParams, TankState


class TestNoActionController:
    """Verifies properties and constraints of the NoActionController baseline."""

    def test_always_returns_zero_action(self):
        ctrl = NoActionController()
        action = ctrl.compute()
        assert action == (0.0, 0.0)
        assert ctrl.compute(1.5, state="dummy", ec=2.0) == (0.0, 0.0)
        assert ctrl.compute_from_reserve(0.1) == (0.0, 0.0)
        assert ctrl() == (0.0, 0.0)

    def test_sequential_3500_steps_zero_action(self):
        ctrl = NoActionController()
        for t in range(3500):
            action = ctrl.compute(timestep=t, random_state=np.random.randn())
            assert action == (0.0, 0.0), f"NoActionController issued non-zero action at step {t}!"

    def test_deterministic(self):
        ctrl1 = NoActionController()
        ctrl2 = NoActionController()
        for _ in range(100):
            assert ctrl1.compute() == ctrl2.compute()

    def test_interface_conformance(self):
        ctrl = NoActionController()
        assert hasattr(ctrl, "reset")
        assert hasattr(ctrl, "compute")
        ctrl.reset()  # Must not raise


class TestRuleBasedController:
    """Verifies properties and threshold logic of the reserve-ratio RuleBasedController."""

    @pytest.fixture
    def controller(self):
        return RuleBasedController()

    @pytest.fixture
    def canonical_actions(self):
        return get_canonical_action_space()

    def test_region_1_sufficient_reserve(self, controller):
        """rho >= rho_target - deadband (>= 0.46) produces (0.0, 0.0)."""
        for rho in [0.46, 0.50, 0.55, 0.80, 1.0]:
            action = controller.compute_from_reserve(rho)
            assert action == (0.0, 0.0), f"Expected (0, 0) at rho={rho}, got {action}"

    def test_deadband_chatter_prevention(self, controller):
        """rho between 0.46 and 0.50 must NOT trigger dosing, preventing chatter near target."""
        for rho in [0.499, 0.48, 0.47, 0.46]:
            action = controller.compute_from_reserve(rho)
            assert action == (0.0, 0.0), f"Deadband violated at rho={rho}: got {action}"

    def test_region_2_moderate_deficit(self, controller):
        """0.20 <= rho < 0.46 produces moderate intervention (2.0 mL/min, 15.0 s)."""
        cfg = controller.config
        expected = (cfg.low_flowrate, cfg.low_duration)  # (2.0, 15.0)
        for rho in [0.20, 0.25, 0.35, 0.45]:
            controller.reset()  # Reset cooldown
            action = controller.compute_from_reserve(rho)
            assert action == expected, f"Expected moderate action {expected} at rho={rho}, got {action}"

    def test_region_3_critical_deficit(self, controller):
        """rho < 0.20 (starvation floor) produces critical intervention (4.0 mL/min, 20.0 s)."""
        cfg = controller.config
        expected = (cfg.critical_flowrate, cfg.critical_duration)  # (4.0, 20.0)
        for rho in [0.0, 0.05, 0.10, 0.199]:
            controller.reset()  # Reset cooldown
            action = controller.compute_from_reserve(rho)
            assert action == expected, f"Expected critical action {expected} at rho={rho}, got {action}"

    def test_action_space_membership(self, controller, canonical_actions):
        """All controller actions must strictly reside within the canonical 61-action discrete space."""
        action_tuples = {tuple(row) for row in canonical_actions}
        test_rhos = np.linspace(0.0, 1.0, 51)
        for rho in test_rhos:
            controller.reset()
            action = controller.compute_from_reserve(rho)
            assert action in action_tuples, f"Action {action} at rho={rho} not in 61-action space!"

    def test_monotonic_dosing_magnitude(self, controller):
        """Increasing deficit must produce monotonically non-decreasing dose volume."""
        rhos = [0.8, 0.5, 0.48, 0.45, 0.30, 0.15, 0.05]
        dose_volumes = []
        for rho in rhos:
            controller.reset()
            fr, dur = controller.compute_from_reserve(rho)
            dose_volumes.append(fr * dur / 60.0)
        # Verify non-decreasing order as rho decreases
        for i in range(len(dose_volumes) - 1):
            assert dose_volumes[i] <= dose_volumes[i + 1]

    def test_cooldown_constraint(self, controller):
        """Consecutive calls within cooldown window (120s = 2 steps of dt=60) must produce (0.0, 0.0)."""
        controller.reset()
        # Step 1: initial dose at depleted reserve
        a1 = controller.compute_from_reserve(0.10, dt=60.0)
        assert a1 == (4.0, 20.0)
        # Step 2: 1 step later (tsld=60s < 120s) -> cooldown active
        a2 = controller.compute_from_reserve(0.10, dt=60.0)
        assert a2 == (0.0, 0.0), f"Cooldown violated: issued {a2} at step 2"
        # Step 3: 2 steps later (tsld still 60s at start of step 3 < 120s) -> cooldown active
        a3 = controller.compute_from_reserve(0.10, dt=60.0)
        assert a3 == (0.0, 0.0), f"Cooldown violated: issued {a3} at step 3"
        # Step 4: 3 steps later (tsld=120s at start of step 4 >= 120s) -> cooldown elapsed
        a4 = controller.compute_from_reserve(0.10, dt=60.0)
        assert a4 == (4.0, 20.0), f"Cooldown should have elapsed at step 4: got {a4}"

    def test_compute_from_state(self, controller):
        """Test compute_from_state with mock TankState."""
        params = TankDynamicsParams()
        state = TankState.create_initial(params)
        # Healthy initial state: reserve ratio is typically around 0.2
        action = controller.compute_from_state(state, internal_capacity=0.5)
        assert isinstance(action, tuple)
        assert len(action) == 2
        assert action[0] >= 0.0 and action[1] >= 0.0

    def test_threshold_sensitivity_audit(self):
        """Verify qualitative behavior across reasonable threshold perturbations."""
        perturbations = [
            (0.48, 0.22),
            (0.46, 0.20),
            (0.44, 0.18),
        ]
        for rho_low, rho_crit in perturbations:
            cfg = RuleBasedConfig(rho_target=0.50, deadband=0.50 - rho_low, critical_low=rho_crit)
            ctrl = RuleBasedController(cfg)
            # Above upper threshold -> zero
            assert ctrl.compute_from_reserve(rho_low + 0.05) == (0.0, 0.0)
            # Between thresholds -> moderate
            ctrl.reset()
            mid = (rho_low + rho_crit) / 2.0
            assert ctrl.compute_from_reserve(mid) == (2.0, 15.0)
            # Below critical -> critical
            ctrl.reset()
            assert ctrl.compute_from_reserve(rho_crit - 0.05) == (4.0, 20.0)
