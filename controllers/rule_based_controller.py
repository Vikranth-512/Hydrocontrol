"""
Reserve-ratio rule-based controller for algae nutrient regulation.

A deterministic heuristic baseline regulating intracellular nutrient reserves
toward the Droop half-quota setpoint.

Threshold and action provenance:
  - rho_target = 0.50: Reuses ecosystem_control.rho_setpoint (Droop half-quota).
  - deadband = 0.04: Reuses pid_behavior.deadband to provide a consistent deadband
    around the established setpoint and prevent boundary chatter. Dosing triggers
    when rho < (rho_target - deadband) = 0.46.
  - critical_low = 0.20: Reuses ecosystem_control.reserve_floor (starvation boundary,
    buffer ~100 steps before maintenance failure).
  - Actions:
      Moderate deficit (0.20 <= rho < 0.46) -> (2.0 mL/min, 15.0 s)
      Critical deficit (rho < 0.20)         -> (4.0 mL/min, 20.0 s)
      Sufficient reserve (rho >= 0.46)      -> (0.0 mL/min, 0.0 s)
    Reused directly from the original RuleBasedConfig parameter definitions.

Units:
  flowrate: mL/min
  duration: s
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from simulation.state_utils import compute_rho


@dataclass
class RuleBasedConfig:
    """
    Configuration for reserve-ratio rule-based heuristic controller.
    
    All defaults inherit grounded parameters from configs/default.yaml and
    historical controller configurations.
    """

    rho_target: float = 0.50             # Reference operating point (Droop half-quota)
    deadband: float = 0.04               # Deadband to suppress chattering near target
    critical_low: float = 0.20           # Starvation boundary from ecosystem_control
    low_flowrate: float = 2.0            # Moderate intervention flowrate (mL/min)
    low_duration: float = 15.0           # Moderate intervention duration (s)
    critical_flowrate: float = 4.0       # Severe intervention flowrate (mL/min)
    critical_duration: float = 20.0      # Severe intervention duration (s)
    high_flowrate: float = 0.0           # No-action flowrate (mL/min)
    high_duration: float = 0.0           # No-action duration (s)
    min_time_between_doses: float = 120.0  # Actuator cooldown (s)

    # Legacy EC-tracking thresholds (preserved for backward compatibility)
    ec_low_threshold: float = 0.9
    ec_high_threshold: float = 1.4
    ec_target: float = 1.2
    ec_critical_low: float = 0.6

    @property
    def rho_low_threshold(self) -> float:
        """Trigger threshold for moderate dosing intervention."""
        return self.rho_target - self.deadband


class RuleBasedController:
    """
    Heuristic rule-based baseline controller regulating reserve ratio.

    Distinguishes three operating regions:
      1. Sufficient reserve (rho >= rho_target - deadband):
         No intervention -> (0.0 mL/min, 0.0 s).
      2. Moderate reserve deficit (critical_low <= rho < rho_target - deadband):
         Moderate intervention -> (low_flowrate, low_duration).
      3. Critical reserve deficit (rho < critical_low):
         Strong intervention -> (critical_flowrate, critical_duration).
    """

    def __init__(self, config: Optional[RuleBasedConfig] = None) -> None:
        self.config = config or RuleBasedConfig()
        self.time_since_last_dose: float = float("inf")

    def reset(self) -> None:
        """Reset internal timer state."""
        self.time_since_last_dose = float("inf")

    def compute_from_reserve(
        self,
        rho: float,
        time_since_last_dose: Optional[float] = None,
        dt: float = 60.0,
    ) -> tuple[float, float]:
        """
        Evaluate heuristic dosing rule given normalized reserve ratio rho in [0, 1].

        Args:
            rho: Intracellular reserve ratio in [0.0, 1.0].
            time_since_last_dose: Optional time in seconds since last active dose.
            dt: Simulation timestep in seconds.

        Returns:
            (flowrate mL/min, duration s)
        """
        cfg = self.config

        # Check actuator cooldown if provided
        tsld = (
            time_since_last_dose
            if time_since_last_dose is not None
            else self.time_since_last_dose
        )
        if tsld < cfg.min_time_between_doses:
            self.time_since_last_dose = tsld + dt
            return 0.0, 0.0

        # Region 3: Critical reserve deficit (below starvation boundary)
        if rho < cfg.critical_low:
            action = (cfg.critical_flowrate, cfg.critical_duration)
        # Region 2: Moderate reserve deficit (below setpoint - deadband)
        elif rho < cfg.rho_low_threshold:
            action = (cfg.low_flowrate, cfg.low_duration)
        # Region 1: Sufficient reserve (above or within deadband)
        else:
            action = (cfg.high_flowrate, cfg.high_duration)

        # Update cooldown tracking
        if action[0] > 0.0 and action[1] > 0.0:
            self.time_since_last_dose = 0.0
        else:
            self.time_since_last_dose = tsld + dt

        return action

    def compute_from_state(
        self,
        state: Any,
        internal_capacity: float = 0.5,
        dt: float = 60.0,
    ) -> tuple[float, float]:
        """
        Evaluate heuristic dosing rule given a TankState or state-like object.

        Uses the canonical compute_rho helper to compute rho with standard
        numerical safeguards.
        """
        rho = compute_rho(state, internal_capacity)
        tsld = getattr(state, "time_since_last_dose", None)
        return self.compute_from_reserve(rho, time_since_last_dose=tsld, dt=dt)

    def compute_from_ec(self, ec: float) -> tuple[float, float]:
        """Legacy EC-tracking rule evaluation (preserved for backward compatibility)."""
        cfg = self.config
        if ec < cfg.ec_critical_low:
            return cfg.critical_flowrate, cfg.critical_duration
        if ec < cfg.ec_low_threshold:
            return cfg.low_flowrate, cfg.low_duration
        if ec > cfg.ec_high_threshold:
            return cfg.high_flowrate, cfg.high_duration
        return 0.0, 0.0

    def compute(self, val: Any, *args: Any, **kwargs: Any) -> tuple[float, float]:
        """
        Dispatch shim for generic callers — prefer the explicit named methods:

          compute_from_reserve(rho)   — primary: normalized reserve ratio rho in [0, 1]
          compute_from_state(state)   — from a TankState or state-like object
          compute_from_ec(ec)         — legacy EC-tracking path (backward compat)

        Dispatch order:
          1. kwargs['is_ec=True']        -> compute_from_ec(val)
          2. val has .algae_biomass attr -> compute_from_state(val)
          3. val is dict                 -> dispatch on 'reserve_ratio'/'rho'/'ec' key
          4. fallback                    -> compute_from_reserve(float(val))
        """
        if kwargs.get("is_ec", False):
            return self.compute_from_ec(float(val))
        if hasattr(val, "algae_biomass") or hasattr(val, "internal_reserve"):
            return self.compute_from_state(val, **kwargs)
        if isinstance(val, dict):
            if "reserve_ratio" in val:
                return self.compute_from_reserve(float(val["reserve_ratio"]), **kwargs)
            if "rho" in val:
                return self.compute_from_reserve(float(val["rho"]), **kwargs)
            if "ec" in val and "biomass" not in val:
                return self.compute_from_ec(float(val["ec"]))
        return self.compute_from_reserve(float(val), **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[float, float]:
        return self.compute(*args, **kwargs)
