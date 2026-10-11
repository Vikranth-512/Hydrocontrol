"""
Algae tank environment wrapper for open-loop and closed-loop simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from simulation.dynamics import (
    DisturbanceInput,
    DisturbanceSpec,
    TankDynamicsParams,
    TankState,
    step_dynamics,
)
from simulation.disturbances import DisturbanceGenerator


@dataclass
class EnvironmentConfig:
    """Runtime constraints and targets."""

    dt_seconds: float = 60.0
    ec_target: float = 1.2
    ec_safe_min: float = 0.4
    ec_safe_max: float = 2.5
    flowrate_min: float = 0.0
    flowrate_max: float = 5.0
    duration_min: float = 0.0
    duration_max: float = 30.0
    min_time_between_doses: float = 120.0
    noise_std: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        defaults = {
            "water_temp": 0.05,
            "ec": 0.02,
            "turbidity": 1.0,
            "ph": 0.02,
            "dissolved_oxygen": 0.1,
        }
        if self.noise_std is None:
            self.noise_std = defaults
        else:
            self.noise_std = {**defaults, **self.noise_std}


class AlgaeTankEnvironment:
    """
    Gym-like interface: reset, step(action), observe noisy sensors.

    Actions: (flowrate, duration) with physical clipping and rate limits.
  Disturbances: DisturbanceSpec schedule with optional sensor bias.
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        dynamics_params: TankDynamicsParams,
        rng: Optional[np.random.Generator] = None,
        include_optional_sensors: bool = True,
        disturbance_generator: Optional[DisturbanceGenerator] = None,
        sensor_noise_schedule: Optional[Dict[str, np.ndarray]] = None,
        sensor_bias_schedule: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> None:
        self.config = config
        self.params = dynamics_params
        self.rng = rng or np.random.default_rng()
        self.include_optional_sensors = include_optional_sensors
        self.disturbance_gen = disturbance_generator
        self.sensor_noise_schedule = sensor_noise_schedule
        self.sensor_bias_schedule = sensor_bias_schedule
        self.state: Optional[TankState] = None
        self._disturbance_schedule: List[DisturbanceInput] = []
        self._sensor_bias_ec = 0.0
        self._step_counter: int = 0

    def reset(
        self,
        initial_state: Optional[TankState] = None,
        disturbance_schedule: Optional[List[DisturbanceInput]] = None,
        sensor_noise_schedule: Optional[Dict[str, np.ndarray]] = None,
        sensor_bias_schedule: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> np.ndarray:
        """Initialize episode; return noisy observation vector."""
        if initial_state is None:
            self.state = TankState.create_initial(self.params, rng=self.rng)
        else:
            self.state = initial_state

        self._disturbance_schedule = disturbance_schedule or []
        if sensor_noise_schedule is not None:
            self.sensor_noise_schedule = sensor_noise_schedule
        if sensor_bias_schedule is not None:
            self.sensor_bias_schedule = sensor_bias_schedule

        self._step_counter = 0
        if self.disturbance_gen:
            self.disturbance_gen.reset()

        if self.sensor_bias_schedule is not None:
            self._sensor_bias_ec = float(self.sensor_bias_schedule[0])
        else:
            self._sensor_bias_ec = 0.0
        return self._observe()

    def _clip_action(self, flowrate: float, duration: float) -> Tuple[float, float]:
        cfg = self.config
        fr = float(np.clip(flowrate, cfg.flowrate_min, cfg.flowrate_max))
        dur = float(np.clip(duration, cfg.duration_min, cfg.duration_max))

        if self.state is not None and self.state.time_since_last_dose < cfg.min_time_between_doses:
            if fr > 0 or dur > 0:
                fr, dur = 0.0, 0.0

        return fr, dur

    def _current_disturbance(self) -> DisturbanceInput:
        assert self.state is not None
        if self._disturbance_schedule and self.state.step_index < len(
            self._disturbance_schedule
        ):
            return self._disturbance_schedule[self.state.step_index]
        return DisturbanceSpec()

    def step(self, action: Tuple[float, float]) -> Tuple[np.ndarray, Dict]:
        """Apply control, advance dynamics, return (observation, info)."""
        assert self.state is not None
        flowrate, duration = self._clip_action(action[0], action[1])
        disturbance = self._current_disturbance()

        self.state = step_dynamics(
            self.state,
            flowrate,
            duration,
            self.config.dt_seconds,
            self.params,
            disturbance=disturbance,
        )

        self._step_counter += 1
        if self.sensor_bias_schedule is not None:
            idx = min(self._step_counter, len(self.sensor_bias_schedule) - 1)
            self._sensor_bias_ec = float(self.sensor_bias_schedule[idx])
        elif self.disturbance_gen:
            self._sensor_bias_ec = self.disturbance_gen.sample_sensor_bias()

        obs = self._observe()
        info = {
            "true_state": self.state.as_dict(),
            "flowrate": flowrate,
            "duration": duration,
            "ec": self.state.ec,
            "pending_nutrients": self.state.pending_nutrients,
            "health_index": self.state.health_index,
        }
        return obs, info

    def _observe(self) -> np.ndarray:
        assert self.state is not None
        s = self.state
        cfg = self.config
        ns = cfg.noise_std

        # Pre-generated CRN sensor noise innovation if available, else sample from RNG
        if self.sensor_noise_schedule is not None:
            idx = min(self._step_counter, len(self.sensor_noise_schedule["ec"]) - 1)
            eps_water_temp = float(self.sensor_noise_schedule["water_temp"][idx])
            eps_ec = float(self.sensor_noise_schedule["ec"][idx])
            eps_turb = float(self.sensor_noise_schedule["turbidity"][idx])
            eps_ph = float(self.sensor_noise_schedule["ph"][idx]) if "ph" in self.sensor_noise_schedule else 0.0
            eps_do = float(self.sensor_noise_schedule["dissolved_oxygen"][idx]) if "dissolved_oxygen" in self.sensor_noise_schedule else 0.0
        else:
            eps_water_temp = float(self.rng.normal(0, ns["water_temp"]))
            eps_ec = float(self.rng.normal(0, ns["ec"]))
            eps_turb = float(self.rng.normal(0, ns["turbidity"]))
            eps_ph = float(self.rng.normal(0, ns["ph"]))
            eps_do = float(self.rng.normal(0, ns["dissolved_oxygen"]))

        obs = {
            "water_temp": s.water_temp + eps_water_temp,
            "ec": s.ec + self._sensor_bias_ec + eps_ec,
            "turbidity": s.turbidity + eps_turb,
            "prev_flowrate": s.prev_flowrate,
            "prev_duration": s.prev_duration,
            "time_since_last_dose": s.time_since_last_dose,
        }

        if self.include_optional_sensors:
            obs["ph"] = s.ph + eps_ph
            obs["dissolved_oxygen"] = s.dissolved_oxygen + eps_do
            obs["ambient_temp"] = s.ambient_temp

        keys = sorted(obs.keys())
        return np.array([obs[k] for k in keys], dtype=np.float64)

    @property
    def observation_keys(self) -> List[str]:
        base = [
            "water_temp",
            "ec",
            "turbidity",
            "prev_flowrate",
            "prev_duration",
            "time_since_last_dose",
        ]
        if self.include_optional_sensors:
            base.extend(["ph", "dissolved_oxygen", "ambient_temp"])
        return sorted(base)
