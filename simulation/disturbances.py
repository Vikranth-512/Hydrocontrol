"""
Disturbance generation: thermal events, sediment/turbidity shocks, correlated noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from simulation.dynamics import DisturbanceSpec


@dataclass
class DisturbanceConfig:
    ec_shock_prob: float = 0.02
    ec_shock_magnitude: float = 0.08
    temp_spike_prob: float = 0.02
    temp_spike_magnitude: float = 3.0
    heatwave_prob: float = 0.008
    heatwave_duration: int = 40
    heatwave_magnitude: float = 4.0
    cold_shock_prob: float = 0.008
    cold_shock_magnitude: float = -3.5
    seasonal_drift_rate: float = 0.002
    mixing_noise_std: float = 0.006
    uptake_pulse_prob: float = 0.025
    uptake_pulse_factor: float = 1.35
    actuator_failure_prob: float = 0.01
    actuator_failure_efficiency: float = 0.35
    nutrient_depletion_prob: float = 0.02
    nutrient_depletion_magnitude: float = 0.12
    turbidity_sediment_prob: float = 0.015
    turbidity_sediment_magnitude: float = 18.0
    correlated_noise_rho: float = 0.85

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DisturbanceConfig":
        return cls(
            ec_shock_prob=cfg.get("ec_shock_prob", 0.02),
            ec_shock_magnitude=cfg.get("ec_shock_magnitude", 0.08),
            temp_spike_prob=cfg.get("temp_spike_prob", 0.02),
            temp_spike_magnitude=cfg.get("temp_spike_magnitude", 3.0),
            heatwave_prob=cfg.get("heatwave_prob", 0.008),
            heatwave_duration=cfg.get("heatwave_duration", 40),
            heatwave_magnitude=cfg.get("heatwave_magnitude", 4.0),
            cold_shock_prob=cfg.get("cold_shock_prob", 0.008),
            cold_shock_magnitude=cfg.get("cold_shock_magnitude", -3.5),
            seasonal_drift_rate=cfg.get("seasonal_drift_rate", 0.002),
            mixing_noise_std=cfg.get("mixing_noise_std", 0.006),
            uptake_pulse_prob=cfg.get("uptake_pulse_prob", 0.025),
            uptake_pulse_factor=cfg.get("uptake_pulse_factor", 1.35),
            actuator_failure_prob=cfg.get("actuator_failure_prob", 0.01),
            actuator_failure_efficiency=cfg.get("actuator_failure_efficiency", 0.35),
            nutrient_depletion_prob=cfg.get("nutrient_depletion_prob", 0.02),
            nutrient_depletion_magnitude=cfg.get("nutrient_depletion_magnitude", 0.12),
            turbidity_sediment_prob=cfg.get("turbidity_sediment_prob", 0.015),
            turbidity_sediment_magnitude=cfg.get("turbidity_sediment_magnitude", 18.0),
            correlated_noise_rho=cfg.get("correlated_noise_rho", 0.85),
        )


class DisturbanceGenerator:
    def __init__(self, config: DisturbanceConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self._mixing_state = 0.0
        self._sensor_bias_ec = 0.0
        self._bias_steps_remaining = 0
        self._heatwave_remaining = 0
        self._seasonal_phase = 0.0

    def reset(self) -> None:
        self._mixing_state = 0.0
        self._sensor_bias_ec = 0.0
        self._bias_steps_remaining = 0
        self._heatwave_remaining = 0
        self._seasonal_phase = 0.0

    def sample(self, scenario: str = "normal", t: int = 0, length: int = 100) -> DisturbanceSpec:
        cfg = self.config
        rho = cfg.correlated_noise_rho
        innov = self.rng.normal(0, cfg.mixing_noise_std)
        self._mixing_state = rho * self._mixing_state + (1 - rho) * innov

        ec_shock = 0.0
        temp_shock = 0.0
        turbidity_shock = 0.0
        uptake_mult = 1.0
        actuator_eff = 1.0

        # Scenario-specific structured events
        if scenario == "temp_spike" or scenario == "heatwave":
            if length // 3 < t < length // 3 + cfg.heatwave_duration:
                temp_shock = cfg.heatwave_magnitude * 0.015
                uptake_mult = 1.2
        elif scenario == "cold_shock":
            if length // 2 < t < length // 2 + 25:
                temp_shock = cfg.cold_shock_magnitude * 0.012
                uptake_mult = 0.85
        elif scenario == "ec_drop" or scenario == "nutrient_depletion":
            if length // 4 < t < length // 4 + 30:
                uptake_mult = 1.35
                ec_shock = -cfg.nutrient_depletion_magnitude * 0.4
        elif scenario == "sediment" or scenario == "turbidity_event":
            if length // 5 < t < length // 5 + 15:
                turbidity_shock = cfg.turbidity_sediment_magnitude * 0.5
        elif scenario == "actuator_failure" and t > length // 2:
            actuator_eff = cfg.actuator_failure_efficiency

        # Stochastic events
        if self._heatwave_remaining > 0:
            self._heatwave_remaining -= 1
            temp_shock += cfg.heatwave_magnitude * 0.012
        elif self.rng.random() < cfg.heatwave_prob:
            self._heatwave_remaining = cfg.heatwave_duration

        if self.rng.random() < cfg.temp_spike_prob:
            temp_shock += self.rng.normal(0, cfg.temp_spike_magnitude * 0.015)

        if self.rng.random() < cfg.cold_shock_prob:
            temp_shock += cfg.cold_shock_magnitude * 0.01

        if self.rng.random() < cfg.ec_shock_prob:
            ec_shock += self.rng.choice([-1, 1]) * cfg.ec_shock_magnitude * 0.5

        if self.rng.random() < cfg.uptake_pulse_prob:
            uptake_mult *= cfg.uptake_pulse_factor

        if self.rng.random() < cfg.actuator_failure_prob:
            actuator_eff = min(actuator_eff, cfg.actuator_failure_efficiency)

        if self.rng.random() < cfg.nutrient_depletion_prob:
            ec_shock -= cfg.nutrient_depletion_magnitude * 0.6

        if self.rng.random() < cfg.turbidity_sediment_prob:
            turbidity_shock += self.rng.exponential(cfg.turbidity_sediment_magnitude * 0.3)

        # Slow seasonal thermal drift
        self._seasonal_phase += cfg.seasonal_drift_rate
        temp_shock += 0.3 * np.sin(self._seasonal_phase)

        return DisturbanceSpec(
            ec_shock=ec_shock,
            temp_shock=temp_shock,
            mixing_noise=self._mixing_state,
            uptake_multiplier=uptake_mult,
            actuator_efficiency=actuator_eff,
            turbidity_shock=turbidity_shock,
        )

    def sample_sensor_bias(self, duration_steps: int = 30) -> float:
        if self._bias_steps_remaining <= 0 and self.rng.random() < 0.01:
            self._sensor_bias_ec = self.rng.normal(0, 0.06)
            self._bias_steps_remaining = duration_steps
        if self._bias_steps_remaining > 0:
            self._bias_steps_remaining -= 1
        else:
            self._sensor_bias_ec *= 0.9
        return self._sensor_bias_ec

    def build_schedule(self, scenario: str, length: int) -> List[DisturbanceSpec]:
        self.reset()
        return [self.sample(scenario, t, length) for t in range(length)]
