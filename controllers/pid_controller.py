"""
Research-grade PID for EC setpoint → (flowrate, duration).

Features: anti-windup, derivative filtering, deadband, output saturation, rate limiting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PIDGains:
    kp: float = 2.0
    ki: float = 0.05
    kd: float = 0.3


@dataclass
class PIDConfig:
    """Behavioral options beyond raw gains."""

    integral_min: float = -8.0
    integral_max: float = 8.0
    derivative_alpha: float = 0.85
    deadband: float = 0.04
    max_delta_flowrate: float = 1.5
    max_delta_duration: float = 8.0
    min_control_u: float = 0.02
    flowrate_scale: float = 2.0
    duration_scale: float = 10.0
    min_flowrate_when_active: float = 0.4
    min_duration_when_active: float = 5.0


class PIDController:
    """
    u = Kp*e + Ki*∫e + Kd*de_filtered/dt

    Maps u → (flowrate, duration) with saturation, rate limits, and deadband.
    """

    def __init__(
        self,
        setpoint: float = 1.2,
        gains: PIDGains | None = None,
        config: PIDConfig | None = None,
        flowrate_max: float = 5.0,
        duration_max: float = 30.0,
        flowrate_min: float = 0.0,
        duration_min: float = 0.0,
        dt: float = 60.0,
    ) -> None:
        self.setpoint = setpoint
        self.gains = gains or PIDGains()
        self.config = config or PIDConfig()
        self.flowrate_max = flowrate_max
        self.duration_max = duration_max
        self.flowrate_min = flowrate_min
        self.duration_min = duration_min
        self.dt = dt
        self._integral = 0.0
        self._prev_error = 0.0
        self._filtered_derivative = 0.0
        self._prev_flowrate = 0.0
        self._prev_duration = 0.0
        self._saturated = False

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._filtered_derivative = 0.0
        self._prev_flowrate = 0.0
        self._prev_duration = 0.0
        self._saturated = False

    def _rate_limit(self, target: float, previous: float, max_delta: float) -> float:
        delta = float(np.clip(target - previous, -max_delta, max_delta))
        return previous + delta

    def compute(self, ec: float) -> tuple[float, float]:
        cfg = self.config
        error = self.setpoint - ec

        if abs(error) < cfg.deadband:
            self._prev_error = error
            return 0.0, 0.0

        raw_derivative = (error - self._prev_error) / max(self.dt, 1e-6)
        self._filtered_derivative = (
            cfg.derivative_alpha * self._filtered_derivative
            + (1.0 - cfg.derivative_alpha) * raw_derivative
        )
        self._prev_error = error

        u_p = self.gains.kp * error
        u_d = self.gains.kd * self._filtered_derivative

        u_unsat = u_p + self.gains.ki * self._integral + u_d

        if u_unsat <= cfg.min_control_u:
            self._saturated = u_unsat <= 0
            self._prev_flowrate = 0.0
            self._prev_duration = 0.0
            return 0.0, 0.0

        flowrate = min(
            self.flowrate_max,
            max(cfg.min_flowrate_when_active, u_unsat * cfg.flowrate_scale),
        )
        duration = min(
            self.duration_max,
            max(cfg.min_duration_when_active, u_unsat * cfg.duration_scale),
        )

        flowrate = self._rate_limit(flowrate, self._prev_flowrate, cfg.max_delta_flowrate)
        duration = self._rate_limit(duration, self._prev_duration, cfg.max_delta_duration)
        flowrate = float(np.clip(flowrate, self.flowrate_min, self.flowrate_max))
        duration = float(np.clip(duration, self.duration_min, self.duration_max))

        u_sat = flowrate / max(cfg.flowrate_scale, 1e-6)
        self._saturated = abs(u_unsat - u_sat) > 0.05

        if not self._saturated:
            self._integral += error * self.dt
            self._integral = float(
                np.clip(self._integral, cfg.integral_min, cfg.integral_max)
            )

        self._prev_flowrate = flowrate
        self._prev_duration = duration
        return float(flowrate), float(duration)
