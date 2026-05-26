"""
Rule-based threshold controller for EC maintenance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuleBasedConfig:
    ec_low_threshold: float = 0.9
    ec_high_threshold: float = 1.4
    ec_target: float = 1.2
    low_flowrate: float = 2.0
    low_duration: float = 15.0
    high_flowrate: float = 0.0
    high_duration: float = 0.0
    critical_low: float = 0.6
    critical_flowrate: float = 4.0
    critical_duration: float = 20.0


class RuleBasedController:
    """Piecewise threshold logic on EC."""

    def __init__(self, config: RuleBasedConfig | None = None) -> None:
        self.config = config or RuleBasedConfig()

    def reset(self) -> None:
        pass

    def compute(self, ec: float) -> tuple[float, float]:
        cfg = self.config
        if ec < cfg.critical_low:
            return cfg.critical_flowrate, cfg.critical_duration
        if ec < cfg.ec_low_threshold:
            return cfg.low_flowrate, cfg.low_duration
        if ec > cfg.ec_high_threshold:
            return cfg.high_flowrate, cfg.high_duration
        return 0.0, 0.0
