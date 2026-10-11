from typing import Any

from controllers.no_action_controller import NoActionController
from controllers.pid_controller import PIDConfig, PIDController, PIDGains
from controllers.rule_based_controller import RuleBasedConfig, RuleBasedController
from controllers.pid_tuner import PIDTuner, evaluate_pid, run_pid_tuning
from controllers.mpc_controller import MPCReserveConfig, MPCReserveController


def get_controller(name: str, config: Any = None, **kwargs: Any) -> Any:
    """
    Controller factory returning an initialized controller instance.
    
    Supported controller names:
      - 'no_action': NoActionController
      - 'rule_based_reserve' or 'rule_based': RuleBasedController
      - 'pid': PIDController
      - 'mpc_reserve' or 'mpc': MPCReserveController
    """
    key = name.lower().strip()
    if key in ("no_action", "none", "zero"):
        return NoActionController()
    if key in ("rule_based_reserve", "rule_based", "rule"):
        cfg = config if isinstance(config, RuleBasedConfig) else RuleBasedConfig(**(config or {}))
        return RuleBasedController(cfg)
    if key == "pid":
        if isinstance(config, PIDController):
            return config
        return PIDController(config=config, **kwargs)
    if key in ("mpc_reserve", "mpc"):
        cfg = config if isinstance(config, MPCReserveConfig) else MPCReserveConfig(**(config or {}))
        return MPCReserveController(cfg)
    raise ValueError(f"Unknown controller '{name}'. Choose from: no_action, rule_based_reserve, pid, mpc_reserve")


__all__ = [
    "NoActionController",
    "PIDConfig",
    "PIDController",
    "PIDGains",
    "RuleBasedConfig",
    "RuleBasedController",
    "MPCReserveConfig",
    "MPCReserveController",
    "PIDTuner",
    "evaluate_pid",
    "run_pid_tuning",
    "get_controller",
]
