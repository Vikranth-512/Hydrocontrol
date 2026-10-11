"""
No-Action Controller baseline for algae nutrient regulation benchmark.

Open-loop uncontrolled baseline that never issues a dosing intervention.
At every control timestep:
    flowrate = 0.0 mL/min
    duration = 0.0 s

This baseline provides the empirical open-loop lower bound for evaluating
controller discriminability.
"""

from __future__ import annotations

from typing import Any


class NoActionController:
    """
    Uncontrolled baseline controller.

    Always issues (0.0 mL/min, 0.0 s) regardless of state, observation,
    or historical actions.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def reset(self) -> None:
        """Reset internal controller state (no-op for open-loop baseline)."""
        pass

    def compute(self, *args: Any, **kwargs: Any) -> tuple[float, float]:
        """Return zero dosing action (0.0 mL/min, 0.0 s)."""
        return 0.0, 0.0

    def compute_from_reserve(self, *args: Any, **kwargs: Any) -> tuple[float, float]:
        """Interface compatibility method returning zero dosing action."""
        return 0.0, 0.0

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[float, float]:
        """Callable interface returning zero dosing action."""
        return 0.0, 0.0
