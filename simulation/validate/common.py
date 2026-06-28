"""
Core utilities and data structures for the validation framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.dynamics import (
    TankDynamicsParams,
    TankState,
    step_dynamics,
    DisturbanceInput,
    DisturbanceSpec,
)


@dataclass
class Experiment:
    """Metadata and execution hook for a scientific validation experiment."""
    id: str
    name: str
    category: str
    hypothesis: str
    metrics: List[str]
    plots: List[str]
    execute: Callable[[Path, TankDynamicsParams], Dict[str, Any]]


def simulate_and_record(
    params: TankDynamicsParams,
    length: int,
    dt: float = 60.0,
    actions: Optional[List[Tuple[float, float]]] = None,
    initial_state: Optional[TankState] = None,
    disturbances: Optional[List[DisturbanceInput]] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Execute an open-loop simulation recording every physical state,
    sensor output, and cumulative mass tracker at every timestep.
    """
    rng = rng or np.random.default_rng(42)
    s = initial_state if initial_state is not None else TankState.create_initial(params, rng=rng)
    actions = actions or [(0.0, 0.0)] * length
    disturbances = disturbances or [0.0] * length

    rows: List[Dict[str, Any]] = []

    for t in range(length):
        fr, dur = actions[t] if t < len(actions) else (0.0, 0.0)
        d = disturbances[t] if t < len(disturbances) else 0.0

        row = s.as_dict()
        row["time_min"] = t * dt / 60.0
        row["flowrate"] = fr
        row["duration"] = dur
        row["total_mass"] = s.compute_total_mass(params)
        rows.append(row)

        s = step_dynamics(s, fr, dur, dt, params, disturbance=d, rng=rng)

    # Append final state
    row = s.as_dict()
    row["time_min"] = length * dt / 60.0
    row["flowrate"] = 0.0
    row["duration"] = 0.0
    row["total_mass"] = s.compute_total_mass(params)
    rows.append(row)

    return pd.DataFrame(rows)


def make_initial_state(
    params: TankDynamicsParams,
    dissolved_mass: float = 1.5,
    biomass: float = 80.0,
    water_temp: Optional[float] = None,
    internal_reserve: Optional[float] = None,
    health_index: float = 1.0,
    damage_index: float = 0.0,
    dead_biomass_pool: float = 0.0,
) -> TankState:
    """
    Factory for creating TankState with full control over initial conditions.
    This wraps the standard create_initial and overrides specific fields.
    """
    wt = water_temp if water_temp is not None else params.ambient_temp_mean
    reserve = internal_reserve if internal_reserve is not None else biomass * params.internal_capacity * 0.5

    s = TankState(
        water_temp=float(wt),
        ph=7.2,
        dissolved_oxygen=8.0,
        ambient_temp=params.ambient_temp_mean,
        nutrient_queue=np.zeros(params.delay_steps, dtype=np.float64),
        dissolved_nutrient_mass=float(dissolved_mass),
        algae_biomass=float(biomass),
        internal_reserve=float(reserve),
        dead_biomass_pool=float(dead_biomass_pool),
        health_index=float(health_index),
        damage_index=float(damage_index),
    )
    s.ec = params.sensor_gain_ec * s.dissolved_nutrient_mass
    s.turbidity = params.biomass_optical_factor * s.algae_biomass
    return s
