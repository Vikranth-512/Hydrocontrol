"""
I. Physical Conservation Experiments

Verify the simulator obeys fundamental physics:
  - Strict mass conservation across all compartments
  - Correct dilution bookkeeping
  - Correct flux accounting through every transfer boundary
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from simulation.dynamics import TankDynamicsParams, TankState, step_dynamics
from simulation.validate.common import Experiment, make_initial_state, simulate_and_record
from simulation.validate.plotting import plot_conservation_error, plot_multi_panel


# ---------------------------------------------------------------------------
# PHYS-01  Strict Mass Conservation (10 000 steps, random dosing)
# ---------------------------------------------------------------------------
def run_mass_conservation(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    length = 10_000
    dt = 60.0
    rng = np.random.default_rng(42)
    actions = [(rng.uniform(0.0, 5.0), rng.uniform(0.0, 30.0)) for _ in range(length)]

    s = make_initial_state(params, dissolved_mass=2.0, biomass=150.0)
    initial_total = s.compute_total_mass(params)

    errors = []
    times = []
    for t in range(length):
        fr, dur = actions[t]
        expected = initial_total + s.cumulative_nutrients
        actual = s.compute_total_mass(params)
        errors.append(actual - expected)
        times.append(t * dt / 60.0)
        s = step_dynamics(s, fr, dur, dt, params, rng=rng)

    # final
    expected = initial_total + s.cumulative_nutrients
    actual = s.compute_total_mass(params)
    errors.append(actual - expected)
    times.append(length * dt / 60.0)

    errors = np.array(errors)
    times = np.array(times)

    max_err = float(np.max(np.abs(errors)))
    rms_err = float(np.sqrt(np.mean(errors ** 2)))

    pd.DataFrame({"time_min": times, "mass_error": errors}).to_csv(
        output_dir / "mass_conservation.csv", index=False
    )
    plot_conservation_error(times, errors, output_dir / "conservation_error.png")

    return {
        "status": "PASS" if max_err < 1e-6 else "FAIL",
        "metrics": {"Max Absolute Error": max_err, "RMS Error": rms_err},
    }


# ---------------------------------------------------------------------------
# PHYS-02  Dilution-Only Bookkeeping
# ---------------------------------------------------------------------------
def run_dilution_bookkeeping(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    """Zero dosing: only dilution should remove mass from dissolved pool."""
    length = 2000
    dt = 60.0
    s = make_initial_state(params, dissolved_mass=5.0, biomass=0.001, internal_reserve=0.0)

    df = simulate_and_record(params, length, dt, initial_state=s)
    df.to_csv(output_dir / "dilution_bookkeeping.csv", index=False)

    plot_multi_panel(
        df,
        [
            (["dissolved_nutrient_mass"], "Dissolved"),
            (["cumulative_dilution"], "Cum. Dilution"),
        ],
        "Dilution-Only Bookkeeping (no biomass)",
        output_dir / "dilution_bookkeeping.png",
    )

    # With near-zero biomass, uptake ≈ 0.  Mass loss ≈ cumulative dilution.
    final_dissolved = df["dissolved_nutrient_mass"].iloc[-1]
    cum_dilution = df["cumulative_dilution"].iloc[-1]
    initial_dissolved = df["dissolved_nutrient_mass"].iloc[0]
    lost = initial_dissolved - final_dissolved
    frac_accounted = cum_dilution / (lost + 1e-15)

    return {
        "status": "PASS" if 0.95 < frac_accounted < 1.05 else "FAIL",
        "metrics": {
            "Mass Lost": float(lost),
            "Cumulative Dilution": float(cum_dilution),
            "Fraction Accounted": float(frac_accounted),
        },
    }


# ---------------------------------------------------------------------------
# PHYS-03  Queue Conservation (impulse through transport)
# ---------------------------------------------------------------------------
def run_queue_conservation(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    """Inject a single impulse; verify all mass exits the queue into the dissolved pool."""
    dt = 60.0
    length = 200
    # Single impulse at t=5
    actions = [(0.0, 0.0)] * length
    actions[5] = (5.0, 30.0)

    s = make_initial_state(params, dissolved_mass=0.0, biomass=0.001, internal_reserve=0.0)
    df = simulate_and_record(params, length, dt, actions=actions, initial_state=s)
    df.to_csv(output_dir / "queue_conservation.csv", index=False)

    plot_multi_panel(
        df,
        [
            (["pending_nutrients"], "Queue Mass"),
            (["dissolved_nutrient_mass"], "Dissolved"),
        ],
        "Transport Queue Conservation (single impulse)",
        output_dir / "queue_conservation.png",
    )

    # After enough steps the queue should be nearly empty
    final_queue = df["pending_nutrients"].iloc[-1]
    status = "PASS" if final_queue < 0.01 else "FAIL"
    return {
        "status": status,
        "metrics": {"Final Queue Residual": float(final_queue)},
    }


# ---- Registry ----
REGISTERED_EXPERIMENTS = [
    Experiment(
        id="PHYS-01",
        name="Strict Mass Conservation",
        category="I. Physical Conservation",
        hypothesis="Total nutrient mass (all compartments + cumulative losses) exactly equals initial mass + cumulative doses at every timestep.",
        execute=run_mass_conservation,
        metrics=["Max Absolute Error", "RMS Error"],
        plots=["conservation_error.png"],
    ),
    Experiment(
        id="PHYS-02",
        name="Dilution-Only Bookkeeping",
        category="I. Physical Conservation",
        hypothesis="With near-zero biomass, mass loss is explained entirely by tracked cumulative dilution.",
        execute=run_dilution_bookkeeping,
        metrics=["Mass Lost", "Cumulative Dilution", "Fraction Accounted"],
        plots=["dilution_bookkeeping.png"],
    ),
    Experiment(
        id="PHYS-03",
        name="Transport Queue Conservation",
        category="I. Physical Conservation",
        hypothesis="An impulse dose placed in the transport queue eventually exits entirely into the dissolved pool.",
        execute=run_queue_conservation,
        metrics=["Final Queue Residual"],
        plots=["queue_conservation.png"],
    ),
]
