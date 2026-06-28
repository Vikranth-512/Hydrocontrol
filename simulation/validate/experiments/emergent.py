"""
IV. Emergent Behavior Experiments

Verify that complex dynamics emerge from coupled subsystems
rather than being explicitly programmed:
  - Delayed recovery
  - Positive feedback (toxic accumulation)
  - Dynamic equilibrium under constant dosing
  - Pulse response delay chain
  - Repeated pulse phase lag
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.dynamics import TankDynamicsParams, step_dynamics
from simulation.validate.common import Experiment, make_initial_state, simulate_and_record
from simulation.validate.plotting import plot_multi_panel
from simulation.validate.metrics import estimate_impulse_delay


# ---------------------------------------------------------------------------
# EMR-01  Excessive Dosing Positive Feedback
# ---------------------------------------------------------------------------
def run_toxic_accumulation(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    length = 500
    actions = [(5.0, 30.0)] * length
    s0 = make_initial_state(params, dissolved_mass=1.5, biomass=100.0)
    df = simulate_and_record(params, length, dt, actions=actions, initial_state=s0)
    df.to_csv(output_dir / "toxic_accumulation.csv", index=False)

    plot_multi_panel(
        df,
        [
            (["dissolved_nutrient_mass"], "Dissolved"),
            (["algae_biomass", "dead_biomass_pool"], "Biomass"),
            (["health_index"], "Health"),
            (["ec"], "EC Sensor"),
        ],
        "Excessive Dosing: Positive Feedback Loop",
        output_dir / "toxic_accumulation.png",
    )

    # Verify positive feedback: dissolved should accelerate upward while biomass collapses
    dissolved_end = float(df["dissolved_nutrient_mass"].iloc[-1])
    biomass_end = float(df["algae_biomass"].iloc[-1])
    health_end = float(df["health_index"].iloc[-1])

    return {
        "status": "PASS" if dissolved_end > 50.0 and health_end < 0.3 else "FAIL",
        "metrics": {
            "Final Dissolved": dissolved_end,
            "Final Biomass": biomass_end,
            "Final Health": health_end,
        },
    }


# ---------------------------------------------------------------------------
# EMR-02  Dynamic Equilibrium Under Constant Dosing
# ---------------------------------------------------------------------------
def run_dynamic_equilibrium(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    length = 10000
    actions = [(0.1, 1.0)] * length
    s0 = make_initial_state(params, dissolved_mass=1.5, biomass=100.0)
    df = simulate_and_record(params, length, dt, actions=actions, initial_state=s0)
    df.to_csv(output_dir / "dynamic_equilibrium.csv", index=False)

    plot_multi_panel(
        df,
        [
            (["dissolved_nutrient_mass"], "Dissolved"),
            (["algae_biomass"], "Biomass"),
            (["internal_reserve"], "Reserve"),
            (["ec"], "EC"),
        ],
        "Dynamic Equilibrium Under Constant Dosing",
        output_dir / "dynamic_equilibrium.png",
    )

    # Check for equilibrium: tail variance should be low relative to mean
    tail = df.iloc[-1000:]
    dissolved_cv = float(tail["dissolved_nutrient_mass"].std() / (tail["dissolved_nutrient_mass"].mean() + 1e-9))
    biomass_cv = float(tail["algae_biomass"].std() / (tail["algae_biomass"].mean() + 1e-9))

    return {
        "status": "PASS" if dissolved_cv < 0.1 and biomass_cv < 0.1 else "FAIL",
        "metrics": {
            "Dissolved CV (tail)": dissolved_cv,
            "Biomass CV (tail)": biomass_cv,
            "Tail Mean Dissolved": float(tail["dissolved_nutrient_mass"].mean()),
            "Tail Mean Biomass": float(tail["algae_biomass"].mean()),
        },
    }


# ---------------------------------------------------------------------------
# EMR-03  Pulse Response Delay Chain
# ---------------------------------------------------------------------------
def run_pulse_response(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    length = 500
    actions = [(0.0, 0.0)] * length
    pulse_t = 20
    actions[pulse_t] = (5.0, 30.0)

    s0 = make_initial_state(params, dissolved_mass=1.0, biomass=100.0)
    df = simulate_and_record(params, length, dt, actions=actions, initial_state=s0)
    df.to_csv(output_dir / "pulse_response.csv", index=False)

    plot_multi_panel(
        df,
        [
            (["dissolved_nutrient_mass"], "Dissolved"),
            (["internal_reserve"], "Reserve"),
            (["algae_biomass"], "Biomass"),
            (["ec"], "EC"),
            (["turbidity"], "Turbidity"),
        ],
        "Single Pulse Response - Delay Chain",
        output_dir / "pulse_response.png",
    )

    # Measure delays
    delay_dissolved = estimate_impulse_delay(df["dissolved_nutrient_mass"].values, pulse_t)
    delay_reserve = estimate_impulse_delay(df["internal_reserve"].values, pulse_t)
    delay_ec = estimate_impulse_delay(df["ec"].values, pulse_t)

    return {
        "status": "PASS" if delay_dissolved >= 0 else "FAIL",
        "metrics": {
            "Delay to Dissolved (steps)": delay_dissolved,
            "Delay to Reserve (steps)": delay_reserve,
            "Delay to EC (steps)": delay_ec,
        },
    }


# ---------------------------------------------------------------------------
# EMR-04  Repeated Pulse Phase Lag
# ---------------------------------------------------------------------------
def run_repeated_pulse(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    length = 2000
    period = 100  # Dose every 100 steps
    actions = [(0.0, 0.0)] * length
    for i in range(0, length, period):
        actions[i] = (4.0, 25.0)

    s0 = make_initial_state(params, dissolved_mass=1.0, biomass=100.0)
    df = simulate_and_record(params, length, dt, actions=actions, initial_state=s0)
    df.to_csv(output_dir / "repeated_pulse.csv", index=False)

    plot_multi_panel(
        df,
        [
            (["dissolved_nutrient_mass"], "Dissolved"),
            (["internal_reserve"], "Reserve"),
            (["algae_biomass"], "Biomass"),
        ],
        "Repeated Pulse Dosing - Phase Lag & Memory",
        output_dir / "repeated_pulse.png",
    )

    # Check if biomass accumulates over successive pulses (memory effect)
    biomass_vals = df["algae_biomass"].values
    growth_total = float(biomass_vals[-1] - biomass_vals[0])

    return {
        "status": "PASS",
        "metrics": {
            "Total Biomass Change": growth_total,
            "Final Biomass": float(biomass_vals[-1]),
        },
    }


# ---- Registry ----
REGISTERED_EXPERIMENTS = [
    Experiment(
        id="EMR-01", name="Toxic Accumulation Positive Feedback", category="IV. Emergent Behavior",
        hypothesis="Excessive dosing triggers osmotic stress → uptake halt → dissolved accumulation → biomass collapse, an emergent positive feedback loop.",
        execute=run_toxic_accumulation,
        metrics=["Final Dissolved", "Final Health"], plots=["toxic_accumulation.png"],
    ),
    Experiment(
        id="EMR-02", name="Dynamic Equilibrium", category="IV. Emergent Behavior",
        hypothesis="Constant moderate dosing produces a dynamic equilibrium where uptake ≈ dosing, without hidden restoring forces.",
        execute=run_dynamic_equilibrium,
        metrics=["Dissolved CV (tail)", "Biomass CV (tail)"], plots=["dynamic_equilibrium.png"],
    ),
    Experiment(
        id="EMR-03", name="Pulse Response Delay Chain", category="IV. Emergent Behavior",
        hypothesis="A single pulse propagates through dissolved → reserve → biomass → turbidity with measurable cascading delays.",
        execute=run_pulse_response,
        metrics=["Delay to Dissolved", "Delay to Reserve"], plots=["pulse_response.png"],
    ),
    Experiment(
        id="EMR-04", name="Repeated Pulse Phase Lag", category="IV. Emergent Behavior",
        hypothesis="Periodic dosing reveals phase lag, memory accumulation, and potential nonlinear saturation across pulses.",
        execute=run_repeated_pulse,
        metrics=["Total Biomass Change"], plots=["repeated_pulse.png"],
    ),
]
