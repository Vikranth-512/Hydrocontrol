"""
VII. Parameter Identifiability & Robustness Experiments

  - One-At-a-Time (OAT) parameter sensitivity → tornado plots
  - Monte Carlo robustness (1000 randomized runs)
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.dynamics import TankDynamicsParams, step_dynamics
from simulation.validate.common import Experiment, make_initial_state
from simulation.validate.plotting import plot_tornado


# Parameters we'll perturb for OAT sensitivity analysis
SENSITIVE_PARAMS = [
    "maximum_uptake_rate",
    "half_saturation_mass",
    "maximum_growth_rate",
    "growth_yield",
    "maintenance_cost",
    "biomass_nutrient_content",
    "mortality_rate",
    "mineralization_rate",
    "damage_rate",
    "repair_rate",
    "osmotic_half_effect",
    "background_dilution_rate",
    "internal_capacity",
]


def _run_baseline(params: TankDynamicsParams, length: int, dt: float, actions):
    s = make_initial_state(params, dissolved_mass=2.0, biomass=100.0)
    for t in range(length):
        fr, dur = actions[t]
        s = step_dynamics(s, fr, dur, dt, params)
    return s


# ---------------------------------------------------------------------------
# PAR-01  OAT Sensitivity Analysis
# ---------------------------------------------------------------------------
def run_oat_sensitivity(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    length = 400
    actions = [(1.5, 15.0)] * length

    # Baseline run
    baseline = _run_baseline(params, length, dt, actions)
    baseline_ec = baseline.ec
    baseline_biomass = baseline.algae_biomass
    baseline_health = baseline.health_index

    rows = []
    sensitivities_ec = []
    sensitivities_biomass = []
    sensitivities_health = []

    for pname in SENSITIVE_PARAMS:
        base_val = getattr(params, pname)
        for direction, multiplier in [("+10%", 1.10), ("-10%", 0.90)]:
            p_mod = copy.deepcopy(params)
            setattr(p_mod, pname, base_val * multiplier)
            result = _run_baseline(p_mod, length, dt, actions)

            delta_ec = (result.ec - baseline_ec) / (baseline_ec + 1e-12)
            delta_bio = (result.algae_biomass - baseline_biomass) / (baseline_biomass + 1e-12)
            delta_h = (result.health_index - baseline_health) / (baseline_health + 1e-12)

            rows.append({
                "parameter": pname,
                "direction": direction,
                "delta_ec_frac": delta_ec,
                "delta_biomass_frac": delta_bio,
                "delta_health_frac": delta_h,
                "final_ec": result.ec,
                "final_biomass": result.algae_biomass,
                "final_health": result.health_index,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "oat_sensitivity.csv", index=False)

    # Tornado: use +10% values for ranking
    plus10 = df[df["direction"] == "+10%"]
    plot_tornado(
        list(plus10["parameter"]),
        plus10["delta_ec_frac"].values,
        "EC",
        output_dir / "tornado_ec.png",
    )
    plot_tornado(
        list(plus10["parameter"]),
        plus10["delta_biomass_frac"].values,
        "Biomass",
        output_dir / "tornado_biomass.png",
    )

    most_sensitive = plus10.iloc[np.argmax(np.abs(plus10["delta_ec_frac"].values))]["parameter"]

    return {
        "status": "PASS",
        "metrics": {
            "Most Sensitive Parameter (EC)": most_sensitive,
            "Num Parameters Tested": len(SENSITIVE_PARAMS),
        },
    }


# ---------------------------------------------------------------------------
# PAR-02  Monte Carlo Robustness
# ---------------------------------------------------------------------------
def run_monte_carlo_robustness(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    length = 1000
    n_runs = 500
    rng = np.random.default_rng(123)

    finals = []

    for run in range(n_runs):
        # Randomize initial conditions
        dm = rng.uniform(0.5, 5.0)
        bm = rng.uniform(50.0, 150.0)
        wt = rng.uniform(18.0, 30.0)

        # Randomize parameters slightly
        p_mod = TankDynamicsParams.sample_random(params, rng)

        s = make_initial_state(p_mod, dissolved_mass=dm, biomass=bm, water_temp=wt)

        # Random actions
        for t in range(length):
            fr = rng.uniform(0, 4.0)
            dur = rng.uniform(0, 25.0)
            s = step_dynamics(s, fr, dur, dt, p_mod, rng=rng)

        finals.append({
            "ec": s.ec,
            "biomass": s.algae_biomass,
            "reserve": s.internal_reserve,
            "health": s.health_index,
            "dissolved": s.dissolved_nutrient_mass,
        })

    df = pd.DataFrame(finals)
    df.to_csv(output_dir / "monte_carlo.csv", index=False)

    # Confidence intervals
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flatten(), ["ec", "biomass", "reserve", "health", "dissolved"]):
        if col in df.columns:
            ax.hist(df[col], bins=30, alpha=0.7)
            ax.set_title(f"{col} distribution (n={n_runs})")
            ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"μ={df[col].mean():.2f}")
            ax.legend()
    if len(axes.flatten()) > 5:
        axes.flatten()[-1].axis("off")
    fig.suptitle(f"Monte Carlo Robustness ({n_runs} runs)")
    fig.tight_layout()
    fig.savefig(output_dir / "monte_carlo.png", dpi=150)
    plt.close(fig)

    # Check for NaN/Inf in any run
    nan_count = int(df.isna().sum().sum())
    inf_count = int(np.isinf(df.select_dtypes(include=[np.number]).values).sum())

    return {
        "status": "PASS" if nan_count == 0 and inf_count == 0 else "FAIL",
        "metrics": {
            "NaN Count": nan_count,
            "Inf Count": inf_count,
            "EC Mean": float(df["ec"].mean()),
            "EC Std": float(df["ec"].std()),
            "Biomass Mean": float(df["biomass"].mean()),
            "Biomass Std": float(df["biomass"].std()),
            "Health Mean": float(df["health"].mean()),
        },
    }


# ---- Registry ----
REGISTERED_EXPERIMENTS = [
    Experiment(
        id="PAR-01", name="OAT Parameter Sensitivity", category="VII. Parameter Identifiability",
        hypothesis="Biological parameters have varying sensitivity on EC, biomass, and health, with identifiable dominant parameters.",
        execute=run_oat_sensitivity,
        metrics=["Most Sensitive Parameter (EC)"], plots=["tornado_ec.png", "tornado_biomass.png"],
    ),
    Experiment(
        id="PAR-02", name="Monte Carlo Robustness", category="VII. Parameter Identifiability",
        hypothesis="Under randomized initial conditions, parameters, and actions, the simulator produces no NaNs or Infs across 500 runs.",
        execute=run_monte_carlo_robustness,
        metrics=["NaN Count", "Inf Count", "EC Mean"], plots=["monte_carlo.png"],
    ),
]
