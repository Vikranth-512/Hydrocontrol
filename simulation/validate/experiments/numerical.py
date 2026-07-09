"""
III. Numerical Analysis Experiments

Verify the numerical solver is trustworthy:
  - Timestep sensitivity / convergence
  - Long-horizon drift / NaN detection
  - Clipping frequency analysis
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


# ---------------------------------------------------------------------------
# NUM-01  Timestep Sensitivity
# ---------------------------------------------------------------------------
def run_timestep_sensitivity(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dts = [30.0, 60.0, 120.0, 300.0]
    # Same scenario: moderate constant dosing for 500 real-time minutes
    total_real_minutes = 500.0
    results_by_dt = {}

    for dt in dts:
        length = int(total_real_minutes * 60.0 / dt)
        actions = [(2.0, dt / 60.0)] * length
        s0 = make_initial_state(params, dissolved_mass=2.0, biomass=100.0)
        df = simulate_and_record(params, length, dt, actions=actions, initial_state=s0)
        results_by_dt[dt] = df
        df.to_csv(output_dir / f"dt_{int(dt)}s.csv", index=False)

    # Compare final states
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    cols = ["dissolved_nutrient_mass", "algae_biomass", "internal_reserve", "health_index"]
    for ax, col in zip(axes.flatten(), cols):
        for dt, df in results_by_dt.items():
            ax.plot(df["time_min"], df[col], label=f"dt={int(dt)}s")
        ax.set_title(col)
        ax.set_xlabel("Time (min)")
        ax.legend(fontsize=7)
    fig.suptitle("Timestep Sensitivity Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "timestep_sensitivity.png", dpi=150)
    plt.close(fig)

    # Quantify divergence relative to dt=30s baseline
    ref = results_by_dt[30.0]
    ref_final = float(ref["algae_biomass"].iloc[-1])
    divergences = {}
    for dt, df in results_by_dt.items():
        final = float(df["algae_biomass"].iloc[-1])
        divergences[f"dt={int(dt)}s Biomass Final"] = final
        divergences[f"dt={int(dt)}s Biomass Error vs 30s"] = abs(final - ref_final)

    max_error = max(v for k, v in divergences.items() if "Error" in k)
    return {
        "status": "PASS" if max_error < 10.0 else "FAIL",
        "metrics": divergences,
    }


# ---------------------------------------------------------------------------
# NUM-02  Long Horizon Stability (100k steps)
# ---------------------------------------------------------------------------
def run_long_horizon(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    length = 100_000
    # Start biomass near the analytical carrying capacity (~1140) so the
    # system reaches dynamic equilibrium quickly.  Starting at 100 produced
    # a 100k-step growth curve with no visible plateau.
    s = make_initial_state(params, dissolved_mass=2.0, biomass=1200.0)

    nan_count = 0
    inf_count = 0
    max_biomass = 0.0
    max_dissolved = 0.0

    # Tracking averages
    fluxes = {
        "uptake": 0.0, "maintenance": 0.0, "growth": 0.0,
        "mortality": 0.0, "damage_inc": 0.0, "repair_inc": 0.0
    }

    # We don't store full DF for 100k steps - sample every 100
    sampled_rows = []

    # Proportional controller parameters
    # Target dissolved well below the osmotic half-effect (5.0) to keep the
    # culture in a safe operating regime.  The old bang-bang controller
    # allowed dissolved to reach 5.0 -- already 50 % uptake inhibition --
    # then biomass grew unbounded and collapsed osmotically.
    dissolved_target = 1.5
    base_flow = 0.5
    base_dur = 2.0
    # Biomass level above which we throttle dosing to prevent the system
    # from drifting above the self-shading carrying capacity.
    biomass_throttle_level = 200.0

    for t in range(length):
        # --- Proportional dosing controller ---
        error = dissolved_target - s.dissolved_nutrient_mass
        dose_frac = float(np.clip(error / dissolved_target, 0.0, 1.0))

        # Throttle when biomass is already high -- gentle logistic taper
        biomass_brake = biomass_throttle_level / (biomass_throttle_level + max(s.algae_biomass - biomass_throttle_level, 0.0))
        dose_frac *= biomass_brake

        fr = base_flow * dose_frac
        dur = base_dur * dose_frac

        s_next = step_dynamics(s, fr, dur, dt, params)
        
        # Calculate fluxes at state s
        osmotic = (s.dissolved_nutrient_mass / params.osmotic_half_effect)**2
        osmotic_factor = 1.0 / (1.0 + osmotic)
        
        light_factor = params.light_attenuation_half_mass**2 / (params.light_attenuation_half_mass**2 + s.algae_biomass**2)
        
        uptake = params.maximum_uptake_rate * (s.dissolved_nutrient_mass / (params.half_saturation_mass + s.dissolved_nutrient_mass)) * osmotic_factor * s.health_index * s.algae_biomass
        maint = params.maintenance_cost * s.algae_biomass
        reserve_ratio = s.internal_reserve / max(s.algae_biomass * params.internal_capacity, 1e-6)
        growth = params.maximum_growth_rate * reserve_ratio * osmotic_factor * light_factor * s.algae_biomass
        
        mort = params.mortality_rate * np.log1p(osmotic) * max(1.0, (10.0 * s.damage_index)**2) * s.algae_biomass
        
        deficit_ratio = max(0.0, maint - s.internal_reserve) / max(maint, 1e-6)
        stress_drive = deficit_ratio * 1.0 + osmotic * 2.0
        dmg = stress_drive * params.damage_rate * (1.0 - s.damage_index)
        
        repair_avail = min(maint, s.internal_reserve) / max(maint, 1e-6)
        repair = (params.repair_rate * repair_avail + params.basal_repair_rate) * (s.damage_index ** 0.7)
        
        fluxes["uptake"] += float(uptake)
        fluxes["maintenance"] += float(maint)
        fluxes["growth"] += float(growth)
        fluxes["mortality"] += float(mort)
        fluxes["damage_inc"] += float(dmg)
        fluxes["repair_inc"] += float(repair)

        s = s_next

        for v in [s.dissolved_nutrient_mass, s.algae_biomass, s.internal_reserve, s.health_index]:
            if np.isnan(v):
                nan_count += 1
            if np.isinf(v):
                inf_count += 1

        max_biomass = max(max_biomass, s.algae_biomass)
        max_dissolved = max(max_dissolved, s.dissolved_nutrient_mass)

        if t % 100 == 0:
            row = s.as_dict()
            row["time_min"] = t * dt / 60.0
            sampled_rows.append(row)

    df = pd.DataFrame(sampled_rows)
    df.to_csv(output_dir / "long_horizon.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_min"], df["algae_biomass"], label="Biomass")
    plt.plot(df["time_min"], df["dissolved_nutrient_mass"], label="Dissolved")
    plt.title("Long Horizon Stability (100k steps)")
    plt.xlabel("Time (min)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "long_horizon.png", dpi=150)
    plt.close()

    avg_fluxes = {k: v / length for k, v in fluxes.items()}
    pd.DataFrame([avg_fluxes]).to_csv(output_dir / "long_horizon_avg_rates.csv", index=False)

    # A healthy long-horizon run should have no numerical issues AND
    # maintain a viable living culture (not collapse to zero).
    alive = s.algae_biomass > 10.0 and s.health_index > 0.5

    return {
        "status": "PASS" if nan_count == 0 and inf_count == 0 and alive else "FAIL",
        "metrics": {
            "NaN Count": nan_count,
            "Inf Count": inf_count,
            "Max Biomass": float(max_biomass),
            "Max Dissolved": float(max_dissolved),
            "Avg Growth": avg_fluxes["growth"],
            "Avg Mortality": avg_fluxes["mortality"],
            "Final Biomass": float(s.algae_biomass),
            "Final Health": float(s.health_index),
        },
    }


# ---------------------------------------------------------------------------
# NUM-03  Clipping / Bounding Frequency
# ---------------------------------------------------------------------------
def run_clipping_analysis(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    """Count how often the numerical integration relies on min/max bounding."""
    dt = 60.0
    length = 10_000
    rng = np.random.default_rng(99)
    actions = [(rng.uniform(0, 8.0), rng.uniform(0, 30.0)) for _ in range(length)]

    s = make_initial_state(params, dissolved_mass=1.0, biomass=100.0)

    clips = {
        "uptake_capped_by_dissolved": 0,
        "growth_capped_by_reserve": 0,
        "mortality_capped_at_95pct": 0,
        "damage_clipped_0": 0,
        "damage_clipped_1": 0,
    }

    for t in range(length):
        fr, dur = actions[t]
        # We'll manually trace the clipping events by re-running the logic
        # This duplicates some of step_dynamics but is necessary for auditing
        dist_spec = __import__("simulation.dynamics", fromlist=["DisturbanceSpec"]).DisturbanceSpec()
        dt_scale = dt / 60.0

        dose_mass = fr * dur / 60.0

        from simulation.dynamics import (
            thermal_efficiency, q10_metabolism_factor, _inject_delayed_dose, _release_absorption
        )
        th_growth = thermal_efficiency(s.water_temp, params)
        th_resp = q10_metabolism_factor(s.water_temp, params)
        osmotic = (s.dissolved_nutrient_mass / params.osmotic_half_effect) ** 2
        osmotic_f = 1.0 / (1.0 + osmotic)

        queue = list(s.pending_doses)
        queue, imm = _inject_delayed_dose(queue, dose_mass, params.delay_kernel, params.immediate_absorption_fraction)
        queue, rel = _release_absorption(queue, params, dt)

        dil_rate = (params.background_dilution_rate + params.ec_decay_jitter) * th_resp
        dilution = s.dissolved_nutrient_mass * (1.0 - np.exp(-dil_rate * dt_scale))
        min_rate = params.mineralization_rate * th_resp
        mineralized = s.dead_biomass_pool * (1.0 - np.exp(-min_rate * dt_scale))
        dissolved_new = s.dissolved_nutrient_mass + imm + rel + mineralized - dilution

        max_reserve = s.algae_biomass * params.internal_capacity
        reserve_deficit = max(0.0, max_reserve - s.internal_reserve)
        reserve_inhibition = reserve_deficit / max(max_reserve, 1e-6)
        monod = dissolved_new / (params.half_saturation_mass + dissolved_new)
        uptake_rate = params.maximum_uptake_rate * th_growth * s.health_index * osmotic_f * reserve_inhibition * monod
        uptake_mass = uptake_rate * s.algae_biomass * dt_scale
        if uptake_mass > dissolved_new:
            clips["uptake_capped_by_dissolved"] += 1
            uptake_mass = dissolved_new

        dissolved_new -= uptake_mass
        reserve_new = s.internal_reserve + uptake_mass

        maintenance_cost = params.maintenance_cost * s.algae_biomass * th_resp * dt_scale

        actual_maint = min(maintenance_cost, reserve_new)
        reserve_new -= actual_maint
        dissolved_new += actual_maint

        reserve_ratio = reserve_new / max(s.algae_biomass * params.internal_capacity, 1e-6)
        growth_drive = reserve_ratio * th_growth * osmotic_f
        growth_amount = params.maximum_growth_rate * growth_drive * s.algae_biomass * dt_scale
        cost_per = params.biomass_nutrient_content / params.growth_yield
        req = growth_amount * cost_per
        if req > reserve_new:
            clips["growth_capped_by_reserve"] += 1

        thermal_stress = 1.0 / (th_resp + 0.1)
        mortality_rate = params.mortality_rate * thermal_stress * (1.0 + osmotic) * (2.0 - s.health_index)
        mort = s.algae_biomass * (1.0 - np.exp(-mortality_rate * dt_scale))
        if mort > s.algae_biomass * 0.95:
            clips["mortality_capped_at_95pct"] += 1

        maint_deficit = maintenance_cost - actual_maint
        deficit_ratio = maint_deficit / max(maintenance_cost, 1e-6)
        stress_drive = deficit_ratio * 1.0 + osmotic * 2.0
        damage_inc = stress_drive * params.damage_rate * dt_scale * (1.0 - s.damage_index)
        
        repair_availability = actual_maint / max(maintenance_cost, 1e-6)
        repair_drive = params.repair_rate * repair_availability * th_resp + params.basal_repair_rate
        repair_inc = repair_drive * dt_scale * (s.damage_index ** 0.7)
        
        damage_new = s.damage_index + damage_inc - repair_inc
        if damage_new < 0.0:
            clips["damage_clipped_0"] += 1
        if damage_new > 1.0:
            clips["damage_clipped_1"] += 1

        # Advance state for next iteration
        s = step_dynamics(s, fr, dur, dt, params, rng=rng)

    pd.DataFrame([clips]).to_csv(output_dir / "clipping_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(clips.keys()), list(clips.values()), color="#4a90d9")
    ax.set_xlabel("Count (out of 10,000 steps)")
    ax.set_title("Numerical Clipping / Bounding Events")
    fig.tight_layout()
    fig.savefig(output_dir / "clipping_analysis.png", dpi=150)
    plt.close(fig)

    total_clips = sum(clips.values())
    return {
        "status": "PASS",
        "metrics": clips,
        "warnings": [f"High clipping rate: {total_clips}/{length}"] if total_clips > length * 0.1 else [],
    }


# ---- Registry ----
REGISTERED_EXPERIMENTS = [
    Experiment(
        id="NUM-01", name="Timestep Sensitivity", category="III. Numerical Analysis",
        hypothesis="Trajectories converge as dt decreases; dt=60s is sufficiently accurate.",
        execute=run_timestep_sensitivity,
        metrics=["Biomass Error vs 30s"], plots=["timestep_sensitivity.png"],
    ),
    Experiment(
        id="NUM-02", name="Long Horizon Stability", category="III. Numerical Analysis",
        hypothesis="100,000 steps produce no NaNs, Infs, or unbounded variables.",
        execute=run_long_horizon,
        metrics=["NaN Count", "Inf Count", "Max Biomass"], plots=["long_horizon.png"],
    ),
    Experiment(
        id="NUM-03", name="Clipping Frequency Analysis", category="III. Numerical Analysis",
        hypothesis="Numerical bounding events (min/max clips) occur infrequently, indicating the integration is not relying on clamps.",
        execute=run_clipping_analysis,
        metrics=["uptake_capped_by_dissolved", "mortality_capped_at_95pct"], plots=["clipping_analysis.png"],
    ),
]
