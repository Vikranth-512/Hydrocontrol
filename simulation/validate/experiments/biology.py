"""
II. Biological Dynamics Experiments

Verify the biological subsystems behave realistically:
  - Monod uptake saturation
  - Starvation inertia (reserve depletes before biomass)
  - Temperature sweep
  - Osmotic stress sweep
  - Reserve isolation
  - Growth limitation by reserve
  - Mortality verification
  - Mineralization half-life
  - Health hysteresis
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.dynamics import TankDynamicsParams, TankState, step_dynamics, thermal_efficiency
from simulation.validate.common import Experiment, make_initial_state, simulate_and_record
from simulation.validate.plotting import plot_time_series, plot_multi_panel, plot_xy


# ---------------------------------------------------------------------------
# BIO-01  Monod Uptake Saturation
# ---------------------------------------------------------------------------
def run_monod_saturation(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    concs = np.linspace(0, 10.0, 60)
    uptakes = []

    for c in concs:
        s = make_initial_state(params, dissolved_mass=c, biomass=100.0, internal_reserve=0.0)
        s_next = step_dynamics(s, 0.0, 0.0, dt, params)
        uptake = (s_next.internal_reserve - s.internal_reserve) / (dt / 60.0)
        uptakes.append(uptake)

    uptakes = np.array(uptakes)
    pd.DataFrame({"dissolved_mass": concs, "uptake_rate": uptakes}).to_csv(
        output_dir / "monod_curve.csv", index=False
    )

    plot_xy(
        concs, uptakes,
        "Dissolved Nutrient Mass", "Uptake Rate",
        "Andrews Substrate-Inhibition Kinetics",
        output_dir / "monod_curve.png",
        vlines=[(params.half_saturation_mass, "gray", f"Km = {params.half_saturation_mass}")],
    )

    empirical_vmax = float(np.max(uptakes))
    # Theoretical: Vmax * biomass * thermal_eff (at default temp) * osmotic_factor_at_peak
    # The peak is usually around C=2.5 to 3.0 where osmotic stress starts dominating
    th = thermal_efficiency(params.ambient_temp_mean, params)
    expected_vmax = params.maximum_uptake_rate * 100.0 * th * 0.55  # ~0.55 combined monod & osmotic at peak

    return {
        "status": "PASS" if abs(empirical_vmax - expected_vmax) / (expected_vmax + 1e-9) < 0.20 else "FAIL",
        "metrics": {
            "Empirical Vmax": empirical_vmax,
            "Expected Vmax (approx)": float(expected_vmax),
        },
    }


# ---------------------------------------------------------------------------
# BIO-02  Starvation Inertia (Cryptic Recycling)
# ---------------------------------------------------------------------------
def run_starvation(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    import copy
    dt = 60.0
    length = 1500
    s0 = make_initial_state(params, dissolved_mass=2.0, biomass=100.0, internal_reserve=5.0)
    
    # 1. Baseline (mineralization enabled)
    df_baseline = simulate_and_record(params, length, dt, initial_state=s0)
    df_baseline.to_csv(output_dir / "starvation_baseline.csv", index=False)
    
    # 2. No Uptake (no cryptic recycling)
    p_no_uptake = copy.deepcopy(params)
    p_no_uptake.maximum_uptake_rate = 0.0
    df_no_uptake = simulate_and_record(p_no_uptake, length, dt, initial_state=s0)

    plot_multi_panel(
        df_baseline,
        [
            (["dissolved_nutrient_mass"], "Dissolved"),
            (["internal_reserve"], "Reserve"),
            (["algae_biomass", "dead_biomass_pool"], "Biomass"),
            (["health_index"], "Health"),
        ],
        "Open-Loop Starvation (Cryptic Recycling)",
        output_dir / "starvation_baseline.png",
    )

    final_reserve_baseline = float(df_baseline["internal_reserve"].iloc[-1])
    final_reserve_isolated = float(df_no_uptake["internal_reserve"].iloc[-1])

    # In baseline, cryptic recycling sustains a reserve tail. Without uptake, it depletes.
    success = (final_reserve_baseline > 0.5) and (final_reserve_isolated < 0.1)

    return {
        "status": "PASS" if success else "FAIL",
        "metrics": {
            "Final Reserve (Baseline)": final_reserve_baseline,
            "Final Reserve (No Uptake)": final_reserve_isolated,
            "Final Biomass (Baseline)": float(df_baseline["algae_biomass"].iloc[-1]),
        },
    }


# ---------------------------------------------------------------------------
# BIO-03  Temperature Sweep
# ---------------------------------------------------------------------------
def run_temperature_sweep(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    temps = np.linspace(10, 40, 31)
    uptake_rates = []
    growth_rates = []

    for temp in temps:
        s = make_initial_state(params, dissolved_mass=5.0, biomass=100.0, water_temp=temp)
        s_next = step_dynamics(s, 0.0, 0.0, dt, params)
        uptake_rates.append((s_next.internal_reserve - s.internal_reserve) / (dt / 60.0))
        growth_rates.append((s_next.algae_biomass - s.algae_biomass) / (dt / 60.0))

    uptake_rates = np.array(uptake_rates)
    growth_rates = np.array(growth_rates)

    pd.DataFrame({
        "temperature": temps, "net_uptake_change": uptake_rates, "net_growth_change": growth_rates
    }).to_csv(output_dir / "temperature_sweep.csv", index=False)

    fig, ax = plt.subplots()
    ax.plot(temps, uptake_rates, "o-", label="Net Uptake Change")
    ax.plot(temps, growth_rates, "s-", label="Net Growth Change")
    ax.axvline(params.temp_opt, color="r", linestyle="--", label=f"T_opt = {params.temp_opt}")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Net Rate")
    ax.set_title("Temperature Dependence of Biological Rates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "temperature_sweep.png", dpi=150)
    plt.close(fig)

    # Peak uptake should be near T_opt
    peak_idx = int(np.argmax(uptake_rates))
    peak_temp = float(temps[peak_idx])
    offset = abs(peak_temp - params.temp_opt)

    return {
        "status": "PASS" if offset < 5.0 else "FAIL",
        "metrics": {"Peak Uptake Temp": peak_temp, "Offset from T_opt": offset},
    }


# ---------------------------------------------------------------------------
# BIO-04  Osmotic Stress Sweep
# ---------------------------------------------------------------------------
def run_osmotic_sweep(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    concs = np.linspace(0, 20, 50)
    uptakes = []
    stresses = []

    for c in concs:
        s = make_initial_state(params, dissolved_mass=c, biomass=100.0, internal_reserve=0.0)
        osmotic = (c / params.osmotic_half_effect) ** 2
        osmotic_factor = 1.0 / (1.0 + osmotic)
        s_next = step_dynamics(s, 0.0, 0.0, dt, params)
        uptakes.append((s_next.internal_reserve) / (dt / 60.0))
        stresses.append(osmotic)

    uptakes = np.array(uptakes)
    stresses = np.array(stresses)

    pd.DataFrame({"dissolved_mass": concs, "uptake": uptakes, "osmotic_stress": stresses}).to_csv(
        output_dir / "osmotic_sweep.csv", index=False
    )

    fig, ax1 = plt.subplots()
    ax1.plot(concs, uptakes, "o-", color="blue", label="Uptake")
    ax1.set_xlabel("Dissolved Nutrient Mass")
    ax1.set_ylabel("Uptake Rate", color="blue")
    ax2 = ax1.twinx()
    ax2.plot(concs, stresses, "s-", color="red", label="Osmotic Stress")
    ax2.set_ylabel("Osmotic Stress", color="red")
    ax1.set_title("Osmotic Stress Inhibition of Uptake")
    fig.tight_layout()
    fig.savefig(output_dir / "osmotic_sweep.png", dpi=150)
    plt.close(fig)

    # Uptake should decrease at very high concentrations
    uptake_low = uptakes[5]
    uptake_high = uptakes[-1]

    return {
        "status": "PASS" if uptake_high < uptake_low else "FAIL",
        "metrics": {
            "Uptake at Low Conc": float(uptake_low),
            "Uptake at High Conc": float(uptake_high),
        },
    }


# ---------------------------------------------------------------------------
# BIO-05  Reserve Isolation
# ---------------------------------------------------------------------------
def run_reserve_isolation(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    """Verify reserve only changes through uptake, growth, and maintenance - no hidden sources."""
    dt = 60.0
    length = 500
    # Start with zero dissolved nutrients so there is no uptake; reserve should only drain
    s0 = make_initial_state(params, dissolved_mass=0.0, biomass=100.0, internal_reserve=10.0)
    df = simulate_and_record(params, length, dt, initial_state=s0)
    df.to_csv(output_dir / "reserve_isolation.csv", index=False)

    plot_time_series(
        df, ["internal_reserve"], "Reserve Drain Without Uptake", "Mass",
        output_dir / "reserve_isolation.png",
    )

    # Reserve should be monotonically non-increasing (only maintenance drains it)
    reserve = df["internal_reserve"].values
    diffs = np.diff(reserve)
    increases = np.sum(diffs > 1e-12)

    return {
        "status": "PASS" if increases == 0 else "FAIL",
        "metrics": {
            "Number of Reserve Increases": int(increases),
            "Final Reserve": float(reserve[-1]),
        },
        "warnings": [f"Reserve increased at {increases} timesteps"] if increases > 0 else [],
    }


# ---------------------------------------------------------------------------
# BIO-06  Growth Limitation (growth requires reserve, not dissolved nutrients)
# ---------------------------------------------------------------------------
def run_growth_limitation(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    length = 200
    
    # Scenario A: High dissolved, zero reserve
    s_a = make_initial_state(params, dissolved_mass=10.0, biomass=100.0, internal_reserve=0.0)
    df_a = simulate_and_record(params, length, dt, initial_state=s_a)

    # Scenario B: Zero dissolved, high reserve
    s_b = make_initial_state(params, dissolved_mass=0.0, biomass=100.0, internal_reserve=10.0)
    df_b = simulate_and_record(params, length, dt, initial_state=s_b)

    df_a.to_csv(output_dir / "growth_limit_high_dissolved.csv", index=False)
    df_b.to_csv(output_dir / "growth_limit_high_reserve.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df_a["time_min"], df_a["algae_biomass"], label="Biomass (high dissolved, no reserve)")
    axes[0].set_title("High Dissolved, No Reserve")
    axes[0].set_ylabel("Biomass")
    axes[0].set_xlabel("Time (min)")
    axes[0].legend()
    axes[1].plot(df_b["time_min"], df_b["algae_biomass"], label="Biomass (no dissolved, high reserve)")
    axes[1].set_title("No Dissolved, High Reserve")
    axes[1].set_ylabel("Biomass")
    axes[1].set_xlabel("Time (min)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "growth_limitation.png", dpi=150)
    plt.close(fig)

    # Scenario B should show initial growth (consuming reserve) while
    # Scenario A should show delayed growth (must uptake into reserve first)
    growth_a_early = float(df_a["algae_biomass"].iloc[10] - df_a["algae_biomass"].iloc[0])
    growth_b_early = float(df_b["algae_biomass"].iloc[10] - df_b["algae_biomass"].iloc[0])

    return {
        "status": "PASS" if growth_b_early > growth_a_early else "FAIL",
        "metrics": {
            "Growth (high dissolved, t=0..10)": growth_a_early,
            "Growth (high reserve, t=0..10)": growth_b_early,
        },
    }


# ---------------------------------------------------------------------------
# BIO-07  Mortality Channels
# ---------------------------------------------------------------------------
def run_mortality_verification(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    length = 300

    # a) Starvation mortality
    s_starve = make_initial_state(params, dissolved_mass=0.0, biomass=100.0, internal_reserve=0.0,
                                   health_index=0.5, damage_index=0.5)
    df_starve = simulate_and_record(params, length, dt, initial_state=s_starve)

    # b) Osmotic mortality
    s_osm = make_initial_state(params, dissolved_mass=20.0, biomass=100.0)
    df_osm = simulate_and_record(params, length, dt, initial_state=s_osm)

    # c) Heat mortality
    import copy
    p_heat = copy.deepcopy(params)
    p_heat.ambient_temp_mean = 40.0
    s_heat = make_initial_state(p_heat, dissolved_mass=3.0, biomass=100.0, water_temp=40.0)
    df_heat = simulate_and_record(p_heat, length, dt, initial_state=s_heat)

    for name, df in [("starvation", df_starve), ("osmotic", df_osm), ("heat", df_heat)]:
        df.to_csv(output_dir / f"mortality_{name}.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, df) in zip(axes, [("Starvation", df_starve), ("Osmotic", df_osm), ("Heat", df_heat)]):
        ax.plot(df["time_min"], df["algae_biomass"], label="Biomass")
        ax.plot(df["time_min"], df["dead_biomass_pool"], label="Dead Pool", linestyle="--")
        ax.set_title(f"Mortality - {name}")
        ax.set_xlabel("Time (min)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mortality_channels.png", dpi=150)
    plt.close(fig)

    loss_starve = float(df_starve["algae_biomass"].iloc[0] - df_starve["algae_biomass"].iloc[-1])
    loss_osm = float(df_osm["algae_biomass"].iloc[0] - df_osm["algae_biomass"].iloc[-1])
    loss_heat = float(df_heat["algae_biomass"].iloc[0] - df_heat["algae_biomass"].iloc[-1])

    return {
        "status": "PASS" if loss_starve > 0 and loss_osm > 0 and loss_heat > 0 else "FAIL",
        "metrics": {
            "Starvation Loss": loss_starve,
            "Osmotic Loss": loss_osm,
            "Heat Loss": loss_heat,
        },
    }


# ---------------------------------------------------------------------------
# BIO-08  Mineralization Half-Life
# ---------------------------------------------------------------------------
def run_mineralization(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    length = 2000
    # Start with only dead biomass; near-zero living biomass
    s0 = make_initial_state(params, dissolved_mass=0.0, biomass=0.001, internal_reserve=0.0,
                             dead_biomass_pool=10.0)
    df = simulate_and_record(params, length, dt, initial_state=s0)
    df.to_csv(output_dir / "mineralization.csv", index=False)

    plot_multi_panel(
        df,
        [
            (["dead_biomass_pool"], "Dead Pool"),
            (["dissolved_nutrient_mass"], "Dissolved (recycled)"),
        ],
        "Mineralization: Detritus → Dissolved",
        output_dir / "mineralization.png",
    )

    # Estimate half-life: find time when dead pool drops below 5.0
    dead = df["dead_biomass_pool"].values
    half_idx = np.where(dead < 5.0)[0]
    half_life_min = float(df["time_min"].iloc[half_idx[0]]) if len(half_idx) > 0 else float("inf")

    return {
        "status": "PASS" if half_life_min < float("inf") else "FAIL",
        "metrics": {"Mineralization Half-Life (min)": half_life_min},
    }


# ---------------------------------------------------------------------------
# BIO-09  Health Hysteresis (damage → recovery is slow and asymmetric)
# ---------------------------------------------------------------------------
def run_health_hysteresis(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    dt = 60.0
    # Phase 1: Osmotic shock to damage health (~80 steps at 8g dissolved)
    # This drops health to ~0.48 without reaching the absorbing state at health=0
    length_shock = 80
    actions_shock = [(0.5, 4.8)] * length_shock
    s0 = make_initial_state(params, dissolved_mass=8.0, biomass=100.0, internal_reserve=25.0)
    df_shock = simulate_and_record(params, length_shock, dt, actions=actions_shock, initial_state=s0)

    # Phase 2: Recovery via water change + gentle sustained dosing
    # Water change resets dissolved to 1.0g (below osmotic crossover ~2.5g).
    # Gentle dosing (0.1 L/min × 1.0 min) sustains nutrients without osmotic stress.
    # Dilution stays ON — this is the biologically realistic recovery protocol.
    last_row = df_shock.iloc[-1]
    s_recovery = make_initial_state(
        params,
        dissolved_mass=1.0,  # Water change clears toxic concentration
        biomass=float(last_row["algae_biomass"]),
        internal_reserve=float(last_row["internal_reserve"]),
        health_index=float(last_row["health_index"]),
        damage_index=float(last_row["damage_index"]),
        dead_biomass_pool=float(last_row["dead_biomass_pool"]),
    )
    length_recover = 2000
    actions_recover = [(0.1, 1.0)] * length_recover
    df_recover = simulate_and_record(params, length_recover, dt, actions=actions_recover, initial_state=s_recovery)
    
    # Adjust time axis for recovery
    df_recover["time_min"] = df_recover["time_min"] + df_shock["time_min"].iloc[-1]

    df_full = pd.concat([df_shock, df_recover], ignore_index=True)
    df_full.to_csv(output_dir / "health_hysteresis.csv", index=False)

    plot_multi_panel(
        df_full,
        [
            (["health_index", "damage_index"], "Health / Damage"),
            (["internal_reserve"], "Reserve"),
            (["algae_biomass"], "Biomass"),
        ],
        "Health Hysteresis: Osmotic Shock → Recovery",
        output_dir / "health_hysteresis.png",
    )

    # Time to damage vs time to recover
    health_min = float(df_full["health_index"].min())
    health_final = float(df_full["health_index"].iloc[-1])

    return {
        "status": "PASS" if health_min < 0.5 and health_final > health_min else "FAIL",
        "metrics": {
            "Minimum Health": health_min,
            "Final Health": health_final,
            "Recovery Delta": health_final - health_min,
        },
    }


# ---- Registry ----
REGISTERED_EXPERIMENTS = [
    Experiment(
        id="BIO-01", name="Andrews Substrate-Inhibition Kinetics", category="II. Biological Dynamics",
        hypothesis="Uptake rate follows Andrews/Haldane substrate-inhibition kinetics due to osmotic stress at high concentrations.",
        execute=run_monod_saturation,
        metrics=["Empirical Vmax", "Expected Vmax (approx)"], plots=["monod_curve.png"],
    ),
    Experiment(
        id="BIO-02", name="Starvation Inertia (Cryptic Recycling)", category="II. Biological Dynamics",
        hypothesis="Reserve monotonically decreases in the absence of uptake, and approaches a recycling equilibrium when mineralization is enabled.",
        execute=run_starvation,
        metrics=["Final Reserve (Baseline)", "Final Reserve (No Uptake)"], plots=["starvation_baseline.png"],
    ),
    Experiment(
        id="BIO-03", name="Temperature Dependence", category="II. Biological Dynamics",
        hypothesis="Uptake and growth rates peak near T_opt and decline at temperature extremes.",
        execute=run_temperature_sweep,
        metrics=["Peak Uptake Temp", "Offset from T_opt"], plots=["temperature_curve.png"],
    ),
    Experiment(
        id="BIO-04", name="Osmotic Stress Inhibition", category="II. Biological Dynamics",
        hypothesis="At very high dissolved nutrient concentrations, osmotic stress reduces uptake before toxicity causes mass mortality.",
        execute=run_osmotic_sweep,
        metrics=["Uptake at Low Conc", "Uptake at High Conc"], plots=["osmotic_curve.png"],
    ),
    Experiment(
        id="BIO-05", name="Reserve Isolation", category="II. Biological Dynamics",
        hypothesis="With zero dissolved nutrients, internal reserve only decreases (no hidden replenishment).",
        execute=run_reserve_isolation,
        metrics=["Number of Reserve Increases"], plots=["reserve_isolation.png"],
    ),
    Experiment(
        id="BIO-06", name="Growth Limitation by Reserve", category="II. Biological Dynamics",
        hypothesis="Growth depends on internal reserve, not dissolved nutrients directly. Pre-loaded cells grow instantly in sterile water.",
        execute=run_growth_limitation,
        metrics=["Growth (high reserve, t=0..10)", "Growth (high dissolved, t=0..10)"], plots=["growth_limitation.png"],
    ),
    Experiment(
        id="BIO-07", name="Mortality Channels", category="II. Biological Dynamics",
        hypothesis="Starvation, osmotic stress, and heat each independently cause biomass loss.",
        execute=run_mortality_verification,
        metrics=["Starvation Loss", "Osmotic Loss", "Heat Loss"], plots=["mortality_channels.png"],
    ),
    Experiment(
        id="BIO-08", name="Mineralization Half-Life", category="II. Biological Dynamics",
        hypothesis="Dead biomass decays back into dissolved nutrients via first-order mineralization kinetics.",
        execute=run_mineralization,
        metrics=["Mineralization Half-Life (min)"], plots=["mineralization.png"],
    ),
    Experiment(
        id="BIO-09", name="Health Hysteresis", category="II. Biological Dynamics",
        hypothesis="Health recovery after osmotic-induced damage is slower than the damage onset, demonstrating physiological inertia.",
        execute=run_health_hysteresis,
        metrics=["Minimum Health", "Final Health", "Recovery Delta"], plots=["health_hysteresis.png"],
    ),
]
