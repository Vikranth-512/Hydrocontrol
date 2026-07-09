"""
Validate v4 simulator: mechanistic biological process model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from simulation.dynamics import (
    TankDynamicsParams,
    TankState,
    simulate_open_loop,
    step_dynamics,
)


def run_validation_suite(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = config.get("simulation", {})
    dyn = config.get("dynamics", {})
    dt = sim.get("dt_seconds", 60.0)
    params = TankDynamicsParams.from_config(dyn)
    
    results: Dict[str, Any] = {}

    # 1. Open-loop nutrient depletion & Starvation dynamics
    length = 500
    t_axis = np.arange(length) * dt / 60.0
    s0 = TankState.create_initial(params, dissolved_mass=3.0, biomass=100.0)
    hist_none = simulate_open_loop(params, length, dt, actions=[(0.0, 0.0)] * length, initial_state=s0)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t_axis, hist_none["dissolved_nutrient_mass"], label="Dissolved Nutrient")
    axes[0].set_ylabel("Mass")
    axes[0].set_title("Open-loop: Depletion & Starvation")
    axes[0].legend()
    axes[1].plot(t_axis, hist_none["internal_reserve"], color="orange", label="Internal Reserve")
    axes[1].set_ylabel("Mass")
    axes[1].legend()
    axes[2].plot(t_axis, hist_none["algae_biomass"], color="green", label="Biomass")
    axes[2].plot(t_axis, hist_none["dead_biomass_pool"], color="gray", label="Dead Pool")
    axes[2].set_ylabel("Mass")
    axes[2].set_xlabel("Time (min)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "01_depletion_starvation.png", dpi=150)
    plt.close(fig)

    results["starvation_reserve_drop"] = float(hist_none["internal_reserve"][0] - min(hist_none["internal_reserve"]))
    results["starvation_biomass_drop"] = float(hist_none["algae_biomass"][0] - hist_none["algae_biomass"][-1])

    # 2. Mass conservation
    # We dose randomly and check if total_mass is conserved
    rng = np.random.default_rng(42)
    s = TankState.create_initial(params, dissolved_mass=1.5, biomass=80.0)
    mass_errors = []
    
    for t in range(200):
        fr = rng.uniform(0.0, 5.0)
        dur = rng.uniform(0.0, 30.0)
        
        expected_total = (
            s.dissolved_nutrient_mass 
            + s.internal_reserve 
            + s.algae_biomass * params.biomass_nutrient_content 
            + s.dead_biomass_pool 
            + s.pending_nutrients
            + s.cumulative_dilution
            + (fr * dur / 60.0) # dose
        )
        s = step_dynamics(s, fr, dur, dt, params)
        actual_total = s.compute_total_mass(params)
        
        mass_errors.append(abs(expected_total - actual_total))
        
    max_error = float(np.max(mass_errors))
    results["max_mass_conservation_error"] = max_error
    
    # 3. High-dose response (Osmotic Stress)
    length_high = 200
    t_axis_high = np.arange(length_high) * dt / 60.0
    s_curr = TankState.create_initial(params, dissolved_mass=1.5, biomass=80.0)
    
    # We will run manually to capture fluxes
    hist_high = {
        "dissolved_nutrient_mass": [],
        "algae_biomass": [],
        "health_index": [],
        "internal_reserve": [],
        "ec": [],
        "uptake_rate": [],
        "maintenance_flux": [],
        "mortality_flux": []
    }
    
    for t in range(length_high):
        s_next = step_dynamics(s_curr, 5.0, 30.0, dt, params)
        hist_high["dissolved_nutrient_mass"].append(float(s_next.dissolved_nutrient_mass))
        hist_high["algae_biomass"].append(float(s_next.algae_biomass))
        hist_high["health_index"].append(float(s_next.health_index))
        hist_high["internal_reserve"].append(float(s_next.internal_reserve))
        hist_high["ec"].append(float(s_next.ec))
        
        # Calculate fluxes (per min)
        # We can approximate them by looking at the change if we isolate it, but
        # since we just need the rate, we can re-calculate the expected rate based on the states
        # or just use the differences. Wait, actually, let's calculate the exact rates at s_curr
        # Uptake
        osmotic = (s_curr.dissolved_nutrient_mass / params.osmotic_half_effect) ** 2
        osmotic_factor = 1.0 / (1.0 + osmotic)
        light_factor = params.light_attenuation_half_mass**2 / (params.light_attenuation_half_mass**2 + s_curr.algae_biomass**2)
        uptake = params.maximum_uptake_rate * (s_curr.dissolved_nutrient_mass / (params.half_saturation_mass + s_curr.dissolved_nutrient_mass)) * osmotic_factor * s_curr.algae_biomass
        
        # Maintenance
        maint = params.maintenance_cost * s_curr.algae_biomass
        
        # Mortality
        mort = params.mortality_rate * np.log1p(osmotic) * max(1.0, (10.0 * s_curr.damage_index)**2) * s_curr.algae_biomass
        
        hist_high["uptake_rate"].append(float(uptake))
        hist_high["maintenance_flux"].append(float(maint))
        hist_high["mortality_flux"].append(float(mort))
        
        s_curr = s_next
        
    for k in hist_high:
        hist_high[k] = np.array(hist_high[k])
    
    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    axes[0].plot(t_axis_high, hist_high["ec"], label="EC Sensor")
    axes[0].plot(t_axis_high, hist_high["dissolved_nutrient_mass"], label="Dissolved Nutrient")
    axes[0].set_ylabel("Mass / EC")
    axes[0].set_title("Continuous High Dosing: Accumulation & Osmotic Stress")
    axes[0].legend()
    
    axes[1].plot(t_axis_high, hist_high["internal_reserve"], color="orange", label="Internal Reserve")
    axes[1].set_ylabel("Mass")
    axes[1].legend()
    
    axes[2].plot(t_axis_high, hist_high["health_index"], color="red", label="Health Index")
    axes[2].set_ylabel("Index (0-1)")
    axes[2].legend()
    
    axes[3].plot(t_axis_high, hist_high["algae_biomass"], color="green", label="Biomass")
    axes[3].set_ylabel("Mass")
    axes[3].legend()
    
    axes[4].plot(t_axis_high, hist_high["uptake_rate"], color="blue", label="Uptake Rate")
    axes[4].plot(t_axis_high, hist_high["maintenance_flux"], color="brown", linestyle="--", label="Maintenance Req")
    axes[4].set_ylabel("Rate (mass/min)")
    axes[4].legend()
    
    axes[5].plot(t_axis_high, hist_high["mortality_flux"], color="purple", label="Mortality Rate")
    axes[5].set_ylabel("Rate (mass/min)")
    axes[5].set_xlabel("Time (min)")
    axes[5].legend()
    
    fig.tight_layout()
    fig.savefig(output_dir / "03_high_dose_stress.png", dpi=150)
    plt.close(fig)
    
    results["high_dose_final_dissolved"] = float(hist_high["dissolved_nutrient_mass"][-1])
    results["high_dose_health_drop"] = float(hist_high["health_index"][0] - hist_high["health_index"][-1])

    # 4. Temperature dependence
    temps = np.linspace(15, 35, 21)
    uptake_rates = []
    growth_rates = []
    
    for temp in temps:
        s = TankState.create_initial(params, dissolved_mass=5.0, biomass=100.0, water_temp=temp)
        # 1 step to see rates
        s_next = step_dynamics(s, 0.0, 0.0, dt, params)
        uptake_rates.append((s_next.internal_reserve - s.internal_reserve) / (dt / 60.0))
        growth_rates.append((s_next.algae_biomass - s.algae_biomass) / (dt / 60.0))
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temps, uptake_rates, "o-", label="Uptake Rate")
    ax.plot(temps, growth_rates, "s-", label="Growth Rate")
    ax.axvline(params.temp_opt, color="r", linestyle="--", label="Optimal Temp")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Rate (mass/min)")
    ax.set_title("Temperature Dependence of Biological Rates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_temperature_dependence.png", dpi=150)
    plt.close(fig)

    # 5. Monod kinetics saturation
    concs = np.linspace(0, 10.0, 50)
    uptake_monod = []
    for c in concs:
        s = TankState.create_initial(params, dissolved_mass=c, biomass=100.0)
        s.internal_reserve = 0.0 # Empty reserve for max uptake
        s_next = step_dynamics(s, 0.0, 0.0, dt, params)
        uptake_monod.append((s_next.internal_reserve - s.internal_reserve) / (dt / 60.0))
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(concs, uptake_monod, "-", label="Uptake Rate vs Concentration")
    ax.axvline(params.half_saturation_mass, color="gray", linestyle=":", label="Km (Half-sat)")
    ax.set_xlabel("Dissolved Nutrient Mass")
    ax.set_ylabel("Uptake Rate")
    ax.set_title("Monod Kinetics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "05_monod_kinetics.png", dpi=150)
    plt.close(fig)

    # 6 & 7. Recovery dynamics
    length_rec = 300
    t_axis_rec = np.arange(length_rec) * dt / 60.0
    s0_rec = TankState.create_initial(params, dissolved_mass=0.1, biomass=80.0)
    s0_rec.internal_reserve = 1.0 # Starved
    # Dose at t=50
    actions_rec = [(0.0, 0.0)] * length_rec
    actions_rec[50] = (5.0, 30.0)
    
    hist_rec = simulate_open_loop(params, length_rec, dt, actions=actions_rec, initial_state=s0_rec)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t_axis_rec, hist_rec["dissolved_nutrient_mass"], label="Dissolved Nutrient")
    axes[0].plot(t_axis_rec, hist_rec["internal_reserve"], label="Internal Reserve")
    axes[0].set_ylabel("Mass")
    axes[0].set_title("Recovery Dynamics after Starvation")
    axes[0].legend()
    axes[1].plot(t_axis_rec, hist_rec["algae_biomass"], color="green", label="Biomass")
    axes[1].set_ylabel("Mass")
    axes[1].set_xlabel("Time (min)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "06_recovery_dynamics.png", dpi=150)
    plt.close(fig)
    
    # 8. Sensor validation
    ec_ratio_errs = np.abs(hist_rec["ec"] / (hist_rec["dissolved_nutrient_mass"] + 1e-9) - params.sensor_gain_ec)
    max_ec_ratio_err = float(np.max(ec_ratio_errs[hist_rec["dissolved_nutrient_mass"] > 0.1]))
    results["max_ec_sensor_error"] = max_ec_ratio_err
    
    # Validation constraints
    results["validation_passed"] = (
        max_error < 1e-9
        and results["starvation_reserve_drop"] > 5.0
        and results["high_dose_final_dissolved"] > 10.0
        and results["high_dose_health_drop"] > 0.1
        and max_ec_ratio_err < 0.05
    )

    with open(output_dir / "validation_summary.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    return results

if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_validation_suite(config, Path("diagnostics/dynamics_v4"))
