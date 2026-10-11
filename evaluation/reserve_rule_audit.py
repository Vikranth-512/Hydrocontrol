"""
Reserve-Rule Controller Mechanistic Audit.

Traces equations, nutrient mass pathways, mathematical coupling (d_rho / d_B = -rho / B),
controller information accessibility, and representative trajectories across scenarios
using the benchmark's exact MultiSeedEvaluator and CRN protocols.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

import numpy as np
import pandas as pd

from evaluation.multiseed_evaluator import MultiSeedEvaluator
from simulation.dynamics import TankDynamicsParams, TankState
from simulation.state_utils import compute_rho


def run_detailed_trajectory_trace(
    mse: MultiSeedEvaluator,
    scenario: str = "normal",
    seed: int = 42,
    horizon: int = 3500,
) -> Dict[str, Any]:
    """
    Run a detailed mechanistic step-by-step trace of the reserve-ratio controller,
    extracting mass flows, compartment balances, and state dynamics.
    """
    crn = mse.generate_crn_schedule(scenario, seed, horizon)
    h = horizon

    # Run paired episode through evaluator's closed loop runner
    res_rb = mse.closed_loop_eval.run_rule_based(
        scenario=scenario,
        custom_length=h,
        custom_seed=seed,
        prebuilt_disturbances=copy.deepcopy(crn["disturbances"]),
        prebuilt_initial_state=copy.deepcopy(crn["initial_state"]),
        prebuilt_sensor_innovations=copy.deepcopy(crn["sensor_noise_schedule"]),
        prebuilt_sensor_bias=copy.deepcopy(crn["sensor_bias_schedule"]),
    )

    traj = res_rb["trajectory"]
    metrics = res_rb["metrics"]
    state_trace: List[TankState] = traj["state_trace"]
    flowrates = traj["flowrate"]
    durations = traj["duration"]
    dyn_params = mse.params

    # Extract trajectory sequences
    biomass_seq = [float(s.algae_biomass) for s in state_trace]
    reserve_seq = [float(s.internal_reserve) for s in state_trace]
    dissolved_seq = [float(s.dissolved_nutrient_mass) for s in state_trace]
    dead_pool_seq = [float(s.dead_biomass_pool) for s in state_trace]
    health_seq = [float(s.health_index) for s in state_trace]
    ec_seq = [float(s.ec) for s in state_trace]
    water_temp_seq = [float(s.water_temp) for s in state_trace]
    rho_seq = [float(compute_rho(s, dyn_params.internal_capacity)) for s in state_trace]

    # Dosing events
    dosing_pulses = []
    for fr, dur in zip(flowrates, durations):
        if fr > 0.0 and dur > 0.0:
            dosing_pulses.append(fr * dur / 60.0)

    total_dosing_events = len(dosing_pulses)
    cumulative_dose_mass = sum(dosing_pulses)
    mean_pulse = float(np.mean(dosing_pulses)) if dosing_pulses else 0.0
    median_pulse = float(np.median(dosing_pulses)) if dosing_pulses else 0.0

    # Compartment Mass Accounting (initial vs final)
    s0 = state_trace[0]
    s_end = state_trace[-1]

    init_dissolved = float(s0.dissolved_nutrient_mass)
    init_reserve = float(s0.internal_reserve)
    init_biomass_nutrient = float(s0.algae_biomass * dyn_params.biomass_nutrient_content)
    init_dead_nutrient = float(s0.dead_biomass_pool * dyn_params.biomass_nutrient_content)
    init_pending = float(s0.pending_nutrients)
    init_total_mass = init_dissolved + init_reserve + init_biomass_nutrient + init_dead_nutrient + init_pending

    final_dissolved = float(s_end.dissolved_nutrient_mass)
    final_reserve = float(s_end.internal_reserve)
    final_biomass = float(s_end.algae_biomass)
    final_biomass_nutrient = float(s_end.algae_biomass * dyn_params.biomass_nutrient_content)
    final_dead_nutrient = float(s_end.dead_biomass_pool * dyn_params.biomass_nutrient_content)
    final_pending = float(s_end.pending_nutrients)
    final_retained_mass = final_dissolved + final_reserve + final_biomass_nutrient + final_dead_nutrient + final_pending

    cumulative_dilution = float(s_end.cumulative_dilution)
    final_rho = float(compute_rho(s_end, dyn_params.internal_capacity))

    return {
        "scenario": scenario,
        "seed": seed,
        "horizon": horizon,
        "total_dosing_events": total_dosing_events,
        "cumulative_dose_volume_ml": float(cumulative_dose_mass),
        "mean_pulse_volume_ml": mean_pulse,
        "median_pulse_volume_ml": median_pulse,
        "initial_dissolved_nutrient": init_dissolved,
        "initial_reserve": init_reserve,
        "initial_biomass": float(s0.algae_biomass),
        "initial_compartment_mass": init_total_mass,
        "final_biomass": final_biomass,
        "final_reserve_ratio": final_rho,
        "final_dissolved_nutrient": final_dissolved,
        "final_reserve": final_reserve,
        "final_biomass_nutrient": final_biomass_nutrient,
        "final_dead_pool_nutrient": final_dead_nutrient,
        "final_pending_nutrients": final_pending,
        "final_retained_mass": final_retained_mass,
        "cumulative_dilution_mass": cumulative_dilution,
        "biomass_min": float(np.min(biomass_seq)),
        "biomass_max": float(np.max(biomass_seq)),
        "biomass_mean": float(np.mean(biomass_seq)),
        "rho_min": float(np.min(rho_seq)),
        "rho_max": float(np.max(rho_seq)),
        "rho_mean": float(np.mean(rho_seq)),
        "final_health": float(s_end.health_index),
        "final_ec": float(s_end.ec),
        "ec_mae": float(metrics.get("ec_mae", 0.0)),
        "episode_collapsed": bool(final_biomass < 50.0),
    }


def audit_representative_seeds(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Run audits across representative seeds:
    - Normal seed 42 (high biomass success ~279.6 g/L)
    - Normal seed 43 (biomass collapse to ~0.98 g/L)
    - Heatwave seed 42 (high biomass success ~273.5 g/L)
    - Nutrient depletion seed 42 (high biomass success ~279.6 g/L)
    - Cold shock seed 42 (biomass collapse to ~0.90 g/L)
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mse = MultiSeedEvaluator(cfg)

    scenarios_to_test = [
        ("normal", 42, "normal_seed42_success"),
        ("normal", 43, "normal_seed43_collapse"),
        ("heatwave", 42, "heatwave_seed42_success"),
        ("nutrient_depletion", 42, "nutrient_depletion_seed42_success"),
        ("cold_shock", 42, "cold_shock_seed42_collapse"),
    ]

    results = {}
    for sc, seed, tag in scenarios_to_test:
        results[tag] = run_detailed_trajectory_trace(mse, scenario=sc, seed=seed, horizon=3500)

    return results


def get_controller_information_table(include_mpc: bool = False) -> pd.DataFrame:
    """Generate structured Controller Information Access Table."""
    rows = [
        {
            "Controller": "no_action",
            "EC": "No (Ignored)",
            "Turbidity": "No (Ignored)",
            "Biomass B": "No (Ignored)",
            "Reserve R": "No (Ignored)",
            "Reserve Ratio rho": "No (Ignored)",
            "Water Temp": "No (Ignored)",
            "Previous Action": "No (Ignored)",
            "Information Asymmetry / Latent State Access": "Open-loop baseline (zero intervention, receives no state feedback)",
        },
        {
            "Controller": "rule_based_reserve",
            "EC": "No (Ignored in reserve mode)",
            "Turbidity": "No",
            "Biomass B": "Yes (Privileged denominator normalization)",
            "Reserve R": "Yes (Privileged numerator state)",
            "Reserve Ratio rho": "Yes (Direct canonical latent state: R / (B * 0.5))",
            "Water Temp": "No (Temperature-blind)",
            "Previous Action": "Yes (Cooldown timer tsld)",
            "Information Asymmetry / Latent State Access": "Direct access to unobservable true latent quota ratio rho = R / (B * C_int). Completely bypasses sensor noise, sensor bias, and measurement delay.",
        },
        {
            "Controller": "pid",
            "EC": "Yes (Soft ceiling override: throttles pump above EC=1.5)",
            "Turbidity": "No",
            "Biomass B": "Yes (Privileged biomass scaling: min(1.0, B/80))",
            "Reserve R": "Yes (via state object)",
            "Reserve Ratio rho": "Yes (via state object)",
            "Water Temp": "No",
            "Previous Action": "Yes (Internal rate limiting and anti-windup)",
            "Information Asymmetry / Latent State Access": "Receives true state object to compute rho, but adds explicit biomass and EC safety overrides. Bypasses sensor noise for primary controlled variable.",
        },
        {
            "Controller": "lstm",
            "EC": "Yes (Noisy sensor + rolling stats)",
            "Turbidity": "Yes (Noisy optical sensor)",
            "Biomass B": "No (Hidden biological state, unobserved)",
            "Reserve R": "No (Hidden biological state, unobserved)",
            "Reserve Ratio rho": "No (Static fallback 0.5 during evaluation)",
            "Water Temp": "Yes (Noisy sensor)",
            "Previous Action": "Yes (Observed prev_flowrate, prev_dur)",
            "Information Asymmetry / Latent State Access": "Strict partial observability: operates entirely on noisy, biased sensor readings and engineered rolling features. Has NO access to true internal quota or biomass.",
        },
    ]
    if include_mpc:
        rows.append({
            "Controller": "mpc_reserve",
            "EC": "No (Reserve-ratio tracking)",
            "Turbidity": "No",
            "Biomass B": "No (Receives normalized rho directly)",
            "Reserve R": "No (Receives normalized rho directly)",
            "Reserve Ratio rho": "Yes (Primary controlled variable rho in [0, 1])",
            "Water Temp": "No",
            "Previous Action": "Yes (Past commanded actions and cooldown timer)",
            "Information Asymmetry / Latent State Access": "Predictive model-based control using permitted reserve-ratio feedback. Uses online estimated response model over past observations; no access to plant state, absorption queue, or future disturbances.",
        })
    return pd.DataFrame(rows)


def run_and_save_reserve_rule_audit(output_dir: Path | str, config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Execute reserve rule audit and save JSON and CSV reports."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectory_audits = audit_representative_seeds(config_path)
    info_table_df = get_controller_information_table()

    json_path = out_dir / "reserve_rule_trajectory_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(trajectory_audits, f, indent=2)

    csv_path = out_dir / "controller_information_table.csv"
    info_table_df.to_csv(csv_path, index=False)

    return {
        "trajectory_audits": trajectory_audits,
        "information_table": info_table_df.to_dict(orient="records"),
    }
