"""
Supervisory Policy Sensitivity Analysis (ORC-04)

Quantifies the sensitivity of the rollout-selected supervisory policy to:
1. Individual objective-function weights (0.5x and 2.0x perturbations)
2. Rollout horizon H in {25, 50, 100, 150, 200}

Evaluated on a fixed set of representative states sampled from the benchmark dataset.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation.dynamics import (
    TankDynamicsParams,
    TankState,
    q10_metabolism_factor,
    step_dynamics,
)
from simulation.labeling_objective import (
    CostBreakdown,
    LabelingObjectiveConfig,
    evaluate_rollout,
)
from simulation.optimization_labeler import OptimizationLabeler
from simulation.validate.common import Experiment
from utils import load_config


# ---------------------------------------------------------------------------
# Representative State Sampling
# ---------------------------------------------------------------------------

def select_representative_states(
    dataset_path: Path,
    n_states: int = 500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[TankState]]:
    """
    Sample a fixed set of representative states from the benchmark dataset.
    
    Retains scenario and trajectory metadata for reproducibility.
    """
    df = pd.read_csv(dataset_path)
    if len(df) < n_states:
        raise ValueError(f"Dataset has only {len(df)} rows; cannot sample {n_states}.")

    rng = np.random.default_rng(seed)
    sampled_indices = np.sort(rng.choice(len(df), size=n_states, replace=False))
    sample_df = df.iloc[sampled_indices].copy().reset_index()
    sample_df.rename(columns={"index": "source_row_index"}, inplace=True)

    # Convert rows to TankState using existing OptimizationLabeler helper
    dummy_config: Dict[str, Any] = {"labeling": {}, "simulation": {}, "dynamics": {}}
    dummy_labeler = OptimizationLabeler(dummy_config)

    states: List[TankState] = []
    for _, row in sample_df.iterrows():
        s = dummy_labeler._row_to_state(row)
        states.append(s)

    return sample_df, states


# ---------------------------------------------------------------------------
# Worker for precomputing candidate rollouts per state
# ---------------------------------------------------------------------------

def _rollout_candidate_200(
    s0: TankState,
    fr: float,
    dur: float,
    dt: float,
    params: TankDynamicsParams,
    ic: float,
    max_h: int = 200,
) -> Tuple[List[Dict[str, float]], Dict[int, TankState]]:
    """
    Simulate single candidate action for up to max_h steps.
    Returns (eco_trace, dict of final_states keyed by horizon step).
    """
    eco_trace: List[Dict[str, float]] = []
    horizon_checkpoints = {25, 50, 100, 150, 200}
    final_states: Dict[int, TankState] = {}

    s = s0
    for k in range(max_h):
        fr_k = fr if k == 0 else 0.0
        dur_k = dur if k == 0 else 0.0

        s = step_dynamics(s, fr_k, dur_k, dt, params)
        B = max(s.algae_biomass, 1e-6)
        rho = float(np.clip(s.internal_reserve / (B * ic), 0.0, 1.0))
        dose = fr_k * dur_k / 60.0
        thermal_resp = q10_metabolism_factor(s.water_temp, params)
        maint_cost = params.maintenance_cost * s.algae_biomass * thermal_resp * (dt / 60.0)
        osmotic_stress = (s.dissolved_nutrient_mass / params.osmotic_half_effect) ** 2

        eco_trace.append({
            "ec": s.ec,
            "rho": rho,
            "health": s.health_index,
            "damage": s.damage_index,
            "biomass": s.algae_biomass,
            "dose": dose,
            "dissolved": s.dissolved_nutrient_mass,
            "maint_cost": maint_cost,
            "osmotic": osmotic_stress,
            "detritus": s.dead_biomass_pool,
        })

        step_num = k + 1
        if step_num in horizon_checkpoints:
            final_states[step_num] = s

    return eco_trace, final_states


def _evaluate_single_state_candidates(
    args: Tuple[int, TankState, List[Tuple[float, float]], float, TankDynamicsParams, Dict[str, Any]],
) -> Tuple[int, Dict[int, List[CostBreakdown]], Dict[int, Tuple[float, float]]]:
    """
    Worker task: runs all 61 candidates for one state across horizons.
    Returns (state_idx, breakdowns_by_h, baseline_actions_by_h).
    """
    state_idx, s0, candidates, dt, params, cfg_dict = args
    base_obj_cfg = LabelingObjectiveConfig.from_yaml(
        cfg_dict.get("labeling", {}),
        cfg_dict.get("simulation", {}),
        cfg_dict.get("dynamics", {}),
        eco=cfg_dict.get("ecosystem_control", {}),
    )
    ic = base_obj_cfg.internal_capacity
    tie_break_epsilon = float(cfg_dict.get("labeling", {}).get("tie_break_epsilon", 0.005))
    horizons = [25, 50, 100, 150, 200]

    # Precompute candidate traces up to 200 steps
    # For each candidate, evaluate CostBreakdown for each horizon
    breakdowns_by_h: Dict[int, List[CostBreakdown]] = {h: [] for h in horizons}
    candidate_scores_by_h: Dict[int, List[float]] = {h: [] for h in horizons}

    for fr, dur in candidates:
        eco_trace, final_states = _rollout_candidate_200(s0, fr, dur, dt, params, ic, max_h=200)
        for h in horizons:
            s_h = final_states[h]
            trace_h = eco_trace[:h]
            bd = evaluate_rollout(trace_h, s_h, s0, fr, dur, base_obj_cfg)
            breakdowns_by_h[h].append(bd)
            candidate_scores_by_h[h].append(bd.total)

    # Pick baseline action for each horizon
    actions_by_h: Dict[int, Tuple[float, float]] = {}
    for h in horizons:
        scores = candidate_scores_by_h[h]
        best_idx = int(np.argmin(scores))
        best_score = scores[best_idx]

        if tie_break_epsilon > 0:
            threshold = best_score * (1.0 + tie_break_epsilon)
            tied_indices = [idx for idx, sc in enumerate(scores) if sc <= threshold]
            best_idx = min(tied_indices, key=lambda idx: candidates[idx][0] * candidates[idx][1])

        actions_by_h[h] = candidates[best_idx]

    return state_idx, breakdowns_by_h, actions_by_h


# ---------------------------------------------------------------------------
# Policy Metrics Computation
# ---------------------------------------------------------------------------

def compute_policy_comparison_metrics(
    baseline_actions: List[Tuple[float, float]],
    tested_actions: List[Tuple[float, float]],
) -> Dict[str, float]:
    """
    Calculate policy comparison metrics between baseline and tested policies:
    - exact label agreement
    - no-action fraction
    - intervention fraction
    - intervention decision change rate
    - conditional flowrate MAE (where baseline action is non-zero)
    - conditional duration MAE (where baseline action is non-zero)
    """
    N = len(baseline_actions)
    if N == 0:
        return {}

    b_arr = np.array(baseline_actions, dtype=np.float64)
    t_arr = np.array(tested_actions, dtype=np.float64)

    # Exact agreement
    exact_matches = np.all(b_arr == t_arr, axis=1)
    label_agreement = float(np.mean(exact_matches))

    # No-action / Intervention
    t_is_zero = (t_arr[:, 0] == 0.0) & (t_arr[:, 1] == 0.0)
    no_action_frac = float(np.mean(t_is_zero))
    intervention_frac = float(1.0 - no_action_frac)

    # Intervention decision change rate
    b_is_zero = (b_arr[:, 0] == 0.0) & (b_arr[:, 1] == 0.0)
    decision_change = (b_is_zero & ~t_is_zero) | (~b_is_zero & t_is_zero)
    decision_change_rate = float(np.mean(decision_change))

    # Conditional flowrate & duration MAE (where baseline action > 0)
    b_intervene_mask = ~b_is_zero
    if np.any(b_intervene_mask):
        cond_fr_diff = np.abs(b_arr[b_intervene_mask, 0] - t_arr[b_intervene_mask, 0])
        cond_dur_diff = np.abs(b_arr[b_intervene_mask, 1] - t_arr[b_intervene_mask, 1])
        cond_fr_mae = float(np.mean(cond_fr_diff))
        cond_dur_mae = float(np.mean(cond_dur_diff))
    else:
        cond_fr_mae = 0.0
        cond_dur_mae = 0.0

    return {
        "label_agreement": label_agreement,
        "label_agreement_pct": label_agreement * 100.0,
        "no_action_fraction": no_action_frac,
        "intervention_fraction": intervention_frac,
        "intervention_decision_change_rate": decision_change_rate,
        "conditional_flowrate_mae": cond_fr_mae,
        "conditional_duration_mae": cond_dur_mae,
    }


# ---------------------------------------------------------------------------
# Sensitivity Experiment Runner
# ---------------------------------------------------------------------------

def run_supervisory_policy_sensitivity_analysis(
    config: Dict[str, Any],
    dataset_path: Optional[Path] = None,
    n_states: int = 500,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute full Supervisory Policy Sensitivity Analysis.
    """
    if dataset_path is None:
        dataset_path = Path("data/processed/all_trajectories_labeled.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Benchmark labeled dataset not found at {dataset_path}")

    if output_dir is None:
        output_dir = Path("reports/supervisory_policy_sensitivity")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample representative states
    sample_df, states = select_representative_states(dataset_path, n_states=n_states, seed=seed)
    
    # Save sampled states metadata
    states_csv_path = output_dir / "selected_representative_states.csv"
    meta_cols = [c for c in sample_df.columns if c in [
        "source_row_index", "trajectory_id", "timestep", "scenario",
        "optimal_flowrate", "optimal_duration", "label_score",
        "true_dissolved_nutrient_mass", "true_algae_biomass", "true_internal_reserve",
        "true_health_index", "true_damage_index", "true_ec", "true_water_temp"
    ]]
    sample_df[meta_cols].to_csv(states_csv_path, index=False)

    # 2. Build candidate actions
    labeler = OptimizationLabeler(config)
    candidates = labeler.candidates
    dt = labeler.dt
    params = labeler.base_params
    base_obj_cfg = labeler.obj_cfg
    tie_break_epsilon = labeler.tie_break_epsilon

    # 3. Precompute candidate rollouts across representative states
    num_cpus = num_workers or min(os.cpu_count() or 4, 12)
    tasks = [(i, s, candidates, dt, params, config) for i, s in enumerate(states)]

    print(f"Precomputing 200-step rollouts for {n_states} states across {len(candidates)} actions using {num_cpus} workers...")
    
    if num_cpus > 1 and len(states) > 1:
        ctx = mp.get_context("spawn") if os.name == "nt" else mp.get_context()
        with ctx.Pool(processes=num_cpus) as pool:
            results = pool.map(_evaluate_single_state_candidates, tasks)
    else:
        results = [_evaluate_single_state_candidates(t) for t in tasks]

    # Sort results by state index
    results.sort(key=lambda x: x[0])
    breakdowns_by_state = [r[1] for r in results]
    baseline_actions_by_h = {
        h: [r[2][h] for r in results]
        for h in [25, 50, 100, 150, 200]
    }

    baseline_actions_h100 = baseline_actions_by_h[100]

    # Baseline stats
    baseline_no_action = float(np.mean([a == (0.0, 0.0) for a in baseline_actions_h100]))
    baseline_intervention = 1.0 - baseline_no_action

    # 4. Objective-Weight Sensitivity (H=100)
    # Sourced directly from implementation config
    active_weights: Dict[str, float] = config.get("labeling", {}).get("weights", {})
    if not active_weights:
        # Fallback to defaults from LabelingObjectiveConfig dataclass
        active_weights = {
            "health": base_obj_cfg.health,
            "damage": base_obj_cfg.damage,
            "reserve": base_obj_cfg.reserve,
            "biomass_rate": base_obj_cfg.biomass_rate,
            "biomass_envelope": base_obj_cfg.biomass_envelope,
            "maintenance": base_obj_cfg.maintenance,
            "detritus": base_obj_cfg.detritus,
            "dissolved_inventory": base_obj_cfg.dissolved_inventory,
            "dissolved_accumulation": base_obj_cfg.dissolved_accumulation,
            "osmotic": base_obj_cfg.osmotic,
            "nutrient_cost": base_obj_cfg.nutrient_cost,
            "action_smoothness": base_obj_cfg.action_smoothness,
        }

    obj_rows: List[Dict[str, Any]] = []

    print(f"Evaluating objective weight perturbations across {len(active_weights)} configurable terms...")

    for term_name, base_weight in active_weights.items():
        if not hasattr(base_obj_cfg, term_name):
            continue

        for mult in [0.5, 2.0]:
            perturbed_val = base_weight * mult
            perturbed_cfg = copy.deepcopy(base_obj_cfg)
            setattr(perturbed_cfg, term_name, perturbed_val)

            # Re-evaluate best action for each state
            perturbed_actions: List[Tuple[float, float]] = []
            for s_idx in range(n_states):
                h100_breakdowns = breakdowns_by_state[s_idx][100]
                scores = [bd.weighted_total(perturbed_cfg) for bd in h100_breakdowns]
                best_idx = int(np.argmin(scores))
                best_score = scores[best_idx]

                if tie_break_epsilon > 0:
                    threshold = best_score * (1.0 + tie_break_epsilon)
                    tied_indices = [idx for idx, sc in enumerate(scores) if sc <= threshold]
                    best_idx = min(tied_indices, key=lambda idx: candidates[idx][0] * candidates[idx][1])

                perturbed_actions.append(candidates[best_idx])

            m = compute_policy_comparison_metrics(baseline_actions_h100, perturbed_actions)
            row = {
                "objective_term": term_name,
                "perturbation": f"{mult}x",
                "perturbation_factor": mult,
                "baseline_weight": base_weight,
                "tested_weight": perturbed_val,
                "label_agreement": m["label_agreement"],
                "label_agreement_pct": m["label_agreement_pct"],
                "no_action_fraction": m["no_action_fraction"],
                "intervention_fraction": m["intervention_fraction"],
                "intervention_decision_change_rate": m["intervention_decision_change_rate"],
                "conditional_flowrate_mae": m["conditional_flowrate_mae"],
                "conditional_duration_mae": m["conditional_duration_mae"],
            }
            obj_rows.append(row)

    obj_df = pd.DataFrame(obj_rows)
    obj_csv_path = output_dir / "objective_weight_sensitivity.csv"
    obj_df.to_csv(obj_csv_path, index=False)

    # 5. Rollout Horizon Sensitivity
    print("Evaluating rollout horizon sensitivity across H in {25, 50, 100, 150, 200}...")
    horizon_rows: List[Dict[str, Any]] = []

    for h in [25, 50, 100, 150, 200]:
        h_actions = baseline_actions_by_h[h]
        m = compute_policy_comparison_metrics(baseline_actions_h100, h_actions)
        row = {
            "horizon": h,
            "label_agreement_with_H100": m["label_agreement"],
            "label_agreement_with_H100_pct": m["label_agreement_pct"],
            "no_action_fraction": m["no_action_fraction"],
            "intervention_fraction": m["intervention_fraction"],
            "intervention_decision_change_rate": m["intervention_decision_change_rate"],
            "conditional_flowrate_mae": m["conditional_flowrate_mae"],
            "conditional_duration_mae": m["conditional_duration_mae"],
        }
        horizon_rows.append(row)

    horizon_df = pd.DataFrame(horizon_rows)
    horizon_csv_path = output_dir / "horizon_sensitivity.csv"
    horizon_df.to_csv(horizon_csv_path, index=False)

    # 6. Generate Simple Plots (Two Plots)
    generate_plots(obj_df, horizon_df, output_dir)

    # 7. Generate Summary Report
    summary_text = generate_summary_report(obj_df, horizon_df, n_states, baseline_no_action, baseline_intervention)
    report_path = output_dir / "supervisory_policy_sensitivity_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # 8. Save JSON Summary
    summary_json_path = output_dir / "summary.json"
    summary_dict = {
        "experiment_id": "ORC-04",
        "name": "Supervisory Policy Sensitivity Analysis",
        "n_states": n_states,
        "seed": seed,
        "baseline_horizon": 100,
        "baseline_no_action_fraction": baseline_no_action,
        "baseline_intervention_fraction": baseline_intervention,
        "objective_weights_tested": list(active_weights.keys()),
        "horizons_tested": [25, 50, 100, 150, 200],
        "objective_sensitivity": obj_df.to_dict(orient="records"),
        "horizon_sensitivity": horizon_df.to_dict(orient="records"),
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print(summary_text)

    return {
        "status": "PASS",
        "metrics": {
            "Representative States": n_states,
            "Objective Terms Tested": len(active_weights),
            "Min Agreement (Obj Weights)": f"{obj_df['label_agreement_pct'].min():.1f}%",
            "Max Agreement (Obj Weights)": f"{obj_df['label_agreement_pct'].max():.1f}%",
            "Agreement H=50 vs H=100": f"{horizon_df.loc[horizon_df['horizon'] == 50, 'label_agreement_with_H100_pct'].values[0]:.1f}%",
            "Agreement H=100 vs H=100": f"{horizon_df.loc[horizon_df['horizon'] == 100, 'label_agreement_with_H100_pct'].values[0]:.1f}%",
            "Agreement H=200 vs H=100": f"{horizon_df.loc[horizon_df['horizon'] == 200, 'label_agreement_with_H100_pct'].values[0]:.1f}%",
        },
        "plots": [
            str(output_dir / "objective_weight_sensitivity.png"),
            str(output_dir / "horizon_sensitivity.png"),
        ],
        "summary": summary_dict,
    }


# ---------------------------------------------------------------------------
# Plot Generation (Plot 1 & Plot 2)
# ---------------------------------------------------------------------------

def generate_plots(obj_df: pd.DataFrame, horizon_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot 1: Grouped bar chart of label agreement by objective perturbation (0.5x vs 2.0x).
    Plot 2: Line plot of label agreement vs rollout horizon H.
    """
    # --- Plot 1: Objective-Weight Sensitivity ---
    terms = obj_df["objective_term"].unique()
    x = np.arange(len(terms))
    width = 0.35

    df_05 = obj_df[obj_df["perturbation_factor"] == 0.5].set_index("objective_term").reindex(terms)
    df_20 = obj_df[obj_df["perturbation_factor"] == 2.0].set_index("objective_term").reindex(terms)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rects1 = ax.bar(x - width / 2, df_05["label_agreement_pct"], width, label="0.5× Weight", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x + width / 2, df_20["label_agreement_pct"], width, label="2.0× Weight", color="#d95f02", alpha=0.9)

    ax.set_ylabel("Exact Label Agreement (%)", fontsize=11)
    ax.set_title("Supervisory Policy Sensitivity to Objective Weights", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=35, ha="right", fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.axhline(100.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95)

    # Bar label annotations
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, rotation=45)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, rotation=45)

    fig.tight_layout()
    fig.savefig(output_dir / "objective_weight_sensitivity.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Rollout Horizon Sensitivity ---
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    h_vals = horizon_df["horizon"].values
    agr_vals = horizon_df["label_agreement_with_H100_pct"].values

    ax.plot(h_vals, agr_vals, marker="o", linewidth=2.0, color="#1b9e77", markersize=7, label="Agreement with H=100")
    for h, agr in zip(h_vals, agr_vals):
        ax.annotate(f"{agr:.1f}%", xy=(h, agr), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Rollout Horizon H (steps)", fontsize=11)
    ax.set_ylabel("Label Agreement with H=100 (%)", fontsize=11)
    ax.set_title("Supervisory Policy Sensitivity to Rollout Horizon", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(h_vals)
    ax.set_ylim(min(max(min(agr_vals) - 15, 0), 80), 108)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "horizon_sensitivity.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary Report Text Formatter
# ---------------------------------------------------------------------------

def generate_summary_report(
    obj_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    n_states: int,
    no_action_frac: float,
    intervention_frac: float,
) -> str:
    """Generate clean human-readable summary report matching Section 18."""
    lines = [
        "=" * 60,
        "SUPERVISORY POLICY SENSITIVITY ANALYSIS",
        "=" * 60,
        "",
        "Baseline:",
        "    Rollout horizon: 100 steps",
        f"    Representative states: {n_states}",
        "    Candidate actions: 61",
        "",
        "-" * 60,
        "OBJECTIVE-WEIGHT SENSITIVITY",
        "-" * 60,
        f"{'Objective term':<24} {'Perturbation':<15} {'Agreement':<12}",
        "-" * 60,
    ]

    for _, r in obj_df.iterrows():
        term = r["objective_term"]
        pert = r["perturbation"]
        agr = f"{r['label_agreement_pct']:.2f}%"
        lines.append(f"{term:<24} {pert:<15} {agr:<12}")

    lines.extend([
        "",
        "-" * 60,
        "ROLLOUT-HORIZON SENSITIVITY",
        "-" * 60,
        f"{'Horizon':<14} {'Agreement with H=100':<22}",
        "-" * 60,
    ])

    for _, r in horizon_df.iterrows():
        h = int(r["horizon"])
        agr = f"{r['label_agreement_with_H100_pct']:.2f}%"
        lines.append(f"{h:<14} {agr:<22}")

    lines.extend([
        "",
        "-" * 60,
        "INTERVENTION BEHAVIOUR",
        "-" * 60,
        f"Baseline no-action fraction: {no_action_frac * 100.0:.2f}%",
        f"Baseline intervention fraction: {intervention_frac * 100.0:.2f}%",
        "",
        "-" * 60,
        "CONCLUSION",
        "-" * 60,
    ])

    # Dynamic interpretation based on actual results
    min_obj_agr = obj_df["label_agreement_pct"].min()
    mean_obj_agr = obj_df["label_agreement_pct"].mean()
    h50_agr = horizon_df.loc[horizon_df["horizon"] == 50, "label_agreement_with_H100_pct"].values[0]

    if min_obj_agr >= 80.0:
        obj_interp = "The selected supervisory policy remained largely consistent under the evaluated weight perturbations."
    elif min_obj_agr >= 50.0:
        obj_interp = "The supervisory policy exhibited moderate dependence on specific objective terms (e.g. nutrient/osmotic tradeoffs)."
    else:
        obj_interp = "The supervisory policy was substantially influenced by the corresponding objective configuration."

    if h50_agr >= 85.0:
        h_interp = "Prediction horizon truncations down to 50 steps exhibited high policy consistency with the reference H=100."
    else:
        h_interp = "Prediction horizon length materially determines the long-horizon supervisory trade-offs."

    lines.extend([
        f"{obj_interp}",
        f"{h_interp}",
        "",
        "The sensitivity analysis quantifies the dependence of the",
        "rollout-selected supervisory policy on objective weighting",
        "and prediction horizon. The objective function remains",
        "configurable by design, while the baseline configuration",
        "provides a fixed reproducible reference for the benchmark.",
        "=" * 60,
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation Framework Hook
# ---------------------------------------------------------------------------

def run_experiment_adapter(output_dir: Path, params: TankDynamicsParams) -> Dict[str, Any]:
    """Hook executed by validation runner."""
    config_path = Path("configs/default.yaml")
    config = load_config(config_path) if config_path.exists() else {}
    return run_supervisory_policy_sensitivity_analysis(
        config=config,
        output_dir=output_dir,
    )


REGISTERED_EXPERIMENTS = [
    Experiment(
        id="ORC-04",
        name="Supervisory Policy Sensitivity Analysis",
        category="IX. Supervisory Policy Robustness",
        hypothesis=(
            "The rollout-selected supervisory policy exhibits quantified sensitivity "
            "to objective weight variations and rollout horizons, confirming that configurable "
            "weights and prediction horizons systematically guide supervisory labels."
        ),
        execute=run_experiment_adapter,
        metrics=[
            "Representative States",
            "Min Agreement (Obj Weights)",
            "Agreement H=50 vs H=100",
            "Agreement H=200 vs H=100",
        ],
        plots=[
            "objective_weight_sensitivity.png",
            "horizon_sensitivity.png",
        ],
    ),
]
