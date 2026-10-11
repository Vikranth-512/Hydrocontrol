"""
Paired Multi-Seed 3500-Step Statistical Evaluator and ML-Discriminability Audit.

Executes a statistically rigorous evaluation replacing the single-seed 3500-step
comparison with 120 paired scenario-seed episodes across 4 controllers:
  no_action, rule_based_reserve, pid, lstm
under true Common Random Numbers (CRN) with decoupled sensor noise innovations.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml

from evaluation.statistical_analysis import (
    METRIC_SPECS,
    aggregate_cross_scenarios_bootstrap,
    compute_controller_win_rate,
    compute_paired_bootstrap_ci,
    compute_preregistered_threshold_exceedance,
    compute_wilcoxon_paired,
    evaluate_practical_equivalence,
)
from models.lstm_policy import LSTMPolicy
from preprocessing.normalization import FeatureNormalizer
from simulation.disturbances import DisturbanceConfig, DisturbanceGenerator
from simulation.dynamics import TankDynamicsParams, TankState
from simulation.environment import EnvironmentConfig
from simulation.scenario_registry import get_multiseed_evaluation_scenarios
from simulation_runner.closed_loop_eval import ClosedLoopEvaluator


class MultiSeedEvaluator:
    """Coordinates paired multi-seed, multi-regime long-horizon evaluation."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[LSTMPolicy] = None,
        normalizer: Optional[FeatureNormalizer] = None,
        feature_columns: Optional[List[str]] = None,
        device: str = "cpu",
        controllers: Optional[List[str]] = None,
        include_mpc: Optional[bool] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.normalizer = normalizer
        self.feature_columns = feature_columns
        self.device = device

        ev_cfg = config.get("evaluation", {}).get("multiseed_long_horizon", {})
        default_controllers = ["no_action", "rule_based_reserve", "pid", "lstm"]
        should_include_mpc = (
            include_mpc
            if include_mpc is not None
            else (
                ev_cfg.get("include_mpc", False)
                or ("mpc_reserve" in ev_cfg.get("controllers", []))
            )
        )
        if should_include_mpc and "mpc_reserve" not in default_controllers:
            default_controllers = default_controllers + ["mpc_reserve"]
        self.controllers: List[str] = list(controllers or ev_cfg.get("controllers", default_controllers))
        # Use authoritative scenario registry for multiseed evaluation (excludes missing_data)
        registry_scenarios = get_multiseed_evaluation_scenarios()
        config_scenarios = ev_cfg.get("scenarios")
        if config_scenarios is not None:
            valid_scenarios = [s for s in config_scenarios if s in registry_scenarios]
            if not valid_scenarios:
                self.scenarios = registry_scenarios
            else:
                if set(config_scenarios) != set(valid_scenarios):
                    print(f"WARNING: Excluded invalid or deprecated scenarios from config: {set(config_scenarios) - set(valid_scenarios)}")
                self.scenarios = valid_scenarios
        else:
            self.scenarios = registry_scenarios
        self.n_seeds: int = int(ev_cfg.get("n_seeds", 30))
        self.base_seed: int = int(ev_cfg.get("base_seed", 42))
        self.horizon: int = int(ev_cfg.get("horizon", 3500))
        self.bootstrap_replicates: int = int(ev_cfg.get("bootstrap_replicates", 5000))
        self.bootstrap_seed: int = int(ev_cfg.get("bootstrap_seed", 12345))
        self.confidence_level: float = float(ev_cfg.get("confidence_level", 0.95))
        self.practical_thresholds: Dict[str, float] = ev_cfg.get("practical_thresholds", {})

        self.report_dir = Path(ev_cfg.get("report_dir", "reports/multiseed_long_horizon"))
        self.figure_dir = Path(ev_cfg.get("figure_dir", "figures/multiseed_long_horizon"))
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.closed_loop_eval = ClosedLoopEvaluator(config, seed=self.base_seed)
        self.params = self.closed_loop_eval.params
        self.env_config = self.closed_loop_eval.env_config
        self.dist_config = self.closed_loop_eval.disturbance_config
        self.ec_target = self.closed_loop_eval.ec_target

        self.seed_schedule = self._build_seed_schedule()

    def _build_seed_schedule(self) -> Dict[str, List[int]]:
        """Construct deterministic seed schedule for each scenario."""
        schedule: Dict[str, List[int]] = {}
        for s_idx, sc in enumerate(self.scenarios):
            schedule[sc] = [
                int(self.base_seed + 1000 * s_idx + i) for i in range(self.n_seeds)
            ]
        return schedule

    def generate_crn_schedule(
        self,
        scenario: str,
        seed: int,
        length: int,
    ) -> Dict[str, Any]:
        """
        Pre-generate complete Common Random Numbers (CRN) disturbance bundle:
        1. Exogenous disturbances (List[DisturbanceSpec])
        2. Common initial state s0
        3. Exogenous sensor bias sequence
        4. Exogenous sensor noise innovations epsilon_t for y_t = h(x_t) + epsilon_t
        """
        # 1. Exogenous disturbance schedule
        dist_rng = np.random.default_rng(seed)
        dist_gen = DisturbanceGenerator(self.dist_config, dist_rng)
        mapped_scenario = "nutrient_depletion" if scenario == "ec_drift" else scenario
        disturbances = dist_gen.build_schedule(mapped_scenario, length)

        # 2. Common initial state
        s0_rng = np.random.default_rng(seed)
        temp = self.params.ambient_temp_mean + s0_rng.normal(0, 1.5)
        ec0 = self.ec_target + s0_rng.normal(0, 0.1)
        s0 = TankState.create_initial(
            self.params,
            dissolved_mass=ec0 / self.params.sensor_gain_ec,
            water_temp=temp,
            rng=s0_rng,
        )
        s0_hash = f"B={s0.algae_biomass:.3f}_dm={s0.dissolved_nutrient_mass:.3f}_T={s0.water_temp:.2f}"

        # 3. Exogenous sensor bias process across length + 1 steps
        bias_rng = np.random.default_rng(seed + 100000)
        bias_gen = DisturbanceGenerator(self.dist_config, bias_rng)
        sensor_bias_schedule = np.array(
            [bias_gen.sample_sensor_bias() for _ in range(length + 1)], dtype=np.float64
        )

        # 4. Exogenous sensor noise innovations epsilon_t
        # For noisy_sensors: triple the noise std to implement canonical 3x noise regime.
        # The latent state is unaffected; only the observation noise innovations are amplified.
        noise_rng = np.random.default_rng(seed + 200000)
        ns = self.env_config.noise_std
        noise_mult = 3.0 if scenario == "noisy_sensors" else 1.0
        sensor_noise_schedule = {
            "water_temp": noise_rng.normal(0, ns["water_temp"] * noise_mult, size=length + 1),
            "ec": noise_rng.normal(0, ns["ec"] * noise_mult, size=length + 1),
            "turbidity": noise_rng.normal(0, ns["turbidity"] * noise_mult, size=length + 1),
            "ph": noise_rng.normal(0, ns["ph"] * noise_mult, size=length + 1),
            "dissolved_oxygen": noise_rng.normal(0, ns["dissolved_oxygen"] * noise_mult, size=length + 1),
        }

        return {
            "disturbances": disturbances,
            "initial_state": s0,
            "s0_hash": s0_hash,
            "sensor_bias_schedule": sensor_bias_schedule,
            "sensor_noise_schedule": sensor_noise_schedule,
            "seed": seed,
            "scenario": scenario,
            "horizon": length,
        }

    def run_paired_episode(
        self,
        scenario: str,
        seed: int,
        horizon: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute one paired episode for all 4 controllers with shared CRN inputs:
        Controller state -> true sensor state + same noise innovation -> controller observation.
        """
        h = horizon or self.horizon
        crn = self.generate_crn_schedule(scenario, seed, h)

        episode_results: Dict[str, Dict[str, Any]] = {}

        # 1. No-Action
        res_na = self.closed_loop_eval.run_no_action(
            scenario=scenario,
            custom_length=h,
            custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(crn["disturbances"]),
            prebuilt_initial_state=copy.deepcopy(crn["initial_state"]),
            prebuilt_sensor_innovations=copy.deepcopy(crn["sensor_noise_schedule"]),
            prebuilt_sensor_bias=copy.deepcopy(crn["sensor_bias_schedule"]),
        )
        res_na["metrics"].update({
            "scenario": scenario,
            "seed": seed,
            "controller": "no_action",
            "s0_hash": crn["s0_hash"],
        })
        episode_results["no_action"] = res_na

        # 2. Reserve-Ratio Rule-Based
        res_rb = self.closed_loop_eval.run_rule_based(
            scenario=scenario,
            custom_length=h,
            custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(crn["disturbances"]),
            prebuilt_initial_state=copy.deepcopy(crn["initial_state"]),
            prebuilt_sensor_innovations=copy.deepcopy(crn["sensor_noise_schedule"]),
            prebuilt_sensor_bias=copy.deepcopy(crn["sensor_bias_schedule"]),
        )
        res_rb["metrics"].update({
            "scenario": scenario,
            "seed": seed,
            "controller": "rule_based_reserve",
            "s0_hash": crn["s0_hash"],
        })
        episode_results["rule_based_reserve"] = res_rb

        # 3. PID Controller
        res_pid = self.closed_loop_eval.run_pid(
            scenario=scenario,
            custom_length=h,
            custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(crn["disturbances"]),
            prebuilt_initial_state=copy.deepcopy(crn["initial_state"]),
            prebuilt_sensor_innovations=copy.deepcopy(crn["sensor_noise_schedule"]),
            prebuilt_sensor_bias=copy.deepcopy(crn["sensor_bias_schedule"]),
        )
        res_pid["metrics"].update({
            "scenario": scenario,
            "seed": seed,
            "controller": "pid",
            "s0_hash": crn["s0_hash"],
        })
        episode_results["pid"] = res_pid

        # 4. Learned LSTM Policy
        if self.model is not None and self.normalizer is not None and self.feature_columns is not None:
            res_lstm = self.closed_loop_eval.run_learned_policy(
                model=self.model,
                normalizer=self.normalizer,
                feature_columns=self.feature_columns,
                scenario=scenario,
                device=self.device,
                custom_length=h,
                custom_seed=seed,
                prebuilt_disturbances=copy.deepcopy(crn["disturbances"]),
                prebuilt_initial_state=copy.deepcopy(crn["initial_state"]),
                prebuilt_sensor_innovations=copy.deepcopy(crn["sensor_noise_schedule"]),
                prebuilt_sensor_bias=copy.deepcopy(crn["sensor_bias_schedule"]),
                optimize_feature_engineering=True,
            )
            res_lstm["metrics"].update({
                "scenario": scenario,
                "seed": seed,
                "controller": "lstm",
                "s0_hash": crn["s0_hash"],
            })
            episode_results["lstm"] = res_lstm

        # 5. Reserve-Ratio MPC Controller
        if "mpc_reserve" in self.controllers:
            res_mpc = self.closed_loop_eval.run_mpc(
                scenario=scenario,
                custom_length=h,
                custom_seed=seed,
                prebuilt_disturbances=copy.deepcopy(crn["disturbances"]),
                prebuilt_initial_state=copy.deepcopy(crn["initial_state"]),
                prebuilt_sensor_innovations=copy.deepcopy(crn["sensor_noise_schedule"]),
                prebuilt_sensor_bias=copy.deepcopy(crn["sensor_bias_schedule"]),
            )
            res_mpc["metrics"].update({
                "scenario": scenario,
                "seed": seed,
                "controller": "mpc_reserve",
                "s0_hash": crn["s0_hash"],
            })
            episode_results["mpc_reserve"] = res_mpc

        return episode_results

    def run_full_evaluation(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        """
        Execute the full 120-paired-episode evaluation grid (480 controller executions)
        and compute all statistical analyses, confidence intervals, and tables.
        """
        total_episodes = len(self.scenarios) * self.n_seeds
        print(f"\nStarting Paired Multi-Seed Evaluation:")
        print(f"  Scenarios ({len(self.scenarios)}): {self.scenarios}")
        print(f"  Controllers ({len(self.controllers)}): {self.controllers}")
        print(f"  Seeds per scenario: {self.n_seeds}")
        n_controllers = len(self.controllers)
        total_controller_executions = total_episodes * n_controllers
        print(f"  Total paired episodes: {total_episodes} ({total_controller_executions} controller executions)")
        print(f"  Horizon per episode: {self.horizon} steps")
        print(f"  Bootstrap replicates: {self.bootstrap_replicates} (seed={self.bootstrap_seed})\n")

        all_episode_rows: List[Dict[str, Any]] = []
        representative_trajectories: Dict[str, Dict[str, Any]] = {}
        completed_episodes = 0

        for sc in self.scenarios:
            seeds = self.seed_schedule[sc]
            for rep_idx, seed in enumerate(seeds):
                paired_res = self.run_paired_episode(sc, seed, self.horizon)
                
                # Save first seed of 'normal' as representative illustrative trajectory
                if sc == "normal" and rep_idx == 0:
                    for c_name, c_res in paired_res.items():
                        representative_trajectories[c_name] = {
                            "ec": c_res["trajectory"]["ec"].tolist(),
                            "biomass": c_res["trajectory"]["biomass"].tolist(),
                            "reserve_ratio": c_res["trajectory"]["reserve_ratio"].tolist(),
                            "flowrate": c_res["trajectory"]["flowrate"].tolist(),
                            "duration": c_res["trajectory"]["duration"].tolist(),
                        }

                for c_name, c_res in paired_res.items():
                    m = c_res["metrics"]
                    all_episode_rows.append(m)

                completed_episodes += 1
                if completed_episodes % 5 == 0 or completed_episodes == total_episodes:
                    pct = 100.0 * completed_episodes / total_episodes
                    print(f"  [Progress] {completed_episodes}/{total_episodes} paired episodes completed ({pct:.1f}%) - {sc} seed={seed}")
                if progress_callback:
                    progress_callback(completed_episodes, total_episodes)

        df_episodes = pd.DataFrame(all_episode_rows)

        # Validate scheduled episode count (computed from config)
        n_controllers = len(self.controllers)
        n_scheduled = len(self.scenarios) * self.n_seeds * n_controllers
        n_actual = len(df_episodes)
        expected_scenarios = len(self.scenarios)
        expected_cells = len(self.scenarios) * n_controllers
        expected_episodes = expected_cells * self.n_seeds
        
        if n_actual != n_scheduled:
            print(f"WARNING: Expected {n_scheduled} controller executions, got {n_actual}. "
                  f"Some controllers may not have run (e.g., LSTM checkpoint missing).")
        else:
            print(f"Episode count validated: {n_actual} controller executions as expected.")
            print(f"  Scenarios: {len(self.scenarios)} (expected: {expected_scenarios})")
            print(f"  Condition cells: {expected_cells} (controllers × scenarios)")
            print(f"  Total episodes: {expected_episodes} (cells × seeds)")
        
        # Validate scenario count
        assert len(self.scenarios) >= 1, "At least one scenario must be configured"
        assert "missing_data" not in self.scenarios, "missing_data should be excluded from evaluation"

        # Process paired differences and statistical analysis
        print("\nComputing paired differences and 95% bootstrap confidence intervals...")
        analysis = self._analyze_results(df_episodes, representative_trajectories)

        # Export artifacts
        print(f"Exporting evaluation artifacts to {self.report_dir}...")
        self._export_artifacts(analysis)

        return analysis

    def _analyze_results(
        self,
        df_episodes: pd.DataFrame,
        representative_trajectories: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform statistical analysis on collected episode DataFrame."""
        comparison_pairs = [
            ("lstm", "pid", "LSTM - PID (Primary Reviewer Comparison)"),
            ("lstm", "rule_based_reserve", "LSTM - ReserveRule (ML Discriminability Audit)"),
            ("pid", "rule_based_reserve", "PID - ReserveRule (Classical vs Mechanistic Feedback)"),
        ]
        if "mpc_reserve" in self.controllers:
            comparison_pairs.extend([
                ("mpc_reserve", "rule_based_reserve", "MPC - ReserveRule (Predictive vs Heuristic Reserve Control)"),
                ("mpc_reserve", "pid", "MPC - PID (Predictive vs Classical Feedback)"),
                ("lstm", "mpc_reserve", "LSTM - MPC (Learned vs Model Predictive Control)"),
            ])

        metrics_to_analyze = [
            "ec_mae",
            "ec_rmse",
            "final_biomass",
            "final_reserve_ratio",
            "reserve_min",
            "final_health",
            "dissolved_nutrient_mean",
            "total_dose",
            "intervention_fraction",
            "n_interventions",
        ]

        paired_diff_rows: List[Dict[str, Any]] = []
        bootstrap_ci_rows: List[Dict[str, Any]] = []
        wilcoxon_rows: List[Dict[str, Any]] = []
        equivalence_rows: List[Dict[str, Any]] = []

        # 1. Compute Episode-Level Paired Differences
        for sc in self.scenarios:
            df_sc = df_episodes[df_episodes["scenario"] == sc]
            seeds = sorted(df_sc["seed"].unique())

            for seed in seeds:
                df_seed = df_sc[df_sc["seed"] == seed]
                row_dict: Dict[str, Any] = {"scenario": sc, "seed": seed}

                for ctrl_a, ctrl_b, pair_label in comparison_pairs:
                    row_a = df_seed[df_seed["controller"] == ctrl_a]
                    row_b = df_seed[df_seed["controller"] == ctrl_b]

                    if len(row_a) == 0 or len(row_b) == 0:
                        continue

                    m_a = row_a.iloc[0].to_dict()
                    m_b = row_b.iloc[0].to_dict()

                    pair_key = f"{ctrl_a}_vs_{ctrl_b}"
                    for m_key in metrics_to_analyze:
                        val_a = float(m_a.get(m_key, np.nan))
                        val_b = float(m_b.get(m_key, np.nan))
                        diff = val_a - val_b
                        row_dict[f"{pair_key}_{m_key}_diff"] = diff

                paired_diff_rows.append(row_dict)

        df_paired_diffs = pd.DataFrame(paired_diff_rows)

        # 2. Scenario-Specific and Cross-Scenario Bootstrap CIs & Equivalence Tests
        for ctrl_a, ctrl_b, pair_label in comparison_pairs:
            pair_key = f"{ctrl_a}_vs_{ctrl_b}"

            for m_key in metrics_to_analyze:
                spec = METRIC_SPECS.get(m_key)
                if spec is None:
                    continue

                col_name = f"{pair_key}_{m_key}_diff"
                scenario_diffs_map: Dict[str, np.ndarray] = {}

                for sc in self.scenarios:
                    df_sc_diff = df_paired_diffs[df_paired_diffs["scenario"] == sc]
                    diff_vals = df_sc_diff[col_name].dropna().values
                    scenario_diffs_map[sc] = diff_vals

                    # Bootstrap CI for this scenario
                    ci_res = compute_paired_bootstrap_ci(
                        diff_vals,
                        n_replicates=self.bootstrap_replicates,
                        confidence_level=self.confidence_level,
                        seed=self.bootstrap_seed,
                    )

                    # Win rates
                    win_res = compute_controller_win_rate(
                        diff_vals,
                        higher_is_better=spec.higher_is_better,
                    )

                    # Pre-registered threshold exceedance
                    delta_thresh = self.practical_thresholds.get(m_key, spec.practical_threshold)
                    thresh_res = compute_preregistered_threshold_exceedance(
                        diff_vals,
                        threshold=delta_thresh,
                        higher_is_better=spec.higher_is_better,
                    )

                    # Supplementary Wilcoxon test
                    wilc_res = compute_wilcoxon_paired(diff_vals)
                    wilcoxon_rows.append({
                        "comparison": pair_key,
                        "scenario": sc,
                        "metric": m_key,
                        "statistic": wilc_res.get("statistic", np.nan),
                        "p_value": wilc_res.get("p_value", np.nan),
                        "n_pairs": wilc_res.get("n_pairs", len(diff_vals)),
                    })

                    # Practical Equivalence analysis
                    eq_res = evaluate_practical_equivalence(
                        ci_res["ci95_lower"],
                        ci_res["ci95_upper"],
                        threshold_delta=delta_thresh,
                        higher_is_better=spec.higher_is_better,
                    )

                    equivalence_rows.append({
                        "comparison": pair_key,
                        "scenario": sc,
                        "metric": m_key,
                        "threshold_delta": delta_thresh,
                        "threshold_justification": spec.threshold_justification,
                        "ci95_lower": ci_res["ci95_lower"],
                        "ci95_upper": ci_res["ci95_upper"],
                        "mean_diff": ci_res["mean_difference"],
                        "verdict": eq_res["verdict"],
                        "interpretation": eq_res["interpretation"],
                    })

                    bootstrap_ci_rows.append({
                        "comparison": pair_key,
                        "scenario": sc,
                        "metric": m_key,
                        "display_name": spec.display_name,
                        "unit": spec.unit,
                        "higher_is_better": spec.higher_is_better,
                        "mean_difference": ci_res["mean_difference"],
                        "median_difference": ci_res["median_difference"],
                        "std_difference": ci_res["std_difference"],
                        "ci95_lower": ci_res["ci95_lower"],
                        "ci95_upper": ci_res["ci95_upper"],
                        "win_rate": win_res["win_rate"],
                        "win_count": win_res["win_count"],
                        "tie_count": win_res["tie_count"],
                        "n_episodes": ci_res["n_episodes"],
                        "threshold_delta": delta_thresh,
                        "episodes_exceeding_preregistered_effect_threshold": thresh_res["fraction"],
                        "episodes_exceeding_count": thresh_res["count"],
                    })

                # Cross-Scenario Equal-Weighted Aggregate
                agg_res = aggregate_cross_scenarios_bootstrap(
                    scenario_diffs_map,
                    n_replicates=self.bootstrap_replicates,
                    confidence_level=self.confidence_level,
                    seed=self.bootstrap_seed,
                )
                all_sc_vals = np.concatenate([scenario_diffs_map[s] for s in self.scenarios])
                win_agg = compute_controller_win_rate(all_sc_vals, higher_is_better=spec.higher_is_better)
                delta_thresh = self.practical_thresholds.get(m_key, spec.practical_threshold)
                thresh_agg = compute_preregistered_threshold_exceedance(
                    all_sc_vals, threshold=delta_thresh, higher_is_better=spec.higher_is_better
                )
                eq_agg = evaluate_practical_equivalence(
                    agg_res["ci95_lower"],
                    agg_res["ci95_upper"],
                    threshold_delta=delta_thresh,
                    higher_is_better=spec.higher_is_better,
                )

                bootstrap_ci_rows.append({
                    "comparison": pair_key,
                    "scenario": "equal_weighted_aggregate",
                    "metric": m_key,
                    "display_name": spec.display_name,
                    "unit": spec.unit,
                    "higher_is_better": spec.higher_is_better,
                    "mean_difference": agg_res["mean_difference"],
                    "median_difference": agg_res["median_difference"],
                    "std_difference": agg_res["std_difference"],
                    "ci95_lower": agg_res["ci95_lower"],
                    "ci95_upper": agg_res["ci95_upper"],
                    "win_rate": win_agg["win_rate"],
                    "win_count": win_agg["win_count"],
                    "tie_count": win_agg["tie_count"],
                    "n_episodes": len(all_sc_vals),
                    "threshold_delta": delta_thresh,
                    "episodes_exceeding_preregistered_effect_threshold": thresh_agg["fraction"],
                    "episodes_exceeding_count": thresh_agg["count"],
                })

                equivalence_rows.append({
                    "comparison": pair_key,
                    "scenario": "equal_weighted_aggregate",
                    "metric": m_key,
                    "threshold_delta": delta_thresh,
                    "threshold_justification": spec.threshold_justification,
                    "ci95_lower": agg_res["ci95_lower"],
                    "ci95_upper": agg_res["ci95_upper"],
                    "mean_diff": agg_res["mean_difference"],
                    "verdict": eq_agg["verdict"],
                    "interpretation": eq_agg["interpretation"],
                })

        df_bootstrap_ci = pd.DataFrame(bootstrap_ci_rows)
        df_wilcoxon = pd.DataFrame(wilcoxon_rows)
        df_equivalence = pd.DataFrame(equivalence_rows)

        # 3. Controller Summary across all scenarios
        ctrl_summary_rows: List[Dict[str, Any]] = []
        for ctrl in self.controllers:
            df_c = df_episodes[df_episodes["controller"] == ctrl]
            for sc in self.scenarios:
                df_cs = df_c[df_c["scenario"] == sc]
                if len(df_cs) == 0:
                    continue

                ci_ec = compute_paired_bootstrap_ci(df_cs["ec_mae"].values, n_replicates=self.bootstrap_replicates, seed=self.bootstrap_seed)
                ci_bio = compute_paired_bootstrap_ci(df_cs["final_biomass"].values, n_replicates=self.bootstrap_replicates, seed=self.bootstrap_seed)
                ci_rho = compute_paired_bootstrap_ci(df_cs["final_reserve_ratio"].values, n_replicates=self.bootstrap_replicates, seed=self.bootstrap_seed)
                ci_dose = compute_paired_bootstrap_ci(df_cs["total_dose"].values, n_replicates=self.bootstrap_replicates, seed=self.bootstrap_seed)

                ctrl_summary_rows.append({
                    "controller": ctrl,
                    "scenario": sc,
                    "ec_mae_mean": ci_ec["mean_difference"],
                    "ec_mae_ci95_lower": ci_ec["ci95_lower"],
                    "ec_mae_ci95_upper": ci_ec["ci95_upper"],
                    "final_biomass_mean": ci_bio["mean_difference"],
                    "final_biomass_ci95_lower": ci_bio["ci95_lower"],
                    "final_biomass_ci95_upper": ci_bio["ci95_upper"],
                    "final_reserve_mean": ci_rho["mean_difference"],
                    "final_reserve_ci95_lower": ci_rho["ci95_lower"],
                    "final_reserve_ci95_upper": ci_rho["ci95_upper"],
                    "total_dose_mean": ci_dose["mean_difference"],
                    "total_dose_ci95_lower": ci_dose["ci95_lower"],
                    "total_dose_ci95_upper": ci_dose["ci95_upper"],
                    "effective_horizon_mean": float(df_cs["effective_horizon"].mean()),
                    "terminated_count": int(df_cs["episode_terminated"].sum()),
                    "n_episodes": len(df_cs),
                })

        df_ctrl_summary = pd.DataFrame(ctrl_summary_rows)

        # 4. Primary Comparison Tables: LSTM vs PID and LSTM vs Reserve-Rule
        df_lstm_vs_pid = df_bootstrap_ci[df_bootstrap_ci["comparison"] == "lstm_vs_pid"].copy()
        df_lstm_vs_reserve = df_bootstrap_ci[df_bootstrap_ci["comparison"] == "lstm_vs_rule_based_reserve"].copy()
        df_mpc_vs_reserve = df_bootstrap_ci[df_bootstrap_ci["comparison"] == "mpc_reserve_vs_rule_based_reserve"].copy()
        df_mpc_vs_pid = df_bootstrap_ci[df_bootstrap_ci["comparison"] == "mpc_reserve_vs_pid"].copy()

        return {
            "df_episodes": df_episodes,
            "df_paired_diffs": df_paired_diffs,
            "df_bootstrap_ci": df_bootstrap_ci,
            "df_wilcoxon": df_wilcoxon,
            "df_equivalence": df_equivalence,
            "df_ctrl_summary": df_ctrl_summary,
            "df_lstm_vs_pid": df_lstm_vs_pid,
            "df_lstm_vs_reserve": df_lstm_vs_reserve,
            "df_mpc_vs_reserve": df_mpc_vs_reserve,
            "df_mpc_vs_pid": df_mpc_vs_pid,
            "representative_trajectories": representative_trajectories,
            "seed_schedule": self.seed_schedule,
            "config": {
                "scenarios": self.scenarios,
                "n_seeds": self.n_seeds,
                "base_seed": self.base_seed,
                "horizon": self.horizon,
                "bootstrap_replicates": self.bootstrap_replicates,
                "bootstrap_seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
                "practical_thresholds": self.practical_thresholds,
            },
        }

    def _export_artifacts(self, analysis: Dict[str, Any]) -> None:
        """Save all CSV, JSON, YAML and Markdown artifacts to report directory."""
        # 1. Episode results
        df_ep = analysis["df_episodes"]
        df_ep.to_csv(self.report_dir / "episode_results.csv", index=False)
        df_ep.to_json(self.report_dir / "episode_results.json", orient="records", indent=2)

        # 2. Paired differences
        analysis["df_paired_diffs"].to_csv(self.report_dir / "paired_differences.csv", index=False)

        # 3. Bootstrap CIs
        analysis["df_bootstrap_ci"].to_csv(self.report_dir / "bootstrap_confidence_intervals.csv", index=False)

        # 4. Equivalence analysis
        analysis["df_equivalence"].to_csv(self.report_dir / "equivalence_analysis.csv", index=False)

        # 5. Wilcoxon supplementary tests
        analysis["df_wilcoxon"].to_csv(self.report_dir / "wilcoxon_results.csv", index=False)

        # 6. Controller summary
        analysis["df_ctrl_summary"].to_csv(self.report_dir / "controller_summary.csv", index=False)

        # 7. Comparison subsets
        analysis["df_lstm_vs_pid"].to_csv(self.report_dir / "lstm_vs_pid.csv", index=False)
        analysis["df_lstm_vs_reserve"].to_csv(self.report_dir / "lstm_vs_reserve_rule.csv", index=False)
        if "df_mpc_vs_reserve" in analysis and len(analysis["df_mpc_vs_reserve"]) > 0:
            analysis["df_mpc_vs_reserve"].to_csv(self.report_dir / "mpc_vs_reserve_rule.csv", index=False)
        if "df_mpc_vs_pid" in analysis and len(analysis["df_mpc_vs_pid"]) > 0:
            analysis["df_mpc_vs_pid"].to_csv(self.report_dir / "mpc_vs_pid.csv", index=False)

        # 8. Seed schedule and config
        with open(self.report_dir / "seed_schedule.json", "w") as f:
            json.dump(analysis["seed_schedule"], f, indent=2)

        with open(self.report_dir / "evaluation_config.yaml", "w") as f:
            yaml.dump(analysis["config"], f, default_flow_style=False)
        
        # 8.5. Evaluation manifest (for plot stage to know current run scenarios)
        manifest = {
            "evaluation_stage": "multiseed_long_horizon",
            "scenarios": self.scenarios,
            "controllers": self.controllers,
            "n_seeds_per_condition": self.n_seeds,
            "horizon": self.horizon,
            "total_episodes": len(analysis["df_episodes"]),
            "condition_cells": len(self.scenarios) * len(self.controllers),
            "base_seed": self.base_seed,
            "excluded_scenarios": ["missing_data"],
        }
        with open(self.report_dir / "evaluation_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        with open(self.report_dir / "representative_trajectories.json", "w") as f:
            json.dump(analysis["representative_trajectories"], f, indent=2)

        # 9. Coverage matrix: scenarios x controllers (episodes completed)
        df_ep = analysis["df_episodes"]
        controllers = list(self.controllers)  # use the evaluator's authoritative list
        coverage_rows = []
        for sc in self.scenarios:
            row = {"scenario": sc}
            for ctrl in controllers:
                n = len(df_ep[(df_ep["scenario"] == sc) & (df_ep["controller"] == ctrl)])
                row[ctrl] = f"{n}/{self.n_seeds}"
            row["total"] = f"{sum(len(df_ep[(df_ep['scenario']==sc)&(df_ep['controller']==c)]) for c in controllers)}/{self.n_seeds * len(controllers)}"
            coverage_rows.append(row)
        pd.DataFrame(coverage_rows).to_csv(self.report_dir / "coverage_matrix.csv", index=False)

        # 10. Failure/collapse probability table by controller and scenario
        fail_rows = []
        for ctrl in controllers:
            for sc in self.scenarios:
                df_cs = df_ep[(df_ep["controller"] == ctrl) & (df_ep["scenario"] == sc)]
                n_total = len(df_cs)
                if n_total == 0:
                    continue
                n_fail = int(df_cs["failure"].sum()) if "failure" in df_cs.columns else 0
                n_collapse = int(df_cs["collapsed"].sum()) if "collapsed" in df_cs.columns else 0
                n_survival = int(df_cs["survival"].sum()) if "survival" in df_cs.columns else n_total
                fail_rows.append({
                    "controller": ctrl,
                    "scenario": sc,
                    "n_episodes": n_total,
                    "failure_count": n_fail,
                    "failure_rate": round(n_fail / n_total, 4) if n_total > 0 else float('nan'),
                    "collapse_count": n_collapse,
                    "collapse_rate": round(n_collapse / n_total, 4) if n_total > 0 else float('nan'),
                    "survival_count": n_survival,
                    "survival_rate": round(n_survival / n_total, 4) if n_total > 0 else float('nan'),
                })
        df_fail = pd.DataFrame(fail_rows)
        df_fail.to_csv(self.report_dir / "failure_probability_table.csv", index=False)

        # Aggregate failure probability by controller (across all scenarios)
        agg_rows = []
        for ctrl in controllers:
            df_c = df_ep[df_ep["controller"] == ctrl]
            n_total = len(df_c)
            agg_rows.append({
                "controller": ctrl,
                "n_episodes": n_total,
                "failure_rate": round(df_c["failure"].mean(), 4) if ("failure" in df_c.columns and n_total > 0) else float('nan'),
                "collapse_rate": round(df_c["collapsed"].mean(), 4) if ("collapsed" in df_c.columns and n_total > 0) else float('nan'),
                "survival_rate": round(df_c["survival"].mean(), 4) if ("survival" in df_c.columns and n_total > 0) else float('nan'),
            })
        pd.DataFrame(agg_rows).to_csv(self.report_dir / "failure_probability_aggregate.csv", index=False)

        # 11. Scenario orthogonality table from registry
        try:
            from simulation.scenario_registry import get_scenario_orthogonality_matrix
            get_scenario_orthogonality_matrix().to_csv(
                self.report_dir / "scenario_orthogonality_table.csv", index=False
            )
        except Exception as e:
            print(f"Warning: could not export orthogonality table: {e}")

        # 12. Publication-ready Markdown Report
        md_content = self._generate_statistical_summary_markdown(analysis)
        with open(self.report_dir / "statistical_summary.md", "w") as f:
            f.write(md_content)

    def _generate_statistical_summary_markdown(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive, factual, publication-ready statistical summary markdown."""
        df_ci = analysis["df_bootstrap_ci"]
        df_eq = analysis["df_equivalence"]
        df_ctrl = analysis["df_ctrl_summary"]
        df_ep = analysis["df_episodes"]
        cfg = analysis["config"]

        n_total_episodes = len(df_ep)
        n_paired_episodes = len(analysis["df_paired_diffs"])
        terminated_count = int(df_ep["episode_terminated"].sum())

        lines: List[str] = [
            "# Paired Multi-Seed Long-Horizon Statistical Evaluation & ML-Discriminability Audit",
            "",
            "## 1. Evaluation Design",
            "",
            f"- **Controllers Evaluated**: `no_action`, `rule_based_reserve`, `pid`, `lstm`, `mpc_reserve`",
            f"- **Operating Scenarios** ({len(cfg['scenarios'])}): {', '.join(f'`{s}`' for s in cfg['scenarios'])}",
            f"- **Sample Size**: {cfg['n_seeds']} independent seeds per scenario $\\rightarrow$ **{n_paired_episodes} paired episodes** (480 total controller rollouts)",
            f"- **Horizon**: {cfg['horizon']} simulation steps per episode (60 s timestep $\\rightarrow$ 58.33 operating hours)",
            f"- **Statistical Unit**: Paired episode difference ($\\Delta_i = M_{{A,i}} - M_{{B,i}}$, $N={cfg['n_seeds']}$ paired observations per scenario)",
            f"- **Bootstrap Confidence Level**: {int(cfg['confidence_level']*100)}% ({cfg['bootstrap_replicates']:,} replicates, deterministic seed {cfg['bootstrap_seed']})",
            "",
            "## 2. Common Random Numbers (CRN) & Sensor Noise Decoupling",
            "",
            "To ensure rigorous statistical pairing without artificially decoupling observations from controller dynamics:",
            "1. **Exogenous Disturbances**: A complete 3500-step stochastic schedule (thermal events, sediment shocks, uptake pulses, mixing noise) is pre-generated and shared across all 4 controllers.",
            "2. **Initial State ($s_0$)**: A single initial state is constructed per seed and `deepcopy` assigned independently to each controller.",
            "3. **Sensor Noise Innovations ($\\epsilon_t$)**: The sensor observation model obeys:",
            "   $$y_t^{(c)} = h(x_t^{(c)}) + \\epsilon_t$$",
            "   where identical Gaussian innovations $\\epsilon_t$ are shared across controllers, but evaluated at each controller's genuine internal tank state $x_t^{(c)}$.",
            "",
            "## 3. Reviewer Primary Comparison: LSTM vs PID",
            "",
            "| Scenario | Metric | Paired Mean $\\Delta$ | 95% Bootstrap CI | LSTM Win Rate | Exceeding Preregistered $\\delta$ |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |",
        ]

        # Primary comparison table: LSTM vs PID for key metrics
        primary_metrics = ["ec_mae", "final_biomass", "final_reserve_ratio", "total_dose"]
        df_lstm_pid = df_ci[df_ci["comparison"] == "lstm_vs_pid"]

        for sc in self.scenarios + ["equal_weighted_aggregate"]:
            sc_label = sc.replace("_", " ").title()
            for m in primary_metrics:
                row = df_lstm_pid[(df_lstm_pid["scenario"] == sc) & (df_lstm_pid["metric"] == m)]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                lines.append(
                    f"| {sc_label} | {r['display_name']} ({r['unit']}) | {r['mean_difference']:+.4f} | "
                    f"[{r['ci95_lower']:+.4f}, {r['ci95_upper']:+.4f}] | {r['win_rate']*100:.1f}% | "
                    f"{r['episodes_exceeding_preregistered_effect_threshold']*100:.1f}% |"
                )

        lines.extend([
            "",
            "## 4. Benchmark ML-Discriminability Audit: LSTM vs Reserve-Ratio Rule",
            "",
            "Evaluating whether the learned policy provides meaningful advantage over simple mechanistically informed feedback ($-\\delta < \\Delta < +\\delta$ practical equivalence):",
            "",
            "| Scenario | Metric | Paired Mean $\\Delta$ | 95% Bootstrap CI | Win Rate | Equivalence Margin | Practical Verdict |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        ])

        df_lstm_rb = df_ci[df_ci["comparison"] == "lstm_vs_rule_based_reserve"]
        for sc in self.scenarios + ["equal_weighted_aggregate"]:
            sc_label = sc.replace("_", " ").title()
            for m in primary_metrics:
                row = df_lstm_rb[(df_lstm_rb["scenario"] == sc) & (df_lstm_rb["metric"] == m)]
                eq_row = df_eq[(df_eq["comparison"] == "lstm_vs_rule_based_reserve") & (df_eq["scenario"] == sc) & (df_eq["metric"] == m)]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                eq_v = eq_row.iloc[0]["verdict"] if len(eq_row) > 0 else "N/A"
                delta_val = r["threshold_delta"]
                lines.append(
                    f"| {sc_label} | {r['display_name']} | {r['mean_difference']:+.4f} | "
                    f"[{r['ci95_lower']:+.4f}, {r['ci95_upper']:+.4f}] | {r['win_rate']*100:.1f}% | "
                    f"[$\\pm${delta_val:.2f}] | **{eq_v}** |"
                )

        lines.extend([
            "",
            "## 5. Control Effort Adjustment & Trade-Off Analysis",
            "",
            "To prevent misinterpreting higher regulation precision that is achieved merely by excessive chemical dosing, performance differences are directly coupled with dosing effort:",
            "",
            "| Scenario | $\\Delta$ Biomass (LSTM - Rule) | $\\Delta$ EC MAE (LSTM - Rule) | $\\Delta$ Total Dose (LSTM - Rule) | $\\Delta$ Total Dose (LSTM - PID) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])

        for sc in self.scenarios + ["equal_weighted_aggregate"]:
            sc_label = sc.replace("_", " ").title()
            bio_r = df_lstm_rb[(df_lstm_rb["scenario"] == sc) & (df_lstm_rb["metric"] == "final_biomass")].iloc[0]
            ec_r = df_lstm_rb[(df_lstm_rb["scenario"] == sc) & (df_lstm_rb["metric"] == "ec_mae")].iloc[0]
            dose_rb = df_lstm_rb[(df_lstm_rb["scenario"] == sc) & (df_lstm_rb["metric"] == "total_dose")].iloc[0]
            dose_pid = df_lstm_pid[(df_lstm_pid["scenario"] == sc) & (df_lstm_pid["metric"] == "total_dose")].iloc[0]
            lines.append(
                f"| {sc_label} | {bio_r['mean_difference']:+.2f} g/L [{bio_r['ci95_lower']:+.2f}, {bio_r['ci95_upper']:+.2f}] | "
                f"{ec_r['mean_difference']:+.4f} [{ec_r['ci95_lower']:+.4f}, {ec_r['ci95_upper']:+.4f}] | "
                f"{dose_rb['mean_difference']:+.2f} mL [{dose_rb['ci95_lower']:+.2f}, {dose_rb['ci95_upper']:+.2f}] | "
                f"{dose_pid['mean_difference']:+.2f} mL [{dose_pid['ci95_lower']:+.2f}, {dose_pid['ci95_upper']:+.2f}] |"
            )

        lines.extend([
            "",
            "## 6. Aggregate Controller Summary (Mean ± 95% Bootstrap CI Across Seeds)",
            "",
            "| Controller | Scenario | EC MAE (mS/cm) | Final Biomass (g/L) | Final Reserve (rho) | Total Dose (mL) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |",
        ])

        for _, r in df_ctrl.iterrows():
            c_name = r["controller"].replace("_", " ").title()
            sc_name = r["scenario"].replace("_", " ").title()
            lines.append(
                f"| {c_name} | {sc_name} | {r['ec_mae_mean']:.4f} [{r['ec_mae_ci95_lower']:.4f}, {r['ec_mae_ci95_upper']:.4f}] | "
                f"{r['final_biomass_mean']:.2f} [{r['final_biomass_ci95_lower']:.2f}, {r['final_biomass_ci95_upper']:.2f}] | "
                f"{r['final_reserve_mean']:.4f} [{r['final_reserve_ci95_lower']:.4f}, {r['final_reserve_ci95_upper']:.4f}] | "
                f"{r['total_dose_mean']:.2f} [{r['total_dose_ci95_lower']:.2f}, {r['total_dose_ci95_upper']:.2f}] |"
            )

        lines.extend([
            "",
            "## 7. Episode Termination & Horizon Integrity",
            "",
            f"- **Total Controller Executions**: {len(df_ep)}",
            f"- **Completed Full Horizon (3500 steps)**: {len(df_ep) - terminated_count}",
            f"- **Early Terminations**: {terminated_count}",
        ])
        if terminated_count > 0:
            df_term = df_ep[df_ep["episode_terminated"] == True]
            term_breakdown = df_term.groupby(["controller", "scenario", "termination_reason"]).size().reset_index(name="count")
            lines.append("")
            lines.append("| Controller | Scenario | Termination Reason | Episodes Terminated |")
            lines.append("| :--- | :--- | :--- | :---: |")
            for _, tr in term_breakdown.iterrows():
                lines.append(f"| {tr['controller']} | {tr['scenario']} | {tr['termination_reason']} | {tr['count']} |")
        else:
            lines.append("- All controllers maintained healthy biological survival across all 3,500 simulation steps in all scenarios.")

        lines.extend([
            "",
            "## 8. Benchmark Discriminability Findings",
            "",
            "1. **Statistical Separation from PID**: Across repeated stochastic trajectories, paired differences between LSTM and PID confirm whether the learned policy exhibits statistically distinct regulation and biomass yields with confidence intervals excluding zero.",
            "2. **Residual Control Complexity Beyond Reserve Heuristic**: Practical equivalence testing examines whether the simple reserve-ratio controller ($-\\delta < \\Delta < +\\delta$) captures the actionable control structure under baseline conditions, and highlights which stress regimes (e.g., heatwave vs nutrient depletion) reveal significant divergences.",
            "3. **Control Effort Discrepancy**: The observed performance difference is evaluated in context with nutrient dosage volumes to assess whether learned regulation relies on different dosing efficiency.",
            "",
            "## 9. Reproducibility & Provenance Information",
            "",
            f"- **Seed Schedule**: Saved in `seed_schedule.json` ({cfg['n_seeds']} seeds per scenario)",
            f"- **Configuration Snapshot**: Saved in `evaluation_config.yaml`",
            f"- **Raw Episode Data**: Saved in `episode_results.csv` and `episode_results.json`",
            f"- **Full Paired Differences**: Saved in `paired_differences.csv`",
            f"- **Bootstrap CIs**: Saved in `bootstrap_confidence_intervals.csv`",
            f"- **Equivalence Analysis**: Saved in `equivalence_analysis.csv`",
            f"- **Supplementary Wilcoxon Tests**: Saved in `wilcoxon_results.csv`",
        ])

        return "\n".join(lines)
