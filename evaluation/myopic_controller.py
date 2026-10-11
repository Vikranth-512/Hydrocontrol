"""
Myopic 1-Step Greedy Controller — Temporal Control Complexity Diagnostic.

Implements a deterministic 1-step greedy controller that:
1. Enumerates the same valid (flowrate, duration) action set used by the
   rollout-selection labeler (candidate_flowrates x candidate_durations,
   valid-pair filter identical to OptimizationLabeler.candidates).
2. Simulates each candidate for exactly ONE simulation step (step_dynamics).
3. Evaluates the same multi-term ecosystem objective as evaluate_rollout
   but truncated to a 1-step trace (H=1).
4. Selects the lowest-score action and applies it.

This controller is a diagnostic instrument ONLY:
- It does NOT modify or replace any existing controller.
- It does NOT write to any existing evaluation directory.
- All results are saved exclusively under reports/myopic_complexity_test/.

Purpose: Determine whether meaningful performance depends on anticipating
delayed consequences (long-horizon dynamics).  Raw metrics are reported and
compared against the existing 4-controller benchmark results.  No automated
complexity label is assigned.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from evaluation.failure_analysis import wilson_score_interval
from evaluation.statistical_analysis import compute_paired_bootstrap_ci
from simulation.disturbances import DisturbanceConfig, DisturbanceGenerator
from simulation.dynamics import (
    TankDynamicsParams,
    TankState,
    q10_metabolism_factor,
    step_dynamics,
)
from simulation.environment import AlgaeTankEnvironment, EnvironmentConfig
from simulation.labeling_objective import (
    LabelingObjectiveConfig,
    evaluate_rollout,
)
from simulation.state_utils import compute_rho
from training.evaluation import compute_ecosystem_metrics


# ---------------------------------------------------------------------------
# 1-Step Greedy Controller
# ---------------------------------------------------------------------------

class MyopicController:
    """
    Deterministic 1-step greedy controller.

    At each timestep:
      1. Enumerate the feasible discrete action set (same candidates used
         by OptimizationLabeler).
      2. For each candidate (flowrate, duration): call step_dynamics once to
         obtain the post-action state s'.
      3. Build a length-1 eco_trace from s' and evaluate the labeling objective
         (H=1 rollout — identical cost function, zero lookahead).
      4. Select the candidate with minimum objective score.
      5. The environment enforces min_time_between_doses via _clip_action, so
         the controller does NOT need to duplicate that constraint.

    Note: the labeling objective is evaluated on the 1-step trace, which
    makes the terminal sustainability term V(s_T) active after exactly 1
    step.  This is intentional — it ensures the objective signal is
    identical in structure to what the long-horizon labeler uses, differing
    only in lookahead depth.
    """

    def __init__(
        self,
        obj_cfg: LabelingObjectiveConfig,
        params: TankDynamicsParams,
        dt: float = 60.0,
        candidate_flowrates: Optional[List[float]] = None,
        candidate_durations: Optional[List[float]] = None,
        flowrate_max: float = 5.0,
        duration_max: float = 30.0,
    ) -> None:
        self.obj_cfg = obj_cfg
        self.params = params
        self.dt = dt
        self.flowrate_max = flowrate_max
        self.duration_max = duration_max

        fr_list = candidate_flowrates or [
            0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
        ]
        dur_list = candidate_durations or [
            0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0
        ]

        # Valid actions: (0,0) no-op, or (positive flowrate, positive duration)
        # Identical filter to OptimizationLabeler.candidates
        self.candidates: List[Tuple[float, float]] = [
            (fr, dur)
            for fr in fr_list
            for dur in dur_list
            if (fr == 0.0 and dur == 0.0) or (fr > 0.0 and dur > 0.0)
            if fr <= flowrate_max and dur <= duration_max
        ]

    def _score_action(
        self,
        state: TankState,
        flowrate: float,
        duration: float,
    ) -> float:
        """
        Simulate one step and evaluate the labeling objective on the 1-step trace.
        Returns the total cost (lower is better).
        """
        s1 = step_dynamics(state, flowrate, duration, self.dt, self.params)

        if not (
            np.isfinite(s1.ec)
            and np.isfinite(s1.algae_biomass)
            and np.isfinite(s1.internal_reserve)
        ):
            return 1e9

        ic = self.obj_cfg.internal_capacity
        B = max(s1.algae_biomass, 1e-6)
        rho = float(np.clip(s1.internal_reserve / (B * ic), 0.0, 1.0))
        dose = flowrate * duration / 60.0
        thermal_resp = q10_metabolism_factor(s1.water_temp, self.params)
        maint_cost = self.params.maintenance_cost * s1.algae_biomass * thermal_resp
        osmotic_stress = (
            s1.dissolved_nutrient_mass / self.params.osmotic_half_effect
        ) ** 2

        eco_trace = [
            {
                "ec": s1.ec,
                "rho": rho,
                "health": s1.health_index,
                "damage": s1.damage_index,
                "biomass": s1.algae_biomass,
                "dose": dose,
                "dissolved": s1.dissolved_nutrient_mass,
                "maint_cost": maint_cost,
                "osmotic": osmotic_stress,
                "detritus": s1.dead_biomass_pool,
            }
        ]

        bd = evaluate_rollout(eco_trace, s1, state, flowrate, duration, self.obj_cfg)
        return bd.total if np.isfinite(bd.total) else 1e9

    def select_action(self, state: TankState) -> Tuple[float, float]:
        """
        Return (flowrate, duration) with minimum 1-step objective score.
        Falls back to (0.0, 0.0) if all candidates produce non-finite scores.
        """
        best_score = float("inf")
        best_action = (0.0, 0.0)

        for fr, dur in self.candidates:
            sc = self._score_action(state, fr, dur)
            if sc < best_score:
                best_score = sc
                best_action = (fr, dur)

        return best_action


# ---------------------------------------------------------------------------
# Episode runner (mirrors ClosedLoopEvaluator.run_episode interface)
# ---------------------------------------------------------------------------

def run_myopic_episode(
    myopic_ctrl: MyopicController,
    env_config: EnvironmentConfig,
    params: TankDynamicsParams,
    dist_config: DisturbanceConfig,
    scenario: str,
    horizon: int,
    seed: int,
    prebuilt_disturbances: Optional[list] = None,
    prebuilt_initial_state: Optional[TankState] = None,
    prebuilt_sensor_innovations: Optional[Dict[str, np.ndarray]] = None,
    prebuilt_sensor_bias: Optional[np.ndarray] = None,
    ec_target: float = 1.2,
    dt: float = 60.0,
) -> Dict[str, Any]:
    """
    Run one closed-loop episode with the MyopicController.

    Shares the same CRN inputs (disturbances, initial state, sensor noise)
    as the other controllers for strict experimental pairing.
    """
    episode_rng = np.random.default_rng(seed)
    dist_gen = DisturbanceGenerator(dist_config, episode_rng)

    env = AlgaeTankEnvironment(
        env_config,
        params,
        rng=episode_rng,
        disturbance_generator=dist_gen,
        sensor_noise_schedule=prebuilt_sensor_innovations,
        sensor_bias_schedule=prebuilt_sensor_bias,
    )

    s0 = copy.deepcopy(prebuilt_initial_state) if prebuilt_initial_state is not None else None
    obs = env.reset(
        initial_state=s0,
        disturbance_schedule=prebuilt_disturbances,
        sensor_noise_schedule=prebuilt_sensor_innovations,
        sensor_bias_schedule=prebuilt_sensor_bias,
    )
    keys = env.observation_keys

    history: Dict[str, list] = {
        k: []
        for k in keys
        + ["ec_true", "turbidity_true", "flowrate", "duration",
           "pending_nutrients", "health_index", "biomass", "reserve_ratio"]
    }
    state_trace: List[TankState] = []

    episode_terminated = False
    termination_step = horizon
    termination_reason = "completed"

    for t in range(horizon):
        state_trace.append(env.state)
        obs_dict = {keys[i]: float(obs[i]) for i in range(len(keys))}

        # Use the actual physical state for greedy action selection
        fr, dur = myopic_ctrl.select_action(env.state)

        curr_rho = compute_rho(env.state, params.internal_capacity)
        curr_biomass = float(env.state.algae_biomass)

        for k, v in obs_dict.items():
            history[k].append(v)
        history["ec_true"].append(env.state.ec)
        history["turbidity_true"].append(env.state.turbidity)
        history["pending_nutrients"].append(env.state.pending_nutrients)
        history["health_index"].append(env.state.health_index)
        history["biomass"].append(curr_biomass)
        history["reserve_ratio"].append(curr_rho)
        history["flowrate"].append(fr)
        history["duration"].append(dur)

        obs, _info = env.step((fr, dur))

        # Early termination
        if env.state and (
            env.state.health_index <= 0.0 or env.state.algae_biomass <= 0.0
        ):
            episode_terminated = True
            termination_step = t + 1
            termination_reason = (
                "biomass_extinction"
                if env.state.algae_biomass <= 0.0
                else "health_collapse"
            )
            break

    return {
        "ec": np.array(history["ec_true"]),
        "flowrate": np.array(history["flowrate"]),
        "duration": np.array(history["duration"]),
        "state_trace": state_trace,
        "history": {k: np.array(v) for k, v in history.items()},
        "health": np.array(history["health_index"]),
        "biomass": np.array(history["biomass"]),
        "reserve_ratio": np.array(history["reserve_ratio"]),
        "damage": np.array([s.damage_index for s in state_trace]),
        "episode_terminated": episode_terminated,
        "termination_step": termination_step,
        "termination_reason": termination_reason,
        "effective_horizon": len(state_trace),
        "nominal_horizon": horizon,
    }


# ---------------------------------------------------------------------------
# Multi-Seed Myopic Evaluator
# ---------------------------------------------------------------------------

class MyopicEvaluator:
    """
    Orchestrates the myopic complexity diagnostic:
    - Same 4 scenarios, 30 seeds, 3500-step horizon as the primary benchmark
    - Strict CRN pairing: identical initial states, disturbances, and sensor
      noise innovations to those used in the primary evaluation
    - Results saved ONLY to reports/myopic_complexity_test/ (isolated)
    - Does NOT modify any primary benchmark artifacts

    The evaluator re-generates CRN schedules using the *same* seed arithmetic
    as MultiSeedEvaluator.generate_crn_schedule so results are exactly paired
    with the primary benchmark episode_results.csv.
    """

    CONTROLLER_NAME = "myopic_1step"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        sim = config.get("simulation", {})
        dyn = config.get("dynamics", {})
        ev_cfg = config.get("evaluation", {}).get("multiseed_long_horizon", {})
        lab = config.get("labeling", {})

        # Match primary benchmark settings exactly
        self.scenarios: List[str] = ev_cfg.get(
            "scenarios", ["normal", "heatwave", "nutrient_depletion", "cold_shock"]
        )
        self.n_seeds: int = int(ev_cfg.get("n_seeds", 30))
        self.base_seed: int = int(ev_cfg.get("base_seed", 42))
        self.horizon: int = int(ev_cfg.get("horizon", 3500))
        self.bootstrap_replicates: int = int(ev_cfg.get("bootstrap_replicates", 5000))
        self.bootstrap_seed: int = int(ev_cfg.get("bootstrap_seed", 12345))

        self.ec_target: float = float(sim.get("ec_target", 1.2))
        self.dt: float = float(sim.get("dt_seconds", 60.0))

        self.env_config = EnvironmentConfig(
            dt_seconds=self.dt,
            ec_target=self.ec_target,
            ec_safe_min=float(sim.get("ec_safe_min", 0.4)),
            ec_safe_max=float(sim.get("ec_safe_max", 2.5)),
            flowrate_min=float(sim.get("flowrate_min", 0.0)),
            flowrate_max=float(sim.get("flowrate_max", 5.0)),
            duration_min=float(sim.get("duration_min", 0.0)),
            duration_max=float(sim.get("duration_max", 30.0)),
            min_time_between_doses=float(sim.get("min_time_between_doses", 120.0)),
            noise_std=sim.get("noise_std"),
        )
        self.params = TankDynamicsParams.from_config(dyn)
        self.dist_config = DisturbanceConfig.from_config(
            config.get("disturbances", {})
        )

        # Build the labeling objective — same config used by OptimizationLabeler
        obj_cfg = LabelingObjectiveConfig.from_yaml(
            lab, sim, dyn,
            eco=config.get("ecosystem_control", {}),
        )
        self.myopic_ctrl = MyopicController(
            obj_cfg=obj_cfg,
            params=self.params,
            dt=self.dt,
            candidate_flowrates=lab.get(
                "candidate_flowrates",
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            ),
            candidate_durations=lab.get(
                "candidate_durations",
                [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            ),
            flowrate_max=float(sim.get("flowrate_max", 5.0)),
            duration_max=float(sim.get("duration_max", 30.0)),
        )

        # Seed schedule: identical arithmetic to MultiSeedEvaluator._build_seed_schedule
        self.seed_schedule: Dict[str, List[int]] = {
            sc: [int(self.base_seed + 1000 * s_idx + i) for i in range(self.n_seeds)]
            for s_idx, sc in enumerate(self.scenarios)
        }

        # Isolated output directory
        self.report_dir = Path(
            config.get("myopic_complexity_test", {}).get(
                "report_dir", "reports/myopic_complexity_test"
            )
        )
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CRN schedule — identical to MultiSeedEvaluator.generate_crn_schedule
    # ------------------------------------------------------------------

    def _generate_crn_schedule(
        self,
        scenario: str,
        seed: int,
        length: int,
    ) -> Dict[str, Any]:
        """
        Reproduce the exact same CRN inputs as the primary benchmark evaluator.

        Seed arithmetic (matches MultiSeedEvaluator.generate_crn_schedule):
          disturbances:   seed
          initial state:  seed           (same rng, same call sequence)
          sensor bias:    seed + 100000
          sensor noise:   seed + 200000
        """
        mapped_scenario = (
            "nutrient_depletion" if scenario == "ec_drift" else scenario
        )

        # 1. Disturbances
        dist_rng = np.random.default_rng(seed)
        dist_gen = DisturbanceGenerator(self.dist_config, dist_rng)
        disturbances = dist_gen.build_schedule(mapped_scenario, length)

        # 2. Initial state (same rng, same call order as primary)
        s0_rng = np.random.default_rng(seed)
        temp = self.params.ambient_temp_mean + s0_rng.normal(0, 1.5)
        ec0 = self.ec_target + s0_rng.normal(0, 0.1)
        s0 = TankState.create_initial(
            self.params,
            dissolved_mass=ec0 / self.params.sensor_gain_ec,
            water_temp=temp,
            rng=s0_rng,
        )

        # 3. Sensor bias
        bias_rng = np.random.default_rng(seed + 100000)
        bias_gen = DisturbanceGenerator(self.dist_config, bias_rng)
        sensor_bias_schedule = np.array(
            [bias_gen.sample_sensor_bias() for _ in range(length + 1)],
            dtype=np.float64,
        )

        # 4. Sensor noise innovations
        noise_rng = np.random.default_rng(seed + 200000)
        ns = self.env_config.noise_std
        sensor_noise_schedule = {
            "water_temp": noise_rng.normal(0, ns["water_temp"], size=length + 1),
            "ec": noise_rng.normal(0, ns["ec"], size=length + 1),
            "turbidity": noise_rng.normal(0, ns["turbidity"], size=length + 1),
            "ph": noise_rng.normal(0, ns["ph"], size=length + 1),
            "dissolved_oxygen": noise_rng.normal(
                0, ns["dissolved_oxygen"], size=length + 1
            ),
        }

        return {
            "disturbances": disturbances,
            "initial_state": s0,
            "sensor_bias_schedule": sensor_bias_schedule,
            "sensor_noise_schedule": sensor_noise_schedule,
            "seed": seed,
            "scenario": scenario,
        }

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------

    def run_episode(self, scenario: str, seed: int) -> Dict[str, Any]:
        """Run one myopic episode and return metrics dict."""
        crn = self._generate_crn_schedule(scenario, seed, self.horizon)

        traj = run_myopic_episode(
            myopic_ctrl=self.myopic_ctrl,
            env_config=self.env_config,
            params=self.params,
            dist_config=self.dist_config,
            scenario=scenario,
            horizon=self.horizon,
            seed=seed,
            prebuilt_disturbances=copy.deepcopy(crn["disturbances"]),
            prebuilt_initial_state=copy.deepcopy(crn["initial_state"]),
            prebuilt_sensor_innovations=copy.deepcopy(crn["sensor_noise_schedule"]),
            prebuilt_sensor_bias=copy.deepcopy(crn["sensor_bias_schedule"]),
            ec_target=self.ec_target,
            dt=self.dt,
        )

        metrics = compute_ecosystem_metrics(
            traj["state_trace"],
            traj["flowrate"],
            traj["duration"],
            self.params,
            self.dt,
            ec_target=self.ec_target,
            episode_terminated=traj["episode_terminated"],
            termination_step=traj["termination_step"],
            termination_reason=traj["termination_reason"],
            nominal_horizon=traj["nominal_horizon"],
        )
        metrics.update({
            "scenario": scenario,
            "seed": seed,
            "controller": self.CONTROLLER_NAME,
        })
        return metrics

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Execute the full myopic complexity diagnostic:
        4 scenarios x 30 seeds = 120 episodes.

        Returns analysis dict and saves all artifacts to self.report_dir.
        """
        total = len(self.scenarios) * self.n_seeds
        print(f"\nMyopic Complexity Diagnostic")
        print(f"  Controller:  1-step greedy (myopic)")
        print(f"  Scenarios:   {self.scenarios}")
        print(f"  Seeds:       {self.n_seeds} per scenario")
        print(f"  Episodes:    {total}")
        print(f"  Horizon:     {self.horizon} steps")
        print(f"  Output dir:  {self.report_dir}\n")

        rows: List[Dict[str, Any]] = []
        done = 0
        for sc in self.scenarios:
            seeds = self.seed_schedule[sc]
            for seed in seeds:
                m = self.run_episode(sc, seed)
                rows.append(m)
                done += 1
                if done % 10 == 0 or done == total:
                    pct = 100.0 * done / total
                    print(
                        f"  [Progress] {done}/{total} ({pct:.1f}%)"
                        f" -- {sc} seed={seed}"
                    )

        df = pd.DataFrame(rows)
        analysis = self._analyze(df)
        self._export(df, analysis)
        return analysis

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute per-scenario summary statistics and collapse probability
        analysis for the myopic controller.

        Note: collapse stats are computed directly here (not via
        analyze_failures_and_collapses) because that helper has a hardcoded
        controller allowlist that excludes the myopic controller name.
        """
        primary_metrics = [
            "ec_mae",
            "final_biomass",
            "total_dose",
            "final_reserve_ratio",
        ]

        summary_rows: List[Dict[str, Any]] = []
        collapse_rows: List[Dict[str, Any]] = []
        COLLAPSE_THRESH = 50.0
        SECONDARY_THRESH = 20.0

        for sc in self.scenarios:
            df_sc = df[df["scenario"] == sc]
            if len(df_sc) == 0:
                continue
            n = len(df_sc)

            # Summary statistics
            row: Dict[str, Any] = {"controller": self.CONTROLLER_NAME, "scenario": sc}
            for m in primary_metrics:
                vals = df_sc[m].values
                ci = compute_paired_bootstrap_ci(
                    vals,
                    n_replicates=self.bootstrap_replicates,
                    seed=self.bootstrap_seed,
                )
                row[f"{m}_mean"] = ci["mean_difference"]
                row[f"{m}_ci95_lower"] = ci["ci95_lower"]
                row[f"{m}_ci95_upper"] = ci["ci95_upper"]
            row["effective_horizon_mean"] = float(df_sc["effective_horizon"].mean())
            row["terminated_count"] = int(df_sc["episode_terminated"].sum())
            row["n_episodes"] = n
            summary_rows.append(row)

            # Collapse probability (computed directly to bypass controller allowlist)
            n_collapsed = int((df_sc["final_biomass"] < COLLAPSE_THRESH).sum())
            n_collapsed_20 = int((df_sc["final_biomass"] < SECONDARY_THRESH).sum())
            n_failed = int(df_sc["episode_terminated"].sum())
            col_lo, col_hi = wilson_score_interval(n_collapsed, n)
            col20_lo, col20_hi = wilson_score_interval(n_collapsed_20, n)
            fail_lo, fail_hi = wilson_score_interval(n_failed, n)
            biomass_vals = df_sc["final_biomass"].values
            n_early = int((df_sc["effective_horizon"] < df_sc["nominal_horizon"]).sum())

            collapse_rows.append({
                "scenario": sc,
                "controller": self.CONTROLLER_NAME,
                "n": n,
                "failures": n_failed,
                "failure_probability": n_failed / n,
                "failure_ci_lower": fail_lo,
                "failure_ci_upper": fail_hi,
                "collapse_count": n_collapsed,
                "collapse_probability": n_collapsed / n,
                "collapse_ci_lower": col_lo,
                "collapse_ci_upper": col_hi,
                "collapse_20_count": n_collapsed_20,
                "collapse_20_probability": n_collapsed_20 / n,
                "collapse_20_ci_lower": col20_lo,
                "collapse_20_ci_upper": col20_hi,
                "terminal_biomass_mean": float(np.mean(biomass_vals)),
                "terminal_biomass_median": float(np.median(biomass_vals)),
                "terminal_biomass_min": float(np.min(biomass_vals)),
                "terminal_biomass_max": float(np.max(biomass_vals)),
                "early_terminated_count": n_early,
            })

        df_summary = pd.DataFrame(summary_rows)
        df_collapse = pd.DataFrame(collapse_rows)

        # Aggregate across all scenarios
        collapse_agg_rows: List[Dict[str, Any]] = []
        n_all = len(df)
        if n_all > 0:
            n_col_all = int((df["final_biomass"] < COLLAPSE_THRESH).sum())
            n_fail_all = int(df["episode_terminated"].sum())
            col_lo_a, col_hi_a = wilson_score_interval(n_col_all, n_all)
            fail_lo_a, fail_hi_a = wilson_score_interval(n_fail_all, n_all)
            biomass_all = df["final_biomass"].values
            collapse_agg_rows.append({
                "controller": self.CONTROLLER_NAME,
                "n_total": n_all,
                "total_failures": n_fail_all,
                "failure_probability": n_fail_all / n_all,
                "failure_ci_lower": fail_lo_a,
                "failure_ci_upper": fail_hi_a,
                "total_collapses": n_col_all,
                "collapse_probability": n_col_all / n_all,
                "collapse_ci_lower": col_lo_a,
                "collapse_ci_upper": col_hi_a,
                "terminal_biomass_mean": float(np.mean(biomass_all)),
                "terminal_biomass_median": float(np.median(biomass_all)),
                "terminal_biomass_min": float(np.min(biomass_all)),
                "terminal_biomass_max": float(np.max(biomass_all)),
                "early_terminated_count": int(
                    (df["effective_horizon"] < df["nominal_horizon"]).sum()
                ),
            })
        df_collapse_agg = pd.DataFrame(collapse_agg_rows)

        return {
            "df_episodes": df,
            "df_summary": df_summary,
            "df_collapse": df_collapse,
            "df_collapse_agg": df_collapse_agg,
            "seed_schedule": self.seed_schedule,
            "config": {
                "controller": self.CONTROLLER_NAME,
                "scenarios": self.scenarios,
                "n_seeds": self.n_seeds,
                "base_seed": self.base_seed,
                "horizon": self.horizon,
                "bootstrap_replicates": self.bootstrap_replicates,
                "bootstrap_seed": self.bootstrap_seed,
                "n_candidates": len(self.myopic_ctrl.candidates),
            },
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> None:
        """Save all diagnostic artifacts to self.report_dir (isolated)."""
        rd = self.report_dir

        # 1. Episode results
        df.to_csv(rd / "episode_results.csv", index=False)
        df.to_json(rd / "episode_results.json", orient="records", indent=2)

        # 2. Summary statistics
        analysis["df_summary"].to_csv(rd / "controller_summary.csv", index=False)

        # 3. Collapse probability tables
        analysis["df_collapse"].to_csv(rd / "collapse_analysis.csv", index=False)
        analysis["df_collapse_agg"].to_csv(
            rd / "collapse_analysis_aggregate.csv", index=False
        )

        # 4. Seed schedule
        with open(rd / "seed_schedule.json", "w") as f:
            json.dump(analysis["seed_schedule"], f, indent=2)

        # 5. Config snapshot
        with open(rd / "diagnostic_config.yaml", "w") as f:
            yaml.dump(analysis["config"], f, default_flow_style=False)

        # 6. Comparison report
        primary_path = Path("reports/multiseed_long_horizon/episode_results.csv")
        if primary_path.exists():
            md = self._generate_comparison_report(df, analysis, primary_path)
        else:
            md = self._generate_standalone_report(df, analysis)

        with open(rd / "myopic_complexity_report.md", "w") as f:
            f.write(md)

        print(f"\nMyopic complexity diagnostic complete.")
        print(f"  Artifacts saved to: {rd}")
        print(f"  Episodes:           {len(df)}")
        print(f"  Terminations:       {int(df['episode_terminated'].sum())} / {len(df)}")

    # ------------------------------------------------------------------
    # Report generators
    # ------------------------------------------------------------------

    def _generate_standalone_report(
        self, df: pd.DataFrame, analysis: Dict[str, Any]
    ) -> str:
        """Standalone report when primary benchmark results are unavailable."""
        lines = [
            "# Myopic 1-Step Greedy Controller -- Temporal Complexity Diagnostic",
            "",
            "> **Note**: Primary benchmark results "
            "(`reports/multiseed_long_horizon/episode_results.csv`) were not found.  "
            "Re-run after the primary benchmark to generate a comparison.",
            "",
            "## Evaluation Configuration",
            "",
            f"- **Controller**: `{self.CONTROLLER_NAME}` (1-step greedy, deterministic)",
            f"- **Scenarios**: {', '.join(f'`{s}`' for s in self.scenarios)}",
            f"- **Seeds per scenario**: {self.n_seeds}",
            f"- **Horizon**: {self.horizon} steps",
            f"- **Action space**: {len(self.myopic_ctrl.candidates)} discrete candidates",
            "",
            "## Myopic Controller Results",
            "",
            "| Scenario | EC MAE | Final Biomass (g/L) | Total Dose (mL) "
            "| Final Reserve (rho) | Collapses |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        df_sum = analysis["df_summary"]
        df_col = analysis["df_collapse"]

        for sc in self.scenarios:
            r = df_sum[df_sum["scenario"] == sc]
            if len(r) == 0:
                continue
            r = r.iloc[0]

            col_r = df_col[
                (df_col["scenario"] == sc)
                & (df_col["controller"] == self.CONTROLLER_NAME)
            ]
            n_col = int(col_r["collapse_count"].iloc[0]) if len(col_r) > 0 else "N/A"
            p_col = (
                f"{float(col_r['collapse_probability'].iloc[0]):.3f}"
                if len(col_r) > 0
                else "N/A"
            )

            lines.append(
                f"| {sc} "
                f"| {r['ec_mae_mean']:.4f} [{r['ec_mae_ci95_lower']:.4f},"
                f" {r['ec_mae_ci95_upper']:.4f}] "
                f"| {r['final_biomass_mean']:.2f} [{r['final_biomass_ci95_lower']:.2f},"
                f" {r['final_biomass_ci95_upper']:.2f}] "
                f"| {r['total_dose_mean']:.2f} [{r['total_dose_ci95_lower']:.2f},"
                f" {r['total_dose_ci95_upper']:.2f}] "
                f"| {r['final_reserve_ratio_mean']:.4f}"
                f" [{r['final_reserve_ratio_ci95_lower']:.4f},"
                f" {r['final_reserve_ratio_ci95_upper']:.4f}] "
                f"| {n_col}/{self.n_seeds} (p={p_col}) |"
            )

        return "\n".join(lines)

    def _generate_comparison_report(
        self,
        df_myopic: pd.DataFrame,
        analysis: Dict[str, Any],
        primary_path: Path,
    ) -> str:
        """
        Comparison report: myopic controller vs primary benchmark controllers.
        All primary benchmark data is read from the existing CSV (read-only).
        """
        df_primary = pd.read_csv(primary_path)
        primary_controllers = ["rule_based_reserve", "pid", "lstm"]
        primary_metrics = [
            ("ec_mae", "EC MAE (mS/cm)", False),
            ("final_biomass", "Final Biomass (g/L)", True),
            ("total_dose", "Total Dose (mL)", False),
            ("final_reserve_ratio", "Final Reserve (rho)", True),
        ]

        lines = [
            "# Myopic 1-Step Greedy Controller -- Temporal Complexity Diagnostic",
            "",
            "## Purpose",
            "",
            "This diagnostic evaluates whether meaningful performance in the algae "
            "tank benchmark depends on anticipating delayed consequences "
            "(long-horizon temporal complexity).",
            "",
            "- **If the 1-step controller performs close to existing long-horizon "
            "controllers**: provides evidence that the control problem may be "
            "relatively myopic (low temporal complexity).",
            "- **If the 1-step controller performs substantially worse** (especially "
            "in final biomass, collapse probability, or control effort): provides "
            "evidence that delayed dynamics materially affect control difficulty.",
            "",
            "> Scientific interpretation is left to the reviewer.  "
            "No automated complexity label is assigned.",
            "",
            "## Evaluation Configuration",
            "",
            f"- **Myopic Controller**: `{self.CONTROLLER_NAME}` -- 1-step greedy, "
            "deterministic; identical objective and action set to the supervisory labeler",
            f"- **Scenarios**: {', '.join(f'`{s}`' for s in self.scenarios)}",
            f"- **Seeds per scenario**: {self.n_seeds} "
            "(same seed schedule as primary benchmark)",
            f"- **Horizon**: {self.horizon} steps (60 s/step -> 58.33 h)",
            f"- **CRN pairing**: Identical initial states, disturbances, and sensor "
            "noise innovations (seed arithmetic reproduced from MultiSeedEvaluator)",
            f"- **Action space**: {len(self.myopic_ctrl.candidates)} discrete candidates "
            "(flowrate x duration, identical to OptimizationLabeler)",
            "",
        ]

        df_sum = analysis["df_summary"]

        for m_key, m_label, _higher_is_better in primary_metrics:
            lines += [
                f"## {m_label}",
                "",
                "Mean [95% Bootstrap CI] across 30 seeds per scenario.",
                "",
                f"| Scenario | Myopic (1-step) | Reserve Rule | PID | LSTM |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]

            for sc in self.scenarios:
                rm = df_sum[df_sum["scenario"] == sc]
                if len(rm) == 0:
                    continue
                rm = rm.iloc[0]
                myopic_cell = (
                    f"{rm[f'{m_key}_mean']:.4f} "
                    f"[{rm[f'{m_key}_ci95_lower']:.4f},"
                    f" {rm[f'{m_key}_ci95_upper']:.4f}]"
                )

                cells = [f"**{sc}**", myopic_cell]
                for ctrl in primary_controllers:
                    df_c = df_primary[
                        (df_primary["controller"] == ctrl)
                        & (df_primary["scenario"] == sc)
                    ]
                    if len(df_c) == 0:
                        cells.append("N/A")
                        continue
                    ci = compute_paired_bootstrap_ci(
                        df_c[m_key].values,
                        n_replicates=self.bootstrap_replicates,
                        seed=self.bootstrap_seed,
                    )
                    cells.append(
                        f"{ci['mean_difference']:.4f} "
                        f"[{ci['ci95_lower']:.4f}, {ci['ci95_upper']:.4f}]"
                    )

                lines.append("| " + " | ".join(cells) + " |")

            lines.append("")

        # Collapse probability
        lines += [
            "## Collapse Probability (Final Biomass < 50 g/L)",
            "",
            "Wilson 95% CI.  Primary benchmark data from "
            "`reports/multiseed_long_horizon/episode_results.csv` (read-only).",
            "",
            "| Scenario | Myopic (1-step) | Reserve Rule | PID | LSTM |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]

        df_col = analysis["df_collapse"]
        for sc in self.scenarios:
            mc = df_col[
                (df_col["scenario"] == sc)
                & (df_col["controller"] == self.CONTROLLER_NAME)
            ]
            if len(mc) > 0:
                mc = mc.iloc[0]
                myopic_cell = (
                    f"{mc['collapse_probability']:.3f} "
                    f"[{mc['collapse_ci_lower']:.3f}, {mc['collapse_ci_upper']:.3f}]"
                    f" ({int(mc['collapse_count'])}/{self.n_seeds})"
                )
            else:
                myopic_cell = "N/A"

            cells = [f"**{sc}**", myopic_cell]
            for ctrl in primary_controllers:
                df_c = df_primary[
                    (df_primary["controller"] == ctrl)
                    & (df_primary["scenario"] == sc)
                ]
                if len(df_c) == 0:
                    cells.append("N/A")
                    continue
                n = len(df_c)
                n_col = int((df_c["final_biomass"] < 50.0).sum())
                ci_lo, ci_hi = wilson_score_interval(n_col, n)
                cells.append(
                    f"{n_col/n:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
                    f" ({n_col}/{n})"
                )
            lines.append("| " + " | ".join(cells) + " |")

        # Termination summary
        lines += [
            "",
            "## Termination Summary",
            "",
            "| Scenario | Episodes | Early Terminations | Termination Rate |",
            "| :--- | :---: | :---: | :---: |",
        ]
        for sc in self.scenarios:
            df_sc = df_myopic[df_myopic["scenario"] == sc]
            n_term = int(df_sc["episode_terminated"].sum())
            lines.append(
                f"| {sc} | {len(df_sc)} | {n_term}"
                f" | {n_term/max(len(df_sc),1):.3f} |"
            )

        lines += [
            "",
            "## Scientific Interpretation Guidance",
            "",
            "Compare each metric column (myopic vs reserve rule, PID, LSTM) "
            "across scenarios:",
            "",
            "- **Similar EC MAE across all controllers**: Regulation may not require "
            "long-horizon anticipation.",
            "- **Substantially lower final biomass or higher collapse probability for "
            "myopic**: Suggests delayed uptake / reserve kinetics require anticipation "
            "beyond 1 step.",
            "- **Similar or higher total dose for myopic**: Greedy dosing may be less "
            "efficient -- overdosing without awareness of osmotic stress accumulation.",
            "- **Scenario-specific divergence** (e.g., cold_shock only): Temporal "
            "complexity is conditionally present under physiological stress.",
            "",
            "---",
            f"*Report generated by `evaluation/myopic_controller.py`.*",
            f"*Primary benchmark data read-only from "
            f"`reports/multiseed_long_horizon/`.*",
            f"*This diagnostic does NOT modify any existing pipeline or result.*",
        ]

        return "\n".join(lines)
