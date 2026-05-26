"""
Optimization-based pseudo-optimal control label generation.

For each timestep, search candidate (flowrate, duration) pairs via short-horizon
rollout and weighted scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from simulation.dynamics import TankDynamicsParams, TankState, rollout_horizon
from simulation.environment import EnvironmentConfig


class OptimizationLabeler:
    """
    Generate labels: optimal_flowrate, optimal_duration per timestep.

    score = w1*ec_error + w2*nutrient_cost + w3*instability + w4*overshoot
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        lab = config.get("labeling", {})
        sim = config.get("simulation", {})
        dyn = config.get("dynamics", {})

        self.horizon = lab.get("horizon_steps", 15)
        self.candidate_flowrates = lab.get(
            "candidate_flowrates",
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        )
        self.candidate_durations = lab.get(
            "candidate_durations", [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        )
        w = lab.get("weights", {})
        self.w_ec = w.get("ec_error", 1.0)
        self.w_cost = w.get("nutrient_cost", 0.3)
        self.w_instab = w.get("instability", 0.5)
        self.w_overshoot = w.get("overshoot", 0.8)

        self.dt = sim.get("dt_seconds", 60.0)
        self.ec_target = sim.get("ec_target", 1.2)
        self.ec_safe_min = sim.get("ec_safe_min", 0.4)
        self.ec_safe_max = sim.get("ec_safe_max", 2.5)
        self.flowrate_max = sim.get("flowrate_max", 5.0)
        self.duration_max = sim.get("duration_max", 30.0)
        self.min_time_between_doses = sim.get("min_time_between_doses", 120.0)

        self.base_params = TankDynamicsParams.from_config(
            dyn, ec_target=self.ec_target
        )

    def _row_to_state(self, row: pd.Series) -> TankState:
        """Reconstruct TankState from trajectory row (prefer true sensors if present)."""
        def _get(name: str, default: float) -> float:
            col = f"true_{name}" if f"true_{name}" in row.index else name
            return float(row[col]) if col in row.index and pd.notna(row[col]) else default

        n_delay = self.base_params.delay_steps
        queue = np.zeros(n_delay, dtype=np.float64)
        pending = _get("pending_absorption", 0.0)
        if pending > 0 and n_delay > 0:
            queue[0] = pending

        return TankState(
            water_temp=_get("water_temp", 22.0),
            ec=_get("ec", self.ec_target),
            turbidity=_get("turbidity", 50.0),
            prev_flowrate=float(row.get("prev_flowrate", row.get("flowrate", 0.0))),
            prev_duration=float(row.get("prev_duration", row.get("duration", 0.0))),
            time_since_last_dose=float(row.get("time_since_last_dose", 999.0)),
            ph=_get("ph", 7.2),
            dissolved_oxygen=_get("dissolved_oxygen", 8.0),
            ambient_temp=_get("ambient_temp", 22.0),
            cumulative_nutrients=_get("cumulative_nutrients", 0.0),
            step_index=int(row.get("timestep", 0)),
            absorption_queue=queue,
            algae_biomass=_get("algae_biomass", _get("turbidity", 80.0)),
            nutrient_memory=_get("nutrient_memory", 0.0),
            biomass_memory=_get("biomass_memory", 0.5),
            biomass_growth_drive=_get("biomass_growth_drive", 0.5),
            health_index=_get("health_index", 1.0),
            ec_velocity=_get("ec_velocity", 0.0),
            assimilation_pool=_get("assimilation_pool", 0.0),
        )

    def _score_action(
        self,
        state: TankState,
        flowrate: float,
        duration: float,
        params: TankDynamicsParams,
    ) -> float:
        """Lower is better."""
        if state.time_since_last_dose < self.min_time_between_doses and (
            flowrate > 0 or duration > 0
        ):
            return 1e6

        if state.ec < self.ec_safe_min or state.ec > self.ec_safe_max:
            if flowrate == 0 and duration == 0:
                return 5e5

        ec_trace, final_s = rollout_horizon(
            state, flowrate, duration, self.horizon, self.dt, params
        )

        ec_errors = (ec_trace - self.ec_target) ** 2
        ec_error = float(np.mean(ec_errors))

        nutrient_cost = flowrate * duration / 60.0

        # Instability: variance of EC over horizon + penalty for oscillation
        instability = float(np.var(ec_trace)) + float(np.mean(np.abs(np.diff(ec_trace))))

        # Overshoot above target
        overshoot = float(np.mean(np.maximum(0.0, ec_trace - self.ec_target) ** 2))

        # Penalize unsafe or starving final EC
        if final_s.ec > self.ec_safe_max or final_s.ec < self.ec_safe_min:
            ec_error += 10.0
        if final_s.ec < params.ec_healthy_min:
            ec_error += 5.0 * (params.ec_healthy_min - final_s.ec)

        score = (
            self.w_ec * ec_error
            + self.w_cost * nutrient_cost
            + self.w_instab * instability
            + self.w_overshoot * overshoot
        )
        return score

    def label_action(
        self, state: TankState, params: Optional[TankDynamicsParams] = None
    ) -> Tuple[float, float, float]:
        """Return (optimal_flowrate, optimal_duration, best_score)."""
        params = params or self.base_params
        best_score = float("inf")
        best_fr, best_dur = 0.0, 0.0

        for fr in self.candidate_flowrates:
            for dur in self.candidate_durations:
                if fr > self.flowrate_max or dur > self.duration_max:
                    continue
                s = self._score_action(state, fr, dur, params)
                if s < best_score:
                    best_score = s
                    best_fr, best_dur = fr, dur

        return best_fr, best_dur, best_score

    def label_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add optimal_flowrate, optimal_duration columns."""
        params = self.base_params
        opt_fr, opt_dur, opt_score = [], [], []

        for _, row in df.iterrows():
            state = self._row_to_state(row)
            fr, dur, sc = self.label_action(state, params)
            opt_fr.append(fr)
            opt_dur.append(dur)
            opt_score.append(sc)

        out = df.copy()
        out["optimal_flowrate"] = opt_fr
        out["optimal_duration"] = opt_dur
        out["label_score"] = opt_score
        return out

    def label_dataset(self, input_dir: Path, output_dir: Path) -> Path:
        """Label all trajectory CSVs in input_dir."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob("trajectory_*.csv"))
        labeled_dfs = []

        for f in files:
            df = pd.read_csv(f)
            labeled = self.label_trajectory(df)
            out_path = output_dir / f.name.replace(".csv", "_labeled.csv")
            labeled.to_csv(out_path, index=False)
            labeled_dfs.append(labeled)

        combined = pd.concat(labeled_dfs, ignore_index=True)
        combined_path = output_dir / "all_trajectories_labeled.csv"
        combined.to_csv(combined_path, index=False)
        return combined_path
