"""
Closed-loop control evaluation in the algae tank simulator.

Compares learned LSTM policy vs PID vs rule-based under disturbances.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from controllers.pid_controller import PIDConfig, PIDController, PIDGains
from controllers.rule_based_controller import RuleBasedController, RuleBasedConfig
from models.lstm_policy import LSTMPolicy
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.normalization import FeatureNormalizer
from simulation.disturbances import DisturbanceConfig, DisturbanceGenerator
from simulation.dynamics import TankDynamicsParams, TankState
from simulation.environment import AlgaeTankEnvironment, EnvironmentConfig
from training.evaluation import compute_ecosystem_metrics


class ClosedLoopEvaluator:
    """Run controllers in simulated environment and collect trajectories."""

    def __init__(self, config: Dict[str, Any], seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        sim = config.get("simulation", {})
        dyn = config.get("dynamics", {})
        ev = config.get("evaluation", {})

        self.env_config = EnvironmentConfig(
            dt_seconds=sim.get("dt_seconds", 60.0),
            ec_target=sim.get("ec_target", 1.2),
            ec_safe_min=sim.get("ec_safe_min", 0.4),
            ec_safe_max=sim.get("ec_safe_max", 2.5),
            flowrate_min=sim.get("flowrate_min", 0.0),
            flowrate_max=sim.get("flowrate_max", 5.0),
            duration_min=sim.get("duration_min", 0.0),
            duration_max=sim.get("duration_max", 30.0),
            min_time_between_doses=sim.get("min_time_between_doses", 120.0),
            noise_std=sim.get("noise_std"),
        )
        self.params = TankDynamicsParams.from_config(dyn)
        self.disturbance_config = DisturbanceConfig.from_config(
            config.get("disturbances", {})
        )
        self.steps = ev.get("closed_loop_steps", 500)
        self.ec_target = sim.get("ec_target", 1.2)
        self.dt = sim.get("dt_seconds", 60.0)
        self.feature_engineer = FeatureEngineer(
            ec_target=self.ec_target,
            rolling_window=config.get("preprocessing", {}).get("rolling_window", 8),
            internal_capacity=dyn.get("internal_capacity", 0.5),
        )
        self.sequence_length = config.get("preprocessing", {}).get("sequence_length", 32)

    def _build_disturbance_schedule(self, scenario: str, length: int) -> list:
        mapped = scenario
        if scenario == "ec_drift":
            mapped = "nutrient_depletion"
        gen = DisturbanceGenerator(self.disturbance_config, self.rng)
        return gen.build_schedule(mapped, length)

    def run_episode(
        self,
        controller_fn: Callable[[Dict[str, float], int], Tuple[float, float]],
        scenario: str = "normal",
        noise_multiplier: float = 1.0,
        custom_length: Optional[int] = None,
        custom_seed: Optional[int] = None,
        custom_water_temp: Optional[float] = None,
        custom_initial_ec: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Single closed-loop rollout."""
        length = custom_length if custom_length is not None else self.steps
        disturbances = self._build_disturbance_schedule(scenario, length)

        noise_std = {
            k: v * noise_multiplier for k, v in self.env_config.noise_std.items()
        }
        env_cfg = EnvironmentConfig(
            dt_seconds=self.env_config.dt_seconds,
            ec_target=self.env_config.ec_target,
            ec_safe_min=self.env_config.ec_safe_min,
            ec_safe_max=self.env_config.ec_safe_max,
            flowrate_min=self.env_config.flowrate_min,
            flowrate_max=self.env_config.flowrate_max,
            duration_min=self.env_config.duration_min,
            duration_max=self.env_config.duration_max,
            min_time_between_doses=self.env_config.min_time_between_doses,
            noise_std=noise_std,
        )

        params = self.params
        if scenario == "delayed_response":
            params = TankDynamicsParams(
                **{
                    **params.__dict__,
                    "immediate_absorption_fraction": 0.08,
                }
            )

        # Use custom seed if provided
        episode_rng = np.random.default_rng(custom_seed) if custom_seed is not None else self.rng
        dist_gen = DisturbanceGenerator(self.disturbance_config, episode_rng)
        env = AlgaeTankEnvironment(
            env_cfg, params, rng=episode_rng, disturbance_generator=dist_gen
        )
        
        # Handle custom initial conditions
        if custom_water_temp is not None or custom_initial_ec is not None:
            temp = custom_water_temp if custom_water_temp is not None else params.ambient_temp_mean
            ec0 = custom_initial_ec if custom_initial_ec is not None else self.ec_target
            s0 = TankState.create_initial(params, dissolved_mass=ec0 / params.sensor_gain_ec, water_temp=temp, rng=episode_rng)
            obs = env.reset(initial_state=s0, disturbance_schedule=disturbances)
        else:
            obs = env.reset(disturbance_schedule=disturbances)
        keys = env.observation_keys

        history = {
            k: []
            for k in keys
            + ["ec_true", "turbidity_true", "flowrate", "duration", "pending_nutrients", "health_index", "biomass", "reserve_ratio"]
        }
        rolling_rows = []
        state_trace: List[TankState] = []

        for t in range(length):
            state_trace.append(env.state)
            obs_dict = {keys[i]: float(obs[i]) for i in range(len(keys))}
            if scenario == "missing_data" and t % 17 == 0:
                obs_dict["ec"] = float("nan")

            # Engineer features from history
            row = {**obs_dict, "timestep": t, "trajectory_id": 0}
            rolling_rows.append(row)
            df_roll = pd.DataFrame(rolling_rows)
            df_eng = self.feature_engineer._transform_single(df_roll)
            eng = df_eng.iloc[-1].to_dict()

            if scenario == "missing_data" and t % 17 == 0:
                ec_for_ctrl = self.ec_target
            else:
                ec_for_ctrl = obs_dict.get("ec", self.ec_target)

            fr, dur = controller_fn({**obs_dict, **eng, "ec": ec_for_ctrl}, t)

            if scenario == "actuator_saturation":
                fr = min(fr, 1.0)
                dur = min(dur, 5.0)

            for k, v in obs_dict.items():
                history[k].append(v)
            history["ec_true"].append(env.state.ec if env.state else ec_for_ctrl)
            history["turbidity_true"].append(env.state.turbidity if env.state else 0.0)
            history["pending_nutrients"].append(
                env.state.pending_nutrients if env.state else 0.0
            )
            history["health_index"].append(env.state.health_index if env.state else 0.0)
            history["biomass"].append(env.state.algae_biomass if env.state else 0.0)
            if env.state:
                reserve_ratio = env.state.internal_reserve / (env.state.algae_biomass * self.params.internal_capacity + 1e-6)
            else:
                reserve_ratio = 0.0
            history["reserve_ratio"].append(reserve_ratio)
            history["flowrate"].append(fr)
            history["duration"].append(dur)

            obs, info = env.step((fr, dur))

        ec_trace = np.array(history["ec_true"])
        return {
            "ec": ec_trace,
            "flowrate": np.array(history["flowrate"]),
            "duration": np.array(history["duration"]),
            "state_trace": state_trace,
            "history": {k: np.array(v) for k, v in history.items()},
            "health": np.array(history["health_index"]),
            "biomass": np.array(history["biomass"]),
            "reserve_ratio": np.array(history["reserve_ratio"]),
            "damage": np.array([s.damage_index for s in state_trace]),
        }

    def run_pid(self, scenario: str = "normal") -> Dict[str, Any]:
        ev = self.config.get("evaluation", {})
        behavior = self.config.get("pid_tuning", {}).get("pid_behavior", {})
        
        # Try to load tuned PID gains from tuning results
        import yaml
        from pathlib import Path
        tuned_pid_path = Path("data/processed/pid_tuned_gains.yaml")
        
        if tuned_pid_path.exists():
            with open(tuned_pid_path, "r") as f:
                tuned_config = yaml.safe_load(f)
                pid_gains = tuned_config.get("evaluation", {}).get("pid", {})
                kp = pid_gains.get("kp", ev.get("pid", {}).get("kp", 2.0))
                ki = pid_gains.get("ki", ev.get("pid", {}).get("ki", 0.05))
                kd = pid_gains.get("kd", ev.get("pid", {}).get("kd", 0.3))
                print(f"Using tuned PID gains: kp={kp}, ki={ki}, kd={kd}")
        else:
            # Fallback to config values
            pid_cfg = ev.get("pid", {})
            kp = pid_cfg.get("kp", 2.0)
            ki = pid_cfg.get("ki", 0.05)
            kd = pid_cfg.get("kd", 0.3)
            print(f"Using config PID gains: kp={kp}, ki={ki}, kd={kd}")
        
        # Use reserve-based control like the tuner (not EC-based)
        reserve_target = self.config.get("ecosystem_control", {}).get("rho_setpoint", 0.50)
        
        pid = PIDController(
            setpoint=reserve_target,  # Use reserve target like tuner
            gains=PIDGains(kp=kp, ki=ki, kd=kd),
            config=PIDConfig(**behavior) if behavior else None,
            flowrate_max=self.env_config.flowrate_max,
            duration_max=self.env_config.duration_max,
            flowrate_min=self.env_config.flowrate_min,
            duration_min=self.env_config.duration_min,
            dt=self.dt,
        )

        def ctrl(obs_dict, t):
            # Use reserve-based control like the tuner
            # Reconstruct TankState-like object from observation dict
            biomass = obs_dict.get("biomass", 1.0)
            reserve_ratio = obs_dict.get("reserve_ratio", 0.5)
            internal_capacity = self.params.internal_capacity
            
            # Create a simple object with the required attributes for compute_from_reserve
            class SimpleState:
                def __init__(self, biomass, reserve_ratio, internal_capacity):
                    self.algae_biomass = biomass
                    self.internal_reserve = reserve_ratio * biomass * internal_capacity
            
            state = SimpleState(biomass, reserve_ratio, internal_capacity)
            return pid.compute_from_reserve(state, internal_capacity)

        traj = self.run_episode(ctrl, scenario=scenario)
        metrics = compute_ecosystem_metrics(
            traj["state_trace"], traj["flowrate"], traj["duration"], self.params, self.dt
        )
        return {"trajectory": traj, "metrics": metrics, "controller": "pid"}

    def run_rule_based(self, scenario: str = "normal") -> Dict[str, Any]:
        ev = self.config.get("evaluation", {})
        rb_cfg = ev.get("rule_based", {})
        rb = RuleBasedController(
            RuleBasedConfig(
                ec_low_threshold=rb_cfg.get("ec_low_threshold", 0.9),
                ec_high_threshold=rb_cfg.get("ec_high_threshold", 1.4),
                ec_target=self.ec_target,
            )
        )

        def ctrl(obs_dict, t):
            return rb.compute(obs_dict["ec"])

        traj = self.run_episode(ctrl, scenario=scenario)
        metrics = compute_ecosystem_metrics(
            traj["state_trace"], traj["flowrate"], traj["duration"], self.params, self.dt
        )
        return {"trajectory": traj, "metrics": metrics, "controller": "rule_based"}

    def run_learned_policy(
        self,
        model: LSTMPolicy,
        normalizer: FeatureNormalizer,
        feature_columns: List[str],
        scenario: str = "normal",
        device: str = "cpu",
        custom_length: Optional[int] = None,
        custom_seed: Optional[int] = None,
        custom_water_temp: Optional[float] = None,
        custom_initial_ec: Optional[float] = None,
    ) -> Dict[str, Any]:
        model = model.to(device)
        model.eval()
        buffer: deque = deque(maxlen=self.sequence_length)

        def ctrl(obs_dict, t):
            vec = np.array([obs_dict[c] for c in feature_columns], dtype=np.float32)
            buffer.append(vec)
            if len(buffer) < self.sequence_length:
                return 0.0, 0.0
            seq = np.stack(list(buffer))
            seq_norm = normalizer._feature_scaler.transform(seq)
            x = torch.from_numpy(seq_norm).float().unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x).cpu().numpy()
            action = normalizer.inverse_transform_targets(out)[0]
            return float(np.clip(action[0], 0, self.env_config.flowrate_max)), float(
                np.clip(action[1], 0, self.env_config.duration_max)
            )

        traj = self.run_episode(ctrl, scenario=scenario, custom_length=custom_length, custom_seed=custom_seed, custom_water_temp=custom_water_temp, custom_initial_ec=custom_initial_ec)
        metrics = compute_ecosystem_metrics(
            traj["state_trace"], traj["flowrate"], traj["duration"], self.params, self.dt
        )
        return {"trajectory": traj, "metrics": metrics, "controller": "lstm_policy"}

    def run_learned_policy_pid_conditions(
        self,
        model: LSTMPolicy,
        normalizer: FeatureNormalizer,
        feature_columns: List[str],
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Run LSTM with same conditions as PID tuning (long horizon, normal disturbance, seed 42)."""
        # PID tuning validation conditions (from pid_tuner.py lines 628-639)
        val_cfg = self.config.get("pid_tuning", {}).get("validation", {})
        long_length = val_cfg.get("long_horizon_length", 3500)
        
        # Use same random initialization as PID evaluation
        # PID: temp = params.ambient_temp_mean + ep_rng.normal(0, 1.5)
        # PID: ec0 = ec_target + ep_rng.normal(0, 0.1)
        pid_rng = np.random.default_rng(42)
        temp = self.params.ambient_temp_mean + pid_rng.normal(0, 1.5)
        ec0 = self.ec_target + pid_rng.normal(0, 0.1)
        
        return self.run_learned_policy(
            model, normalizer, feature_columns,
            scenario="normal",
            device=device,
            custom_length=long_length,
            custom_seed=42,
            custom_water_temp=temp,
            custom_initial_ec=ec0,
        )

    def compare_all(
        self,
        model: LSTMPolicy,
        normalizer: FeatureNormalizer,
        feature_columns: List[str],
        scenarios: Optional[List[str]] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        scenarios = scenarios or [
            "normal",
            "sensor_noise",
            "ec_drift",
            "temp_spike",
            "delayed_response",
            "actuator_saturation",
        ]
        results = {}
        for sc in scenarios:
            noise_mult = 3.0 if sc == "sensor_noise" else 1.0
            results[sc] = {
                "lstm": self.run_learned_policy(
                    model, normalizer, feature_columns, sc, device
                ),
            }
        return results
