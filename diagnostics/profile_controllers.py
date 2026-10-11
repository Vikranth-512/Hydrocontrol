"""
Profile 3500-step single episode runtime and metrics for all 5 controllers.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from models.lstm_policy import LSTMPolicy
from preprocessing.normalization import FeatureNormalizer
from simulation.disturbances import DisturbanceGenerator
from simulation.dynamics import TankState
from simulation_runner.closed_loop_eval import ClosedLoopEvaluator
from utils import load_config

ROOT = Path(__file__).resolve().parent.parent


def main():
    config = load_config(ROOT / "configs" / "default.yaml")
    evaluator = ClosedLoopEvaluator(config, seed=42)
    length = 3500
    seed = 42
    scenario = "normal"

    # Pre-generate identical CRN disturbances and initial state
    gen = DisturbanceGenerator(evaluator.disturbance_config, np.random.default_rng(seed))
    disturbances = gen.build_schedule(scenario, length)
    s0 = TankState.create_initial(evaluator.params, rng=np.random.default_rng(seed))

    results = {}

    controllers = [
        ("no_action", lambda: evaluator.run_no_action(
            scenario=scenario, custom_length=length, custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(disturbances),
            prebuilt_initial_state=copy.deepcopy(s0),
        )),
        ("rule_based_reserve", lambda: evaluator.run_rule_based(
            scenario=scenario, custom_length=length, custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(disturbances),
            prebuilt_initial_state=copy.deepcopy(s0),
        )),
        ("pid", lambda: evaluator.run_pid(
            scenario=scenario, custom_length=length, custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(disturbances),
            prebuilt_initial_state=copy.deepcopy(s0),
        )),
        ("mpc_reserve", lambda: evaluator.run_mpc(
            scenario=scenario, custom_length=length, custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(disturbances),
            prebuilt_initial_state=copy.deepcopy(s0),
        )),
    ]

    # LSTM
    ckpt_path = ROOT / "checkpoints" / "best_model.pt"
    if ckpt_path.exists():
        with open(ROOT / "data" / "processed" / "feature_columns.json", "r") as f:
            feature_cols = json.load(f)["features"]
        normalizer = FeatureNormalizer.load(ROOT / "data" / "processed" / "scalers")
        model = LSTMPolicy(
            input_size=len(feature_cols),
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            output_size=2,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.eval()

        controllers.append(("lstm", lambda: evaluator.run_learned_policy(
            model=model, normalizer=normalizer, feature_columns=feature_cols,
            scenario=scenario, custom_length=length, custom_seed=seed,
            prebuilt_disturbances=copy.deepcopy(disturbances),
            prebuilt_initial_state=copy.deepcopy(s0),
            optimize_feature_engineering=True,
        )))

    print("=" * 95)
    print(f"BENCHMARK PROFILE: 3500-step Paired Episode (scenario={scenario}, seed={seed})")
    print("=" * 95)
    print(f"{'Controller':<20} | {'Runtime (s)':<11} | {'EC MAE':<8} | {'Final Bio':<10} | {'Final Rho':<10} | {'Total Dose':<10}")
    print("-" * 95)

    profile_data = {}
    for name, run_fn in controllers:
        t0 = time.perf_counter()
        res = run_fn()
        dur = time.perf_counter() - t0
        m = res["metrics"]
        print(f"{name:<20} | {dur:<11.2f} | {m['ec_mae']:<8.4f} | {m['final_biomass']:<10.2f} | {m['final_reserve_ratio']:<10.4f} | {m['total_dose']:<10.2f}")
        profile_data[name] = {
            "runtime_seconds": dur,
            "metrics": m,
        }
        if name == "mpc_reserve" and "controller_instance" in res:
            ctrl_inst = res["controller_instance"]
            profile_data[name]["avg_decision_latency_ms"] = ctrl_inst.average_decision_latency_ms

    print("=" * 95)
    return profile_data


if __name__ == "__main__":
    main()
