"""
Execute full benchmark matrix for mpc_reserve across all 11 scenarios x 30 seeds (330 episodes)
using exact CRN pairing matching the established 1,320-episode baseline matrix.
"""

from __future__ import annotations

import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from evaluation.multiseed_evaluator import MultiSeedEvaluator
from utils import load_config


def run_mpc_task(args):
    scenario, seed, horizon, config = args
    mse = MultiSeedEvaluator(config, include_mpc=True)
    crn = mse.generate_crn_schedule(scenario, seed, horizon)
    
    res_mpc = mse.closed_loop_eval.run_mpc(
        scenario=scenario,
        custom_length=horizon,
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
    
    rep_traj = None
    if scenario == "normal" and seed == 42:
        rep_traj = {
            "ec": res_mpc["trajectory"]["ec"].tolist(),
            "biomass": res_mpc["trajectory"]["biomass"].tolist(),
            "reserve_ratio": res_mpc["trajectory"]["reserve_ratio"].tolist(),
            "flowrate": res_mpc["trajectory"]["flowrate"].tolist(),
            "duration": res_mpc["trajectory"]["duration"].tolist(),
        }
        
    return res_mpc["metrics"], rep_traj


def main():
    config = load_config(ROOT / "configs" / "default.yaml")
    mse = MultiSeedEvaluator(config, include_mpc=True)
    
    scenarios = mse.scenarios
    seed_schedule = mse.seed_schedule
    horizon = mse.horizon
    
    tasks = []
    for sc in scenarios:
        for seed in seed_schedule[sc]:
            tasks.append((sc, seed, horizon, config))
            
    total_tasks = len(tasks)
    print(f"Launching {total_tasks} parallel MPC episodes (11 scenarios x 30 seeds)...")
    
    mpc_rows = []
    mpc_rep_traj = None
    t0 = time.perf_counter()
    completed = 0
    
    # Run across 8 worker processes
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_mpc_task, t): t for t in tasks}
        for future in as_completed(futures):
            metrics, rep_traj = future.result()
            mpc_rows.append(metrics)
            if rep_traj is not None:
                mpc_rep_traj = rep_traj
            completed += 1
            if completed % 25 == 0 or completed == total_tasks:
                elapsed = time.perf_counter() - t0
                pct = 100.0 * completed / total_tasks
                rate = completed / elapsed
                eta = (total_tasks - completed) / rate if rate > 0 else 0
                print(f"  [Progress] {completed}/{total_tasks} MPC episodes ({pct:.1f}%) in {elapsed:.1f}s (ETA: {eta:.1f}s)")
                
    total_time = time.perf_counter() - t0
    print(f"\nAll {total_tasks} MPC episodes completed in {total_time:.1f}s (avg {total_time/total_tasks:.2f}s per episode across 8 workers).")
    
    # Load existing 1,320 baseline episode rows
    df_existing = pd.read_csv(ROOT / "reports" / "multiseed_long_horizon" / "episode_results.csv")
    df_baselines = df_existing[df_existing["controller"] != "mpc_reserve"]
    print(f"Loaded {len(df_baselines)} baseline episode rows ({df_baselines['controller'].unique().tolist()}).")
    
    df_mpc = pd.DataFrame(mpc_rows)
    print(f"Created {len(df_mpc)} MPC episode rows.")
    
    # Combine into full 1,650 episode matrix
    df_full = pd.concat([df_baselines, df_mpc], ignore_index=True)
    print(f"Combined full benchmark dataset: {len(df_full)} episodes across {df_full['controller'].nunique()} controllers.")
    
    # Load representative trajectories
    rep_traj_file = ROOT / "reports" / "multiseed_long_horizon" / "representative_trajectories.json"
    with open(rep_traj_file, "r") as f:
        rep_trajectories = json.load(f)
    if mpc_rep_traj is not None:
        rep_trajectories["mpc_reserve"] = mpc_rep_traj
        
    # Run full statistical analysis
    print("\nComputing statistical analysis, paired differences, and 95% bootstrap CIs...")
    analysis = mse._analyze_results(df_full, rep_trajectories)
    
    # Export artifacts to reports/multiseed_long_horizon
    print(f"Exporting updated artifacts to {mse.report_dir}...")
    mse._export_artifacts(analysis)
    
    print("\n=== BENCHMARK VALIDATION COMPLETE ===")
    print(f"Total episodes: {len(df_full)}")
    print(f"Controllers: {sorted(df_full['controller'].unique())}")
    print(f"Scenarios: {len(df_full['scenario'].unique())}")
    print("Mean dose by controller:")
    for ctrl, grp in df_full.groupby("controller"):
        print(f"  {ctrl:<20}: total_dose={grp['total_dose'].mean():.2f} mL, biomass={grp['final_biomass'].mean():.2f} g/L, ec_mae={grp['ec_mae'].mean():.4f}")


if __name__ == "__main__":
    main()
