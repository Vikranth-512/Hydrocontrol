"""
Algae nutrient dosing — learned control policy pipeline.

Usage:
    python main.py --config configs/default.yaml --stage all
    python main.py --stage generate
    python main.py --stage label
    python main.py --stage train
    python main.py --stage evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.normalization import FeatureNormalizer
from preprocessing.sequence_builder import SequenceBuilder
from simulation.optimization_labeler import OptimizationLabeler
from simulation.scenario_registry import get_multiseed_evaluation_scenarios
from simulation.synthetic_generator import SyntheticTrajectoryGenerator
from simulation_runner.closed_loop_eval import ClosedLoopEvaluator
from training.evaluation import compute_prediction_metrics
from training.train import Trainer
from utils import (
    ensure_dirs,
    get_trajectory_index_padding,
    load_config,
    list_trajectory_files,
    save_experiment_metadata,
    set_seed,
)
from simulation.validate_dynamics import run_validation_suite
from simulation.validate.empirical_runner import run_empirical_validation
from visualization.plots import Plotter
from controllers.pid_tuner import run_pid_tuning


def stage_generate(config: dict, paths: dict) -> Path:
    padding = get_trajectory_index_padding(config)
    print("=== Stage: Synthetic trajectory generation ===")
    print(f"  Trajectory filename padding: {padding}")
    gen = SyntheticTrajectoryGenerator(config, seed=config.get("seed", 42))
    return gen.generate_dataset(paths["synthetic"])


def stage_label(config: dict, paths: dict, with_diagnostics: bool = False) -> Path:
    padding = get_trajectory_index_padding(config)
    expected = config.get("simulation", {}).get("num_trajectories", 200)
    print("=== Stage: Optimization-based labeling (precision regulation) ===")
    print(f"  Trajectory filename padding: {padding}")
    print(f"  Expected trajectories: {expected}")
    lab = config.get("labeling", {})
    print(f"  horizon_steps={lab.get('horizon_steps', 60)}, mode=option_b")
    if lab.get("debug_mode", False):
        print("  debug_mode=ON (verbose cost/candidate logging)")
    labeler = OptimizationLabeler(config)
    fig_dir = paths["figures"] / "labeling_diagnostics" if with_diagnostics else None
    combined_path = labeler.label_dataset(
        paths["synthetic"],
        paths["processed"],
        run_diagnostics=with_diagnostics,
        figures_dir=fig_dir,
        expected_trajectories=expected,
    )
    df = pd.read_csv(combined_path)
    unique_ids = df["trajectory_id"].nunique()
    print(f"[LABEL] Dataset validation passed: {unique_ids} unique trajectory IDs")
    return combined_path


def stage_label_diagnostics(config: dict, paths: dict) -> Path:
    """Re-label a single trajectory with full diagnostic capture + plots."""
    print("=== Stage: Labeling diagnostics ===")
    syn = paths["synthetic"]
    traj_files = list_trajectory_files(syn, labeled=False)
    if not traj_files:
        raise FileNotFoundError(f"No trajectories in {syn}; run generate first.")
    df = pd.read_csv(traj_files[0])
    labeler = OptimizationLabeler(config)
    labeler.label_trajectory(df.head(min(80, len(df))), sample_diagnostics=True)
    fig_dir = paths["figures"] / "labeling_diagnostics"
    from visualization.labeling_diagnostics import LabelingDiagnosticPlotter

    plotter = LabelingDiagnosticPlotter(fig_dir)
    plotter.plot_all(labeler.get_diagnostic_samples(), labeler.obj_cfg)
    out = paths["processed"] / "labeling_diagnostic_samples.json"
    import json

    with open(out, "w") as f:
        json.dump(labeler.get_diagnostic_samples(), f, indent=2)
    print(f"Diagnostics saved: {fig_dir}")
    return out


def stage_preprocess(config: dict, paths: dict) -> dict:
    print("=== Stage: Feature engineering & sequences ===")
    labeled_path = paths["processed"] / "all_trajectories_labeled.csv"
    if not labeled_path.exists():
        stage_label(config, paths)
    df = pd.read_csv(labeled_path)

    sim = config.get("simulation", {})
    prep = config.get("preprocessing", {})
    dyn = config.get("dynamics", {})
    fe = FeatureEngineer(
        ec_target=sim.get("ec_target", 1.2),
        rolling_window=prep.get("rolling_window", 8),
        include_biological=prep.get("include_biological_features", False),
        internal_capacity=dyn.get("internal_capacity", 0.5),
    )
    df = fe.transform(df)
    feature_cols = fe.get_available_feature_columns(df)
    target_cols = ["optimal_flowrate", "optimal_duration"]

    sb = SequenceBuilder(
        sequence_length=prep.get("sequence_length", 32),
        prediction_horizon=prep.get("prediction_horizon", 1),
        feature_columns=feature_cols,
        target_columns=target_cols,
    )
    X, y, traj_ids = sb.build_from_dataframe(df)
    splits = sb.train_val_test_split(
        X, y, traj_ids,
        train_ratio=prep.get("train_ratio", 0.7),
        val_ratio=prep.get("val_ratio", 0.15),
        seed=config.get("seed", 42),
    )

    normalizer = FeatureNormalizer(scaler_type=prep.get("scaler_type", "standard"))
    normalizer.fit(
        splits["X_train"], splits["y_train"], feature_cols, target_cols
    )
    splits["X_train"] = normalizer.transform_features(splits["X_train"])
    splits["X_val"] = normalizer.transform_features(splits["X_val"])
    splits["X_test"] = normalizer.transform_features(splits["X_test"])
    splits["y_train"] = normalizer.transform_targets(splits["y_train"])
    splits["y_val"] = normalizer.transform_targets(splits["y_val"])
    splits["y_test"] = normalizer.transform_targets(splits["y_test"])

    scaler_dir = paths.get("scalers", paths["processed"] / "scalers")
    normalizer.save(scaler_dir)

    np.savez(
        paths["processed"] / "sequences.npz",
        X_train=splits["X_train"],
        y_train=splits["y_train"],
        X_val=splits["X_val"],
        y_val=splits["y_val"],
        X_test=splits["X_test"],
        y_test=splits["y_test"],
    )
    with open(paths["processed"] / "feature_columns.json", "w") as f:
        json.dump({"features": feature_cols, "targets": target_cols}, f, indent=2)

    return {"splits": splits, "normalizer": normalizer, "feature_cols": feature_cols}


def stage_train(config: dict, paths: dict, preprocessed: dict) -> Trainer:
    print("=== Stage: LSTM policy training ===")
    splits = preprocessed["splits"]
    normalizer = preprocessed["normalizer"]

    trainer = Trainer(config)
    input_size = splits["X_train"].shape[2]
    trainer.build_model(input_size)
    trainer.train(
        splits["X_train"],
        splits["y_train"],
        splits["X_val"],
        splits["y_val"],
        normalizer,
        paths["checkpoints"],
    )
    return trainer


def stage_evaluate(config: dict, paths: dict, trainer: Trainer, preprocessed: dict) -> dict:
    print("=== Stage: Offline & closed-loop evaluation ===")
    normalizer = preprocessed["normalizer"]
    splits = preprocessed["splits"]
    feature_cols = preprocessed["feature_cols"]

    y_pred = trainer.predict(splits["X_test"], normalizer)
    y_true = normalizer.inverse_transform_targets(splits["y_test"])
    pred_metrics = compute_prediction_metrics(y_true, y_pred)

    print("Test prediction metrics:", pred_metrics)

    plotter = Plotter(paths["figures"])
    plotter.plot_predicted_vs_optimal(y_true, y_pred)
    plotter.plot_training_curves(trainer.history)
    plotter.plot_error_distribution(y_true[:, 0] - y_pred[:, 0])

    device = trainer.device
    model = trainer.model
    evaluator = ClosedLoopEvaluator(config, seed=config.get("seed", 42))
    
    # Run LSTM with PID tuning conditions (3500 long horizon run)
    lstm_pid_conditions = evaluator.run_learned_policy_pid_conditions(
        model, normalizer, feature_cols, device=device
    )

    # Generate new evaluation figures (3 figures for 3500 long horizon run)
    dt = config["simulation"]["dt_seconds"]
    reserve_target = config.get("ecosystem_control", {}).get("rho_setpoint", 0.50)
    
    print("Generating LSTM evaluation figures for 3500 long horizon run...")
    plotter.plot_long_horizon_overview(lstm_pid_conditions, dt, reserve_target)
    plotter.plot_lstm_smoothness(lstm_pid_conditions, dt)
    plotter.plot_nutrient_efficiency(lstm_pid_conditions, dt)
    print("All 3 evaluation figures generated successfully.")

    # Save results - only save specific metrics and trajectory arrays (exclude state_trace)
    results_path = paths["processed"] / "evaluation_results.json"
    
    # Extract only the requested metrics
    requested_metrics = [
        "health_mean", "health_mean_tail", "damage_mean", "damage_cumulative",
        "reserve_mean", "reserve_floor_frac", "starvation_fraction",
        "biomass_mean", "biomass_cv_tail", "bloom_fraction",
        "collapse_free_duration", "biomass_persistence",
        "total_dose", "dose_per_health", "control_smoothness",
        "ec_mean", "ec_std", "ec_min", "ec_max", "ec_range"
    ]
    
    metrics_dict = lstm_pid_conditions["metrics"]
    filtered_metrics = {k: metrics_dict.get(k) for k in requested_metrics}
    
    # Only save prediction metrics and LSTM metrics (no trajectory data)
    serializable = {
        "prediction_metrics": pred_metrics,
        "long_horizon_lstm": {
            "metrics": filtered_metrics,
        },
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    
    return {"long_horizon_lstm": lstm_pid_conditions}


def stage_compare(config: dict, paths: dict) -> Dict[str, Any]:
    """
    Run paired 3500-step long-horizon benchmark across four controllers:
      1. No-action (uncontrolled baseline)
      2. Reserve-ratio rule-based heuristic
      3. PID controller
      4. Learned LSTM policy
    Evaluated under identical Common Random Numbers (CRN) and initial conditions.
    """
    from pathlib import Path
    from models.lstm_policy import LSTMPolicy
    
    # Create compare and reports directories
    compare_dir = paths["figures"] / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"Comparison figures will be saved to: {compare_dir}")
    print(f"Comparison reports will be saved to: {reports_dir}")
    
    # Check if trained LSTM model and preprocessed feature metadata exist
    ckpt_path = paths["checkpoints"] / "best_model.pt"
    scalers_dir = paths.get("scalers", paths["processed"] / "scalers")
    feat_cols_path = paths["processed"] / "feature_columns.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if ckpt_path.exists() and feat_cols_path.exists() and scalers_dir.exists():
        print(f"Loading existing LSTM checkpoint from {ckpt_path}...")
        with open(feat_cols_path, "r") as f:
            feat_meta = json.load(f)
        feature_cols = feat_meta["features"]
        
        normalizer = FeatureNormalizer.load(scalers_dir)
        mcfg = config.get("model", {})
        model = LSTMPolicy(
            input_size=len(feature_cols),
            hidden_size=mcfg.get("hidden_size", 128),
            num_layers=mcfg.get("num_layers", 2),
            dropout=mcfg.get("dropout", 0.2),
            output_size=mcfg.get("output_size", 2),
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    else:
        print("Checkpoint not found. Running preprocessing and training...")
        preprocessed = stage_preprocess(config, paths)
        trainer = stage_train(config, paths, preprocessed)
        model = trainer.model
        normalizer = preprocessed["normalizer"]
        feature_cols = preprocessed["feature_cols"]
        device = trainer.device

    # Run Paired 3500-Step Benchmark (All 4 controllers, exact CRN disturbances & s0)
    print("\n" + "=" * 70)
    print("RUNNING PAIRED 3500-STEP LONG-HORIZON BENCHMARK (CRN DISTURBANCES)")
    print("=" * 70)
    evaluator = ClosedLoopEvaluator(config, seed=config.get("seed", 42))
    benchmark_results = evaluator.run_paired_benchmark_3500(
        model=model,
        normalizer=normalizer,
        feature_columns=feature_cols,
        device=device,
    )
    
    # Extract metrics dictionary
    metrics_by_ctrl = {k: v["metrics"] for k, v in benchmark_results.items()}
    
    # Format comparison table
    rows = []
    ctrl_display_names = [
        ("no_action", "No-action"),
        ("rule_based_reserve", "Rule-based"),
        ("pid", "PID"),
        ("lstm", "LSTM"),
        ("mpc_reserve", "MPC Reserve"),
    ]
    for c_key, c_name in ctrl_display_names:
        if c_key not in metrics_by_ctrl:
            continue
        m = metrics_by_ctrl.get(c_key, {})
        rows.append({
            "Controller": c_name,
            "EC MAE": f"{m.get('ec_mae', 0.0):.4f}",
            "EC RMSE": f"{m.get('ec_rmse', 0.0):.4f}",
            "Final EC": f"{m.get('final_ec', 0.0):.4f}",
            "Final Biomass": f"{m.get('final_biomass', 0.0):.2f}",
            "Final Reserve Ratio": f"{m.get('final_reserve_ratio', 0.0):.4f}",
            "Intervention Fraction": f"{m.get('intervention_fraction', 0.0):.4f}",
            "Total Dose": f"{m.get('total_dose', 0.0):.2f}",
            "Effective Horizon": m.get("effective_horizon", 3500),
            "Terminated": "YES" if m.get("episode_terminated", False) else "NO",
        })
    df_comp = pd.DataFrame(rows)
    
    # Print formatted markdown table to stdout
    print("\n" + "=" * 70)
    print("3500-STEP LONG-HORIZON BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    headers = list(df_comp.columns)
    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, r in df_comp.iterrows():
        md_lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    comp_table_str = "\n".join(md_lines)
    print(comp_table_str)
    
    # Save CSV and Markdown reports
    csv_report_path = reports_dir / "long_horizon_comparison.csv"
    md_report_path = reports_dir / "long_horizon_comparison.md"
    df_comp.to_csv(csv_report_path, index=False)
    with open(md_report_path, "w") as f:
        f.write("# 3500-Step Long-Horizon Controller Benchmark Comparison\n\n")
        f.write(comp_table_str + "\n\n")
        f.write("### Controller Information-Set Audit\n\n")
        f.write("| Controller | Reserve Ratio (rho) | Biomass (B) | EC | 17-Feature Sensor History | Nature |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :--- |\n")
        f.write("| **No-action** | — | — | — | — | Uncontrolled baseline, open-loop |\n")
        f.write("| **Rule-based** | Yes | — | — | — | Reserve-ratio feedback heuristic |\n")
        f.write("| **PID** | Yes | Yes (scale) | Yes (headroom) | — | Classical feedback control |\n")
        f.write("| **LSTM** | Yes | (via features) | Yes | Yes | Learned policy (imitation) |\n")
        f.write("| **MPC Reserve** | Yes | — | — | — | Predictive model-based baseline |\n")
    print(f"\nSaved CSV comparison: {csv_report_path}")
    print(f"Saved Markdown comparison: {md_report_path}")
    
    # Update data/processed/evaluation_results.json
    results_path = paths["processed"] / "evaluation_results.json"
    eval_json = {}
    if results_path.exists():
        with open(results_path, "r") as f:
            try:
                eval_json = json.load(f)
            except Exception:
                eval_json = {}
    
    for c_key, c_field in [
        ("no_action", "long_horizon_no_action"),
        ("rule_based_reserve", "long_horizon_rule_based"),
        ("pid", "long_horizon_pid"),
        ("lstm", "long_horizon_lstm"),
        ("mpc_reserve", "long_horizon_mpc_reserve"),
    ]:
        if c_key in metrics_by_ctrl:
            eval_json[c_field] = {"metrics": metrics_by_ctrl.get(c_key, {})}
    
    with open(results_path, "w") as f:
        json.dump(eval_json, f, indent=2)
    print(f"Updated benchmark records in: {results_path}")
    
    # Generate 10 Generic Comparison Figures
    print("\n" + "=" * 70)
    print("GENERATING 10 CONTROLLER BENCHMARK COMPARISON PLOTS")
    print("=" * 70)
    plotter = Plotter(compare_dir)
    dt = config["simulation"]["dt_seconds"]
    reserve_target = config.get("ecosystem_control", {}).get("rho_setpoint", 0.50)
    
    plotter.plot_compare_biomass(benchmark_results, dt=dt)
    plotter.plot_compare_health(benchmark_results, dt=dt)
    plotter.plot_compare_reserve(benchmark_results, dt=dt, reserve_target=reserve_target)
    plotter.plot_compare_dosing(benchmark_results, dt=dt)
    plotter.plot_compare_ec(benchmark_results, dt=dt)
    plotter.plot_compare_biomass_vs_dose(metrics_by_ctrl)
    plotter.plot_compare_cumulative_dose(benchmark_results, dt=dt)
    plotter.plot_compare_radar(metrics_by_ctrl)
    plotter.plot_compare_tradeoff(metrics_by_ctrl)
    plotter.plot_compare_parallel_coordinates(metrics_by_ctrl)
    
    print(f"\nAll 10 comparison plots successfully generated in: {compare_dir}")
    print("Stage compare complete!")
    return benchmark_results



def stage_export(config: dict, paths: dict, trainer: Trainer, preprocessed: dict) -> None:
    print("=== Stage: Model export ===")
    model = trainer.model
    assert model is not None
    model.eval()
    seq_len = config["preprocessing"]["sequence_length"]
    n_feat = preprocessed["splits"]["X_test"].shape[2]
    example = torch.randn(1, seq_len, n_feat)

    export_fmt = config.get("export", {}).get("format", "torchscript")
    if export_fmt == "onnx":
        path = paths["checkpoints"] / "policy.onnx"
        model.export_onnx(example, str(path))
    else:
        path = paths["checkpoints"] / "policy.pt"
        model.export_torchscript(example, str(path))
def stage_multiseed_long_horizon(config: dict, paths: dict, dry_run: bool = False) -> Dict[str, Any]:
    """
    Run paired multi-seed, multi-regime 3500-step statistical evaluation across four controllers:
      1. No-action (uncontrolled baseline)
      2. Reserve-ratio rule-based heuristic
      3. PID controller
      4. Learned LSTM policy
    Evaluated under Common Random Numbers (CRN) with decoupled sensor noise innovations.
    """
    from evaluation.multiseed_evaluator import MultiSeedEvaluator
    from visualization.multiseed_plots import generate_all_plots
    from models.lstm_policy import LSTMPolicy

    # Dry-run mode: print evaluation design and exit without executing
    if dry_run:
        ev_cfg = config.get("evaluation", {}).get("multiseed_long_horizon", {})
        scenarios = ev_cfg.get("scenarios", get_multiseed_evaluation_scenarios())
        n_seeds = int(ev_cfg.get("n_seeds", 30))
        horizon = int(ev_cfg.get("horizon", 3500))
        controllers = ["no_action", "rule_based_reserve", "pid", "lstm", "mpc_reserve"]
        n_total = len(scenarios) * n_seeds * len(controllers)
        n_cells = len(scenarios) * len(controllers)
        expected_scenarios = 11
        expected_cells = 55   # 11 scenarios × 5 controllers
        expected_episodes = 1650  # 11 × 30 × 5
        
        print("\n" + "=" * 70)
        print("DRY RUN: Paired Multi-Seed Long-Horizon Evaluation Design")
        print("=" * 70)
        print(f"  Scenarios ({len(scenarios)}): {scenarios}")
        print(f"  Controllers ({len(controllers)}): {controllers}")
        print(f"  Seeds per scenario: {n_seeds}")
        print(f"  Horizon per episode: {horizon} steps")
        print(f"  Total paired episodes: {len(scenarios) * n_seeds}")
        print(f"  Total controller executions: {n_total}")
        print(f"  Condition cells (controllers × scenarios): {n_cells}")
        print(f"  Expected scenarios: {expected_scenarios}")
        print(f"  Expected cells: {expected_cells}")
        print(f"  Expected episodes: {expected_episodes}")
        print(f"  missing_data: excluded")
        print("=" * 70 + "\n")
        
        # Validate counts
        assert len(scenarios) == expected_scenarios, f"Expected {expected_scenarios} scenarios, got {len(scenarios)}"
        assert n_cells == expected_cells, f"Expected {expected_cells} cells, got {n_cells}"
        assert n_total == expected_episodes, f"Expected {expected_episodes} episodes, got {n_total}"
        assert "missing_data" not in scenarios, "missing_data should be excluded from scenarios"
        
        return {}

    
    ckpt_path = paths["checkpoints"] / "best_model.pt"
    scalers_dir = paths.get("scalers", paths["processed"] / "scalers")
    feat_cols_path = paths["processed"] / "feature_columns.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if ckpt_path.exists() and feat_cols_path.exists() and scalers_dir.exists():
        print(f"Loading existing LSTM checkpoint from {ckpt_path}...")
        with open(feat_cols_path, "r") as f:
            feat_meta = json.load(f)
        feature_cols = feat_meta["features"]
        
        normalizer = FeatureNormalizer.load(scalers_dir)
        mcfg = config.get("model", {})
        model = LSTMPolicy(
            input_size=len(feature_cols),
            hidden_size=mcfg.get("hidden_size", 128),
            num_layers=mcfg.get("num_layers", 2),
            dropout=mcfg.get("dropout", 0.2),
            output_size=mcfg.get("output_size", 2),
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    else:
        print("Checkpoint or feature scalers not found. Running preprocessing and training...")
        preprocessed = stage_preprocess(config, paths)
        trainer = stage_train(config, paths, preprocessed)
        model = trainer.model
        normalizer = preprocessed["normalizer"]
        feature_cols = preprocessed["feature_cols"]

    evaluator = MultiSeedEvaluator(
        config=config,
        model=model,
        normalizer=normalizer,
        feature_columns=feature_cols,
        device=device,
        include_mpc=True,
    )
    
    analysis = evaluator.run_full_evaluation()
    
    # Generate distribution and paired plots
    print(f"\nGenerating publication-quality figures in {evaluator.figure_dir}...")
    generate_all_plots(analysis, evaluator.figure_dir)
    
    # Print publication tables to stdout
    print("\n" + "=" * 80)
    print("PAIRED MULTI-SEED 3500-STEP EVALUATION: LSTM vs PID (Primary Reviewer Comparison)")
    print("=" * 80)
    df_pid = analysis["df_lstm_vs_pid"]
    disp_cols = [
        "scenario",
        "metric",
        "mean_difference",
        "ci95_lower",
        "ci95_upper",
        "win_rate",
        "episodes_exceeding_preregistered_effect_threshold",
    ]
    print(df_pid[disp_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("ML-DISCRIMINABILITY AUDIT: LSTM vs RESERVE-RULE (Practical Equivalence Margin)")
    print("=" * 80)
    df_eq = analysis["df_equivalence"]
    df_eq_sub = df_eq[df_eq["comparison"] == "lstm_vs_rule_based_reserve"][
        ["scenario", "metric", "mean_diff", "ci95_lower", "ci95_upper", "threshold_delta", "verdict"]
    ]
    print(df_eq_sub.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"Artifacts successfully saved to: {evaluator.report_dir}")
    print(f"Figures successfully saved to:   {evaluator.figure_dir}")
    print("=" * 80 + "\n")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Algae dosing policy learning pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print evaluation design (scenarios, controllers, seeds, horizon, episode count) and exit without running."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all",
            "validate_dynamics",
            "empirical_biological_validation",
            "generate",
            "label",
            "label_diagnostics",
            "preprocess",
            "train",
            "evaluate",
            "export",
            "tune_pid",
            "compare",
            "evaluate_intervention",
            "multiseed_long_horizon",
        ],
    )
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    set_seed(config.get("seed", 42))
    paths = ensure_dirs(config)
    save_experiment_metadata(
        paths.get("metadata", paths["processed"] / "metadata"),
        config,
        {"stage": args.stage},
    )

    preprocessed = None
    trainer = None

    if args.stage in ("all", "validate_dynamics"):
        print("=== Stage: Dynamics validation ===")
        val_dir = paths["figures"] / "dynamics_validation"
        val_results = run_validation_suite(config, val_dir)
        print("Validation results:", val_results)
        if not val_results.get("validation_passed", False):
            print("WARNING: Dynamics validation criteria not fully met — review plots in", val_dir)

    if args.stage == "empirical_biological_validation":
        print("=== Stage: Empirical Biological Validation ===")
        empirical_results = run_empirical_validation(paths["figures"])
        print("Empirical validation results:", empirical_results)

    if args.stage in ("all", "generate"):
        stage_generate(config, paths)

    label_ok = True
    if args.stage in ("all", "label"):
        try:
            stage_label(
                config,
                paths,
                with_diagnostics=config.get("labeling", {}).get(
                    "run_diagnostics_on_label", False
                ),
            )
        except Exception as e:
            label_ok = False
            print(f"LABELING STAGE FAILED: {e}")
            if args.stage == "label":
                raise

    if args.stage == "all" and not label_ok:
        print("Pipeline finished with labeling errors — not complete.")
        sys.exit(1)

    if args.stage == "label_diagnostics":
        stage_label_diagnostics(config, paths)

    if args.stage in ("all", "preprocess", "train", "evaluate", "export"):
        preprocessed = stage_preprocess(config, paths)

    if args.stage in ("all", "train", "evaluate", "export"):
        trainer = stage_train(config, paths, preprocessed)

    if args.stage in ("all", "evaluate"):
        stage_evaluate(config, paths, trainer, preprocessed)

    if args.stage in ("all", "export"):
        stage_export(config, paths, trainer, preprocessed)

    if args.stage == "tune_pid":
        if not config.get("pid_tuning", {}).get("enabled", True):
            print("pid_tuning.enabled is false — skipping")
        else:
            print("=== Stage: PID gain tuning ===")
            out_json = paths["processed"] / "pid_tuning_results.json"
            fig_dir = paths["figures"] / "pid_tuning"
            results = run_pid_tuning(config, out_json, fig_dir)
            best = results.get("best", {})
            print("\n" + "=" * 50)
            print("BEST PID GAINS:")
            print(f"  Kp = {best.get('kp', 'N/A')}")
            print(f"  Ki = {best.get('ki', 'N/A')}")
            print(f"  Kd = {best.get('kd', 'N/A')}")
            print("=" * 50)
            if best.get("validation"):
                v = best["validation"]
                print(f"Validation mean score: {v.get('mean_score', 0):.4f}")
                print(f"Validation std:      {v.get('std_score', 0):.4f}")
                print(f"Long-horizon score:  {v.get('long_horizon_mean', 0):.4f}")
            
            # Save PID metrics as separate JSON file
            if best.get("metrics"):
                metrics = best["metrics"]
                pid_metrics = {
                    "health_mean": metrics.get("health_mean"),
                    "health_mean_tail": metrics.get("health_mean_tail"),
                    "damage_mean": metrics.get("damage_mean"),
                    "damage_cumulative": metrics.get("damage_cumulative"),
                    "reserve_mean": metrics.get("reserve_mean"),
                    "reserve_floor_frac": metrics.get("reserve_floor_frac", metrics.get("reserve_floor_fraction")),
                    "starvation_fraction": metrics.get("starvation_fraction"),
                    "biomass_mean": metrics.get("biomass_mean_tail"),  # PID uses biomass_mean_tail
                    "biomass_cv_tail": metrics.get("biomass_cv_tail"),
                    "bloom_fraction": metrics.get("bloom_fraction"),
                    "collapse_free_duration": metrics.get("collapse_free_duration"),
                    "biomass_persistence": metrics.get("biomass_persistence"),
                    "total_dose": metrics.get("total_dose", metrics.get("nutrient_usage")),
                    "dose_per_health": metrics.get("dose_per_health"),
                    "control_smoothness": metrics.get("control_smoothness"),
                    "ec_mean": metrics.get("ec_mean"),
                    "ec_std": metrics.get("ec_std"),
                    "ec_min": metrics.get("ec_min"),
                    "ec_max": metrics.get("ec_max"),
                    "ec_range": metrics.get("ec_range"),
                }
                pid_metrics_path = fig_dir / "pid_metrics.json"
                with open(pid_metrics_path, "w") as f:
                    json.dump(pid_metrics, f, indent=2)
                print(f"PID metrics saved to: {pid_metrics_path}")
                print("Metrics:", best["metrics"])
            print(f"\nResults saved: {out_json}")
            print(f"Figures saved:  {fig_dir}")
            tuned_pid_path = paths["processed"] / "pid_tuned_gains.yaml"
            import yaml
            with open(tuned_pid_path, "w") as f:
                yaml.dump(
                    {"evaluation": {"pid": {"kp": best["kp"], "ki": best["ki"], "kd": best["kd"]}}},
                    f,
                )
            print(f"Tuned gains YAML: {tuned_pid_path}")

    if args.stage == "compare":
        print("=== Stage: PID vs LSTM Comparison ===")
        stage_compare(config, paths)

    if args.stage == "evaluate_intervention":
        print("=== Stage: Reviewer-Response Intervention Evaluation ===")
        from run_intervention_evaluation import run_evaluation
        out_dir = ROOT / "reports" / "intervention_evaluation"
        ckpt_path = ROOT / "checkpoints" / "best_model.pt"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run_evaluation(config, out_dir, ckpt_path, device=device)

    if args.stage == "multiseed_long_horizon":
        print("=== Stage: Paired Multi-Seed 3500-Step Statistical Evaluation ===")
        stage_multiseed_long_horizon(config, paths, dry_run=args.dry_run)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
