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


def stage_compare(config: dict, paths: dict) -> None:
    """Run both LSTM and PID pipelines and generate comparison plots."""
    from pathlib import Path
    import yaml
    
    # Create compare directory
    compare_dir = paths["figures"] / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    print(f"Comparison figures will be saved to: {compare_dir}")
    
    # Step 1: Run LSTM evaluation (3500 steps)
    print("\n" + "=" * 60)
    print("STEP 1: Running LSTM evaluation (3500 steps)")
    print("=" * 60)
    
    # Need to run the full pipeline up to evaluation
    # Run feature engineering and training first
    print("Running feature engineering...")
    preprocessed = stage_preprocess(config, paths)
    
    print("Training LSTM model...")
    trainer = stage_train(config, paths, preprocessed)
    
    print("Running LSTM evaluation...")
    lstm_results = stage_evaluate(config, paths, trainer, preprocessed)
    
    # Load LSTM metrics from evaluation results
    lstm_metrics_path = paths["processed"] / "evaluation_results.json"
    with open(lstm_metrics_path, "r") as f:
        lstm_data = json.load(f)
    lstm_metrics = lstm_data["long_horizon_lstm"]["metrics"]
    
    # Step 2: Run PID tuning (3500 steps)
    print("\n" + "=" * 60)
    print("STEP 2: Running PID tuning (3500 steps)")
    print("=" * 60)
    
    out_json = paths["processed"] / "pid_tuning_results.json"
    fig_dir = paths["figures"] / "pid_tuning"
    pid_results = run_pid_tuning(config, out_json, fig_dir)
    pid_best = pid_results.get("best", {})
    
    # Load PID metrics from the saved file
    pid_metrics_path = fig_dir / "pid_metrics.json"
    with open(pid_metrics_path, "r") as f:
        pid_metrics = json.load(f)
    
    # Step 3: Generate comparison plots
    print("\n" + "=" * 60)
    print("STEP 3: Generating comparison plots")
    print("=" * 60)
    
    plotter = Plotter(compare_dir)
    dt = config["simulation"]["dt_seconds"]
    reserve_target = config.get("ecosystem_control", {}).get("rho_setpoint", 0.50)
    
    # Generate 10 comparison plots
    print("Generating comparison plots...")
    plotter.plot_compare_biomass(lstm_results["long_horizon_lstm"], pid_best, dt)
    plotter.plot_compare_health(lstm_results["long_horizon_lstm"], pid_best, dt)
    plotter.plot_compare_reserve(lstm_results["long_horizon_lstm"], pid_best, dt, reserve_target)
    plotter.plot_compare_dosing(lstm_results["long_horizon_lstm"], pid_best, dt)
    plotter.plot_compare_ec(lstm_results["long_horizon_lstm"], pid_best, dt)
    plotter.plot_compare_biomass_vs_dose(lstm_metrics, pid_metrics)
    plotter.plot_compare_cumulative_dose(lstm_results["long_horizon_lstm"], pid_best, dt)
    plotter.plot_compare_radar(lstm_metrics, pid_metrics)
    plotter.plot_compare_tradeoff(lstm_metrics, pid_metrics)
    plotter.plot_compare_parallel_coordinates(lstm_metrics, pid_metrics)
    
    print(f"\nAll comparison plots saved to: {compare_dir}")
    print("Comparison complete!")


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
    print(f"Exported to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Algae dosing policy learning pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all",
            "validate_dynamics",
            "generate",
            "label",
            "label_diagnostics",
            "preprocess",
            "train",
            "evaluate",
            "export",
            "tune_pid",
            "compare",
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

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
