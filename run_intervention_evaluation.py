"""
Runner script for Reviewer-Response Intervention Evaluation.

Executes offline evaluation on the held-out test split, comparing the trained LSTM
policy against the Always-No-Action baseline under the canonical 61-action representation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.intervention_evaluator import (
    InterventionEvaluator,
    discretize_action_nearest,
    get_canonical_action_space,
    is_intervention_vec,
    load_canonical_test_split,
)
from models.lstm_policy import LSTMPolicy
from preprocessing.sequence_builder import SequenceDataset
from utils import load_config, set_seed
from visualization.intervention_plots import (
    plot_intervention_confusion_matrix,
    plot_scenario_intervention_metrics,
)

def df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as standard markdown table without external dependencies."""
    headers = list(df.columns)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def run_evaluation(
    config: Dict[str, Any],
    output_dir: Path,
    checkpoint_path: Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Execute complete intervention evaluation pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load canonical test split
    print("[1/6] Loading canonical held-out test split...")
    test_split = load_canonical_test_split(config, project_root=ROOT)
    X_test = test_split.X_test
    y_test_phys = test_split.y_test_physical
    scenarios = test_split.scenarios
    normalizer = test_split.normalizer
    
    n_samples = len(y_test_phys)
    print(f"      Total test samples: {n_samples:,}")
    
    # 2. Load trained LSTM checkpoint
    print(f"[2/6] Loading model checkpoint from {checkpoint_path}...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    mcfg = config.get("model", {})
    input_size = X_test.shape[2]
    model = LSTMPolicy(
        input_size=input_size,
        hidden_size=mcfg.get("hidden_size", 128),
        num_layers=mcfg.get("num_layers", 2),
        dropout=mcfg.get("dropout", 0.2),
        output_size=mcfg.get("output_size", 2),
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 3. Generate raw predictions in physical units
    print("[3/6] Generating LSTM predictions on test split...")
    ds = SequenceDataset(X_test, np.zeros((n_samples, 2)))
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    preds_norm_list = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds_norm_list.append(out)
    preds_norm = np.vstack(preds_norm_list)
    y_pred_phys = normalizer.inverse_transform_targets(preds_norm)
    
    # 4. Run Two-Stage Intervention Evaluation
    print("[4/6] Evaluating Stage 1 & Stage 2 metrics...")
    evaluator = InterventionEvaluator()
    lstm_metrics = evaluator.evaluate(y_test_phys, y_pred_phys)
    baseline_metrics = evaluator.evaluate_baseline_always_no_action(y_test_phys)
    
    # Operating regime breakdown
    primary_regimes = ["normal", "heatwave", "nutrient_depletion", "cold_shock"]
    scenario_df = evaluator.evaluate_by_regime(
        y_test_phys, y_pred_phys, scenarios, target_regimes=primary_regimes
    )
    
    # All present regimes breakdown (comprehensive supplementary)
    all_present_regimes = sorted(list(np.unique(scenarios)))
    all_scenario_df = evaluator.evaluate_by_regime(
        y_test_phys, y_pred_phys, scenarios, target_regimes=all_present_regimes
    )
    
    # Secondary magnitude bins
    y_pred_snapped = discretize_action_nearest(y_pred_phys, evaluator.action_space)
    magnitude_df = evaluator.evaluate_magnitude_bins(y_test_phys, y_pred_phys, y_pred_snapped)
    
    # Consolidated comparison table & text
    comp_df = evaluator.build_comparison_table(lstm_metrics, baseline_metrics)
    interpretation_text = evaluator.generate_reviewer_interpretation(
        lstm_metrics, baseline_metrics, scenario_df
    )
    
    # 5. Generate Figures
    print("[5/6] Generating publication-quality figures...")
    cm_counts = {
        "true_negatives": lstm_metrics["true_negatives"],
        "false_positives": lstm_metrics["false_positives"],
        "false_negatives": lstm_metrics["false_negatives"],
        "true_positives": lstm_metrics["true_positives"],
    }
    plot_intervention_confusion_matrix(
        cm_counts, output_dir / "intervention_confusion_matrix.png"
    )
    plot_scenario_intervention_metrics(
        scenario_df, output_dir / "scenario_intervention_f1.png"
    )
    
    # 6. Save Artifacts
    print("[6/6] Saving evaluation artifacts...")
    # JSON metrics
    json_out = {
        "metadata": {
            "model_checkpoint": str(checkpoint_path),
            "checkpoint_epoch": ckpt.get("epoch", "unknown"),
            "dataset_version": "all_trajectories_labeled.csv",
            "test_split_samples": n_samples,
            "action_space_size": len(evaluator.action_space),
            "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": config.get("seed", 42),
        },
        "lstm_metrics": lstm_metrics,
        "always_no_action_baseline": baseline_metrics,
    }
    with open(output_dir / "intervention_metrics.json", "w") as f:
        json.dump(json_out, f, indent=2)
        
    # CSVs
    comp_df.to_csv(output_dir / "baseline_comparison.csv", index=False)
    scenario_df.to_csv(output_dir / "scenario_metrics.csv", index=False)
    all_scenario_df.to_csv(output_dir / "all_scenarios_metrics.csv", index=False)
    
    cm_df = pd.DataFrame(
        [
            {"Predicted": "No Action", "True No Action": cm_counts["true_negatives"], "True Intervention": cm_counts["false_negatives"]},
            {"Predicted": "Intervention", "True No Action": cm_counts["false_positives"], "True Intervention": cm_counts["true_positives"]},
        ],
        columns=["Predicted", "True No Action", "True Intervention"],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv", index=False)
    
    if not magnitude_df.empty:
        magnitude_df.to_csv(output_dir / "magnitude_bins.csv", index=False)
        
    # Flat metrics CSV
    flat_metrics = []
    for k, v in lstm_metrics.items():
        flat_metrics.append({"metric": k, "lstm_value": v, "baseline_value": baseline_metrics.get(k, np.nan)})
    pd.DataFrame(flat_metrics).to_csv(output_dir / "intervention_metrics.csv", index=False)
    
    # Markdown summary report
    summary_md = [
        "# Reviewer-Response Evaluation Report: Hydroponic Nutrient Regulation\n",
        f"**Evaluation Date (UTC)**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Model Checkpoint**: `{checkpoint_path.name}`\n",
        f"**Test Set Samples**: {n_samples:,} (No Action: {lstm_metrics['true_no_action_count']:,} [{lstm_metrics['true_no_action_fraction']*100:.2f}%], Non-Zero Intervention: {lstm_metrics['true_intervention_count']:,} [{lstm_metrics['true_intervention_fraction']*100:.2f}%])\n",
        f"**Action Space**: Canonical 61 Discrete Candidate Pairs (1 no-op `(0,0)` + 60 non-zero pairs)\n",
        "## Consolidated Benchmark Comparison\n",
        df_to_markdown(comp_df),
        "\n## Operating Regime Stratification\n",
        df_to_markdown(scenario_df),
        "\n",
    ]
    if not magnitude_df.empty:
        summary_md.extend([
            "## Dose Magnitude Bin Analysis (Supplementary)\n",
            df_to_markdown(magnitude_df),
            "\n",
        ])
        
    summary_md.append(interpretation_text)
    
    with open(output_dir / "reviewer_summary.md", "w") as f:
        f.write("\n".join(summary_md))
        
    print("\n" + "=" * 70)
    print("REVIEWER-RESPONSE BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(comp_df.to_string(index=False))
    print("\n" + "=" * 70)
    print("OPERATING REGIME STRATIFICATION")
    print("=" * 70)
    print(scenario_df.to_string(index=False))
    print("\n" + "=" * 70)
    print(interpretation_text)
    print("=" * 70)
    print(f"\nArtifacts saved successfully to: {output_dir}")
    
    return json_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reviewer-Response Intervention Evaluation Extension"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--output-dir", type=str, default="reports/intervention_evaluation"
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()
    
    config = load_config(ROOT / args.config)
    set_seed(config.get("seed", 42))
    
    out_dir = ROOT / args.output_dir
    ckpt_path = ROOT / args.checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_evaluation(config, out_dir, ckpt_path, device=device)


if __name__ == "__main__":
    main()
