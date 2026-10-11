"""
Standalone execution entry point for Supervisory Policy Sensitivity Analysis (ORC-04).

Usage:
    python run_supervisory_sensitivity.py
    python run_supervisory_sensitivity.py --n-states 500 --seed 42 --output-dir reports/supervisory_policy_sensitivity
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on PYTHONPATH
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.validate.experiments.supervisory_sensitivity import (
    run_supervisory_policy_sensitivity_analysis,
)
from utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Run Supervisory Policy Sensitivity Analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to pipeline configuration YAML file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/all_trajectories_labeled.csv",
        help="Path to labeled benchmark dataset CSV",
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=500,
        help="Number of representative states to sample (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed random seed for representative state sampling (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/supervisory_policy_sensitivity",
        help="Output directory for CSVs, plots, and summary report",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: autodetect)",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path) if config_path.exists() else {}

    print("=" * 60)
    print("  SUPERVISORY POLICY SENSITIVITY ANALYSIS (ORC-04)")
    print("=" * 60)
    print(f"Config:     {args.config}")
    print(f"Dataset:    {args.dataset}")
    print(f"States:     {args.n_states}")
    print(f"Seed:       {args.seed}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    result = run_supervisory_policy_sensitivity_analysis(
        config=config,
        dataset_path=Path(args.dataset),
        n_states=args.n_states,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        num_workers=args.workers,
    )

    print(f"\nExecution finished with status: {result['status']}")
    print("Generated Artifacts:")
    for p in result.get("plots", []):
        print(f"  - {p}")
    print(f"  - {Path(args.output_dir) / 'objective_weight_sensitivity.csv'}")
    print(f"  - {Path(args.output_dir) / 'horizon_sensitivity.csv'}")
    print(f"  - {Path(args.output_dir) / 'summary.json'}")
    print(f"  - {Path(args.output_dir) / 'supervisory_policy_sensitivity_report.txt'}")


if __name__ == "__main__":
    main()
