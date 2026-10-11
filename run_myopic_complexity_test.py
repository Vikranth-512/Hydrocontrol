"""
Standalone entry point: Myopic 1-Step Greedy Controller Complexity Diagnostic.

Runs the temporal control complexity test that determines whether meaningful
performance in the algae tank benchmark depends on anticipating delayed
consequences.

Usage:
    python run_myopic_complexity_test.py
    python run_myopic_complexity_test.py --config configs/default.yaml
    python run_myopic_complexity_test.py --output-dir reports/myopic_complexity_test

This script:
  - Does NOT modify the existing benchmark pipeline.
  - Does NOT overwrite any existing result files.
  - Writes all outputs exclusively to reports/myopic_complexity_test/
    (or the directory specified via --output-dir).

Runtime estimate: ~120 episodes x 3500 steps x 71 candidates = ~14 ms/step
(on a 12-core machine) -> approximately 25-45 minutes total wall time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Myopic 1-Step Greedy Controller — Temporal Complexity Diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for diagnostic results "
            "(default: reports/myopic_complexity_test)"
        ),
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Override number of seeds per scenario (default: use config value, 30)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override horizon steps (default: use config value, 3500)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Import after arg parsing to avoid slow imports at --help time
    from utils import load_config
    from evaluation.myopic_controller import MyopicEvaluator

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(str(config_path))

    # Apply CLI overrides (do not mutate nested dicts if key absent)
    if args.output_dir is not None:
        config.setdefault("myopic_complexity_test", {})["report_dir"] = args.output_dir

    if args.n_seeds is not None:
        config.setdefault("evaluation", {}).setdefault(
            "multiseed_long_horizon", {}
        )["n_seeds"] = args.n_seeds

    if args.horizon is not None:
        config.setdefault("evaluation", {}).setdefault(
            "multiseed_long_horizon", {}
        )["horizon"] = args.horizon

    print("=" * 60)
    print("  MYOPIC 1-STEP COMPLEXITY DIAGNOSTIC")
    print("=" * 60)
    print(f"  Config:  {config_path}")
    ev_cfg = config.get("evaluation", {}).get("multiseed_long_horizon", {})
    n_seeds = ev_cfg.get("n_seeds", 30)
    horizon = ev_cfg.get("horizon", 3500)
    scenarios = ev_cfg.get(
        "scenarios", ["normal", "heatwave", "nutrient_depletion", "cold_shock"]
    )
    print(f"  Scenarios ({len(scenarios)}): {scenarios}")
    print(f"  Seeds per scenario: {n_seeds}")
    print(f"  Horizon: {horizon} steps")
    print(f"  Total episodes: {len(scenarios) * n_seeds}")
    print(
        f"\n  NOTE: This diagnostic saves results ONLY to "
        f"reports/myopic_complexity_test/."
    )
    print(
        f"  It does NOT modify any existing benchmark results or pipeline.\n"
    )

    evaluator = MyopicEvaluator(config)

    t0 = time.perf_counter()
    analysis = evaluator.run()
    elapsed = time.perf_counter() - t0

    print(f"\n  Wall time: {elapsed/60:.1f} min ({elapsed:.1f} s)")
    print(f"\nResults saved to: {evaluator.report_dir}")
    print(f"  episode_results.csv        — raw episode metrics")
    print(f"  controller_summary.csv     — per-scenario mean + 95% CI")
    print(f"  collapse_analysis.csv      — collapse probability by scenario")
    print(f"  myopic_complexity_report.md — comparison vs primary benchmark")
    print(f"  diagnostic_config.yaml     — parameter snapshot")
    print(f"  seed_schedule.json         — reproducibility provenance")

    # Quick summary to stdout
    df = analysis["df_episodes"]
    df_sum = analysis["df_summary"]
    print("\nQuick Summary (Mean across seeds):")
    print(f"  {'Scenario':<22} {'EC MAE':>9} {'Fin.Biomass':>12} {'TotDose':>10} {'Fin.Rho':>10} {'Terminated':>12}")
    print("  " + "-" * 78)
    for sc in evaluator.scenarios:
        r = df_sum[df_sum["scenario"] == sc]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        df_sc = df[df["scenario"] == sc]
        n_term = int(df_sc["episode_terminated"].sum())
        print(
            f"  {sc:<22}"
            f" {r['ec_mae_mean']:>9.4f}"
            f" {r['final_biomass_mean']:>12.2f}"
            f" {r['total_dose_mean']:>10.2f}"
            f" {r['final_reserve_ratio_mean']:>10.4f}"
            f" {n_term:>6}/{len(df_sc)}"
        )

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
