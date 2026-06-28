"""
Validation framework runner.

Discovers, executes, and reports on all registered experiments
across the 8 scientific validation categories.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from simulation.dynamics import TankDynamicsParams
from simulation.validate.common import Experiment
from simulation.validate.report import generate_report

# Import all experiment modules
from simulation.validate.experiments import physical
from simulation.validate.experiments import biology
from simulation.validate.experiments import numerical
from simulation.validate.experiments import emergent
from simulation.validate.experiments import sensors
from simulation.validate.experiments import benchmark
from simulation.validate.experiments import parameter
from simulation.validate.experiments import control


def get_all_experiments() -> List[Experiment]:
    """Registry of all validation experiments across 8 categories."""
    experiments: List[Experiment] = []

    # I. Physical Conservation
    experiments.extend(physical.REGISTERED_EXPERIMENTS)

    # II. Biological Dynamics
    experiments.extend(biology.REGISTERED_EXPERIMENTS)

    # III. Numerical Analysis
    experiments.extend(numerical.REGISTERED_EXPERIMENTS)

    # IV. Emergent Behavior
    experiments.extend(emergent.REGISTERED_EXPERIMENTS)

    # V. Sensor Validation
    experiments.extend(sensors.REGISTERED_EXPERIMENTS)

    # VI. RL Benchmark Characterization
    experiments.extend(benchmark.REGISTERED_EXPERIMENTS)

    # VII. Parameter Identifiability
    experiments.extend(parameter.REGISTERED_EXPERIMENTS)

    # VIII. Controller Independence
    experiments.extend(control.REGISTERED_EXPERIMENTS)

    return experiments


def main():
    t_start = time.time()
    print("=" * 72)
    print("  SCIENTIFIC VALIDATION FRAMEWORK")
    print("  Mechanistic Algae Tank Simulator - Publication Audit")
    print("=" * 72)

    base_out_dir = Path("validation_report")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    params = TankDynamicsParams()

    experiments = get_all_experiments()
    results: List[Dict[str, Any]] = []

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i:02d}/{len(experiments):02d}] [{exp.id}] {exp.name}")
        print(f"       Category: {exp.category}")
        hyp_safe = exp.hypothesis[:80].encode("ascii", errors="replace").decode("ascii")
        print(f"       Hypothesis: {hyp_safe}...")

        exp_dir = base_out_dir / exp.id
        exp_dir.mkdir(parents=True, exist_ok=True)

        try:
            t0 = time.time()
            exp_result = exp.execute(exp_dir, params)
            elapsed = time.time() - t0

            exp_result["experiment"] = exp
            if "status" not in exp_result:
                exp_result["status"] = "FAIL"
            exp_result.setdefault("warnings", [])
            exp_result.setdefault("metrics", {})

            results.append(exp_result)

            status_icon = "[PASS]" if exp_result["status"] == "PASS" else "[FAIL]"
            print(f"       {status_icon} {exp_result['status']}  ({elapsed:.1f}s)")

            # Print key metrics
            for k, v in list(exp_result["metrics"].items())[:4]:
                if isinstance(v, float):
                    print(f"         {k}: {v:.6g}")
                else:
                    print(f"         {k}: {v}")

            for w in exp_result.get("warnings", []):
                print(f"         WARNING: {w}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"       [ERROR] CRITICAL: {e}")
            results.append({
                "experiment": exp,
                "status": "ERROR",
                "metrics": {},
                "warnings": [f"Exception: {e}"],
            })

    # Generate Summary Report
    summary_dir = base_out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    generate_report(results, summary_dir)

    elapsed_total = time.time() - t_start
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] in ("FAIL", "ERROR"))
    total = len(results)

    print("\n" + "=" * 72)
    print(f"  VALIDATION COMPLETE  -  {passed}/{total} PASSED, {failed} FAILED")
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"  Report: {summary_dir / 'Validation_Report.md'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
