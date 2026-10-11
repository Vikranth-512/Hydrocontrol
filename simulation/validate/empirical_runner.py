"""
Empirical Biological Plausibility Validation Runner

Runs the seven literature-grounded biological plausibility tests and generates
comprehensive reports. This is a separate validation stage from the existing
34 automated validation tests, which focus on mathematical/mechanical verification.

The empirical validation tests assess whether the simulator reproduces biological
response structures commonly reported in nutrient-regulated microalgal cultivation
and photobioreactor systems.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from simulation.dynamics import TankDynamicsParams
from simulation.validate.common import Experiment
from simulation.validate.experiments import empirical_biological


def get_empirical_experiments() -> List[Experiment]:
    """Registry of empirical biological validation experiments."""
    return empirical_biological.REGISTERED_EXPERIMENTS


def run_empirical_validation(
    output_dir: Path,
    params: TankDynamicsParams | None = None,
) -> Dict[str, Any]:
    """
    Run all seven empirical biological validation experiments.
    
    Args:
        output_dir: Base directory for validation outputs
        params: Dynamics parameters (uses defaults if None)
    
    Returns:
        Dictionary containing all experiment results and summary metrics
    """
    t_start = time.time()
    print("=" * 72)
    print("  EMPIRICAL BIOLOGICAL PLAUSIBILITY VALIDATION")
    print("  Literature-Grounded Biological Response Structure Assessment")
    print("=" * 72)
    
    # Create output directory structure
    base_out_dir = Path(output_dir) / "empirical_biological"
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = base_out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    figures_dir = base_out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = base_out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    report_dir = base_out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Load empirical reference metadata
    reference_metadata_path = Path("validation/empirical/references/empirical_reference_metadata.yaml")
    reference_metadata = {}
    if reference_metadata_path.exists():
        with open(reference_metadata_path, "r") as f:
            reference_metadata = yaml.safe_load(f)
    
    # Use default parameters if not provided
    if params is None:
        params = TankDynamicsParams()
    
    # Run all empirical experiments
    experiments = get_empirical_experiments()
    results: List[Dict[str, Any]] = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i:02d}/{len(experiments):02d}] [{exp.id}] {exp.name}")
        print(f"       Category: {exp.category}")
        hyp_safe = exp.hypothesis[:80].encode("ascii", errors="replace").decode("ascii")
        print(f"       Hypothesis: {hyp_safe}...")
        
        exp_dir = raw_dir / exp.id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            t0 = time.time()
            exp_result = exp.execute(exp_dir, params)
            elapsed = time.time() - t0
            
            exp_result["experiment"] = exp
            exp_result.setdefault("warnings", [])
            
            # Move figures to figures directory
            for plot in exp.plots:
                src = exp_dir / plot
                if src.exists():
                    dst = figures_dir / plot
                    if dst.exists():
                        dst.unlink()
                    src.rename(dst)
            
            results.append(exp_result)
            
            assessment = exp_result.get("assessment", "unknown")
            print(f"       [COMPLETE] Assessment: {assessment.upper()}  ({elapsed:.1f}s)")
            
            # Print primary evidence
            primary_evidence = exp_result.get("primary_evidence", "")
            if primary_evidence:
                print(f"         Primary evidence: {primary_evidence}")
            
            for w in exp_result.get("warnings", []):
                print(f"         WARNING: {w}")
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"       [ERROR] CRITICAL: {e}")
            results.append({
                "id": exp.id,
                "name": exp.name,
                "experiment": exp,
                "assessment": "test_design_requires_revision",
                "primary_evidence": f"Exception: {e}",
                "secondary_evidence": "",
                "limitations": "Test execution failed",
                "warnings": [f"Exception: {e}"],
            })
    
    # Generate summary table
    summary_rows = []
    for res in results:
        summary_rows.append({
            "Test": res.get("id", "unknown"),
            "Biological phenomenon": res.get("name", "unknown"),
            "Assessment": res.get("assessment", "unknown"),
            "Primary evidence": res.get("observed_behavior", ""),
        })
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(results_dir / "empirical_validation_summary.csv", index=False)
    
    # Generate machine-readable JSON summary with standardized schema
    json_summary = {
        "validation_type": "Empirical Biological Plausibility Validation",
        "validation_purpose": "Literature-grounded biological response structure assessment",
        "total_experiments": len(results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": []
    }
    
    for res in results:
        exp: Experiment = res["experiment"]
        # Clean metrics for JSON serialization
        primary_metrics = res.get("primary_metrics", {})
        secondary_metrics = res.get("secondary_metrics", {})
        
        # Convert any non-JSON-serializable values
        def clean_dict(d):
            if not isinstance(d, dict):
                return d
            cleaned = {}
            for k, v in d.items():
                if isinstance(v, (bool, int, float, str, type(None))):
                    cleaned[k] = v
                elif isinstance(v, list):
                    cleaned[k] = [clean_dict(item) if isinstance(item, dict) else item for item in v]
                else:
                    # Try to convert to string
                    cleaned[k] = str(v)
            return cleaned
        
        primary_metrics = clean_dict(primary_metrics)
        secondary_metrics = clean_dict(secondary_metrics)
        
        json_summary["experiments"].append({
            "id": res.get("id", exp.id),
            "name": res.get("name", exp.name),
            "validation_type": res.get("validation_type", "Empirical Biological Plausibility Validation"),
            "validation_purpose": res.get("validation_purpose", "Literature-grounded biological response structure assessment"),
            "assessment": res.get("assessment", "unknown"),
            "expected_behavior": res.get("expected_behavior", exp.hypothesis),
            "observed_behavior": res.get("observed_behavior", ""),
            "primary_metrics": primary_metrics,
            "secondary_metrics": secondary_metrics,
            "thresholds": res.get("thresholds", {}),
            "experimental_conditions": res.get("experimental_conditions", {}),
            "limitations": res.get("limitations", []),
            "assessment_rationale": res.get("assessment_rationale", ""),
        })
    
    import json
    with open(results_dir / "empirical_validation_summary.json", "w") as f:
        json.dump(json_summary, f, indent=2)
    
    # Generate machine-readable validation summary
    validation_summary = {
        "validation_suite": "Empirical Biological Plausibility",
        "experiments": {}
    }
    
    for res in results:
        exp_id = res.get("id", "unknown")
        validation_summary["experiments"][exp_id] = {
            "name": res.get("name", "unknown"),
            "assessment": res.get("assessment", "unknown"),
            "primary_evidence": res.get("primary_evidence", ""),
            "secondary_evidence": res.get("secondary_evidence", ""),
            "limitations": res.get("limitations", []),
        }
    
    with open(base_out_dir / "validation_summary.json", "w") as f:
        json.dump(validation_summary, f, indent=2)
    
    # Generate human-readable report
    generate_empirical_report(results, reference_metadata, report_dir)
    
    # Generate README with experimental definitions
    generate_readme(results, base_out_dir)
    
    elapsed_total = time.time() - t_start
    
    print("\n" + "=" * 72)
    print(f"  EMPIRICAL BIOLOGICAL PLAUSIBILITY VALIDATION COMPLETE")
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"  Results: {results_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Report: {report_dir / 'empirical_biological_plausibility_report.md'}")
    print("=" * 72)
    
    return {
        "status": "COMPLETE",
        "total_experiments": len(results),
        "elapsed_time": elapsed_total,
        "results_dir": str(results_dir),
        "figures_dir": str(figures_dir),
        "report_dir": str(report_dir),
        "summary": df_summary.to_dict(),
    }


def generate_empirical_report(
    results: List[Dict[str, Any]],
    reference_metadata: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate human-readable empirical validation report."""
    report_path = output_dir / "empirical_biological_plausibility_report.md"
    
    lines = [
        "# Empirical Biological Plausibility Validation Report",
        "",
        "## Overview",
        "",
        "The existing automated validation suite establishes mathematical and mechanistic "
        "consistency of the simulator. To provide an external biological check, seven "
        "literature-grounded behavioral experiments were added. These experiments assess whether "
        "the simulator reproduces biological response structures commonly reported in "
        "nutrient-regulated microalgal cultivation and photobioreactor systems.",
        "",
        "## Validation Framework",
        "",
        "**Important distinctions:**",
        "",
        "- **Mechanistic verification**: The 34 automated validation tests confirm that the "
          "simulator replicates behaviors defined by its equations (mathematical/mechanical consistency).",
        "- **Empirical biological plausibility**: These seven experiments assess whether the "
          "simulator's response structures are consistent with established biological behavior "
          "reported in the literature.",
        "",
        "This validation is **NOT**:",
        "",
        "- Parameter identification or species-specific calibration",
        "- Experimental calibration of the benchmark dynamics",
        "- Proof of biological authenticity",
        "- Digital-twin validation",
        "",
        "The simulator is a synthetic benchmark and is not intended to represent one specific "
        "algal species, crop cultivar, or commercial photobioreactor. The validation assesses "
        "response-structure plausibility, not species-specific predictive accuracy.",
        "",
        "## Assessment Categories",
        "",
        "**Consistent:**",
        "The simulated response exhibits the predefined qualitative behavior being assessed.",
        "",
        "**Partially consistent:**",
        "The principal response direction or mechanism is present, but one or more secondary "
        "characteristics differ from the expected behavior or cannot be fully assessed.",
        "",
        "**Inconsistent:**",
        "The predefined biological response is not reproduced under the stated experimental conditions.",
        "",
        "**Test-design-requires-revision:**",
        "The experiment does not provide a valid test of the hypothesis because the measured "
        "quantity, baseline, controllability, or detection criterion is inappropriate.",
        "",
        "## Experimental Results",
        "",
    ]
    
    for res in results:
        exp_id = res.get("id", "unknown")
        exp_name = res.get("name", "unknown")
        assessment = res.get("assessment", "unknown")
        
        lines.extend([
            f"### {exp_id}: {exp_name}",
            "",
            f"**Hypothesis:** {res.get('expected_behavior', 'N/A')}",
            "",
            f"**Assessment:** {assessment.upper()}",
            "",
        ])
        
        lines.append("**Primary evidence:**")
        lines.append(res.get("primary_evidence", "None"))
        lines.append("")
        
        if res.get("secondary_evidence"):
            lines.append("**Secondary evidence:**")
            lines.append(res.get("secondary_evidence", ""))
            lines.append("")
        
        if res.get("limitations"):
            lines.append("**Limitations:**")
            for lim in res.get("limitations", []):
                lines.append(f"- {lim}")
            lines.append("")
        
        if res.get("warnings"):
            lines.append("**Warnings:**")
            for w in res["warnings"]:
                lines.append(f"- WARNING: {w}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Add summary table
    lines.extend([
        "## Cross-Test Summary",
        "",
        "| Test | Biological Phenomenon | Assessment | Primary Evidence |",
        "|------|---------------------|------------|------------------|",
    ])
    
    for res in results:
        exp_id = res.get("id", "unknown")
        exp_name = res.get("name", "unknown")
        assessment = res.get("assessment", "unknown")
        primary_evidence = res.get("primary_evidence", "")[:60]  # Truncate for table
        lines.append(f"| {exp_id} | {exp_name} | {assessment} | {primary_evidence} |")
    
    lines.extend([
        "",
        "## Limitations",
        "",
        "- The simulator is a generic benchmark, not a species-specific model.",
        "- Comparisons are at the response-structure level, not species-specific parameter values.",
        "- Absolute time scales and biomass magnitudes may differ from specific experimental systems.",
        "- The validation does not claim universal biological correctness across all species.",
        "- Biological processes are represented at an abstraction level appropriate for a benchmark.",
        "- Differences in experimental scale and timescale limit direct quantitative comparison.",
        "- Absence of direct calibration against a single experimental dataset.",
        "",
        "## Conclusion",
        "",
        "The empirical biological plausibility suite evaluates whether selected simulator response "
        "structures are qualitatively consistent with behaviors reported in the biological literature. "
        "The experiments assess directional responses, response shapes, and temporal pathways rather than "
        "species-specific predictive accuracy or experimental calibration. Partial consistency is retained "
        "where the modeled response differs from the simplified hypothesis or where the experiment cannot "
        "fully resolve the pathway. These results complement the simulator's mechanistic and numerical "
        "verification tests rather than constituting direct experimental validation.",
        "",
        "## References",
        "",
        "Empirical references are documented in `validation/empirical/references/empirical_reference_metadata.yaml`.",
        "",
    ])
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Empirical validation report written to {report_path}")


def generate_readme(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generate README documenting experimental definitions and metric formulas."""
    readme_path = output_dir / "README.md"
    
    lines = [
        "# Empirical Biological Plausibility Validation",
        "",
        "This directory contains the empirical biological plausibility validation suite "
        "for the v4 hydroponic nutrient-regulation benchmark simulator.",
        "",
        "## Purpose",
        "",
        "The empirical biological plausibility suite evaluates whether simulated response "
        "structures are consistent with established experimental observations from "
        "microalgal cultivation, photobioreactor systems, and nutrient-regulated cultivation.",
        "",
        "## Experimental Definitions",
        "",
    ]
    
    for res in results:
        exp_id = res.get("id", "unknown")
        exp_name = res.get("name", "unknown")
        assessment = res.get("assessment", "unknown")
        
        lines.extend([
            f"### {exp_id}: {exp_name}",
            "",
            f"**Hypothesis:** {res.get('expected_behavior', 'N/A')}",
            "",
            f"**Assessment:** {assessment}",
            "",
            f"**Primary evidence:** {res.get('primary_evidence', 'None')}",
            "",
            f"**Limitations:**",
        ])
        
        for lim in res.get("limitations", []):
            lines.append(f"- {lim}")
        
        lines.extend([
            "",
            "---",
            "",
        ])
    
    lines.extend([
        "## Metric Formulas",
        "",
        "### Monotonicity Score",
        "```python",
        "monotonicity_score = number of non-decreasing adjacent response intervals / total number of adjacent intervals",
        "```",
        "",
        "### Directional Monotonicity Score",
        "```python",
        "# For expected decreasing behavior:",
        "directional_monotonicity_score = number of intervals satisfying growth[i+1] <= growth[i] / total intervals",
        "# For expected increasing behavior, reverse the inequality",
        "```",
        "",
        "### Incremental Response Ratio (Saturation Detection)",
        "```python",
        "incremental_response_ratio = (high-region slope magnitude) / (low-region slope magnitude)",
        "# Low region: quantile < 0.5",
        "# High region: quantile >= 0.5",
        "```",
        "",
        "### Spearman Rank Correlation",
        "Measures monotonic relationship between variables (non-parametric).",
        "",
        "### Pearson Correlation",
        "Measures linear relationship between variables (parametric).",
        "",
        "## Reproducibility",
        "",
        "All experiments use:",
        "- Fixed random seed (42)",
        "- Zero stochastic disturbances",
        "- Identical initial states between conditions",
        "- Controlled one-factor-at-a-time perturbations",
        "- Explicit documentation of all experimental parameters",
        "",
        "## Output Structure",
        "",
        "```",
        "empirical_biological/",
        "├── raw/                    # Raw simulator outputs per experiment",
        "│   ├── EBV-01/",
        "│   ├── EBV-02/",
        "│   └── ...",
        "├── figures/                # Publication-quality figures",
        "│   ├── EBV_01_uptake_response.png",
        "│   ├── EBV_02_growth_response.png",
        "│   └── ...",
        "├── results/                # Metrics and summaries",
        "│   ├── empirical_validation_summary.csv",
        "│   ├── empirical_validation_summary.json",
        "│   └── EBV_XX_metrics.json",
        "├── report/                 # Human-readable reports",
        "│   └── empirical_biological_plausibility_report.md",
        "├── validation_summary.json  # Machine-readable summary",
        "└── README.md               # This file",
        "```",
        "",
        "## Running the Validation",
        "",
        "```bash",
        "python main.py --stage empirical_biological_validation",
        "```",
        "",
        "## Interpretation",
        "",
        "- **Consistent**: The simulator reproduces the expected qualitative response pattern.",
        "- **Partially consistent**: The simulator shows some aspects of the expected pattern but with deviations.",
        "- **Inconsistent**: The simulator does not reproduce the expected response pattern.",
        "- **Test-design-requires-revision**: The experiment design is inappropriate for testing the hypothesis.",
        "",
        "These assessments are based on literature-established qualitative behavior patterns. "
        "Where quantitative empirical data are available and directly comparable, additional "
        "statistical metrics are reported.",
        "",
        "## Important Notes",
        "",
        "- This is NOT species-specific experimental calibration.",
        "- No simulator parameters were modified to improve results.",
        "- No literature values were fabricated.",
        "- The suite does NOT produce an aggregate 'biological accuracy score'.",
        "- The assessment concerns consistency with predefined literature-grounded response structure.",
        "",
    ])
    
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"README written to {readme_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation_report")
    run_empirical_validation(output_dir)
