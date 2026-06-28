from pathlib import Path
from typing import Any, Dict, List

from simulation.validate.common import Experiment


def generate_report(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generates the final Validation_Report.md aggregating all experiment results."""
    report_path = output_dir / "Validation_Report.md"
    
    # Group results by category
    categories = {}
    for res in results:
        exp: Experiment = res["experiment"]
        if exp.category not in categories:
            categories[exp.category] = []
        categories[exp.category].append(res)
        
    total_exps = len(results)
    passed_exps = sum(1 for r in results if r["status"] == "PASS")
    pass_rate = (passed_exps / total_exps * 100) if total_exps > 0 else 0.0

    lines = [
        "# Simulation Validation Report",
        "",
        "This document contains the automated scientific validation results for the mechanistic algae tank simulator.",
        "",
        f"**Overall Status:** {passed_exps}/{total_exps} passed ({pass_rate:.1f}%)",
        "",
        "---",
    ]

    for cat_name, cat_results in categories.items():
        lines.extend([
            f"## {cat_name}",
            "---",
            ""
        ])
        
        for res in cat_results:
            exp: Experiment = res["experiment"]
            status_symbol = "[PASS]" if res["status"] == "PASS" else "[FAIL]"
            lines.append(f"### {status_symbol} {exp.name} ({exp.id})")
            lines.append(f"**Hypothesis:** {exp.hypothesis}")
            lines.append(f"**Status:** {res['status']}")
            lines.append("")
            
            if res.get("metrics"):
                lines.append("**Metrics:**")
                for m_key, m_val in res["metrics"].items():
                    # format float if needed
                    if isinstance(m_val, float):
                        lines.append(f"- {m_key}: {m_val:.4g}")
                    else:
                        lines.append(f"- {m_key}: {m_val}")
                lines.append("")
            
            if res.get("warnings"):
                lines.append("**Warnings:**")
                for w in res["warnings"]:
                    lines.append(f"- WARNING: {w}")
                lines.append("")
                
            if exp.plots:
                lines.append("**Plots Generated:**")
                for p in exp.plots:
                    lines.append(f"- `{p}`")
                lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Validation report successfully written to {report_path}")
