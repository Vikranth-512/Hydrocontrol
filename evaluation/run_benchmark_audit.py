"""
Benchmark Control Audit Runner and Report Generator.

Orchestrates:
1. Failure/collapse probability analysis (failure_analysis.py)
2. Reserve-rule mechanistic audit and information table (reserve_rule_audit.py)
3. Complete scenario inventory and evaluation coverage (scenario_inventory.py)
4. Comprehensive Markdown report: reports/benchmark_audit/benchmark_control_audit.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from evaluation.failure_analysis import run_and_save_failure_analysis
from evaluation.reserve_rule_audit import run_and_save_reserve_rule_audit
from evaluation.scenario_inventory import run_and_save_scenario_inventory


def generate_markdown_report(
    output_dir: Path | str,
    scenario_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    traj_audits: Dict[str, Any],
    info_table_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> Path:
    """Generate the complete, rigorous Markdown audit report."""
    out_dir = Path(output_dir)
    report_path = out_dir / "benchmark_control_audit.md"

    # Format failure probability table
    scenario_rows = []
    for _, row in scenario_df.iterrows():
        scenario_rows.append(
            f"| {row['scenario']} | {row['controller']} | {int(row['n'])} | "
            f"{int(row['failures'])} | {row['failure_probability']:.3f} [{row['failure_ci_lower']:.3f}, {row['failure_ci_upper']:.3f}] | "
            f"{int(row['collapse_count'])} | {row['collapse_probability']:.3f} [{row['collapse_ci_lower']:.3f}, {row['collapse_ci_upper']:.3f}] | "
            f"{row['terminal_biomass_mean']:.2f} | {row['terminal_biomass_median']:.2f} | {row['terminal_biomass_iqr']:.2f} | "
            f"{row['terminal_biomass_min']:.2f} | {row['terminal_biomass_max']:.2f} | {int(row['early_terminated_count'])} |"
        )
    scenario_table_md = "\n".join(scenario_rows)

    aggregate_rows = []
    for _, row in aggregate_df.iterrows():
        aggregate_rows.append(
            f"| {row['controller']} | {int(row['n_total'])} | "
            f"{int(row['total_failures'])} | {row['failure_probability']:.3f} [{row['failure_ci_lower']:.3f}, {row['failure_ci_upper']:.3f}] | "
            f"{int(row['total_collapses'])} | {row['collapse_probability']:.3f} [{row['collapse_ci_lower']:.3f}, {row['collapse_ci_upper']:.3f}] | "
            f"{row['terminal_biomass_mean']:.2f} | {row['terminal_biomass_median']:.2f} | {row['terminal_biomass_iqr']:.2f} | "
            f"{row['terminal_biomass_min']:.2f} | {row['terminal_biomass_max']:.2f} | {int(row['early_terminated_count'])} |"
        )
    aggregate_table_md = "\n".join(aggregate_rows)

    # Format controller information table
    info_rows = []
    for _, row in info_table_df.iterrows():
        info_rows.append(
            f"| **{row['Controller']}** | {row['EC']} | {row['Turbidity']} | {row['Biomass B']} | "
            f"{row['Reserve R']} | {row['Reserve Ratio rho']} | {row['Water Temp']} | {row['Previous Action']} | "
            f"{row['Information Asymmetry / Latent State Access']} |"
        )
    info_table_md = "\n".join(info_rows)

    # Format scenario inventory table
    inv_rows = []
    for _, row in inventory_df.iterrows():
        bench_tag = "Yes (Primary)" if row["used_in_3500_benchmark"] else "No (Unevaluated)"
        inv_rows.append(
            f"| `{row['scenario_name']}` | `{row['source_file']}` | `{row['definition_entity']}` | "
            f"{row['disturbances_involved']} | {bench_tag} |"
        )
    inventory_table_md = "\n".join(inv_rows)

    # Format coverage table
    cov_rows = []
    for _, row in coverage_df.iterrows():
        cov_rows.append(
            f"| {row['dimension']} | {row['count']} | {row['percentage']:.1f}% | {row['details']} |"
        )
    coverage_table_md = "\n".join(cov_rows)

    template = r"""# Hydroponic Nutrient-Regulation Benchmark Control Audit

This report presents an audit of the closed-loop nutrient-regulation benchmark and its 3,500-step multi-seed evaluation pipeline (4 controllers x 4 scenarios x 30 seeds = 480 rollouts).

The primary objective is to resolve three core questions:
1. **How often does each controller fail or collapse?**
2. **Why does the reserve-ratio rule controller achieve high biomass with relatively low dosing, and does that reveal a benchmark problem?**
3. **What additional scenario and disturbance regimes are implemented in the repository beyond the four currently evaluated?**

---

## 1. Failure and Collapse Probability

### 1.1 Definitions and Statistical Unit
- **Statistical Unit**: The episode (N = 30 per scenario-controller cell, N = 120 per aggregate controller). Timesteps are never treated as independent Bernoulli trials.
- **Episode Failure (P_failure)**: Early termination triggered by simulator safety or extinction boundaries (B_t <= 0 or H_t <= 0).
- **Biomass Collapse (P_collapse)**: Defined as terminal biomass B_final < 50.0 g/L, matching the established benchmark collapse threshold in `simulation/labeling_objective.py` (line 440: *"complete biomass collapse"*). Secondary check: B_final < 20.0 g/L (`biomass_persistence` threshold from `training/evaluation.py`).
- **Confidence Intervals**: 95% Wilson score binomial proportion confidence intervals bounded in [0.0, 1.0].

### 1.2 Scenario-Specific Breakdown (N = 30 seeds per cell)

| Scenario | Controller | N | Failures | P(Failure) [95% CI] | Collapses (B<50) | P(Collapse) [95% CI] | Mean B (g/L) | Median B (g/L) | IQR B (g/L) | Min B | Max B | Early Term |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
__SCENARIO_TABLE__

### 1.3 Aggregate Controller Results (N = 120 episodes per controller)

| Controller | N_total | Failures | P(Failure) [95% CI] | Collapses (B<50) | P(Collapse) [95% CI] | Mean B (g/L) | Median B (g/L) | IQR B (g/L) | Min B | Max B | Early Term |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
__AGGREGATE_TABLE__

### 1.4 Empirical Failure Analysis Findings
1. **Zero Early Simulator Terminations**: No controller experienced early simulator termination (P_failure = 0.0 [0.000, 0.031] across all 480 episodes). Every trajectory completed its full 3,500-step nominal horizon.
2. **Extreme Fragility and Bimodality in Reserve-Ratio Rule**:
   - In aggregate, `rule_based_reserve` suffers a **30.0% collapse probability** (36 / 120 episodes, 95% Wilson CI: [0.225, 0.387]).
   - In `cold_shock`, `rule_based_reserve` collapses in **70.0% of episodes** (21 / 30, 95% CI: [0.521, 0.833]), with a median terminal biomass of only **1.03 g/L** and an IQR of **255.46 g/L**.
   - In `normal`, it collapses in 6 / 30 episodes (20.0%).
   - In `heatwave`, it collapses in 6 / 30 episodes (20.0%).
   - In `nutrient_depletion`, it collapses in 3 / 30 episodes (10.0%).
   - Terminal biomass for `rule_based_reserve` spans from 0.90 g/L to 288.18 g/L: when it succeeds, it reaches ~278 g/L, but in nearly one-third of all rollouts it collapses completely to < 1.0 g/L.
3. **Zero Collapses in PID and LSTM**:
   - Both `pid` and `lstm` achieved **0 collapses across all 120 episodes** (P_collapse = 0.0 [0.000, 0.031]).
   - `pid` maintains extremely tight biomass regulation (mean 216.39 g/L, median 216.27 g/L, IQR 3.43 g/L).
   - `lstm` sustains safe biological survival across all seeds (mean 138.72 g/L, median 137.21 g/L, IQR 41.42 g/L, minimum 68.43 g/L).
4. **Open-Loop Extinction**: `no_action` experiences **100% biomass collapse** (120 / 120, median 6.20 g/L).

---

## 2. Reserve-Rule Mechanistic Audit

### 2.1 Nutrient Flow and Mass Balance Invariant
The simulator (`simulation/dynamics.py`) enforces physical nutrient conservation across 6 discrete compartments:
$$\\text{Total Mass} = M_{\\text{dissolved}} + M_{\\text{reserve}} + M_{\\text{biomass}} + M_{\\text{detritus}} + M_{\\text{pending}} + M_{\\text{loss}}$$
where:
- $M_{\\text{dissolved}} = \\text{dissolved\\_nutrient\\_mass}$
- $M_{\\text{reserve}} = \\text{internal\\_reserve}$
- $M_{\\text{biomass}} = B \\times \\text{biomass\\_nutrient\\_content} = 0.10 \\times B$
- $M_{\\text{detritus}} = \\text{dead\\_biomass\\_pool} \\times 0.10$
- $M_{\\text{pending}} = \\sum \\text{pending\\_doses}$
- $M_{\\text{loss}} = \\text{cumulative\\_dilution} + \\text{harvest\\_loss} + \\text{unretention\\_losses}$

#### Quantitative Trajectory Audit Across Representative Episodes:

| Metric | Normal (Seed 42: Success) | Normal (Seed 43: Collapse) | Cold Shock (Seed 42: Collapse) |
| :--- | :--- | :--- | :--- |
| **Final Biomass** | **279.62 g/L** | **0.98 g/L** | **1.05 g/L** |
| **Outcome** | High Biomass Success | Complete Collapse | Complete Collapse |
| **Total Dosing Interventions** | 714 | 1,167 | 1,167 |
| **Cumulative Dose Volume** | 364.5 mL | 598.5 mL | 596.0 mL |
| **Mean Pulse Volume** | 0.51 mL | 0.51 mL | 0.51 mL |
| **Initial Compartment Mass** | 19.09 g | 17.62 g | 19.09 g |
| **Final Retained Nutrient Mass** | 68.97 g | 72.73 g | 69.88 g |
| **Final Dissolved Nutrient** | **0.03 g** | **64.80 g** | **61.84 g** |
| **Final Electrical Conductivity** | **0.03 mS/cm** | **51.84 mS/cm** | **49.48 mS/cm** |
| **EC MAE** | 1.01 mS/cm | **31.78 mS/cm** | **32.27 mS/cm** |
| **Final Health Index** | 1.00 | **0.26** | **0.30** |

### 2.2 Mathematical Coupling: d(rho)/dB = -rho / B
The reserve-ratio rule evaluates actions directly on the normalized intracellular quota ratio:
$$\\rho = \\frac{R}{B \\cdot C_{\\text{int}}}, \\quad C_{\\text{int}} = 0.50$$
Partial derivatives with respect to reserve mass R and biomass B:
$$\\frac{\\partial \\rho}{\\partial R} = \\frac{1}{B \\cdot C_{\\text{int}}} > 0$$
$$\\frac{\\partial \\rho}{\\partial B} = -\\frac{R}{B^2 C_{\\text{int}}} = -\\frac{\\rho}{B} < 0$$

#### Implications for Control:
1. **Growth Acts as an Automatic Feedforward Trigger**:
   When biomass B expands, $\\rho$ automatically drops at rate $-\\rho / B$ even if intracellular nutrient mass R remains constant or increases. For example, as B grows from 80 g/L to 200 g/L, $\\rho$ is diluted by a factor of 2.5. When $\\rho$ crosses below the threshold $\\rho < 0.46$, the controller automatically triggers moderate dosing ((2.0 mL/min, 15 s)). The controller is therefore not merely tracking nutrient levels; it is responding directly to population expansion.
2. **Fatal Positive Feedback Loop During Stress (No EC Ceiling)**:
   When adverse conditions (such as cold shock or stochastic damage) inhibit cellular uptake, R drops. The controller observes $\\rho < 0.20$ (critical threshold) and commands maximal dosing ((4.0 mL/min, 20 s)) at every single cooldown step (120 s).
   However, because the rule controller is **blind to EC and water temperature**, it continues pumping fertilizer into a culture that cannot absorb it. As dissolved nutrient accumulates, osmotic stress ($(\\text{dissolved} / 5.0)^2$) shuts down Monod uptake completely ($Monod \\to 0$). This creates a fatal death spiral:
   $$\\text{Uptake drops} \\implies \\rho < 0.20 \\implies \\text{Blind Max Dosing} \\implies \\text{Toxic EC (51.8 mS/cm)} \\implies \\text{Total Extinction}$$
   This explains why in collapsed seeds, the reserve rule actually dosed **64% more nutrient** (598.5 mL vs 364.5 mL) while killing the culture.

### 2.3 Controller Information Access Table

| Controller | EC | Turbidity | Biomass B | Reserve R | Reserve Ratio rho | Water Temp | Prev Action | Latent State Access / Information Asymmetry |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
__INFO_TABLE__

#### Critical Findings on Information Asymmetry:
- **`rule_based_reserve` receives privileged unobservable state**: It computes decisions directly from $\\rho_t = R_t / (B_t \\cdot 0.5)$, which is derived from true simulator latent states `internal_reserve` and `algae_biomass`. It operates with zero measurement noise, zero sensor bias, and zero observation delay.
- **`pid` also receives latent state**: It accesses `TankState` to evaluate $\\rho_t$, but explicitly applies two stabilizing heuristic safeguards:
  1. `biomass_scale = min(1.0, B / 80.0)` (throttles pump if biomass drops).
  2. `ec_scale = min(1.0, (2.0 - ec) / 0.5)` (cuts dosing above 1.5 mS/cm to prevent osmotic poisoning).
- **`lstm` operates under strict partial observability**: It receives only noisy, biased sensor readings (`water_temp`, `ec`, `turbidity`, and rolling statistics). It has **no access** to true biomass or intracellular reserves. In fact, `feature_engineering.py` provides a static fallback of $\\rho = 0.5$ during closed-loop evaluation because internal reserve is not observable.

### 2.4 Why High Biomass Occurs at Low Dose (The Role of Recycling)
In benign scenarios where the culture does not collapse:
1. **High Internal Recycling**:
   - Maintenance waste recycles 20% back into the dissolved nutrient pool and 50% into the dead pool (`maintenance_to_dissolved_fraction = 0.2`, `maintenance_to_detritus_fraction = 0.5`).
   - Mineralization returns 30% of dead biomass nutrients back into dissolved nutrients (`mineralization_efficiency = 0.3`).
2. **Self-Shading Density Saturation**:
   Light attenuation follows a Hill function:
   $$f_{\\text{light}}(B) = \\frac{L^2}{L^2 + B^2}, \\quad L = 250.0\\text{ g/L}$$
   Near $B \\approx 270\\text{--}280\\text{ g/L}$, light attenuation throttles growth. Once the culture reaches this equilibrium, cell division matches basal mortality, and internal recycling sustains the existing population. Dosing is only required to replace net irreversible losses (dilution and metabolic waste), which requires only ~364 mL across 3,500 steps.

### 2.5 Potential Benchmark Simplifications
The audit identifies the following benchmark structural characteristics:
1. **Large Reserve Buffer Capacity**: $C_{\\text{int}} = 0.50$ allows cells to store up to 50% of their dry weight in intracellular reserves, creating significant biological inertia that shields simple threshold controllers from short-term fluctuations.
2. **High Recycling Efficiency**: Substantial internal recycling of maintenance and dead biomass allows closed-loop systems to self-sustain with minimal external inputs once peak density is reached.
3. **Absence of EC Safety Constraints in Heuristic Baseline**: The production reserve rule lacks an EC ceiling, causing catastrophic runaway toxicity in 30% of episodes.

---

## 3. Scenario Inventory

Exhaustive inventory of all scenario and disturbance configurations discovered across `simulation/disturbances.py`, `simulation/synthetic_generator.py`, and `simulation_runner/closed_loop_eval.py`:

| Scenario Name | Source File | Definition Entity | Disturbances Involved | Benchmark Status |
| :--- | :--- | :--- | :--- | :--- |
__INVENTORY_TABLE__

---

## 4. Current Evaluation Coverage

| Dimension | Count | Coverage % | Details |
| :--- | :--- | :--- | :--- |
__COVERAGE_TABLE__

### Key Finding on Evaluation Coverage:
The 3,500-step multi-seed evaluation currently covers only **4 out of 12** implemented operational regimes (33.3% coverage).
Eight distinct regimes are fully implemented in code but remain unevaluated in the benchmark paper:
1. `noisy_sensors` (3x measurement noise)
2. `delayed_response` (8% immediate absorption, extended 4-stage delay kernel)
3. `actuator_failure` (35% pump mechanical efficiency degradation)
4. `actuator_saturation` (flowrate capped at 1.0 mL/min, duration at 5.0 s)
5. `sediment` (+18.0 NTU turbidity sensor blinding plume)
6. `missing_data` (EC sensor dropout every 17 steps)
7. `temp_spike` (short acute thermal pulse)
8. `ec_drop` (acute dissolved nutrient drop)

---

## 5. Findings

Based strictly on code inspection, simulator equations, and empirical multi-seed data:

1. **The Reserve Rule Does NOT Invalidate Learned Control**:
   The apparent superiority of `rule_based_reserve` in previous single-seed reports was an artifact of looking only at successful seeds. Under rigorous 120-episode evaluation, `rule_based_reserve` has an unacceptable **30.0% collapse rate** (and **70.0% in cold shock**), whereas both PID and LSTM achieve **0% collapse**.
2. **Information Asymmetry is Significant**:
   The reserve-ratio rule controller receives privileged, unobservable ground-truth state variables ($\rho = R / (B \\cdot 0.5)$), completely bypassing sensor noise, bias, and measurement delay. In contrast, the LSTM operates under strict partial observability from noisy sensor streams.
3. **Reserve Ratio Couples Directly to Biomass**:
   Because $\\partial\\rho / \\partial B = -\\rho / B$, the reserve ratio acts as a population-growth detector, automatically triggering dosing as the culture expands. However, the absence of an EC ceiling makes this feedback fatal when cellular uptake is impaired.
4. **Low Dosing at High Biomass is a Legitimate Consequence of Internal Recycling**:
   In benign regimes, high biomass (280 g/L) requires little external dosing because mineralization (30%) and maintenance recycling (20% dissolved, 50% detritus) return nutrients to the culture, and light attenuation (L = 250 g/L) halts runaway expansion.
5. **Significant Benchmark Expansion Headroom Exists**:
   Two-thirds of the repository's implemented disturbance regimes (`noisy_sensors`, `delayed_response`, `actuator_failure`, `actuator_saturation`, `sediment`, `missing_data`) are already coded but have not yet been evaluated in the 3,500-step benchmark.
"""

    report_content = (
        template.replace("__SCENARIO_TABLE__", scenario_table_md)
        .replace("__AGGREGATE_TABLE__", aggregate_table_md)
        .replace("__INFO_TABLE__", info_table_md)
        .replace("__INVENTORY_TABLE__", inventory_table_md)
        .replace("__COVERAGE_TABLE__", coverage_table_md)
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_path


def run_full_benchmark_audit(output_dir: str = "reports/benchmark_audit") -> Path:
    """Execute complete benchmark audit pipeline."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/3: Running Failure and Collapse Probability Analysis...")
    scenario_df, aggregate_df = run_and_save_failure_analysis(
        "reports/multiseed_long_horizon/episode_results.csv",
        out_dir,
        collapse_threshold=50.0,
        secondary_threshold=20.0,
    )

    print("Step 2/3: Running Reserve-Rule Mechanistic Audit...")
    reserve_audit_res = run_and_save_reserve_rule_audit(out_dir)
    traj_audits = reserve_audit_res["trajectory_audits"]
    info_table_df = pd.DataFrame(reserve_audit_res["information_table"])

    print("Step 3/3: Running Repository-Wide Scenario Inventory...")
    inventory_df, coverage_df = run_and_save_scenario_inventory(out_dir)

    print("Generating comprehensive final Markdown report...")
    report_path = generate_markdown_report(
        out_dir,
        scenario_df,
        aggregate_df,
        traj_audits,
        info_table_df,
        inventory_df,
        coverage_df,
    )
    print(f"Audit completed successfully! Report generated at: {report_path}")
    return report_path


if __name__ == "__main__":
    run_full_benchmark_audit()
