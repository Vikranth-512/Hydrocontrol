"""
Publication-Quality Visualizations for Paired Multi-Seed Long-Horizon Controller Evaluation.

Generates:
1. Plot 1: Paired Controller Differences (LSTM vs PID) with individual seed points & 95% bootstrap CIs
2. Plot 2: Paired Controller Differences (LSTM vs Reserve-Rule) with practical equivalence margins
3. Plot 3: Multi-Controller Episode Distributions (boxplots with overlaid jittered points)
4. Plot 4: Performance vs Control Effort Trade-Off (Biomass vs Total Dose scatter)
5. Plot 5: Representative Illustrative Trajectories (Seed 42 illustrative 3500-step time series)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Color palette for controllers
CONTROLLER_COLORS = {
    "no_action": "#7f7f7f",          # Slate Gray
    "rule_based_reserve": "#2ca02c",  # Forest Green
    "pid": "#1f77b4",                 # Deep Blue
    "lstm": "#d62728",                # Crimson Red
}

CONTROLLER_LABELS = {
    "no_action": "No-Action",
    "rule_based_reserve": "Reserve-Ratio Rule",
    "pid": "PID Controller",
    "lstm": "LSTM Policy",
}

SCENARIO_LABELS = {
    "normal": "Normal",
    "heatwave": "Heatwave",
    "nutrient_depletion": "Nutrient Depletion",
    "cold_shock": "Cold Shock",
    "noisy_sensors": "Noisy Sensors",
    "delayed_response": "Delayed Response",
    "actuator_failure": "Actuator Failure",
    "actuator_saturation": "Actuator Saturation",
    "sediment": "Sediment Plume",
    "temp_spike": "Temperature Spike",
    "ec_drop": "EC Drop",
}


def plot_paired_differences(
    df_diffs: pd.DataFrame,
    df_ci: pd.DataFrame,
    comparison: str = "lstm_vs_pid",
    output_path: Optional[Path] = None,
    show_equivalence_margin: bool = False,
) -> plt.Figure:
    """
    Plot paired episode-level differences across scenarios for 4 key metrics:
    EC MAE, Final Biomass, Final Reserve Ratio, Total Dose.
    Shows individual seed points, zero line, mean, and 95% bootstrap CI bars.
    """
    metrics = [
        ("final_biomass", "Final Biomass (g/L)", True),
        ("ec_mae", "EC MAE (mS/cm)", False),
        ("final_reserve_ratio", "Final Reserve Ratio", True),
        ("total_dose", "Total Dose (mL)", False),
    ]

    # Dynamically determine scenarios from the data (excluding equal_weighted_aggregate)
    scenarios = sorted([s for s in df_diffs["scenario"].unique() if s != "equal_weighted_aggregate"])
    n_scenarios = len(scenarios)
    
    # Use recommended figure dimensions for publication quality
    # 6.5 × 5.5 inches at 300 DPI = 1950 × 1650 pixels (1.18:1 aspect ratio)
    # This provides good vertical space for subplots and rotated labels
    fig_width = 6.5
    fig_height = 5.5
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), dpi=300)
    axes = axes.flatten()

    for idx, (m_key, m_label, higher_better) in enumerate(metrics):
        ax = axes[idx]
        diff_col = f"{comparison}_{m_key}_diff"

        for s_idx, sc in enumerate(scenarios):
            sc_diffs = df_diffs[df_diffs["scenario"] == sc][diff_col].dropna().values
            n_pts = len(sc_diffs)
            # Add subtle jitter for scatter points
            jitter = np.random.default_rng(42 + s_idx).uniform(-0.15, 0.15, size=n_pts)
            x_pos = s_idx + jitter

            ax.scatter(
                x_pos,
                sc_diffs,
                color="#4a4a4a",
                alpha=0.45,
                s=28,
                edgecolors="none",
                zorder=2,
            )

            # Retrieve bootstrap CI for this scenario
            ci_row = df_ci[
                (df_ci["comparison"] == comparison)
                & (df_ci["scenario"] == sc)
                & (df_ci["metric"] == m_key)
            ]
            if len(ci_row) > 0:
                mean_val = ci_row.iloc[0]["mean_difference"]
                ci_low = ci_row.iloc[0]["ci95_lower"]
                ci_high = ci_row.iloc[0]["ci95_upper"]

                # Plot mean point and error bar
                ax.errorbar(
                    s_idx,
                    mean_val,
                    yerr=[[mean_val - ci_low], [ci_high - mean_val]],
                    fmt="o",
                    color="#d62728" if comparison.startswith("lstm") else "#1f77b4",
                    ecolor="#d62728" if comparison.startswith("lstm") else "#1f77b4",
                    elinewidth=2.2,
                    capsize=4.5,
                    capthick=1.5,
                    markersize=6.5,
                    zorder=4,
                )

                # Show equivalence margin if requested
                if show_equivalence_margin and "threshold_delta" in ci_row.columns:
                    delta = ci_row.iloc[0]["threshold_delta"]
                    ax.axhspan(-delta, delta, color="#2ca02c", alpha=0.08, zorder=1)
                    ax.axhline(delta, color="#2ca02c", linestyle=":", alpha=0.5, linewidth=0.9)
                    ax.axhline(-delta, color="#2ca02c", linestyle=":", alpha=0.5, linewidth=0.9)

        # Zero reference line
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)

        ax.set_title(m_label, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(scenarios)))
        
        # Format x-axis labels for better readability with many scenarios
        scenario_labels = [SCENARIO_LABELS.get(s, s) for s in scenarios]
        if n_scenarios > 6:
            # Rotate labels for better readability with many scenarios
            ax.set_xticklabels(scenario_labels, fontsize=9, rotation=45, ha='right')
        else:
            ax.set_xticklabels(scenario_labels, fontsize=10)
        
        ax.set_ylabel(f"Paired Difference ($\\Delta$)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)

    comp_title = (
        "Paired Controller Differences: LSTM vs PID (30 Seeds per Regime)"
        if comparison == "lstm_vs_pid"
        else "ML-Discriminability Audit: LSTM vs Reserve-Ratio Rule (With Equivalence Margins)"
    )
    fig.suptitle(comp_title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_episode_distributions(
    df_episodes: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot multi-controller episode-level distributions (boxplots with overlaid points)
    for EC MAE, Final Biomass, and Total Dose across scenarios.
    """
    metrics = [
        ("final_biomass", "Final Biomass (g/L)"),
        ("ec_mae", "EC MAE (mS/cm)"),
        ("total_dose", "Total Dose (mL)"),
        ("final_reserve_ratio", "Final Reserve Ratio (rho)"),
    ]

    controllers = ["no_action", "rule_based_reserve", "pid", "lstm"]
    # Dynamically determine scenarios from the data
    scenarios = sorted(df_episodes["scenario"].unique())
    
    # Adjust figure size based on number of scenarios for better readability
    n_scenarios = len(scenarios)
    width_factor = max(1.0, n_scenarios / 4.0)  # Scale width proportionally
    fig_width = 14 * width_factor
    fig_height = 10
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), dpi=300)
    axes = axes.flatten()

    for idx, (m_key, m_label) in enumerate(metrics):
        ax = axes[idx]
        positions = []
        pos_counter = 1
        pos_labels = []

        for sc in scenarios:
            df_sc = df_episodes[df_episodes["scenario"] == sc]
            for c_key in controllers:
                vals = df_sc[df_sc["controller"] == c_key][m_key].dropna().values
                box = ax.boxplot(
                    vals,
                    positions=[pos_counter],
                    widths=0.6,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor=CONTROLLER_COLORS[c_key], alpha=0.35),
                    medianprops=dict(color=CONTROLLER_COLORS[c_key], linewidth=1.8),
                )

                # Overlaid jittered points
                jitter = np.random.default_rng(10 + pos_counter).uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(
                    pos_counter + jitter,
                    vals,
                    color=CONTROLLER_COLORS[c_key],
                    alpha=0.6,
                    s=14,
                    edgecolors="none",
                )
                pos_counter += 1
            pos_counter += 1  # Gap between scenarios

        # Set scenario group labels (dynamic based on number of scenarios)
        # Each scenario group has 4 controllers + 1 gap
        group_width = len(controllers) + 1
        tick_pos = [group_width * (i + 0.5) - 0.5 for i in range(len(scenarios))]
        ax.set_xticks(tick_pos)
        
        # Format x-axis labels for better readability with many scenarios
        scenario_labels = [SCENARIO_LABELS.get(s, s) for s in scenarios]
        if n_scenarios > 6:
            # Rotate labels for better readability with many scenarios
            ax.set_xticklabels(scenario_labels, fontsize=9, rotation=45, ha='right', fontweight='bold')
        else:
            ax.set_xticklabels(scenario_labels, fontsize=10, fontweight='bold')
        
        ax.set_ylabel(m_label, fontsize=10)
        ax.set_title(m_label, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.4)

    # Custom legend for controllers
    custom_handles = [
        plt.Line2D([0], [0], color=CONTROLLER_COLORS[c], lw=3, label=CONTROLLER_LABELS[c])
        for c in controllers
    ]
    fig.legend(handles=custom_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.99), fontsize=11)
    
    # Adjust layout to accommodate rotated labels if needed
    if n_scenarios > 6:
        plt.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)  # Extra padding for rotated labels
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_performance_vs_effort(
    df_episodes: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot biological yield (Final Biomass) vs Control Effort (Total Dose)
    and EC MAE vs Total Dose for every episode/controller, revealing whether
    performance gains are driven by excessive dosing volume.
    """
    controllers = ["no_action", "rule_based_reserve", "pid", "lstm"]
    # Dynamically determine scenarios from the data
    scenarios = sorted(df_episodes["scenario"].unique())
    n_scenarios = len(scenarios)
    
    # Create marker mapping for scenarios (cycle through available markers)
    markers = ["o", "^", "s", "D", "v", "p", "*", "h", "d", "X", "+", "x"]
    scenario_markers = {sc: markers[i % len(markers)] for i, sc in enumerate(scenarios)}

    # Adjust figure size based on number of scenarios for better legend space
    fig_width = max(14, 14 + (n_scenarios - 4) * 0.5)  # Scale width for legend space
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 6), dpi=300)

    for c_key in controllers:
        df_c = df_episodes[df_episodes["controller"] == c_key]
        for sc, marker in scenario_markers.items():
            df_cs = df_c[df_c["scenario"] == sc]
            if len(df_cs) == 0:
                continue

            # Plot 1: Biomass vs Total Dose
            ax1.scatter(
                df_cs["total_dose"],
                df_cs["final_biomass"],
                color=CONTROLLER_COLORS[c_key],
                marker=marker,
                alpha=0.65,
                s=40,
                edgecolors="none",
            )

            # Plot 2: EC MAE vs Total Dose
            ax2.scatter(
                df_cs["total_dose"],
                df_cs["ec_mae"],
                color=CONTROLLER_COLORS[c_key],
                marker=marker,
                alpha=0.65,
                s=40,
                edgecolors="none",
            )

    ax1.set_xlabel("Total Nutrient Dose (mL)", fontsize=11)
    ax1.set_ylabel("Final Algae Biomass (g/L)", fontsize=11)
    ax1.set_title("Biological Yield vs Control Effort", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.set_xlabel("Total Nutrient Dose (mL)", fontsize=11)
    ax2.set_ylabel("EC MAE (mS/cm)", fontsize=11)
    ax2.set_title("Regulation Precision vs Control Effort", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Controller legend
    ctrl_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CONTROLLER_COLORS[c], markersize=9, label=CONTROLLER_LABELS[c])
        for c in controllers
    ]
    # Scenario legend
    sc_handles = [
        plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="#555555", markersize=8, label=SCENARIO_LABELS.get(s, s))
        for s, m in scenario_markers.items()
    ]

    # Adjust legend layout based on number of scenarios
    if n_scenarios > 6:
        # Use more columns for scenario legend when there are many scenarios
        sc_ncols = min(6, n_scenarios)
        leg1 = fig.legend(handles=ctrl_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.32, -0.15), fontsize=10, title="Controller")
        leg2 = fig.legend(handles=sc_handles, loc="lower center", ncol=sc_ncols, bbox_to_anchor=(0.72, -0.15), fontsize=9, title="Operating Regime")
        plt.tight_layout(rect=[0, 0, 1, 0.90])  # Extra bottom padding for legends
    else:
        leg1 = fig.legend(handles=ctrl_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.32, -0.05), fontsize=10, title="Controller")
        leg2 = fig.legend(handles=sc_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.72, -0.05), fontsize=10, title="Operating Regime")
        plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_representative_trajectories(
    trajectories: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot single representative 3500-step time series for Seed 42 (Normal scenario),
    clearly labeled as illustrative.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True, dpi=300)

    for c_key, data in trajectories.items():
        ec = data.get("ec", [])
        biomass = data.get("biomass", [])
        reserve = data.get("reserve_ratio", [])
        steps = np.arange(len(ec))

        ax1.plot(steps, ec, label=CONTROLLER_LABELS.get(c_key, c_key), color=CONTROLLER_COLORS.get(c_key, "#333333"), linewidth=1.2, alpha=0.85)
        ax2.plot(steps, biomass, label=CONTROLLER_LABELS.get(c_key, c_key), color=CONTROLLER_COLORS.get(c_key, "#333333"), linewidth=1.2, alpha=0.85)
        ax3.plot(steps, reserve, label=CONTROLLER_LABELS.get(c_key, c_key), color=CONTROLLER_COLORS.get(c_key, "#333333"), linewidth=1.2, alpha=0.85)

    ax1.axhline(1.2, color="black", linestyle="--", alpha=0.6, label="EC Target (1.2 mS/cm)")
    ax1.set_ylabel("EC (mS/cm)", fontsize=10)
    ax1.set_title("Electrical Conductivity Regulation", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.set_ylabel("Biomass (g/L)", fontsize=10)
    ax2.set_title("Algae Biomass Dynamics", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)

    ax3.axhline(0.50, color="black", linestyle="--", alpha=0.6, label="Quota Setpoint (rho=0.50)")
    ax3.axhline(0.20, color="red", linestyle=":", alpha=0.6, label="Starvation Floor (rho=0.20)")
    ax3.set_ylabel("Reserve Ratio (rho)", fontsize=10)
    ax3.set_xlabel("Simulation Timestep (60 s / step)", fontsize=10)
    ax3.set_title("Intracellular Nutrient Reserve Ratio (rho)", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.5)

    ax1.legend(loc="upper right", ncol=3, fontsize=9)

    fig.suptitle(
        "Representative Illustrative Trajectory (Normal Scenario, Seed 42)\n"
        "[Note: Representative trajectory; statistical conclusions are based on the full multi-seed evaluation.]",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def generate_all_plots(analysis: Dict[str, Any], output_dir: Path) -> None:
    """Generate all 5 publication-ready figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df_diffs = analysis["df_paired_diffs"]
    df_ci = analysis["df_bootstrap_ci"]
    df_episodes = analysis["df_episodes"]
    trajectories = analysis.get("representative_trajectories", {})
    
    # Log which scenarios are being plotted
    scenarios_in_data = sorted(df_episodes["scenario"].unique())
    print(f"Scenarios found in evaluation data: {scenarios_in_data}")
    print(f"Total scenarios to plot: {len(scenarios_in_data)}")
    
    # Check for missing_data in data (should not be present)
    if "missing_data" in scenarios_in_data:
        print("WARNING: missing_data found in data - it should be excluded from evaluation")
    
    # Check evaluation manifest if available
    manifest_path = output_dir.parent / "evaluation_manifest.json"
    if manifest_path.exists():
        import json
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        manifest_scenarios = set(manifest.get("scenarios", []))
        data_scenarios = set(scenarios_in_data)
        if manifest_scenarios != data_scenarios:
            print(f"WARNING: Scenarios in manifest differ from data")
            print(f"  Manifest: {sorted(manifest_scenarios)}")
            print(f"  Data: {sorted(data_scenarios)}")

    print(f"Generating Figure 1: Paired Controller Differences (LSTM vs PID)...")
    plot_paired_differences(
        df_diffs, df_ci, comparison="lstm_vs_pid",
        output_path=output_dir / "plot1_paired_differences_lstm_vs_pid.png",
        show_equivalence_margin=False,
    )

    print(f"Generating Figure 2: ML-Discriminability Audit (LSTM vs Reserve-Rule)...")
    plot_paired_differences(
        df_diffs, df_ci, comparison="lstm_vs_rule_based_reserve",
        output_path=output_dir / "plot2_paired_differences_lstm_vs_reserve_rule.png",
        show_equivalence_margin=True,
    )

    print(f"Generating Figure 3: Multi-Controller Episode Distributions...")
    plot_episode_distributions(
        df_episodes,
        output_path=output_dir / "plot3_episode_distributions.png",
    )

    print(f"Generating Figure 4: Performance vs Control Effort Trade-Off...")
    plot_performance_vs_effort(
        df_episodes,
        output_path=output_dir / "plot4_performance_vs_effort.png",
    )

    if trajectories:
        print(f"Generating Figure 5: Representative Illustrative Trajectories...")
        plot_representative_trajectories(
            trajectories,
            output_path=output_dir / "plot5_representative_trajectories.png",
        )
