"""
Matplotlib visualization for policy learning experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


class Plotter:
    """Generate publication-grade figures."""

    def __init__(self, output_dir: Path, style: str = "seaborn-v0_8-whitegrid") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("ggplot")

    def plot_training_curves(self, history: Dict[str, list], name: str = "training") -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history.get("train_loss", []), label="train")
        axes[0].plot(history.get("val_loss", []), label="val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].set_title("Training / Validation Loss")

        axes[1].plot(history.get("val_rmse", []), label="RMSE")
        axes[1].plot(history.get("val_mae", []), label="MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].set_title("Validation Metrics")

        fig.tight_layout()
        path = self.output_dir / f"{name}_curves.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_predicted_vs_optimal(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        name: str = "actions",
    ) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = ["Flowrate", "Duration"]
        for i, ax in enumerate(axes):
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=8)
            lims = [
                min(y_true[:, i].min(), y_pred[:, i].min()),
                max(y_true[:, i].max(), y_pred[:, i].max()),
            ]
            ax.plot(lims, lims, "r--", lw=1)
            ax.set_xlabel(f"Optimal {labels[i]}")
            ax.set_ylabel(f"Predicted {labels[i]}")
            ax.set_title(labels[i])
        fig.tight_layout()
        path = self.output_dir / f"{name}_pred_vs_optimal.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_ec_trajectory(
        self,
        ec: np.ndarray,
        ec_target: float,
        dt: float,
        name: str = "ec_traj",
    ) -> Path:
        t = np.arange(len(ec)) * dt / 60.0
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, ec, label="EC")
        ax.axhline(ec_target, color="r", linestyle="--", label="Target")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("EC (mS/cm)")
        ax.legend()
        ax.set_title("EC Trajectory")
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_health_trajectory(
        self,
        health: np.ndarray,
        damage: np.ndarray,
        dt: float,
        name: str = "health_traj",
    ) -> Path:
        t = np.arange(len(health)) * dt / 60.0
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        axes[0].plot(t, health, label="Health Index", color="green")
        axes[0].axhline(0.7, color="orange", linestyle="--", label="Health Floor")
        axes[0].axhline(0.2, color="red", linestyle="--", label="Collapse Threshold")
        axes[0].set_ylabel("Health Index")
        axes[0].legend()
        axes[0].set_title("Ecosystem Health Over Time")
        axes[0].set_ylim([0, 1.05])
        
        axes[1].plot(t, damage, label="Damage Index", color="red")
        axes[1].set_ylabel("Damage Index")
        axes[1].set_xlabel("Time (min)")
        axes[1].legend()
        axes[1].set_title("Cumulative Damage")
        axes[1].set_ylim([0, 1.05])
        
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_reserve_trajectory(
        self,
        reserve_ratio: np.ndarray,
        dt: float,
        name: str = "reserve_traj",
    ) -> Path:
        t = np.arange(len(reserve_ratio)) * dt / 60.0
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, reserve_ratio, label="Reserve Ratio", color="blue")
        ax.axhline(0.30, color="orange", linestyle="--", label="Reserve Floor")
        ax.axhline(0.15, color="red", linestyle="--", label="Starvation Threshold")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Reserve Ratio")
        ax.legend()
        ax.set_title("Nutrient Reserve Over Time")
        ax.set_ylim([0, 1.05])
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_biomass_trajectory(
        self,
        biomass: np.ndarray,
        dt: float,
        name: str = "biomass_traj",
    ) -> Path:
        t = np.arange(len(biomass)) * dt / 60.0
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, biomass, label="Algae Biomass", color="green")
        ax.axhline(300.0, color="orange", linestyle="--", label="Bloom Threshold")
        ax.axhline(20.0, color="red", linestyle="--", label="Persistence Threshold")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Biomass (g)")
        ax.legend()
        ax.set_title("Algae Biomass Over Time")
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_dosing_behavior(
        self,
        flowrate: np.ndarray,
        duration: np.ndarray,
        dt: float,
        name: str = "dosing",
    ) -> Path:
        t = np.arange(len(flowrate)) * dt / 60.0
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t, flowrate)
        axes[0].set_ylabel("Flowrate (mL/min)")
        axes[1].plot(t, duration)
        axes[1].set_ylabel("Duration (s)")
        axes[1].set_xlabel("Time (min)")
        fig.suptitle("Nutrient Dosing Behavior")
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_controller_comparison(
        self,
        results: Dict[str, Any],
        ec_target: float,
        dt: float,
        scenario: str = "normal",
        name: str = "controller_cmp",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(11, 5))
        t = None
        for ctrl_name, color in [
            ("pid", "C0"),
            ("rule_based", "C1"),
            ("lstm", "C2"),
        ]:
            if scenario not in results:
                continue
            data = results[scenario].get(ctrl_name, {})
            traj = data.get("trajectory", {})
            ec = traj.get("ec")
            if ec is None:
                continue
            t = np.arange(len(ec)) * dt / 60.0
            ax.plot(t, ec, label=ctrl_name, color=color, alpha=0.85)
        ax.axhline(ec_target, color="k", linestyle="--", label="Target")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("EC (mS/cm)")
        ax.legend()
        ax.set_title(f"Closed-Loop EC — {scenario}")
        fig.tight_layout()
        path = self.output_dir / f"{name}_{scenario}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_error_distribution(
        self,
        errors: np.ndarray,
        name: str = "errors",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("EC Error")
        ax.set_ylabel("Count")
        ax.set_title("EC Error Distribution")
        fig.tight_layout()
        path = self.output_dir / f"{name}_dist.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_stability_analysis(
        self,
        metrics_by_controller: Dict[str, Dict[str, float]],
        name: str = "stability",
    ) -> Path:
        controllers = list(metrics_by_controller.keys())
        keys = ["stability_variance", "overshoot", "cumulative_dosing_cost"]
        x = np.arange(len(controllers))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, key in enumerate(keys):
            vals = [metrics_by_controller[c].get(key, 0) for c in controllers]
            ax.bar(x + i * width, vals, width, label=key)
        ax.set_xticks(x + width)
        ax.set_xticklabels(controllers)
        ax.legend()
        ax.set_title("Stability & Cost Comparison")
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_metrics_table(
        self,
        comparison: Dict[str, Dict[str, Dict[str, float]]],
        name: str = "metrics_summary",
    ) -> Path:
        """Bar chart of EC MAE across scenarios and controllers."""
        fig, ax = plt.subplots(figsize=(12, 6))
        scenarios = list(comparison.keys())
        controllers = ["pid", "rule_based", "lstm"]
        x = np.arange(len(scenarios))
        width = 0.25
        for i, ctrl in enumerate(controllers):
            vals = [
                comparison[sc].get(ctrl, {}).get("metrics", {}).get("ec_mae", np.nan)
                for sc in scenarios
            ]
            ax.bar(x + i * width, vals, width, label=ctrl)
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios, rotation=15)
        ax.set_ylabel("EC MAE")
        ax.legend()
        ax.set_title("Robustness: EC MAE by Scenario")
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_turbidity_trajectory(
        self,
        turbidity: np.ndarray,
        dt: float,
        name: str = "turbidity_traj",
    ) -> Path:
        t = np.arange(len(turbidity)) * dt / 60.0
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, turbidity, color="C2")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Turbidity (NTU)")
        ax.set_title("Turbidity / algae proxy trajectory")
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_delayed_absorption(
        self,
        pending: np.ndarray,
        ec: np.ndarray,
        dt: float,
        name: str = "delayed_absorption",
    ) -> Path:
        t = np.arange(len(ec)) * dt / 60.0
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(t, ec, label="EC", color="C0")
        ax1.set_ylabel("EC")
        ax2 = ax1.twinx()
        ax2.plot(t, pending, "--", label="Pending absorption", color="C1")
        ax2.set_ylabel("Queue mass")
        ax1.set_xlabel("Time (min)")
        ax1.set_title("Delayed nutrient absorption vs EC")
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # NEW EVALUATION PLOTTING FUNCTIONS (3500 Long Horizon LSTM)
    
    def plot_long_horizon_overview(
        self,
        lstm_results: Dict[str, Any],
        dt: float,
        reserve_target: float = 0.50,
    ) -> Path:
        """Plot 1: Reserve, Health, Biomass across 3500 sim run with thresholds"""
        traj = lstm_results["trajectory"]
        t = np.arange(len(traj["health"])) * dt / 60.0
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # Health plot
        axes[0].plot(t, traj["health"], color="C2", linewidth=2, label="Health Index")
        axes[0].axhline(0.7, color="orange", linestyle="--", alpha=0.7, label="Health Floor")
        axes[0].axhline(0.2, color="red", linestyle="--", alpha=0.7, label="Collapse")
        axes[0].set_ylabel("Health Index")
        axes[0].set_ylim([0, 1.05])
        axes[0].legend(loc='lower right')
        axes[0].set_title("LSTM Health Trajectory (3500 Steps)")
        axes[0].grid(True, alpha=0.3)
        
        # Reserve plot
        axes[1].plot(t, traj["reserve_ratio"], color="C0", linewidth=2, label="Reserve Ratio")
        axes[1].axhline(reserve_target, color="green", linestyle="--", alpha=0.7, label=f"Target ({reserve_target})")
        axes[1].axhline(0.20, color="red", linestyle="--", alpha=0.7, label="Floor (0.20)")
        axes[1].axhline(0.15, color="darkred", linestyle="--", alpha=0.7, label="Starvation (0.15)")
        axes[1].set_ylabel("Reserve Ratio")
        axes[1].set_ylim([0, 1.05])
        axes[1].legend(loc='lower right')
        axes[1].set_title("LSTM Reserve Dynamics (3500 Steps)")
        axes[1].grid(True, alpha=0.3)
        
        # Biomass plot
        axes[2].plot(t, traj["biomass"], color="C1", linewidth=2, label="Biomass")
        axes[2].axhline(20.0, color="red", linestyle="--", alpha=0.7, label="Persistence Threshold")
        axes[2].axhline(300.0, color="orange", linestyle="--", alpha=0.7, label="Bloom Threshold")
        axes[2].set_xlabel("Time (min)")
        axes[2].set_ylabel("Biomass (g)")
        axes[2].legend(loc='upper right')
        axes[2].set_title("LSTM Biomass Trajectory (3500 Steps)")
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle("LSTM Long Horizon Overview (3500 Steps, PID Conditions)", fontsize=16, fontweight='bold')
        fig.tight_layout()
        path = self.output_dir / "long_horizon_overview.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path
    
    def plot_lstm_smoothness(
        self,
        lstm_results: Dict[str, Any],
        dt: float,
    ) -> Path:
        """Plot 2: LSTM smoothness across 3500 sim run"""
        traj = lstm_results["trajectory"]
        t = np.arange(len(traj["flowrate"])) * dt / 60.0
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        # Flowrate plot
        axes[0].plot(t, traj["flowrate"], color="C2", linewidth=2, label="Flowrate")
        axes[0].set_ylabel("Flowrate (mL/min)")
        axes[0].legend(loc='upper right')
        axes[0].set_title("LSTM Flowrate Trajectory (3500 Steps)")
        axes[0].grid(True, alpha=0.3)
        
        # Duration plot
        axes[1].plot(t, traj["duration"], color="C3", linewidth=2, label="Duration")
        axes[1].set_xlabel("Time (min)")
        axes[1].set_ylabel("Duration (s)")
        axes[1].legend(loc='upper right')
        axes[1].set_title("LSTM Duration Trajectory (3500 Steps)")
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle("LSTM Controller Smoothness (3500 Steps, PID Conditions)", fontsize=16, fontweight='bold')
        fig.tight_layout()
        path = self.output_dir / "lstm_smoothness.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path
    
    def plot_nutrient_efficiency(
        self,
        lstm_results: Dict[str, Any],
        dt: float,
    ) -> Path:
        """Plot 3: Nutrient efficiency across 3500 sim run"""
        traj = lstm_results["trajectory"]
        t = np.arange(len(traj["flowrate"])) * dt / 60.0
        
        # Calculate dose and cumulative dose
        dose = traj["flowrate"] * traj["duration"] / 60.0
        cumulative_dose = np.cumsum(dose)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        # Instantaneous dose plot
        axes[0].plot(t, dose, color="C2", linewidth=2, label="Instantaneous Dose")
        axes[0].set_ylabel("Dose (mL)")
        axes[0].legend(loc='upper right')
        axes[0].set_title("LSTM Instantaneous Nutrient Dose (3500 Steps)")
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative dose plot
        axes[1].plot(t, cumulative_dose, color="C4", linewidth=2, label="Cumulative Dose")
        axes[1].set_xlabel("Time (min)")
        axes[1].set_ylabel("Cumulative Dose (mL)")
        axes[1].legend(loc='upper right')
        axes[1].set_title("LSTM Cumulative Nutrient Efficiency (3500 Steps)")
        axes[1].grid(True, alpha=0.3)
        
        # Add total dose annotation
        total_dose = cumulative_dose[-1]
        axes[1].text(0.02, 0.95, f"Total Dose: {total_dose:.2f} mL", 
                    transform=axes[1].transAxes, fontsize=12, 
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        fig.suptitle("LSTM Nutrient Efficiency (3500 Steps, PID Conditions)", fontsize=16, fontweight='bold')
        fig.tight_layout()
        path = self.output_dir / "nutrient_efficiency.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path
    
    def figure1_long_term_health(
        self,
        comparison: Dict[str, Dict[str, Dict[str, Any]]],
        dt: float,
        reserve_target: float = 0.50,
    ) -> Path:
        """Figure 1: Long-term Ecosystem Health (2x3 panel) - LSTM only"""
        scenarios = ["normal", "sensor_noise", "ec_drift", "temp_spike", "delayed_response", "actuator_saturation"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            if scenario not in comparison:
                ax.text(0.5, 0.5, f"No data for {scenario}", ha='center', va='center')
                ax.set_title(scenario.replace('_', ' ').title())
                continue
                
            # LSTM only
            data = comparison[scenario].get("lstm", {})
            traj = data.get("trajectory", {})
            health = traj.get("health")
            if health is not None:
                t = np.arange(len(health)) * dt / 60.0
                ax.plot(t, health, label="LSTM", color="C2", alpha=0.85, linewidth=2)
            
            ax.axhline(0.7, color="orange", linestyle="--", alpha=0.5, label="Health Floor")
            ax.axhline(0.2, color="red", linestyle="--", alpha=0.5, label="Collapse")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Health Index")
            ax.set_ylim([0, 1.05])
            ax.set_title(scenario.replace('_', ' ').title())
            if idx == 0:
                ax.legend(loc='lower right', fontsize=8)
        
        fig.suptitle("Figure 1: Long-term Ecosystem Health (LSTM Performance)", fontsize=14, fontweight='bold')
        fig.tight_layout()
        path = self.output_dir / "figure1_long_term_health.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def figure2_reserve_dynamics(
        self,
        comparison: Dict[str, Dict[str, Dict[str, Any]]],
        dt: float,
        reserve_target: float = 0.50,
        reserve_floor: float = 0.20,
    ) -> Path:
        """Figure 2: Reserve Dynamics with floor/target lines - LSTM only"""
        scenarios = ["normal", "delayed_response", "ec_drift"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            if scenario not in comparison:
                ax.text(0.5, 0.5, f"No data for {scenario}", ha='center', va='center')
                ax.set_title(scenario.replace('_', ' ').title())
                continue
                
            # LSTM only
            data = comparison[scenario].get("lstm", {})
            traj = data.get("trajectory", {})
            reserve = traj.get("reserve_ratio")
            if reserve is not None:
                t = np.arange(len(reserve)) * dt / 60.0
                ax.plot(t, reserve, label="LSTM", color="C2", alpha=0.85, linewidth=2)
            
            ax.axhline(reserve_target, color="green", linestyle="--", alpha=0.7, label=f"Target ({reserve_target})")
            ax.axhline(reserve_floor, color="red", linestyle="--", alpha=0.7, label=f"Floor ({reserve_floor})")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Reserve Ratio")
            ax.set_ylim([0, 1.05])
            ax.set_title(scenario.replace('_', ' ').title())
            if idx == 0:
                ax.legend(loc='lower right', fontsize=9)
        
        fig.suptitle("Figure 2: Reserve Dynamics (LSTM Performance)", fontsize=14, fontweight='bold')
        fig.tight_layout()
        path = self.output_dir / "figure2_reserve_dynamics.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def figure3_biomass_persistence(
        self,
        comparison: Dict[str, Dict[str, Dict[str, Any]]],
        dt: float,
    ) -> Path:
        """Figure 3: Biomass Persistence - LSTM only"""
        scenarios = ["normal", "sensor_noise", "ec_drift", "temp_spike", "delayed_response", "actuator_saturation"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            if scenario not in comparison:
                ax.text(0.5, 0.5, f"No data for {scenario}", ha='center', va='center')
                ax.set_title(scenario.replace('_', ' ').title())
                continue
                
            # LSTM only
            data = comparison[scenario].get("lstm", {})
            traj = data.get("trajectory", {})
            biomass = traj.get("biomass")
            if biomass is not None:
                t = np.arange(len(biomass)) * dt / 60.0
                ax.plot(t, biomass, label="LSTM", color="C2", alpha=0.85, linewidth=2)
            
            ax.axhline(20.0, color="red", linestyle="--", alpha=0.5, label="Persistence Threshold")
            ax.axhline(300.0, color="orange", linestyle="--", alpha=0.5, label="Bloom Threshold")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Biomass (g)")
            ax.set_title(scenario.replace('_', ' ').title())
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        fig.suptitle("Figure 3: Biomass Persistence (LSTM Performance)", fontsize=14, fontweight='bold')
        fig.tight_layout()
        path = self.output_dir / "figure3_biomass_persistence.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def figure4_disturbance_recovery(
        self,
        comparison: Dict[str, Dict[str, Dict[str, Any]]],
        dt: float,
    ) -> Path:
        """Figure 4: Disturbance Recovery (zoom plots) - LSTM only"""
        scenarios = ["sensor_noise", "ec_drift", "temp_spike"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            if scenario not in comparison:
                ax.text(0.5, 0.5, f"No data for {scenario}", ha='center', va='center')
                ax.set_title(scenario.replace('_', ' ').title())
                continue
                
            # LSTM only
            data = comparison[scenario].get("lstm", {})
            traj = data.get("trajectory", {})
            health = traj.get("health")
            if health is not None:
                t = np.arange(len(health)) * dt / 60.0
                # Zoom around disturbance (middle 30% of simulation)
                zoom_start = int(len(health) * 0.35)
                zoom_end = int(len(health) * 0.65)
                ax.plot(t[zoom_start:zoom_end], health[zoom_start:zoom_end], 
                       label="LSTM", color="C2", alpha=0.85, linewidth=2)
            
            ax.axvline(t[zoom_start], color="gray", linestyle=":", alpha=0.5)
            ax.axvline(t[zoom_end], color="gray", linestyle=":", alpha=0.5)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Health Index")
            ax.set_ylim([0, 1.05])
            ax.set_title(f"{scenario.replace('_', ' ').title()} (Zoom)")
            if idx == 0:
                ax.legend(loc='lower right', fontsize=9)
        
        fig.suptitle("Figure 4: Disturbance Recovery (LSTM Performance)", fontsize=14, fontweight='bold')
        fig.tight_layout()
        path = self.output_dir / "figure4_disturbance_recovery.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def figure5_nutrient_efficiency(
        self,
        comparison: Dict[str, Dict[str, Dict[str, Any]]],
        dt: float,
    ) -> Path:
        """Figure 5: Nutrient Efficiency (cumulative dosing) - LSTM only"""
        scenarios = ["normal", "sensor_noise", "ec_drift", "temp_spike"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            if scenario not in comparison:
                ax.text(0.5, 0.5, f"No data for {scenario}", ha='center', va='center')
                ax.set_title(scenario.replace('_', ' ').title())
                continue
                
            # LSTM only
            data = comparison[scenario].get("lstm", {})
            traj = data.get("trajectory", {})
            flowrate = traj.get("flowrate")
            duration = traj.get("duration")
            if flowrate is not None and duration is not None:
                dose = flowrate * duration / 60.0
                cumulative = np.cumsum(dose)
                t = np.arange(len(cumulative)) * dt / 60.0
                ax.plot(t, cumulative, label="LSTM", color="C2", alpha=0.85, linewidth=2)
            
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Cumulative Nutrient Added (mL)")
            ax.set_title(scenario.replace('_', ' ').title())
            if idx == 0:
                ax.legend(loc='upper left', fontsize=9)
        
        fig.suptitle("Figure 5: Nutrient Efficiency (LSTM Performance)", fontsize=14, fontweight='bold')
        fig.tight_layout()
        path = self.output_dir / "figure5_nutrient_efficiency.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def figure6_controller_smoothness(
        self,
        comparison: Dict[str, Dict[str, Dict[str, Any]]],
        dt: float,
    ) -> Path:
        """Figure 6: Controller Smoothness - LSTM only"""
        scenario = "normal"
        fig, ax = plt.subplots(figsize=(12, 5))
        
        if scenario in comparison:
            # LSTM only
            data = comparison[scenario].get("lstm", {})
            traj = data.get("trajectory", {})
            flowrate = traj.get("flowrate")
            if flowrate is not None:
                t = np.arange(len(flowrate)) * dt / 60.0
                ax.plot(t, flowrate, label="LSTM", color="C2", alpha=0.85, linewidth=2)
        
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Flowrate (mL/min)")
        ax.set_title("Figure 6: Controller Smoothness (LSTM Performance)")
        ax.legend(loc='upper right', fontsize=10)
        fig.tight_layout()
        path = self.output_dir / "figure6_controller_smoothness.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def figure7_phase_portrait(
        self,
        comparison: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Path:
        """Figure 7: Phase Portrait (reserve vs health) - LSTM only"""
        scenarios = ["normal", "delayed_response", "ec_drift"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            if scenario not in comparison:
                ax.text(0.5, 0.5, f"No data for {scenario}", ha='center', va='center')
                ax.set_title(scenario.replace('_', ' ').title())
                continue
                
            # LSTM only
            data = comparison[scenario].get("lstm", {})
            traj = data.get("trajectory", {})
            health = traj.get("health")
            reserve = traj.get("reserve_ratio")
            if health is not None and reserve is not None:
                ax.plot(reserve, health, label="LSTM", color="C2", alpha=0.7, linewidth=1.5)
                # Mark start and end points
                ax.scatter(reserve[0], health[0], color="C2", s=50, marker='o', alpha=0.8)
                ax.scatter(reserve[-1], health[-1], color="C2", s=50, marker='s', alpha=0.8)
            
            ax.set_xlabel("Reserve Ratio")
            ax.set_ylabel("Health Index")
            ax.set_xlim([0, 1.05])
            ax.set_ylim([0, 1.05])
            ax.set_title(scenario.replace('_', ' ').title())
            if idx == 0:
                ax.legend(loc='lower right', fontsize=9)
        
        fig.suptitle("Figure 7: Phase Portrait (LSTM Performance)", fontsize=14, fontweight='bold')
        fig.tight_layout()
        path = self.output_dir / "figure7_phase_portrait.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def figure8_performance_radar(
        self,
        comparison: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Path:
        """Figure 8: Performance Radar Chart - LSTM only (absolute values)"""
        # Aggregate metrics across scenarios for LSTM only
        metrics_to_plot = [
            ("health_mean", True),      # Higher is better
            ("damage_mean", False),     # Lower is better  
            ("reserve_mean", True),     # Higher is better
            ("biomass_mean", True),     # Higher is better
            ("total_dose", False),      # Lower is better
            ("control_smoothness", True), # Higher is better
            ("collapse_free_duration", True), # Higher is better
            ("starvation_fraction", False),   # Lower is better
        ]
        
        # Calculate average metrics for LSTM across scenarios
        lstm_metrics = []
        for metric, higher_better in metrics_to_plot:
            vals = []
            for scenario, data in comparison.items():
                if scenario == "pid_conditions":
                    continue
                ctrl_data = data.get("lstm", {})
                metrics = ctrl_data.get("metrics", {})
                val = metrics.get(metric)
                if val is not None and not np.isnan(val):
                    vals.append(val)
            if vals:
                lstm_metrics.append(np.mean(vals))
            else:
                lstm_metrics.append(0.5)
        
        # Normalize to 0-1 scale for visualization
        normalized_metrics = []
        for i, (val) in enumerate(lstm_metrics):
            higher_better = metrics_to_plot[i][1]
            # Simple normalization: assume reasonable ranges
            if metrics_to_plot[i][0] in ["health_mean", "reserve_mean", "control_smoothness"]:
                norm = min(max(val, 0), 1)  # 0-1 range
            elif metrics_to_plot[i][0] in ["damage_mean", "starvation_fraction"]:
                norm = 1 - min(max(val, 0), 1)  # Invert for lower is better
            elif metrics_to_plot[i][0] == "biomass_mean":
                norm = min(max(val / 500, 0), 1)  # Assume max biomass ~500
            elif metrics_to_plot[i][0] == "total_dose":
                norm = 1 - min(max(val / 10000, 0), 1)  # Invert, assume max dose ~10000
            elif metrics_to_plot[i][0] == "collapse_free_duration":
                norm = min(max(val / 3500, 0), 1)  # Normalize by max steps
            else:
                norm = 0.5
            normalized_metrics.append(norm)
        
        # Create radar chart
        labels = ["Health", "Damage", "Reserve", "Biomass", "Dose", "Smoothness", "Persistence", "Starvation"]
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        values = normalized_metrics + normalized_metrics[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label="LSTM", color="C2")
        ax.fill(angles, values, alpha=0.15, color="C2")
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.set_title("Figure 8: Performance Radar Chart (LSTM Metrics)", fontsize=14, fontweight='bold', pad=20)
        
        fig.tight_layout()
        path = self.output_dir / "figure8_performance_radar.png"
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return path

    def figure9_robustness_heatmap(
        self,
        comparison: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Path:
        """Figure 9: Scenario Robustness Heatmap - LSTM only"""
        scenarios = ["normal", "sensor_noise", "ec_drift", "temp_spike", "delayed_response", "actuator_saturation"]
        metrics = ["health_mean", "reserve_mean", "damage_mean", "collapse_free_duration", "total_dose", "starvation_fraction"]
        
        # Aggregate data for LSTM
        data_matrix = np.zeros((len(scenarios), len(metrics)))
        
        for i, scenario in enumerate(scenarios):
            if scenario not in comparison:
                continue
            lstm_data = comparison[scenario].get("lstm", {})
            lstm_metrics = lstm_data.get("metrics", {})
            for j, metric in enumerate(metrics):
                val = lstm_metrics.get(metric)
                if val is not None and not np.isnan(val):
                    data_matrix[i, j] = val
        
        # Normalize each column (metric) to 0-1 for coloring
        normalized_data = data_matrix.copy()
        for j in range(len(metrics)):
            col = data_matrix[:, j]
            if np.max(col) > np.min(col):
                # For metrics where higher is better (health, reserve, collapse_free_duration)
                if metrics[j] in ["health_mean", "reserve_mean", "collapse_free_duration"]:
                    normalized_data[:, j] = (col - np.min(col)) / (np.max(col) - np.min(col))
                # For metrics where lower is better (damage, dose, starvation)
                else:
                    normalized_data[:, j] = 1 - (col - np.min(col)) / (np.max(col) - np.min(col))
            else:
                normalized_data[:, j] = 0.5
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(normalized_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(scenarios)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax.set_yticklabels([s.replace('_', ' ').title() for s in scenarios])
        
        # Add text annotations
        for i in range(len(scenarios)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Scenarios")
        ax.set_title("Figure 9: Scenario Robustness Heatmap (LSTM Performance)", fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance (Green=Good, Red=Poor)')
        
        fig.tight_layout()
        path = self.output_dir / "figure9_robustness_heatmap.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def figure10_controller_tradeoff(
        self,
        comparison: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Path:
        """Figure 10: Controller Trade-off (dose vs health) - LSTM only"""
        scenarios = ["normal", "sensor_noise", "ec_drift", "temp_spike", "delayed_response", "actuator_saturation"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # LSTM only
        dose_vals = []
        health_vals = []
        scenario_labels = []
        for scenario in scenarios:
            if scenario not in comparison:
                continue
            ctrl_data = comparison[scenario].get("lstm", {})
            metrics = ctrl_data.get("metrics", {})
            dose = metrics.get("total_dose")
            health = metrics.get("health_mean")
            if dose is not None and health is not None and not np.isnan(dose) and not np.isnan(health):
                dose_vals.append(dose)
                health_vals.append(health)
                scenario_labels.append(scenario.replace('_', ' ').title())
        
        if dose_vals and health_vals:
            ax.scatter(dose_vals, health_vals, label="LSTM", 
                      color="C2", marker="^", s=100, alpha=0.7, edgecolors='black')
            # Add scenario labels
            for i, label in enumerate(scenario_labels):
                ax.annotate(label, (dose_vals[i], health_vals[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel("Total Dose (mL)")
        ax.set_ylabel("Mean Health Index")
        ax.set_title("Figure 10: Controller Trade-off (LSTM Performance Across Scenarios)", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add ideal direction arrow
        ax.annotate('Ideal', xy=(0.1, 0.95), xytext=(0.3, 0.8),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=12, color='red', fontweight='bold')
        
        fig.tight_layout()
        path = self.output_dir / "figure10_controller_tradeoff.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path
