"""
Matplotlib visualization for policy learning experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


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
