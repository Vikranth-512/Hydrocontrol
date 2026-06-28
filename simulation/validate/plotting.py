"""
Standardized plotting utilities for the validation framework.
All plots are publication-quality with consistent styling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.figsize": (10, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_time_series(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    ylabel: str,
    save_path: Path,
    xlabel: str = "Time (min)",
    x_col: str = "time_min",
    logy: bool = False,
):
    """Standard multi-line time-series plot."""
    fig, ax = plt.subplots()
    for col in columns:
        if col in df.columns:
            ax.plot(df[x_col], df[col], label=col)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_multi_panel(
    df: pd.DataFrame,
    panels: List[Tuple[List[str], str]],
    title: str,
    save_path: Path,
    x_col: str = "time_min",
):
    """Multi-panel subplot sharing x-axis. Each panel is (columns, ylabel)."""
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (cols, ylabel) in zip(axes, panels):
        for col in cols:
            if col in df.columns:
                ax.plot(df[x_col], df[col], label=col)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (min)")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_xy(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
    vlines: Optional[List[Tuple[float, str, str]]] = None,
    label: str = "",
):
    """Simple x-y scatter/line plot with optional vertical reference lines."""
    fig, ax = plt.subplots()
    ax.plot(x, y, "o-", markersize=3, label=label or ylabel)
    if vlines:
        for xv, color, lbl in vlines:
            ax.axvline(xv, color=color, linestyle="--", label=lbl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_conservation_error(
    time: np.ndarray,
    error: np.ndarray,
    save_path: Path,
):
    """Specialized plot for mass conservation error over time."""
    fig, ax = plt.subplots()
    ax.plot(time, error, color="red", linewidth=0.8)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.set_title("Mass Conservation Error Over Time")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Absolute Error (mass units)")
    if np.max(np.abs(error)) > 1e-15:
        ax.set_yscale("symlog", linthresh=1e-13)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_tornado(
    param_names: List[str],
    sensitivities: np.ndarray,
    metric_name: str,
    save_path: Path,
):
    """Tornado (horizontal bar) chart for parameter sensitivity."""
    idx = np.argsort(np.abs(sensitivities))
    fig, ax = plt.subplots(figsize=(8, max(4, len(param_names) * 0.35)))
    colors = ["#d9534f" if s < 0 else "#5cb85c" for s in sensitivities[idx]]
    ax.barh(range(len(param_names)), sensitivities[idx], color=colors)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels([param_names[i] for i in idx])
    ax.set_xlabel(f"Delta {metric_name} (fractional)")
    ax.set_title(f"Parameter Sensitivity - {metric_name}")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
