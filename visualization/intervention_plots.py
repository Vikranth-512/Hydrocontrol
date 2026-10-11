"""
Publication-quality figures for reviewer-response intervention evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_intervention_confusion_matrix(
    cm_counts: Dict[str, int],
    output_path: Union[str, Path],
    title: str = "Binary Intervention Confusion Matrix",
) -> Path:
    """
    Generate publication-quality confusion matrix heatmap.
    
    cm_counts must contain:
    - true_negatives (TN)
    - false_positives (FP)
    - false_negatives (FN)
    - true_positives (TP)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    tn = cm_counts["true_negatives"]
    fp = cm_counts["false_positives"]
    fn = cm_counts["false_negatives"]
    tp = cm_counts["true_positives"]
    
    matrix = np.array([[tn, fp], [fn, tp]])
    total = np.sum(matrix)
    norm_matrix = matrix / total * 100.0
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    # Custom color palette (coolwarm / Blues)
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Sample Count", fontsize=11, fontweight="medium")
    
    classes = ["No Action", "Intervention"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11, fontweight="medium")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=11, fontweight="medium")
    
    # Text labels in each cell
    thresh = matrix.max() / 2.0
    labels = [
        [f"TN\n{tn:,}\n({norm_matrix[0, 0]:.1f}%)", f"FP\n{fp:,}\n({norm_matrix[0, 1]:.1f}%)"],
        [f"FN\n{fn:,}\n({norm_matrix[1, 0]:.1f}%)", f"TP\n{tp:,}\n({norm_matrix[1, 1]:.1f}%)"],
    ]
    
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > thresh else "black"
            ax.text(
                j, i, labels[i][j],
                ha="center", va="center",
                color=color, fontsize=12, fontweight="bold",
            )
            
    ax.set_ylabel("True Supervisory Action", fontsize=12, fontweight="bold")
    ax.set_xlabel("LSTM Predicted Action", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_scenario_intervention_metrics(
    scenario_df: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "Intervention Metrics Across Operating Regimes",
) -> Path:
    """
    Generate publication-grade bar chart comparing Intervention F1 and Overall Accuracy
    across evaluated operating regimes.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = scenario_df.dropna(subset=["Intervention F1"]).copy()
    if df.empty:
        df = scenario_df.copy()
        
    regimes = df["Regime"].tolist()
    int_f1 = df["Intervention F1"].fillna(0.0).values
    acc = df["Overall Accuracy"].fillna(0.0).values * 100.0 if "Overall Accuracy" in df else None
    
    x = np.arange(len(regimes))
    width = 0.35 if acc is not None else 0.5
    
    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)
    
    color_f1 = "#1f77b4"
    rects1 = ax1.bar(
        x - (width / 2 if acc is not None else 0),
        int_f1,
        width,
        label="Intervention F1 Score",
        color=color_f1,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
    )
    ax1.set_ylabel("Intervention F1 Score", fontsize=11, fontweight="bold", color=color_f1)
    ax1.set_ylim(0.0, 1.05)
    ax1.tick_params(axis="y", labelcolor=color_f1)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Value annotations on bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )
        
    if acc is not None:
        ax2 = ax1.twinx()
        color_acc = "#2ca02c"
        rects2 = ax2.bar(
            x + width / 2,
            acc,
            width,
            label="Overall Accuracy (%)",
            color=color_acc,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )
        ax2.set_ylabel("Overall Accuracy (%)", fontsize=11, fontweight="bold", color=color_acc)
        ax2.set_ylim(0, 110)
        ax2.tick_params(axis="y", labelcolor=color_acc)
        ax2.grid(False)
        
        for rect in rects2:
            height = rect.get_height()
            ax2.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )
            
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes, fontsize=10, fontweight="bold", rotation=15, ha="right")
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=12)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
