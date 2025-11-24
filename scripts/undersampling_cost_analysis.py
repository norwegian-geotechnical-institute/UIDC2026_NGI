#!/usr/bin/env python3
"""
Undersampling Cost Analysis Script

Performs grid search over undersampling fractions to analyze the relationship
between class balance and prediction cost. Generates plots and saves confusion
matrices for analysis.

Usage:
    python scripts/undersampling_cost_analysis.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import track
from rich.table import Table

sys.path.append(str(Path(__file__).parent.parent / "src"))

from tbm_ml.plotting import plot_tbm_confusion_matrix
from tbm_ml.schema_config import Config
from tbm_ml.train_eval_funcs import (
    calculate_prediction_costs,
    load_data,
    train_predict,
)


def create_results_directory(base_path: str = "analyses/undersampling_analysis") -> Path:
    """Create a timestamped directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(base_path) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def analyze_undersampling_fraction(
    undersample_fraction: float,
    model_name: str,
    model_params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    oversample_level: int,
    cost_matrix: dict,
    class_mapping: dict,
    random_seed: int,
) -> dict:
    """
    Train and evaluate model with a specific undersampling fraction.
    
    Parameters:
    -----------
    undersample_fraction : float
        Ratio of majority to minority class (e.g., 2.0 means 2:1 ratio)
    
    Returns:
    --------
    dict with metrics and predictions
    """
    # Calculate undersample_level based on fraction
    minority_count = int(y_train.sum())  # Assuming binary with 1 as minority
    undersample_level = int(minority_count * undersample_fraction)
    
    # Train and predict
    y_pred = train_predict(
        model_name=model_name,
        model_params=model_params,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        undersample_level=undersample_level,
        oversample_level=oversample_level,
        save_model=False,
        random_seed=random_seed,
    )
    
    # Calculate cost metrics
    cost_metrics = calculate_prediction_costs(
        y_test,
        y_pred,
        tn_cost=cost_matrix["tn_cost"],
        fp_cost=cost_matrix["fp_cost"],
        fn_cost=cost_matrix["fn_cost"],
        tp_cost=cost_matrix["tp_cost"],
    )
    
    # Calculate accuracy metrics
    total_samples = len(y_test)
    tn = cost_metrics["tn_count"]
    fp = cost_metrics["fp_count"]
    fn = cost_metrics["fn_count"]
    tp = cost_metrics["tp_count"]
    
    accuracy = (tp + tn) / total_samples
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    
    return {
        "undersample_fraction": undersample_fraction,
        "undersample_level": undersample_level,
        "y_pred": y_pred,
        "cost_total": cost_metrics["total_cost"],
        "cost_average": cost_metrics["average_cost"],
        "tn_count": cost_metrics["tn_count"],
        "fp_count": cost_metrics["fp_count"],
        "fn_count": cost_metrics["fn_count"],
        "tp_count": cost_metrics["tp_count"],
        "tn_total_cost": cost_metrics["tn_total_cost"],
        "fp_total_cost": cost_metrics["fp_total_cost"],
        "fn_total_cost": cost_metrics["fn_total_cost"],
        "tp_total_cost": cost_metrics["tp_total_cost"],
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def plot_cost_vs_undersampling(
    results: list[dict],
    results_dir: Path,
    cost_matrix: dict,
    colors: dict[str, str],
) -> tuple[Path, Path]:
    """Create comprehensive plots of cost vs undersampling fraction."""
    
    fractions = [r["undersample_fraction"] for r in results]
    total_costs = [r["cost_total"] for r in results]
    avg_costs = [r["cost_average"] for r in results]
    
    # Extract confusion matrix components
    fp_costs = [r["fp_total_cost"] for r in results]
    fn_costs = [r["fn_total_cost"] for r in results]
    tp_costs = [r["tp_total_cost"] for r in results]
    tn_costs = [r["tn_total_cost"] for r in results]
    
    # ============================================================================
    # MAIN FIGURE: Total Cost and Cost Component Breakdown (stacked vertically)
    # ============================================================================
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle("Undersampling Fraction vs Prediction Cost Analysis", fontsize=16, fontweight="bold")
    
    # Plot 1: Total Cost
    ax1 = axes[0]
    ax1.plot(fractions, total_costs, marker="o", linewidth=2, markersize=6, color="#2E86AB")
    ax1.set_xlabel("Undersampling Fraction, Majority:Minority (x:1)", fontsize=11)
    ax1.set_ylabel("Total Cost", fontsize=11)
    ax1.set_title("Total Cost vs Undersampling Fraction", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Mark minimum
    min_idx = np.argmin(total_costs)
    ax1.scatter([fractions[min_idx]], [total_costs[min_idx]], 
                color="red", s=100, zorder=5, label=f"Min: {fractions[min_idx]:.1f}")
    ax1.legend()
    
    # Plot 2: Cost Components Breakdown (Stacked Area)
    ax2 = axes[1]
    ax2.stackplot(fractions, tn_costs, fp_costs, fn_costs, tp_costs,
                  labels=["TN Cost", "FP Cost", "FN Cost", "TP Cost"],
                  colors=[colors['tn_color'], colors['fp_color'], colors['fn_color'], colors['tp_color']],
                  alpha=0.8)
    ax2.set_xlabel("Undersampling Fraction, Majority:Minority (x:1)", fontsize=11)
    ax2.set_ylabel("Cost Components", fontsize=11)
    ax2.set_title("Cost Component Breakdown", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save main plot
    main_plot_path = results_dir / "cost_vs_undersampling.png"
    plt.savefig(main_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # ============================================================================
    # SEPARATE FIGURE: All Confusion Matrix Counts (Twin Y-Axes)
    # ============================================================================
    fig_errors, ax_left = plt.subplots(1, 1, figsize=(10, 6))
    
    fp_counts = [r["fp_count"] for r in results]
    fn_counts = [r["fn_count"] for r in results]
    tp_counts = [r["tp_count"] for r in results]
    tn_counts = [r["tn_count"] for r in results]
    
    # Get colors from colors parameter (passed from config)
    fp_color = colors.get('fp_color', '#FFD700')
    fn_color = colors.get('fn_color', '#FF6B6B')
    tp_color = colors.get('tp_color', '#4ECDC4')
    tn_color = colors.get('tn_color', '#90EE90')
    
    # Create twin axis for True Negatives
    ax_right = ax_left.twinx()
    
    # Plot FP, FN, TP on left axis
    line_fp = ax_left.plot(fractions, fp_counts, marker="s", linewidth=2, markersize=6, 
                            label="False Positives", color=fp_color)
    line_fn = ax_left.plot(fractions, fn_counts, marker="^", linewidth=2, markersize=6,
                            label="False Negatives", color=fn_color)
    line_tp = ax_left.plot(fractions, tp_counts, marker="d", linewidth=2, markersize=6,
                            label="True Positives", color=tp_color)
    
    # Plot TN on right axis
    line_tn = ax_right.plot(fractions, tn_counts, marker="o", linewidth=2, markersize=6, 
                             label="True Negatives", color=tn_color)
    
    # Configure left axis
    ax_left.set_xlabel("Undersampling Fraction, Majority:Minority (x:1)", fontsize=12)
    ax_left.set_ylabel("Count (FP, FN, TP)", fontsize=12, color="black")
    ax_left.tick_params(axis='y', labelcolor="black")
    ax_left.grid(True, alpha=0.3)
    
    # Set left axis limits: max value at ~40% of axis height, add small margin at bottom only
    left_max = max(max(fp_counts), max(fn_counts), max(tp_counts))
    left_margin = left_max * 0.05  # 5% margin at bottom
    ax_left.set_ylim(-left_margin, left_max * 2.5)
    
    # Configure right axis
    ax_right.set_ylabel("Count (TN)", fontsize=12, color=tn_color)
    ax_right.tick_params(axis='y', labelcolor=tn_color)
    
    # Set right axis limits: min value at ~40% from bottom, add small margin at top only
    tn_min = min(tn_counts)
    tn_max = max(tn_counts)
    tn_range = tn_max - tn_min
    tn_margin = tn_range * 0.05  # 5% margin at top
    ax_right.set_ylim(tn_min - tn_range * 1.5, tn_max + tn_margin)
    
    # Title
    ax_left.set_title("Confusion Matrix Counts vs Undersampling Fraction", fontsize=14, fontweight="bold")
    
    # Combine legends from both axes
    lines = line_fp + line_fn + line_tp + line_tn
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, fontsize=11, loc="center right")
    
    plt.tight_layout()
    
    # Save error counts plot
    error_plot_path = results_dir / "error_counts_vs_undersampling.png"
    plt.savefig(error_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return main_plot_path, error_plot_path


def plot_accuracy_metrics(
    results: list[dict],
    results_dir: Path,
) -> Path:
    """Create plot of accuracy metrics vs undersampling fraction."""
    
    fractions = [r["undersample_fraction"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    balanced_accuracies = [r["balanced_accuracy"] for r in results]
    sensitivities = [r["sensitivity"] for r in results]
    
    # Find minimum cost point
    costs = [r["cost_average"] for r in results]
    min_cost_idx = np.argmin(costs)
    min_cost_fraction = fractions[min_cost_idx]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot metrics
    ax.plot(fractions, accuracies, linewidth=2, 
            label="Accuracy", color="#2E86AB")
    ax.plot(fractions, balanced_accuracies, linewidth=2,
            label="Balanced Accuracy", color="#A23B72")
    ax.plot(fractions, sensitivities, linewidth=2,
            label="Recall", color="#F18F01")
    
    # Mark maximum points for each metric
    max_acc_idx = np.argmax(accuracies)
    max_bal_acc_idx = np.argmax(balanced_accuracies)
    max_sens_idx = np.argmax(sensitivities)
    
    ax.scatter([fractions[max_acc_idx]], [accuracies[max_acc_idx]], 
              color="#2E86AB", s=100, zorder=5, marker="*", edgecolors="black", linewidths=1.5)
    ax.scatter([fractions[max_bal_acc_idx]], [balanced_accuracies[max_bal_acc_idx]], 
              color="#A23B72", s=100, zorder=5, marker="*", edgecolors="black", linewidths=1.5)
    ax.scatter([fractions[max_sens_idx]], [sensitivities[max_sens_idx]], 
              color="#F18F01", s=100, zorder=5, marker="*", edgecolors="black", linewidths=1.5)
    
    # Add vertical line at minimum cost
    ax.axvline(x=min_cost_fraction, color="red", linestyle="--", alpha=0.5, linewidth=1.5,
               label="Minimum cost")
    
    # Add star symbol to legend
    from matplotlib.lines import Line2D
    legend_elements = ax.get_legend_handles_labels()
    star_marker = Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
                        markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                        label='Maximum', linestyle='None')
    handles, labels = legend_elements
    handles.append(star_marker)
    
    ax.set_xlabel("Undersampling Fraction, Majority:Minority (x:1)", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Accuracy Metrics vs Undersampling Fraction", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(handles=handles, fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    accuracy_plot_path = results_dir / "accuracy_metrics_vs_undersampling.png"
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return accuracy_plot_path


def save_confusion_matrices(
    results: list[dict],
    y_test: pd.Series,
    class_mapping: dict,
    results_dir: Path,
    cell_colors: dict[str, str],
    n_points: int = 5,
) -> list[Path]:
    """
    Save confusion matrices for n evenly spaced points along the curve.
    
    Parameters:
    -----------
    n_points : int
        Number of confusion matrices to save (evenly distributed)
    """
    # Select n_points evenly distributed along the results
    n_results = len(results)
    if n_points > n_results:
        indices = list(range(n_results))
    else:
        indices = [int(i * (n_results - 1) / (n_points - 1)) for i in range(n_points)]
    
    cm_dir = results_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    
    saved_paths = []
    
    for idx in indices:
        result = results[idx]
        fraction = result["undersample_fraction"]
        y_pred = result["y_pred"]
        
        # Create confusion matrix plot
        fig = plot_tbm_confusion_matrix(
            y_test, 
            y_pred, 
            class_mapping, 
            normalize="true", 
            show_percentages=True,
            cell_colors=cell_colors
        )
        
        # Add title with cost information
        fig.suptitle(
            f"Undersampling Fraction: {fraction:.1f}:1\n"
            f"Total Cost: {result['cost_total']:.2f} | Avg Cost: {result['cost_average']:.4f}",
            fontsize=12,
            fontweight="bold",
            y=1.05,
        )
        
        # Save with adjusted layout to prevent title overlap
        cm_path = cm_dir / f"cm_fraction_{fraction:.1f}.png"
        fig.savefig(cm_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        
        saved_paths.append(cm_path)
    
    return saved_paths


def save_results_table(results: list[dict], results_dir: Path) -> Path:
    """Save detailed results table to CSV."""
    df = pd.DataFrame(results)
    
    # Drop y_pred column (contains arrays, not suitable for CSV)
    if "y_pred" in df.columns:
        df = df.drop(columns=["y_pred"])
    
    csv_path = results_dir / "results_table.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function for undersampling cost analysis."""
    
    console = Console()
    
    console.print("\n[bold magenta]🔍 Undersampling Cost Analysis[/bold magenta]\n")
    
    # Parse config
    pcfg = Config(**OmegaConf.to_object(cfg))
    
    # Create results directory
    results_dir = create_results_directory()
    console.print(f"[bold blue]📁 Results directory:[/bold blue] {results_dir}\n")
    
    # Load data
    console.print("[bold green]Loading data...[/bold green]")
    X_train, X_test, y_train, y_test = load_data(
        pcfg.dataset.path_model_ready_train,
        pcfg.dataset.path_model_ready_test,
        pcfg.experiment.label,
        pcfg.experiment.features,
    )
    
    minority_count = int(y_train.sum())
    majority_count = len(y_train) - minority_count
    
    console.print(f"  Training samples: {len(X_train)}")
    console.print(f"  Test samples: {len(X_test)}")
    console.print(f"  Minority class count: {minority_count}")
    console.print(f"  Majority class count: {majority_count}")
    console.print(f"  Original ratio: {majority_count/minority_count:.2f}:1\n")
    
    # Prepare cost matrix
    cost_matrix = {
        "tn_cost": pcfg.experiment.cost_matrix.tn_cost,
        "fp_cost": pcfg.experiment.cost_matrix.fp_cost,
        "fn_cost": pcfg.experiment.cost_matrix.fn_cost,
        "tp_cost": pcfg.experiment.cost_matrix.tp_cost,
    }
    
    # Prepare color configuration
    colors = {
        'tn_color': pcfg.experiment.confusion_matrix_colors.tn_color,
        'fp_color': pcfg.experiment.confusion_matrix_colors.fp_color,
        'fn_color': pcfg.experiment.confusion_matrix_colors.fn_color,
        'tp_color': pcfg.experiment.confusion_matrix_colors.tp_color,
    }
    
    console.print("[bold cyan]Cost Matrix:[/bold cyan]")
    console.print(f"  TN Cost: {cost_matrix['tn_cost']}")
    console.print(f"  FP Cost: {cost_matrix['fp_cost']}")
    console.print(f"  FN Cost: {cost_matrix['fn_cost']}")
    console.print(f"  TP Cost: {cost_matrix['tp_cost']}\n")
    
    # Define undersampling fractions to test
    min_fraction = 0.5
    max_fraction = 8
    num_points = 30
    
    undersampling_fractions = np.linspace(min_fraction, max_fraction, num=num_points)
    
    console.print(f"[bold yellow]Testing {len(undersampling_fractions)} undersampling fractions[/bold yellow]")
    console.print(f"  Range: {min_fraction:.1f}:1 to {max_fraction:.1f}:1\n")
    
    # Run grid search
    results = []
    for fraction in track(undersampling_fractions, description="Running grid search..."):
        result = analyze_undersampling_fraction(
            undersample_fraction=fraction,
            model_name=pcfg.model.name,
            model_params=pcfg.model.params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            oversample_level=pcfg.experiment.oversample_level,
            cost_matrix=cost_matrix,
            class_mapping=pcfg.experiment.tbm_classification,
            random_seed=pcfg.experiment.seed,
    )
        results.append(result)
    
    # Find optimal fraction
    optimal_idx = np.argmin([r["cost_average"] for r in results])
    optimal_result = results[optimal_idx]
    
    console.print("\n[bold green]✅ Grid search complete![/bold green]\n")
    
    # Display summary table
    table = Table(title="Top 5 Undersampling Configurations by Average Cost")
    table.add_column("Rank", style="cyan", justify="center")
    table.add_column("Fraction", style="magenta")
    table.add_column("Avg Cost", style="green")
    table.add_column("Total Cost", style="yellow")
    table.add_column("FP", style="red")
    table.add_column("FN", style="red")
    
    sorted_results = sorted(results, key=lambda x: x["cost_average"])
    for i, result in enumerate(sorted_results, 1):
        table.add_row(
            str(i),
            f"{result['undersample_fraction']:.1f}:1",
            f"{result['cost_average']:.4f}",
            f"{result['cost_total']:.2f}",
            str(result['fp_count']),
            str(result['fn_count']),
        )
    
    console.print(table)
    
    console.print(f"\n[bold green]🎯 Optimal Undersampling Fraction:[/bold green] {optimal_result['undersample_fraction']:.1f}:1")
    console.print(f"   Average Cost: {optimal_result['cost_average']:.4f}")
    console.print(f"   Total Cost: {optimal_result['cost_total']:.2f}\n")
    
    # Generate plots
    console.print("[bold blue]📊 Generating plots...[/bold blue]")
    main_plot_path, error_plot_path = plot_cost_vs_undersampling(results, results_dir, cost_matrix, colors)
    console.print(f"  Saved: {main_plot_path}")
    console.print(f"  Saved: {error_plot_path}")
    
    accuracy_plot_path = plot_accuracy_metrics(results, results_dir)
    console.print(f"  Saved: {accuracy_plot_path}")
    
    # Save confusion matrices for key points
    n_cm_points = len(undersampling_fractions)  # Save confusion matrices for all tested fractions
    console.print(f"\n[bold blue]💾 Saving {n_cm_points} confusion matrices...[/bold blue]")
    cm_paths = save_confusion_matrices(
        results, y_test, pcfg.experiment.tbm_classification, results_dir, colors, n_points=n_cm_points
    )
    for path in cm_paths:
        console.print(f"  Saved: {path}")
    
    # Save results table
    console.print("\n[bold blue]📋 Saving results table...[/bold blue]")
    csv_path = save_results_table(results, results_dir)
    console.print(f"  Saved: {csv_path}")
    
    console.print("\n[bold green]🎉 Analysis complete![/bold green]")
    console.print(f"[bold blue]All results saved to:[/bold blue] {results_dir}\n")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"

    main()
