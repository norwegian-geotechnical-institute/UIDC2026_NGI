#!/usr/bin/env python3
"""
Tunnel Gap Analysis Script

Analyzes the spacing between consecutive tunnel length measurements to identify
appropriate gap thresholds and detect outliers in the data.

Usage:
    python scripts/analyze_tunnel_gaps.py
"""

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from rich.console import Console
from rich.table import Table
from omegaconf import DictConfig, OmegaConf

from tbm_ml.schema_config import Config


def analyze_tunnel_spacing(df, console):
    """
    Analyze the spacing between consecutive tunnel length measurements.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data containing tunnel length
    console : Console
        Rich console for output
    """
    tunnel_length = df["Tunnellength [m]"]
    
    # Check for NaN values
    console.print(f"[yellow]Tunnel length column info:[/yellow]")
    console.print(f"  Total values: {len(tunnel_length)}")
    console.print(f"  Non-null values: {tunnel_length.notna().sum()}")
    console.print(f"  Null values: {tunnel_length.isna().sum()}")
    
    # Remove NaN values
    tunnel_length = tunnel_length.dropna()
    
    if len(tunnel_length) < 2:
        console.print("[bold red]Error: Not enough valid tunnel length data![/bold red]")
        return None
    
    # Calculate differences between consecutive measurements
    diffs = tunnel_length.diff().dropna()
    
    # Remove any NaN or infinite values from diffs
    diffs = diffs[np.isfinite(diffs)]
    
    if len(diffs) == 0:
        console.print("[bold red]Error: No valid spacing differences found![/bold red]")
        return None
    
    # Basic statistics
    console.print("\n[bold cyan]Tunnel Length Spacing Statistics[/bold cyan]\n")
    
    stats_table = Table(title="Basic Statistics", show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan", justify="left")
    stats_table.add_column("Value", style="green", justify="right")
    
    stats_table.add_row("Total data points", f"{len(df):,}")
    stats_table.add_row("Mean spacing", f"{diffs.mean():.4f} m")
    stats_table.add_row("Median spacing", f"{diffs.median():.4f} m")
    stats_table.add_row("Std deviation", f"{diffs.std():.4f} m")
    stats_table.add_row("Min spacing", f"{diffs.min():.4f} m")
    stats_table.add_row("Max spacing", f"{diffs.max():.4f} m")
    
    console.print(stats_table)
    
    # Percentile analysis
    console.print("\n[bold cyan]Percentile Analysis[/bold cyan]\n")
    
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
    percentile_table = Table(title="Spacing Percentiles", show_header=True, header_style="bold magenta")
    percentile_table.add_column("Percentile", style="cyan", justify="right")
    percentile_table.add_column("Spacing (m)", style="green", justify="right")
    percentile_table.add_column("Count Above", style="yellow", justify="right")
    
    for p in percentiles:
        value = np.percentile(diffs, p)
        count_above = (diffs > value).sum()
        percentile_table.add_row(f"{p}%", f"{value:.4f}", f"{count_above:,}")
    
    console.print(percentile_table)
    
    # Identify potential gap thresholds
    console.print("\n[bold cyan]Suggested Gap Thresholds[/bold cyan]\n")
    
    # Use IQR method for outlier detection
    q1 = diffs.quantile(0.25)
    q3 = diffs.quantile(0.75)
    iqr = q3 - q1
    
    # Various multipliers for different sensitivity levels
    thresholds_table = Table(title="Threshold Options", show_header=True, header_style="bold magenta")
    thresholds_table.add_column("Method", style="cyan", justify="left", width=30)
    thresholds_table.add_column("Threshold (m)", style="green", justify="right")
    thresholds_table.add_column("Gaps Detected", style="yellow", justify="right")
    thresholds_table.add_column("Total Gap Length (m)", style="red", justify="right")
    
    methods = [
        ("Mean + 2*Std", diffs.mean() + 2*diffs.std()),
        ("Mean + 3*Std", diffs.mean() + 3*diffs.std()),
        ("Q3 + 1.5*IQR (Outliers)", q3 + 1.5*iqr),
        ("Q3 + 3*IQR (Extreme outliers)", q3 + 3*iqr),
        ("95th percentile", np.percentile(diffs, 95)),
        ("99th percentile", np.percentile(diffs, 99)),
        ("99.5th percentile", np.percentile(diffs, 99.5)),
    ]
    
    for method_name, threshold in methods:
        gaps = diffs[diffs > threshold]
        num_gaps = len(gaps)
        total_gap_length = gaps.sum()
        thresholds_table.add_row(
            method_name,
            f"{threshold:.4f}",
            f"{num_gaps}",
            f"{total_gap_length:.2f}"
        )
    
    console.print(thresholds_table)
    
    # Show largest gaps
    console.print("\n[bold cyan]Top 20 Largest Gaps[/bold cyan]\n")
    
    top_gaps = diffs.nlargest(20).reset_index()
    top_gaps.columns = ['Index', 'Gap Size (m)']
    top_gaps['Start Position (m)'] = tunnel_length.iloc[top_gaps['Index'] - 1].values
    top_gaps['End Position (m)'] = tunnel_length.iloc[top_gaps['Index']].values
    
    gaps_table = Table(title="Largest Gaps in Data", show_header=True, header_style="bold magenta")
    gaps_table.add_column("Rank", style="cyan", justify="right")
    gaps_table.add_column("Gap Size (m)", style="green", justify="right")
    gaps_table.add_column("Start (m)", style="yellow", justify="right")
    gaps_table.add_column("End (m)", style="yellow", justify="right")
    
    for idx, row in top_gaps.iterrows():
        gaps_table.add_row(
            f"{idx + 1}",
            f"{row['Gap Size (m)']:.4f}",
            f"{row['Start Position (m)']:.2f}",
            f"{row['End Position (m)']:.2f}"
        )
    
    console.print(gaps_table)
    
    return diffs


def create_visualizations(diffs, output_dir="plots"):
    """
    Create visualization plots for spacing analysis.
    
    Parameters:
    -----------
    diffs : pd.Series
        Differences between consecutive tunnel measurements
    output_dir : str
        Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Tunnel Length Spacing Analysis", fontsize=16, fontweight='bold')
    
    # 1. Histogram of all spacings
    ax1 = axes[0, 0]
    ax1.hist(diffs, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel("Spacing between consecutive measurements (m)", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Distribution of All Spacings", fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.axvline(diffs.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {diffs.mean():.4f} m')
    ax1.axvline(diffs.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {diffs.median():.4f} m')
    ax1.legend()
    
    # 2. Box plot
    ax2 = axes[0, 1]
    bp = ax2.boxplot(diffs, vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    ax2.set_ylabel("Spacing (m)", fontsize=11)
    ax2.set_title("Box Plot (showing outliers)", fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_xticklabels(['Spacing'])
    
    # 3. Zoomed histogram (excluding extreme outliers)
    ax3 = axes[1, 0]
    # Show up to 99th percentile
    p99 = np.percentile(diffs, 99)
    diffs_zoomed = diffs[diffs <= p99]
    ax3.hist(diffs_zoomed, bins=100, color='coral', edgecolor='black', alpha=0.7)
    ax3.set_xlabel("Spacing between consecutive measurements (m)", fontsize=11)
    ax3.set_ylabel("Frequency", fontsize=11)
    ax3.set_title("Distribution of Spacings (up to 99th percentile)", fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.axvline(diffs_zoomed.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {diffs_zoomed.mean():.4f} m')
    ax3.axvline(diffs_zoomed.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {diffs_zoomed.median():.4f} m')
    ax3.legend()
    
    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    sorted_diffs = np.sort(diffs)
    cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs) * 100
    ax4.plot(sorted_diffs, cumulative, linewidth=2, color='darkgreen')
    ax4.set_xlabel("Spacing (m)", fontsize=11)
    ax4.set_ylabel("Cumulative Percentage (%)", fontsize=11)
    ax4.set_title("Cumulative Distribution", fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)
    ax4.set_xscale('log')
    
    # Add percentile lines
    for p in [90, 95, 99]:
        val = np.percentile(diffs, p)
        ax4.axvline(val, color='red', linestyle='--', alpha=0.5)
        ax4.text(val, 50, f'{p}%', rotation=90, verticalalignment='center')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / "tunnel_spacing_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main analysis function."""
    
    console = Console()
    
    console.print("\n[bold magenta]🔍 Tunnel Gap Analysis[/bold magenta]\n")
    
    # Parse config
    pcfg = Config(**OmegaConf.to_object(cfg))
    
    # Load data
    console.print("[bold green]Loading data...[/bold green]")
    df = pd.read_csv(pcfg.dataset.path_raw_dataset)
    
    # Note: Column names in CSV have varying whitespace, but "Tunnellength [m]" doesn't have leading space
    
    console.print(f"Loaded {len(df):,} data points\n")
    
    # Analyze spacing
    diffs = analyze_tunnel_spacing(df, console)
    
    # Create visualizations
    console.print("\n[bold blue]📊 Creating visualizations...[/bold blue]")
    plot_path = create_visualizations(diffs)
    console.print(f"[bold green]✓[/bold green] Saved plot to: {plot_path}\n")
    
    console.print("[bold green]✅ Analysis complete![/bold green]\n")


if __name__ == "__main__":
    main()
