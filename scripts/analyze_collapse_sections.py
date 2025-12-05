#!/usr/bin/env python3
"""
Analyze collapse sections in the TBM dataset.

This script provides detailed analysis of continuous collapse sections,
including distribution, sizes, and recommendations for train/test splitting.

Usage:
    python scripts/analyze_collapse_sections.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_collapse_sections(data_path: str = "data/model_ready/dataset_total.csv"):
    """Analyze collapse sections in the dataset."""
    
    # Load data
    df = pd.read_csv(data_path)
    
    print("=" * 80)
    print("COLLAPSE SECTION ANALYSIS")
    print("=" * 80)
    
    # Basic statistics
    total_samples = len(df)
    collapse_samples = df['collapse'].sum()
    non_collapse_samples = (df['collapse'] == 0).sum()
    
    print(f"\nDataset Overview:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Collapse samples: {collapse_samples:,} ({collapse_samples/total_samples*100:.2f}%)")
    print(f"  Non-collapse samples: {non_collapse_samples:,} ({non_collapse_samples/total_samples*100:.2f}%)")
    
    # Collapse section analysis
    collapse_df = df[df['collapse_section'] > 0]
    n_sections = collapse_df['collapse_section'].nunique()
    
    print(f"\nCollapse Sections:")
    print(f"  Total continuous collapse sections: {n_sections}")
    
    # Section sizes
    section_sizes = collapse_df.groupby('collapse_section').size().sort_values(ascending=False)
    
    print(f"\nSection Size Statistics:")
    print(f"  Mean: {section_sizes.mean():.1f} samples")
    print(f"  Median: {section_sizes.median():.1f} samples")
    print(f"  Min: {section_sizes.min()} samples (Section {section_sizes.idxmin()})")
    print(f"  Max: {section_sizes.max()} samples (Section {section_sizes.idxmax()})")
    print(f"  Std Dev: {section_sizes.std():.1f} samples")
    
    print(f"\nAll Collapse Sections (sorted by size):")
    print("-" * 40)
    for section_id, size in section_sizes.items():
        print(f"  Section {section_id:2d}: {size:3d} samples")
    
    # Visualization
    print("\n" + "=" * 80)
    print("Generating visualization...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Bar chart of section sizes
    ax1 = axes[0]
    section_sizes_sorted = section_sizes.sort_index()
    ax1.bar(section_sizes_sorted.index, section_sizes_sorted.values, 
            color='#EF4927', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Collapse Section ID', fontsize=11)
    ax1.set_ylabel('Number of Samples', fontsize=11)
    ax1.set_title('Collapse Section Sizes', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(range(1, n_sections + 1))
    
    # Add mean line
    ax1.axhline(y=section_sizes.mean(), color='blue', linestyle='--', 
                linewidth=2, label=f'Mean: {section_sizes.mean():.1f}')
    ax1.legend(fontsize=10)
    
    # Plot 2: Cumulative distribution
    ax2 = axes[1]
    cumsum_sorted = section_sizes_sorted.cumsum()
    ax2.plot(cumsum_sorted.index, cumsum_sorted.values, 
             marker='o', linewidth=2, markersize=6, color='#1B9AD7')
    ax2.fill_between(cumsum_sorted.index, 0, cumsum_sorted.values, alpha=0.3, color='#1B9AD7')
    ax2.set_xlabel('Collapse Section ID', fontsize=11)
    ax2.set_ylabel('Cumulative Collapse Samples', fontsize=11)
    ax2.set_title('Cumulative Distribution of Collapse Samples', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, n_sections + 1))
    
    # Add 75% line for train/test split reference
    target_75 = collapse_samples * 0.75
    ax2.axhline(y=target_75, color='red', linestyle='--', 
                linewidth=2, label=f'75% split: {target_75:.0f} samples')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("analyses") / "collapse_section_analysis.png"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Save detailed report
    report_path = Path("analyses") / "collapse_section_report.csv"
    section_report = pd.DataFrame({
        'collapse_section': section_sizes.index,
        'n_samples': section_sizes.values,
        'cumulative_samples': section_sizes.sort_index().cumsum().values,
        'percentage_of_total_collapses': (section_sizes.values / collapse_samples * 100).round(2)
    })
    section_report.to_csv(report_path, index=False)
    print(f"Detailed report saved to: {report_path}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    analyze_collapse_sections()
