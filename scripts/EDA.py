import hydra
import matplotlib.pyplot as plt
import pandas as pd

from rich.console import Console
from rich.table import Table
from omegaconf import DictConfig, OmegaConf

from tbm_ml.schema_config import Config
from tbm_ml.plotting import pairplot, plot_tbm_parameters


def calculate_data_summary(df, gap_threshold=1.0):
    """
    Calculate summary statistics about the tunnel data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data containing tunnel information
    gap_threshold : float
        Minimum gap size to count as a gap
    
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    tunnel_length = df["Tunnellength [m]"]
    collapse = df["collapse"]
    
    # Total tunnel length
    total_length = tunnel_length.max() - tunnel_length.min()
    
    # Detect gaps
    gaps = []
    for i in range(len(tunnel_length) - 1):
        diff = tunnel_length.iloc[i + 1] - tunnel_length.iloc[i]
        if diff > gap_threshold:
            gaps.append((tunnel_length.iloc[i], tunnel_length.iloc[i + 1], diff))
    
    # Calculate data coverage
    total_gap_length = sum(gap[2] for gap in gaps)
    data_coverage = total_length - total_gap_length
    coverage_percentage = (data_coverage / total_length * 100) if total_length > 0 else 0
    
    # Collapse statistics
    num_collapse_points = int(collapse.sum())
    num_non_collapse_points = int((collapse == 0).sum())
    
    # Estimate collapse section length (assuming each data point represents some length)
    # Use median spacing between consecutive points
    spacings = tunnel_length.diff().dropna()
    median_spacing = spacings[spacings <= gap_threshold].median()
    collapse_length = num_collapse_points * median_spacing if not pd.isna(median_spacing) else 0
    
    return {
        "total_length": total_length,
        "data_coverage": data_coverage,
        "coverage_percentage": coverage_percentage,
        "num_gaps": len(gaps),
        "num_collapse_points": num_collapse_points,
        "num_non_collapse_points": num_non_collapse_points,
        "collapse_length": collapse_length,
        "num_data_points": len(df),
    }


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pcfg = Config(**OmegaConf.to_object(cfg))
    console = Console()
    console.print(pcfg)

    console.print("[bold green]Loading data...[/bold green]")
    # Load data
    df = pd.read_csv(pcfg.dataset.path_raw_dataset)

    # rename column names to be more readable
    rename_dict = {
        'p': 'penetration\n[mm/rev]',
        'Pr': 'advance rate\n[mm/min]',
        'RPM': 'cutterhead rotations\n[rpm]',
        'Total T': 'cutterhead torque\n[kNm]',
        'Total F': 'thrust\n[kN]'
        }
    df.rename(rename_dict, axis=1, inplace=True)

    # Plot lineplot
    console.print("[bold green]Plotting TBM parameters...[/bold green]")
    gap_threshold = 5.0
    fig1: plt.Figure = plot_tbm_parameters(df, gap_threshold=gap_threshold, show_gap_markers=False)
    fig1.tight_layout()
    fig1.savefig("plots/tbm_parameters.png")

    # Pairplot
    console.print("[bold green]Plotting pairplot...[/bold green]")
    fig2: plt.Figure = pairplot(df[list(rename_dict.values()) + ['collapse']],
                                parameters=list(rename_dict.values()),
                                target=pcfg.experiment.label,
                                figsize=(18, 18))
    fig2.tight_layout()
    fig2.savefig("plots/pairplot.png")

    # Calculate and display data summary
    console.print("\n[bold cyan]Data Summary[/bold cyan]")
    summary = calculate_data_summary(df, gap_threshold=gap_threshold)
    
    table = Table(title="Tunnel Data Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", justify="left", width=35)
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Total Tunnel Length", f"{summary['total_length']:.1f} m")
    table.add_row("Data Coverage", f"{summary['data_coverage']:.1f} m")
    table.add_row("Data Coverage Percentage", f"{summary['coverage_percentage']:.1f} %")
    table.add_row("Number of Data Points", f"{summary['num_data_points']:,}")
    table.add_row("Number of Data Gaps", f"{summary['num_gaps']}")
    table.add_row("Number of Collapse Data Points", f"{summary['num_collapse_points']}")
    table.add_row("Number of Non-Collapse Data Points", f"{summary['num_non_collapse_points']:,}")
    table.add_row("Estimated Collapse Section Length", f"{summary['collapse_length']:.1f} m")
    
    console.print(table)
    console.print()


if __name__ == "__main__":
    main()