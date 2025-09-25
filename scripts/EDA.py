import hydra
import matplotlib.pyplot as plt
import pandas as pd

from rich.console import Console
from omegaconf import DictConfig, OmegaConf

from tbm_ml.schema_config import Config
from tbm_ml.plotting import pairplot, plot_tbm_parameters


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
        'Total F': 'thrust\n[kN]',
        'Fpi': 'Field Penetration Index',
        'Tpi': 'drilling efficiency index\nTPI'
        }
    df.rename(rename_dict, axis=1, inplace=True)

    # Plot lineplot
    console.print("[bold green]Plotting TBM parameters...[/bold green]")
    fig1: plt.Figure = plot_tbm_parameters(df)
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


if __name__ == "__main__":
    main()