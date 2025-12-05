import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from tbm_ml.preprocess_funcs import preprocess_data
from tbm_ml.schema_config import Config
from tbm_ml.utility import info_dataset
from tbm_ml.collapse_section_split import split_by_section_range


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pcfg = Config(**OmegaConf.to_object(cfg))
    console = Console()
    console.print(pcfg)

    df: pd.DataFrame = preprocess_data(
        path_file=pcfg.dataset.path_raw_dataset,
        features=pcfg.experiment.features,
        site_features=pcfg.experiment.site_info,
        labels=pcfg.experiment.label,
        outlier_feature=pcfg.preprocess.outlier_feature,
        remove_duplicates=pcfg.preprocess.remove_duplicates,
        remove_outliers_hard=pcfg.preprocess.remove_outliers_hard,
        remove_outliers_uni=pcfg.preprocess.remove_outliers_uni,
        remove_outliers_multi=pcfg.preprocess.remove_outliers_multi,
        univariate_threshold=pcfg.preprocess.univariate_threshold,
        multivariate_threshold=pcfg.preprocess.multivariate_threshold,
    )

    df.to_csv("data/model_ready/dataset_total.csv", index=False)

    # Use balanced size split by collapse sections
    # Train: sections 1-15 (199 collapse samples, 75.4%)
    # Test: sections 16-18 (65 collapse samples, 24.6%)
    console.print("\n[bold cyan]Splitting data by collapse sections (balanced size split)...[/bold cyan]")
    train_sections = list(range(1, 16))  # Sections 1-15
    test_sections = list(range(16, 19))   # Sections 16-18
    train_df, test_df = split_by_section_range(
        df, 
        train_sections=train_sections,
        test_sections=test_sections,
        label_column=pcfg.experiment.label
    )
    
    train_df.to_csv("data/model_ready/dataset_train.csv", index=False)
    test_df.to_csv("data/model_ready/dataset_test.csv", index=False)
    info_dataset(df, train_df, test_df, label=pcfg.experiment.label)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
