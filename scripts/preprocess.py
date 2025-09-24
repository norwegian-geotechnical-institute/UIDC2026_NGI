import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from sklearn.model_selection import train_test_split

from tbm_ml.preprocess_funcs import preprocess_data
from tbm_ml.schema_config import Config
from tbm_ml.utility import info_dataset


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

    train_df, test_df = train_test_split(df, test_size=(1-pcfg.experiment.train_fraction), random_state=pcfg.experiment.seed, stratify=df[pcfg.experiment.label])
    train_df.to_csv("data/model_ready/dataset_train.csv", index=False)
    test_df.to_csv("data/model_ready/dataset_test.csv", index=False)
    info_dataset(df, train_df, test_df, label=pcfg.experiment.label)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
