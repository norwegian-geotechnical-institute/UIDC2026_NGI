import os
from datetime import datetime

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from sklearn.model_selection import train_test_split

from tbm_ml.hyperparameter_optimisation import run_optimisation
from tbm_ml.preprocess_funcs import get_dataset
from tbm_ml.schema_config import Config


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pcfg = Config(**OmegaConf.to_object(cfg))
    console = Console()
    console.print(pcfg)
    pcfg.mlflow.experiment_name = "train_test"

    # Load data - only use the training data for hyperparameter optimisation
    console.print("[bold green]Loading training and testing data...[/bold green]")
    df = get_dataset(pcfg.dataset.path_model_ready_train)

    X = df[pcfg.experiment.features]
    y = df[pcfg.experiment.label]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=(1 - pcfg.experiment.train_fraction),
                                                        random_state=pcfg.experiment.seed,
                                                        stratify=y)

    # optimally run using cross validation
    study = run_optimisation(
        X_train,
        X_test,
        y_train,
        y_test,
        pcfg.experiment.oversample_level,
        pcfg.experiment.undersample_level,
        n_trials=pcfg.optuna.n_trials,
    )

    console.rule("Study statistics")
    console.print("Number of finished trials: ", len(study.trials))

    console.print("Best trial:")
    trial = study.best_trial

    console.print("Trial number: \t", trial.number)
    console.print("Balanced accuracy: \t", trial.value)

    console.print("Params: ")
    for key, value in trial.params.items():
        console.print(f"{key}: {value}")

    # Save best parameters to a YAML file
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_dir = "./experiments/hyperparameters"
    os.makedirs(save_dir, exist_ok=True)
    yaml_filename = os.path.join(
        save_dir, f"best_hyperparameters_xgboost_{timestamp}.yaml"
    )

    with open(yaml_filename, "w") as yaml_file:
        yaml.dump(trial.params, yaml_file)

    console.print(
        f"[bold blue]Best hyperparameters saved to {yaml_filename}[/bold blue]"
    )


if __name__ == "__main__":
    main()
