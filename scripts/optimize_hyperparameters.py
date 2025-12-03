import os
from datetime import datetime
from pathlib import Path

import hydra
import optuna
import yaml
import joblib
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from tbm_ml.hyperparameter_optimization import run_optimization
from tbm_ml.preprocess_funcs import get_dataset
from tbm_ml.schema_config import Config


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pcfg = Config(**OmegaConf.to_object(cfg))
    console = Console()
    console.print(pcfg)

    # Get model name from config
    model_name = pcfg.model.name

    # Set MLflow experiment name for single model optimization
    pcfg.mlflow.experiment_name = f"hyperparameter_optimization_{model_name}"

    console.print(
        f"[bold green]Optimizing hyperparameters for: {model_name}[/bold green]"
    )

    # Load data - use all training data for cross-validation
    console.print("[bold green]Loading training data...[/bold green]")
    df = get_dataset(pcfg.dataset.path_model_ready_train)

    X = df[pcfg.experiment.features]
    y = df[pcfg.experiment.label]

    console.print(
        f"[bold blue]Starting hyperparameter optimization for {model_name}...[/bold blue]"
    )
    console.print(f"Total samples: {len(X)}")
    console.print(f"Number of trials: {pcfg.optuna.n_trials}")
    console.print(
        f"Using {pcfg.optuna.cv_folds}-fold cross-validation for hyperparameter optimization"
    )

    # Prepare cost matrix for optimization
    cost_matrix = {
        "tn_cost": pcfg.experiment.cost_matrix.tn_cost,
        "fp_cost": pcfg.experiment.cost_matrix.fp_cost,
        "fn_cost": pcfg.experiment.cost_matrix.fn_cost,
        "tp_cost": pcfg.experiment.cost_matrix.tp_cost,
        "time_per_regular_advance": pcfg.experiment.cost_matrix.time_per_regular_advance,
    }

    # Run optimization with cross-validation and MLflow logging
    study = run_optimization(
        model_name=pcfg.model.name,
        X=X,
        y=y,
        oversample_level=pcfg.experiment.oversample_level,
        undersample_level=pcfg.experiment.undersample_level,
        undersample_ratio=pcfg.experiment.undersample_ratio,
        cost_matrix=cost_matrix,
        n_trials=pcfg.optuna.n_trials,
        cv_folds=pcfg.optuna.cv_folds,
        mlflow_path=Path(pcfg.mlflow.path),
        experiment_name=pcfg.mlflow.experiment_name,
        log_to_mlflow=True,
    )

    console.rule("Study statistics")
    console.print("Number of finished trials: ", len(study.trials))
    console.print(
        "Number of pruned trials: ",
        len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])),
    )
    console.print(
        "Number of complete trials: ",
        len(
            study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        ),
    )

    console.print("Best trial:")
    trial = study.best_trial

    console.print("Trial number: \t", trial.number)
    console.print("Best balanced accuracy: \t", f"{trial.value:.4f}")

    console.print("Best params: ")
    for key, value in trial.params.items():
        console.print(f"  {key}: {value}")

    # Save best parameters to a YAML file
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_dir = Path(pcfg.optuna.path_results)
    save_dir.mkdir(parents=True, exist_ok=True)

    yaml_filename = save_dir / f"best_hyperparameters_{model_name}_{timestamp}.yaml"

    # Create complete config with model info
    best_config = {
        "name": model_name,
        "params": trial.params,
        "performance": {
            "balanced_accuracy": trial.value,
            "trial_number": trial.number,
            "optimization_date": timestamp,
        },
    }

    with open(yaml_filename, "w") as yaml_file:
        yaml.dump(best_config, yaml_file, default_flow_style=False, indent=2)

    console.print(
        f"[bold blue]Best hyperparameters saved to {yaml_filename}[/bold blue]"
    )

    # Optionally save study object for further analysis
    study_filename = save_dir / f"optuna_study_{model_name}_{timestamp}.pkl"

    joblib.dump(study, study_filename)
    console.print(f"[bold blue]Complete study saved to {study_filename}[/bold blue]")


if __name__ == "__main__":
    main()
