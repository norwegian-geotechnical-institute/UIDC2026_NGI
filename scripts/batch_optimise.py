#!/usr/bin/env python3
"""
Simple batch hyperparameter optimisation script.
This script runs optimisation for each model directly without subprocess.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import Progress, track

# from sklearn.model_selection import train_test_split

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tbm_ml.hyperparameter_optimisation import run_optimisation
from tbm_ml.preprocess_funcs import get_dataset
from tbm_ml.schema_config import Config

# List of all models to optimise
MODELS = [
    # "xgboost_native",
    "xgboost",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "catboost",
    "lightgbm",
    "logistic_regression",
    "svm",
    "knn",
    # "gaussian_process",
]


def optimise_single_model(
    model_name: str, console: Console, base_config: DictConfig
) -> dict:
    """optimise a single model and return results."""
    console.print(
        f"\n[bold blue]🚀 Starting optimisation for {model_name}...[/bold blue]"
    )

    try:
        start_time = time.time()

        # Create config for this model
        cfg = base_config.copy()
        cfg.model.name = model_name

        pcfg = Config(**OmegaConf.to_object(cfg))

        # Load data once (reuse for all models)
        console.print("[bold green]Loading training data...[/bold green]")
        df = get_dataset(pcfg.dataset.path_model_ready_train)

        X = df[pcfg.experiment.features]
        y = df[pcfg.experiment.label]

        console.print(f"Total samples: {len(X)}")
        console.print(f"Number of trials: {pcfg.optuna.n_trials}")
        console.print(
            f"Using {pcfg.optuna.cv_folds}-fold cross-validation for hyperparameter optimization"
        )

        # Run optimisation with cross-validation and MLflow logging
        study = run_optimisation(
            model_name=model_name,
            X=X,
            y=y,
            oversample_level=pcfg.experiment.oversample_level,
            undersample_level=pcfg.experiment.undersample_level,
            n_trials=pcfg.optuna.n_trials,
            cv_folds=pcfg.optuna.cv_folds,
            mlflow_path=Path(pcfg.mlflow.path),
            experiment_name=pcfg.mlflow.experiment_name,
            log_to_mlflow=True,
        )

        end_time = time.time()
        duration = (end_time - start_time) / 60  # minutes

        best_trial = study.best_trial
        best_score = best_trial.value

        console.print(
            f"[bold green]✅ {model_name} completed successfully![/bold green]"
        )
        console.print(f"⏱️ Duration: {duration:.1f} minutes")
        console.print(f"🎯 Best balanced accuracy: {best_score:.4f}")

        # Save results
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        save_dir = Path(pcfg.optuna.path_results)
        save_dir.mkdir(parents=True, exist_ok=True)

        yaml_filename = save_dir / f"best_hyperparameters_{model_name}_{timestamp}.yaml"

        best_config = {
            "name": model_name,
            "params": best_trial.params,
            "performance": {
                "balanced_accuracy": best_score,
                "trial_number": best_trial.number,
                "optimisation_date": timestamp,
                "duration_minutes": duration,
            },
        }

        with open(yaml_filename, "w") as f:
            yaml.dump(best_config, f, default_flow_style=False, indent=2)

        console.print(f"📁 Results saved to: {yaml_filename}")

        return {
            "model": model_name,
            "success": True,
            "score": best_score,
            "duration": duration,
            "trials": len(study.trials),
            "file": str(yaml_filename),
        }

    except Exception as e:
        end_time = time.time()
        duration = (end_time - start_time) / 60

        console.print(f"[bold red]❌ {model_name} failed: {str(e)}[/bold red]")

        return {
            "model": model_name,
            "success": False,
            "error": str(e),
            "duration": duration,
        }


@hydra.main(version_base="1.3", config_path="config", config_name="main")
def main(cfg: DictConfig) -> int:
    console = Console()

    console.print(
        "[bold magenta]🎯 Batch Hyperparameter Optimisation for All Models[/bold magenta]"
    )
    console.print(f"Models to optimise: {', '.join(MODELS)}")
    console.print(f"Total models: {len(MODELS)}")

    # Initialize Hydra and load base configuration
    try:
        # Track results
        results = []
        start_time = time.time()

        # Run optimisation for each model
        for i, model in enumerate(MODELS, 1):
            console.rule(f"Model {i}/{len(MODELS)}: {model}")

            result = optimise_single_model(model, console, cfg)
            results.append(result)

        # Final summary
        end_time = time.time()
        total_duration = (end_time - start_time) / 3600  # hours

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        console.rule("[bold]Final Results")
        console.print(
            f"[bold green]✅ Successful optimisations ({len(successful)}):[/bold green]"
        )
        for result in successful:
            console.print(
                f"  • {result['model']}: {result['score']:.4f} ({result['duration']:.1f} min)"
            )

        if failed:
            console.print(
                f"\n[bold red]❌ Failed optimisations ({len(failed)}):[/bold red]"
            )
            for result in failed:
                console.print(
                    f"  • {result['model']}: {result.get('error', 'Unknown error')}"
                )

        console.print(f"\n[bold blue]📊 Summary:[/bold blue]")
        console.print(f"  • Total time: {total_duration:.2f} hours")
        console.print(
            f"  • Success rate: {len(successful)}/{len(MODELS)} ({len(successful)/len(MODELS)*100:.1f}%)"
        )

        # Save summary
        summary_file = Path("experiments/hyperparameters/batch_summary.txt")
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_file, "w") as f:
            f.write("Batch Hyperparameter optimisation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total duration: {total_duration:.2f} hours\n")
            f.write(
                f"Success rate: {len(successful)}/{len(MODELS)} ({len(successful)/len(MODELS)*100:.1f}%)\n\n"
            )

            f.write("Successful optimisations:\n")
            for result in successful:
                f.write(f"  - {result['model']}: {result['score']:.4f}\n")

            if failed:
                f.write(f"\nFailed optimisations:\n")
                for result in failed:
                    f.write(
                        f"  - {result['model']}: {result.get('error', 'Unknown error')}\n"
                    )

        console.print(f"\n[bold blue]📝 Summary saved to: {summary_file}[/bold blue]")
        console.print("[bold green]🎉 Batch optimisation complete![/bold green]")

        return 0 if not failed else 1

    except Exception as e:
        console.print(f"[bold red]💥 Failed to initialize: {str(e)}[/bold red]")
        return 1


if __name__ == "__main__":
    # Set environment for better error reporting
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)

    sys.exit(main())
