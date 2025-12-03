#!/usr/bin/env python3
"""
Simple batch hyperparameter optimization script.
This script runs optimization for each model directly without subprocess.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import Progress, track

# from sklearn.model_selection import train_test_split

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tbm_ml.hyperparameter_optimization import run_optimization
from tbm_ml.preprocess_funcs import get_dataset
from tbm_ml.schema_config import Config
from tbm_ml.train_eval_funcs import train_predict, evaluate_model

# List of all models to optimize
MODELS_TO_OPTIMIZE = [
    "xgboost",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "catboost",
    "lightgbm",
    "svm",
    "knn",
    "gaussian_process",
    "logistic_regression",
]

# Baseline models to evaluate without optimization
BASELINE_MODELS = [
    # "logistic_regression",
    "dummy",
]


def optimize_single_model(
    model_name: str, console: Console, base_config: DictConfig
) -> dict:
    """optimize a single model and return results."""
    console.print(
        f"\n[bold blue]🚀 Starting optimization for {model_name}...[/bold blue]"
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
            model_name=model_name,
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
            random_seed=pcfg.experiment.seed,
        )

        end_time = time.time()
        duration = (end_time - start_time) / 60  # minutes

        best_trial = study.best_trial
        best_score = best_trial.value

        console.print(
            f"[bold green]✅ {model_name} completed successfully![/bold green]"
        )
        console.print(f"⏱️ Duration: {duration:.1f} minutes")
        console.print(f"🎯 Best average cost: {best_score:.4f}")
        
        # Evaluate on test set with best parameters
        console.print("[bold green]Evaluating on test set...[/bold green]")
        df_test = get_dataset(pcfg.dataset.path_model_ready_test)
        X_test = df_test[pcfg.experiment.features]
        y_test = df_test[pcfg.experiment.label]
        
        # Train final model with best parameters
        y_pred = train_predict(
            model_name=model_name,
            model_params=best_trial.params,
            X_train=X,
            X_test=X_test,
            y_train=y,
            y_test=y_test,
            undersample_level=pcfg.experiment.undersample_level,
            undersample_ratio=pcfg.experiment.undersample_ratio,
            oversample_level=pcfg.experiment.oversample_level,
            save_model=False,
            random_seed=pcfg.experiment.seed,
        )
        
        # Evaluate and get metrics
        cost_matrix = {
            "tn_cost": pcfg.experiment.cost_matrix.tn_cost,
            "fp_cost": pcfg.experiment.cost_matrix.fp_cost,
            "fn_cost": pcfg.experiment.cost_matrix.fn_cost,
            "tp_cost": pcfg.experiment.cost_matrix.tp_cost,
            "time_per_regular_advance": pcfg.experiment.cost_matrix.time_per_regular_advance,
        }
        
        test_metrics, _ = evaluate_model(
            y_test=y_test,
            y_pred=y_pred,
            class_mapping=pcfg.experiment.tbm_classification,
            cost_matrix=cost_matrix,
        )
        
        console.print(f"📊 Test Accuracy: {test_metrics['accuracy']:.4f}")
        console.print(f"📊 Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        console.print(f"📊 Test Recall: {test_metrics['recall']:.4f}")
        console.print(f"📊 Test Precision: {test_metrics['precision']:.4f}")
        console.print(f"📊 Test F1-Score: {test_metrics['f1']:.4f}")
        console.print(f"💰 Test Average Cost: {test_metrics['cost_average']:.4f}")

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
                "optimization_date": timestamp,
                "duration_minutes": duration,
            },
            "test_metrics": {
                "accuracy": float(test_metrics["accuracy"]),
                "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
                "recall": float(test_metrics["recall"]),
                "precision": float(test_metrics["precision"]),
                "f1": float(test_metrics["f1"]),
                "cost_average": float(test_metrics["cost_average"]),
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
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_recall": test_metrics["recall"],
            "test_precision": test_metrics["precision"],
            "test_f1": test_metrics["f1"],
            "test_cost_average": test_metrics["cost_average"],
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


def evaluate_baseline_model(
    model_name: str, console: Console, base_config: DictConfig
) -> dict:
    """Evaluate a baseline model with default parameters (no optimization)."""
    console.print(
        f"\n[bold blue]📊 Evaluating baseline model: {model_name}...[/bold blue]"
    )

    try:
        start_time = time.time()

        # Create config for this model
        cfg = base_config.copy()
        cfg.model.name = model_name

        pcfg = Config(**OmegaConf.to_object(cfg))

        # Load data
        console.print("[bold green]Loading data...[/bold green]")
        df_train = get_dataset(pcfg.dataset.path_model_ready_train)
        df_test = get_dataset(pcfg.dataset.path_model_ready_test)

        X_train = df_train[pcfg.experiment.features]
        y_train = df_train[pcfg.experiment.label]
        X_test = df_test[pcfg.experiment.features]
        y_test = df_test[pcfg.experiment.label]

        # Use default parameters for baseline models
        if model_name == "dummy":
            model_params = {"strategy": "most_frequent", "random_state": pcfg.experiment.seed}
        elif model_name == "logistic_regression":
            model_params = {
                "C": 1.0,
                "penalty": "l2",
                "solver": "liblinear",
                "max_iter": 1000,
                "random_state": pcfg.experiment.seed,
            }
        else:
            model_params = {}

        # Train and predict
        y_pred = train_predict(
            model_name=model_name,
            model_params=model_params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            undersample_level=pcfg.experiment.undersample_level,
            undersample_ratio=pcfg.experiment.undersample_ratio,
            oversample_level=pcfg.experiment.oversample_level,
            save_model=False,
            random_seed=pcfg.experiment.seed,
        )

        # Evaluate and get metrics
        cost_matrix = {
            "tn_cost": pcfg.experiment.cost_matrix.tn_cost,
            "fp_cost": pcfg.experiment.cost_matrix.fp_cost,
            "fn_cost": pcfg.experiment.cost_matrix.fn_cost,
            "tp_cost": pcfg.experiment.cost_matrix.tp_cost,
            "time_per_regular_advance": pcfg.experiment.cost_matrix.time_per_regular_advance,
        }

        test_metrics, _ = evaluate_model(
            y_test=y_test,
            y_pred=y_pred,
            class_mapping=pcfg.experiment.tbm_classification,
            cost_matrix=cost_matrix,
        )

        end_time = time.time()
        duration = (end_time - start_time) / 60  # minutes

        console.print(
            f"[bold green]✅ {model_name} evaluated successfully![/bold green]"
        )
        console.print(f"⏱️ Duration: {duration:.1f} minutes")
        console.print(f"📊 Test Accuracy: {test_metrics['accuracy']:.4f}")
        console.print(f"📊 Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        console.print(f"📊 Test Recall: {test_metrics['recall']:.4f}")
        console.print(f"📊 Test Precision: {test_metrics['precision']:.4f}")
        console.print(f"📊 Test F1-Score: {test_metrics['f1']:.4f}")
        console.print(f"💰 Test Average Cost: {test_metrics['cost_average']:.4f}")

        return {
            "model": model_name,
            "success": True,
            "score": test_metrics["cost_average"],  # Use cost as score for consistency
            "duration": duration,
            "trials": 0,  # No trials for baseline
            "file": "N/A",
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_recall": test_metrics["recall"],
            "test_precision": test_metrics["precision"],
            "test_f1": test_metrics["f1"],
            "test_cost_average": test_metrics["cost_average"],
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

    total_models = len(MODELS_TO_OPTIMIZE) + len(BASELINE_MODELS)
    console.print(
        "[bold magenta]🎯 Batch Model Evaluation[/bold magenta]"
    )
    console.print(f"Models to optimize: {', '.join(MODELS_TO_OPTIMIZE)}")
    console.print(f"Baseline models: {', '.join(BASELINE_MODELS)}")
    console.print(f"Total models: {total_models}")

    # Set MLflow experiment name for batch optimization
    cfg.mlflow.experiment_name = "batch_hyperparameter_optimization"

    # Initialize Hydra and load base configuration
    try:
        # Track results
        results = []
        start_time = time.time()

        # Run optimization for models that need it
        for i, model in enumerate(MODELS_TO_OPTIMIZE, 1):
            console.rule(f"Model {i}/{total_models}: {model} (with optimization)")
            result = optimize_single_model(model, console, cfg)
            results.append(result)

        # Evaluate baseline models without optimization
        for i, model in enumerate(BASELINE_MODELS, len(MODELS_TO_OPTIMIZE) + 1):
            console.rule(f"Model {i}/{total_models}: {model} (baseline)")
            result = evaluate_baseline_model(model, console, cfg)
            results.append(result)

        # Final summary
        end_time = time.time()
        total_duration = (end_time - start_time) / 3600  # hours

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        console.rule("[bold]Final Results")
        console.print(
            f"[bold green]✅ Successful evaluations ({len(successful)}):[/bold green]"
        )
        for result in successful:
            console.print(
                f"  • {result['model']}: {result['score']:.4f} ({result['duration']:.1f} min)"
            )

        if failed:
            console.print(
                f"\n[bold red]❌ Failed optimizations ({len(failed)}):[/bold red]"
            )
            for result in failed:
                console.print(
                    f"  • {result['model']}: {result.get('error', 'Unknown error')}"
                )

        console.print(f"\n[bold blue]📊 Summary:[/bold blue]")
        console.print(f"  • Total time: {total_duration:.2f} hours")
        console.print(
            f"  • Success rate: {len(successful)}/{total_models} ({len(successful)/total_models*100:.1f}%)"
        )

        # Save summary
        summary_file = Path("experiments/hyperparameters/batch_summary.txt")
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_file, "w") as f:
            f.write("Batch Model Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total duration: {total_duration:.2f} hours\n")
            f.write(
                f"Success rate: {len(successful)}/{total_models} ({len(successful)/total_models*100:.1f}%)\n\n"
            )

            f.write("Successful evaluations:\n")
            for result in successful:
                f.write(f"  - {result['model']}: {result['score']:.4f}\n")

            if failed:
                f.write(f"\nFailed evaluations:\n")
                for result in failed:
                    f.write(
                        f"  - {result['model']}: {result.get('error', 'Unknown error')}\n"
                    )

        console.print(f"\n[bold blue]📝 Summary saved to: {summary_file}[/bold blue]")
        
        # Save CSV with all metrics
        if successful:
            csv_file = Path("experiments/hyperparameters/batch_results.csv")
            
            # Create DataFrame with results
            csv_data = []
            for result in successful:
                csv_data.append({
                    "Classifier": result["model"].upper().replace("_", ""),
                    "Accuracy ↑": f"{result['test_accuracy']:.4f}",
                    "Balanced Accuracy ↑": f"{result['test_balanced_accuracy']:.4f}",
                    "Recall ↑": f"{result['test_recall']:.4f}",
                    "Precision ↑": f"{result['test_precision']:.4f}",
                    "F1-Score ↑": f"{result['test_f1']:.4f}",
                    "Average Cost ↓": f"{result['test_cost_average']:.4f}",
                })
            
            df_results = pd.DataFrame(csv_data)
            df_results.to_csv(csv_file, index=False)
            
            console.print(f"[bold blue]📊 Results CSV saved to: {csv_file}[/bold blue]")
        
        console.print("[bold green]🎉 Batch evaluation complete![/bold green]")

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
