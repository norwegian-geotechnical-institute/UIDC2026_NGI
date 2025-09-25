from pprint import pformat
from typing import Any
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from rich.console import Console
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from tbm_ml.train_eval_funcs import xgb_native_pipeline, train_predict


def get_hyperparameter_space(model_name: str, trial: optuna.Trial) -> dict[str, Any]:
    """Define hyperparameter search spaces for different models."""

    if model_name == "xgboost_native":
        return {
            "objective": "binary:logistic",
            "device": "gpu",
            "random_state": 42,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-3, 10.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        }

    elif model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
        }

    elif model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": 42,
        }

    elif model_name == "extra_trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": 42,
        }

    elif model_name == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 50, 300),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
            "max_bins": trial.suggest_categorical("max_bins", [128, 255]),
            "random_state": 42,
        }

    elif model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "border_count": trial.suggest_categorical("border_count", [128, 254]),
            "random_seed": 42,
            "verbose": False,
            "task_type": "CPU",
        }

    elif model_name == "lightgbm":
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "random_state": 42,
            "verbose": -1,
        }

    elif model_name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-3, 100, log=True),
            "kernel": trial.suggest_categorical(
                "kernel", ["rbf", "linear", "poly", "sigmoid"]
            ),
            "gamma": (
                trial.suggest_categorical("gamma", ["scale", "auto"])
                if trial.params.get("kernel") == "rbf"
                else "scale"
            ),
            "degree": (
                trial.suggest_int("degree", 2, 5)
                if trial.params.get("kernel") == "poly"
                else 3
            ),
            "random_state": 42,
        }

    elif model_name == "logistic_regression":
        return {
            "penalty": trial.suggest_categorical(
                "penalty", ["l1", "l2", "elasticnet", None]
            ),
            "C": trial.suggest_float("C", 1e-3, 100, log=True),
            "solver": trial.suggest_categorical(
                "solver", ["liblinear", "lbfgs", "saga"]
            ),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "random_state": 42,
        }

    elif model_name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "leaf_size": trial.suggest_int("leaf_size", 10, 100),
            "p": trial.suggest_int("p", 1, 3),
        }

    elif model_name == "gaussian_process":
        return {
            "optimizer": trial.suggest_categorical("optimizer", ["fmin_l_bfgs_b"]),
            "n_restarts_optimizer": trial.suggest_int("n_restarts_optimizer", 0, 5),
            "max_iter_predict": trial.suggest_int("max_iter_predict", 50, 200),
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "rbf", "polynomial"]
            ),
            "length_scale": trial.suggest_float("length_scale", 1e-2, 1e2, log=True),
            "random_state": 42,
        }

    else:
        raise ValueError(f"Hyperparameter space not defined for model: {model_name}")


def create_objective_function(model_name: str, cv_folds: int = 5):
    """Create an objective function for a specific model with cross-validation."""

    def objective(
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        oversample_level: int,
        undersample_level: int,
    ) -> float:

        console = Console()

        # Get hyperparameters for this model
        model_params = get_hyperparameter_space(model_name, trial)

        console.print(f"\nModel: {model_name}")
        console.print(f"Trial {trial.number}: {pformat(trial.params)}")

        try:
            # Use stratified k-fold cross-validation
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                # Use appropriate training function
                if model_name == "xgboost_native":
                    y_pred_fold = xgb_native_pipeline(
                        X_train_fold,
                        X_val_fold,
                        y_train_fold,
                        y_val_fold,
                        model_params,
                        oversample_level,
                        undersample_level,
                    )
                else:
                    y_pred_fold = train_predict(
                        model_name,
                        model_params,
                        X_train_fold,
                        X_val_fold,
                        y_train_fold,
                        y_val_fold,
                        undersample_level,
                        oversample_level,
                    )

                # Calculate balanced accuracy for this fold
                fold_score = balanced_accuracy_score(y_val_fold, y_pred_fold)
                cv_scores.append(fold_score)

            # Return mean cross-validation score
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)

            console.print(
                f"CV Balanced accuracy: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})"
            )

            return mean_cv_score

        except Exception as e:
            console.print(f"[red]Error in trial {trial.number}: {str(e)}[/red]")
            return 0.0  # Return worst possible score on error

    return objective


# Run Optuna optimisation
def run_optimisation(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    oversample_level: int,
    undersample_level: int,
    n_trials: int = 100,
    cv_folds: int = 5,
    study_name: str = None,
    mlflow_path: Path = None,
    experiment_name: str = None,
    log_to_mlflow: bool = False,
) -> Any:

    if study_name is None:
        study_name = f"{model_name}_hyperparameter_optimisation"

    # Create objective function for this model
    objective_func = create_objective_function(model_name, cv_folds)

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        direction="maximize", study_name=study_name, sampler=sampler
    )

    # Track optimization time
    import time

    start_time = time.time()

    study.optimize(
        lambda trial: objective_func(trial, X, y, oversample_level, undersample_level),
        n_trials=n_trials,
    )

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60

    # Log to MLflow if requested
    if log_to_mlflow and mlflow_path and experiment_name:
        try:
            log_optuna_study_to_mlflow(
                study=study,
                model_name=model_name,
                mlflow_path=mlflow_path,
                experiment_name=experiment_name,
                oversample_level=oversample_level,
                undersample_level=undersample_level,
                cv_folds=cv_folds,
                duration_minutes=duration_minutes,
            )
        except Exception as e:
            console = Console()
            console.print(f"[red]Warning: Failed to log to MLflow: {e}[/red]")

    return study


def log_optuna_study_to_mlflow(
    study: optuna.Study,
    model_name: str,
    mlflow_path: Path,
    experiment_name: str,
    oversample_level: int,
    undersample_level: int,
    cv_folds: int,
    duration_minutes: float,
) -> None:
    """
    Log Optuna study results to MLflow.

    Parameters:
    -----------
    study : optuna.Study
        Completed Optuna study
    model_name : str
        Name of the model being optimized
    mlflow_path : Path
        Path to MLflow tracking directory
    experiment_name : str
        Name of MLflow experiment
    oversample_level : int
        Oversampling level used
    undersample_level : int
        Undersampling level used
    cv_folds : int
        Number of cross-validation folds
    duration_minutes : float
        Total optimization duration in minutes
    """

    # Use the same MLflow setup pattern as log_mlflow_metrics_and_model
    mlflow.set_tracking_uri(mlflow_path)
    mlflow.set_experiment(experiment_name=experiment_name)

    # Start MLflow run for this hyperparameter optimization
    with mlflow.start_run(run_name=f"{model_name}_hyperopt"):

        # Log study metadata
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_trials", len(study.trials))
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("oversample_level", oversample_level)
        mlflow.log_param("undersample_level", undersample_level)
        mlflow.log_param("optimization_direction", study.direction.name)

        # Log timing information
        mlflow.log_metric("duration_minutes", duration_minutes)
        mlflow.log_metric("duration_hours", duration_minutes / 60)

        # Log best trial information
        best_trial = study.best_trial
        mlflow.log_metric("best_balanced_accuracy", best_trial.value)
        mlflow.log_param("best_trial_number", best_trial.number)

        # Log best hyperparameters
        for param_name, param_value in best_trial.params.items():
            mlflow.log_param(f"best_{param_name}", param_value)

        # Log study statistics
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ]
        failed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
        ]

        mlflow.log_metric("n_completed_trials", len(completed_trials))
        mlflow.log_metric("n_pruned_trials", len(pruned_trials))
        mlflow.log_metric("n_failed_trials", len(failed_trials))
        mlflow.log_metric(
            "success_rate",
            len(completed_trials) / len(study.trials) if study.trials else 0,
        )

        # Log score statistics from completed trials
        if completed_trials:
            scores = [t.value for t in completed_trials]
            mlflow.log_metric("mean_cv_score", np.mean(scores))
            mlflow.log_metric("std_cv_score", np.std(scores))
            mlflow.log_metric("min_cv_score", np.min(scores))
            mlflow.log_metric("max_cv_score", np.max(scores))
            mlflow.log_metric("median_cv_score", np.median(scores))

        # Create and log optimization history plot
        try:
            import optuna.visualization as vis
            import matplotlib.pyplot as plt

            # Create optimization history plot
            fig_history = vis.matplotlib.plot_optimization_history(study)
            # Handle both figure and axes objects
            if hasattr(fig_history, "figure"):
                mlflow.log_figure(fig_history.figure, "optimization_history.png")
                plt.close(fig_history.figure)
            elif hasattr(fig_history, "get_figure"):
                mlflow.log_figure(fig_history.get_figure(), "optimization_history.png")
                plt.close(fig_history.get_figure())
            else:
                # Create a new figure and save the axes to it
                fig, ax = plt.subplots(figsize=(10, 6))
                fig_history.figure.savefig(
                    "temp_history.png", dpi=150, bbox_inches="tight"
                )
                mlflow.log_artifact("temp_history.png", artifact_path="plots")
                plt.close("all")
                Path("temp_history.png").unlink(missing_ok=True)

            # Create parameter importance plot if enough trials
            if (
                len(completed_trials) >= 5
            ):  # Lowered threshold since we have fewer trials in testing
                fig_importance = vis.matplotlib.plot_param_importances(study)
                if hasattr(fig_importance, "figure"):
                    mlflow.log_figure(fig_importance.figure, "param_importances.png")
                    plt.close(fig_importance.figure)
                elif hasattr(fig_importance, "get_figure"):
                    mlflow.log_figure(
                        fig_importance.get_figure(), "param_importances.png"
                    )
                    plt.close(fig_importance.get_figure())

            # Create parallel coordinate plot
            if len(completed_trials) >= 3:  # Lowered threshold for testing
                fig_parallel = vis.matplotlib.plot_parallel_coordinate(study)
                if hasattr(fig_parallel, "figure"):
                    mlflow.log_figure(fig_parallel.figure, "parallel_coordinate.png")
                    plt.close(fig_parallel.figure)
                elif hasattr(fig_parallel, "get_figure"):
                    mlflow.log_figure(
                        fig_parallel.get_figure(), "parallel_coordinate.png"
                    )
                    plt.close(fig_parallel.get_figure())

        except ImportError:
            console = Console()
            console.print(
                "[yellow]Warning: optuna.visualization not available, skipping plots[/yellow]"
            )
        except Exception as e:
            console = Console()
            console.print(
                f"[yellow]Warning: Could not create Optuna plots: {e}[/yellow]"
            )
