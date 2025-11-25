from pprint import pformat
from typing import Any
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend before importing pyplot

import mlflow
import numpy as np
import optuna
import pandas as pd
from rich.console import Console
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from tbm_ml.train_eval_funcs import xgb_native_pipeline, train_predict


def get_hyperparameter_space(model_name: str, trial: optuna.Trial) -> dict[str, Any]:
    """Define hyperparameter search spaces for different models.

    Optimized for binary classification with limited data (~3000 samples).
    Focus on regularization and overfitting prevention.
    """

    if model_name == "xgboost_native":
        return {
            "objective": "binary:logistic",
            "device": "cpu",  # More stable for small datasets
            "random_state": 42,
            # CRITICAL: Tree structure (most important)
            "max_depth": trial.suggest_int("max_depth", 3, 8),  # Reduced max depth
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            # CRITICAL: Regularization (prevent overfitting)
            "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),  # L2 reg
            "alpha": trial.suggest_float("alpha", 1e-3, 1.0, log=True),  # L1 reg
            # HIGH: Sample control
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),  # Less aggressive
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            # MEDIUM: Tree complexity
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),  # Reduced range
        }

    elif model_name == "xgboost":
        return {
            # CRITICAL: Tree control
            "max_depth": trial.suggest_int("max_depth", 3, 8),  # Reduced for small data
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),  # Reduced range
            # CRITICAL: Regularization
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            # HIGH: Sampling
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "random_state": 42,
        }

    elif model_name == "random_forest":
        return {
            # CRITICAL: Tree control
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),  # Reduced range
            "max_depth": trial.suggest_int("max_depth", 5, 20),  # Reduced max depth
            # CRITICAL: Overfitting control
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 5, 20
            ),  # Increased min
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 2, 10
            ),  # Increased min
            # HIGH: Feature randomness
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),  # Removed None
            # LOW: Bootstrap (less critical with small data)
            "bootstrap": True,  # Fixed to True for stability
            "random_state": 42,
        }

    elif model_name == "extra_trees":
        return {
            # CRITICAL: Tree control (similar to random forest but can be slightly deeper)
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            # CRITICAL: Overfitting control
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
            # HIGH: Feature randomness
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            # MEDIUM: Bootstrap (Extra Trees can handle without bootstrap better)
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": 42,
        }

    elif model_name == "hist_gradient_boosting":
        return {
            # CRITICAL: Learning control
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),  # Reduced max
            "max_iter": trial.suggest_int(
                "max_iter", 50, 200
            ),  # Reduced for small data
            # CRITICAL: Tree structure
            "max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes", 15, 50
            ),  # Reduced max
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 20, 50
            ),  # Increased min
            # HIGH: Regularization
            "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 1.0),
            # LOW: Binning (less critical for small data)
            "max_bins": 128,  # Fixed for consistency
            "random_state": 42,
        }

    elif model_name == "catboost":
        return {
            # CRITICAL: Learning control
            "iterations": trial.suggest_int("iterations", 100, 500),  # Reduced max
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 8),  # Reduced max depth
            # CRITICAL: Regularization
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20),  # Increased min
            # MEDIUM: Other parameters
            "border_count": 128,  # Fixed for small datasets
            "random_seed": 42,
            "verbose": False,
            "task_type": "CPU",
        }

    elif model_name == "lightgbm":
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            # CRITICAL: Tree structure
            "num_leaves": trial.suggest_int("num_leaves", 10, 50),  # Reduced max
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),  # Reduced range
            # CRITICAL: Regularization
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 50
            ),  # Increased min
            # HIGH: Feature control
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),  # Reduced max
            "random_state": 42,
            "verbose": -1,
        }

    elif model_name == "svm":
        return {
            # CRITICAL: Regularization vs complexity trade-off
            "C": trial.suggest_float(
                "C", 0.01, 10, log=True
            ),  # Reduced range for stability
            # CRITICAL: Kernel choice
            "kernel": trial.suggest_categorical(
                "kernel", ["rbf", "linear"]
            ),  # Simplified
            # HIGH: RBF kernel parameter (only if RBF selected)
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "random_state": 42,
        }

    elif model_name == "logistic_regression":
        return {
            # CRITICAL: Regularization strength
            "C": trial.suggest_float("C", 0.01, 10, log=True),  # Focused range
            # HIGH: Regularization type
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),  # Simplified
            # MEDIUM: Solver (depends on penalty)
            "solver": "liblinear",  # Fixed for stability with small data
            "max_iter": 1000,  # Fixed sufficient value
            "random_state": 42,
        }

    elif model_name == "knn":
        return {
            # CRITICAL: Number of neighbors (most important for KNN)
            "n_neighbors": trial.suggest_int(
                "n_neighbors", 3, 25
            ),  # Reduced max for small data
            # HIGH: Distance weighting
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            # MEDIUM: Distance metric
            "p": trial.suggest_int("p", 1, 2),  # Manhattan or Euclidean
            # LOW: Algorithm choice (auto is usually best)
            "algorithm": "auto",  # Fixed for simplicity
            "leaf_size": 30,  # Fixed default value
        }

    elif model_name == "gaussian_process":
        # SIMPLIFIED: GP is computationally expensive and complex for hyperopt with small data
        # Fix most parameters to reasonable defaults and tune only the most critical ones
        return {
            # MEDIUM: Optimizer (fixed to most robust option)
            "optimizer": "fmin_l_bfgs_b",
            "n_restarts_optimizer": 2,  # Fixed reasonable value
            "max_iter_predict": 100,  # Fixed reasonable value
            # LOW: Kernel choice (simplified, RBF is usually good)
            "kernel": None,  # Use default RBF kernel
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
        undersample_level: int | None = None,
        undersample_ratio: float | None = None,
    ) -> float:

        console = Console()

        # Get hyperparameters for this model
        model_params = get_hyperparameter_space(model_name, trial)

        console.print(f"\nModel: {model_name}")
        console.print(f"Trial {trial.number}: {pformat(trial.params)}")

        try:
            # Use stratified k-fold cross-validation
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_metrics = {
                "balanced_accuracy": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
            }

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
                        undersample_level=undersample_level,
                        undersample_ratio=undersample_ratio,
                        oversample_level=oversample_level,
                    )

                # Calculate all metrics for this fold
                cv_metrics["balanced_accuracy"].append(
                    balanced_accuracy_score(y_val_fold, y_pred_fold)
                )
                cv_metrics["accuracy"].append(accuracy_score(y_val_fold, y_pred_fold))
                cv_metrics["precision"].append(
                    precision_score(y_val_fold, y_pred_fold, zero_division=0)
                )
                cv_metrics["recall"].append(
                    recall_score(y_val_fold, y_pred_fold, zero_division=0)
                )
                cv_metrics["f1"].append(
                    f1_score(y_val_fold, y_pred_fold, zero_division=0)
                )

            # Calculate mean scores for all metrics
            mean_metrics = {
                metric: np.mean(scores) for metric, scores in cv_metrics.items()
            }
            std_metrics = {
                metric: np.std(scores) for metric, scores in cv_metrics.items()
            }

            # Store metrics in trial for later MLflow logging
            for metric, mean_score in mean_metrics.items():
                trial.set_user_attr(f"cv_mean_{metric}", mean_score)
                trial.set_user_attr(f"cv_std_{metric}", std_metrics[metric])

            console.print(
                f"CV Metrics - Balanced Accuracy: {mean_metrics['balanced_accuracy']:.4f} "
                f"(+/- {std_metrics['balanced_accuracy']:.4f}), "
                f"Accuracy: {mean_metrics['accuracy']:.4f}, "
                f"Precision: {mean_metrics['precision']:.4f}, "
                f"Recall: {mean_metrics['recall']:.4f}, "
                f"F1: {mean_metrics['f1']:.4f}"
            )

            # Return balanced accuracy as the optimization target
            return mean_metrics["balanced_accuracy"]

        except Exception as e:
            console.print(f"[red]Error in trial {trial.number}: {str(e)}[/red]")
            return 0.0  # Return worst possible score on error

    return objective


# Run Optuna optimization
def run_optimization(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    oversample_level: int,
    undersample_level: int | None = None,
    undersample_ratio: float | None = None,
    n_trials: int = 100,
    cv_folds: int = 5,
    study_name: str = None,
    mlflow_path: Path = None,
    experiment_name: str = None,
    log_to_mlflow: bool = False,
) -> Any:

    if study_name is None:
        study_name = f"{model_name}_hyperparameter_optimization"

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
        lambda trial: objective_func(trial, X, y, oversample_level, undersample_level, undersample_ratio),
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

        # Log all metrics from the best trial
        metrics_to_log = ["balanced_accuracy", "accuracy", "precision", "recall", "f1"]
        for metric in metrics_to_log:
            mean_key = f"cv_mean_{metric}"
            std_key = f"cv_std_{metric}"
            if mean_key in best_trial.user_attrs:
                mlflow.log_metric(
                    f"best_{metric}_mean", best_trial.user_attrs[mean_key]
                )
                mlflow.log_metric(f"best_{metric}_std", best_trial.user_attrs[std_key])

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
            # Log balanced accuracy statistics (primary optimization metric)
            balanced_accuracy_scores = [t.value for t in completed_trials]
            mlflow.log_metric("mean_cv_score", np.mean(balanced_accuracy_scores))
            mlflow.log_metric("std_cv_score", np.std(balanced_accuracy_scores))
            mlflow.log_metric("min_cv_score", np.min(balanced_accuracy_scores))
            mlflow.log_metric("max_cv_score", np.max(balanced_accuracy_scores))
            mlflow.log_metric("median_cv_score", np.median(balanced_accuracy_scores))

            # Log statistics for all metrics across all completed trials
            metrics_to_log = [
                "balanced_accuracy",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
            for metric in metrics_to_log:
                mean_key = f"cv_mean_{metric}"
                std_key = f"cv_std_{metric}"

                # Collect scores for this metric from all completed trials
                metric_scores = []
                for trial in completed_trials:
                    if mean_key in trial.user_attrs:
                        metric_scores.append(trial.user_attrs[mean_key])

                if metric_scores:
                    mlflow.log_metric(
                        f"all_trials_mean_{metric}", np.mean(metric_scores)
                    )
                    mlflow.log_metric(f"all_trials_std_{metric}", np.std(metric_scores))
                    mlflow.log_metric(f"all_trials_min_{metric}", np.min(metric_scores))
                    mlflow.log_metric(f"all_trials_max_{metric}", np.max(metric_scores))

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
