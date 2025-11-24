from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from rich.pretty import pprint
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC as SVClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from tbm_ml.plotting import plot_confusion_matrix, plot_tbm_confusion_matrix


def calculate_prediction_costs(
    y_true: pd.Series,
    y_pred: pd.Series,
    tn_cost: float = 0.0,
    fp_cost: float = 5.0,
    fn_cost: float = 100.0,
    tp_cost: float = 0.0,
) -> dict[str, float]:
    """
    Calculate the cost of predictions based on confusion matrix and cost values.
    
    Parameters:
    -----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    tn_cost : float
        Cost for True Negatives (correctly predicted regular excavation)
    fp_cost : float
        Cost for False Positives (predicted collapse but was regular - false alarm)
    fn_cost : float
        Cost for False Negatives (predicted regular but was collapse - missed collapse)
    tp_cost : float
        Cost for True Positives (correctly predicted collapse)
    
    Returns:
    --------
    dict with cost metrics:
        - total_cost: Sum of all prediction costs
        - average_cost: Average cost per prediction
        - tn_count, fp_count, fn_count, tp_count: Confusion matrix counts
        - tn_total_cost, fp_total_cost, fn_total_cost, tp_total_cost: Individual cost components
    """
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate individual cost components
    tn_total_cost = tn * tn_cost
    fp_total_cost = fp * fp_cost
    fn_total_cost = fn * fn_cost
    tp_total_cost = tp * tp_cost
    
    # Calculate total and average cost
    total_cost = tn_total_cost + fp_total_cost + fn_total_cost + tp_total_cost
    average_cost = total_cost / len(y_true) if len(y_true) > 0 else 0.0
    
    return {
        "total_cost": total_cost,
        "average_cost": average_cost,
        "tn_count": int(tn),
        "fp_count": int(fp),
        "fn_count": int(fn),
        "tp_count": int(tp),
        "tn_total_cost": tn_total_cost,
        "fp_total_cost": fp_total_cost,
        "fn_total_cost": fn_total_cost,
        "tp_total_cost": tp_total_cost,
    }


def calculate_expected_cost(
    y_true: pd.Series,
    y_pred: pd.Series,
    cost_matrix: dict[str, float],
) -> float:
    """
    Calculate the expected cost per prediction.
    
    This is a convenience function that returns just the average cost,
    useful for optimization objectives.
    
    Parameters:
    -----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    cost_matrix : dict
        Dictionary with keys: 'tn_cost', 'fp_cost', 'fn_cost', 'tp_cost'
    
    Returns:
    --------
    float : Average cost per prediction (lower is better)
    """

    # Make sure all required cost values are provided and not None
    required_keys = ["tn_cost", "fp_cost", "fn_cost", "tp_cost"]
    assert all(
        key in cost_matrix and cost_matrix[key] is not None for key in required_keys
    ), f"Cost matrix must contain all required keys {required_keys} with non-None values"

    costs = calculate_prediction_costs(
        y_true,
        y_pred,
        tn_cost=cost_matrix["tn_cost"],
        fp_cost=cost_matrix["fp_cost"],
        fn_cost=cost_matrix["fn_cost"],
        tp_cost=cost_matrix["tp_cost"],
    )
    return costs["average_cost"]


def load_data(
    train_data_path: Path,
    test_data_path: Path,
    target_column: str,
    features_columns: list,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    train_data = pd.read_csv(Path(train_data_path))
    test_data = pd.read_csv(Path(test_data_path))

    X_train = train_data[features_columns]
    y_train = train_data[target_column]
    X_test = test_data[features_columns]
    y_test = test_data[target_column]

    return X_train, X_test, y_train, y_test


def xgb_native_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_params: dict,
    oversample_level: int,
    undersample_level: int,
    model_save_path: str = "models/xgb_model.json",  # Path to save the model
    save_model: bool = False,
    random_seed: int = 42,
) -> pd.Series:

    # Convert labels to integers to ensure consistency
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    undersample_dict = {
        cls: undersample_level
        for cls in y_train.value_counts().index
        if y_train.value_counts()[cls] > undersample_level
    }
    oversample_dict = {
        cls: oversample_level
        for cls in y_train.value_counts().index
        if y_train.value_counts()[cls] < oversample_level
    }

    # Apply RandomUnderSampler to reduce majority classes
    undersampler = RandomUnderSampler(
        sampling_strategy=undersample_dict, random_state=random_seed
    )

    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

    # Apply SMOTE to oversample minority classes
    smote = SMOTE(sampling_strategy=oversample_dict, random_state=random_seed)
    X_train_final, y_train_final = smote.fit_resample(
        X_train_resampled, y_train_resampled
    )

    # For binary classification, ensure labels are 0 and 1
    # No need to adjust labels if they're already 0 and 1
    y_train = y_train_final
    y_test = y_test

    # Convert the balanced training data to DMatrix (with GPU support)
    dtrain = xgb.DMatrix(X_train_final, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set default parameters for XGBoost if not provided
    # Note: num_class is not needed for binary classification
    params = {}
    model_params.update(params)

    # Train the XGBoost model
    xgb_model = xgb.train(model_params, dtrain, num_boost_round=100)

    # optionally save the model
    if save_model:
        xgb_model.save_model(model_save_path)
        pprint(f"Model saved at {model_save_path}")

    # Make predictions (output is probabilities for binary:logistic)
    y_pred_proba = xgb_model.predict(dtest)
    # Convert probabilities to class labels (0 or 1)
    y_pred = (y_pred_proba > 0.5).astype(int)
    return y_pred


def train_predict(
    model_name: str,
    model_params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    undersample_level: int,
    oversample_level: int,
    save_model: bool = False,
    random_seed: int = 42,
) -> Pipeline:

    undersample_dict = {
        cls: undersample_level
        for cls in y_train.value_counts().index
        if y_train.value_counts()[cls] > undersample_level
    }
    oversample_dict = {
        cls: oversample_level
        for cls in y_train.value_counts().index
        if y_train.value_counts()[cls] < oversample_level
    }

    # Select classifier based on model_name
    match model_name:
        case "knn":
            classifier = KNeighborsClassifier(**model_params)
        case "xgboost":
            classifier = XGBClassifier(**model_params)
        case "xgboost_native":
            classifier = None
        case "dummy":
            classifier = DummyClassifier(**model_params)
        case "logistic_regression":
            classifier = LogisticRegression(**model_params)
        case "extra_trees":
            classifier = ExtraTreesClassifier(**model_params)
        case "random_forest":
            classifier = RandomForestClassifier(**model_params)
        case "hist_gradient_boosting":
            classifier = HistGradientBoostingClassifier(**model_params)
        case "catboost":
            classifier = CatBoostClassifier(**model_params)
        case "lightgbm":
            classifier = LGBMClassifier(**model_params)
        case "svm":
            classifier = SVClassifier(**model_params)
        case "gaussian_process":
            classifier = GaussianProcessClassifier(**model_params)
        case _:
            raise NotImplementedError(f"Model {model_name} is not implemented yet.")

    if model_name == "xgboost_native":
        y_pred = xgb_native_pipeline(
            X_train,
            X_test,
            y_train,
            y_test,
            model_params,
            oversample_level,
            undersample_level,
            save_model=save_model,
            random_seed=random_seed,
        )
    else:
        pipeline = make_pipeline(
            StandardScaler(),
            RandomUnderSampler(sampling_strategy=undersample_dict, random_state=random_seed),
            SMOTE(sampling_strategy=oversample_dict, random_state=random_seed),
            classifier,
        )
        # changes the shape of y_train to (n_samples, )
        y_train = y_train.values.ravel()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

    return y_pred


def evaluate_model(
    y_test: pd.Series,
    y_pred: pd.Series,
    class_mapping: dict,
    cost_matrix: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, plt.Figure]]:

    # Calculate traditional metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Store metrics in dictionary
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }

    # Calculate cost-based metrics if cost_matrix is provided
    if cost_matrix is not None:
        cost_metrics = calculate_prediction_costs(
            y_test,
            y_pred,
            tn_cost=cost_matrix["tn_cost"],
            fp_cost=cost_matrix["fp_cost"],
            fn_cost=cost_matrix["fn_cost"],
            tp_cost=cost_matrix["tp_cost"],
        )
        
        # Add cost metrics to the metrics dictionary with 'cost_' prefix
        metrics.update({
            "cost_total": cost_metrics["total_cost"],
            "cost_average": cost_metrics["average_cost"],
            "cost_tn_count": cost_metrics["tn_count"],
            "cost_fp_count": cost_metrics["fp_count"],
            "cost_fn_count": cost_metrics["fn_count"],
            "cost_tp_count": cost_metrics["tp_count"],
            "cost_tn_total": cost_metrics["tn_total_cost"],
            "cost_fp_total": cost_metrics["fp_total_cost"],
            "cost_fn_total": cost_metrics["fn_total_cost"],
            "cost_tp_total": cost_metrics["tp_total_cost"],
        })
        
        # Print cost breakdown
        pprint("\n[bold cyan]Cost Analysis:[/bold cyan]")
        pprint(f"Total Cost: {cost_metrics['total_cost']:.2f}")
        pprint(f"Average Cost per Prediction: {cost_metrics['average_cost']:.4f}")
        pprint("\nCost Breakdown:")
        pprint(f"  TN: {cost_metrics['tn_count']} × {cost_matrix.get('tn_cost', 0.0)} = {cost_metrics['tn_total_cost']:.2f}")
        pprint(f"  FP: {cost_metrics['fp_count']} × {cost_matrix.get('fp_cost', 5.0)} = {cost_metrics['fp_total_cost']:.2f}")
        pprint(f"  FN: {cost_metrics['fn_count']} × {cost_matrix.get('fn_cost', 100.0)} = {cost_metrics['fn_total_cost']:.2f}")
        pprint(f"  TP: {cost_metrics['tp_count']} × {cost_matrix.get('tp_cost', 0.0)} = {cost_metrics['tp_total_cost']:.2f}")

    # Plot confusion matrix using TBM-specific styling
    cm_fig = plot_tbm_confusion_matrix(
        y_test, y_pred, class_mapping, normalize="true", show_percentages=True
    )
    artifacts = {"confusion_matrix": cm_fig}

    # Print classification report
    pprint("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics, artifacts


def log_mlflow_metrics_and_model(
    mlflow_path: Path,
    experiment_name: str,
    metrics: dict,
    artifacts: dict,
    model_name: str,
    model_params: dict,
    undersample_level: int,
    oversample_level: int,
    hydra_cfg_dir: Path,
) -> None:
    # Setting MLflow experiment and tracking URI
    mlflow.set_tracking_uri(mlflow_path)
    mlflow.set_experiment(experiment_name=experiment_name)

    # Logging to MLflow
    with mlflow.start_run():
        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model details
        model_details = {
            "model_name": model_name,
            "scaler": "StandardScaler",
            "undersample_level": undersample_level,
            "oversample_level": oversample_level,
        }
        mlflow.log_params(model_details)
        mlflow.log_params(model_params)

        # Log confusion matrix, and eventual other figures as artifact
        for name, fig in artifacts.items():
            mlflow.log_figure(fig, f"{name}.png")

        # Log Hydra config files as artifacts
        hydra_configs = [f for f in hydra_cfg_dir.iterdir() if f.suffix == ".yaml"]

        # Log paths of Hydra config files as MLflow parameters
        hydra_cfg_paths = []
        for config_file in hydra_configs:
            mlflow.log_artifact(str(config_file), artifact_path="hydra_configs")
            hydra_cfg_paths.append(str(config_file))

        # Log paths as MLflow parameters
        hydra_cfg_path_str = ", ".join(hydra_cfg_paths)
        mlflow.log_param("hydra_config_paths", hydra_cfg_path_str)
