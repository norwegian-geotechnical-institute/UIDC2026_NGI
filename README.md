# UIDC2026_NGI

Repository for code and results of NGI's contribution to the 1st International Underground Infrastructure Digital Challenge (UIDC 2026)

## Overview

This project develops machine learning models for predicting Tunnel Boring Machine (TBM) collapse events using operational data from the Yinsong Water Diversion Project. The codebase includes comprehensive data preprocessing, feature engineering, hyperparameter optimization, and model evaluation pipelines with MLflow experiment tracking.

## Project Structure

```
├── scripts/                          # Main execution scripts
│   ├── batch_optimise.py            # Batch hyperparameter optimization for multiple models
│   ├── optimise_hyperparameters.py  # Single model hyperparameter optimization
│   ├── train.py                     # Model training with best hyperparameters
│   ├── preprocess.py                # Data preprocessing pipeline
│   ├── select_features.py           # Feature selection utilities
│   ├── EDA.py                       # Exploratory data analysis
│   └── config/                      # Hydra configuration files
│       └── main.yaml                # Main configuration (CV folds, sampling, MLflow)
├── src/tbm_ml/                      # Core ML library
│   ├── hyperparameter_optimisation.py  # Optuna-based hyperparameter search
│   ├── train_eval_funcs.py          # Training pipelines and evaluation
│   ├── preprocess_funcs.py          # Data preprocessing functions
│   ├── plotting.py                  # Visualization utilities
│   └── schema_config.py             # Pydantic configuration schemas
├── data/                            # Dataset storage
│   ├── raw/                         # Raw data files
│   ├── intermediate/                # Processed intermediate data
│   └── model_ready/                 # Train/test splits ready for modeling
├── experiments/                     # Experiment outputs
│   ├── mlruns/                      # MLflow tracking data
│   └── hyperparameters/             # Best hyperparameter YAML files
└── docs/                            # Documentation

```

## Key Features

### Data Preprocessing

- **Outlier Detection**: Isolation Forest-based outlier removal
- **Class Imbalance Handling**:
  - RandomUnderSampler for majority class reduction
  - SMOTE for minority class oversampling
- **Feature Scaling**: StandardScaler normalization
- **Train/Test Split**: Stratified split preserving class distributions

### Model Support

The framework supports 11 machine learning models:

- **Gradient Boosting**: XGBoost (native & sklearn), LightGBM, CatBoost, HistGradientBoosting
- **Ensemble Methods**: Random Forest, Extra Trees
- **Traditional ML**: Logistic Regression, SVM, KNN
- **Gaussian Process**: For probabilistic predictions

### Hyperparameter Optimization

- **Framework**: Optuna with TPE (Tree-structured Parzen Estimator) sampler
- **Cross-Validation**: Stratified K-Fold (configurable, default 5-fold)
- **Metrics**: Balanced accuracy (primary), precision, recall, F1-score
- **Trials**: 100 trials per model (configurable)
- **Logging**: MLflow integration for experiment tracking and visualization

### Experiment Tracking

- **MLflow Integration**: Automatic logging of metrics, parameters, and artifacts
- **Visualizations**: Optimization history and parameter importance plots
- **Version Control**: All hyperparameters saved as timestamped YAML files

## Installation

### Prerequisites

- Python 3.12+
- uv (Python package manager)

### Setup

```bash
# Clone the repository
git clone https://github.com/norwegian-geotechnical-institute/UIDC2026_NGI.git
cd UIDC2026_NGI

# Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate      # Linux/Mac

# Install dependencies (if using uv)
uv sync
```

## Usage

### 1. Data Preprocessing

```bash
python scripts/preprocess.py
```

Processes raw TBM data, handles outliers, and creates train/test splits.

### 2. Hyperparameter Optimization

#### Single Model

```bash
python scripts/optimise_hyperparameters.py model_name=xgboost
```

#### Batch Optimization (Multiple Models)

```bash
python scripts/batch_optimise.py
```

Runs optimization for all configured models sequentially.

### 3. Model Training

```bash
python scripts/train.py
```

Trains models using the best hyperparameters found during optimization.

### 4. MLflow UI

```bash
python -m mlflow ui --backend-store-uri ./experiments/mlruns --port 5000
```

View experiments at <http://127.0.0.1:5000>

## Configuration

Main configuration file: `scripts/config/main.yaml`

Key parameters:

```yaml
data:
  train_path: "data/model_ready/dataset_train.csv"
  test_path: "data/model_ready/dataset_test.csv"

preprocessing:
  undersample_level: 2000  # Majority class samples
  oversample_level: 0      # Minority class SMOTE (0=disabled)
  outlier_removal: true

optimization:
  n_trials: 100            # Optuna trials per model
  cv_folds: 5              # Cross-validation folds
  metric: "balanced_accuracy"

mlflow:
  path: "./experiments/mlruns"
  experiment_name: null    # Set programmatically by scripts
```

## Models Configuration

Edit `scripts/batch_optimise.py` to select models:

```python
MODELS = [
    "xgboost",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "catboost",
    "lightgbm",
    "logistic_regression",
    "svm",
    "knn",
]
```

## Output Files

### Hyperparameter Files

Best hyperparameters saved to `experiments/hyperparameters/`:

```
best_hyperparameters_{model_name}_{timestamp}.yaml
```

### MLflow Artifacts

- Optimization history plots
- Parameter importance plots
- Model metrics and parameters

## Data

The data in folder 'YS-IWHR-main' is a copy of the data provided in the main branch of the repository <https://github.com/ChenZuyuIWHR/YS-IWHR>
