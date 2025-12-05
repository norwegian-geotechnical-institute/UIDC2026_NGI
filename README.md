# UIDC2026_NGI

Repository for code and results of NGI's contribution to the 1st International Underground Infrastructure Digital Challenge (UIDC 2026)

## Overview

This project develops machine learning models for predicting Tunnel Boring Machine (TBM) collapse events using operational data from the Yinsong Water Diversion Project. The codebase includes data preprocessing, feature engineering, hyperparameter optimization, and model evaluation pipelines with MLflow experiment tracking.

## Project Structure

```
├── scripts/                          # Main execution scripts
│   ├── batch_optimize.py            # Batch hyperparameter optimization for multiple models
│   ├── optimize_hyperparameters.py  # Single model hyperparameter optimization
│   ├── train.py                     # Model training with best hyperparameters
│   ├── preprocess.py                # Data preprocessing pipeline
│   ├── undersampling_cost_analysis.py  # Analyze undersampling fraction vs prediction cost
│   ├── generate_cv_results_table.py # Generate results table from YAML files
│   ├── analyze_collapse_sections.py # Analyze collapse section distributions
│   ├── permutation_feature_importance.py  # Feature importance analysis
│   ├── EDA.py                       # Exploratory data analysis
│   └── config/                      # Hydra configuration files
│       └── main.yaml                # Main configuration (CV folds, sampling, MLflow)
├── src/tbm_ml/                      # Core ML library
│   ├── hyperparameter_optimization.py  # Optuna-based hyperparameter search with StratifiedGroupKFold
│   ├── train_eval_funcs.py          # Training pipelines and evaluation
│   ├── preprocess_funcs.py          # Data preprocessing functions
│   ├── collapse_section_split.py    # Collapse section-aware train/test splitting
│   ├── plotting.py                  # Visualization utilities
│   └── schema_config.py             # Pydantic configuration schemas
├── data/                            # Dataset storage
│   ├── raw/                         # Raw data files
│   ├── intermediate/                # Processed intermediate data
│   └── model_ready/                 # Train/test splits ready for modeling
├── experiments/                     # Experiment outputs
│   ├── mlruns/                      # MLflow tracking data
│   └── hyperparameters/             # Best hyperparameter YAML files
├── analyses/                        # Analysis outputs
│   ├── undersampling_analysis/      # Cost vs undersampling fraction analysis
│   └── profile_report/              # Data profiling reports
└── docs/                            # Documentation
```

## Key Features

### Data Preprocessing

- **Outlier Detection**: Isolation Forest-based outlier removal
- **Collapse Section Tracking**: Identifies and indexes continuous collapse sections for group-aware CV
- **Class Imbalance Handling**:
  - RandomUnderSampler for majority class reduction (ratio-based or absolute)
  - SMOTE for minority class oversampling (not used in final models)
  - Configurable undersampling ratio (e.g., 1.0 = 1:1 majority:minority ratio)
- **Feature Scaling**: StandardScaler normalization
- **Train/Test Split**: Section-aware split (sections 1-15 train, 16-18 test) preserving collapse section integrity

### Model Support

The framework supports 11 machine learning models:

- **Gradient Boosting**: XGBoost (native & sklearn), LightGBM, CatBoost, HistGradientBoosting
- **Ensemble Methods**: Random Forest, Extra Trees
- **Traditional ML**: Logistic Regression, SVM, KNN
- **Gaussian Process**: For probabilistic predictions

### Hyperparameter Optimization

- **Framework**: Optuna with TPE (Tree-structured Parzen Estimator) sampler
- **Cross-Validation**: StratifiedGroupKFold (5-fold) to keep collapse sections together during CV
- **Metrics**: Cost-based optimization (minimizing expected prediction cost), balanced accuracy, precision, recall, F1-score
- **Cost Matrix**: Configurable costs for TN, FP, FN, TP with time per regular advance multiplier
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

### 0. Run the preliminary tests script

Note: This is just to generate the merged csv file used in subsequent steps.

```bash
python scripts/preliminary_tests.py
```

### 1. Data Preprocessing

```bash
python scripts/preprocess.py
```

Processes raw TBM data, handles outliers, and creates train/test splits.

### 2. Hyperparameter Optimization

#### Single Model

```bash
python scripts/optimize_hyperparameters.py model_name=xgboost
```

#### Batch Optimization (Multiple Models)

```bash
python scripts/batch_optimize.py
```

Runs optimization for all configured models sequentially.

### 3. Model Training

```bash
python scripts/train.py
```

Trains models using the best hyperparameters found during optimization.

### 4. Undersampling Cost Analysis

```bash
python scripts/undersampling_cost_analysis.py
```

Analyzes how different undersampling fractions affect prediction cost. Generates:

- Cost vs undersampling fraction plots (CV and test set)
- Accuracy metrics plots (CV and test set)
- Confusion matrices for different undersampling ratios
- Results CSV with optimal undersampling fraction

### 5. MLflow UI

```bash
python -m mlflow ui --backend-store-uri ./experiments/mlruns --port 5000
```

View experiments at <http://127.0.0.1:5000>

## Configuration

Main configuration file: `scripts/config/main.yaml`

Key parameters:

```yaml
experiment:
  undersample_level: null  # Absolute number (overrides ratio if set), or null to use ratio
  undersample_ratio: 1.0   # Majority:minority ratio (1.0 = 1:1, 2.0 = 2:1, etc.)
  oversample_level: 0      # Minority class SMOTE (0=disabled)
  
  # Cost matrix for prediction cost calculation (in hours per regular advance)
  cost_matrix:
    tn_cost: 1.0           # True Regular (correct prediction of regular)
    fp_cost: 10.0          # False Collapse (false alarm)
    fn_cost: 240.0         # False Regular (missed collapse - most costly)
    tp_cost: 10.0          # True Collapse (correct prediction of collapse)
    time_per_regular_advance: 1.0  # Time unit multiplier

optuna:
  n_trials: 100            # Optuna trials per model
  cv_folds: 5              # Cross-validation folds (uses StratifiedGroupKFold)
  path_results: experiments/hyperparameters

mlflow:
  path: ./experiments/mlruns
  experiment_name: null    # Set programmatically by scripts
```

## Models Configuration

The models are configured in `scripts/batch_optimize.py`:

```python
# Models that undergo hyperparameter optimization
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

# Baseline models (evaluated without optimization)
BASELINE_MODELS = [
    "dummy",  # Always predicts majority class
]
```

## Output Files

### Hyperparameter Files

Best hyperparameters saved to `experiments/hyperparameters/`:

```
best_hyperparameters_{model_name}_{timestamp}.yaml
```

Each file contains:

- Optimized hyperparameters
- CV performance metrics (cost-based)
- Test set metrics (accuracy, balanced accuracy, recall, precision, F1, cost)
- Optimization metadata (trial number, duration, timestamp)

### Analysis Outputs

- `analyses/undersampling_analysis/{timestamp}/` - Undersampling analysis plots and data

### MLflow Artifacts

- Optimization history plots
- Parameter importance plots
- Model metrics and parameters

## Data

The data in folder 'YS-IWHR-main' is a copy of the data provided in the main branch of the repository <https://github.com/ChenZuyuIWHR/YS-IWHR>
