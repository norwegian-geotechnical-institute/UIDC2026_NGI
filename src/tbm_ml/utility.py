from typing import Callable, Optional, Tuple

import pandas as pd
import xgboost as xgb
from rich.console import Console


def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    # Load the dataset from the specified path
    df = pd.read_csv(file_path)

    # Clean column names to avoid issues with whitespace or special characters
    df.columns = df.columns.str.strip()

    # Define selected features
    SELECTED_FEATURES = [
        'T(P,V/N)-b1',
        'T(P,V/N)-b2',
        'T(P,V/N)-a',
        'T(P,V/N)-R2',
        'T(V/N)-b',
        'T(V/N)-a',
        'T(V/N)-R2',
        'P(V)-slope',
        'P(V)-a',
        'P(V)-R2',
        'Wp(WT)-slope',
        'Wp(WT)-a',
        'Wp(WT)-R2',
        'T(P)-slope',
        'T(P)-a',
        'T(P)-R2',
        'T-m',
        'T-slope',
        'T-R2',
        'P-m',
        'P-slope',
        'P-R2',
        'V-m',
        'V-slope',
        'V-R2',
        'Vs-m',
        'Vs-slope',
        'TN-m',
        'TN-slope',
        'TN-R2',
        'PV-m',
        'PV-slope',
        'PV-R2',
        'V/N-m',
        'V/N-slope',
        'V/N-R2',
        'N-m',
        'Ns-m',
        'I-m',
        'V0-m',
        'V/Vs-min',
        'T/P-min',
        'T/P-max',
        'T/P-m',
        'T/P-std',
        'Tpi-min',
        'Tpi-max',
        'Tpi-m',
        'Tpi-std',
        'Fpi-min',
        'Fpi-max',
        'Fpi-m',
        'Fpi-std',
        'Tpi-slope',
        'Tpi-R2:1-sse/sst',
        'WT/WP-slope',
        'WT/WP-R2:1-sse/sst',
        'T(P*V/N)-slope',
        'T(P*V/N)-a',
        'T(P*V/N)-R2',
        'Fpi',
        'P*V/N',
        'T/P',
        'Tpi',
        'dt',
        'dz',
        'workP',
        'workT',
        'Total T', 'Total F', 'RPM', 'Pr', 'p'
    ]

    # Extract the features
    features = df[SELECTED_FEATURES].copy()

    # Extract labels if the column exists
    labels = None
    if "collapse" in df.columns:
        labels = df["collapse"].copy()

    return df, features, labels


# Loading the saved model for future use
def load_xgb_model(model_path: str) -> xgb.Booster:
    loaded_model = xgb.Booster()
    loaded_model.load_model(model_path)
    return loaded_model


def track_sample_num(func: Callable) -> Callable:
    """Tracking number of samples of a dataframe before and after processing."""
    console = Console()

    def df_processing(*args: int, **kwargs: int) -> pd.DataFrame:
        res = func(*args, **kwargs)
        for data in args:
            if isinstance(data, pd.DataFrame):
                console.print("--------------------------------")
                console.print(
                    f"Number of samples before processing with {func.__name__} function"
                    f" (rows,cols): {data.shape}"
                )
                console.print(
                    f"Number of samples after processing with {func.__name__} function"
                    f" (rows,cols): {res.shape}"
                )
                console.print("--------------------------------")
        return res

    return df_processing


def info_dataset(
    df_main: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    label: str,
) -> None:
    """Print info about dataset."""
    console = Console()
    console.rule()
    console.print("\nFirst five rows:")
    console.print(df_main.head())
    console.rule()
    console.print(df_main.info())
    console.rule()
    console.print(
        f"\nA fantastic dataset of {df_main.shape[0]} samples is built :smiley:"
    )
    console.print(f"Num samples trainset: {df_train.shape[0]}")
    console.print(f"Num samples testset: {df_test.shape[0]}")
    console.rule()
    # value counts train
    console.print("\nValue counts trainset:")
    console.print(df_train[label].value_counts())
    # value counts test
    console.print("\nValue counts testset:")
    console.print(df_test[label].value_counts())
