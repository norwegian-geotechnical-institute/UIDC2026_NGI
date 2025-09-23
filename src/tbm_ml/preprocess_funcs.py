from pathlib import Path

import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.mad import MAD
from rich.pretty import pprint
from sklearn.model_selection import train_test_split

from tbm_ml.utility import track_sample_num


def get_dataset(path_file: Path) -> pd.DataFrame:
    """Read dataset."""
    df = pd.read_csv(path_file, header=0, sep=",")
    return df


@track_sample_num
def choose_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Choose features for dataset."""
    df = df[features]
    return df


@track_sample_num
def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NA values."""
    df = df.dropna()
    return df


@track_sample_num
def drop_duplicates(df: pd.DataFrame, duplicate_features: list[str]) -> pd.DataFrame:
    """Drop duplicated rows."""
    df = df.drop_duplicates(subset=duplicate_features)
    return df


@track_sample_num
def remove_outliers_hardcoded(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers based on hardcoded values."""
    # If any hardcoded removal is needed, do it here
    # Example: Remove rows where 'feature1' > 100 or 'feature2'
    # return df
    raise NotImplementedError("Hardcoded outlier removal not implemented yet.")


@track_sample_num
def remove_outliers_univariate(
    df: pd.DataFrame, feature: str, threshold: float
) -> pd.DataFrame:
    """
    Removes outliers from a dataframe based on the MAD (Median Absolute Deviation) method.
    """
    # Initialize the MAD model with the provided threshold
    mad = MAD(threshold=threshold)

    # Fit the model on the specified feature
    mad.fit(df[[feature]])

    # Predict outliers (1 for outlier, 0 for inlier)
    outliers = mad.predict(df[[feature]])

    # Filter the DataFrame to exclude outliers
    df_no_outliers = df[outliers == 0]

    return df_no_outliers


@track_sample_num
def remove_outliers_multivariate(
    df: pd.DataFrame, features: list[str], confidence_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Removes outliers from a DataFrame using the Isolation Forest model.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list[str]): List of feature column names to consider for outlier detection.
        confidence_threshold (float): The threshold for outlier confidence. Defaults to 0.95.

    Returns:
        pd.DataFrame: A DataFrame excluding detected outliers.
    """
    # Initialize and fit the Isolation Forest model
    iforest = IForest(n_estimators=100)
    iforest.fit(df[features])

    # Get the outlier probabilities
    probs = iforest.predict_proba(df[features])[:, 1]

    # Create a mask for outliers based on the confidence threshold
    is_outlier = probs > confidence_threshold

    # Display results
    outliers = df[is_outlier]
    num_outliers = len(outliers)
    print(f"Number of outliers with Isolation Forest: {num_outliers}")
    print(f"Percentage of outliers: {num_outliers / len(df):.4f}")
    print("Outlier samples:\n", outliers)

    # Return DataFrame excluding outliers
    return df[~is_outlier]


def preprocess_data(
    path_file: str,
    features: list,
    site_features: list,
    labels: str,
    outlier_feature: str,
    remove_duplicates: bool,
    remove_outliers_hard: bool,
    remove_outliers_uni: bool,
    remove_outliers_multi: bool,
    univariate_threshold: int = 3,
    multivariate_threshold=0.95,
) -> pd.DataFrame:
    """Preprocess dataset."""
    df = get_dataset(path_file)
    pprint("Dataset loaded")
    df = choose_features(df, features=site_features + features + [labels])
    df = drop_na(df)
    pprint("NA values dropped")
    if remove_duplicates:
        df = drop_duplicates(df, features)
        pprint("Duplicates dropped")
    if remove_outliers_hard:
        df = remove_outliers_hardcoded(df)
        pprint("Hardcoded outliers removed")
    if remove_outliers_uni:
        df = remove_outliers_univariate(
            df, outlier_feature, threshold=univariate_threshold
        )
        pprint("Univariate outliers removed")
    if remove_outliers_multi:
        df = remove_outliers_multivariate(
            df, features, confidence_threshold=multivariate_threshold
        )
        pprint("Multivariate outliers removed")
    if df.empty:
        raise ValueError("DataFrame is empty after preprocessing")
    return df
