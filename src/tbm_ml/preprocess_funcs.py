from pathlib import Path

import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.mad import MAD
from rich.pretty import pprint

from tbm_ml.utility import track_sample_num


def get_dataset(path_file: Path) -> pd.DataFrame:
    """Read dataset."""
    df = pd.read_csv(path_file, header=0, sep=",")
    return df


def add_collapse_section_index(df: pd.DataFrame, label_column: str = "collapse", 
                                 sort_column: str = "Chainage", ascending: bool = False) -> pd.DataFrame:
    """
    Add a column identifying continuous collapse sections.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        label_column (str): Name of the collapse label column. Defaults to "collapse".
        sort_column (str): Column to use for ordering (e.g., "Chainage" or "Tunnellength [m]").
        ascending (bool): Sort order. Use False for Chainage (decreasing along tunnel), 
                         True for Tunnellength (increasing along tunnel).
    
    Returns:
        pd.DataFrame: DataFrame with added 'collapse_section' column.
                     Non-collapse rows have collapse_section = 0.
                     Collapse rows are numbered 1, 2, 3, ... for each continuous section.
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Sort by the specified column to ensure proper ordering along the tunnel
    if sort_column in df.columns:
        df = df.sort_values(sort_column, ascending=ascending).reset_index(drop=True)
    else:
        pprint(f"Warning: Sort column '{sort_column}' not found. Using existing order.")
    
    # Identify continuous collapse sections
    # A new section starts when current row is collapse and previous row is not collapse
    df['is_collapse'] = df[label_column] == 1
    df['collapse_section'] = (df['is_collapse'] & (~df['is_collapse'].shift(1, fill_value=False))).cumsum()
    
    # Set section to 0 for non-collapse rows
    df.loc[~df['is_collapse'], 'collapse_section'] = 0
    
    # Convert to integer type
    df['collapse_section'] = df['collapse_section'].astype(int)
    
    # Drop temporary column
    df = df.drop(columns=['is_collapse'])
    
    # Report statistics
    n_sections = df[df['collapse_section'] > 0]['collapse_section'].nunique()
    if n_sections > 0:
        section_sizes = df[df['collapse_section'] > 0].groupby('collapse_section').size()
        pprint(f"Found {n_sections} continuous collapse sections")
        pprint(f"Section sizes - Min: {section_sizes.min()}, Max: {section_sizes.max()}, "
               f"Mean: {section_sizes.mean():.1f}, Median: {section_sizes.median():.1f}")
    else:
        pprint("No collapse sections found")
    
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
    add_collapse_sections: bool = True,
    sort_column: str = "Chainage",
    sort_ascending: bool = False,
) -> pd.DataFrame:
    """
    Preprocess dataset.
    
    Args:
        path_file: Path to the raw dataset file
        features: List of feature column names
        site_features: List of site information column names
        labels: Name of the label column
        outlier_feature: Feature to use for univariate outlier detection
        remove_duplicates: Whether to remove duplicate rows
        remove_outliers_hard: Whether to remove hardcoded outliers
        remove_outliers_uni: Whether to remove univariate outliers
        remove_outliers_multi: Whether to remove multivariate outliers
        univariate_threshold: Threshold for univariate outlier detection
        multivariate_threshold: Threshold for multivariate outlier detection
        add_collapse_sections: Whether to add collapse section indices (default: True)
        sort_column: Column to use for sorting when identifying collapse sections
        sort_ascending: Sort order for collapse section identification
    
    Returns:
        pd.DataFrame: Preprocessed dataset with optional collapse_section column
    """
    df = get_dataset(path_file)
    pprint("Dataset loaded")
    
    # Add collapse section index before feature selection if enabled
    if add_collapse_sections:
        df = add_collapse_section_index(df, label_column=labels, 
                                        sort_column=sort_column, ascending=sort_ascending)
        # Include collapse_section in the selected features
        df = choose_features(df, features=site_features + features + [labels, 'collapse_section'])
    else:
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
