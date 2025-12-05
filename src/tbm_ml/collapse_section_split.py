"""
Helper functions for splitting TBM data by collapse sections.

These functions enable stratified splitting that keeps continuous collapse
sections intact, preventing data leakage.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple


def split_by_collapse_sections(
    df: pd.DataFrame,
    train_size: float = 0.75,
    random_state: int = 42,
    label_column: str = "collapse",
    section_column: str = "collapse_section"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset ensuring collapse sections stay together.
    
    This uses GroupShuffleSplit to ensure that all samples from a given
    collapse section are kept together in either train or test set.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The complete dataset with collapse_section column
    train_size : float
        Fraction of data for training (default: 0.75)
    random_state : int
        Random seed for reproducibility
    label_column : str
        Name of the label column (default: "collapse")
    section_column : str
        Name of the collapse section column (default: "collapse_section")
    
    Returns:
    --------
    train_df, test_df : Tuple[pd.DataFrame, pd.DataFrame]
        Training and testing dataframes
    """
    
    # Create groups: use section_column for collapse samples, unique ID for non-collapse
    groups = df[section_column].copy()
    non_collapse_mask = groups == 0
    groups.loc[non_collapse_mask] = -df.loc[non_collapse_mask].index
    
    # Use GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, df[label_column], groups=groups))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # Print statistics
    print(f"\nTrain/Test Split by Collapse Sections:")
    print(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    - Collapse: {train_df[label_column].sum():,}")
    print(f"    - Non-collapse: {(train_df[label_column]==0).sum():,}")
    print(f"  Test: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"    - Collapse: {test_df[label_column].sum():,}")
    print(f"    - Non-collapse: {(test_df[label_column]==0).sum():,}")
    
    train_sections = sorted(train_df[train_df[section_column] > 0][section_column].unique())
    test_sections = sorted(test_df[test_df[section_column] > 0][section_column].unique())
    print(f"\n  Train collapse sections: {train_sections}")
    print(f"  Test collapse sections: {test_sections}")
    
    return train_df, test_df


def split_by_section_range(
    df: pd.DataFrame,
    train_sections: list[int],
    test_sections: list[int],
    label_column: str = "collapse",
    section_column: str = "collapse_section"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by manually specifying which sections go to train/test.
    
    This allows for deterministic splitting based on section IDs, useful for
    temporal or spatial splits where sections have a natural ordering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The complete dataset with collapse_section column
    train_sections : list[int]
        List of collapse section IDs for training
    test_sections : list[int]
        List of collapse section IDs for testing
    label_column : str
        Name of the label column (default: "collapse")
    section_column : str
        Name of the collapse section column (default: "collapse_section")
    
    Returns:
    --------
    train_df, test_df : Tuple[pd.DataFrame, pd.DataFrame]
        Training and testing dataframes
    """
    
    # Separate collapse and non-collapse samples
    collapse_df = df[df[section_column] > 0].copy()
    non_collapse_df = df[df[section_column] == 0].copy()
    
    # Split collapse samples by section
    train_collapse = collapse_df[collapse_df[section_column].isin(train_sections)]
    test_collapse = collapse_df[collapse_df[section_column].isin(test_sections)]
    
    # Calculate proportions for non-collapse samples
    total_collapse = len(train_collapse) + len(test_collapse)
    train_collapse_fraction = len(train_collapse) / total_collapse if total_collapse > 0 else 0.75
    
    # Split non-collapse samples proportionally
    non_collapse_shuffled = non_collapse_df.sample(frac=1, random_state=42)
    n_train_non_collapse = int(len(non_collapse_shuffled) * train_collapse_fraction)
    train_non_collapse = non_collapse_shuffled.iloc[:n_train_non_collapse]
    test_non_collapse = non_collapse_shuffled.iloc[n_train_non_collapse:]
    
    # Combine
    train_df = pd.concat([train_collapse, train_non_collapse], axis=0).sample(frac=1, random_state=42)
    test_df = pd.concat([test_collapse, test_non_collapse], axis=0).sample(frac=1, random_state=42)
    
    # Print statistics
    print(f"\nTrain/Test Split by Section Range:")
    print(f"  Train sections: {train_sections}")
    print(f"  Test sections: {test_sections}")
    print(f"\n  Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    - Collapse: {train_df[label_column].sum():,}")
    print(f"    - Non-collapse: {(train_df[label_column]==0).sum():,}")
    print(f"  Test: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"    - Collapse: {test_df[label_column].sum():,}")
    print(f"    - Non-collapse: {(test_df[label_column]==0).sum():,}")
    
    return train_df, test_df


if __name__ == "__main__":
    # Example usage
    print("Example usage of collapse section splitting:")
    print("\n" + "="*80)
    
    # Load data
    df = pd.read_csv("data/model_ready/dataset_total.csv")
    
    # Method 1: Random group shuffle split
    print("\nMethod 1: Random GroupShuffleSplit (keeps sections together)")
    print("-" * 80)
    train_df1, test_df1 = split_by_collapse_sections(df, train_size=0.75, random_state=42)
    
    # Method 2: Manual section assignment
    print("\n" + "="*80)
    print("\nMethod 2: Manual Section Range Split (sections 1-15 train, 16-18 test)")
    print("-" * 80)
    train_sections = list(range(1, 16))
    test_sections = list(range(16, 19))
    train_df2, test_df2 = split_by_section_range(df, train_sections, test_sections)
    
    print("\n" + "="*80)
    print("Example complete!")
