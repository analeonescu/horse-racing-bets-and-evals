"""Data loading and chronological splitting utilities."""

import pandas as pd


def load_data(path):
    """Load a race dataset from CSV."""
    
    return pd.read_csv(path)


def sort_races_chronologically(df: pd.DataFrame, date_col="race_date"):
    """Return a copy sorted by race date."""
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    
    return result.sort_values(date_col).reset_index(drop=True)


def chronological_split(df: pd.DataFrame, date_col="race_date", test_size=0.2):
    """Split data chronologically without shuffling."""
    result = sort_races_chronologically(df, date_col)
    split_idx = int(len(result) * (1 - test_size))
    
    return result.iloc[:split_idx].copy(), result.iloc[split_idx:].copy()
