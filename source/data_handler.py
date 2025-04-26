# data_handler.py
"""
Handles loading, preprocessing, and splitting of time series data.
"""
import pandas as pd
import numpy as np
import warnings

def load_and_split_data(file_path, train_ratio=0.6, val_ratio=0.2):
    """
    Loads data from CSV, calculates returns, and splits chronologically.

    Args:
        file_path (str): Path to the CSV file.
        train_ratio (float): Proportion of data for the training set.
        val_ratio (float): Proportion of data for the validation set.

    Returns:
        tuple: (train_data, val_data, test_data) pandas DataFrames.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None, None, None

    # Date parsing
    try:
        data['Open time'] = pd.to_datetime(data['Open time'])
        data['Close time'] = pd.to_datetime(data['Close time'])
    except Exception as e:
        warnings.warn(f"Date parsing warning (non-critical): {e}")

    # Calculate future 1-minute returns carefully
    data.sort_values(by='Open time', inplace=True) # Ensure chronological order
    data['Return'] = data['Close'].shift(-1) / data['Close'] - 1

    # Drop rows where 'Return' cannot be calculated (last row)
    # Also drop rows if essential features are missing
    initial_len = len(data)
    essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return']
    data.dropna(subset=essential_cols, inplace=True)
    if len(data) < initial_len:
        print(f"Dropped {initial_len - len(data)} rows due to NaNs in essential columns.")

    if len(data) < 100: # Arbitrary minimum length for meaningful splits
        print("Error: Not enough data after cleaning for splitting.")
        return None, None, None

    # Chronological splitting
    n = len(data)
    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end_idx].copy()
    val_data = data.iloc[train_end_idx:val_end_idx].copy()
    test_data = data.iloc[val_end_idx:].copy()

    print(f"Data Split: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")

    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        print("Error: One or more data splits are empty. Check ratios and data length.")
        return None, None, None

    return train_data, val_data, test_data