"""Utilities for time series data splitting."""

import numpy as np


def train_val_test_split(data, train_ratio=0.7, val_ratio=0.15):
    """Split time series data into train, validation, and test sets."""
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data

