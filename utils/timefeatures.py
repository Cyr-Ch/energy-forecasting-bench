"""Time features extraction for time series datasets."""

import numpy as np
import pandas as pd


def time_features(dates, freq='h'):
    """
    Extract time features from datetime.
    
    Args:
        dates: array-like of datetime objects
        freq: frequency string ('h'=hourly, 't'=minutely, 'd'=daily, etc.)
    
    Returns:
        features: [n, d] array of time features
    """
    dates = pd.to_datetime(dates)
    features = []
    
    if freq == 'h' or freq == 'H':
        # Hourly features
        features.append(dates.month.values)
        features.append(dates.day.values)
        features.append(dates.weekday.values)
        features.append(dates.hour.values)
    elif freq == 't' or freq == 'T':
        # 15-minute features
        features.append(dates.month.values)
        features.append(dates.day.values)
        features.append(dates.weekday.values)
        features.append(dates.hour.values)
        features.append((dates.minute.values // 15).astype(int))
    elif freq == 'd' or freq == 'D':
        # Daily features
        features.append(dates.month.values)
        features.append(dates.day.values)
        features.append(dates.weekday.values)
    elif freq == 'w' or freq == 'W':
        # Weekly features
        features.append(dates.month.values)
        features.append((dates.day // 7).values)
        features.append(dates.weekday.values)
    elif freq == 'm' or freq == 'M':
        # Monthly features
        features.append(dates.month.values)
    elif freq == 'a' or freq == 'A':
        # Yearly features
        features.append(dates.year.values)
    else:
        # Default: hourly features
        features.append(dates.month.values)
        features.append(dates.day.values)
        features.append(dates.weekday.values)
        features.append(dates.hour.values)
    
    features = np.array(features).transpose(1, 0)  # [n, d]
    return features

