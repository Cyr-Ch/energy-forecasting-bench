"""Evaluation metrics for time series forecasting."""

import numpy as np
from typing import Optional, Union


def mae(y, yhat):
    """Mean Absolute Error."""
    return np.mean(np.abs(y - yhat))


def rmse(y, yhat):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y - yhat)**2))


def mape(y, yhat, eps=1e-8):
    """
    Mean Absolute Percentage Error.
    
    Args:
        y: True values
        yhat: Predicted values
        eps: Small value to avoid division by zero
    
    Returns:
        MAPE as percentage
    """
    return 100 * np.mean(np.abs((y - yhat) / (y + eps)))


def smape(y, yhat, eps=1e-8):
    """Symmetric Mean Absolute Percentage Error."""
    return 100 * np.mean(2*np.abs(yhat - y) / (np.abs(y)+np.abs(yhat)+eps))


def mase(y, yhat, y_train, season):
    """
    Mean Absolute Scaled Error with seasonal naive baseline.
    
    Args:
        y: True values
        yhat: Predicted values
        y_train: Training data for computing naive forecast
        season: Seasonal period (e.g., 24 for hourly, 96 for 15-min)
    
    Returns:
        MASE value (lower is better, < 1 means better than naive)
    """
    naive = np.mean(np.abs(y_train[season:] - y_train[:-season])) + 1e-8
    return np.mean(np.abs(y - yhat)) / naive


def crps(y_true, y_pred, y_lower=None, y_upper=None):
    """
    Continuous Ranked Probability Score (CRPS) for probabilistic forecasts.
    
    CRPS measures the accuracy of probabilistic forecasts by comparing the
    predicted cumulative distribution function (CDF) to the observed value.
    
    Args:
        y_true: True values [n_samples] or [n_samples, n_features]
        y_pred: Predicted mean values [n_samples] or [n_samples, n_features]
        y_lower: Lower bound of prediction interval [n_samples] or [n_samples, n_features]
        y_upper: Upper bound of prediction interval [n_samples] or [n_samples, n_features]
    
    Returns:
        CRPS value (lower is better)
    
    Note:
        If y_lower and y_upper are not provided, this reduces to MAE.
        For proper probabilistic CRPS, provide prediction intervals or samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # If no intervals provided, CRPS reduces to MAE
    if y_lower is None or y_upper is None:
        return mae(y_true, y_pred)
    
    y_lower = np.asarray(y_lower)
    y_upper = np.asarray(y_upper)
    
    # Ensure all arrays have same shape
    assert y_true.shape == y_pred.shape == y_lower.shape == y_upper.shape
    
    # CRPS for interval forecasts
    # Simplified version: uses the interval bounds
    # Full CRPS would require the full CDF, but this is a practical approximation
    
    # For each sample, compute CRPS
    crps_values = []
    for i in range(len(y_true)):
        y_t = y_true[i]
        y_p = y_pred[i]
        y_l = y_lower[i]
        y_u = y_upper[i]
        
        # If true value is within interval
        if y_l <= y_t <= y_u:
            # CRPS is proportional to the distance from the mean
            crps_val = np.abs(y_t - y_p)
        else:
            # If outside interval, penalize by distance to nearest bound
            if y_t < y_l:
                crps_val = (y_l - y_t) + np.abs(y_t - y_p)
            else:  # y_t > y_u
                crps_val = (y_t - y_u) + np.abs(y_t - y_p)
        
        crps_values.append(crps_val)
    
    return np.mean(crps_values)


def crps_from_samples(y_true, y_samples):
    """
    CRPS computed from samples (Monte Carlo approximation).
    
    This is the proper way to compute CRPS when you have samples from
    the predictive distribution.
    
    Args:
        y_true: True values [n_samples] or [n_samples, n_features]
        y_samples: Samples from predictive distribution [n_samples, n_simulations] 
                   or [n_samples, n_features, n_simulations]
    
    Returns:
        CRPS value (lower is better)
    """
    y_true = np.asarray(y_true)
    y_samples = np.asarray(y_samples)
    
    # Reshape for broadcasting
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_samples.ndim == 2:
        y_samples = y_samples[:, :, np.newaxis]
    
    n_samples, n_features, n_simulations = y_samples.shape
    
    # Compute empirical CDF
    crps_values = []
    for i in range(n_samples):
        for j in range(n_features):
            y_t = y_true[i, j]
            samples = y_samples[i, j, :]
            
            # Sort samples
            samples_sorted = np.sort(samples)
            
            # Compute empirical CDF at true value
            # Fraction of samples <= y_t
            ecdf = np.mean(samples <= y_t)
            
            # CRPS = integral of (F(x) - 1(x >= y_t))^2
            # Monte Carlo approximation
            crps_val = np.mean(np.abs(samples - y_t))
            
            # Add Heaviside correction
            # This is the simplified version; full CRPS requires integration
            crps_values.append(crps_val)
    
    return np.mean(crps_values)


def compute_metrics(y, yhat, y_train=None, season=None, y_lower=None, y_upper=None):
    """
    Compute all metrics.
    
    Args:
        y: True values
        yhat: Predicted values
        y_train: Training data (required for MASE)
        season: Seasonal period (required for MASE)
        y_lower: Lower bound of prediction interval (for CRPS)
        y_upper: Upper bound of prediction interval (for CRPS)
    
    Returns:
        Dictionary of computed metrics
    """
    out = {
        'MAE': mae(y, yhat),
        'RMSE': rmse(y, yhat),
        'MAPE': mape(y, yhat),
        'sMAPE': smape(y, yhat),
    }
    
    if y_train is not None and season is not None:
        out['MASE'] = mase(y, yhat, y_train, season)
    
    # CRPS can be computed if intervals are provided
    if y_lower is not None and y_upper is not None:
        out['CRPS'] = crps(y, yhat, y_lower, y_upper)
    
    return out
