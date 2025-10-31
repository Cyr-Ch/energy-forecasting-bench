"""Evaluation metrics for time series forecasting."""

import numpy as np


def mae(y, yhat):
    """Mean Absolute Error."""
    return np.mean(np.abs(y - yhat))


def rmse(y, yhat):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y - yhat)**2))


def smape(y, yhat, eps=1e-8):
    """Symmetric Mean Absolute Percentage Error."""
    return 100 * np.mean(2*np.abs(yhat - y) / (np.abs(y)+np.abs(yhat)+eps))


def mase(y, yhat, y_train, season):
    """Mean Absolute Scaled Error."""
    naive = np.mean(np.abs(y_train[season:] - y_train[:-season])) + 1e-8
    return np.mean(np.abs(y - yhat)) / naive


def compute_metrics(y, yhat, y_train=None, season=None):
    """Compute all metrics."""
    out = {
        'MAE': mae(y, yhat),
        'RMSE': rmse(y, yhat),
        'sMAPE': smape(y, yhat),
    }
    if y_train is not None and season is not None:
        out['MASE'] = mase(y, yhat, y_train, season)
    return out
