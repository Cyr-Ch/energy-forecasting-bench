"""Utilities for plotting time series results."""

import matplotlib.pyplot as plt
import numpy as np


def plot_forecast(y_true, y_pred, title="Forecast"):
    """Plot true values vs predictions."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="True", alpha=0.7)
    plt.plot(y_pred, label="Predicted", alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_multivariate_forecast(y_true, y_pred, feature_idx=0, title="Forecast"):
    """Plot forecasts for a specific feature in multivariate time series."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:, feature_idx], label="True", alpha=0.7)
    plt.plot(y_pred[:, feature_idx], label="Predicted", alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

