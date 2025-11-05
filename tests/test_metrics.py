"""Test evaluation metrics."""

import pytest
import numpy as np
from metrics import mae, rmse, mape, smape, mase, crps


def test_mae():
    """Test MAE calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 2.5, 2.5, 4.5, 5.5])
    
    result = mae(y_true, y_pred)
    expected = np.mean(np.abs(y_true - y_pred))
    
    assert np.isclose(result, expected), f"MAE: expected {expected}, got {result}"
    print("✓ MAE calculation")


def test_rmse():
    """Test RMSE calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 2.5, 2.5, 4.5, 5.5])
    
    result = rmse(y_true, y_pred)
    expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    assert np.isclose(result, expected), f"RMSE: expected {expected}, got {result}"
    print("✓ RMSE calculation")


def test_mape():
    """Test MAPE calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])
    
    result = mape(y_true, y_pred)
    assert result >= 0, "MAPE should be non-negative"
    print("✓ MAPE calculation")


def test_smape():
    """Test sMAPE calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])
    
    result = smape(y_true, y_pred)
    assert result >= 0, "sMAPE should be non-negative"
    assert result <= 200, "sMAPE should be <= 200"
    print("✓ sMAPE calculation")


def test_mase():
    """Test MASE calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])
    y_train = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    seasonal_period = 1
    
    result = mase(y_true, y_pred, y_train, seasonal_period)
    assert result >= 0, "MASE should be non-negative"
    print("✓ MASE calculation")


def test_crps():
    """Test CRPS calculation."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    y_lower = np.array([0.9, 1.9, 2.7])
    y_upper = np.array([1.3, 2.3, 3.1])
    
    result = crps(y_true, y_pred, y_lower, y_upper)
    assert result >= 0, "CRPS should be non-negative"
    print("✓ CRPS calculation")


def test_compute_metrics():
    """Test compute_metrics function."""
    from metrics import compute_metrics
    
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])
    
    metrics = compute_metrics(y_true, y_pred)
    
    assert 'MAE' in metrics
    assert 'RMSE' in metrics
    assert 'MAPE' in metrics
    assert 'sMAPE' in metrics
    
    print("✓ compute_metrics function")
    print(f"  Metrics: {list(metrics.keys())}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


