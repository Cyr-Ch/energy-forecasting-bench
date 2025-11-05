"""Test model initialization and forward passes."""

import pytest
import torch
import numpy as np
from models.registry import get_model


def test_patchtst_forward():
    """Test PatchTST model forward pass."""
    model = get_model(
        'patchtst',
        d_in=1,
        out_len=96,
        d_model=64,
        n_heads=2,
        n_layers=2,
        dropout=0.1,
        patch_len=16,
        stride=8,
        revin=True
    )
    
    # Create dummy input
    batch_size = 2
    seq_len = 336
    batch_x = torch.randn(batch_size, seq_len, 1)
    batch_x_mark = torch.randn(batch_size, seq_len, 4)  # time features
    batch_dec = None
    batch_y_mark = torch.randn(batch_size, 96, 4)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch_x, batch_x_mark, batch_dec, batch_y_mark)
    
    assert output.shape == (batch_size, 96), f"Expected shape (2, 96), got {output.shape}"
    print("✓ PatchTST forward pass")


def test_autoformer_forward():
    """Test Autoformer model forward pass."""
    model = get_model(
        'autoformer',
        d_in=1,
        out_len=96,
        d_model=64,
        n_heads=2,
        e_layers=2,
        d_layers=1,
        d_ff=128,
        dropout=0.1
    )
    
    batch_size = 2
    seq_len = 336
    batch_x = torch.randn(batch_size, seq_len, 1)
    batch_x_mark = torch.randn(batch_size, seq_len, 4)
    batch_dec = torch.randn(batch_size, 48, 1)  # label_len
    batch_y_mark = torch.randn(batch_size, 144, 4)  # label_len + pred_len
    
    model.eval()
    with torch.no_grad():
        output = model(batch_x, batch_x_mark, batch_dec, batch_y_mark)
    
    assert output.shape == (batch_size, 96), f"Expected shape (2, 96), got {output.shape}"
    print("✓ Autoformer forward pass")


def test_informer_forward():
    """Test Informer model forward pass."""
    model = get_model(
        'informer',
        d_in=1,
        out_len=96,
        d_model=64,
        n_heads=2,
        e_layers=2,
        d_layers=1,
        d_ff=128,
        dropout=0.1
    )
    
    batch_size = 2
    seq_len = 336
    batch_x = torch.randn(batch_size, seq_len, 1)
    batch_x_mark = torch.randn(batch_size, seq_len, 4)
    batch_dec = torch.randn(batch_size, 48, 1)
    batch_y_mark = torch.randn(batch_size, 144, 4)
    
    model.eval()
    with torch.no_grad():
        output = model(batch_x, batch_x_mark, batch_dec, batch_y_mark)
    
    assert output.shape == (batch_size, 96), f"Expected shape (2, 96), got {output.shape}"
    print("✓ Informer forward pass")


def test_classical_models_init():
    """Test classical model initialization."""
    from models.classical.xgboost import XGBoostModel
    from models.classical.prophet import ProphetModel
    from models.classical.arima import ARIMAModel
    
    # XGBoost
    xgb_model = XGBoostModel(n_estimators=10, max_depth=3)
    assert xgb_model is not None
    print("✓ XGBoost initialization")
    
    # Prophet
    prophet_model = ProphetModel(config={'target_column': 'OT', 'pred_len': 96})
    assert prophet_model is not None
    print("✓ Prophet initialization")
    
    # ARIMA
    arima_model = ARIMAModel(auto_select=False, order=(1, 1, 1))
    assert arima_model is not None
    print("✓ ARIMA initialization")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

