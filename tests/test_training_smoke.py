"""Smoke tests for training script (quick tests without full training)."""

import pytest
import torch
import sys
from pathlib import Path


def test_training_imports():
    """Test that training script can be imported."""
    # Check that train.py exists
    train_path = Path('train.py')
    assert train_path.exists(), "train.py should exist"
    
    # Check that eval.py exists
    eval_path = Path('eval.py')
    assert eval_path.exists(), "eval.py should exist"
    
    print("✓ Training scripts exist")


def test_neural_model_quick_init():
    """Quick test of neural model initialization for training."""
    from models.registry import get_model
    
    # Test PatchTST
    model = get_model(
        'patchtst',
        d_in=1,
        out_len=24,  # Short horizon for quick test
        d_model=32,  # Small model
        n_heads=2,
        n_layers=1,
        dropout=0.1,
        patch_len=8,
        stride=4,
        revin=True
    )
    
    # Test forward pass with small batch
    batch_x = torch.randn(1, 48, 1)  # Small batch
    batch_x_mark = torch.randn(1, 48, 4)
    batch_y_mark = torch.randn(1, 24, 4)
    
    model.eval()
    with torch.no_grad():
        output = model(batch_x, batch_x_mark, None, batch_y_mark)
    
    assert output.shape == (1, 24), f"Expected (1, 24), got {output.shape}"
    print("✓ Neural model quick initialization test")


def test_classical_model_init():
    """Test classical model initialization for training."""
    from models.classical.xgboost import XGBoostModel
    
    model = XGBoostModel(
        n_estimators=5,  # Very small for quick test
        max_depth=2,
        learning_rate=0.1,
        use_lag_features=True,
        lag_window=3,
        use_rolling_features=False,  # Disable for speed
        random_state=42
    )
    
    assert model is not None
    print("✓ Classical model initialization test")


def test_config_files_exist():
    """Test that config files exist."""
    config_dir = Path('configs/models')
    
    expected_configs = [
        'patchtst.yaml',
        'autoformer.yaml',
        'informer.yaml',
        'xgboost.yaml',
        'prophet.yaml',
    ]
    
    for config_file in expected_configs:
        config_path = config_dir / config_file
        assert config_path.exists(), f"Config file {config_file} should exist"
    
    print("✓ Config files exist")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


