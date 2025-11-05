"""Test dataset loading and data processing."""

import pytest
import numpy as np
from pathlib import Path


def test_dataset_imports():
    """Test that dataset modules can be imported."""
    from datasets.ettd import Dataset_ETT_hour, Dataset_ETT_minute
    from datasets.registry import get_dataset
    
    print("✓ Dataset imports successful")


def test_dataset_loading():
    """Test dataset loading (may fail if data not downloaded)."""
    from datasets.ettd import Dataset_ETT_hour
    
    data_path = Path('data/raw/etth/ETTh1.csv')
    
    if not data_path.exists():
        pytest.skip("ETTh1.csv not found. Run download tool first.")
    
    try:
        dataset = Dataset_ETT_hour(
            root_path='data/raw/etth',
            flag='train',
            size=[96, 48, 96],
            features='S',
            data_path='ETTh1.csv',
            target='OT',
            scale=True,
            timeenc=0,
            freq='h'
        )
        
        # Check dataset length
        assert len(dataset) > 0, "Dataset should have samples"
        
        # Check data shape
        seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
        
        assert seq_x.shape[0] == 96, "seq_x should have seq_len=96"
        assert seq_y.shape[0] == 144, "seq_y should have label_len+pred_len=144"
        assert seq_x_mark.shape[0] == 96, "seq_x_mark should match seq_x length"
        
        print("✓ Dataset loading successful")
        print(f"  Dataset size: {len(dataset)} samples")
        print(f"  seq_x shape: {seq_x.shape}")
        print(f"  seq_y shape: {seq_y.shape}")
        
    except Exception as e:
        pytest.fail(f"Dataset loading failed: {e}")


def test_dataset_scaling():
    """Test that dataset scaling works correctly."""
    from datasets.ettd import Dataset_ETT_hour
    
    data_path = Path('data/raw/etth/ETTh1.csv')
    
    if not data_path.exists():
        pytest.skip("ETTh1.csv not found. Run download tool first.")
    
    dataset = Dataset_ETT_hour(
        root_path='data/raw/etth',
        flag='train',
        size=[96, 48, 96],
        features='S',
        data_path='ETTh1.csv',
        target='OT',
        scale=True,
        timeenc=0,
        freq='h'
    )
    
    # Check that scaler exists
    assert hasattr(dataset, 'scaler'), "Dataset should have scaler when scale=True"
    assert dataset.scaler is not None, "Scaler should be initialized"
    
    # Test inverse transform
    seq_x, seq_y, _, _ = dataset[0]
    original = dataset.inverse_transform(seq_x)
    
    assert original.shape == seq_x.shape, "Inverse transform should preserve shape"
    
    print("✓ Dataset scaling works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


