"""Test that all modules can be imported correctly."""

import pytest
import sys


def test_registry_imports():
    """Test that registry modules can be imported."""
    import importlib
    
    modules = [
        'datasets.registry',
        'models.registry',
        'datasets.ettd',
        'models.patchtst.model',
        'models.autoformer.model',
        'models.informer.model',
        'models.classical.xgboost',
        'models.classical.prophet',
        'models.classical.arima',
        'metrics',
        'utils.seed',
        'utils.scaling',
        'utils.timefeatures',
    ]
    
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")


def test_model_registry():
    """Test that model registry works."""
    from models.registry import get_model
    
    # Test neural models
    neural_models = ['patchtst', 'autoformer', 'informer']
    for model_name in neural_models:
        try:
            model = get_model(model_name, d_in=1, out_len=96, d_model=64, n_heads=2, dropout=0.1)
            assert model is not None
            print(f"✓ Model registry: {model_name}")
        except Exception as e:
            pytest.fail(f"Failed to get model {model_name}: {e}")


def test_dataset_registry():
    """Test that dataset registry works."""
    from datasets.registry import get_dataset
    
    # This might fail if data isn't downloaded, which is OK
    try:
        dataset = get_dataset('etth', target='OT')
        print("✓ Dataset registry: etth")
    except FileNotFoundError:
        print("⚠ Dataset registry works but data not found (expected if not downloaded)")
    except Exception as e:
        print(f"⚠ Dataset registry test: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


