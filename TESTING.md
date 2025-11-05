# Testing Guide

This guide explains how to test the energy-forecasting-bench repository.

## Quick Start

### 1. Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### 2. Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Test Categories

### 1. Import Tests (`tests/test_imports.py`)

Tests that all modules can be imported correctly:

```bash
pytest tests/test_imports.py -v
```

**What it tests:**
- Registry modules (datasets.registry, models.registry)
- Model modules (PatchTST, Autoformer, Informer, XGBoost, Prophet, ARIMA)
- Dataset modules
- Utility modules

### 2. Model Tests (`tests/test_models.py`)

Tests model initialization and forward passes:

```bash
pytest tests/test_models.py -v
```

**What it tests:**
- Neural models (PatchTST, Autoformer, Informer) forward passes
- Classical models (XGBoost, Prophet, ARIMA) initialization
- Output shapes are correct

### 3. Dataset Tests (`tests/test_datasets.py`)

Tests dataset loading and processing:

```bash
pytest tests/test_datasets.py -v
```

**What it tests:**
- Dataset imports
- Dataset loading (requires data to be downloaded)
- Data scaling and inverse transform
- Data shapes and formats

**Note:** These tests will skip if data files are not found.

### 4. Metrics Tests (`tests/test_metrics.py`)

Tests evaluation metrics:

```bash
pytest tests/test_metrics.py -v
```

**What it tests:**
- MAE, RMSE, MAPE, sMAPE calculations
- MASE calculation (with seasonal naive)
- CRPS calculation (for probabilistic forecasts)
- compute_metrics function

### 5. Training Smoke Tests (`tests/test_training_smoke.py`)

Quick smoke tests for training infrastructure:

```bash
pytest tests/test_training_smoke.py -v
```

**What it tests:**
- Training script existence
- Quick model initialization
- Config file existence

### 6. Training Pipeline Tests (`tests/test_training_pipeline.py`)

Full training and evaluation pipeline tests:

```bash
pytest tests/test_training_pipeline.py -v
```

**What it tests:**
- Complete training pipeline for classical models
- Complete training pipeline for neural models
- Model saving and loading
- Results file generation

## Integration Testing

### Full Pipeline Test

1. **Download data** (if not already done):
```bash
python tools/download_data.py --dataset etth
```

2. **Train a model**:
```bash
python train.py --model patchtst --dataset etth --epochs 2 --batch_size 4 --exp_name integration_test
```

3. **Evaluate the model**:
```bash
python eval.py --exp_dir runs/integration_test --split test
```

4. **Check results**:
```bash
# Check if results file exists
ls runs/integration_test/
```

## CI Testing

The repository includes GitHub Actions CI that runs on every push/PR. The CI workflow:

1. Checks out code
2. Sets up Python 3.11
3. Installs dependencies from `requirements.txt`
4. Runs preprocessing on tiny dataset subset
5. Tests registry imports

To run CI checks locally:

```bash
# Test registry imports
python -c "
import importlib
for m in ['datasets.registry', 'models.registry']:
    importlib.import_module(m)
print('Registries import OK')
"
```

## Test Coverage

To generate a test coverage report:

```bash
pip install pytest-cov

# Run tests with coverage
pytest tests/ --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
# or
start htmlcov/index.html  # Windows
```

## Troubleshooting

### Tests Fail with "Module not found"

**Solution:** Make sure you're in the repository root and have installed all dependencies:
```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

### Dataset Tests Skip

**Solution:** Download the dataset first:
```bash
python tools/download_data.py --dataset etth
```

### Model Tests Fail with Shape Errors

**Solution:** Check that model parameters are compatible. Some models require specific input shapes.

### Training Tests Fail

**Solution:** Make sure you have:
1. Data downloaded
2. Sufficient disk space for checkpoints
3. Correct model parameters

## Adding New Tests

1. **Create test file** in `tests/` directory following naming convention `test_*.py`
2. **Run the test**:
```bash
pytest tests/test_new_feature.py -v
```

3. **Follow naming conventions**:
- Test files: `test_*.py`
- Test functions: `test_*`
- Use descriptive names

## Best Practices

1. **Run tests before committing**:
```bash
pytest tests/ -v
```

2. **Test both neural and classical models** when adding new features

3. **Test with small models/data** for quick feedback

4. **Test edge cases** (empty data, wrong shapes, etc.)

5. **Keep tests independent** - each test should work in isolation
