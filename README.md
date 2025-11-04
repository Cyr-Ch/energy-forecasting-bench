# energy-forecasting-bench

A clean, reproducible benchmark repo for **time series forecasting** on the **ETT (Electricity Transformer Temperature)** dataset.
---

## 🔧 Quick Start

1. **Create environment**: Create a conda environment named `energybench` with Python 3.11 and activate it.

2. **Install dependencies**: Install all required packages from `requirements.txt`.

3. **Download dataset**: Download the ETT dataset using the download tool.

4. **Train model**: Train any model (PatchTST, Autoformer, Informer, XGBoost, Prophet, or ARIMA) on the ETT dataset with default hyperparameters.

5. **Evaluate and report**: Evaluate the trained model and generate a leaderboard CSV file.

---

## ✅ Scope & Principles

* **ETT dataset focus** — ETT (Electricity Transformer Temperature) datasets with curated loaders + preprocessing.
* **Reproducible splits** (time-based: train/val/test with rolling-origin).
* **Leakage-safe** scaling (fit on train, apply on val/test).
* **Standard metrics**: MAE, RMSE, MAPE, sMAPE, **MASE** (with seasonal naive), CRPS (probabilistic models).
* **Strong baselines**: Fully implemented classical models (ARIMA, Prophet, XGBoost) vs. SOTA deep learning models (PatchTST, Autoformer, Informer).
* **Config-first** (Hydra/OmegaConf) & experiment tracking.
* **CI sanity checks** on tiny subsets of ETT datasets.

---

## 📊 Datasets (built-in loaders)

* **ETT (Electricity Transformer Temperature)** — transformer station load/temperature data:
  * **ETTh1, ETTh2** — hourly data
  * **ETTm1, ETTm2** — 15-minute data

### Using ETT Dataset Loaders

The repository provides PyTorch dataset loaders for ETT datasets. These loaders handle data scaling, time feature extraction, and proper train/val/test splits.

#### Basic Usage

**ETTh (Hourly) Dataset:**
```python
from datasets.ettd import Dataset_ETT_hour

# Load ETTh1 training data
train_data = Dataset_ETT_hour(
    root_path='data/raw/etth',
    flag='train',
    size=[96, 48, 96],  # [seq_len, label_len, pred_len]
    features='S',  # 'S'=univariate, 'M'=multivariate, 'MS'=multivariate single target
    data_path='ETTh1.csv',
    target='OT',  # Target column name
    scale=True,  # Scale data using StandardScaler
    timeenc=0,  # 0=manual time features, 1=time_features function
    freq='h'  # 'h' for hourly
)

# Load validation and test data
val_data = Dataset_ETT_hour(
    root_path='data/raw/etth',
    flag='val',
    size=[96, 48, 96],
    features='S',
    data_path='ETTh1.csv',
    target='OT',
    scale=True,
    timeenc=0,
    freq='h'
)

test_data = Dataset_ETT_hour(
    root_path='data/raw/etth',
    flag='test',
    size=[96, 48, 96],
    features='S',
    data_path='ETTh1.csv',
    target='OT',
    scale=True,
    timeenc=0,
    freq='h'
)
```

**ETTm (15-minute) Dataset:**
```python
from datasets.ettd import Dataset_ETT_minute

# Load ETTm1 training data
train_data = Dataset_ETT_minute(
    root_path='data/raw/ettm',
    flag='train',
    size=[96, 48, 96],
    features='S',
    data_path='ETTm1.csv',
    target='OT',
    scale=True,
    timeenc=0,
    freq='t'  # 't' for 15-minute intervals
)
```

#### Dataset Parameters

- **`root_path`**: Root directory containing the CSV data files
- **`flag`**: Dataset split - `'train'`, `'val'`, or `'test'`
- **`size`**: `[seq_len, label_len, pred_len]` - Sequence lengths for encoder input, decoder label, and prediction horizon
  - Default: `[384, 96, 96]` if `None`
- **`features`**: Feature mode
  - `'S'`: Univariate (single target column)
  - `'M'`: Multivariate (all features)
  - `'MS'`: Multivariate with single target
- **`data_path`**: CSV filename (e.g., `'ETTh1.csv'`, `'ETTh2.csv'`, `'ETTm1.csv'`, `'ETTm2.csv'`)
- **`target`**: Target column name (default: `'OT'` - Oil Temperature)
- **`scale`**: Whether to scale data using StandardScaler (fitted on training data)
- **`timeenc`**: Time encoding mode
  - `0`: Manual time features (month, day, weekday, hour, minute)
  - `1`: Uses `time_features` function for time embedding
- **`freq`**: Frequency string
  - `'h'`: Hourly (for ETTh)
  - `'t'`: 15-minute intervals (for ETTm)

#### Data Splits

The loaders use the splits:
- **ETTh**: 12 months train, 4 months val, 4 months test
- **ETTm**: 12 months train, 4 months val, 4 months test (with 4 samples per hour)

#### Dataset Output Format

Each dataset returns tuples of `(seq_x, seq_y, seq_x_mark, seq_y_mark)`:

```python
seq_x, seq_y, seq_x_mark, seq_y_mark = train_data[0]

# seq_x: [seq_len, features] - Encoder input sequence
# seq_y: [label_len + pred_len, features] - Decoder target sequence
# seq_x_mark: [seq_len, time_features] - Encoder time features
# seq_y_mark: [label_len + pred_len, time_features] - Decoder time features
```

#### Using with DataLoaders

```python
from torch.utils.data import DataLoader

# Create DataLoaders
train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_data,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# Iterate over batches
for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
    # batch_x: [B, seq_len, features]
    # batch_y: [B, label_len + pred_len, features]
    # batch_x_mark: [B, seq_len, time_features]
    # batch_y_mark: [B, label_len + pred_len, time_features]
    pass
```

#### Inverse Transform

To get predictions back to original scale:

```python
# Get scaled prediction from model
scaled_pred = model(batch_x, batch_x_mark, batch_dec, batch_y_mark)

# Inverse transform to original scale
pred_original = train_data.inverse_transform(scaled_pred)
```

#### Using with Registry

You can also use the dataset registry:

```python
from datasets.registry import get_dataset

# Get dataset via registry (if configured)
dataset = get_dataset('etth', root_path='data/raw/etth', flag='train', ...)
```

---

## 🧮 Evaluation Protocols

* **Fixed split** (default):

  * Train: first 70% of timeline
  * Val: next 10%
  * Test: last 20%
* **Rolling-origin**: multiple forecast origins across test span.
* **Multi-horizon**: report H ∈ {24, 48, 96, 168} with context L ∈ {96, 168, 336, 720}.
* **Seasonality-aware MASE**: seasonal period = 24 (hourly), = 96 (15-min).
---

## 🧠 Models

The repository supports multiple forecasting models, from classical baselines to state-of-the-art deep learning architectures. All models are implemented to be compatible with the ETT dataset format.

### Deep Learning Models

#### PatchTST

**PatchTST** — Patch-based Transformer for long-term time series forecasting. This implementation matches the official PatchTST architecture (ICLR 2023) with:

- **Patching**: Segments time series into patches (sub-series) as tokens
- **Channel Independence**: Each channel processed as separate univariate series sharing weights
- **RevIN**: Reversible Instance Normalization for handling distribution shift
- **Efficient Architecture**: Reduces sequence length and computational complexity

**Usage:**
```python
from models.registry import get_model

model = get_model(
    'patchtst',
    d_in=1,           # Number of input channels
    out_len=96,       # Prediction length
    patch_len=16,     # Patch length
    stride=8,         # Stride for patching
    d_model=512,      # Model dimension
    n_heads=8,        # Number of attention heads
    n_layers=3,       # Number of transformer layers
    revin=True        # Use RevIN
)
```

**Documentation**: See [`models/patchtst/README.md`](models/patchtst/README.md) for detailed architecture explanation, paper reference, and usage guide.

**Paper**: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730) (ICLR 2023)  

#### Autoformer

**Autoformer** — Decomposition Transformer with Auto-Correlation for long-term forecasting. Uses series decomposition and auto-correlation mechanism for efficient long-sequence modeling.

**Usage:**
```python
model = get_model(
    'autoformer',
    d_in=1,
    out_len=96,
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1
)
```

**Documentation**: See [`models/autoformer/README.md`](models/autoformer/README.md) for detailed architecture explanation.

#### Informer

**Informer** — Efficient Transformer for long sequence forecasting using ProbSparse attention mechanism.

**Usage:**
```python
model = get_model(
    'informer',
    d_in=1,
    out_len=96,
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1
)
```

**Documentation**: See [`models/informer/README.md`](models/informer/README.md) for detailed architecture explanation.

### Classical Baselines

#### XGBoost

**XGBoost** — Gradient boosting framework for time series with automatic feature engineering (lag features, rolling statistics, time features).

**Features:**
- Automatic feature engineering (lags, rolling stats, time features)
- Multi-step forecasting
- Compatible with ETT dataset format

**Usage:**
```python
from models.classical.xgboost import XGBoostModel

model = XGBoostModel(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_lag_features=True,
    lag_window=10,
    use_rolling_features=True
)
model.fit(X_train, y_train)
predictions = model.predict(X_val, horizon=96)
```

**Documentation**: See [`models/classical/README.md`](models/classical/README.md#xgboost-model) for detailed usage guide.

#### Prophet

**Prophet** — Facebook's additive regression model for time series forecasting with automatic seasonality detection and holiday effects.

**Features:**
- Automatic seasonality detection (yearly, weekly, daily)
- Holiday effects and custom events
- Uncertainty intervals
- Robust to missing data and outliers
- Component breakdown (trend, seasonality, holidays)

**Usage:**
```python
from models.classical.prophet import ProphetModel
import pandas as pd

model = ProphetModel(
    config={
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': 'auto',
        'weekly_seasonality': 'auto',
        'daily_seasonality': 'auto',
        'target_column': 'OT',
        'pred_len': 96
    }
)
model.fit(df)  # DataFrame with datetime index
forecast = model.predict(steps=96)
```

**Documentation**: See [`models/classical/README.md`](models/classical/README.md#prophet-model) for comprehensive parameter tuning guide and examples.

#### ARIMA

**ARIMA** — AutoRegressive Integrated Moving Average model for statistical time series forecasting with automatic order selection.

**Features:**
- Automatic order selection using AIC
- SARIMA support for seasonal patterns
- Confidence intervals
- Stationarity handling

**Usage:**
```python
from models.classical.arima import ARIMAModel

model = ARIMAModel(
    auto_select=True,      # Automatically select best order
    max_p=5,
    max_d=2,
    max_q=5,
    seasonal_period=24,    # For hourly data
    trend='c'
)
model.fit(data)
predictions = model.predict(steps=96)
```

**Documentation**: See [`models/classical/README.md`](models/classical/README.md#arima-model) for detailed usage guide and parameter tuning tips.

### Model Configuration

All models can be configured via YAML files in `configs/models/`. Each model has a corresponding configuration file:

- `configs/models/patchtst.yaml` - PatchTST configuration
- `configs/models/autoformer.yaml` - Autoformer configuration
- `configs/models/informer.yaml` - Informer configuration
- `configs/models/xgboost.yaml` - XGBoost configuration
- `configs/models/prophet.yaml` - Prophet configuration

Use the `--config` flag to load a custom configuration file, or override specific parameters directly via command-line arguments.

### Model Registry

All models are registered in the model registry for easy access:

```python
from models.registry import get_model

# Get any model by name
model = get_model('patchtst', d_in=1, out_len=96)
model = get_model('autoformer', d_in=1, out_len=96)
model = get_model('informer', d_in=1, out_len=96)
```

For classical models, import directly:

```python
from models.classical.xgboost import XGBoostModel
from models.classical.prophet import ProphetModel
from models.classical.arima import ARIMAModel
```

### Adding a New Model

1. Create a model file in `models/your_model/` directory. Import `register_model` from `models.registry` and `torch.nn`. Decorate your model class with `@register_model('your_model')`. Implement `__init__` to set up the architecture and `forward` for the forward pass.

2. Add a configuration file in `configs/models/your_model.yaml` with your model's hyperparameters.

3. Test the model by running the training script with `--model your_model --dataset etth`.

---

## 🏃 Training & Evaluation

### Training a Model

**Basic training**: Run `train.py` with required arguments including `--model`, `--dataset`, `--target`, `--context_len`, `--horizon`, and `--seed`. Additional options include `--epochs`, `--batch_size`, and `--lr` for learning rate.

**With custom config**: Use the `--config` flag to load settings from a YAML configuration file, while still specifying the model and dataset via command-line arguments.

**Training options**:
- `--model`: Model name (patchtst, autoformer, informer, xgboost, etc.)
- `--dataset`: Dataset name (etth, ettm, etth1, etth2, ettm1, ettm2)
- `--target`: Target column name (default: 'load')
- `--context_len`: Input sequence length (default: 336)
- `--horizon`: Forecast horizon (default: 96)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--seed`: Random seed for reproducibility
- `--device`: Device (cuda/cpu)
- `--output_dir`: Output directory for checkpoints

### Evaluation

**Evaluate a trained model**: Run `eval.py` with `--exp_dir` pointing to the experiment directory containing model checkpoints and configurations. Optionally specify `--checkpoint` to use a specific checkpoint file (defaults to `best_model.pt`).

**Evaluation options**:
- `--exp_dir`: Experiment directory containing model and config
- `--checkpoint`: Specific checkpoint file (default: 'best_model.pt')
- `--split`: Split to evaluate (train/val/test)
- `--metrics`: Comma-separated metrics (MAE,RMSE,sMAPE,MASE)

**Metrics computed**:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **sMAPE**: Symmetric Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error (requires seasonal period)

### Batch Training

Use the provided shell scripts in the `scripts/` directory for batch training. Run `scripts/run_all_etth.sh` to train all models on the ETTh dataset, or `scripts/run_all_ettm.sh` for the ETTm dataset.

### Leaderboard Generation

Generate a leaderboard comparing all models by running `tools/make_leaderboard.py` with `--results_dir` pointing to the directory containing experiment results, `--save` specifying the output filename, and `--format` for the output format.

The leaderboard includes:
- Model name
- Dataset
- Context length and horizon
- All computed metrics
- Training time and parameters

---

## 📥 Data Download Helpers

### Downloading Datasets

**Basic download**: Run `tools/download_data.py` with `--dataset` specifying the dataset name to download.

**Download with subset**: Add the `--subset` flag to download a specific subset size (e.g., `small` or `full`). The `small` subset is useful for quick testing, while `full` downloads the complete dataset.

**Available datasets**:
- `etth`: ETTh1/ETTh2 (hourly transformer data)
- `ettm`: ETTm1/ETTm2 (15-min transformer data)
- `etth1`: ETTh1 dataset (hourly)
- `etth2`: ETTh2 dataset (hourly)
- `ettm1`: ETTm1 dataset (15-min)
- `ettm2`: ETTm2 dataset (15-min)

**Download options**:
- `--dataset`: Dataset name (required)
- `--subset`: Subset size (small/full, default: 'full')
- `--output`: Output directory (default: 'data/raw')
- `--force`: Force re-download even if exists

### Custom Dataset

To add support for a new dataset:

1. Create a dataset loader file in `datasets/your_dataset.py`. Import `EnergyDataset` from `datasets.base`, `register_dataset` from `datasets.registry`, and `pandas`. Decorate your loader function with `@register_dataset('your_dataset')`. The function should load the data, convert date columns to datetime, set them as index, and return an `EnergyDataset` instance with the appropriate frequency.

2. Add a configuration file in `configs/datasets/your_dataset.yaml` with dataset-specific settings.

3. Test the dataset loader by running the download tool with your dataset name.

---

## 🧪 CI (smoke tests)

### Running Tests Locally

**Test imports**: Test that registry modules can be imported correctly by importing `datasets.registry` and `models.registry` using `importlib`.

**Run preprocessing test**: Test the preprocessing pipeline by running `tools/preprocess.py` with a tiny dataset subset.

**Test model initialization**: Verify that models can be initialized correctly by importing from the model registry and creating an instance. Similarly test dataset loading (which may fail if data hasn't been downloaded, which is expected).

### GitHub Actions CI

The repository includes GitHub Actions CI that runs on every push/PR:

**CI workflow** (`.github/workflows/ci.yaml`):
1. Checks out code
2. Sets up Python 3.11
3. Installs dependencies from `requirements.txt`
4. Runs preprocessing on tiny dataset subset
5. Tests registry imports

**Viewing CI results**:
- Go to the "Actions" tab on GitHub
- Click on the latest workflow run
- Check individual job status and logs

### Adding Tests

To add new tests:

1. **Unit tests**: Create a `tests/` directory and add test files like `test_models.py`. Import necessary modules (`torch`, `get_model` from registry), create test functions that initialize models, pass sample inputs, and assert expected output shapes.

2. **Integration tests**: Create integration test files like `test_pipeline.py` that test the full pipeline including dataset loading, data splitting, model initialization, and forward passes.

3. **Run tests**: Execute all tests using `pytest tests/` from the project root.

---

## 🧭 Contribution Guide (short)

* One PR per model/dataset.
* Include `configs/` + `scripts/` for batch runs.
* Add tiny fixture (≤200 KB) for CI.

---

