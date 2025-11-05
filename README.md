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

## 📊 Datasets

* **ETT (Electricity Transformer Temperature)** — transformer station load/temperature data:
  * **ETTh1, ETTh2** — hourly data
  * **ETTm1, ETTm2** — 15-minute data

The repository provides PyTorch dataset loaders for ETT datasets. These loaders handle data scaling, time feature extraction, and proper train/val/test splits.

**Data Splits**:
- **ETTh**: 12 months train, 4 months val, 4 months test
- **ETTm**: 12 months train, 4 months val, 4 months test (with 4 samples per hour)

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

The repository supports multiple forecasting models, from classical baselines to state-of-the-art deep learning architectures.

### Deep Learning Models

#### PatchTST
**PatchTST** — Patch-based Transformer for long-term time series forecasting. This implementation matches the official PatchTST architecture (ICLR 2023) with patching, channel independence, RevIN, and efficient architecture.

**Documentation**: See [`models/patchtst/README.md`](models/patchtst/README.md) for detailed architecture explanation, paper reference, and usage guide.

**Paper**: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730) (ICLR 2023)

#### Autoformer
**Autoformer** — Decomposition Transformer with Auto-Correlation for long-term forecasting. Uses series decomposition and auto-correlation mechanism for efficient long-sequence modeling.

**Documentation**: See [`models/autoformer/README.md`](models/autoformer/README.md) for detailed architecture explanation.

#### Informer
**Informer** — Efficient Transformer for long sequence forecasting using ProbSparse attention mechanism.

**Documentation**: See [`models/informer/README.md`](models/informer/README.md) for detailed architecture explanation.

### Classical Baselines

#### XGBoost
**XGBoost** — Gradient boosting framework for time series with automatic feature engineering (lag features, rolling statistics, time features).

**Documentation**: See [`models/classical/README.md`](models/classical/README.md#xgboost-model) for detailed usage guide.

#### Prophet
**Prophet** — Facebook's additive regression model for time series forecasting with automatic seasonality detection and holiday effects.

**Documentation**: See [`models/classical/README.md`](models/classical/README.md#prophet-model) for comprehensive parameter tuning guide and examples.

#### ARIMA
**ARIMA** — AutoRegressive Integrated Moving Average model for statistical time series forecasting with automatic order selection.

**Documentation**: See [`models/classical/README.md`](models/classical/README.md#arima-model) for detailed usage guide and parameter tuning tips.

### Model Configuration

All models can be configured via YAML files in `configs/models/`. Each model has a corresponding configuration file:
- `configs/models/patchtst.yaml` - PatchTST configuration
- `configs/models/autoformer.yaml` - Autoformer configuration
- `configs/models/informer.yaml` - Informer configuration
- `configs/models/xgboost.yaml` - XGBoost configuration
- `configs/models/prophet.yaml` - Prophet configuration

Use the `--config` flag to load a custom configuration file, or override specific parameters directly via command-line arguments.

---

## 🏃 Training

### Neural Models (PatchTST, Autoformer, Informer)

**Basic training**:
```bash
python train.py --model patchtst --dataset etth1 --epochs 10 --batch_size 32
```

**With config file**:
```bash
python train.py --config configs/models/patchtst.yaml --dataset etth1
```

**Resume training**:
```bash
python train.py --resume runs/experiment_name/last_model.pt
```

**Neural model training options**:
- `--model`: Model name (`patchtst`, `autoformer`, `informer`)
- `--dataset`: Dataset name (`etth1`, `etth2`, `ettm1`, `ettm2`)
- `--target`: Target column name (default: `'OT'`)
- `--context_len`: Input sequence length (default: `336`)
- `--horizon`: Forecast horizon (default: `96`)
- `--epochs`: Number of training epochs (default: `10`)
- `--batch_size`: Batch size (default: `32`)
- `--lr`: Learning rate (default: `1e-3`)
- `--weight_decay`: Weight decay (default: `1e-4`)
- `--dropout`: Dropout rate (default: `0.1`)
- `--patience`: Early stopping patience (default: `10`)
- `--save_best_after`: Only save best model after N epochs (default: `0` = immediately)
- `--scheduler`: Learning rate scheduler (`cosine`, `step`, `plateau`, `none`)
- `--seed`: Random seed for reproducibility (default: `42`)
- `--device`: Device (`cuda`/`cpu`, auto-detects if not specified)
- `--output_dir`: Output directory for experiments (default: `'runs'`)
- `--exp_name`: Experiment name (auto-generated if not specified)
- `--config`: Path to YAML config file
- `--resume`: Resume from checkpoint path

### Classical Models (XGBoost, Prophet, ARIMA)

**Basic training**:
```bash
python train.py --model xgboost --dataset etth1
python train.py --model prophet --dataset etth1 --horizon 96
python train.py --model arima --dataset etth1 --horizon 96
```

**With config file**:
```bash
python train.py --config configs/models/xgboost.yaml --dataset etth1
```

**Classical model training options**:
- `--model`: Model name (`xgboost`, `prophet`, `arima`)
- `--dataset`: Dataset name (`etth1`, `etth2`, `ettm1`, `ettm2`)
- `--target`: Target column name (default: `'OT'`)
- `--context_len`: Input sequence length for feature extraction (default: `336`)
- `--horizon`: Forecast horizon (default: `96`)
- `--seed`: Random seed for reproducibility (default: `42`)
- `--output_dir`: Output directory for experiments (default: `'runs'`)
- `--exp_name`: Experiment name (auto-generated if not specified)
- `--config`: Path to YAML config file with model-specific parameters

### Batch Training

Use the provided shell scripts in the `scripts/` directory for batch training:

**Train all models on ETTh datasets**:
```bash
bash scripts/run_all_etth.sh
```

**Train all models on ETTm datasets**:
```bash
bash scripts/run_all_ettm.sh
```

These scripts train all 6 models (xgboost, prophet, arima, patchtst, autoformer, informer) on all dataset variants (etth1/etth2 or ettm1/ettm2).

---

## 📈 Evaluation

The evaluation script (`eval.py`) supports both neural and classical models with comprehensive metrics.

**Evaluate a trained model**:
```bash
python eval.py --exp_dir runs/experiment_name --split test
```

**Specify checkpoint**:
```bash
python eval.py --exp_dir runs/experiment_name --checkpoint best_model.pt --split test
```

**Custom metrics**:
```bash
python eval.py --exp_dir runs/experiment_name --metrics MAE,RMSE,MAPE,sMAPE,MASE --split test
```

**Evaluation options**:
- `--exp_dir`: Experiment directory containing model and config (required)
- `--checkpoint`: Specific checkpoint file (default: `'best_model.pt'` for neural, `'trained_model.pkl'` for classical)
- `--split`: Split to evaluate (`train`, `val`, `test`, default: `'test'`)
- `--batch_size`: Batch size for neural models (default: `32`)
- `--device`: Device (`cuda`/`cpu`, auto-detects if not specified)
- `--metrics`: Comma-separated metrics (default: `'MAE,RMSE,MAPE,sMAPE,MASE'`)
- `--seasonal_period`: Seasonal period for MASE (default: `24` for hourly, `96` for 15-min)
- `--output_file`: Output file for results (default: saves to experiment directory as `results_{split}.json`)

**Metrics computed**:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **sMAPE**: Symmetric Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error (requires training data and seasonal period)
- **CRPS**: Continuous Ranked Probability Score (for probabilistic models with prediction intervals)

**Results location**: Results are saved to `{exp_dir}/results_{split}.json` and also printed to the console.

---

## 🏆 Leaderboard Generation

Generate a leaderboard comparing multiple experiments by running `tools/make_leaderboard.py`.

**Compare multiple experiments**:
```bash
python tools/make_leaderboard.py runs/exp1 runs/exp2 runs/exp3
```

**Compare all experiments in runs directory** (bash):
```bash
python tools/make_leaderboard.py runs/*
```

**Save to CSV**:
```bash
python tools/make_leaderboard.py runs/exp1 runs/exp2 --output leaderboard.csv
```

**Sort by different metric**:
```bash
python tools/make_leaderboard.py runs/exp1 runs/exp2 --sort-by RMSE
```

**Leaderboard options**:
- `exp_dirs`: One or more experiment directory paths (required)
- `--output`, `-o`: Output CSV file path (default: don't save)
- `--sort-by`: Metric to sort by (default: `MAE`)

**Leaderboard includes**:
- Rank (sorted by specified metric)
- Model name
- Dataset name
- Split (train/val/test)
- All metrics found in results files (MAE, RMSE, MSE, MAPE, sMAPE, MASE, etc.)
- Experiment directory and name (for reference)

The script automatically finds `results.json` (from training) or `results_{split}.json` (from eval.py) files in each experiment directory and extracts all metrics.

---

## 📥 Data Download

**Basic download**:
```bash
python tools/download_data.py --dataset etth1
```

**Download all ETTh datasets**:
```bash
python tools/download_data.py --dataset etth
```

**Download all ETTm datasets**:
```bash
python tools/download_data.py --dataset ettm
```

**Available datasets**:
- `etth`: ETTh1/ETTh2 (hourly transformer data)
- `ettm`: ETTm1/ETTm2 (15-min transformer data)
- `etth1`: ETTh1 dataset (hourly)
- `etth2`: ETTh2 dataset (hourly)
- `ettm1`: ETTm1 dataset (15-min)
- `ettm2`: ETTm2 dataset (15-min)

**Download options**:
- `--dataset`: Dataset name (required)
- `--subset`: Subset size (`small`/`full`, default: `'full'`)
- `--output`: Output directory (default: `'data/raw'`)
- `--force`: Force re-download even if exists

---

## 🧪 Testing

**Install test dependencies**:
```bash
pip install pytest pytest-cov
```

**Run all tests**:
```bash
pytest tests/ -v
```

**Run specific test categories**:
```bash
# Test imports
pytest tests/test_imports.py -v

# Test models
pytest tests/test_models.py -v

# Test datasets (requires data download)
pytest tests/test_datasets.py -v

# Test metrics
pytest tests/test_metrics.py -v

# Test training pipeline
pytest tests/test_training_pipeline.py -v
```

**Quick training test** (1 epoch, small model):
```bash
python train.py --model patchtst --dataset etth1 --epochs 1 --batch_size 2 \
    --context_len 96 --horizon 24 --d_model 32 --n_heads 2 --n_layers 1 \
    --exp_name test_run
```

See [`TESTING.md`](TESTING.md) for detailed testing instructions, including:
- Running tests with coverage
- Integration testing
- Troubleshooting
- Adding new tests
- Best practices

---

## 🧭 Contribution Guide

* One PR per model/dataset.
* Include `configs/` + `scripts/` for batch runs.
* Add tiny fixture (≤200 KB) for CI.

---
