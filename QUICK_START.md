# Quick Start Guide - Training on Downloaded Data

## Prerequisites
1. ✅ Data downloaded (ETTh1.csv is in `data/raw/etth/`)
2. ✅ Dependencies installed (`pip install -r requirements.txt`)

## Step 1: Basic Training Commands

### Train XGBoost (Fastest)
```bash
python train.py --model xgboost --dataset etth1
```

### Train Prophet
```bash
python train.py --model prophet --dataset etth1 --horizon 96
```

### Train ARIMA
```bash
python train.py --model arima --dataset etth1 --horizon 96
```

## Step 2: Customize Training Parameters

### With Custom Horizon and Context Length
```bash
python train.py \
    --model xgboost \
    --dataset etth1 \
    --horizon 168 \
    --context_len 336 \
    --exp_name my_experiment
```

### Using Config File
```bash
python train.py --config configs/models/xgboost.yaml --dataset etth1
```

## Step 3: Check Results

After training completes, check:
- `runs/{model}_{dataset}_{timestamp}/trained_model.pkl` - Your trained model
- `runs/{model}_{dataset}_{timestamp}/results.json` - Validation metrics
- `runs/{model}_{dataset}_{timestamp}/config.yaml` - Training configuration

## Step 4: Evaluate the Model

```bash
python eval.py --exp_dir runs/{your_experiment_name} --split test
```

## Common Parameters

- `--model`: `xgboost`, `prophet`, or `arima`
- `--dataset`: `etth1`, `etth2`, `ettm1`, `ettm2`
- `--horizon`: Prediction length (default: 96)
- `--context_len`: Input sequence length (default: 336)
- `--exp_name`: Custom experiment name
- `--seed`: Random seed (default: 42)

## Troubleshooting

### Data not found?
- Check that CSV file is in `data/raw/etth/ETTh1.csv`
- Verify file name matches exactly (case-sensitive)

### Import errors?
- Run: `pip install -r requirements.txt`
- Activate your virtual environment if using one

### Model not training?
- Check console output for error messages
- Verify dataset path is correct

