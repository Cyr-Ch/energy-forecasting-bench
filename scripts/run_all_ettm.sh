#!/bin/bash
# Run all models on ETTm dataset

# Define models to run
models=("xgboost" "prophet" "arima" "patchtst" "autoformer" "informer")

# Define datasets (ETTm1 and ETTm2)
datasets=("ettm1" "ettm2")

# Run each model on each dataset
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "=========================================="
        echo "Training $model on $dataset"
        echo "=========================================="
        python train.py --model $model --dataset $dataset
        echo ""
    done
done

echo "All models trained on ETTm datasets!"
