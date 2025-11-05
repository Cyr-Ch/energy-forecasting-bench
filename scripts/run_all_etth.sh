#!/bin/bash
# Run all models on ETTh dataset

# Define models to run
models=("xgboost" "prophet" "arima" "patchtst" "autoformer" "informer")

# Define datasets (ETTh1 and ETTh2)
datasets=("etth1" "etth2")

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

echo "All models trained on ETTh datasets!"
