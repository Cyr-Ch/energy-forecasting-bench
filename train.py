"""Training script for time series forecasting models."""

import argparse
import os
import json
import torch
import numpy as np
from datasets.registry import get_dataset
from models.registry import get_model
from utils.seed import set_seed
from utils.serialization import save_config
from metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='patchtst')
    parser.add_argument('--dataset', type=str, default='ecl')
    parser.add_argument('--target', type=str, default='load')
    parser.add_argument('--context_len', type=int, default=336)
    parser.add_argument('--horizon', type=int, default=96)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 1) load data
    ds = get_dataset(args.dataset, target=args.target)
    split = ds.split()
    
    # 2) build dataloaders (omitted for brevity)
    # TODO: Implement data loading logic
    
    # 3) init model
    model = get_model(args.model, d_in=1, out_len=args.horizon)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 4) train loop (brevity)
    for epoch in range(10):
        pass
    
    # 5) save
    os.makedirs('runs', exist_ok=True)
    save_config(vars(args), 'runs/last_config.json')


if __name__ == "__main__":
    main()
