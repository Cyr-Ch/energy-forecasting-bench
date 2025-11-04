"""Evaluation script for time series forecasting models."""

import argparse
import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.registry import get_dataset
from models.registry import get_model
from utils.serialization import load_model, load_config
from utils.seed import set_seed
from metrics import compute_metrics


def evaluate_deep_model(model, data_loader, device, scaler=None):
    """Evaluate a deep learning model (PyTorch)."""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(data_loader, desc="Evaluating"):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            # Forward pass
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.float().to(device)
            if batch_y_mark is not None:
                batch_y_mark = batch_y_mark.float().to(device)
            
            # Model prediction
            pred = model(batch_x, batch_x_mark, None, batch_y_mark)
            
            # Extract target (last pred_len steps)
            target = batch_y[:, -pred.shape[-1]:, 0] if batch_y.ndim == 3 else batch_y[:, -pred.shape[-1]:]
            
            # Handle output shape
            if pred.ndim == 2:
                pred = pred.cpu().numpy()
            elif pred.ndim == 3:
                pred = pred[:, :, 0].cpu().numpy()  # Take first channel for univariate
            
            target = target.cpu().numpy()
            
            predictions.append(pred)
            targets.append(target)
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Inverse transform if scaler provided
    if scaler is not None:
        # Reshape for inverse transform (needs 2D)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        targets = scaler.inverse_transform(targets)
        # Flatten back
        predictions = predictions.flatten()
        targets = targets.flatten()
    
    return predictions, targets


def evaluate_classical_model(model, data_loader, scaler=None):
    """Evaluate a classical model (XGBoost, Prophet, ARIMA)."""
    predictions = []
    targets = []
    
    for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(data_loader, desc="Evaluating"):
        # Extract target
        target = batch_y[:, -model.config.get('pred_len', 96):, 0].numpy() if batch_y.ndim == 3 else batch_y[:, -model.config.get('pred_len', 96):].numpy()
        
        # For classical models, we need to prepare data differently
        # This is a simplified version - may need model-specific handling
        if hasattr(model, 'predict'):
            # Convert batch to appropriate format
            batch_x_np = batch_x.numpy()
            batch_y_np = batch_y.numpy()
            
            # Model-specific prediction (simplified)
            # In practice, you'd need to handle each model type differently
            batch_pred = []
            for i in range(batch_x.shape[0]):
                # Extract sequence
                seq = batch_x_np[i, :, 0] if batch_x.ndim == 3 else batch_x_np[i, :]
                
                # Make prediction (this is simplified - actual implementation depends on model)
                if hasattr(model, 'predict'):
                    try:
                        pred = model.predict(steps=target.shape[1])
                        if isinstance(pred, pd.DataFrame):
                            pred = pred['yhat'].values if 'yhat' in pred.columns else pred.values.flatten()
                        pred = np.array(pred).flatten()
                    except:
                        # Fallback: use last value
                        pred = np.repeat(seq[-1], target.shape[1])
                else:
                    pred = np.repeat(seq[-1], target.shape[1])
                
                batch_pred.append(pred[:target.shape[1]])
            
            batch_pred = np.array(batch_pred)
            predictions.append(batch_pred)
            targets.append(target)
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Inverse transform if scaler provided
    if scaler is not None:
        # Reshape for inverse transform (needs 2D)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        targets = scaler.inverse_transform(targets)
        # Flatten back
        predictions = predictions.flatten()
        targets = targets.flatten()
    
    return predictions, targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate time series forecasting model")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory containing model and config")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt",
                        help="Model checkpoint filename (default: best_model.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Split to evaluate (default: test)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation (default: 32)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu). Auto-detects if not specified")
    parser.add_argument("--metrics", type=str, default="MAE,RMSE,MAPE,sMAPE,MASE",
                        help="Comma-separated metrics to compute (default: MAE,RMSE,MAPE,sMAPE,MASE)")
    parser.add_argument("--seasonal_period", type=int, default=None,
                        help="Seasonal period for MASE (default: auto-detect from dataset)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results (default: results.json in exp_dir)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load experiment directory
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory not found: {exp_dir}")
    
    # Load config
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        config_path = exp_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Try to load from pickle
            config_path = exp_dir / "config.pkl"
            if config_path.exists():
                config = load_config(str(config_path))
            else:
                raise ValueError(f"Config file not found in {exp_dir}")
    else:
        with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract model parameters
    model_name = config.get('model', 'patchtst')
    dataset_name = config.get('dataset', 'etth')
    target = config.get('target', 'OT')
    context_len = config.get('context_len', config.get('seq_len', 336))
    horizon = config.get('horizon', config.get('pred_len', 96))
    features = config.get('features', 'S')
    data_path = config.get('data_path', 'ETTh1.csv')
    root_path = config.get('root_path', 'data/raw/etth')
    scale = config.get('scale', True)
    timeenc = config.get('timeenc', 0)
    freq = config.get('freq', 'h')
    seed = config.get('seed', 42)
    
    set_seed(seed)
    
    # Determine dataset type
    if 'ettm' in dataset_name.lower() or 'minute' in dataset_name.lower():
        from datasets.ettd import Dataset_ETT_minute
        DatasetClass = Dataset_ETT_minute
        if freq == 'h':
            freq = 't'  # 15-minute intervals
    else:
        from datasets.ettd import Dataset_ETT_hour
        DatasetClass = Dataset_ETT_hour
        if freq == 't':
            freq = 'h'  # Hourly
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    eval_dataset = DatasetClass(
        root_path=root_path,
        flag=args.split,
        size=[context_len, context_len // 2, horizon],
        features=features,
        data_path=data_path,
        target=target,
        scale=scale,
        timeenc=timeenc,
        freq=freq
    )
    
    # Load training dataset for MASE
    train_dataset = DatasetClass(
        root_path=root_path,
        flag='train',
        size=[context_len, context_len // 2, horizon],
        features=features,
        data_path=data_path,
        target=target,
        scale=scale,
        timeenc=timeenc,
        freq=freq
    )
    
    # Create data loader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Determine seasonal period
    if args.seasonal_period is None:
        if freq == 'h':
            seasonal_period = 24  # Hourly: daily seasonality
        elif freq == 't':
            seasonal_period = 96  # 15-min: daily seasonality
        else:
            seasonal_period = 24
    else:
        seasonal_period = args.seasonal_period
    
    # Initialize model
    print(f"Initializing model: {model_name}")
    if model_name in ['patchtst', 'autoformer', 'informer']:
        # Deep learning model
        d_in = eval_dataset.data_x.shape[1] if eval_dataset.data_x.ndim == 2 else 1
        
        # Get model parameters from config
        model_kwargs = {
            'd_in': d_in,
            'out_len': horizon,
        }
        
        # Add model-specific parameters
        if model_name == 'patchtst':
            model_kwargs.update({
                'patch_len': config.get('patch_len', 16),
                'stride': config.get('stride', 8),
                'd_model': config.get('d_model', 512),
                'n_heads': config.get('n_heads', 8),
                'n_layers': config.get('n_layers', 3),
                'd_ff': config.get('d_ff', 2048),
                'dropout': config.get('dropout', 0.1),
                'revin': config.get('revin', True),
            })
        elif model_name in ['autoformer', 'informer']:
            model_kwargs.update({
                'd_model': config.get('d_model', 512),
                'n_heads': config.get('n_heads', 8),
                'e_layers': config.get('e_layers', 2),
                'd_layers': config.get('d_layers', 1),
                'd_ff': config.get('d_ff', 2048),
                'dropout': config.get('dropout', 0.1),
            })
        
        model = get_model(model_name, **model_kwargs)
        model = model.to(device)
        
        # Load checkpoint
        checkpoint_path = exp_dir / args.checkpoint
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}, using untrained model")
        
        # Evaluate
        predictions, targets = evaluate_deep_model(
            model, eval_loader, device, 
            scaler=eval_dataset.scaler if hasattr(eval_dataset, 'scaler') else None
        )
    else:
        # Classical model (not fully implemented in eval - would need model-specific handling)
        print(f"Warning: Classical model evaluation ({model_name}) may need model-specific handling")
        print("This is a simplified implementation. For classical models, use their native evaluation methods.")
        return
    
    # Flatten for metrics computation
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    # Get training data for MASE
    train_data_flat = train_dataset.data_x.flatten()
    
    # Parse metrics to compute
    metrics_to_compute = [m.strip() for m in args.metrics.split(',')]
    
    # Compute metrics
    print("\nComputing metrics...")
    results = {}
    
    # Always compute basic metrics
    if 'MAE' in metrics_to_compute or 'all' in metrics_to_compute:
        from metrics import mae
        results['MAE'] = float(mae(targets_flat, predictions_flat))
    
    if 'RMSE' in metrics_to_compute or 'all' in metrics_to_compute:
        from metrics import rmse
        results['RMSE'] = float(rmse(targets_flat, predictions_flat))
    
    if 'MAPE' in metrics_to_compute or 'all' in metrics_to_compute:
        from metrics import mape
        results['MAPE'] = float(mape(targets_flat, predictions_flat))
    
    if 'sMAPE' in metrics_to_compute or 'all' in metrics_to_compute:
        from metrics import smape
        results['sMAPE'] = float(smape(targets_flat, predictions_flat))
    
    # MASE requires training data
    if 'MASE' in metrics_to_compute or 'all' in metrics_to_compute:
        if len(train_data_flat) > seasonal_period:
            from metrics import mase
            results['MASE'] = float(mase(targets_flat, predictions_flat, train_data_flat, seasonal_period))
        else:
            print(f"Warning: Training data too short for MASE (need > {seasonal_period}, got {len(train_data_flat)})")
    
    # CRPS (if intervals available - would need model to provide them)
    if 'CRPS' in metrics_to_compute or 'all' in metrics_to_compute:
        print("Warning: CRPS requires prediction intervals. Not computing CRPS.")
    
    # Print results
    print("\n" + "="*50)
    print(f"Evaluation Results ({args.split.upper()} set)")
    print("="*50)
    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")
    print("="*50)
    
    # Save results
    if args.output_file is None:
        output_file = exp_dir / f"results_{args.split}.json"
    else:
        output_file = Path(args.output_file)
    
    results_dict = {
        'model': model_name,
        'dataset': dataset_name,
        'split': args.split,
        'metrics': results,
        'config': {
            'context_len': context_len,
            'horizon': horizon,
            'features': features,
            'seasonal_period': seasonal_period,
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

