"""Training script for time series forecasting models."""

import argparse
import os
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.registry import get_dataset
from models.registry import get_model
from utils.seed import set_seed
from utils.serialization import save_config, save_model, load_model
from metrics import compute_metrics, mae, rmse
import pickle


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(train_loader, desc="Training"):
        # Move to device
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        
        if batch_x_mark is not None:
            batch_x_mark = batch_x_mark.float().to(device)
        if batch_y_mark is not None:
            batch_y_mark = batch_y_mark.float().to(device)
        
        # Forward pass
        pred = model(batch_x, batch_x_mark, None, batch_y_mark)
        
        # Extract target (last pred_len steps)
        if pred.ndim == 2:
            target = batch_y[:, -pred.shape[-1]:, 0] if batch_y.ndim == 3 else batch_y[:, -pred.shape[-1]:]
        elif pred.ndim == 3:
            target = batch_y[:, -pred.shape[1]:, :]
            pred = pred.squeeze(-1) if pred.shape[-1] == 1 else pred
            target = target.squeeze(-1) if target.shape[-1] == 1 else target
        
        # Compute loss
        loss = criterion(pred, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate_neural(model, val_loader, criterion, device, scaler=None):
    """Validate neural model."""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    num_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(val_loader, desc="Validating"):
            # Move to device
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.float().to(device)
            if batch_y_mark is not None:
                batch_y_mark = batch_y_mark.float().to(device)
            
            # Forward pass
            pred = model(batch_x, batch_x_mark, None, batch_y_mark)
            
            # Extract target
            if pred.ndim == 2:
                target = batch_y[:, -pred.shape[-1]:, 0] if batch_y.ndim == 3 else batch_y[:, -pred.shape[-1]:]
            elif pred.ndim == 3:
                target = batch_y[:, -pred.shape[1]:, :]
                pred = pred.squeeze(-1) if pred.shape[-1] == 1 else pred
                target = target.squeeze(-1) if target.shape[-1] == 1 else target
            
            # Compute loss
            loss = criterion(pred, target)
            total_loss += loss.item()
            
            # Store predictions and targets for metrics
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            
            # Inverse transform if scaler provided
            if scaler is not None:
                if pred_np.ndim == 1:
                    pred_np = pred_np.reshape(-1, 1)
                if target_np.ndim == 1:
                    target_np = target_np.reshape(-1, 1)
                pred_np = scaler.inverse_transform(pred_np).flatten()
                target_np = scaler.inverse_transform(target_np).flatten()
            
            predictions.append(pred_np)
            targets.append(target_np)
            num_batches += 1
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Compute metrics
    val_loss = total_loss / num_batches
    val_mae = mae(targets, predictions)
    val_rmse = rmse(targets, predictions)
    
    return val_loss, val_mae, val_rmse


def train_classical_model(model_name, train_dataset, val_dataset, args, config, exp_dir):
    """Train a classical model (XGBoost, Prophet, ARIMA)."""
    import pandas as pd
    
    print(f"Training classical model: {model_name}")
    
    # Prepare training data
    print("Preparing training data...")
    if model_name == 'xgboost':
        from models.classical.xgboost import XGBoostModel
        
        # Extract sequences for XGBoost
        X_train = []
        y_train = []
        timestamps_train = []
        
        for i in tqdm(range(len(train_dataset)), desc="Preparing train data"):
            seq_x, seq_y, seq_x_mark, seq_y_mark = train_dataset[i]
            X_train.append(seq_x)
            y_train.append(seq_y[-args.horizon:, 0] if seq_y.ndim == 2 else seq_y[-args.horizon:])
            timestamps_train.append(seq_x_mark)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        timestamps_train = np.array(timestamps_train)
        
        # Initialize model
        model_kwargs = {
            'n_estimators': config.get('n_estimators', 100),
            'max_depth': config.get('max_depth', 6),
            'learning_rate': config.get('learning_rate', 0.1),
            'use_lag_features': config.get('use_lag_features', True),
            'lag_window': config.get('lag_window', 10),
            'use_rolling_features': config.get('use_rolling_features', True),
            'rolling_windows': config.get('rolling_windows', [3, 7, 14]),
            'use_time_features': config.get('use_time_features', False),
            'random_state': args.seed,
        }
        model = XGBoostModel(**model_kwargs)
        
        # Fit model
        print("Fitting XGBoost model...")
        model.fit(X_train, y_train, timestamps=timestamps_train if model_kwargs['use_time_features'] else None)
        
        # Prepare validation data
        X_val = []
        y_val = []
        timestamps_val = []
        
        for i in tqdm(range(len(val_dataset)), desc="Preparing val data"):
            seq_x, seq_y, seq_x_mark, seq_y_mark = val_dataset[i]
            X_val.append(seq_x)
            y_val.append(seq_y[-args.horizon:, 0] if seq_y.ndim == 2 else seq_y[-args.horizon:])
            timestamps_val.append(seq_x_mark)
        
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        timestamps_val = np.array(timestamps_val)
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(X_val, timestamps=timestamps_val if model_kwargs['use_time_features'] else None, horizon=args.horizon)
        
        # Inverse transform
        if hasattr(train_dataset, 'scaler') and train_dataset.scaler is not None:
            predictions = train_dataset.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            y_val = train_dataset.scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        
        # Compute metrics
        predictions_flat = predictions.flatten()
        targets_flat = y_val.flatten()
        
    elif model_name == 'prophet':
        from models.classical.prophet import ProphetModel
        
        # Convert to DataFrame format
        train_df = pd.DataFrame(train_dataset.data_x, index=pd.date_range(start='2020-01-01', periods=len(train_dataset.data_x), freq=args.freq))
        train_df.columns = [args.target] if train_df.shape[1] == 1 else [f'col_{i}' for i in range(train_df.shape[1])]
        
        val_df = pd.DataFrame(val_dataset.data_x, index=pd.date_range(start=train_df.index[-1] + pd.Timedelta(hours=1 if args.freq == 'h' else 15), 
                                                                      periods=len(val_dataset.data_x), 
                                                                      freq=args.freq))
        val_df.columns = train_df.columns
        
        # Initialize model
        prophet_config = {
            'changepoint_prior_scale': config.get('changepoint_prior_scale', 0.05),
            'seasonality_prior_scale': config.get('seasonality_prior_scale', 10.0),
            'holidays_prior_scale': config.get('holidays_prior_scale', 10.0),
            'seasonality_mode': config.get('seasonality_mode', 'additive'),
            'yearly_seasonality': config.get('yearly_seasonality', 'auto'),
            'weekly_seasonality': config.get('weekly_seasonality', 'auto'),
            'daily_seasonality': config.get('daily_seasonality', 'auto'),
            'target_column': args.target if args.target in train_df.columns else train_df.columns[0],
            'pred_len': args.horizon,
        }
        model = ProphetModel(config=prophet_config)
        
        # Fit model
        print("Fitting Prophet model...")
        model.fit(train_df)
        
        # Make predictions
        print("Making predictions...")
        forecast = model.predict(steps=args.horizon)
        predictions = forecast['yhat'].values
        
        # Get validation targets
        targets = val_df.iloc[:args.horizon, 0].values if args.target in val_df.columns else val_df.iloc[:args.horizon, 0].values
        
        # Inverse transform
        if hasattr(train_dataset, 'scaler') and train_dataset.scaler is not None:
            predictions = train_dataset.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            targets = train_dataset.scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
        
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()
        
    elif model_name == 'arima':
        from models.classical.arima import ARIMAModel
        
        # Extract training series (univariate)
        train_series = train_dataset.data_x[:, 0] if train_dataset.data_x.ndim == 2 else train_dataset.data_x.flatten()
        val_series = val_dataset.data_x[:, 0] if val_dataset.data_x.ndim == 2 else val_dataset.data_x.flatten()
        
        # Initialize model
        arima_config = {
            'auto_select': config.get('auto_select', True),
            'max_p': config.get('max_p', 5),
            'max_d': config.get('max_d', 2),
            'max_q': config.get('max_q', 5),
            'seasonal_period': config.get('seasonal_period', 24 if args.freq == 'h' else 96),
            'trend': config.get('trend', 'c'),
        }
        model = ARIMAModel(**arima_config)
        
        # Fit model
        print("Fitting ARIMA model...")
        model.fit(train_series)
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(steps=args.horizon)
        
        # Get validation targets
        targets = val_series[:args.horizon]
        
        # Inverse transform
        if hasattr(train_dataset, 'scaler') and train_dataset.scaler is not None:
            predictions = train_dataset.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            targets = train_dataset.scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
        
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()
    
    else:
        raise ValueError(f"Unknown classical model: {model_name}")
    
    # Compute metrics
    val_mae = mae(targets_flat, predictions_flat)
    val_rmse = rmse(targets_flat, predictions_flat)
    val_mse = np.mean((targets_flat - predictions_flat) ** 2)
    
    print(f"\nValidation Metrics:")
    print(f"  MSE: {val_mse:.6f}")
    print(f"  MAE: {val_mae:.6f}")
    print(f"  RMSE: {val_rmse:.6f}")
    
    # Save model
    model_path = exp_dir / 'trained_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    # Save results
    results = {
        'model': model_name,
        'dataset': args.dataset,
        'metrics': {
            'MSE': float(val_mse),
            'MAE': float(val_mae),
            'RMSE': float(val_rmse),
        },
        'config': vars(args)
    }
    
    results_path = exp_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    return model, val_mse, val_mae, val_rmse


def main():
    parser = argparse.ArgumentParser(description="Train time series forecasting model")
    
    # Config argument (needs to be parsed first)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (YAML)')
    
    # Parse known args first to get config path
    args, remaining_argv = parser.parse_known_args()
    
    # Load config if provided (before parsing other args)
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Model and dataset arguments
    # Check for both 'model' and 'model_name' in config (some configs use model_name)
    model_default = config.get('model') or config.get('model_name') or 'patchtst'
    parser.add_argument('--model', type=str, default=model_default, 
                       help='Model name (patchtst, autoformer, informer for neural models; xgboost, prophet, arima for classical models)')
    parser.add_argument('--dataset', type=str, default=config.get('dataset', 'etth'),
                       help='Dataset name (etth, ettm, etth1, etth2, ettm1, ettm2)')
    parser.add_argument('--target', type=str, default=config.get('target', 'OT'),
                       help='Target column name')
    parser.add_argument('--features', type=str, default=config.get('features', 'S'),
                       choices=['S', 'M', 'MS'],
                       help='Feature mode: S=univariate, M=multivariate, MS=multivariate single target')
    parser.add_argument('--data_path', type=str, default=config.get('data_path', 'ETTh1.csv'),
                       help='Data file path')
    parser.add_argument('--root_path', type=str, default=config.get('root_path', 'data/raw/etth'),
                       help='Root path for data')
    
    # Sequence arguments
    parser.add_argument('--context_len', type=int, default=config.get('context_len', 336),
                       help='Input sequence length (context window)')
    parser.add_argument('--horizon', type=int, default=config.get('horizon', 96),
                       help='Prediction horizon (forecast length)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=config.get('epochs', 10),
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 32),
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=config.get('lr', 1e-3),
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=config.get('weight_decay', 1e-4),
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=config.get('dropout', 0.1),
                       help='Dropout rate')
    
    # Model-specific arguments
    parser.add_argument('--d_model', type=int, default=config.get('d_model', 512),
                       help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=config.get('n_heads', 8),
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=config.get('n_layers', 3),
                       help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=config.get('d_ff', 2048),
                       help='Feed-forward dimension')
    parser.add_argument('--e_layers', type=int, default=config.get('e_layers', 2),
                       help='Number of encoder layers (Autoformer/Informer)')
    parser.add_argument('--d_layers', type=int, default=config.get('d_layers', 1),
                       help='Number of decoder layers (Autoformer/Informer)')
    parser.add_argument('--patch_len', type=int, default=config.get('patch_len', 16),
                       help='Patch length (PatchTST)')
    parser.add_argument('--stride', type=int, default=config.get('stride', 8),
                       help='Stride for patching (PatchTST)')
    parser.add_argument('--revin', action='store_true', default=config.get('revin', True),
                       help='Use RevIN (PatchTST)')
    
    # Training settings
    parser.add_argument('--seed', type=int, default=config.get('seed', 42),
                       help='Random seed')
    parser.add_argument('--device', type=str, default=config.get('device', None),
                       help='Device (cuda/cpu). Auto-detects if not specified')
    parser.add_argument('--num_workers', type=int, default=config.get('num_workers', 0),
                       help='Number of data loader workers')
    parser.add_argument('--patience', type=int, default=config.get('patience', 10),
                       help='Early stopping patience')
    parser.add_argument('--save_best_after', type=int, default=config.get('save_best_after', 0),
                       help='Only save best model after N epochs (0 = save immediately)')
    parser.add_argument('--scheduler', type=str, default=config.get('scheduler', 'cosine'),
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--warmup_steps', type=int, default=config.get('warmup_steps', 400),
                       help='Warmup steps for cosine scheduler')
    
    # Data settings
    parser.add_argument('--scale', action='store_true', default=config.get('scale', True),
                       help='Scale data')
    parser.add_argument('--timeenc', type=int, default=config.get('timeenc', 0),
                       choices=[0, 1],
                       help='Time encoding: 0=manual, 1=time_features')
    parser.add_argument('--freq', type=str, default=config.get('freq', 'h'),
                       help='Data frequency: h=hourly, t=15-min')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=config.get('output_dir', 'runs'),
                       help='Output directory for experiments')
    parser.add_argument('--exp_name', type=str, default=config.get('exp_name', None),
                       help='Experiment name (auto-generated if not specified)')
    parser.add_argument('--resume', type=str, default=config.get('resume', None),
                       help='Resume from checkpoint')
    
    # Parse all arguments (command line overrides config)
    args = parser.parse_args(remaining_argv, namespace=args)
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Determine dataset class
    if 'ettm' in args.dataset.lower() or 'minute' in args.dataset.lower():
        from datasets.ettd import Dataset_ETT_minute
        DatasetClass = Dataset_ETT_minute
        if args.freq == 'h':
            args.freq = 't'  # 15-minute intervals
    else:
        from datasets.ettd import Dataset_ETT_hour
        DatasetClass = Dataset_ETT_hour
        if args.freq == 't':
            args.freq = 'h'  # Hourly
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = DatasetClass(
        root_path=args.root_path,
        flag='train',
        size=[args.context_len, args.context_len // 2, args.horizon],
        features=args.features,
        data_path=args.data_path,
        target=args.target,
        scale=args.scale,
        timeenc=args.timeenc,
        freq=args.freq
    )
    
    val_dataset = DatasetClass(
        root_path=args.root_path,
        flag='val',
        size=[args.context_len, args.context_len // 2, args.horizon],
        features=args.features,
        data_path=args.data_path,
        target=args.target,
        scale=args.scale,
        timeenc=args.timeenc,
        freq=args.freq
    )
    
    # Setup experiment directory
    if args.exp_name is None:
        from datetime import datetime
        args.exp_name = f"{args.model}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    exp_dir = Path(args.output_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment directory: {exp_dir}")
    
    # Determine model type
    neural_models = ['patchtst', 'autoformer', 'informer']
    classical_models = ['xgboost', 'prophet', 'arima']
    
    is_neural = args.model in neural_models
    is_classical = args.model in classical_models
    
    if not is_neural and not is_classical:
        raise ValueError(
            f"Unknown model: {args.model}\n"
            f"Neural models: {', '.join(neural_models)}\n"
            f"Classical models: {', '.join(classical_models)}"
        )
    
    # Route to appropriate training method
    if is_classical:
        # Train classical model
        model, val_mse, val_mae, val_rmse = train_classical_model(
            args.model, train_dataset, val_dataset, args, config, exp_dir
        )
        
        # Save config
        config_dict = vars(args)
        config_dict['num_params'] = 'N/A (classical model)'
        with open(exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        print(f"\nTraining completed!")
        print(f"Results saved to: {exp_dir}")
        
    else:
        # Train neural model
        # Determine input dimension
        d_in = train_dataset.data_x.shape[1] if train_dataset.data_x.ndim == 2 else 1
        
        # Initialize model
        print(f"Initializing neural model: {args.model}")
        
        model_kwargs = {
            'd_in': d_in,
            'out_len': args.horizon,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'dropout': args.dropout,
        }
        
        # Model-specific parameters
        if args.model == 'patchtst':
            model_kwargs.update({
                'n_layers': args.n_layers,
                'd_ff': args.d_ff,
                'patch_len': args.patch_len,
                'stride': args.stride,
                'revin': args.revin,
            })
        elif args.model in ['autoformer', 'informer']:
            model_kwargs.update({
                'e_layers': args.e_layers,
                'd_layers': args.d_layers,
                'd_ff': args.d_ff,
            })
            if args.model == 'autoformer':
                model_kwargs.update({
                    'factor': config.get('factor', 3),
                    'moving_avg': config.get('moving_avg', 25),
                })
            elif args.model == 'informer':
                model_kwargs.update({
                    'factor': config.get('factor', 5),
                    'distil': config.get('distil', True),
                })
        
        model = get_model(args.model, **model_kwargs)
        model = model.to(device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        if args.scheduler == 'cosine':
            total_steps = len(train_loader) * args.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps
            )
        elif args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.epochs // 3, gamma=0.1
            )
        elif args.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        else:
            scheduler = None
        
        # Save config
        config_dict = vars(args)
        config_dict['num_params'] = num_params
        with open(exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch}")
        
        # Training loop
        print("\nStarting training...")
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss, val_mae, val_rmse = validate_neural(
                model, val_loader, criterion, device,
                scaler=train_dataset.scaler if hasattr(train_dataset, 'scaler') else None
            )
            
            # Update learning rate
            if scheduler is not None:
                if args.scheduler == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print metrics
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"Val MAE: {val_mae:.6f}, Val RMSE: {val_rmse:.6f}, LR: {current_lr:.6f}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'best_val_loss': best_val_loss,
            }
            
            # Save last checkpoint
            torch.save(checkpoint, exp_dir / 'last_model.pt')
            
            # Save best model (only after save_best_after epochs if specified)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model if we've passed the minimum epoch threshold
                if epoch + 1 >= args.save_best_after:
                    torch.save(checkpoint, exp_dir / 'best_model.pt')
                    print(f"✓ New best model saved (Val Loss: {val_loss:.6f}, Epoch {epoch+1})")
                else:
                    print(f"  Best validation loss improved to {val_loss:.6f}, but waiting until epoch {args.save_best_after} to save")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping after {epoch+1} epochs (patience: {args.patience})")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
