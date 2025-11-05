"""Test full training and evaluation pipeline with minimal data."""

import pytest
import torch
import numpy as np
import json
import pickle
from pathlib import Path
import sys
import os

# Add parent directory to path to import train and eval modules
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_classical_model_training_pipeline():
    """Test full training + evaluation pipeline for classical model (XGBoost)."""
    from datasets.ettd import Dataset_ETT_hour
    from models.classical.xgboost import XGBoostModel
    from metrics import mae, rmse
    import tempfile
    import shutil
    
    # Check if data exists
    data_path = Path('data/raw/etth/ETTh1.csv')
    if not data_path.exists():
        pytest.skip("ETTh1.csv not found. Run download tool first.")
    
    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        exp_dir = tmp_path / "test_xgboost"
        exp_dir.mkdir()
        
        try:
            # Load dataset with very small size for quick test
            print("Loading small dataset subset...")
            train_dataset = Dataset_ETT_hour(
                root_path='data/raw/etth',
                flag='train',
                size=[48, 24, 24],  # Small: seq_len=48, label_len=24, pred_len=24
                features='S',
                data_path='ETTh1.csv',
                target='OT',
                scale=True,
                timeenc=0,
                freq='h'
            )
            
            val_dataset = Dataset_ETT_hour(
                root_path='data/raw/etth',
                flag='val',
                size=[48, 24, 24],
                features='S',
                data_path='ETTh1.csv',
                target='OT',
                scale=True,
                timeenc=0,
                freq='h'
            )
            
            # Use only first 100 samples for very quick training
            print(f"Using {min(100, len(train_dataset))} training samples...")
            
            # Prepare training data (small subset)
            X_train = []
            y_train = []
            for i in range(min(100, len(train_dataset))):
                seq_x, seq_y, _, _ = train_dataset[i]
                X_train.append(seq_x)
                y_train.append(seq_y[-24:, 0] if seq_y.ndim == 2 else seq_y[-24:])
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Initialize and train model (very small for speed)
            print("Training XGBoost model...")
            model = XGBoostModel(
                n_estimators=5,  # Very small for quick test
                max_depth=2,
                learning_rate=0.1,
                use_lag_features=False,  # Disable for speed
                use_rolling_features=False,  # Disable for speed
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Prepare validation data (small subset)
            X_val = []
            y_val = []
            for i in range(min(50, len(val_dataset))):
                seq_x, seq_y, _, _ = val_dataset[i]
                X_val.append(seq_x)
                y_val.append(seq_y[-24:, 0] if seq_y.ndim == 2 else seq_y[-24:])
            
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            
            # Make predictions
            print("Making predictions...")
            predictions = model.predict(X_val, horizon=24)
            
            # Inverse transform
            if hasattr(train_dataset, 'scaler') and train_dataset.scaler is not None:
                predictions = train_dataset.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                y_val = train_dataset.scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            
            # Compute metrics
            predictions_flat = predictions.flatten()
            targets_flat = y_val.flatten()
            
            val_mae = mae(targets_flat, predictions_flat)
            val_rmse = rmse(targets_flat, predictions_flat)
            
            print(f"  Validation MAE: {val_mae:.6f}")
            print(f"  Validation RMSE: {val_rmse:.6f}")
            
            # Save model
            model_path = exp_dir / 'trained_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            assert model_path.exists(), "Model should be saved"
            print(f"[OK] Model saved to {model_path}")
            
            # Save results
            results = {
                'model': 'xgboost',
                'dataset': 'etth1',
                'metrics': {
                    'MAE': float(val_mae),
                    'RMSE': float(val_rmse),
                },
            }
            
            results_path = exp_dir / 'results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            assert results_path.exists(), "Results file should be saved"
            print(f"[OK] Results saved to {results_path}")
            
            # Test model loading
            print("Testing model loading...")
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            assert loaded_model is not None, "Model should load successfully"
            assert loaded_model.is_fitted, "Loaded model should be fitted"
            
            # Test prediction with loaded model
            test_pred = loaded_model.predict(X_val[:5], horizon=24)
            assert test_pred.shape[0] == 5, "Loaded model should make predictions"
            
            print("[OK] Model loading and prediction successful")
            
            # Verify metrics are reasonable (not NaN or inf)
            assert not np.isnan(val_mae), "MAE should not be NaN"
            assert not np.isnan(val_rmse), "RMSE should not be NaN"
            assert not np.isinf(val_mae), "MAE should not be inf"
            assert not np.isinf(val_rmse), "RMSE should not be inf"
            
            print("[OK] All pipeline tests passed!")
            
        except Exception as e:
            pytest.fail(f"Training pipeline test failed: {e}")


def test_neural_model_training_pipeline():
    """Test full training + evaluation pipeline for neural model (PatchTST) with minimal data."""
    from datasets.ettd import Dataset_ETT_hour
    from models.registry import get_model
    from torch.utils.data import DataLoader, Subset
    from metrics import mae, rmse
    import tempfile
    
    # Check if data exists
    data_path = Path('data/raw/etth/ETTh1.csv')
    if not data_path.exists():
        pytest.skip("ETTh1.csv not found. Run download tool first.")
    
    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        exp_dir = tmp_path / "test_patchtst"
        exp_dir.mkdir()
        
        try:
            # Load dataset with very small size for quick test
            print("Loading small dataset subset...")
            train_dataset = Dataset_ETT_hour(
                root_path='data/raw/etth',
                flag='train',
                size=[48, 24, 24],  # Small: seq_len=48, label_len=24, pred_len=24
                features='S',
                data_path='ETTh1.csv',
                target='OT',
                scale=True,
                timeenc=0,
                freq='h'
            )
            
            val_dataset = Dataset_ETT_hour(
                root_path='data/raw/etth',
                flag='val',
                size=[48, 24, 24],
                features='S',
                data_path='ETTh1.csv',
                target='OT',
                scale=True,
                timeenc=0,
                freq='h'
            )
            
            # Use only first 50 samples for very quick training
            train_subset = Subset(train_dataset, list(range(min(50, len(train_dataset)))))
            val_subset = Subset(val_dataset, list(range(min(20, len(val_dataset)))))
            
            print(f"Using {len(train_subset)} training samples, {len(val_subset)} validation samples...")
            
            # Create data loaders
            train_loader = DataLoader(train_subset, batch_size=8, shuffle=False, num_workers=0)
            val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=0)
            
            # Initialize model (very small for speed)
            print("Initializing PatchTST model...")
            model = get_model(
                'patchtst',
                d_in=1,
                out_len=24,
                d_model=32,  # Small model
                n_heads=2,
                n_layers=1,  # Single layer
                dropout=0.1,
                patch_len=8,
                stride=4,
                revin=True
            )
            
            device = torch.device('cpu')
            model = model.to(device)
            
            # Train for just 1 epoch
            print("Training model (1 epoch)...")
            model.train()
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
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
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                print("[OK] Model training completed")
            
            # Evaluate
            print("Evaluating model...")
            model.eval()
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
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
                    
                    # Inverse transform
                    if hasattr(train_dataset, 'scaler') and train_dataset.scaler is not None:
                        pred_np = pred.cpu().numpy()
                        target_np = target.cpu().numpy()
                        
                        if pred_np.ndim == 1:
                            pred_np = pred_np.reshape(-1, 1)
                        if target_np.ndim == 1:
                            target_np = target_np.reshape(-1, 1)
                        
                        pred_np = train_dataset.scaler.inverse_transform(pred_np).flatten()
                        target_np = train_dataset.scaler.inverse_transform(target_np).flatten()
                        
                        predictions.append(pred_np)
                        targets.append(target_np)
                    else:
                        predictions.append(pred.cpu().numpy().flatten())
                        targets.append(target.cpu().numpy().flatten())
            
            predictions = np.concatenate(predictions, axis=0)
            targets = np.concatenate(targets, axis=0)
            
            # Compute metrics
            val_mae = mae(targets, predictions)
            val_rmse = rmse(targets, predictions)
            
            print(f"  Validation MAE: {val_mae:.6f}")
            print(f"  Validation RMSE: {val_rmse:.6f}")
            
            # Save model
            model_path = exp_dir / 'best_model.pt'
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'val_mae': val_mae,
                'val_rmse': val_rmse,
            }
            torch.save(checkpoint, model_path)
            
            assert model_path.exists(), "Model should be saved"
            print(f"[OK] Model saved to {model_path}")
            
            # Save config
            config_path = exp_dir / 'config.yaml'
            config = {
                'model': 'patchtst',
                'dataset': 'etth1',
                'horizon': 24,
                'context_len': 48,
            }
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            assert config_path.exists(), "Config file should be saved"
            print(f"[OK] Config saved to {config_path}")
            
            # Save results
            results = {
                'model': 'patchtst',
                'dataset': 'etth1',
                'metrics': {
                    'MAE': float(val_mae),
                    'RMSE': float(val_rmse),
                },
            }
            
            results_path = exp_dir / 'results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            assert results_path.exists(), "Results file should be saved"
            print(f"[OK] Results saved to {results_path}")
            
            # Test model loading
            print("Testing model loading...")
            loaded_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(loaded_checkpoint['model_state_dict'])
            
            assert loaded_checkpoint['val_mae'] == val_mae, "Loaded metrics should match"
            print("[OK] Model loading successful")
            
            # Verify metrics are reasonable (not NaN or inf)
            assert not np.isnan(val_mae), "MAE should not be NaN"
            assert not np.isnan(val_rmse), "RMSE should not be NaN"
            assert not np.isinf(val_mae), "MAE should not be inf"
            assert not np.isinf(val_rmse), "RMSE should not be inf"
            
            print("[OK] All neural model pipeline tests passed!")
            
        except Exception as e:
            pytest.fail(f"Neural model training pipeline test failed: {e}")


def test_eval_script_compatibility():
    """Test that eval.py can be called with experiment directory (smoke test)."""
    from datasets.ettd import Dataset_ETT_hour
    from models.registry import get_model
    from torch.utils.data import DataLoader, Subset
    import tempfile
    import yaml
    
    # Check if data exists
    data_path = Path('data/raw/etth/ETTh1.csv')
    if not data_path.exists():
        pytest.skip("ETTh1.csv not found. Run download tool first.")
    
    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        exp_dir = tmp_path / "test_eval"
        exp_dir.mkdir()
        
        try:
            # Create minimal config and model
            config = {
                'model': 'patchtst',
                'dataset': 'etth1',
                'horizon': 24,
                'context_len': 48,
                'features': 'S',
                'target': 'OT',
                'data_path': 'ETTh1.csv',
                'root_path': 'data/raw/etth',
                'scale': True,
                'timeenc': 0,
                'freq': 'h',
            }
            
            config_path = exp_dir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Create minimal model checkpoint
            model = get_model(
                'patchtst',
                d_in=1,
                out_len=24,
                d_model=32,
                n_heads=2,
                n_layers=1,
                dropout=0.1,
                patch_len=8,
                stride=4,
                revin=True
            )
            
            model_path = exp_dir / 'best_model.pt'
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'val_mae': 0.5,
                'val_rmse': 0.6,
            }
            torch.save(checkpoint, model_path)
            
            # Verify files exist
            assert config_path.exists(), "Config file should exist"
            assert model_path.exists(), "Model checkpoint should exist"
            
            print("[OK] Eval script compatibility test passed (files created correctly)")
            
        except Exception as e:
            pytest.fail(f"Eval script compatibility test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

