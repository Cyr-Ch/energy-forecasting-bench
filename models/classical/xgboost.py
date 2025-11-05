"""XGBoost model implementation for time series forecasting."""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional, Union, Dict, Any


class XGBoostModel:
    """XGBoost model for time series forecasting using sliding window features."""
    
    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        random_state=None,
        n_jobs=-1,
        use_lag_features=True,
        lag_window=10,
        use_rolling_features=True,
        rolling_windows=[3, 7, 14],
        use_time_features=False,
        **kwargs
    ):
        """
        Initialize XGBoost model for time series forecasting.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features when constructing each tree
            min_child_weight: Minimum sum of instance weight in a child
            gamma: Minimum loss reduction required to make a split
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            random_state: Random seed
            n_jobs: Number of parallel threads
            use_lag_features: Whether to create lag features
            lag_window: Number of lag features to create
            use_rolling_features: Whether to create rolling statistics
            rolling_windows: List of window sizes for rolling statistics
            use_time_features: Whether to extract time features from timestamps
            **kwargs: Additional XGBoost parameters
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Feature engineering parameters
        self.use_lag_features = use_lag_features
        self.lag_window = lag_window
        self.use_rolling_features = use_rolling_features
        self.rolling_windows = rolling_windows
        self.use_time_features = use_time_features
        
        # Additional parameters
        self.kwargs = kwargs
        
        # Model and scaler
        self.model = None
        self.feature_names_ = None
        self.is_fitted = False
    
    def _create_features(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create features from time series data using sliding window approach.
        
        Args:
            data: Time series data [n_samples, n_features]
            timestamps: Optional timestamp array for time features
        
        Returns:
            features: Feature matrix [n_samples, n_features_engineered]
        """
        n_samples, n_features = data.shape
        
        # Start with original features
        features_list = [data]
        
        # Lag features
        if self.use_lag_features:
            for lag in range(1, self.lag_window + 1):
                lag_data = np.zeros_like(data)
                lag_data[lag:] = data[:-lag]
                features_list.append(lag_data)
        
        # Rolling statistics
        if self.use_rolling_features:
            for window in self.rolling_windows:
                if window < n_samples:
                    # Rolling mean
                    rolling_mean = pd.DataFrame(data).rolling(window=window, min_periods=1).mean().values
                    features_list.append(rolling_mean)
                    
                    # Rolling std
                    rolling_std = pd.DataFrame(data).rolling(window=window, min_periods=1).std().values
                    rolling_std = np.nan_to_num(rolling_std, nan=0.0)
                    features_list.append(rolling_std)
                    
                    # Rolling min
                    rolling_min = pd.DataFrame(data).rolling(window=window, min_periods=1).min().values
                    features_list.append(rolling_min)
                    
                    # Rolling max
                    rolling_max = pd.DataFrame(data).rolling(window=window, min_periods=1).max().values
                    features_list.append(rolling_max)
        
        # Time features from timestamps
        if self.use_time_features and timestamps is not None:
            try:
                timestamps = pd.to_datetime(timestamps.flatten())
                time_features = np.column_stack([
                    timestamps.month.values,
                    timestamps.day.values,
                    timestamps.weekday.values,
                    timestamps.hour.values,
                ])
                # Reshape to match n_samples
                if len(time_features) != n_samples:
                    # Broadcast or interpolate
                    if len(time_features) == 1:
                        time_features = np.tile(time_features, (n_samples, 1))
                    else:
                        # Interpolate using numpy
                        indices = np.linspace(0, len(time_features) - 1, n_samples)
                        time_features = np.array([np.interp(indices, range(len(time_features)), time_features[:, i]) 
                                                 for i in range(time_features.shape[1])]).T
                features_list.append(time_features)
            except Exception:
                # If time feature extraction fails, skip it
                pass
        
        # Concatenate all features
        features = np.hstack(features_list)
        return features
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
            timestamps: Optional[np.ndarray] = None):
        """
        Fit XGBoost model on time series data.
        
        Args:
            X: Input features [n_samples, n_features] or [n_samples, seq_len, n_features]
            y: Target values [n_samples, horizon] or [n_samples, seq_len, n_features]
               If None, uses last feature of X as target
            timestamps: Optional timestamps for time features [n_samples, seq_len]
        """
        # Handle different input formats
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Save original X shape before flattening
        X_original_shape = X.shape
        X_original_ndim = X.ndim
        
        # Handle sequence data (from dataset loaders)
        if X.ndim == 3:
            # [n_samples, seq_len, n_features] -> flatten to [n_samples * seq_len, n_features]
            n_samples, seq_len, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
            
            # Flatten timestamps if provided
            if timestamps is not None and timestamps.ndim == 2:
                timestamps = timestamps.flatten()
        elif X.ndim == 2:
            X_flat = X
            n_samples = X.shape[0]
            seq_len = 1
        else:
            raise ValueError(f"Expected 2D or 3D input, got {X.ndim}D")
        
        # Handle target
        if y is None:
            # Use last feature as target
            y = X_flat[:, -1]
        else:
            if isinstance(y, pd.DataFrame):
                y = y.values
            
            if y.ndim == 3:
                # [n_samples, seq_len, n_features] -> flatten to [n_samples * seq_len, n_features]
                # Then take last feature or flatten
                y_flat = y.reshape(-1, y.shape[-1])
                y = y_flat[:, -1] if y_flat.shape[1] > 1 else y_flat.flatten()
            elif y.ndim == 2:
                # y is [n_samples, horizon] or [n_samples, 1]
                if X_original_ndim == 3:
                    # X was flattened from [n_samples, seq_len, n_features] to [n_samples * seq_len, n_features]
                    # We need to repeat y for each timestep in each sequence
                    if y.shape[1] == 1:
                        # [n_samples, 1] -> repeat for each timestep -> [n_samples * seq_len]
                        y = np.repeat(y.flatten(), seq_len)
                    else:
                        # [n_samples, horizon] -> use mean of horizon for each sample, then repeat
                        y_mean = y.mean(axis=1)  # [n_samples]
                        y = np.repeat(y_mean, seq_len)  # [n_samples * seq_len]
                else:
                    # X is 2D, y should match
                    if y.shape[1] == 1:
                        y = y.flatten()
                    else:
                        # Use mean or last value
                        y = y.mean(axis=1) if y.shape[1] > 1 else y.flatten()
        
        # Ensure y is 1D for regression
        if y.ndim > 1:
            y = y.flatten() if y.shape[1] == 1 else y[:, -1]
        
        # Final check: ensure y matches X_flat.shape[0]
        if len(y) != X_flat.shape[0]:
            # If mismatch, take first X_flat.shape[0] elements or repeat
            if len(y) < X_flat.shape[0]:
                # Repeat y to match
                repeat_factor = X_flat.shape[0] // len(y)
                y = np.tile(y, repeat_factor)[:X_flat.shape[0]]
            else:
                # Take first X_flat.shape[0] elements
                y = y[:X_flat.shape[0]]
        
        # Create features
        X_features = self._create_features(X_flat, timestamps)
        
        # Store feature names
        self.feature_names_ = [f'feature_{i}' for i in range(X_features.shape[1])]
        
        # Initialize and train XGBoost model
        params = {
            'objective': 'reg:squarederror',
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            **self.kwargs
        }
        
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X_features, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame], timestamps: Optional[np.ndarray] = None, 
                horizon: int = 1) -> np.ndarray:
        """
        Make predictions using fitted model.
        
        Args:
            X: Input features [n_samples, n_features] or [n_samples, seq_len, n_features]
            timestamps: Optional timestamps for time features [n_samples, seq_len]
            horizon: Forecast horizon (number of steps ahead to predict)
        
        Returns:
            predictions: [n_samples, horizon] predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Handle different input formats
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Handle sequence data
        if X.ndim == 3:
            n_samples, seq_len, n_features = X.shape
            # For forecasting, use only the last timestep of each sequence
            # This gives us the most recent context for each sequence
            X_flat = X[:, -1, :]  # [n_samples, n_features] - last timestep of each sequence
            
            if timestamps is not None and timestamps.ndim == 2:
                # Use last timestep's timestamp for each sequence
                timestamps = timestamps[:, -1]  # [n_samples]
        elif X.ndim == 2:
            X_flat = X
            n_samples = X.shape[0]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {X.ndim}D")
        
        # For multi-step forecasting, use recursive prediction
        # Start with the last timestep of each sequence
        predictions = []
        current_X = X_flat.copy()  # [n_samples, n_features]
        current_timestamps = timestamps.copy() if timestamps is not None else None
        
        for step in range(horizon):
            # Create features from current state
            X_features = self._create_features(current_X, current_timestamps)
            
            # Predict one step ahead for each sample
            pred = self.model.predict(X_features)  # [n_samples]
            predictions.append(pred)
            
            # Update input for next step (recursive forecasting)
            if step < horizon - 1:
                # Update current_X by shifting and appending the prediction
                # For univariate case, replace the last feature with prediction
                if current_X.shape[1] == 1:
                    current_X = pred.reshape(-1, 1)
                else:
                    # Shift features and append prediction as new value
                    current_X = np.roll(current_X, -1, axis=1)
                    current_X[:, -1] = pred
        
        # Stack predictions [horizon, n_samples] -> [n_samples, horizon]
        predictions = np.array(predictions).T  # [n_samples, horizon]
        
        return predictions
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'use_lag_features': self.use_lag_features,
            'lag_window': self.lag_window,
            'use_rolling_features': self.use_rolling_features,
            'rolling_windows': self.rolling_windows,
            'use_time_features': self.use_time_features,
            **self.kwargs
        }
    
    def set_params(self, **params) -> 'XGBoostModel':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self
