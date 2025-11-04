"""ARIMA model implementation for time series forecasting."""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Dict, Any
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings('ignore')


class ARIMAModel:
    """ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting."""
    
    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        auto_select: bool = True,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        seasonal_period: Optional[int] = None,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2,
        trend: Optional[str] = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        method: str = 'lbfgs',
        **kwargs
    ):
        """
        Initialize ARIMA model for time series forecasting.
        
        Args:
            order: (p, d, q) order for ARIMA model. If None and auto_select=True, will be auto-selected
            seasonal_order: (P, D, Q, s) seasonal order for SARIMA model
            auto_select: Whether to automatically select best order using AIC
            max_p: Maximum value of p for auto-selection
            max_d: Maximum value of d for auto-selection
            max_q: Maximum value of q for auto-selection
            seasonal_period: Seasonal period for SARIMA (e.g., 24 for hourly, 96 for 15-min)
            max_P: Maximum value of P for seasonal auto-selection
            max_D: Maximum value of D for seasonal auto-selection
            max_Q: Maximum value of Q for seasonal auto-selection
            trend: Trend component ('c' for constant, 't' for linear, 'ct' for both, None for no trend)
            enforce_stationarity: Whether to enforce stationarity
            enforce_invertibility: Whether to enforce invertibility
            method: Fitting method ('lbfgs', 'nm', 'cg', 'powell')
            **kwargs: Additional parameters for ARIMA model
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_select = auto_select
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal_period = seasonal_period
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.method = method
        self.kwargs = kwargs
        
        # Model and fitted data
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.best_order = None
        self.best_seasonal_order = None
    
    def _check_stationarity(self, data: np.ndarray) -> bool:
        """Check if data is stationary using Augmented Dickey-Fuller test."""
        try:
            result = adfuller(data)
            return result[1] <= 0.05  # p-value <= 0.05 means stationary
        except Exception:
            return False
    
    def _select_order(self, data: np.ndarray, seasonal: bool = False) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int, int]]]:
        """
        Automatically select best ARIMA order using AIC.
        
        Args:
            data: Time series data
            seasonal: Whether to consider seasonal components
        
        Returns:
            best_order: (p, d, q) order
            best_seasonal_order: (P, D, Q, s) seasonal order or None
        """
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_seasonal_order = None
        
        # Determine d (differencing)
        d = 0
        if self.enforce_stationarity:
            current_data = data.copy()
            for i in range(self.max_d + 1):
                if self._check_stationarity(current_data):
                    d = i
                    break
                if i < self.max_d:
                    current_data = np.diff(current_data)
        
        # Grid search for (p, q)
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                try:
                    order = (p, d, q)
                    model = ARIMA(data, order=order, trend=self.trend, **self.kwargs)
                    fitted = model.fit(method=self.method, disp=0)
                    aic = fitted.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_order = order
                except Exception:
                    continue
        
        # Seasonal component if specified
        if seasonal and self.seasonal_period is not None:
            best_seasonal_aic = np.inf
            best_seasonal = None
            
            for P in range(self.max_P + 1):
                for D in range(self.max_D + 1):
                    for Q in range(self.max_Q + 1):
                        try:
                            seasonal_order = (P, D, Q, self.seasonal_period)
                            model = ARIMA(
                                data,
                                order=best_order,
                                seasonal_order=seasonal_order,
                                trend=self.trend,
                                **self.kwargs
                            )
                            fitted = model.fit(method=self.method, disp=0)
                            aic = fitted.aic
                            
                            if aic < best_seasonal_aic:
                                best_seasonal_aic = aic
                                best_seasonal = seasonal_order
                        except Exception:
                            continue
            
            if best_seasonal is not None and best_seasonal_aic < best_aic:
                best_seasonal_order = best_seasonal
        
        return best_order, best_seasonal_order
    
    def fit(self, data: Union[np.ndarray, pd.Series, pd.DataFrame]):
        """
        Fit ARIMA model on time series data.
        
        Args:
            data: Time series data [n_samples] or [n_samples, n_features]
                 If 2D, uses first column or last feature as target
        """
        # Handle different input formats
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        elif isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                # Use first column or last feature
                data = data[:, 0] if data.shape[1] > 0 else data.flatten()
            elif data.ndim > 2:
                # Flatten if 3D
                data = data.flatten()
        
        # Convert to 1D array
        data = np.array(data).flatten()
        
        # Remove NaN and inf values
        mask = np.isfinite(data)
        if not np.all(mask):
            data = data[mask]
        
        if len(data) < 10:
            raise ValueError("Data too short for ARIMA fitting (need at least 10 samples)")
        
        # Select order if auto_select
        if self.auto_select:
            self.best_order, self.best_seasonal_order = self._select_order(
                data,
                seasonal=(self.seasonal_period is not None)
            )
            order = self.best_order
            seasonal_order = self.best_seasonal_order
        else:
            order = self.order or (1, 1, 1)
            seasonal_order = self.seasonal_order
        
        # Initialize and fit model
        try:
            self.model = ARIMA(
                data,
                order=order,
                seasonal_order=seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
                **self.kwargs
            )
            self.fitted_model = self.model.fit(method=self.method, disp=0)
            self.is_fitted = True
            self.order = order
            self.seasonal_order = seasonal_order
        except Exception as e:
            raise ValueError(f"Failed to fit ARIMA model: {e}")
        
        return self
    
    def predict(self, steps: int = 1, start: Optional[int] = None, end: Optional[int] = None) -> np.ndarray:
        """
        Make predictions using fitted model.
        
        Args:
            steps: Number of steps ahead to predict (forecast horizon)
            start: Start index for prediction (if None, starts from end of training data)
            end: End index for prediction (if None, uses start + steps - 1)
        
        Returns:
            predictions: [steps] array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            if start is None and end is None:
                # Forecast future steps
                forecast = self.fitted_model.forecast(steps=steps)
            else:
                # Predict specific range
                forecast = self.fitted_model.predict(start=start, end=end)
            
            # Convert to numpy array
            if isinstance(forecast, pd.Series):
                forecast = forecast.values
            else:
                forecast = np.array(forecast)
            
            # Ensure correct shape
            if forecast.ndim == 0:
                forecast = np.array([forecast])
            elif forecast.ndim > 1:
                forecast = forecast.flatten()
            
            # Return requested number of steps
            return forecast[:steps]
        except Exception as e:
            raise ValueError(f"Failed to make predictions: {e}")
    
    def get_forecast(self, steps: int = 1) -> Dict[str, np.ndarray]:
        """
        Get forecast with confidence intervals.
        
        Args:
            steps: Number of steps ahead to forecast
        
        Returns:
            Dictionary with 'mean', 'conf_int' (lower and upper bounds)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making forecasts")
        
        try:
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            forecast_mean = forecast_result.predicted_mean.values
            conf_int = forecast_result.conf_int().values
            
            return {
                'mean': forecast_mean,
                'conf_int': conf_int,
                'lower': conf_int[:, 0],
                'upper': conf_int[:, 1]
            }
        except Exception as e:
            raise ValueError(f"Failed to get forecast: {e}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'auto_select': self.auto_select,
            'max_p': self.max_p,
            'max_d': self.max_d,
            'max_q': self.max_q,
            'seasonal_period': self.seasonal_period,
            'trend': self.trend,
            'method': self.method,
            **self.kwargs
        }
    
    def set_params(self, **params) -> 'ARIMAModel':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self
    
    def summary(self) -> str:
        """Get model summary."""
        if not self.is_fitted:
            return "Model not fitted yet"
        return str(self.fitted_model.summary())
