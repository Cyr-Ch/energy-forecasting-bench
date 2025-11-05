"""Prophet model implementation."""

import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Union, Optional, List


class ProphetModel:
    """
    Facebook Prophet model for time series forecasting.
    
    Prophet is an additive regression model that automatically detects changepoints,
    seasonality, and holidays in time series data.
    
    Key Parameters Users Can Tweak for Performance:
    ================================================
    
    1. **Growth Parameters:**
       - growth: 'linear' or 'logistic' - Controls trend type
       - changepoint_prior_scale: Controls flexibility of trend changes (default: 0.05)
         * Higher values = more flexible trend (can overfit)
         * Lower values = less flexible trend (can underfit)
       - n_changepoints: Number of potential changepoints (default: 25)
       - changepoints: Manual list of changepoint dates (optional)
    
    2. **Seasonality Parameters:**
       - yearly_seasonality: Fit yearly seasonality (default: 'auto' or True/False)
       - weekly_seasonality: Fit weekly seasonality (default: 'auto' or True/False)
       - daily_seasonality: Fit daily seasonality (default: 'auto' or True/False)
       - seasonality_mode: 'additive' or 'multiplicative' (default: 'additive')
         * Use 'multiplicative' when seasonality scales with trend
       - seasonality_prior_scale: Controls strength of seasonality (default: 10.0)
         * Higher values = stronger seasonality effect
         * Lower values = weaker seasonality effect
    
    3. **Holiday Parameters:**
       - holidays: DataFrame with 'holiday' and 'ds' columns (optional)
       - holidays_prior_scale: Controls strength of holiday effects (default: 10.0)
    
    4. **Uncertainty Parameters:**
       - interval_width: Width of uncertainty intervals (default: 0.80)
       - uncertainty_samples: Number of samples for uncertainty estimation (default: 1000)
       - mcmc_samples: If > 0, uses MCMC for uncertainty (slower but more accurate)
    
    5. **Other Parameters:**
       - regressors: List of additional regressor column names (optional)
       - add_country_holidays: Add country holidays (e.g., 'US', 'UK')
    """
    
    def __init__(self, config: dict):
        """
        Initialize Prophet model with configuration.
        
        Args:
            config: Dictionary containing model parameters. Common parameters:
                - growth: 'linear' or 'logistic' (default: 'linear')
                - changepoint_prior_scale: float (default: 0.05)
                - seasonality_prior_scale: float (default: 10.0)
                - holidays_prior_scale: float (default: 10.0)
                - seasonality_mode: 'additive' or 'multiplicative' (default: 'additive')
                - yearly_seasonality: bool or 'auto' (default: 'auto')
                - weekly_seasonality: bool or 'auto' (default: 'auto')
                - daily_seasonality: bool or 'auto' (default: 'auto')
                - n_changepoints: int (default: 25)
                - interval_width: float (default: 0.80)
                - uncertainty_samples: int (default: 1000)
                - mcmc_samples: int (default: 0)
                - add_country_holidays: str (e.g., 'US', 'UK') or None
                - holidays: pd.DataFrame or None
                - regressors: list of str or None
                - target_column: str - Name of target column in data (default: uses first numeric column)
                - date_column: str - Name of date column (default: uses index if DatetimeIndex)
        """
        self.config = config
        
        # Extract Prophet-specific parameters with defaults
        prophet_params = {
            'growth': config.get('growth', 'linear'),
            'changepoint_prior_scale': config.get('changepoint_prior_scale', 0.05),
            'seasonality_prior_scale': config.get('seasonality_prior_scale', 10.0),
            'holidays_prior_scale': config.get('holidays_prior_scale', 10.0),
            'seasonality_mode': config.get('seasonality_mode', 'additive'),
            'yearly_seasonality': config.get('yearly_seasonality', 'auto'),
            'weekly_seasonality': config.get('weekly_seasonality', 'auto'),
            'daily_seasonality': config.get('daily_seasonality', 'auto'),
            'n_changepoints': config.get('n_changepoints', 25),
            'interval_width': config.get('interval_width', 0.80),
            'uncertainty_samples': config.get('uncertainty_samples', 1000),
            'mcmc_samples': config.get('mcmc_samples', 0),
        }
        
        # Optional parameters
        if 'add_country_holidays' in config:
            prophet_params['add_country_holidays'] = config['add_country_holidays']
        
        if 'holidays' in config and config['holidays'] is not None:
            prophet_params['holidays'] = config['holidays']
        
        if 'changepoints' in config and config['changepoints'] is not None:
            prophet_params['changepoints'] = config['changepoints']
        
        # Initialize Prophet model
        self.model = Prophet(**prophet_params)
        
        # Store additional regressors if specified
        self.regressors = config.get('regressors', None)
        if self.regressors:
            for regressor in self.regressors:
                self.model.add_regressor(regressor)
        
        # Store data format info
        self.target_column = config.get('target_column', None)
        self.date_column = config.get('date_column', None)
        
        self.is_fitted = False
    
    def _prepare_data(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Convert input data to Prophet format (ds and y columns).
        
        Args:
            data: Input data - can be:
                - pd.DataFrame with datetime index or date column
                - pd.Series with datetime index
                - np.ndarray (requires date_column and target_column in config)
        
        Returns:
            DataFrame with 'ds' (datestamp) and 'y' (target) columns
        """
        if isinstance(data, pd.Series):
            df = data.to_frame()
            if data.name:
                df.columns = [data.name]
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, np.ndarray):
            # Convert numpy array to DataFrame
            if len(data.shape) == 1:
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(data)
            # Need date column from config
            if self.date_column and self.date_column in df.columns:
                df['ds'] = pd.to_datetime(df[self.date_column])
            else:
                raise ValueError("For numpy arrays, provide date_column in config")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Handle datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            df['ds'] = df.index
        elif 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
        elif self.date_column and self.date_column in df.columns:
            df['ds'] = pd.to_datetime(df[self.date_column])
        else:
            # Try to infer date column
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                df['ds'] = pd.to_datetime(df[date_cols[0]])
            else:
                raise ValueError("Could not find date column. Provide datetime index or date_column in config.")
        
        # Handle target column
        if self.target_column and self.target_column in df.columns:
            df['y'] = pd.to_numeric(df[self.target_column], errors='coerce')
        else:
            # Use first numeric column that's not 'ds'
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'ds']
            if len(numeric_cols) == 0:
                raise ValueError("Could not find target column. Provide target_column in config.")
            df['y'] = pd.to_numeric(df[numeric_cols[0]], errors='coerce')
        
        # Drop rows with NaN in y
        df = df.dropna(subset=['y'])
        
        # Ensure ds is datetime
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Sort by date
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Select only required columns (ds, y, and any regressors)
        cols = ['ds', 'y']
        if self.regressors:
            for regressor in self.regressors:
                if regressor in df.columns:
                    cols.append(regressor)
        
        return df[cols]
    
    def fit(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]):
        """
        Fit Prophet model on training data.
        
        Args:
            data: Training data - DataFrame with datetime index/column and target column,
                  or Series with datetime index, or numpy array with date/target columns
        """
        # Prepare data for Prophet
        prophet_df = self._prepare_data(data)
        
        # Fit the model
        self.model.fit(prophet_df)
        self.is_fitted = True
    
    def predict(
        self, 
        steps: Optional[int] = None, 
        future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Make predictions with Prophet model.
        
        Args:
            steps: Number of future steps to predict (if future not provided)
            future: Optional DataFrame with 'ds' column and regressors for specific dates
        
        Returns:
            DataFrame with columns:
                - ds: datestamp
                - yhat: predicted value
                - yhat_lower: lower bound of prediction interval
                - yhat_upper: upper bound of prediction interval
                - trend, seasonal, etc.: component breakdowns
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        if future is not None:
            # Use provided future dataframe
            future_df = future.copy()
            if 'ds' not in future_df.columns:
                raise ValueError("future DataFrame must have 'ds' column")
            future_df['ds'] = pd.to_datetime(future_df['ds'])
        else:
            # Create future dataframe
            if steps is None:
                steps = self.config.get('pred_len', 96)  # Default prediction length
            
            # Get last date from training data
            last_date = self.model.history['ds'].max()
            
            # Determine frequency by calculating time difference from data
            if len(self.model.history) > 1:
                # Calculate time difference from actual data (most reliable)
                time_diff = self.model.history['ds'].diff().dropna()
                if len(time_diff) > 0:
                    # Use the most common time difference
                    freq_delta = time_diff.mode()[0] if len(time_diff.mode()) > 0 else time_diff.iloc[0]
                else:
                    freq_delta = pd.Timedelta(hours=1)
                
                # Try to infer frequency string for pd.date_range
                freq_str = pd.infer_freq(self.model.history['ds'])
                if freq_str is None:
                    # Convert Timedelta to string format for pd.date_range
                    # Common mappings: hours -> 'H', days -> 'D', minutes -> 'T'
                    if freq_delta >= pd.Timedelta(days=1):
                        freq_str = f'{int(freq_delta / pd.Timedelta(days=1))}D'
                    elif freq_delta >= pd.Timedelta(hours=1):
                        freq_str = f'{int(freq_delta / pd.Timedelta(hours=1))}H'
                    elif freq_delta >= pd.Timedelta(minutes=1):
                        freq_str = f'{int(freq_delta / pd.Timedelta(minutes=1))}T'
                    else:
                        freq_str = f'{int(freq_delta / pd.Timedelta(seconds=1))}S'
            else:
                freq_delta = pd.Timedelta(hours=1)
                freq_str = '1H'
            
            # Create future dates
            future_dates = pd.date_range(
                start=last_date + freq_delta,
                periods=steps,
                freq=freq_str
            )
            
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Add regressors if specified (set to 0 or mean as default)
            if self.regressors:
                for regressor in self.regressors:
                    # Use mean value from training data as default
                    if hasattr(self.model, 'history') and regressor in self.model.history.columns:
                        future_df[regressor] = self.model.history[regressor].mean()
                    else:
                        future_df[regressor] = 0
        
        # Make predictions
        forecast = self.model.predict(future_df)
        
        return forecast
    
    def get_components(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Extract component breakdown from forecast.
        
        Args:
            forecast: DataFrame returned by predict()
        
        Returns:
            DataFrame with trend, seasonal components, etc.
        """
        component_cols = ['ds', 'trend']
        
        # Add seasonal components if they exist
        for col in forecast.columns:
            if 'seasonal' in col.lower() or 'weekly' in col.lower() or \
               'yearly' in col.lower() or 'daily' in col.lower():
                component_cols.append(col)
        
        # Add holiday component if exists
        if 'holidays' in forecast.columns:
            component_cols.append('holidays')
        
        # Add regressor components if they exist
        if self.regressors:
            for regressor in self.regressors:
                if regressor in forecast.columns:
                    component_cols.append(regressor)
        
        return forecast[component_cols]

