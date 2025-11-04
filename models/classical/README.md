# Classical Models for Time Series Forecasting

This directory contains classical machine learning models for time series forecasting, including XGBoost, ARIMA, and Prophet.

## XGBoost Model

XGBoost is a gradient boosting framework that can be used for time series forecasting by converting sequences into tabular features using sliding windows.

### Features

- **Automatic Feature Engineering**: Creates lag features, rolling statistics, and optional time features
- **Multi-step Forecasting**: Supports recursive prediction for multiple horizons
- **Flexible Input Format**: Handles both 2D arrays and 3D sequences (compatible with ETT dataset loaders)
- **Comprehensive Parameters**: Full control over XGBoost hyperparameters

### Usage Examples

#### Basic Usage with ETT Dataset

```python
import numpy as np
from torch.utils.data import DataLoader
from datasets.ettd import Dataset_ETT_hour
from models.classical.xgboost import XGBoostModel

# Load ETT dataset
train_data = Dataset_ETT_hour(
    root_path='data/raw/etth',
    flag='train',
    size=[96, 48, 96],  # [seq_len, label_len, pred_len]
    features='S',  # Univariate
    data_path='ETTh1.csv',
    target='OT',
    scale=True,
    timeenc=0,
    freq='h'
)

val_data = Dataset_ETT_hour(
    root_path='data/raw/etth',
    flag='val',
    size=[96, 48, 96],
    features='S',
    data_path='ETTh1.csv',
    target='OT',
    scale=True,
    timeenc=0,
    freq='h'
)

# Prepare data for XGBoost
# XGBoost works with 2D arrays, so we need to extract sequences
X_train = []
y_train = []
timestamps_train = []

for i in range(len(train_data)):
    seq_x, seq_y, seq_x_mark, seq_y_mark = train_data[i]
    # Use encoder input and target
    X_train.append(seq_x)
    y_train.append(seq_y[-96:, 0])  # Last 96 steps (pred_len)
    timestamps_train.append(seq_x_mark)

X_train = np.array(X_train)  # [n_samples, seq_len, n_features]
y_train = np.array(y_train)  # [n_samples, pred_len]
timestamps_train = np.array(timestamps_train)  # [n_samples, seq_len, time_features]

# Initialize XGBoost model
model = XGBoostModel(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_lag_features=True,
    lag_window=10,
    use_rolling_features=True,
    rolling_windows=[3, 7, 14],
    use_time_features=True,  # Use time features from timestamps
    random_state=42
)

# Fit model
model.fit(X_train, y_train, timestamps=timestamps_train)

# Prepare validation data
X_val = []
y_val = []
timestamps_val = []

for i in range(len(val_data)):
    seq_x, seq_y, seq_x_mark, seq_y_mark = val_data[i]
    X_val.append(seq_x)
    y_val.append(seq_y[-96:, 0])
    timestamps_val.append(seq_x_mark)

X_val = np.array(X_val)
y_val = np.array(y_val)
timestamps_val = np.array(timestamps_val)

# Make predictions
predictions = model.predict(X_val, timestamps=timestamps_val, horizon=96)

# Evaluate (example with MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_val.flatten(), predictions.flatten())
print(f"Validation MSE: {mse:.4f}")
```

#### Simple 2D Array Usage

```python
import numpy as np
from models.classical.xgboost import XGBoostModel

# Create sample time series data
n_samples = 1000
n_features = 1
X = np.random.randn(n_samples, n_features).cumsum(axis=0)  # Random walk

# Create target (next 10 steps)
horizon = 10
y = np.array([X[i:i+horizon].flatten() for i in range(n_samples-horizon)])
X_train = X[:-horizon]

# Initialize and fit model
model = XGBoostModel(
    n_estimators=50,
    max_depth=5,
    learning_rate=0.1,
    use_lag_features=True,
    lag_window=5,
    use_rolling_features=True,
    rolling_windows=[3, 7]
)

model.fit(X_train, y)

# Predict
predictions = model.predict(X_train[-10:], horizon=10)
print(f"Predictions shape: {predictions.shape}")  # [10, 10]
```

#### Custom Parameters

```python
from models.classical.xgboost import XGBoostModel

# Initialize with custom XGBoost parameters
model = XGBoostModel(
    # XGBoost parameters
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    random_state=42,
    n_jobs=-1,
    
    # Feature engineering parameters
    use_lag_features=True,
    lag_window=15,  # Create 15 lag features
    use_rolling_features=True,
    rolling_windows=[3, 7, 14, 30],  # Multiple rolling windows
    use_time_features=False,  # Disable time features
)

# Fit and predict as before
model.fit(X_train, y_train)
predictions = model.predict(X_val, horizon=96)
```

#### Without Time Features

```python
from models.classical.xgboost import XGBoostModel

# Initialize model without time features
model = XGBoostModel(
    n_estimators=100,
    max_depth=6,
    use_lag_features=True,
    lag_window=10,
    use_rolling_features=True,
    use_time_features=False  # Disable time features
)

# Fit without timestamps
model.fit(X_train, y_train)

# Predict without timestamps
predictions = model.predict(X_val, horizon=96)
```

#### Feature Engineering Details

The XGBoost model automatically creates the following features:

1. **Original Features**: The input time series data
2. **Lag Features**: Previous values (lag 1, 2, ..., lag_window)
3. **Rolling Statistics**: For each window size:
   - Rolling mean
   - Rolling standard deviation
   - Rolling minimum
   - Rolling maximum
4. **Time Features** (optional): Month, day, weekday, hour extracted from timestamps

Example with `lag_window=5` and `rolling_windows=[3, 7]`:
- Original features: 1
- Lag features: 5
- Rolling features: 3 windows × 4 statistics = 12
- Time features: 4 (if enabled)
- **Total**: ~22 features per sample

### Parameters

#### XGBoost Parameters

- `n_estimators` (default: 100): Number of boosting rounds
- `max_depth` (default: 6): Maximum tree depth
- `learning_rate` (default: 0.1): Learning rate (eta)
- `subsample` (default: 1.0): Subsample ratio of training instances
- `colsample_bytree` (default: 1.0): Subsample ratio of features
- `min_child_weight` (default: 1): Minimum sum of instance weight in a child
- `gamma` (default: 0): Minimum loss reduction for split
- `reg_alpha` (default: 0): L1 regularization
- `reg_lambda` (default: 1): L2 regularization
- `random_state` (default: None): Random seed
- `n_jobs` (default: -1): Number of parallel threads

#### Feature Engineering Parameters

- `use_lag_features` (default: True): Create lag features
- `lag_window` (default: 10): Number of lag features to create
- `use_rolling_features` (default: True): Create rolling statistics
- `rolling_windows` (default: [3, 7, 14]): Window sizes for rolling statistics
- `use_time_features` (default: False): Extract time features from timestamps

### Methods

- `fit(X, y, timestamps=None)`: Fit the model on training data
  - `X`: Input features [n_samples, n_features] or [n_samples, seq_len, n_features]
  - `y`: Target values [n_samples, horizon] or [n_samples, seq_len, n_features]
  - `timestamps`: Optional timestamps [n_samples, seq_len] for time features

- `predict(X, timestamps=None, horizon=1)`: Make predictions
  - `X`: Input features [n_samples, n_features] or [n_samples, seq_len, n_features]
  - `timestamps`: Optional timestamps [n_samples, seq_len]
  - `horizon`: Number of steps ahead to predict
  - Returns: [n_samples, horizon] predictions

- `get_params()`: Get model parameters
- `set_params(**params)`: Set model parameters

### Tips for Best Performance

1. **Feature Engineering**: Enable lag and rolling features for better capture of temporal patterns
2. **Time Features**: Use time features if your data has strong seasonality (hourly, daily, weekly patterns)
3. **Hyperparameter Tuning**: Tune `max_depth`, `learning_rate`, and `n_estimators` for best results
4. **Regularization**: Use `reg_alpha` and `reg_lambda` to prevent overfitting
5. **Early Stopping**: Consider using XGBoost's early stopping with validation set

### Comparison with Deep Learning Models

XGBoost is a good baseline for time series forecasting:
- **Pros**: Fast training, interpretable, works well with tabular features, doesn't require GPU
- **Cons**: Limited to engineered features, may struggle with very long sequences, less flexible than neural networks

For the ETT dataset, XGBoost can serve as a strong baseline before trying deep learning models like Autoformer or PatchTST.

## ARIMA Model

ARIMA (AutoRegressive Integrated Moving Average) is a classical statistical model for time series forecasting. It combines autoregression (AR), differencing (I), and moving average (MA) components.

### Features

- **Automatic Order Selection**: Automatically selects best (p, d, q) order using AIC
- **SARIMA Support**: Supports seasonal ARIMA with configurable seasonal periods
- **Univariate Forecasting**: Works on single time series (univariate)
- **Confidence Intervals**: Provides forecast confidence intervals
- **Stationarity Handling**: Automatically handles non-stationary data with differencing

### Usage Examples

#### Basic Usage with ETT Dataset

```python
import numpy as np
from datasets.ettd import Dataset_ETT_hour
from models.classical.arima import ARIMAModel

# Load ETT dataset (ARIMA works on univariate data)
train_data = Dataset_ETT_hour(
    root_path='data/raw/etth',
    flag='train',
    size=[96, 48, 96],
    features='S',  # Univariate only
    data_path='ETTh1.csv',
    target='OT',
    scale=True,
    timeenc=0,
    freq='h'
)

# Extract time series data
# ARIMA works on a single univariate series
train_series = []
for i in range(len(train_data)):
    seq_x, seq_y, _, _ = train_data[i]
    # Use the last values of the sequence
    train_series.extend(seq_x[:, 0].tolist())

train_series = np.array(train_series)

# Initialize ARIMA model with auto-selection
model = ARIMAModel(
    auto_select=True,  # Automatically select best order
    max_p=5,          # Maximum AR order
    max_d=2,           # Maximum differencing order
    max_q=5,           # Maximum MA order
    seasonal_period=24,  # 24 hours for hourly data
    trend='c'          # Include constant term
)

# Fit model
model.fit(train_series)

# Make predictions (96 steps ahead)
predictions = model.predict(steps=96)

print(f"Predictions shape: {predictions.shape}")  # [96]
print(f"First 10 predictions: {predictions[:10]}")
```

#### Manual Order Specification

```python
from models.classical.arima import ARIMAModel
import numpy as np

# Initialize with manual order (p, d, q) = (2, 1, 2)
model = ARIMAModel(
    order=(2, 1, 2),      # (p, d, q) - AR(2), I(1), MA(2)
    auto_select=False,    # Don't auto-select
    trend='c'             # Include constant
)

# Fit on data
data = np.random.randn(1000).cumsum()  # Sample data
model.fit(data)

# Predict 10 steps ahead
predictions = model.predict(steps=10)
```

#### Seasonal ARIMA (SARIMA)

```python
from models.classical.arima import ARIMAModel
import numpy as np

# SARIMA with seasonal period (e.g., 24 for hourly, 96 for 15-min)
model = ARIMAModel(
    order=(1, 1, 1),           # Non-seasonal order
    seasonal_order=(1, 1, 1, 24),  # Seasonal order (P, D, Q, s)
    seasonal_period=24,        # 24 hours for hourly data
    auto_select=False,
    trend='c'
)

# Fit model
data = np.random.randn(1000).cumsum()
model.fit(data)

# Predict with confidence intervals
forecast = model.get_forecast(steps=24)
print(f"Mean forecast: {forecast['mean']}")
print(f"Confidence interval: {forecast['conf_int']}")
```

#### Auto-Selection with Seasonal Component

```python
from models.classical.arima import ARIMAModel
from datasets.ettd import Dataset_ETT_hour
import numpy as np

# Load data
train_data = Dataset_ETT_hour(
    root_path='data/raw/etth',
    flag='train',
    size=[96, 48, 96],
    features='S',
    data_path='ETTh1.csv',
    target='OT',
    scale=True,
    timeenc=0,
    freq='h'
)

# Extract series
train_series = []
for i in range(len(train_data)):
    seq_x, _, _, _ = train_data[i]
    train_series.extend(seq_x[:, 0].tolist())

train_series = np.array(train_series)

# Auto-select with seasonal component
model = ARIMAModel(
    auto_select=True,
    max_p=5,
    max_d=2,
    max_q=5,
    seasonal_period=24,  # Hourly seasonality
    max_P=2,              # Max seasonal AR
    max_D=1,              # Max seasonal differencing
    max_Q=2,              # Max seasonal MA
    trend='c'
)

# Fit and see selected order
model.fit(train_series)
print(f"Selected order: {model.order}")
print(f"Selected seasonal order: {model.seasonal_order}")

# Get model summary
print(model.summary())
```

#### Forecasting with Confidence Intervals

```python
from models.classical.arima import ARIMAModel
import numpy as np

# Fit model
model = ARIMAModel(
    order=(2, 1, 2),
    auto_select=False,
    trend='c'
)

data = np.random.randn(1000).cumsum()
model.fit(data)

# Get forecast with confidence intervals
forecast = model.get_forecast(steps=24)

print(f"Mean forecast: {forecast['mean']}")
print(f"Lower bound (95%): {forecast['lower']}")
print(f"Upper bound (95%): {forecast['upper']}")
```

### Parameters

#### ARIMA Order Parameters

- `order` (default: None): (p, d, q) tuple for ARIMA model
  - `p`: AR order (autoregressive terms)
  - `d`: Differencing order (integration)
  - `q`: MA order (moving average terms)
- `seasonal_order` (default: None): (P, D, Q, s) tuple for seasonal component
  - `P`: Seasonal AR order
  - `D`: Seasonal differencing order
  - `Q`: Seasonal MA order
  - `s`: Seasonal period (e.g., 24 for hourly, 96 for 15-min)

#### Auto-Selection Parameters

- `auto_select` (default: True): Automatically select best order
- `max_p` (default: 5): Maximum AR order for auto-selection
- `max_d` (default: 2): Maximum differencing order
- `max_q` (default: 5): Maximum MA order
- `max_P` (default: 2): Maximum seasonal AR order
- `max_D` (default: 1): Maximum seasonal differencing order
- `max_Q` (default: 2): Maximum seasonal MA order

#### Other Parameters

- `seasonal_period` (default: None): Seasonal period for SARIMA
  - `24` for hourly data (daily seasonality)
  - `96` for 15-minute data (daily seasonality)
  - `168` for hourly data (weekly seasonality)
- `trend` (default: None): Trend component
  - `'c'`: Constant (intercept)
  - `'t'`: Linear trend
  - `'ct'`: Both constant and linear
  - `None`: No trend
- `method` (default: 'lbfgs'): Fitting method ('lbfgs', 'nm', 'cg', 'powell')
- `enforce_stationarity` (default: True): Enforce stationarity constraints
- `enforce_invertibility` (default: True): Enforce invertibility constraints

### Methods

- `fit(data)`: Fit ARIMA model on time series data
  - `data`: 1D array or series [n_samples]

- `predict(steps, start, end)`: Make predictions
  - `steps`: Number of steps ahead to predict
  - `start`: Start index (optional)
  - `end`: End index (optional)
  - Returns: [steps] array of predictions

- `get_forecast(steps)`: Get forecast with confidence intervals
  - `steps`: Number of steps ahead
  - Returns: Dictionary with 'mean', 'conf_int', 'lower', 'upper'

- `get_params()`: Get model parameters
- `set_params(**params)`: Set model parameters
- `summary()`: Get model summary statistics

### Tips for Best Performance

1. **Stationarity**: ARIMA requires stationary data. The model automatically handles differencing if `enforce_stationarity=True`
2. **Seasonal Period**: Set appropriate `seasonal_period` based on your data frequency
   - Hourly: 24 (daily) or 168 (weekly)
   - 15-minute: 96 (daily) or 672 (weekly)
3. **Auto-Selection**: Use `auto_select=True` for automatic order selection, but be aware it can be slow
4. **Order Selection**: For faster results, manually specify order based on ACF/PACF plots
5. **Trend Component**: Include trend if your data shows clear trend (use `trend='c'` or `trend='ct'`)

### Limitations

- **Univariate Only**: ARIMA works on single time series, not multivariate
- **Linear Model**: Assumes linear relationships, may miss non-linear patterns
- **Computational Cost**: Auto-selection can be slow for large datasets
- **Stationarity Requirement**: Data should be stationary (or made stationary through differencing)

### Comparison with Other Models

ARIMA is a classical baseline:
- **Pros**: Interpretable, statistical foundation, handles seasonality well, provides confidence intervals
- **Cons**: Univariate only, linear assumptions, may struggle with complex patterns

For the ETT dataset, ARIMA serves as a strong statistical baseline alongside XGBoost and deep learning models.

## Prophet Model

Prophet is Facebook's additive regression model for time series forecasting that automatically detects changepoints, seasonality, and holidays. It's particularly well-suited for data with strong seasonal patterns and can handle missing data and outliers robustly.

### Features

- **Automatic Seasonality Detection**: Automatically fits yearly, weekly, and daily seasonality patterns
- **Holiday Effects**: Built-in support for holidays and custom events
- **Robust to Missing Data**: Handles missing values and outliers gracefully
- **Uncertainty Intervals**: Provides prediction intervals, not just point forecasts
- **Component Breakdown**: Separates trend, seasonality, and holiday effects
- **Flexible Input Format**: Handles pandas DataFrames, Series, and numpy arrays with datetime index

### Usage Examples

#### Basic Usage with ETT Dataset

```python
import pandas as pd
import numpy as np
from datasets.ettd import Dataset_ETT_hour
from models.classical.prophet import ProphetModel

# Load ETT dataset
train_data = Dataset_ETT_hour(
    root_path='data/raw/etth',
    flag='train',
    size=[96, 48, 96],  # [seq_len, label_len, pred_len]
    features='S',  # Univariate
    data_path='ETTh1.csv',
    target='OT',
    scale=False,  # Prophet works better with unscaled data
    timeenc=0,
    freq='h'
)

# Convert to DataFrame format for Prophet
# Prophet expects data with datetime index and target column
train_df = train_data.df.copy()
train_df.index = pd.to_datetime(train_df.index)

# Initialize Prophet model
model = ProphetModel(
    config={
        'growth': 'linear',
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': 'auto',
        'weekly_seasonality': 'auto',
        'daily_seasonality': 'auto',
        'target_column': 'OT',
        'pred_len': 96
    }
)

# Fit model
model.fit(train_df)

# Make predictions
forecast = model.predict(steps=96)

# Extract predictions
predictions = forecast['yhat'].values

# Get uncertainty intervals
lower_bound = forecast['yhat_lower'].values
upper_bound = forecast['yhat_upper'].values

# Get component breakdown
components = model.get_components(forecast)
print(components.head())
```

#### Simple DataFrame Usage

```python
import pandas as pd
import numpy as np
from models.classical.prophet import ProphetModel

# Create sample time series data
dates = pd.date_range(start='2020-01-01', periods=1000, freq='H')
values = np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.randn(1000) * 0.1
df = pd.DataFrame({'value': values}, index=dates)

# Initialize Prophet model with default parameters
model = ProphetModel(
    config={
        'target_column': 'value',
        'pred_len': 48
    }
)

# Fit model
model.fit(df)

# Make predictions
forecast = model.predict(steps=48)

# Visualize results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df.index[-100:], df['value'].values[-100:], label='Historical')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                 alpha=0.3, color='red', label='Uncertainty')
plt.legend()
plt.show()
```

#### Custom Parameters for Better Performance

```python
from models.classical.prophet import ProphetModel
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')

# Initialize Prophet with custom parameters for better performance
model = ProphetModel(
    config={
        # Growth parameters - adjust for trend flexibility
        'growth': 'linear',  # or 'logistic' for bounded growth
        'changepoint_prior_scale': 0.10,  # Higher = more flexible trend (can overfit)
        'n_changepoints': 30,  # More changepoints for complex trends
        
        # Seasonality parameters - adjust for seasonal patterns
        'seasonality_mode': 'multiplicative',  # Use when seasonality scales with trend
        'seasonality_prior_scale': 15.0,  # Higher = stronger seasonality effect
        
        # Enable specific seasonalities
        'yearly_seasonality': True,  # For yearly patterns
        'weekly_seasonality': True,  # For weekly patterns
        'daily_seasonality': True,  # For daily patterns
        
        # Holiday parameters
        'holidays_prior_scale': 10.0,
        'add_country_holidays': 'US',  # Add US holidays
        
        # Uncertainty parameters
        'interval_width': 0.95,  # 95% confidence intervals
        'uncertainty_samples': 2000,  # More samples for better uncertainty estimates
        
        # Data format
        'target_column': 'target',  # Your target column name
        'pred_len': 96
    }
)

# Fit and predict
model.fit(df)
forecast = model.predict(steps=96)
```

#### With Custom Holidays

```python
from models.classical.prophet import ProphetModel
import pandas as pd

# Create custom holidays DataFrame
holidays = pd.DataFrame({
    'holiday': 'custom_event',
    'ds': pd.to_datetime(['2024-01-15', '2024-06-20', '2024-12-25']),
    'lower_window': 0,
    'upper_window': 1,
})

# Initialize model with custom holidays
model = ProphetModel(
    config={
        'holidays': holidays,
        'holidays_prior_scale': 15.0,  # Stronger holiday effect
        'target_column': 'sales',
        'pred_len': 30
    }
)

# Fit and predict
model.fit(sales_df)
forecast = model.predict(steps=30)
```

#### With Additional Regressors

```python
from models.classical.prophet import ProphetModel
import pandas as pd

# Prepare data with additional regressors
df = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=1000, freq='D'),
    'target': np.random.randn(1000).cumsum(),
    'regressor1': np.random.randn(1000),
    'regressor2': np.random.randn(1000)
})

# Initialize model with regressors
model = ProphetModel(
    config={
        'regressors': ['regressor1', 'regressor2'],
        'target_column': 'target',
        'date_column': 'date',
        'pred_len': 30
    }
)

# Fit model
model.fit(df)

# For predictions, provide future regressor values
future_df = pd.DataFrame({
    'ds': pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=30, freq='D'),
    'regressor1': np.random.randn(30),  # Future regressor values
    'regressor2': np.random.randn(30)
})

# Predict with future regressors
forecast = model.predict(future=future_df)
```

### Key Parameters for Performance Tuning

#### Growth Parameters

- **`changepoint_prior_scale`** (default: 0.05): Controls flexibility of trend changes
  - **Higher values (0.1-0.5)**: More flexible trend, can capture sudden changes better
  - **Lower values (0.01-0.05)**: Less flexible trend, smoother but may miss changes
  - **Tune if**: Your data has sudden trend changes or looks too smooth/too wiggly

- **`n_changepoints`** (default: 25): Number of potential changepoints
  - **Higher values**: More changepoints, more flexible
  - **Lower values**: Fewer changepoints, smoother trend
  - **Tune if**: You have long time series with many trend changes

#### Seasonality Parameters

- **`seasonality_mode`** (default: 'additive'): How seasonality interacts with trend
  - **'additive'**: Seasonality is added to trend (constant amplitude)
  - **'multiplicative'**: Seasonality scales with trend (amplitude grows with trend)
  - **Use multiplicative if**: Your seasonal patterns get stronger as the trend increases

- **`seasonality_prior_scale`** (default: 10.0): Controls strength of seasonality
  - **Higher values (15-20)**: Stronger seasonality effect
  - **Lower values (5-10)**: Weaker seasonality effect
  - **Tune if**: Seasonality is too strong/too weak in predictions

- **`yearly_seasonality`**, **`weekly_seasonality`**, **`daily_seasonality`**: Enable/disable specific patterns
  - Set to `False` if you know your data doesn't have that pattern
  - Set to `True` to force it even with little data

#### Holiday Parameters

- **`holidays_prior_scale`** (default: 10.0): Controls strength of holiday effects
  - **Higher values**: Stronger holiday effects
  - **Lower values**: Weaker holiday effects
  - **Tune if**: Holidays are over/under-predicted

#### Uncertainty Parameters

- **`interval_width`** (default: 0.80): Width of prediction intervals
  - 0.80 = 80% confidence intervals
  - 0.95 = 95% confidence intervals

- **`uncertainty_samples`** (default: 1000): Number of samples for uncertainty
  - **Higher values**: More accurate but slower
  - **Lower values**: Faster but less accurate uncertainty

### Methods

- **`fit(data)`**: Fit the model on training data
  - `data`: pandas DataFrame with datetime index/column and target column, or Series with datetime index

- **`predict(steps=None, future=None)`**: Make predictions
  - `steps`: Number of future steps to predict (if `future` not provided)
  - `future`: Optional DataFrame with 'ds' column and regressors for specific dates
  - Returns: DataFrame with columns:
    - `ds`: datestamp
    - `yhat`: predicted value
    - `yhat_lower`: lower bound of prediction interval
    - `yhat_upper`: upper bound of prediction interval
    - `trend`, `seasonal`, etc.: component breakdowns

- **`get_components(forecast)`**: Extract component breakdown from forecast
  - Returns DataFrame with trend, seasonal components, holidays, etc.

### Tips for Best Performance

1. **Seasonality Mode**: Use 'multiplicative' when seasonal patterns scale with trend (e.g., sales growing over time have larger seasonal swings)

2. **Changepoint Prior Scale**: Start with 0.05 and increase if trend looks too smooth, decrease if it looks too wiggly

3. **Seasonality Prior Scale**: Increase if seasonality seems too weak, decrease if it's too strong

4. **Disable Unused Seasonalities**: If your data doesn't have daily/weekly/yearly patterns, disable them to reduce overfitting

5. **Holidays**: Always include holidays if your data has them - they significantly improve accuracy

6. **Data Scaling**: Prophet works better with unscaled data compared to deep learning models

7. **Missing Data**: Prophet handles missing data well, but try to fill gaps if possible

8. **Outliers**: Prophet is robust to outliers, but consider removing extreme outliers if they're not real

9. **Long-term Forecasting**: For very long horizons, consider using logistic growth instead of linear

10. **Component Analysis**: Use `get_components()` to understand what's driving your forecasts

### Comparison with Other Models

Prophet is excellent for:
- **Business forecasting**: Interpretable, handles holidays, provides uncertainty
- **Data with strong seasonality**: Automatically detects and fits seasonal patterns
- **Missing data**: Robust to gaps and outliers
- **Interpretability**: Clear breakdown of trend, seasonality, and holiday effects

Prophet may struggle with:
- **Very long sequences**: May miss complex patterns deep learning models can capture
- **High-frequency data**: Can be slow on minute-level data
- **Multi-variate dependencies**: Doesn't capture relationships between multiple time series

For the ETT dataset, Prophet provides a strong interpretable baseline with automatic seasonality detection, making it a good choice for understanding your data before trying more complex models.

