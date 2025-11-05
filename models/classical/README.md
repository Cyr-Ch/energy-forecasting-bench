# Classical Models for Time Series Forecasting

This directory contains classical machine learning models for time series forecasting, including XGBoost, ARIMA, and Prophet.

## XGBoost Model

XGBoost is a gradient boosting framework that can be used for time series forecasting by converting sequences into tabular features using sliding windows.

### Features

- **Automatic Feature Engineering**: Creates lag features, rolling statistics, and optional time features
- **Multi-step Forecasting**: Supports recursive prediction for multiple horizons
- **Flexible Input Format**: Handles both 2D arrays and 3D sequences (compatible with ETT dataset loaders)
- **Comprehensive Parameters**: Full control over XGBoost hyperparameters

### Training with Command Line

**Basic training**:
```bash
python train.py --model xgboost --dataset etth1
```

**With config file**:
```bash
python train.py --config configs/models/xgboost.yaml --dataset etth1
```

**Custom parameters**:
```bash
python train.py \
    --model xgboost \
    --dataset etth1 \
    --horizon 96 \
    --context_len 336 \
    --exp_name my_xgboost_experiment
```

### Feature Engineering Details

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

### Training with Command Line

**Basic training**:
```bash
python train.py --model arima --dataset etth1 --horizon 96
```

**With config file**:
```bash
python train.py --config configs/models/arima.yaml --dataset etth1
```

**Custom parameters**:
```bash
python train.py \
    --model arima \
    --dataset etth1 \
    --horizon 96 \
    --context_len 336
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

- `auto_select` (default: False): Automatically select best order
- `max_p` (default: 3): Maximum AR order for auto-selection
- `max_d` (default: 2): Maximum differencing order
- `max_q` (default: 3): Maximum MA order
- `max_P` (default: 2): Maximum seasonal AR order
- `max_D` (default: 1): Maximum seasonal differencing order
- `max_Q` (default: 2): Maximum seasonal MA order

#### Other Parameters

- `seasonal_period` (default: 24 for hourly, 96 for 15-min): Seasonal period for SARIMA
  - `24` for hourly data (daily seasonality)
  - `96` for 15-minute data (daily seasonality)
  - `168` for hourly data (weekly seasonality)
- `trend` (default: None): Trend component
  - `'c'`: Constant (intercept)
  - `'t'`: Linear trend
  - `'ct'`: Both constant and linear
  - `None`: No trend

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

### Training with Command Line

**Basic training**:
```bash
python train.py --model prophet --dataset etth1 --horizon 96
```

**With config file**:
```bash
python train.py --config configs/models/prophet.yaml --dataset etth1
```

**Custom parameters**:
```bash
python train.py \
    --model prophet \
    --dataset etth1 \
    --horizon 96 \
    --context_len 336
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
