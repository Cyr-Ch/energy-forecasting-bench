# Autoformer: Decomposition Transformers with Auto-Correlation

## Overview

Autoformer is a novel Transformer-based architecture designed specifically for long-term time series forecasting. It was introduced in the paper **"Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"** by Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long, published at NeurIPS 2021.

**Paper**: [arXiv:2106.13008](https://arxiv.org/abs/2106.13008)  
**Official Implementation**: [GitHub Repository](https://github.com/thuml/Autoformer)

## Motivation

Traditional Transformer models face several challenges when applied to time series forecasting:

1. **Self-attention limitations**: Standard self-attention mechanisms capture point-wise dependencies but struggle with series-level periodic patterns that are crucial for time series data.
2. **Computational complexity**: Self-attention has quadratic complexity O(L²), making it inefficient for long sequences.
3. **Trend-seasonal mixing**: Directly modeling raw time series often mixes trend and seasonal components, making it difficult to capture long-term dependencies.

Autoformer addresses these issues by introducing:
- **Series Decomposition**: Progressively separating trend and seasonal components
- **Auto-Correlation Mechanism**: Discovering dependencies based on series periodicity using FFT-based correlation
- **Efficient architecture**: Achieving O(L log L) complexity for series of length L

## Architecture

### Overall Structure

Autoformer follows an encoder-decoder architecture similar to Transformers, but with key modifications:

```
Input → Encoder → Decoder → Output
   ↓         ↓         ↓
Decomposition → Auto-Correlation → Decomposition
```

### Key Components

#### 1. Series Decomposition Block

The Series Decomposition block separates a time series into **trend** and **seasonal** components:

- **Trend extraction**: Uses moving average (avg pooling) to extract the slow-changing trend component
- **Seasonal extraction**: Computed as the residual (original - trend)

This decomposition is embedded throughout the model architecture, allowing progressive refinement of trend and seasonal components at each layer.

**Mathematical Formulation**:
- Trend: `trend = MovingAvg(x)`
- Seasonal: `seasonal = x - trend`

#### 2. Auto-Correlation Mechanism

The Auto-Correlation mechanism replaces standard self-attention and is inspired by stochastic process theory. It discovers dependencies based on **series periodicity** rather than point-wise relationships.

**Key Steps**:

1. **FFT-based Correlation**: Uses Fast Fourier Transform (FFT) to efficiently compute auto-correlations between query and key sequences:
   ```
   R(τ) = FFT(Q) * conj(FFT(K))
   ```

2. **Period Detection**: Identifies top-k periods (lags) with highest correlation strength:
   ```
   TopK = argmax_{τ∈[1,L]} |R(τ)|
   ```

3. **Time Delay Aggregation**: Aggregates values by shifting the value sequence by the detected periods:
   ```
   Output = Σ_{τ∈TopK} R(τ) * Roll(V, τ)
   ```

**Advantages**:
- **Series-level dependency**: Captures periodic patterns at sub-series level
- **Computational efficiency**: O(L log L) complexity using FFT
- **Periodicity awareness**: Automatically discovers and exploits periodic patterns

#### 3. Encoder Architecture

The encoder consists of multiple encoder layers, each containing:

```
Encoder Layer:
  Input → Series Decomposition → Auto-Correlation → Output
            ↓           ↓
        [trend, seasonal] → [seasonal_processed] → trend + seasonal_processed
```

Each encoder layer:
1. Decomposes input into trend and seasonal components
2. Applies Auto-Correlation to the seasonal component
3. Combines processed seasonal with trend

#### 4. Decoder Architecture

The decoder uses a similar structure but includes:
- **Self Auto-Correlation**: Processes decoder input with auto-correlation
- **Cross Attention**: Standard cross-attention between decoder and encoder outputs
- **Progressive Decomposition**: Multiple decomposition steps to refine forecasts

```
Decoder Layer:
  Input → Decomposition 1 → Self Auto-Correlation → 
         → Decomposition 2 → Cross-Attention → Feed-Forward → Output
```

### Forward Pass Flow

1. **Input Projection**: Projects input `[B, C, L]` to model dimension `[B, L, D]`
2. **Encoder Processing**: 
   - Progressively decomposes and processes input through encoder layers
   - Extracts features while separating trend and seasonal components
3. **Decoder Initialization**:
   - Uses last few input values + mean padding for forecast horizon
   - Creates decoder input sequence
4. **Decoder Processing**:
   - Processes decoder input with self auto-correlation
   - Uses encoder output for cross-attention
   - Progressively refines forecast
5. **Output Projection**: Projects decoder output to forecast horizon

## Key Innovations

### 1. Progressive Decomposition

Unlike traditional decomposition that happens once at preprocessing, Autoformer embeds decomposition throughout the architecture. This allows:
- Iterative refinement of trend and seasonal components
- Better handling of complex temporal patterns
- More stable training and forecasting

### 2. Series-Level Auto-Correlation

The auto-correlation mechanism operates at the series level, not point-wise:
- **Discovers periodic patterns**: Automatically identifies recurring sub-series
- **Aggregates similar sub-series**: Groups information from similar periods
- **Efficient computation**: Uses FFT for O(L log L) complexity

### 3. Architecture Design

The encoder-decoder structure with decomposition:
- **Encoder**: Extracts and refines features from input sequence
- **Decoder**: Generates forecasts by combining trend and seasonal predictions
- **Decomposition**: Maintains separation of trend and seasonal throughout

## Implementation Details

### Model Parameters

- `d_in`: Input dimension (number of features/variables)
- `out_len`: Forecast horizon length
- `d_model`: Model dimension (embedding size)
- `n_heads`: Number of attention heads
- `e_layers`: Number of encoder layers
- `d_layers`: Number of decoder layers
- `d_ff`: Feed-forward dimension
- `factor`: Factor for top-k period selection (factor * log(L))
- `kernel_size`: Moving average kernel size for decomposition
- `dropout`: Dropout rate

### Usage Example

```python
from models.registry import get_model

# Initialize Autoformer
model = get_model(
    'autoformer',
    d_in=1,           # Single variable
    out_len=96,       # Forecast 96 steps ahead
    d_model=512,      # Model dimension
    n_heads=8,        # 8 attention heads
    e_layers=2,       # 2 encoder layers
    d_layers=1,       # 1 decoder layer
    d_ff=2048,        # Feed-forward dimension
    factor=3,         # Top-k factor
    kernel_size=25,   # Moving average window
    dropout=0.1      # Dropout rate
)

# Forward pass
# Input: [batch_size, channels, sequence_length]
# Output: [batch_size, out_len]
output = model(input_tensor)
```

## Performance

Autoformer demonstrated state-of-the-art performance on the ETT (Electricity Transformer Temperature) datasets:

- **ETTh1, ETTh2** — hourly transformer temperature data
- **ETTm1, ETTm2** — 15-minute transformer temperature data

**Key Results**:
- Achieved **38% relative improvement** over previous models on ETT datasets
- Outperformed Informer, LogTrans, and other Transformer variants
- Especially effective for long-term forecasting (horizons > 96 steps)

## Comparison with Standard Transformers

| Aspect | Standard Transformer | Autoformer |
|--------|---------------------|------------|
| Attention | Point-wise self-attention | Series-level auto-correlation |
| Complexity | O(L²) | O(L log L) |
| Decomposition | None (preprocessing only) | Progressive (embedded in architecture) |
| Dependency | Point-wise dependencies | Periodic sub-series dependencies |
| Efficiency | Quadratic scaling | Log-linear scaling |

## Mathematical Formulation

### Auto-Correlation

For a series of length L, the auto-correlation at lag τ is:

```
R(τ) = FFT(Q) * conj(FFT(K))
```

Where Q and K are query and key sequences. The mechanism selects top-k lags with highest correlation:

```
τ_top = argmax_{τ∈[1,L]} |R(τ)|
```

### Time Delay Aggregation

The output is computed by aggregating values shifted by the top-k periods:

```
Output = Σ_{τ∈τ_top} R(τ) * Roll(V, τ)
```

Where `Roll(V, τ)` circularly shifts the value sequence V by τ positions.

## References

1. **Original Paper**:
   - Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. *Advances in Neural Information Processing Systems*, 34, 22419-22430.
   - [arXiv:2106.13008](https://arxiv.org/abs/2106.13008)

2. **Official Implementation**:
   - [GitHub: thuml/Autoformer](https://github.com/thuml/Autoformer)

3. **Related Works**:
   - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
   - LogTrans: Log-Transformer for Long-Term Time Series Forecasting
   - Transformer: Attention Is All You Need

## Implementation Notes

This implementation includes:
- ✅ Series Decomposition block with moving average
- ✅ Auto-Correlation mechanism using FFT
- ✅ Encoder-decoder architecture with decomposition
- ✅ Cross-attention in decoder
- ✅ Complete forward pass with proper input/output handling

The implementation follows the paper's architecture while adapting to the project's model interface conventions.

## Usage Examples

### Basic Usage with ETT Dataset

```python
import torch
from torch.utils.data import DataLoader
from datasets.ettd import Dataset_ETT_hour
from models.autoformer import Model

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

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Initialize Autoformer model
model = Model(
    d_in=1,           # Single variable (univariate)
    out_len=96,       # Forecast horizon
    d_model=512,      # Model dimension
    n_heads=8,        # Number of attention heads
    e_layers=2,       # Encoder layers
    d_layers=1,       # Decoder layers
    d_ff=2048,        # Feed-forward dimension
    factor=3,         # Top-k factor for auto-correlation
    moving_avg=25,    # Moving average window for decomposition
    dropout=0.1,      # Dropout rate
    activation='gelu', # Activation function
    embed='fixed',    # Embedding type
    freq='h',         # Frequency ('h' for hourly)
    label_len=48,     # Decoder label length
    output_attention=False
)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

model.train()
for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        # Move to device
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        
        # Forward pass
        # Autoformer expects [B, L, C] format
        batch_x = batch_x.unsqueeze(-1)  # [B, L, 1] if univariate
        pred = model(batch_x, batch_x_mark)
        
        # Extract target (last pred_len steps)
        target = batch_y[:, -96:, 0]  # [B, pred_len]
        
        # Compute loss
        loss = criterion(pred, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    total_loss = 0
    for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        
        batch_x = batch_x.unsqueeze(-1)
        pred = model(batch_x, batch_x_mark)
        target = batch_y[:, -96:, 0]
        
        loss = criterion(pred, target)
        total_loss += loss.item()
    
    print(f"Validation Loss: {total_loss/len(val_loader):.4f}")
```

### Using Model Registry

```python
from models.registry import get_model

# Get Autoformer model via registry
model = get_model(
    'autoformer',
    d_in=1,
    out_len=96,
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    factor=3,
    moving_avg=25,
    dropout=0.1
)
```

### Multivariate Forecasting

```python
# For multivariate forecasting (features='M' or 'MS')
train_data = Dataset_ETT_hour(
    root_path='data/raw/etth',
    flag='train',
    size=[96, 48, 96],
    features='M',  # Multivariate - all features
    data_path='ETTh1.csv',
    target='OT',
    scale=True,
    timeenc=0,
    freq='h'
)

# Initialize model with appropriate d_in
model = Model(
    d_in=7,  # Number of features in ETTh dataset
    out_len=96,
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1
)
```

### Using Time Features

```python
# With time features enabled
train_data = Dataset_ETT_hour(
    root_path='data/raw/etth',
    flag='train',
    size=[96, 48, 96],
    features='S',
    data_path='ETTh1.csv',
    target='OT',
    scale=True,
    timeenc=1,  # Use time_features function
    freq='h'
)

# Time features will be automatically extracted and used in embedding
```

### Prediction Example

```python
# Single prediction
model.eval()
with torch.no_grad():
    # Get a sample
    seq_x, seq_y, seq_x_mark, seq_y_mark = train_data[0]
    
    # Prepare input
    x = seq_x.unsqueeze(0).unsqueeze(-1).float().to(device)  # [1, seq_len, 1]
    x_mark = seq_x_mark.unsqueeze(0).float().to(device)     # [1, seq_len, time_features]
    
    # Make prediction
    prediction = model(x, x_mark)  # [1, pred_len]
    
    # Inverse transform if data was scaled
    prediction_original = train_data.inverse_transform(prediction.cpu().numpy())
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction (original scale): {prediction_original}")
```

