# PatchTST: A Time Series is Worth 64 Words

## Overview

PatchTST is a Transformer-based architecture designed for long-term time series forecasting that addresses the limitations of standard Transformers by introducing **patching** and **channel independence**. It was introduced in the paper **"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"** by Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam, published at ICLR 2023.

**Paper**: [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)  
**Official Implementation**: [GitHub Repository](https://github.com/yuqinie98/PatchTST)

## Motivation

Traditional Transformer models face several challenges when applied to time series forecasting:

1. **Point-wise attention**: Standard self-attention treats each time point as a token, leading to quadratic complexity and difficulty capturing sub-series patterns.
2. **Channel mixing**: Multivariate models mix channels, making it hard to learn univariate patterns.
3. **Distribution shift**: Time series often have non-stationary distributions, making generalization difficult.

PatchTST addresses these issues by introducing:
- **Patching**: Segments time series into patches (sub-series) that serve as tokens
- **Channel Independence**: Each channel is treated as a separate univariate series sharing weights
- **RevIN**: Reversible Instance Normalization for handling distribution shift

## Key Results

According to the paper, PatchTST achieves:
- **21.0%** reduction in MSE compared to best Transformer-based models
- **16.7%** reduction in MAE compared to best Transformer-based models
- Better performance on long look-back windows
- More efficient computation with fewer parameters

## Architecture

### Overall Structure

PatchTST follows a simple but effective architecture:

```
Input [B, L, C] → Channel Independence → 
    For each channel:
        RevIN (norm) → Patch Embedding → 
        Transformer Encoder → 
        Prediction Head → 
        RevIN (denorm)
    → Stack Channels → Output [B, pred_len, C]
```

### Key Components

#### 1. Patching Mechanism

Instead of treating each time point as a token, PatchTST segments the time series into **patches** (sub-series):

- **Patch Length**: Each patch contains `patch_len` consecutive time points (e.g., 16)
- **Stride**: Patches are created with `stride` overlap (e.g., stride=8 means 50% overlap)
- **Number of Patches**: `N_patches = (L - patch_len) // stride + 1`

**Example**:
```
Time series: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
patch_len=4, stride=2:
  Patch 1: [1, 2, 3, 4]
  Patch 2: [3, 4, 5, 6]
  Patch 3: [5, 6, 7, 8]
  ...
```

**Benefits**:
- **Reduced sequence length**: From L tokens to N_patches tokens (N_patches << L)
- **Sub-series patterns**: Captures local patterns within patches
- **Efficiency**: Reduces computational complexity from O(L²) to O(N_patches²)

#### 2. Channel Independence

PatchTST processes each channel (variable) independently:

- Each channel is treated as a **separate univariate time series**
- All channels **share the same** embedding, transformer, and head weights
- This allows the model to learn general temporal patterns across channels

**Why Channel Independence?**
- **Univariate patterns**: Each variable often has its own temporal patterns
- **Parameter efficiency**: Sharing weights reduces overfitting
- **Better generalization**: Learned patterns transfer across channels

**Mathematical Formulation**:
```
For each channel c:
    x_c = x[:, :, c]  # Extract channel c
    y_c = Model(x_c)  # Process as univariate
    # Model weights are shared across channels
Output = Stack([y_1, y_2, ..., y_C])
```

#### 3. RevIN (Reversible Instance Normalization)

RevIN normalizes each instance (time series) independently and is **reversible**:

- **Normalization**: Normalizes input using instance statistics (mean and std)
- **Processing**: Model processes normalized data
- **Denormalization**: Reverses normalization to get original scale predictions

**Why RevIN?**
- **Distribution shift**: Handles non-stationary time series
- **Reversible**: Preserves information through normalization/denormalization
- **Instance-level**: Each sample normalized independently

**Mathematical Formulation**:
```
Normalization:
    mean = mean(x, dim=time)
    std = std(x, dim=time)
    x_norm = (x - mean) / std

Denormalization:
    x_pred = x_pred_norm * std + mean
```

#### 4. Patch Embedding

Transforms patches into embeddings:

- **Value Embedding**: Linear projection from `patch_len` to `d_model`
- **Positional Embedding**: Learnable positional encodings for patch positions
- **Output**: `[B, N_patches, d_model]` tensor

**Structure**:
```
Patch [B, N_patches, patch_len] 
    → Linear(patch_len → d_model) 
    → + Positional Embedding 
    → [B, N_patches, d_model]
```

#### 5. Transformer Encoder

Standard Transformer encoder with MLP-based feed-forward network:

- **Self-Attention**: Multi-head attention over patches
- **Feed-Forward Network**: MLP (not Conv1D) with GELU activation
- **Layer Norm**: Pre-norm architecture
- **Residual Connections**: Skip connections around attention and FFN

**Layer Structure**:
```
x → LayerNorm → Self-Attention → + → LayerNorm → FFN → + → x
```

#### 6. Prediction Head

Reconstructs forecasts from patch predictions:

- Projects from `d_model` to `patch_len`
- Reconstructs full sequence from overlapping patches
- Averages overlapping regions
- Outputs `pred_len` future values

## How to Use

### Basic Usage

#### Using with ETT Dataset

```python
import torch
from datasets.ettd import Dataset_ETT_hour
from models.registry import get_model
from torch.utils.data import DataLoader

# Load dataset
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

# Initialize model
model = get_model(
    'patchtst',
    d_in=1,           # Number of input channels
    out_len=96,       # Prediction length
    d_model=512,      # Model dimension
    n_heads=8,        # Number of attention heads
    n_layers=3,       # Number of transformer layers
    d_ff=2048,        # Feed-forward dimension
    dropout=0.1,      # Dropout rate
    patch_len=16,     # Patch length
    stride=8,         # Stride for patching
    revin=True,       # Use RevIN
    affine=True       # Use affine parameters in RevIN
)

# Prepare data
data_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(10):
    for batch_idx, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(data_loader):
        # seq_x: [B, seq_len, features]
        # seq_y: [B, label_len + pred_len, features]
        
        # Forward pass
        pred = model(seq_x)  # [B, pred_len] or [B, pred_len, C]
        
        # Extract target
        target = seq_y[:, -96:, 0]  # Last 96 steps (pred_len)
        
        # Compute loss
        loss = criterion(pred, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```

#### Simple Example

```python
import torch
from models.registry import get_model

# Create sample data
batch_size = 32
seq_len = 336
n_features = 1
pred_len = 96

# Random time series data: [B, L, C]
x = torch.randn(batch_size, seq_len, n_features)

# Initialize model
model = get_model(
    'patchtst',
    d_in=n_features,
    out_len=pred_len,
    patch_len=16,
    stride=8,
    d_model=512,
    n_heads=8,
    n_layers=3
)

# Forward pass
pred = model(x)  # [B, pred_len] or [B, pred_len, C]

print(f"Input shape: {x.shape}")
print(f"Output shape: {pred.shape}")
```

### Multivariate Forecasting

```python
import torch
from models.registry import get_model

# Multivariate time series
batch_size = 32
seq_len = 336
n_features = 7  # 7 variables
pred_len = 96

x = torch.randn(batch_size, seq_len, n_features)

# Initialize model (channel independence)
model = get_model(
    'patchtst',
    d_in=n_features,  # Number of channels
    out_len=pred_len,
    patch_len=16,
    stride=8,
    d_model=512,
    n_heads=8,
    n_layers=3,
    revin=True  # Use RevIN for each channel
)

# Forward pass
pred = model(x)  # [B, pred_len, n_features]

print(f"Input shape: {x.shape}")      # [32, 336, 7]
print(f"Output shape: {pred.shape}")  # [32, 96, 7]
```

### Using Configuration File

```python
import yaml
import torch
from models.registry import get_model

# Load configuration
with open('configs/models/patchtst.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize model from config
model = get_model(
    'patchtst',
    d_in=1,
    out_len=96,
    **{k: v for k, v in config.items() if k not in ['model']}
)

# Use model
x = torch.randn(32, 336, 1)
pred = model(x)
```

### Custom Configuration

```python
from models.registry import get_model

# Custom configuration
model = get_model(
    'patchtst',
    d_in=1,
    out_len=96,
    # Patch parameters
    patch_len=16,      # Patch length (16 for PatchTST/64, 42 for PatchTST/42)
    stride=8,         # Stride (50% overlap)
    
    # Transformer parameters
    d_model=512,      # Model dimension
    n_heads=8,        # Number of attention heads
    n_layers=3,       # Number of transformer layers
    d_ff=2048,        # Feed-forward dimension (typically 4 * d_model)
    dropout=0.1,      # Dropout rate
    activation='gelu', # Activation function
    
    # RevIN parameters
    revin=True,       # Use RevIN
    affine=True,      # Use affine parameters in RevIN
    
    # Head parameters
    head_dropout=0.0  # Dropout in prediction head
)
```

## Configuration Parameters

### Patch Parameters

- **`patch_len`** (default: 16): Length of each patch
  - PatchTST/64 uses `patch_len=16` (64 words = 4 patches × 16 length)
  - PatchTST/42 uses `patch_len=42`
  - Smaller values: More patches, more tokens, higher computational cost
  - Larger values: Fewer patches, may miss local patterns

- **`stride`** (default: 8): Stride for patch creation
  - Controls overlap between patches
  - `stride = patch_len / 2` gives 50% overlap
  - Smaller stride: More overlap, more patches
  - Larger stride: Less overlap, fewer patches

### Transformer Parameters

- **`d_model`** (default: 512): Model dimension
  - Hidden dimension of the transformer
  - Larger values: More capacity, more parameters

- **`n_heads`** (default: 8): Number of attention heads
  - Multi-head attention heads
  - Should divide `d_model` evenly

- **`n_layers`** (default: 3): Number of transformer encoder layers
  - Deeper networks: More capacity, more parameters, slower training

- **`d_ff`** (default: 2048): Feed-forward dimension
  - Typically 4 × `d_model`
  - Larger values: More capacity in FFN

- **`dropout`** (default: 0.1): Dropout rate
  - Regularization to prevent overfitting
  - Higher values: More regularization

- **`activation`** (default: 'gelu'): Activation function
  - 'gelu' or 'relu'
  - GELU is standard for transformers

### RevIN Parameters

- **`revin`** (default: True): Whether to use RevIN
  - Recommended: True for better generalization
  - False: Skip normalization (may work for stationary data)

- **`affine`** (default: True): Whether to use affine parameters in RevIN
  - Learnable scaling and bias parameters
  - Recommended: True

### Head Parameters

- **`head_dropout`** (default: 0.0): Dropout in prediction head
  - Additional regularization in output layer
  - Usually set to 0.0

## Architecture Details

### Forward Pass Flow

1. **Input**: `[B, L, C]` where B=batch, L=sequence length, C=channels

2. **Channel Independence Loop** (for each channel c):
   - Extract channel: `x_c = x[:, :, c]` → `[B, L]`
   
3. **RevIN Normalization**:
   - Compute statistics: `mean = mean(x_c)`, `std = std(x_c)`
   - Normalize: `x_norm = (x_c - mean) / std`
   
4. **Patch Embedding**:
   - Create patches: `[B, L]` → `[B, N_patches, patch_len]`
   - Project: `[B, N_patches, patch_len]` → `[B, N_patches, d_model]`
   - Add positional encoding: `[B, N_patches, d_model]`
   
5. **Transformer Encoder**:
   - Process patches: `[B, N_patches, d_model]` → `[B, N_patches, d_model]`
   - Multiple layers of self-attention + FFN
   
6. **Prediction Head**:
   - Project: `[B, N_patches, d_model]` → `[B, N_patches, patch_len]`
   - Reconstruct: `[B, N_patches, patch_len]` → `[B, pred_len]`
   
7. **RevIN Denormalization**:
   - Denormalize: `pred = pred_norm * std + mean`
   
8. **Stack Channels**: Stack all channel predictions → `[B, pred_len, C]`

### Computational Complexity

- **Patching**: Reduces sequence length from L to N_patches ≈ L / stride
- **Attention**: O(N_patches²) instead of O(L²)
- **Total**: O(L × d_model) with patching vs O(L² × d_model) without

### Memory Efficiency

- **Channel Independence**: Processes channels sequentially, reducing memory
- **Patching**: Reduces sequence length, reducing memory for attention
- **Shared Weights**: Same weights across channels, fewer parameters

## Tips for Best Performance

1. **Patch Length**: 
   - Use `patch_len=16` for PatchTST/64 (recommended for most cases)
   - Use `patch_len=42` for PatchTST/42 (for longer sequences)

2. **Stride**: 
   - Use `stride = patch_len / 2` for 50% overlap (recommended)
   - Smaller stride: More patches, better capture of patterns, slower
   - Larger stride: Fewer patches, faster, may miss patterns

3. **RevIN**: 
   - Always use RevIN (`revin=True`) for better generalization
   - Especially important for non-stationary time series

4. **Model Size**:
   - `d_model=512`, `n_heads=8`, `n_layers=3` is a good starting point
   - Increase for more capacity, decrease for faster training

5. **Learning Rate**:
   - Start with `lr=1e-3` with AdamW optimizer
   - Use learning rate scheduling (cosine, OneCycleLR)

6. **Batch Size**:
   - Larger batch sizes (32-128) work well
   - Adjust based on GPU memory

## Comparison with Other Models

### vs. Standard Transformers

- **Efficiency**: Patching reduces sequence length, making it more efficient
- **Patterns**: Captures sub-series patterns better than point-wise attention
- **Parameters**: Fewer parameters due to channel independence

### vs. Autoformer/Informer

- **Simplicity**: Simpler architecture (encoder only, no decoder)
- **Efficiency**: More efficient due to patching and channel independence
- **Performance**: Often achieves better performance with fewer parameters

### vs. Linear Models (DLinear)

- **Non-linear**: Can capture non-linear patterns
- **Complexity**: More parameters but better for complex patterns
- **Efficiency**: Still efficient due to patching

## Limitations

1. **Patch Length**: Fixed patch length may not capture all patterns
2. **Channel Independence**: Assumes channels are independent (may miss cross-channel dependencies)
3. **Overlapping Patches**: Reconstruction may average out important patterns
4. **Long Sequences**: Very long sequences may still be computationally expensive

## Citation

If you use PatchTST in your research, please cite:

```bibtex
@inproceedings{Nie2023PatchTST,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Yuqi Nie and Nam H. Nguyen and Phanwadee Sinthong and Jayant Kalagnanam},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023},
  url={https://arxiv.org/abs/2211.14730}
}
```

## References

1. **Paper**: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)
2. **Official Implementation**: [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)
3. **RevIN**: [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://arxiv.org/abs/2210.07206)

