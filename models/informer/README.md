# Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

## Overview

Informer is a Transformer-based architecture designed for efficient long-sequence time-series forecasting. It was introduced in the paper **"Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"** by Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang, published at AAAI 2021.

**Paper**: [arXiv:2012.07436](https://arxiv.org/abs/2012.07436)  
**Official Implementation**: [GitHub Repository](https://github.com/zhouhaoyi/Informer2020)

## Motivation

Traditional Transformer models face several challenges when applied to long-sequence time-series forecasting:

1. **Quadratic complexity**: Self-attention has O(L²) complexity, making it computationally expensive for long sequences.
2. **Memory bottleneck**: Long input sequences consume excessive memory.
3. **Speed limitations**: Sequential decoding is slow for long-term forecasting.

Informer addresses these issues by introducing:
- **ProbSparse Self-Attention**: Reduces complexity from O(L²) to O(L log L)
- **Self-Attention Distilling**: Reduces input dimensions across encoder layers
- **Generative-Style Decoder**: Predicts long sequences in a single forward pass

## Architecture

### Overall Structure

Informer follows an encoder-decoder architecture similar to Transformers, but with key efficiency improvements:

```
Input → Encoder (with Distillation) → Decoder → Output
   ↓         ↓              ↓
Embed → ProbSparse → Full Attention
```

### Key Components

#### 1. ProbSparse Self-Attention

The ProbSparse self-attention mechanism reduces computational complexity by selecting only the most important queries:

**Advantages**:
- **Complexity**: O(L log L) instead of O(L²)
- **Efficiency**: Reduces memory usage significantly
- **Accuracy**: Maintains performance while being more efficient

#### 2. Self-Attention Distilling

The distilling mechanism reduces input dimensions across encoder layers:

```
Layer 1: [B, L, D] → Conv → MaxPool → [B, L/2, D]
Layer 2: [B, L/2, D] → Conv → MaxPool → [B, L/4, D]
...
```

**Benefits**:
- **Memory efficiency**: Progressively reduces sequence length
- **Feature extraction**: Highlights dominant attention patterns
- **Scalability**: Handles extremely long sequences efficiently

#### 3. Generative-Style Decoder

The decoder uses a generative approach to predict long sequences:

- **Start tokens**: Uses last few values from input sequence
- **Target tokens**: Zero-padded future positions
- **Single forward pass**: Predicts entire forecast horizon at once

**Structure**:
```
Decoder Input: [last_values, zeros]
Decoder: Self-Attention (ProbSparse) + Cross-Attention (Full) → Output
```

#### 4. Encoder Architecture

The encoder consists of multiple encoder layers with distillation:

```
Encoder Layer:
  Input → ProbSparse Attention → Feed-Forward → Output
           ↓
      Distillation (if enabled)
```

Each encoder layer:
1. Applies ProbSparse self-attention
2. Applies feed-forward network
3. Optionally applies distillation (reduces sequence length)

#### 5. Decoder Architecture

The decoder uses:
- **Self-Attention**: ProbSparse attention for decoder input
- **Cross-Attention**: Full attention between decoder and encoder
- **Feed-Forward**: Standard feed-forward network

```
Decoder Layer:
  Input → Self ProbSparse Attention → 
         → Cross Full Attention → Feed-Forward → Output
```

### Forward Pass Flow

1. **Input Embedding**: Projects input to model dimension with positional encoding
2. **Encoder Processing**: 
   - Progressively processes input through encoder layers
   - Applies distillation to reduce sequence length
   - Extracts features using ProbSparse attention
3. **Decoder Initialization**:
   - Uses last values from input + zeros for future horizon
   - Creates decoder input sequence
4. **Decoder Processing**:
   - Self-attention on decoder input (ProbSparse)
   - Cross-attention with encoder output (Full)
   - Progressively generates forecast
5. **Output Projection**: Projects decoder output to forecast horizon

## Key Innovations

### 1. ProbSparse Self-Attention

The ProbSparse mechanism selects queries based on sparsity measurement:
- **Sparsity measurement**: Identifies queries with high variance in attention scores
- **Top-k selection**: Selects only the most important queries
- **Efficient computation**: Reduces complexity while maintaining accuracy

### 2. Self-Attention Distilling

The distilling mechanism:
- **Progressive reduction**: Halves sequence length at each encoder layer
- **Feature extraction**: Highlights dominant patterns
- **Memory efficiency**: Reduces memory usage significantly

### 3. Generative-Style Decoder

The decoder design:
- **Single forward pass**: Predicts entire forecast horizon at once
- **Start token approach**: Uses last values as decoder input
- **Zero padding**: Uses zeros for future positions

## Implementation Details

### Model Parameters

- `d_in`: Input dimension (number of features/variables)
- `out_len`: Forecast horizon length
- `d_model`: Model dimension (embedding size)
- `n_heads`: Number of attention heads
- `e_layers`: Number of encoder layers
- `d_layers`: Number of decoder layers
- `d_ff`: Feed-forward dimension
- `factor`: Factor for ProbSparse attention (typically 5)
- `dropout`: Dropout rate
- `activation`: Activation function ('gelu' or 'relu')
- `embed`: Embedding type ('fixed' or 'learned')
- `freq`: Frequency string ('h' for hourly, 't' for 15-minute)
- `distil`: Whether to use distillation in encoder
- `output_attention`: Whether to return attention weights

## How to Use

### Training with Command Line

**Basic training**:
```bash
python train.py --model informer --dataset etth1 --epochs 10 --batch_size 32
```

**With config file**:
```bash
python train.py --config configs/models/informer.yaml --dataset etth1
```

**Custom parameters**:
```bash
python train.py \
    --model informer \
    --dataset etth1 \
    --context_len 336 \
    --horizon 96 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --d_ff 2048 \
    --epochs 10
```

## Performance

Informer demonstrated state-of-the-art performance on the ETT (Electricity Transformer Temperature) datasets:

- **ETTh1, ETTh2** — hourly transformer temperature data
- **ETTm1, ETTm2** — 15-minute transformer temperature data

**Key Results**:
- Achieved significant improvements in efficiency (O(L log L) vs O(L²))
- Reduced memory usage by up to 50% compared to standard Transformers
- Maintained or improved accuracy while being more efficient
- Especially effective for long-term forecasting (horizons > 96 steps)

## Comparison with Standard Transformers

| Aspect | Standard Transformer | Informer |
|--------|---------------------|----------|
| Attention | Full self-attention | ProbSparse self-attention |
| Complexity | O(L²) | O(L log L) |
| Memory | High | Reduced (distillation) |
| Decoding | Sequential | Generative (single pass) |
| Efficiency | Quadratic scaling | Log-linear scaling |
| Long sequences | Limited | Efficient handling |

## Mathematical Formulation

### ProbSparse Self-Attention

The sparsity measurement for query q_i is:

```
M(q_i, K) = max_j(q_i k_j^T / √d) - (1/L) Σ_j(q_i k_j^T / √d)
```

Top-k queries are selected:

```
TopK = argmax_{i∈[1,L]} M(q_i, K)
```

Attention is computed only for selected queries:

```
Attention(Q, K, V) = Softmax(Q_top K^T / √d) V
```

### Self-Attention Distilling

At each encoder layer, the sequence is halved:

```
X_{l+1} = MaxPool(Conv1d(X_l))
```

Where Conv1d reduces dimensions and MaxPool halves the length.

## References

1. **Original Paper**:
   - Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(12), 11106-11115.
   - [arXiv:2012.07436](https://arxiv.org/abs/2012.07436)

2. **Official Implementation**:
   - [GitHub: zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020)

3. **Related Works**:
   - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
   - LogTrans: Log-Transformer for Long-Term Time Series Forecasting
   - Transformer: Attention Is All You Need

## Implementation Notes

This implementation includes:
- ✅ ProbSparse self-attention mechanism
- ✅ Self-attention distilling for encoder efficiency
- ✅ Generative-style decoder
- ✅ Full attention for cross-attention
- ✅ Positional encoding (unlike Autoformer)
- ✅ Complete forward pass with proper input/output handling

The implementation follows the paper's architecture while adapting to the project's model interface conventions.
