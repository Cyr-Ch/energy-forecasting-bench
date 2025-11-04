"""PatchTST layer implementations following the official implementation."""

import torch
import torch.nn as nn
import math


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) for time series.
    
    This normalization is reversible and helps with distribution shift issues.
    Based on: https://github.com/ts-kim/RevIN
    
    For channel independence, each channel is normalized separately.
    """
    
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Args:
            num_features: Number of features/channels
            eps: Epsilon for numerical stability
            affine: Whether to use learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()
    
    def forward(self, x, mode: str):
        """
        Args:
            x: Input tensor [B, L] for univariate input (channel-independent)
            mode: 'norm' for normalization or 'denorm' for denormalization
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise ValueError(f"Mode must be 'norm' or 'denorm', got {mode}")
        return x
    
    def _init_params(self):
        # Initialize affine parameters (one per feature, but we use same for all)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    
    def _get_statistics(self, x):
        # For univariate input [B, L], compute statistics over time dimension
        # x: [B, L]
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
    
    def _normalize(self, x):
        # x: [B, L]
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            # Use first affine parameter (since we're processing channels independently)
            x = x * self.affine_weight[0]
            x = x + self.affine_bias[0]
        return x
    
    def _denormalize(self, x):
        # x: [B, L]
        if self.affine:
            x = x - self.affine_bias[0]
            x = x / (self.affine_weight[0] + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer that segments time series into patches.
    
    This is a key component of PatchTST: "A Time Series is Worth 64 Words"
    """
    
    def __init__(self, d_model, patch_len, stride, dropout, padding_patch=True):
        """
        Args:
            d_model: Model dimension
            patch_len: Length of each patch
            stride: Stride for patch creation
            dropout: Dropout rate
            padding_patch: Whether to pad patches
        """
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.padding_patch_layer = nn.ZeroPad2d((0, 0, 0, stride)) if padding_patch else None
        
        # Linear projection for patches
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = nn.Parameter(torch.randn(1, 1000, d_model))  # Max 1000 patches
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, L] where L is sequence length (channel-independent, univariate)
        Returns:
            x: Patched tensor [B, N_patches, d_model]
        """
        # Ensure input is [B, L]
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [1, L]
        assert x.ndim == 2, f"Expected 2D input [B, L], got {x.shape}"
        
        B, L = x.shape
        
        # Padding if needed
        if self.padding_patch_layer is not None:
            # Pad at the end to ensure we can create patches
            x = x.unsqueeze(-1)  # [B, L, 1]
            x = self.padding_patch_layer(x)  # [B, L+stride, 1]
            x = x.squeeze(-1)  # [B, L+stride]
            L = x.shape[1]
        
        # Calculate number of patches
        num_patch = (L - self.patch_len) // self.stride + 1
        total_length = (num_patch - 1) * self.stride + self.patch_len
        
        # Ensure we have enough length
        if L < total_length:
            # Pad if needed
            padding = total_length - L
            x = torch.nn.functional.pad(x, (0, padding))
        elif L > total_length:
            # Crop if needed
            x = x[:, :total_length]
        
        # Create patches using unfold: [B, num_patch, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Project patches to d_model: [B, num_patch, patch_len] -> [B, num_patch, d_model]
        x = self.value_embedding(x)  # [B, num_patch, d_model]
        
        # Add positional embedding
        if x.shape[1] <= self.position_embedding.shape[1]:
            x = x + self.position_embedding[:, :x.shape[1], :]
        else:
            # If more patches than position embeddings, extend by repeating
            n_repeats = (x.shape[1] // self.position_embedding.shape[1]) + 1
            pos_emb = self.position_embedding.repeat(1, n_repeats, 1)
            x = x + pos_emb[:, :x.shape[1], :]
        
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with MLP-based FFN (not Conv1D).
    This matches the official PatchTST implementation.
    """
    
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation='gelu'):
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network (MLP, not Conv1D)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
    
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # FFN with residual
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        
        return x


class TransformerBackbone(nn.Module):
    """Transformer encoder backbone with multiple layers."""
    
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout=0.1, activation='gelu'):
        super(TransformerBackbone, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PredictionHead(nn.Module):
    """
    Prediction head that reconstructs forecasts from patch predictions.
    
    In PatchTST, we predict patches and then reconstruct the full sequence.
    """
    
    def __init__(self, d_model, patch_len, stride, pred_len, dropout=0.1, head_dropout=0.0):
        """
        Args:
            d_model: Model dimension
            patch_len: Length of each patch
            stride: Stride used in patching
            pred_len: Prediction length
            dropout: Dropout rate
            head_dropout: Head dropout rate
        """
        super(PredictionHead, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.pred_len = pred_len
        
        # Project from d_model to patch_len
        self.head = nn.Linear(d_model, patch_len)
        self.head_dropout = nn.Dropout(head_dropout)
        
    def forward(self, x):
        """
        Args:
            x: Transformer output [B, N_patches, d_model]
        Returns:
            pred: Reconstructed forecast [B, pred_len]
        """
        # Project to patch values: [B, N_patches, d_model] -> [B, N_patches, patch_len]
        x = self.head(x)  # [B, N_patches, patch_len]
        x = self.head_dropout(x)
        
        B, N_patches, patch_len = x.shape
        
        # Reconstruct sequence from patches
        # For overlapping patches, we need to handle them carefully
        # We'll use a simple approach: take the last pred_len values
        
        # Calculate the total length we can reconstruct
        total_len = (N_patches - 1) * self.stride + patch_len
        
        # Reconstruct by placing patches with stride
        # Initialize output with zeros
        output = torch.zeros(B, total_len, device=x.device, dtype=x.dtype)
        count = torch.zeros(B, total_len, device=x.device, dtype=x.dtype)
        
        # Place each patch
        for i in range(N_patches):
            start_idx = i * self.stride
            end_idx = start_idx + patch_len
            if end_idx <= total_len:
                output[:, start_idx:end_idx] += x[:, i, :]
                count[:, start_idx:end_idx] += 1
        
        # Average overlapping regions
        count = torch.clamp(count, min=1.0)  # Avoid division by zero
        output = output / count
        
        # Extract the last pred_len values (predictions)
        if total_len >= self.pred_len:
            output = output[:, -self.pred_len:]
        else:
            # Pad if needed (shouldn't happen in practice)
            padding = self.pred_len - total_len
            output = nn.functional.pad(output, (0, padding))
        
        return output
