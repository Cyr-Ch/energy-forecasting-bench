"""PatchTST layer implementations."""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Patch embedding layer."""
    
    def __init__(self, d_in, d_model, patch_len, stride):
        super().__init__()
        self.proj = nn.Conv1d(d_in, d_model, kernel_size=patch_len, stride=stride)
    
    def forward(self, x):  # x: [B, C, L]
        return self.proj(x).transpose(1, 2)  # [B, N_patches, d_model]


class TransformerBackbone(nn.Module):
    """Transformer encoder backbone."""
    
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, x):
        return self.encoder(x)


class Head(nn.Module):
    """Output head for forecasting."""
    
    def __init__(self, d_model, out_len):
        super().__init__()
        self.proj = nn.Linear(d_model, out_len)
    
    def forward(self, x):
        # pool over patches
        x = x.mean(dim=1)
        return self.proj(x)
