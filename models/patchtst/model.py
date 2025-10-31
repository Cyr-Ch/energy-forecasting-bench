"""PatchTST model implementation."""

import torch
import torch.nn as nn
from .layers import PatchEmbedding, TransformerBackbone, Head
from ..registry import register_model


@register_model('patchtst')
class PatchTST(nn.Module):
    """PatchTST: A Time Series is Worth 64 Words."""
    
    def __init__(self, d_in, out_len, d_model=512, n_heads=8, n_layers=3, d_ff=1024, dropout=0.2, patch_len=16, stride=8):
        super().__init__()
        self.embed = PatchEmbedding(d_in, d_model, patch_len, stride)
        self.backbone = TransformerBackbone(d_model, n_heads, n_layers, d_ff, dropout)
        self.head = Head(d_model, out_len)
    
    def forward(self, x):  # x: [B, C, L]
        z = self.embed(x)
        z = self.backbone(z)
        y = self.head(z)
        return y  # [B, out_len]
