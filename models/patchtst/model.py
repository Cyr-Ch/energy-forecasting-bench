"""PatchTST model implementation following the official implementation.

Based on: https://github.com/yuqinie98/PatchTST
Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)
"""

import torch
import torch.nn as nn
from .layers import PatchEmbedding, TransformerBackbone, PredictionHead, RevIN
from ..registry import register_model


@register_model('patchtst')
class PatchTST(nn.Module):
    """
    PatchTST: A Time Series is Worth 64 Words.
    
    Key features:
    - Channel Independence: Each channel is treated as a separate univariate series
    - Patching: Time series is segmented into patches
    - RevIN: Reversible Instance Normalization for distribution shift
    """
    
    def __init__(
        self, 
        d_in, 
        out_len, 
        d_model=512, 
        n_heads=8, 
        n_layers=3, 
        d_ff=2048, 
        dropout=0.1, 
        patch_len=16, 
        stride=8,
        revin=True,
        affine=True,
        activation='gelu',
        head_dropout=0.0
    ):
        """
        Args:
            d_in: Number of input channels/features
            out_len: Prediction length (forecast horizon)
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension (typically 4 * d_model)
            dropout: Dropout rate
            patch_len: Length of each patch
            stride: Stride for patch creation
            revin: Whether to use RevIN (Reversible Instance Normalization)
            affine: Whether to use affine parameters in RevIN
            activation: Activation function ('gelu' or 'relu')
            head_dropout: Dropout rate in prediction head
        """
        super().__init__()
        
        # Store config
        self.d_in = d_in
        self.out_len = out_len
        self.patch_len = patch_len
        self.stride = stride
        self.revin = revin
        
        # RevIN for normalization (channel-independent)
        if revin:
            self.revin_layer = RevIN(num_features=d_in, affine=affine)
        else:
            self.revin_layer = None
        
        # Patch embedding (channel-independent: each channel processed separately)
        # For channel independence, we process each channel as a separate univariate series
        # All channels share the same embedding and transformer weights
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)
        
        # Transformer backbone (shared across channels)
        self.backbone = TransformerBackbone(d_model, n_heads, n_layers, d_ff, dropout, activation)
        
        # Prediction head (predicts patches and reconstructs)
        # For channel independence, we need separate heads or process separately
        # In practice, we can share the head across channels
        self.head = PredictionHead(d_model, patch_len, stride, out_len, dropout, head_dropout)
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Forward pass with channel independence.
        
        Args:
            x_enc: Encoder input [B, L, C] or [B, C, L] (will auto-detect format)
            x_mark_enc: Optional encoder time features [B, L, d_mark] (not used in PatchTST)
            x_dec: Optional decoder input (not used in PatchTST)
            x_mark_dec: Optional decoder time features (not used in PatchTST)
            enc_self_mask: Optional encoder mask (not used)
            dec_self_mask: Optional decoder mask (not used)
            dec_enc_mask: Optional cross-attention mask (not used)
        
        Returns:
            y: Predictions [B, out_len, C] or [B, out_len] if C=1
        """
        # Handle input format: support both [B, C, L] and [B, L, C]
        if x_enc.dim() == 3:
            if x_enc.shape[1] == self.d_in and x_enc.shape[2] > x_enc.shape[1]:
                # Format is [B, C, L], transpose to [B, L, C]
                x_enc = x_enc.transpose(1, 2)
        
        B, L, C = x_enc.shape
        
        # Channel Independence: Process each channel separately
        # All channels share the same embedding, transformer, and head weights
        outputs = []
        
        for c in range(C):
            # Extract single channel: [B, L]
            x_c = x_enc[:, :, c]
            
            # RevIN normalization
            if self.revin_layer is not None:
                # RevIN expects [B, L] for univariate input
                # We need to reshape for RevIN
                x_c_normalized = self.revin_layer(x_c.unsqueeze(-1), mode='norm')
                x_c_normalized = x_c_normalized.squeeze(-1)
            else:
                x_c_normalized = x_c
            
            # Patch embedding: [B, L] -> [B, N_patches, d_model]
            x_patched = self.patch_embedding(x_c_normalized)
            
            # Transformer backbone: [B, N_patches, d_model] -> [B, N_patches, d_model]
            x_transformed = self.backbone(x_patched)
            
            # Prediction head: [B, N_patches, d_model] -> [B, out_len]
            x_pred = self.head(x_transformed)
            
            # RevIN denormalization
            if self.revin_layer is not None:
                # We need to reconstruct the full sequence for denorm
                # For simplicity, we'll use the prediction directly
                # In practice, you might want to pad with historical data
                x_pred_padded = torch.cat([x_c_normalized, x_pred], dim=-1)
                x_pred_padded = self.revin_layer(x_pred_padded.unsqueeze(-1), mode='denorm')
                x_pred = x_pred_padded.squeeze(-1)[:, -self.out_len:]
            else:
                x_pred = x_pred
            
            outputs.append(x_pred)
        
        # Concatenate channels: [B, out_len, C]
        y = torch.stack(outputs, dim=-1)
        
        # For single-variate output, squeeze to [B, out_len]
        if C == 1:
            y = y.squeeze(-1)
        
        return y
