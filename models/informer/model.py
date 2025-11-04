"""Informer model implementation following the official structure."""

import torch
import torch.nn as nn
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .layers.SelfAttention_Family import ProbAttention, AttentionLayer, FullAttention
from .layers.Embed import DataEmbedding
from ..registry import register_model


class Model(nn.Module):
    """
    Informer with ProbSparse attention in O(LlogL) complexity
    """
    def __init__(
        self,
        d_in,
        out_len,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        factor=5,
        dropout=0.1,
        activation='gelu',
        embed='fixed',
        freq='h',
        distil=True,
        output_attention=False
    ):
        super(Model, self).__init__()
        
        # Store config
        self.pred_len = out_len
        self.output_attention = output_attention
        self.d_in = d_in
        self.d_model = d_model
        self.c_out = d_in
        
        # Embedding
        self.enc_embedding = DataEmbedding(d_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(d_in, d_model, embed, freq, dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout,
                                     output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, self.c_out, bias=True)
        )
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Args:
            x_enc: [B, L_enc, C] encoder input (can also be [B, C, L] which will be transposed)
            x_mark_enc: [B, L_enc, d_mark] optional encoder time features
            x_dec: [B, L_dec, C] optional decoder input
            x_mark_dec: [B, L_dec, d_mark] optional decoder time features
            enc_self_mask: encoder attention mask
            dec_self_mask: decoder attention mask
            dec_enc_mask: cross-attention mask
        Returns:
            dec_out: [B, pred_len, C] or [B, pred_len] if C=1
            attns: attention weights if output_attention=True
        """
        # Handle input format: support both [B, C, L] and [B, L, C]
        if x_enc.dim() == 3 and x_enc.shape[1] != self.d_in:
            # Assume [B, C, L] format and transpose
            x_enc = x_enc.transpose(1, 2)  # [B, L, C]
        
        B, L_enc, C = x_enc.shape
        
        # Decoder input
        if x_dec is None:
            # Create decoder input: last values + zeros for future
            # Use last pred_len values as decoder input
            x_dec = torch.cat([x_enc[:, -self.pred_len:, :], 
                              torch.zeros([B, self.pred_len, C], device=x_enc.device)], dim=1)
        else:
            # Handle x_dec format if provided
            if x_dec.dim() == 3 and x_dec.shape[1] != C:
                x_dec = x_dec.transpose(1, 2)
            # Ensure decoder input has correct length
            if x_dec.shape[1] < 2 * self.pred_len:
                # Pad with zeros if needed
                pad_len = 2 * self.pred_len - x_dec.shape[1]
                x_dec = torch.cat([x_dec, torch.zeros([B, pad_len, C], device=x_dec.device)], dim=1)
        
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        
        # Extract prediction part
        dec_out = dec_out[:, -self.pred_len:, :]  # [B, pred_len, C]
        
        # For single-variate output, squeeze to [B, pred_len]
        if self.c_out == 1:
            dec_out = dec_out.squeeze(-1)
        
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out


# Register with the old name for compatibility
@register_model('informer')
class Informer(Model):
    """Informer model (alias for Model class)."""
    pass
