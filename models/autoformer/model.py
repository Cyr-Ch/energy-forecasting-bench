"""Autoformer model implementation following the official structure."""

import torch
import torch.nn as nn
from .layers.Embed import DataEmbedding_wo_pos
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from ..registry import register_model


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
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
        factor=3,
        moving_avg=25,
        dropout=0.1,
        activation='gelu',
        embed='fixed',
        freq='h',
        label_len=None,
        output_attention=False
    ):
        super(Model, self).__init__()
        
        # Store config
        self.seq_len = None  # Will be set from input
        # label_len should match the dataset's label_len (context_len // 2 typically)
        # If not provided, use a reasonable default based on pred_len
        self.label_len = label_len if label_len is not None else out_len // 2
        self.pred_len = out_len
        self.output_attention = output_attention
        self.d_in = d_in
        self.d_model = d_model
        self.c_out = d_in
        
        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)
        
        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(d_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(d_in, d_model, embed, freq, dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    d_model,
                    self.c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, self.c_out, bias=True)
        )
        
        # Projection for trend_init to match c_out shape
        self.trend_projection = nn.Linear(d_in, self.c_out, bias=True) if d_in != self.c_out else nn.Identity()
    
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
        # Check if last dimension matches d_in (features)
        if x_enc.dim() == 3 and x_enc.shape[-1] != self.d_in and x_enc.shape[1] == self.d_in:
            # Assume [B, C, L] format and transpose to [B, L, C]
            x_enc = x_enc.transpose(1, 2)
        
        B, L_enc, C = x_enc.shape
        self.seq_len = L_enc
        
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)  # [B, pred_len, C]
        zeros = torch.zeros([B, self.pred_len, C], device=x_enc.device)
        # Project mean to c_out shape
        mean = self.trend_projection(mean)  # [B, pred_len, c_out]
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # decoder input
        if x_dec is None:
            # trend_init and seasonal_init are from decomp(x_enc), so they have shape [B, L_enc, C]
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)  # [B, label_len + pred_len, C]
            seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)  # [B, label_len + pred_len, C]
            # Project trend_init to c_out shape to match decoder output
            trend_init = self.trend_projection(trend_init)  # [B, label_len + pred_len, c_out]
        else:
            # Handle x_dec format if provided
            if x_dec.dim() == 3 and x_dec.shape[1] != C:
                x_dec = x_dec.transpose(1, 2)
            seasonal_init, trend_init = self.decomp(x_dec)
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)  # [B, label_len + pred_len, C]
            seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)  # [B, label_len + pred_len, C]
            # Project trend_init to c_out shape to match decoder output
            trend_init = self.trend_projection(trend_init)  # [B, label_len + pred_len, c_out]
        
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        
        # final
        # Ensure both parts have the same shape before adding
        # seasonal_part should be [B, L_dec, c_out] after projection
        # trend_part should be [B, L_dec, c_out] after layer projection
        dec_out = trend_part + seasonal_part  # [B, L_dec, c_out]
        
        # Extract prediction part
        dec_out = dec_out[:, -self.pred_len:, :]  # [B, pred_len, c_out]
        
        # For single-variate output, squeeze to [B, pred_len]
        if self.c_out == 1:
            dec_out = dec_out.squeeze(-1)  # [B, pred_len]
        
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out


# Register with the old name for compatibility
@register_model('autoformer')
class Autoformer(Model):
    """Autoformer model (alias for Model class)."""
    pass
