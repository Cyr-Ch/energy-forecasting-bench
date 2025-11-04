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
        self.seq_len = L_enc
        
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([B, self.pred_len, C], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # decoder input
        if x_dec is None:
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
            seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        else:
            # Handle x_dec format if provided
            if x_dec.dim() == 3 and x_dec.shape[1] != C:
                x_dec = x_dec.transpose(1, 2)
            seasonal_init, trend_init = self.decomp(x_dec)
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
            seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        
        # final
        dec_out = trend_part + seasonal_part
        
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
@register_model('autoformer')
class Autoformer(Model):
    """Autoformer model (alias for Model class)."""
    pass
