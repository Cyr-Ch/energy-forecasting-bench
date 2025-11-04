"""Embedding layers for Autoformer."""

import torch
import torch.nn as nn
import numpy as np


class DataEmbedding_wo_pos(nn.Module):
    """
    Data embedding without position encoding.
    The series-wise connection inherently contains the sequential information.
    Thus, we can discard the position embedding of transformers.
    """
    
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, x_mark=None):
        """
        Args:
            x: [B, L, c_in] input tensor
            x_mark: [B, L, d_mark] optional time features
        Returns:
            x: [B, L, d_model] embedded tensor
        """
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """Token embedding layer."""
    
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular',
            bias=False
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, x):
        """
        Args:
            x: [B, L, c_in]
        Returns:
            x: [B, L, d_model]
        """
        x = x.transpose(1, 2)  # [B, c_in, L]
        x = self.tokenConv(x)  # [B, d_model, L]
        x = x.transpose(1, 2)  # [B, L, d_model]
        return x


class PositionalEmbedding(nn.Module):
    """Positional embedding using fixed sinusoidal encoding."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, L, d_model]
        Returns:
            x: [B, L, d_model]
        """
        return self.pe[:, :x.size(1)]


class TimeFeatureEmbedding(nn.Module):
    """Time feature embedding for temporal information."""
    
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map.get(freq, 4)
        
        self.embed = nn.Linear(d_inp, d_model, bias=False)
    
    def forward(self, x):
        """
        Args:
            x: [B, L, d_inp] time features
        Returns:
            x: [B, L, d_model] embedded time features
        """
        return self.embed(x)

