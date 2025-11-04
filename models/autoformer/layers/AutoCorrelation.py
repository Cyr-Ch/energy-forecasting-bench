"""Auto-Correlation mechanism for Autoformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AutoCorrelation(nn.Module):
    """Auto-Correlation mechanism using FFT-based period detection."""
    
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    
    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * np.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg
    
    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * np.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg
    
    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * np.log(length))
        weights = torch.topk(torch.mean(corr, dim=1), top_k, dim=-1)[0]
        delay = torch.topk(torch.mean(corr, dim=1), top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, :, i].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, :, i].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, length))
        return delays_agg
    
    def forward(self, queries, keys, values, attn_mask):
        """
        Args:
            queries: [B, H, L, D_head]
            keys: [B, H, L, D_head]
            values: [B, H, L, D_head]
            attn_mask: attention mask
        Returns:
            out: [B, H, L, D_head]
            attn: attention weights if output_attention=True
        """
        B, H, L, D = queries.shape
        
        # Compute auto-correlation using FFT on time dimension
        # q_fft and k_fft should be computed on the time dimension (L)
        # queries: [B, H, L, D] -> [B, H, D, L] for FFT
        q_fft = torch.fft.rfft(queries.permute(0, 1, 3, 2).contiguous(), dim=-1)  # [B, H, D, L//2+1]
        k_fft = torch.fft.rfft(keys.permute(0, 1, 3, 2).contiguous(), dim=-1)  # [B, H, D, L//2+1]
        res = q_fft * torch.conj(k_fft)  # [B, H, D, L//2+1]
        corr = torch.fft.irfft(res, n=L, dim=-1)  # [B, H, D, L]
        corr = corr.permute(0, 1, 3, 2).contiguous()  # [B, H, L, D]
        
        # Time delay aggregation
        # values: [B, H, L, D] -> [B, H, D, L] for aggregation
        # corr: [B, H, L, D] -> [B, H, D, L] for aggregation
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 1, 3, 2).contiguous(), corr.permute(0, 1, 3, 2).contiguous())
            V = V.permute(0, 1, 3, 2)  # [B, H, D, L] -> [B, H, L, D]
        else:
            V = self.time_delay_agg_inference(values.permute(0, 1, 3, 2).contiguous(), corr.permute(0, 1, 3, 2).contiguous())
            V = V.permute(0, 1, 3, 2)  # [B, H, D, L] -> [B, H, L, D]
        
        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 2, 3, 1))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    """Auto-Correlation layer wrapper."""
    
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    
    def forward(self, queries, keys, values, attn_mask):
        """
        Args:
            queries: [B, L, D]
            keys: [B, S, D]
            values: [B, S, D]
            attn_mask: attention mask
        Returns:
            out: [B, L, D]
            attn: attention weights
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        queries = queries.transpose(1, 2)  # [B, H, L, d_keys]
        keys = keys.transpose(1, 2)  # [B, H, S, d_keys]
        values = values.transpose(1, 2)  # [B, H, S, d_values]
        
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        
        out = out.transpose(1, 2).contiguous().view(B, L, -1)  # [B, L, H*d_values]
        out = self.out_projection(out)
        
        return out, attn

