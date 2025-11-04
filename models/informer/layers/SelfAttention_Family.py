"""Self-attention family for Informer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ProbAttention(nn.Module):
    """ProbSparse Attention mechanism for Informer."""
    
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    
    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        Compute ProbSparse attention scores.
        
        Args:
            Q: [B, H, L, D]
            K: [B, H, L, D]
            sample_k: number of samples for K
            n_top: number of top queries to select
        
        Returns:
            Q_K: [B, H, n_top, L]
            M_top: [B, H, n_top] indices of top queries
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        
        # Calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=K.device)
        K_sample = K_expand[:, :, torch.arange(L_Q, device=K.device).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)  # [B, H, L_Q, sample_k]
        
        # Find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # [B, H, L_Q]
        M_top = M.topk(n_top, sorted=False)[1]  # [B, H, n_top]
        
        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    M_top, :]  # factor*ln(L_q)*ln(L_k), D
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*ln(L_k), ln(L_k)
        
        return Q_K, M_top
    
    def _get_initial_context(self, V, L_Q):
        """Get initial context for ProbSparse attention."""
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V for mask
            contex = V.cumsum(dim=-2)
        return contex
    
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """Update context with selected queries."""
        B, H, L_V, D = V.shape
        
        if self.mask_flag:
            prob_mask = ProbMask(B, H, L_Q, L_V, index, scores, device=V.device)
            scores.masked_fill_(prob_mask.mask, -np.inf)
        
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        
        context_in[torch.arange(B)[:, None, None],
                  torch.arange(H)[None, :, None],
                  index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)
    
    def forward(self, queries, keys, values, attn_mask):
        """
        Args:
            queries: [B, H, L, D]
            keys: [B, H, L, D]
            values: [B, H, L, D]
            attn_mask: attention mask
        Returns:
            out: [B, H, L, D]
            attn: attention weights if output_attention=True
        """
        B, H, L_Q, E = queries.shape
        _, _, L_K, _ = keys.shape
        
        scale = self.scale or 1. / math.sqrt(E)
        
        U_part = self.factor * math.ceil(math.log(L_K))  # c*ln(L_k)
        u = self.factor * math.ceil(math.log(L_Q))  # c*ln(L_q)
        
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        
        # add scale factor
        scores_top = scores_top * scale
        
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.contiguous(), attn


class FullAttention(nn.Module):
    """Full attention mechanism (standard self-attention)."""
    
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, queries, keys, values, attn_mask):
        """
        Args:
            queries: [B, H, L, D]
            keys: [B, H, L, D]
            values: [B, H, L, D]
            attn_mask: attention mask
        Returns:
            out: [B, H, L, D]
            attn: attention weights if output_attention=True
        """
        B, H, L_Q, E = queries.shape
        _, _, L_K, _ = keys.shape
        
        scale = self.scale or 1. / math.sqrt(E)
        
        scores = torch.einsum("bhqd,bhkd->bhqk", queries, keys)
        
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L_Q, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        attn = torch.softmax(scores * scale, dim=-1)
        attn = self.dropout(attn)
        context = torch.einsum("bhqk,bhvd->bhqd", attn, values)
        
        if self.output_attention:
            return (context.contiguous(), attn)
        else:
            return (context.contiguous(), None)


class AttentionLayer(nn.Module):
    """Attention layer wrapper."""
    
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
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
        
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        
        out = out.transpose(1, 2).contiguous().view(B, L, -1)  # [B, L, H*d_values]
        out = self.out_projection(out)
        
        return out, attn


class ProbMask(nn.Module):
    """ProbSparse attention mask for causal masking."""
    
    def __init__(self, B, H, L_Q, L_K, index, scores, device="cpu"):
        super(ProbMask, self).__init__()
        _mask = torch.ones(L_Q, L_K, dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L_Q, L_K)
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                            torch.arange(H)[None, :, None],
                            index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask


class TriangularCausalMask(nn.Module):
    """Triangular causal mask for attention."""
    
    def __init__(self, B, L, device="cpu"):
        super(TriangularCausalMask, self).__init__()
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    
    @property
    def mask(self):
        return self._mask

