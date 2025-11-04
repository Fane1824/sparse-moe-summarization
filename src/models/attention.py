"""
Grouped Query Attention (GQA) implementation from scratch.
GQA is a memory-efficient variant of multi-head attention where:
- Q has multiple heads (num_heads)
- K and V are grouped into fewer heads (num_kv_heads)
- Each KV head is shared across multiple Q heads

When num_kv_heads == num_heads, this is standard Multi-Head Attention (MHA).
When num_kv_heads == 1, this is Multi-Query Attention (MQA).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention module.
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (must divide num_heads evenly)
        dropout: Dropout probability
        bias: Whether to use bias in linear projections
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.dropout_p = dropout
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert num_heads % self.num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        
        # Q projection: d_model -> num_heads * head_dim
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        
        # K, V projections: d_model -> num_kv_heads * head_dim
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV caching.
        
        Args:
            query: Query tensor of shape (batch, tgt_len, d_model)
            key: Key tensor of shape (batch, src_len, d_model)
            value: Value tensor of shape (batch, src_len, d_model)
            attn_mask: Attention mask of shape (tgt_len, src_len) or (batch, tgt_len, src_len)
                      Values should be 0 for positions to attend to, -inf for positions to mask
            key_padding_mask: Padding mask for keys of shape (batch, src_len)
                             True for positions to mask
            past_key_value: Optional cached (key, value) from previous forward passes
            use_cache: Whether to return key/value for caching
        
        Returns:
            output: Attention output of shape (batch, tgt_len, d_model)
            attn_weights: Attention weights of shape (batch, num_heads, tgt_len, src_len)
            past_key_value: Cached (key, value) if use_cache=True, else None
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        
        # Project Q, K, V
        # Q: (batch, tgt_len, num_heads * head_dim)
        # K, V: (batch, src_len, num_kv_heads * head_dim)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to separate heads
        # Q: (batch, tgt_len, num_heads, head_dim) -> (batch, num_heads, tgt_len, head_dim)
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K, V: (batch, src_len, num_kv_heads, head_dim) -> (batch, num_kv_heads, src_len, head_dim)
        k = k.view(batch_size, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Use cached K/V if provided
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)  # Concatenate along sequence dimension
            v = torch.cat([past_v, v], dim=2)
            src_len = k.shape[2]  # Update src_len to include cached tokens
        
        # Cache K/V BEFORE repeating (for GQA compatibility)
        # Store the unrepeated K/V for next iteration
        k_cache = k
        v_cache = v
        
        # Repeat K and V to match the number of Q heads (for grouped attention)
        if self.num_kv_heads < self.num_heads:
            # Repeat each KV head num_queries_per_kv times
            # (batch, num_kv_heads, src_len, head_dim) -> (batch, num_heads, src_len, head_dim)
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Compute attention scores
        # (batch, num_heads, tgt_len, head_dim) @ (batch, num_heads, head_dim, src_len)
        # -> (batch, num_heads, tgt_len, src_len)
        scaling = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # (tgt_len, src_len) -> (1, 1, tgt_len, src_len)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # (batch, tgt_len, src_len) -> (batch, 1, tgt_len, src_len)
                attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores + attn_mask
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # (batch, src_len) -> (batch, 1, 1, src_len)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (batch, num_heads, tgt_len, src_len) @ (batch, num_heads, src_len, head_dim)
        # -> (batch, num_heads, tgt_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to (batch, tgt_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        
        # Final output projection
        output = self.out_proj(attn_output)
        
        # Return cache if requested
        if use_cache:
            # Return K/V before they were repeated (for GQA compatibility)
            return output, attn_weights, (k_cache, v_cache)
        else:
            return output, attn_weights, None


class MultiHeadAttention(GroupedQueryAttention):
    """
    Standard Multi-Head Attention (MHA) as a special case of GQA.
    This is equivalent to GQA with num_kv_heads == num_heads.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_heads,  # MHA: same number of KV heads as Q heads
            dropout=dropout,
            bias=bias,
        )


class MultiQueryAttention(GroupedQueryAttention):
    """
    Multi-Query Attention (MQA) as a special case of GQA.
    This is equivalent to GQA with num_kv_heads == 1.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=1,  # MQA: single KV head shared across all Q heads
            dropout=dropout,
            bias=bias,
        )
