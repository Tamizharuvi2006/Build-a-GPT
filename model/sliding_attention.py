import torch
import torch.nn as nn
from typing import Optional
import math

from model.rope import RotaryEmbedding
from model.long_context import LongContextExtender
from config.model_config import *

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention. Like Grouped Query Attention but with a sliding window mask.
    Each token can only attend to WINDOW_SIZE previous tokens.
    """
    def __init__(self, embed_dim: int, num_query_heads: int, num_kv_heads: int, head_dim: int, window_size: int, dropout: float = 0.1):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.num_groups = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(embed_dim, num_query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_query_heads * head_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        if 'USE_LONG_CONTEXT' in globals() and USE_LONG_CONTEXT:
            self.rope = LongContextExtender(LONG_CONTEXT_SCALE, head_dim)
        else:
            self.rope = RotaryEmbedding(head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, past_kv: Optional[tuple] = None):
        """
        Forward pass for sliding window attention.
        """
        batch_size, seq_len, embed_dim = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        offset = past_kv[0].shape[2] if past_kv is not None else 0
        
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        current_kv = (k, v)

        k_rep = torch.repeat_interleave(k, self.num_groups, dim=1)
        v_rep = torch.repeat_interleave(v, self.num_groups, dim=1)

        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if seq_len > 1:
            causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            causal_mask = torch.tril(causal_mask)
            window_mask = torch.triu(causal_mask, diagonal=-self.window_size + 1)
            
            if offset > 0:
                past_window_mask = torch.ones(seq_len, offset, dtype=torch.bool, device=x.device)
                window_mask = torch.cat([past_window_mask, window_mask], dim=1)
                
            scores = scores.masked_fill(~window_mask, float('-inf'))
        else:
            if offset > 0:
                total_len = offset + 1
                window_mask = torch.zeros(1, total_len, dtype=torch.bool, device=x.device)
                start_idx = max(0, total_len - self.window_size)
                window_mask[:, start_idx:] = True
                scores = scores.masked_fill(~window_mask, float('-inf'))

        if mask is not None:
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v_rep)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out), current_kv
