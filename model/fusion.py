import torch
import torch.nn as nn
from typing import Optional

class MultiModalFusion(nn.Module):
    """
    Fuses text embeddings with another modality via cross-attention.
    """
    def __init__(self, embed_dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(context_dim, embed_dim)
        self.v_proj = nn.Linear(context_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, embed_dim)
        context: (batch_size, context_len, context_dim)
        """
        batch_size, seq_len, _ = x.size()
        context_len = context.size(1)
        
        head_dim = self.embed_dim // self.num_heads
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(context).view(batch_size, context_len, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(context).view(batch_size, context_len, self.num_heads, head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(out) + x
