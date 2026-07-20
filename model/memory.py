import torch
import torch.nn as nn
from typing import Optional

class MemoryModule(nn.Module):
    """
    External memory bank that transformer layers can read from and write to.
    """
    def __init__(self, mem_size: int, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.mem_size = mem_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.memory = nn.Parameter(torch.randn(1, mem_size, embed_dim))
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reads from memory using attention and augments x.
        """
        batch_size, seq_len, _ = x.size()
        head_dim = self.embed_dim // self.num_heads
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        
        mem = self.memory.expand(batch_size, -1, -1)
        k = self.k_proj(mem).view(batch_size, self.mem_size, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(mem).view(batch_size, self.mem_size, self.num_heads, head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return x + self.out_proj(out)
