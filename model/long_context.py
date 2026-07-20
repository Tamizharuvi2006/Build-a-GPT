import torch
import torch.nn as nn
import math

class LongContextExtender(nn.Module):
    """
    Implements techniques for extending context length beyond training length.
    Uses NTK-aware RoPE scaling.
    """
    def __init__(self, scale_factor: float, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.scale_factor = scale_factor
        self.max_seq_len = max_seq_len
        self.base = base
        
        ntk_base = base * (scale_factor ** (dim / (dim - 2)))
        
        inv_freq = 1.0 / (ntk_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Applies scaled rotary embeddings to x with optional offset.
        """
        seq_len = x.shape[-2]
        t = torch.arange(offset, offset + seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        rotated_x = torch.cat((-x2, x1), dim=-1)
        
        return (x * cos) + (rotated_x * sin)
