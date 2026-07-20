import torch
import torch.nn as nn

class ReasoningLayer(nn.Module):
    """
    Iterative refinement / scratchpad layer.
    Runs multiple passes of attention over the hidden state.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_iterations: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_iterations = num_iterations
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Refines x over multiple iterations.
        """
        batch_size, seq_len, _ = x.size()
        head_dim = self.embed_dim // self.num_heads
        
        refined_x = x
        
        for _ in range(self.num_iterations):
            residual = refined_x
            norm_x = self.norm(refined_x)
            
            qkv = self.qkv_proj(norm_x)
            q, k, v = qkv.chunk(3, dim=-1)
            
            q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
            
            refined_x = residual + self.out_proj(out)
            
        return refined_x
