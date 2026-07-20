import torch
import torch.nn as nn

class OutputHead(nn.Module):
    """
    Flexible output projection. Can do language modeling head, classification head, etc.
    """
    def __init__(self, embed_dim: int, vocab_size: int, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.proj = nn.Linear(embed_dim, vocab_size, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns logits for the given hidden states.
        """
        return self.proj(x)
