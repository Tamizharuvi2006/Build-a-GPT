"""
Token Embedding module for the LLM.
"""

import math
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Token embedding layer that scales embeddings and applies dropout.
    """
    def __init__(self, vocab_size: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for token embedding.
        
        Args:
            x: Input tensor of token IDs.
            
        Returns:
            Embedded and scaled tokens with dropout applied.
        """
        # Scale embeddings by sqrt(embed_dim)
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        return self.dropout(x)
