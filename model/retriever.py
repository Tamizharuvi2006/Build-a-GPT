import torch
import torch.nn as nn

class Retriever(nn.Module):
    """
    Retrieves relevant context from a memory bank using attention-based lookup.
    """
    def __init__(self, embed_dim: int, num_entries: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_entries = num_entries
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query: torch.Tensor, memory_bank: torch.Tensor) -> torch.Tensor:
        """
        query: (batch_size, query_len, embed_dim)
        memory_bank: (batch_size, num_entries, embed_dim)
        Returns retrieved context.
        """
        q = self.q_proj(query)
        k = self.k_proj(memory_bank)
        v = self.v_proj(memory_bank)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        retrieved_context = torch.matmul(attn, v)
        return retrieved_context
