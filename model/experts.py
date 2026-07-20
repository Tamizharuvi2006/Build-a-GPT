import torch
import torch.nn as nn
from typing import List

class Expert(nn.Module):
    """
    A single expert network, which is typically a FeedForward network.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)

class ExpertLayer(nn.Module):
    """
    A collection of N experts.
    """
    def __init__(self, dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor, dispatch_weights: torch.Tensor) -> torch.Tensor:
        """
        Passes tokens through the specified experts and combines their outputs.
        """
        B, T, D = x.shape
        out = torch.zeros_like(x)
        
        x_flat = x.view(-1, D)
        expert_indices_flat = expert_indices.view(-1, expert_indices.size(-1))
        dispatch_weights_flat = dispatch_weights.view(-1, dispatch_weights.size(-1))
        
        for i, expert in enumerate(self.experts):
            expert_mask = (expert_indices_flat == i)
            token_indices, k_indices = torch.where(expert_mask)
            
            if token_indices.numel() > 0:
                expert_inputs = x_flat[token_indices]
                expert_outputs = expert(expert_inputs)
                
                weights = dispatch_weights_flat[token_indices, k_indices].unsqueeze(-1)
                expert_outputs = expert_outputs * weights
                
                out.view(-1, D).index_add_(0, token_indices, expert_outputs)
                
        return out
