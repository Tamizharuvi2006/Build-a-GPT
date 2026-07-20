import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class TopKRouter(nn.Module):
    """
    Routes tokens to top-k experts.
    """
    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_proj = nn.Linear(embed_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            dispatch_weights: Weights for each of the top-k experts
            expert_indices: Indices of the top-k experts
            load_balancing_loss: Optional loss to encourage balanced routing
        """
        logits = self.routing_proj(x)
        routing_probs = F.softmax(logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        dispatch_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        expert_density = routing_probs.mean(dim=(0, 1))
        load_balancing_loss = (expert_density ** 2).sum() * self.num_experts
        
        return dispatch_weights, top_k_indices, load_balancing_loss
