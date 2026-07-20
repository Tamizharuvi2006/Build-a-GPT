import torch
import torch.nn as nn

class PlannerModule(nn.Module):
    """
    Generates a latent plan vector before generating text.
    """
    def __init__(self, embed_dim: int, plan_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.plan_dim = plan_dim
        
        self.proj_in = nn.Linear(embed_dim, plan_dim)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(plan_dim, plan_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects hidden states to plan space.
        """
        plan = self.act(self.proj_in(x))
        plan = self.proj_out(plan)
        return plan
