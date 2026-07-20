import torch
from typing import Dict
import math

class GradientMonitor:
    """Monitors gradients of a PyTorch model for anomalies and norms."""
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def compute_grad_norm(self, norm_type: float = 2.0) -> float:
        """Computes the total gradient norm across all model parameters."""
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
            
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)
        return total_norm.item()
        
    def get_layer_norms(self, norm_type: float = 2.0) -> Dict[str, float]:
        """Returns a dictionary of gradient norms per layer/parameter."""
        norms = {}
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                norms[name] = torch.norm(p.grad.detach(), norm_type).item()
        return norms
        
    def check_for_anomalies(self):
        """Checks for NaN, Inf, or suspiciously large/small gradients."""
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                grad_norm = torch.norm(p.grad.detach(), 2.0).item()
                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    print(f"WARNING: Anomaly detected! Parameter '{name}' has NaN/Inf gradient.")
                elif grad_norm > 1e4:
                    print(f"WARNING: Exploding gradient possible! Parameter '{name}' has high gradient norm: {grad_norm:.4f}")
                elif grad_norm < 1e-6 and grad_norm > 0:
                    print(f"WARNING: Vanishing gradient possible! Parameter '{name}' has small gradient norm: {grad_norm:.4e}")
