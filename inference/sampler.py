from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

class BaseSampler(ABC):
    """Abstract base class for samplers."""
    
    @abstractmethod
    def sample(self, logits):
        """Samples from logits and returns token_id."""
        pass

class GreedySampler(BaseSampler):
    """Always selects the most likely token."""
    
    @torch.no_grad()
    def sample(self, logits):
        # Take the logits of the last token
        if logits.dim() > 1 and logits.size(-2) > 1:
            logits = logits[..., -1, :]
        return torch.argmax(logits, dim=-1, keepdim=True)

class RandomSampler(BaseSampler):
    """Samples purely randomly from the distribution."""
    
    @torch.no_grad()
    def sample(self, logits):
        if logits.dim() > 1 and logits.size(-2) > 1:
            logits = logits[..., -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
