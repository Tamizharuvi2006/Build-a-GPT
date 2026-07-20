import torch
import torch.nn.functional as F
from .temperature import temperature_scale, repetition_penalty
from .topk import top_k_filter
from .topp import top_p_filter

class Sampler:
    """
    Combined Sampler for temperature, top-k, and top-p sampling.
    """
    def __init__(self, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.rep_penalty = repetition_penalty

    @torch.no_grad()
    def sample(self, logits, generated_ids=None):
        """
        Samples a single token from the logits.
        """
        # Ensure we only work with the last token's logits
        if logits.dim() > 1 and logits.size(-2) > 1:
            logits = logits[..., -1, :]
        elif logits.dim() > 2:
            logits = logits[..., -1, :]

        logits = repetition_penalty(logits, generated_ids, self.rep_penalty)
        logits = temperature_scale(logits, self.temperature)
        logits = top_k_filter(logits, self.top_k)
        logits = top_p_filter(logits, self.top_p)
        
        probs = F.softmax(logits, dim=-1)
        
        if self.temperature == 0.0:
            next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token_id = torch.multinomial(probs, num_samples=1)
            
        return next_token_id
