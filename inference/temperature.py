import torch

def temperature_scale(logits, temperature):
    """
    Scales logits by temperature.
    Handles temperature=0 as greedy (argmax).
    """
    if temperature == 0.0:
        max_idx = torch.argmax(logits, dim=-1, keepdim=True)
        greedy_logits = torch.full_like(logits, float('-inf'))
        greedy_logits.scatter_(-1, max_idx, 0.0)
        return greedy_logits
    return logits / temperature

def repetition_penalty(logits, generated_ids, penalty=1.1):
    """
    Applies a repetition penalty to the logits based on previously generated ids.
    """
    if penalty == 1.0 or generated_ids is None or generated_ids.numel() == 0:
        return logits
    
    # Gather logits of previously generated tokens
    score = torch.gather(logits, -1, generated_ids)
    
    # Apply penalty: multiply if score < 0, else divide
    score = torch.where(score < 0, score * penalty, score / penalty)
    
    # Scatter the penalized scores back into the logits
    logits.scatter_(-1, generated_ids, score)
    return logits
