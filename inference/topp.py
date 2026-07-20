import torch

def top_p_filter(logits, p):
    """
    Nucleus sampling: keeps smallest set of tokens whose cumulative probability exceeds p.
    """
    if p >= 1.0:
        return logits
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter back to the original format
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    
    filtered_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return filtered_logits
