import torch

def top_k_filter(logits, k):
    """
    Keeps only the top-k logits, setting the rest to -inf.
    """
    if k == 0 or k >= logits.size(-1):
        return logits
    
    # Find the k-th largest value
    top_k_logits, _ = torch.topk(logits, k, dim=-1)
    min_top_k = top_k_logits[..., -1, None]
    
    # Filter out anything less than the minimum top-k value
    filtered_logits = torch.where(
        logits < min_top_k,
        torch.full_like(logits, float('-inf')),
        logits
    )
    return filtered_logits
