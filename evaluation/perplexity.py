import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate_perplexity(model, dataloader, device):
    """
    Evaluates the perplexity of a model on a dataset.
    
    Args:
        model: The language model.
        dataloader: DataLoader containing validation/test data.
        device: The device to run evaluation on.
        
    Returns:
        float: The calculated perplexity.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        
        # Reshape logits to (B * T, VOCAB_SIZE) and targets to (B * T)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        total_loss += loss.item()
        total_batches += 1
        
    if total_batches == 0:
        return float('inf')
        
    avg_loss = total_loss / total_batches
    return math.exp(avg_loss)
