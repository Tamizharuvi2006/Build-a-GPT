import torch

def token_accuracy(logits, targets):
    """
    Calculates the fraction of correctly predicted next tokens.
    
    Args:
        logits (torch.Tensor): Model predictions of shape (B, T, VOCAB_SIZE).
        targets (torch.Tensor): True labels of shape (B, T).
        
    Returns:
        float: Token accuracy.
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()

@torch.no_grad()
def sequence_accuracy(model, dataloader, device, threshold=0.8):
    """
    Calculates the fraction of sequences with accuracy >= threshold.
    
    Args:
        model: The language model.
        dataloader: DataLoader for the dataset.
        device: Device to run evaluation on.
        threshold (float): Accuracy threshold for a sequence to be considered 'correct'.
        
    Returns:
        float: Sequence accuracy fraction.
    """
    model.eval()
    correct_sequences = 0
    total_sequences = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
        
        # Calculate accuracy per sequence
        seq_acc = (preds == targets).float().mean(dim=1)
        
        correct_sequences += (seq_acc >= threshold).sum().item()
        total_sequences += targets.size(0)
        
    if total_sequences == 0:
        return 0.0
        
    return correct_sequences / total_sequences

def top_k_accuracy(logits, targets, k=5):
    """
    Calculates the top-k accuracy for next token prediction.
    
    Args:
        logits (torch.Tensor): Model predictions of shape (B, T, VOCAB_SIZE).
        targets (torch.Tensor): True labels of shape (B, T).
        k (int): Number of top predictions to consider.
        
    Returns:
        float: Top-k accuracy.
    """
    _, top_k_preds = logits.topk(k, dim=-1)
    # Check if target is in top k predictions
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).float().sum()
    total = targets.numel()
    return (correct / total).item()
