import os
import torch
from pathlib import Path

class BestModelSaver:
    """Saves the best model based on a monitored metric."""
    def __init__(self, checkpoint_dir: str, metric_name: str = 'val_loss', mode: str = 'min'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def __call__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metric: float):
        """
        Check if the metric improved and save the model if it did.
        
        Args:
            model: The PyTorch model.
            optimizer: The optimizer.
            epoch: The current epoch.
            metric: The metric value to monitor.
        """
        if self.mode == 'min':
            improved = metric < self.best_metric
        else:
            improved = metric > self.best_metric
            
        if improved:
            self.best_metric = metric
            save_path = self.checkpoint_dir / "best.pt"
            
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            
            # Simple save function logic based on standard checkpointing
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metric
            }
            torch.save(checkpoint, save_path)
            print(f"[{self.__class__.__name__}] Saved best model to {save_path} with {self.metric_name}: {metric:.4f}")
