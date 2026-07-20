import torch
from typing import List

class LRMonitor:
    """Callback to monitor and log the learning rate."""
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self._history: List[float] = []

    def get_lr(self) -> float:
        """Returns the current learning rate (assumes same LR across all groups)."""
        return self.optimizer.param_groups[0]['lr']

    def log(self):
        """Logs the current learning rate and adds it to history."""
        lr = self.get_lr()
        self._history.append(lr)
        print(f"[{self.__class__.__name__}] Current Learning Rate: {lr:.2e}")

    @property
    def history(self) -> List[float]:
        """Returns the history of learning rates."""
        return self._history
