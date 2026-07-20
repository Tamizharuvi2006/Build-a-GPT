class EarlyStopping:
    """Callback for early stopping during training."""
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.stopped = False

    def __call__(self, metric: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            metric (float): The tracked metric (e.g., validation loss).
            
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.mode == 'min':
            improved = metric < self.best_metric - self.min_delta
        else:
            improved = metric > self.best_metric + self.min_delta

        if improved:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                
        return self.stopped
