import torch

class GradientAccumulator:
    """Manages gradient accumulation over multiple steps."""
    def __init__(self, accumulation_steps: int = 4):
        if accumulation_steps <= 0:
            raise ValueError("Accumulation steps must be greater than 0")
        self._accumulation_steps = accumulation_steps
        
    @property
    def accumulation_steps(self) -> int:
        return self._accumulation_steps
        
    def should_step(self, batch_idx: int) -> bool:
        """Determines if the optimizer should step based on the batch index."""
        # Step when batch_idx + 1 is a multiple of accumulation_steps
        return (batch_idx + 1) % self._accumulation_steps == 0
        
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scales the loss by the accumulation steps."""
        return loss / self._accumulation_steps
        
    def effective_batch_size(self, batch_size: int) -> int:
        """Calculates the effective batch size given the base batch size."""
        return batch_size * self._accumulation_steps
