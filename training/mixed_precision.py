import torch

def create_scaler() -> torch.cuda.amp.GradScaler:
    """Creates a gradient scaler for mixed precision training."""
    return torch.cuda.amp.GradScaler()

def mixed_precision_context(device: torch.device):
    """Returns an autocast context manager for the given device."""
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    return torch.autocast(device_type=device_type)

def setup_optimal_gpu_settings():
    """Auto-detects GPU capabilities and sets the optimal precision and flags."""
    dtype = torch.float16
    if torch.cuda.is_available():
        # Get compute capability of the primary device
        major, minor = torch.cuda.get_device_capability()
        
        # Ampere (8.x) or newer (like RTX 3050, 4050, A100) supports bfloat16 & TF32
        if major >= 8:
            print(f"[Auto-Tuner] Detected Compute Capability {major}.{minor}. Enabling bfloat16 and TF32.")
            dtype = torch.bfloat16
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            print(f"[Auto-Tuner] Detected Compute Capability {major}.{minor}. Using float16.")
            dtype = torch.float16
            
    return dtype

class MixedPrecisionManager:
    """Manages mixed precision training."""
    def __init__(self, enabled: bool = True, dtype: torch.dtype = None):
        self.enabled = enabled
        self.dtype = dtype if dtype is not None else setup_optimal_gpu_settings()
        
        # GradScaler is primarily needed for float16 to prevent underflow.
        # It's not strictly necessary for bfloat16, but we'll conditionally enable it.
        use_scaler = self.enabled and self.dtype == torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        
    def context(self, device: torch.device):
        """Returns the autocast context manager."""
        device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        return torch.autocast(device_type=device_type, dtype=self.dtype, enabled=self.enabled)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scales the loss."""
        return self.scaler.scale(loss)

    def unscale_and_step(self, optimizer: torch.optim.Optimizer):
        """Unscales gradients and steps the optimizer."""
        self.scaler.step(optimizer)
        self.scaler.update()
