import time
import torch

class SimpleProfiler:
    """
    A simple profiler for measuring execution time of code blocks.
    """
    def __init__(self):
        self.start_times = {}
        self.durations = {}
        
    def start(self, name):
        """Starts timing a block."""
        self.start_times[name] = time.perf_counter()
        
    def stop(self, name):
        """Stops timing a block and records the duration."""
        if name in self.start_times:
            duration = time.perf_counter() - self.start_times[name]
            if name not in self.durations:
                self.durations[name] = []
            self.durations[name].append(duration)
            del self.start_times[name]
            
    def summary(self):
        """Returns a string summary of the profiled durations."""
        lines = ["Profiler Summary:", "-" * 30]
        for name, times in self.durations.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            lines.append(f"{name:<15} | Avg: {avg_time:.4f}s | Total: {total_time:.4f}s | Calls: {len(times)}")
        return "\n".join(lines)

def profile_model(model, input_shape, device):
    """
    Profiles the memory and execution time of a model's forward and backward passes.
    
    Args:
        model: The PyTorch model.
        input_shape (tuple): Shape of the input tensor.
        device: Device to run profiling on.
        
    Returns:
        dict: Dictionary containing profiling metrics.
    """
    model.to(device)
    model.train()
    
    # Create dummy input
    # Assuming int64 inputs for an LLM (tokens)
    dummy_input = torch.randint(0, 100, input_shape, dtype=torch.long, device=device)
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated(device)
        
    # Profile forward pass
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    output = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    forward_time = time.perf_counter() - t0
    
    # Fake loss for backward
    if isinstance(output, tuple):
        output = output[0]
    loss = output.sum()
    
    # Profile backward pass
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    backward_time = time.perf_counter() - t1
    
    metrics = {
        'forward_time_ms': forward_time * 1000,
        'backward_time_ms': backward_time * 1000
    }
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device)
        metrics['peak_memory_mb'] = peak_memory / (1024 ** 2)
        
    return metrics
