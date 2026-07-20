def count_parameters(model):
    """
    Counts the total number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model):
    """
    Generates a formatted summary of the model layers and parameters.
    """
    summary = []
    summary.append("Model Summary:")
    summary.append("-" * 50)
    
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        shape = str(list(parameter.shape))
        summary.append(f"{name:<30} | {shape:<15} | {params}")
        total_params += params
        
    summary.append("-" * 50)
    summary.append(f"Total Trainable Params: {total_params}")
    return "\n".join(summary)

def format_time(seconds):
    """
    Formats a duration in seconds into a human-readable string.
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"

def format_number(n):
    """
    Formats a number into a shorter string representation (e.g. 1.2M, 15.3K).
    """
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(n)

def estimate_model_size(model):
    """
    Estimates the memory size of a model.
    """
    # Assuming float32 (4 bytes per parameter)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_mb = (param_size + buffer_size) / 1024**2
    return f"{size_mb:.1f} MB"
