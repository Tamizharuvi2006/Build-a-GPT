"""Benchmarking utilities for model performance."""
import time
import torch
from typing import Dict, Any

def benchmark_inference(model: torch.nn.Module, device: torch.device, seq_len: int = 128, num_runs: int = 100) -> Dict[str, Any]:
    """Benchmark inference speed."""
    model.to(device)
    model.eval()
    dummy_input = torch.randint(0, 8000, (1, seq_len)).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_run = total_time / num_runs
    tokens_per_second = (seq_len * num_runs) / total_time
    
    return {
        "time_per_run_ms": time_per_run * 1000,
        "tokens_per_second": tokens_per_second,
        "model_size_params": sum(p.numel() for p in model.parameters())
    }

def benchmark_training(model: torch.nn.Module, device: torch.device, batch_size: int = 8, seq_len: int = 128) -> Dict[str, Any]:
    """Benchmark training step speed."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    dummy_input = torch.randint(0, 8000, (batch_size, seq_len)).to(device)
    dummy_target = torch.randint(0, 8000, (batch_size, seq_len)).to(device)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        out = model(dummy_input)
        loss = criterion(out.view(-1, 8000), dummy_target.view(-1))
        loss.backward()
        optimizer.step()
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    num_runs = 50
    
    for _ in range(num_runs):
        optimizer.zero_grad()
        out = model(dummy_input)
        loss = criterion(out.view(-1, 8000), dummy_target.view(-1))
        loss.backward()
        optimizer.step()
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    time_per_step = (end_time - start_time) / num_runs
    
    return {
        "training_step_time_ms": time_per_step * 1000,
        "memory_usage_mb": torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0
    }
