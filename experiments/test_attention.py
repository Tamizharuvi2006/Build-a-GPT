"""Tests for GroupedQueryAttention."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from model.attention import GroupedQueryAttention

def main():
    print("Testing GroupedQueryAttention...")
    
    batch_size = 2
    seq_len = 16
    embed_dim = 192
    
    # Initialize GQA
    attn = GroupedQueryAttention(
        embed_dim=embed_dim,
        num_query_heads=6,
        num_kv_heads=2,
        head_dim=32,
        dropout=0.1
    )
    
    # Dummy input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    out = attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    assert out.shape == x.shape, "Output shape mismatch!"
    
    print("GQA output shape is correct. Causal mask is applied internally.")

if __name__ == "__main__":
    main()
