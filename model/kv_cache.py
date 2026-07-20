import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    """
    Key-Value Cache for efficient autoregressive inference.
    """
    def __init__(self, max_batch_size: int, max_seq_len: int, num_kv_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        self.k_cache = {}
        self.v_cache = {}
        self.seq_len = 0

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new key and value for the given layer.
        Returns the cached keys and values up to the current sequence length.
        """
        batch_size, num_heads, seq_len, head_dim = key.size()
        
        if layer_idx not in self.k_cache:
            self.k_cache[layer_idx] = torch.zeros(
                (self.max_batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim),
                device=self.device, dtype=self.dtype
            )
            self.v_cache[layer_idx] = torch.zeros(
                (self.max_batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim),
                device=self.device, dtype=self.dtype
            )
            
        start_idx = self.seq_len
        end_idx = start_idx + seq_len
        
        self.k_cache[layer_idx][:batch_size, :, start_idx:end_idx, :] = key
        self.v_cache[layer_idx][:batch_size, :, start_idx:end_idx, :] = value
        
        cached_k = self.k_cache[layer_idx][:batch_size, :, :end_idx, :]
        cached_v = self.v_cache[layer_idx][:batch_size, :, :end_idx, :]
        return cached_k, cached_v

    def reset(self):
        """Resets the cache sequence length."""
        self.seq_len = 0

    def get_seq_len(self) -> int:
        """Returns the current sequence length."""
        return self.seq_len
