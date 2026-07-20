import torch.nn as nn

from model.transformer_block import TransformerBlock

from config.model_config import *


class HybridTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(NUM_LAYERS)
            ]
        )

    def forward(self, x, use_cache=False, kv_cache=None):
        total_moe_loss = 0.0
        for i, layer in enumerate(self.layers):
            
            past_kv = kv_cache.get(i) if use_cache and kv_cache is not None else None
            
            x, new_kv, lb_loss = layer(x, past_kv=past_kv)
            total_moe_loss += lb_loss
            
            if use_cache and kv_cache is not None:
                kv_cache.update(i, new_kv[0], new_kv[1])

        return x, total_moe_loss