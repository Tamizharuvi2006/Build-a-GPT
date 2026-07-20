import torch.nn as nn

from model.rmsnorm import RMSNorm
from model.attention import GroupedQueryAttention
from model.sliding_attention import SlidingWindowAttention
from model.feedforward import FeedForward
from model.experts import ExpertLayer
from model.router import TopKRouter

from config.model_config import *


class TransformerBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.norm1 = RMSNorm(EMBED_DIM)

        if USE_SLIDING_WINDOW:
            self.attn = SlidingWindowAttention(
                embed_dim=EMBED_DIM,
                num_query_heads=NUM_QUERY_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_dim=HEAD_DIM,
                window_size=WINDOW_SIZE,
                dropout=ATTENTION_DROPOUT,
            )
        else:
            self.attn = GroupedQueryAttention(
                embed_dim=EMBED_DIM,
                num_query_heads=NUM_QUERY_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_dim=HEAD_DIM,
                dropout=ATTENTION_DROPOUT,
            )

        self.norm2 = RMSNorm(EMBED_DIM)

        if 'USE_MOE' in globals() and USE_MOE:
            self.router = TopKRouter(EMBED_DIM, NUM_EXPERTS, TOP_K_EXPERTS)
            self.experts = ExpertLayer(EMBED_DIM, int(EMBED_DIM * FFN_MULTIPLIER), NUM_EXPERTS)
            self.ffn = None
        else:
            self.ffn = FeedForward(dim=EMBED_DIM)
            self.router = None
            self.experts = None

    def forward(self, x, past_kv=None):

        attn_out, new_kv = self.attn(
            self.norm1(x),
            past_kv=past_kv
        )
        x = x + attn_out

        lb_loss = 0.0
        norm_x = self.norm2(x)
        
        if self.ffn is not None:
            ffn_out = self.ffn(norm_x)
        else:
            dispatch_weights, top_k_indices, lb_loss = self.router(norm_x)
            ffn_out = self.experts(norm_x, top_k_indices, dispatch_weights)

        x = x + ffn_out

        return x, new_kv, lb_loss