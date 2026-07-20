import math
import torch
import torch.nn as nn

from model.rope import RotaryEmbedding
from model.long_context import LongContextExtender
from config.model_config import *


class GroupedQueryAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        dropout=0.1,
    ):

        super().__init__()

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(
            embed_dim,
            num_query_heads * head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            embed_dim,
            num_kv_heads * head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            embed_dim,
            num_kv_heads * head_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            num_query_heads * head_dim,
            embed_dim,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)

        if 'USE_LONG_CONTEXT' in globals() and USE_LONG_CONTEXT:
            self.rope = LongContextExtender(LONG_CONTEXT_SCALE, head_dim)
        else:
            self.rope = RotaryEmbedding(head_dim)

    def forward(self, x, past_kv=None):

        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            B,
            T,
            self.num_query_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        offset = past_kv[0].shape[2] if past_kv is not None else 0

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        current_kv = (k, v)

        repeat = self.num_query_heads // self.num_kv_heads

        k_rep = k.repeat_interleave(repeat, dim=1)
        v_rep = v.repeat_interleave(repeat, dim=1)

        scores = torch.matmul(
            q,
            k_rep.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        if T > 1:
            mask = torch.triu(
                torch.ones(T, T, device=x.device),
                diagonal=1,
            ).bool()
            
            if offset > 0:
                past_mask = torch.zeros(T, offset, dtype=torch.bool, device=x.device)
                full_mask = torch.cat([past_mask, mask], dim=1)
                scores = scores.masked_fill(full_mask, float("-inf"))
            else:
                scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        weights = self.dropout(weights)

        out = torch.matmul(weights, v_rep)

        out = out.transpose(1, 2).contiguous()

        out = out.view(
            B,
            T,
            self.num_query_heads * self.head_dim,
        )

        return self.out_proj(out), current_kv