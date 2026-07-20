import torch
import torch.nn as nn

from model.embedding import TokenEmbedding
from model.hybrid_transformer import HybridTransformer
from model.rmsnorm import RMSNorm
from model.reasoning import ReasoningLayer
from model.memory import MemoryModule
from model.planner import PlannerModule
from model.fusion import MultiModalFusion

from config.model_config import *


class FantasyLLM(nn.Module):

    def __init__(self):

        super().__init__()

        # Token Embedding
        self.embedding = TokenEmbedding(
            VOCAB_SIZE,
            EMBED_DIM
        )
        
        if 'USE_PLANNER' in globals() and USE_PLANNER:
            self.planner = PlannerModule(EMBED_DIM, PLAN_DIM)
        else:
            self.planner = None
            
        if 'USE_FUSION' in globals() and USE_FUSION:
            self.fusion = MultiModalFusion(EMBED_DIM, CONTEXT_DIM)
        else:
            self.fusion = None

        # Transformer
        self.transformer = HybridTransformer()

        if 'USE_MEMORY' in globals() and USE_MEMORY:
            self.memory = MemoryModule(MEMORY_SIZE, EMBED_DIM)
        else:
            self.memory = None
            
        if 'USE_REASONING' in globals() and USE_REASONING:
            self.reasoning = ReasoningLayer(EMBED_DIM, NUM_QUERY_HEADS, REASONING_ITERATIONS)
        else:
            self.reasoning = None

        # Final RMSNorm
        self.norm = RMSNorm(
            EMBED_DIM
        )

        # Output Head
        self.lm_head = nn.Linear(
            EMBED_DIM,
            VOCAB_SIZE,
            bias=BIAS if 'BIAS' in globals() else False
        )

        # Weight Tying
        self.lm_head.weight = self.embedding.embedding.weight

    def forward(self, tokens, use_cache=False, kv_cache=None, multimodal_context=None):

        x = self.embedding(tokens)
        
        if self.planner is not None:
            plan = self.planner(x)
            x = x + plan
            
        if self.fusion is not None and multimodal_context is not None:
            x = self.fusion(x, multimodal_context)

        x, moe_loss = self.transformer(x, use_cache=use_cache, kv_cache=kv_cache)

        if self.memory is not None:
            x = self.memory(x)
            
        if self.reasoning is not None:
            x = self.reasoning(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        if self.training:
            return logits, moe_loss
        return logits