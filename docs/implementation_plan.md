# Stretch Goals Integration (Planner, Long Context, Fusion)

This plan outlines the integration of the final 3 experimental stretch goals into the `FantasyLLM` architecture. 

## Open Questions

> [!IMPORTANT]
> 1. **Multi-Modal Context**: The `MultiModalFusion` module requires an external `context` tensor. For now, I will add an optional `multimodal_context` argument to the `FantasyLLM.forward()` method. Are you planning to pass image embeddings or just text metadata?
> 2. **Planner Application**: The `PlannerModule` outputs a latent plan vector. I will configure it so that the plan is added directly to the embeddings before the transformer stack (residual connection). Does this align with your design?

## Proposed Changes

---

### config/model_config.py
Add configuration flags for the new stretch components.
#### [MODIFY] [model_config.py](file:///d:/FantasyData/config/model_config.py)
- Add `USE_PLANNER = True`, `PLAN_DIM = 192`.
- Add `USE_LONG_CONTEXT = True`, `LONG_CONTEXT_SCALE = 4.0`.
- Add `USE_FUSION = True`, `CONTEXT_DIM = 192`.

---

### model/attention.py & model/sliding_attention.py
Replace standard Rotary Position Embeddings with NTK-aware RoPE.
#### [MODIFY] [attention.py](file:///d:/FantasyData/model/attention.py)
#### [MODIFY] [sliding_attention.py](file:///d:/FantasyData/model/sliding_attention.py)
- If `USE_LONG_CONTEXT` is enabled, instantiate `LongContextExtender` instead of `RotaryEmbedding`.

---

### model/llm.py
Incorporate the Planner and Fusion modules.
#### [MODIFY] [llm.py](file:///d:/FantasyData/model/llm.py)
- Instantiate `PlannerModule` and `MultiModalFusion` if configured.
- Update `forward(tokens, use_cache=False, kv_cache=None, multimodal_context=None)`:
  - Run embeddings through the `PlannerModule` and add the plan to the embeddings.
  - Run the `MultiModalFusion` cross-attention if `multimodal_context` is provided.

---

## Verification Plan

### Automated Tests
- Run `python experiments/test_model.py` to ensure the forward pass doesn't crash with all stretch goals enabled.
