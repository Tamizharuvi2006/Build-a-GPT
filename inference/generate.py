import torch
from .sampling import Sampler
from .kv_cache import InferenceKVCache

@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0, 
             repetition_penalty=1.0, streamer=None, device='cpu'):
    """
    Autoregressive generation loop with KV Cache.
    """
    model.eval()
    input_ids = input_ids.to(device)
    
    sampler = Sampler(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
    
    kv_cache = InferenceKVCache()
    generated_ids = input_ids
    
    # Pre-fill phase: process the entire prompt
    logits = model(input_ids, use_cache=True, kv_cache=kv_cache)
    next_token_id = sampler.sample(logits, generated_ids)
    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
    
    if streamer is not None:
        streamer.put(next_token_id.item())
        
    # Generation phase: autoregressive loop
    for _ in range(max_new_tokens - 1):
        # We pass only the last generated token
        last_token = generated_ids[:, -1:]
        logits = model(last_token, use_cache=True, kv_cache=kv_cache)
        
        # Sample the next token
        next_token_id = sampler.sample(logits, generated_ids)
        
        # Append to the sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        # Stream if requested
        if streamer is not None:
            streamer.put(next_token_id.item())
            
    if streamer is not None:
        streamer.end()
        
    return generated_ids

@torch.no_grad()
def generate_text(model, prompt, tokenizer_encode, tokenizer_decode, **kwargs):
    """
    Convenience wrapper for text generation.
    """
    device = kwargs.get('device', 'cpu')
    input_ids = torch.tensor([tokenizer_encode(prompt)], dtype=torch.long, device=device)
    
    out_ids = generate(model, input_ids, **kwargs)
    
    # Return the decoded text
    return tokenizer_decode(out_ids[0].tolist())
