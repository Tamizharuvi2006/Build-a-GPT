"""Story generator using the FantasyLLM."""
import torch
from model.llm import FantasyLLM

from tokenizer.encode import encode
from tokenizer.decode import decode

# Assuming inference.generate provides generate_text
try:
    from inference.generate import generate_text
except ImportError:
    # Dummy fallback if module missing
    def generate_text(model, prompt, tokenizer_encode, tokenizer_decode, max_new_tokens, temperature, top_k, top_p, device):
        pass

class StoryGenerator:
    """Story generation wrapper around FantasyLLM."""
    
    def __init__(self, checkpoint_path, tokenizer_path='data/tokenizer/tokenizer.json', device=None, search_engine=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = FantasyLLM()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer_path = tokenizer_path
        self.search_engine = search_engine
        
    def generate(self, prompt, max_tokens=200, temperature=0.8, top_k=50, top_p=0.9):
        """Generate a story from a prompt."""
        output = generate_text(
            model=self.model,
            prompt=prompt,
            tokenizer_encode=encode,
            tokenizer_decode=decode,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=self.device
        )
        return output if isinstance(output, str) else str(output)
        
    def generate_with_rag(self, prompt, max_tokens=200, temperature=0.8, top_k=50, top_p=0.9):
        """Generate a story incorporating retrieved context."""
        if self.search_engine is None:
            return self.generate(prompt, max_tokens, temperature, top_k, top_p)
            
        results = self.search_engine.search(prompt, top_k=3)
        
        context = "\n".join([f"- {res.chunk}" for res in results])
        augmented_prompt = f"Context:\n{context}\n\nStory Prompt: {prompt}\n\nStory:"
        
        return self.generate(augmented_prompt, max_tokens, temperature, top_k, top_p)
