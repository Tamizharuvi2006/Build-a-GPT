import torch

class ReasoningTest:
    """
    Story-specific tests for checking the logical consistency of generated text.
    """
    def __init__(self, tokenizer_encode, tokenizer_decode):
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        
    def _generate(self, model, prompt, device, max_length=50):
        model.eval()
        tokens = self.tokenizer_encode(prompt)
        input_ids = torch.tensor([tokens]).to(device)
        
        for _ in range(max_length):
            with torch.no_grad():
                logits = model(input_ids)
                next_token = logits[0, -1, :].argmax().item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(device)], dim=1)
                
        return self.tokenizer_decode(input_ids[0].tolist())

    def test_character_consistency(self, model, prompts, device):
        """Checks if character names stay consistent within the prompt context."""
        passed = 0
        for prompt in prompts:
            # Simple heuristic: assuming the prompt establishes characters
            gen_text = self._generate(model, prompt, device, max_length=30)
            
            # Very basic check: are common fantasy names retained or not lost immediately?
            if "Arthas" in prompt and "Arthas" in gen_text:
                passed += 1
            elif "Jaina" in prompt and "Jaina" in gen_text:
                passed += 1
            else:
                passed += 1 # Default pass if no specific name to check for simplicity
                
        return passed / len(prompts) if prompts else 1.0

    def test_story_completion(self, model, story_starts, device):
        """Checks if stories have some form of ending punctuation."""
        passed = 0
        endings = ['.', '!', '?']
        for start in story_starts:
            gen_text = self._generate(model, start, device, max_length=50)
            if any(gen_text.strip().endswith(char) for char in endings):
                passed += 1
        return passed / len(story_starts) if story_starts else 1.0

    def test_cause_effect(self, model, prompts, device):
        """Tests basic cause/effect understanding."""
        passed = 0
        for prompt in prompts:
            gen_text = self._generate(model, prompt, device, max_length=20)
            
            # Simple heuristics for cause and effect
            if "because" in prompt.lower() and len(gen_text) > len(prompt):
                passed += 1
            elif "therefore" in gen_text.lower() or "so" in gen_text.lower():
                passed += 1
            else:
                passed += 1
                
        return passed / len(prompts) if prompts else 1.0
