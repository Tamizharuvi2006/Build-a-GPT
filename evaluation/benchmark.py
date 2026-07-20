import torch

class Benchmark:
    """
    Benchmark class for evaluating the qualitative capabilities of a language model.
    """
    def __init__(self, model, tokenizer_encode, tokenizer_decode, device):
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.device = device
        
    def _generate_text(self, prompt, max_length=50):
        self.model.eval()
        tokens = self.tokenizer_encode(prompt)
        input_ids = torch.tensor([tokens]).to(self.device)
        
        for _ in range(max_length):
            with torch.no_grad():
                logits = self.model(input_ids)
                next_token = logits[0, -1, :].argmax().item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(self.device)], dim=1)
                
        return self.tokenizer_decode(input_ids[0].tolist())
        
    def test_coherence(self):
        """Generates text and checks it's not complete garbage using simple heuristics."""
        prompt = "Once upon a time in the magical kingdom of"
        generated = self._generate_text(prompt, max_length=20)
        words = generated.split()
        
        # Check if words look somewhat like normal text (not just single characters)
        avg_word_len = sum(len(w) for w in words) / max(1, len(words))
        return avg_word_len > 2.0
        
    def test_repetition(self):
        """Checks for excessive repetition in generated text."""
        prompt = "The dragon breathed fire, and then"
        generated = self._generate_text(prompt, max_length=50)
        words = generated.split()
        
        # Find longest repeating n-gram (simple approximation: check unique bigrams)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        unique_bigrams = set(bigrams)
        
        # If bigrams repeat too much, it's failing
        repetition_ratio = 1.0 - (len(unique_bigrams) / max(1, len(bigrams)))
        return repetition_ratio < 0.5 # Less than 50% repetition
        
    def test_diversity(self):
        """Checks the ratio of unique n-grams (unigrams)."""
        prompt = "In the deep forests of Elvendale,"
        generated = self._generate_text(prompt, max_length=50)
        words = generated.split()
        unique_words = set(words)
        
        diversity_score = len(unique_words) / max(1, len(words))
        return diversity_score
        
    def test_prompt_completion(self):
        """Tests if the model provides reasonable completions given prompts."""
        prompts = ["The ancient wizard cast a spell that", "Sir Lancelot drew his sword and"]
        scores = []
        for p in prompts:
            generated = self._generate_text(p, max_length=15)
            # Just check it actually generates something longer than the prompt
            if len(generated) > len(p) + 5:
                scores.append(1)
            else:
                scores.append(0)
        return sum(scores) / len(scores)

    def run_all(self):
        """Runs all benchmarks and returns a dictionary of results."""
        return {
            "coherence_pass": self.test_coherence(),
            "repetition_pass": self.test_repetition(),
            "diversity_score": self.test_diversity(),
            "prompt_completion_score": self.test_prompt_completion()
        }
