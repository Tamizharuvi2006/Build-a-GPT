"""Simple from-scratch Byte-Pair Encoding (BPE) tokenizer."""
import re
from collections import defaultdict
from typing import List, Set, Tuple

class SimpleBPE:
    """Educational BPE implementation."""
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        
    def _get_pairs(self, word: tuple) -> Set[Tuple[str, str]]:
        """Get set of adjacent pairs in a word."""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs
        
    def _merge(self, word: tuple, pair: Tuple[str, str]) -> tuple:
        """Merge all occurrences of pair in word."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)
        
    def train(self, text: str):
        """Train BPE on text data."""
        words = text.strip().split()
        word_freq = defaultdict(int)
        for w in words:
            chars = tuple(list(w) + ['</w>'])
            word_freq[chars] += 1
            
        vocab_size_current = len(set(char for chars in word_freq.keys() for char in chars))
        num_merges = self.vocab_size - vocab_size_current
        
        for i in range(num_merges):
            pairs = defaultdict(int)
            for word, freq in word_freq.items():
                for pair in self._get_pairs(word):
                    pairs[pair] += freq
                    
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            self.merges[best_pair] = "".join(best_pair)
            
            new_word_freq = {}
            for word, freq in word_freq.items():
                new_word = self._merge(word, best_pair)
                new_word_freq[new_word] = freq
            word_freq = new_word_freq
            
        # Build final vocabulary mapping
        idx = 0
        for word in word_freq.keys():
            for token in word:
                if token not in self.vocab:
                    self.vocab[token] = idx
                    idx += 1
                    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        words = text.strip().split()
        token_ids = []
        
        for w in words:
            chars = tuple(list(w) + ['</w>'])
            
            for pair, merged in self.merges.items():
                chars = self._merge(chars, pair)
                
            for token in chars:
                token_ids.append(self.vocab.get(token, 0))
        return token_ids
        
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [reverse_vocab.get(i, '') for i in ids]
        text = "".join(tokens).replace('</w>', ' ')
        return text.strip()
