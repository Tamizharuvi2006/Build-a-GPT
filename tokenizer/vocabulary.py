"""Vocabulary mapping for tokens."""
import json
from typing import List

class Vocabulary:
    """Manages token-to-ID mappings."""
    
    def __init__(self, special_tokens: List[str] = None):
        self.special_tokens = special_tokens or ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
        self.token2id = {}
        self.id2token = {}
        
        for i, token in enumerate(self.special_tokens):
            self.token2id[token] = i
            self.id2token[i] = token
            
    def build_from_tokenizer(self, tokenizer_path: str):
        """Load vocabulary from a HuggingFace-style tokenizer.json."""
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        model = data.get("model", {})
        vocab = model.get("vocab", {})
        
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            if token not in self.token2id:
                new_idx = len(self.token2id)
                self.token2id[token] = new_idx
                self.id2token[new_idx] = token
                
    def token_to_id(self, token: str) -> int:
        return self.token2id.get(token, self.unk_id)
        
    def id_to_token(self, idx: int) -> str:
        return self.id2token.get(idx, '[UNK]')
        
    def __len__(self) -> int:
        return len(self.token2id)
        
    @property
    def pad_id(self) -> int:
        return self.token2id.get('[PAD]', 0)
        
    @property
    def unk_id(self) -> int:
        return self.token2id.get('[UNK]', 1)
        
    @property
    def bos_id(self) -> int:
        return self.token2id.get('[BOS]', 2)
        
    @property
    def eos_id(self) -> int:
        return self.token2id.get('[EOS]', 3)
