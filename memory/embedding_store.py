"""Key-value embedding store for generic memory tasks."""
import torch
import pickle
from typing import List, Tuple

class EmbeddingStore:
    """Store and retrieve embeddings by string key."""
    
    def __init__(self, dim: int, max_entries: int = 10000):
        self.dim = dim
        self.max_entries = max_entries
        self.store = {}
        self.embeddings_tensor = None
        self.keys = []
        
    def add(self, key: str, embedding: torch.Tensor):
        """Add or update an embedding by key."""
        if embedding.shape[-1] != self.dim:
            raise ValueError(f"Embedding dimension must be {self.dim}")
            
        embedding = embedding.detach().cpu().view(1, -1)
        
        if key in self.store:
            idx = self.store[key]
            self.embeddings_tensor[idx] = embedding
        else:
            if len(self.keys) >= self.max_entries:
                oldest_key = self.keys.pop(0)
                del self.store[oldest_key]
                self.embeddings_tensor = self.embeddings_tensor[1:]
                for i, k in enumerate(self.keys):
                    self.store[k] = i
                    
            if self.embeddings_tensor is None:
                self.embeddings_tensor = embedding
            else:
                self.embeddings_tensor = torch.cat([self.embeddings_tensor, embedding], dim=0)
                
            self.keys.append(key)
            self.store[key] = len(self.keys) - 1
            
    def get(self, key: str) -> torch.Tensor:
        """Get an embedding by key."""
        if key not in self.store:
            return None
        return self.embeddings_tensor[self.store[key]]
        
    def search(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for top-k nearest keys by cosine similarity."""
        if self.embeddings_tensor is None or len(self.keys) == 0:
            return []
            
        query_embedding = query_embedding.view(1, -1)
        norm_query = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)
        norm_embeddings = torch.nn.functional.normalize(self.embeddings_tensor, p=2, dim=-1)
        
        similarities = torch.matmul(norm_query, norm_embeddings.transpose(0, 1)).squeeze(0)
        
        top_k = min(top_k, similarities.size(0))
        scores, indices = torch.topk(similarities, top_k)
        
        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            results.append((self.keys[idx], score))
        return results
        
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                "dim": self.dim,
                "max_entries": self.max_entries,
                "store": self.store,
                "keys": self.keys,
                "embeddings_tensor": self.embeddings_tensor
            }, f)
            
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.dim = data["dim"]
            self.max_entries = data["max_entries"]
            self.store = data["store"]
            self.keys = data["keys"]
            self.embeddings_tensor = data["embeddings_tensor"]
