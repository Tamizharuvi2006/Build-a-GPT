"""Flat vector store for batch processing and retrieval."""
import torch
from typing import List, Tuple, Dict, Any

class VectorStore:
    """Simple flat vector store using PyTorch."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = None
        self.metadata: List[Dict[str, Any]] = []
        
    def add(self, vectors: torch.Tensor, metadata: List[Dict[str, Any]]):
        """Add a batch of vectors and corresponding metadata."""
        if vectors.shape[0] != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries.")
        if vectors.shape[-1] != self.dim:
            raise ValueError(f"Vector dimension must be {self.dim}")
            
        vectors = vectors.detach().cpu()
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = torch.cat([self.vectors, vectors], dim=0)
            
        self.metadata.extend(metadata)
        
    def search(self, query_vector: torch.Tensor, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search nearest neighbors using cosine similarity."""
        if self.vectors is None or len(self.metadata) == 0:
            return []
            
        query_vector = query_vector.view(1, -1)
        norm_query = torch.nn.functional.normalize(query_vector, p=2, dim=-1)
        norm_vectors = torch.nn.functional.normalize(self.vectors, p=2, dim=-1)
        
        similarities = torch.matmul(norm_query, norm_vectors.transpose(0, 1)).squeeze(0)
        
        top_k = min(top_k, similarities.size(0))
        scores, indices = torch.topk(similarities, top_k)
        
        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            results.append((self.metadata[idx], score))
        return results
