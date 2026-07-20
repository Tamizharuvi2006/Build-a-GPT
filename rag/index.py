"""Vector index for storing and retrieving document embeddings."""
import torch
import pickle
from typing import List, Tuple
from .documents import Document

class VectorIndex:
    """Simple vector index using cosine similarity."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.embeddings = None
        self.documents = []
        
    def add(self, embeddings: torch.Tensor, documents: List[Document]):
        """Add embeddings and corresponding documents to the index."""
        if embeddings.shape[0] != len(documents):
            raise ValueError("Number of embeddings must match number of documents.")
            
        if self.embeddings is None:
            self.embeddings = embeddings.clone()
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
        self.documents.extend(documents)
        
    def search(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search the index using cosine similarity."""
        if self.embeddings is None or len(self.documents) == 0:
            return []
            
        query_embedding = query_embedding.unsqueeze(0) if query_embedding.dim() == 1 else query_embedding
        norm_query = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)
        norm_embeddings = torch.nn.functional.normalize(self.embeddings, p=2, dim=-1)
        
        similarities = torch.matmul(norm_query, norm_embeddings.transpose(0, 1)).squeeze(0)
        
        top_k = min(top_k, similarities.size(0))
        scores, indices = torch.topk(similarities, top_k)
        
        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            results.append((self.documents[idx], score))
        return results
        
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({"dim": self.dim, "embeddings": self.embeddings, "documents": self.documents}, f)
            
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.dim = data["dim"]
            self.embeddings = data["embeddings"]
            self.documents = data["documents"]
