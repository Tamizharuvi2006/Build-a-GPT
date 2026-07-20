"""Semantic search interface using VectorIndex."""
from dataclasses import dataclass
from typing import List, Any
from .documents import Document
from .index import VectorIndex

@dataclass
class SearchResult:
    """Represents a search result from the index."""
    document: Document
    score: float
    chunk: str

class SemanticSearch:
    """High-level semantic search wrapper."""
    
    def __init__(self, index: VectorIndex, embedder: Any):
        self.index = index
        self.embedder = embedder
        
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search the index for the given query."""
        query_embedding = self.embedder(query)
        results = self.index.search(query_embedding, top_k=top_k)
        
        search_results = []
        for doc, score in results:
            search_results.append(SearchResult(
                document=doc,
                score=score,
                chunk=doc.content
            ))
        return search_results
