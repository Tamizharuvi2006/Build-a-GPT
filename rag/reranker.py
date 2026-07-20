"""Result reranking for RAG."""
from typing import List
from .search import SearchResult

class Reranker:
    """Reranks search results based on simple keyword overlap."""
    
    def __init__(self, model=None):
        self.model = model
        
    def rerank(self, query: str, results: List[SearchResult], top_k: int = 3) -> List[SearchResult]:
        """Rerank results by counting query term occurrences."""
        if not results:
            return []
            
        query_terms = set(query.lower().split())
        
        scored_results = []
        for result in results:
            content_lower = result.chunk.lower()
            overlap_score = sum(1 for term in query_terms if term in content_lower)
            final_score = result.score + (overlap_score * 0.1)
            scored_results.append((final_score, result))
            
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [res for _, res in scored_results[:top_k]]
