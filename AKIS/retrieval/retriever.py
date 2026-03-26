"""
Retriever for AKIS
"""
from typing import List, Dict, Tuple
from AKIS.embeddings.vector_store import VectorStore

class Retriever:
    def __init__(self, vector_store: VectorStore, top_k: int = 3):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Tuple[str, Dict, float]]:
        return self.vector_store.search(query, self.top_k)
