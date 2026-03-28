"""
Retriever for AKIS
"""
from typing import List, Dict, Tuple
from AKIS.embeddings.vector_store import VectorStore


from typing import List, Dict

class Retriever:
    """
    Retriever for AKIS. Supports configurable top_k retrieval.
    """
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top_k most relevant chunks for the query.
        Returns a list of dicts with chunk_id, text, and source.
        """
        results = self.vector_store.search(query, top_k)
        output = []
        for res in results:
            # Support both (text, meta, score) and dict formats
            if isinstance(res, tuple):
                text, meta, *_ = res
                output.append({
                    "chunk_id": meta.get("chunk_id", "-"),
                    "text": text,
                    "source": meta.get("source_file", meta.get("source", "-"))
                })
            elif isinstance(res, dict):
                output.append({
                    "chunk_id": res.get("chunk_id", "-"),
                    "text": res.get("text", ""),
                    "source": res.get("source_file", res.get("source", "-"))
                })
        return output
