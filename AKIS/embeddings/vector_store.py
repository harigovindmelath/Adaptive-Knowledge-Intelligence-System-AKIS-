"""
Vector store using FAISS and bge-small
"""
from typing import List, Dict, Tuple
import faiss
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.data = []  # List of (chunk_text, metadata)

    def add_chunks(self, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.data.extend([(c["text"], c) for c in chunks])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        query_emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(query_emb, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.data):
                text, meta = self.data[idx]
                results.append((text, meta, float(dist)))
        return results
