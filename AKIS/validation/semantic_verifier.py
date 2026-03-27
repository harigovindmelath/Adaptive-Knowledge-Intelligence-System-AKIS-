from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import threading
import re

# --- Model Loading (Singleton) ---
_model = None
_model_lock = threading.Lock()

def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and cache the embedding model (thread-safe singleton).
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(model_name)
    return _model

# --- Helper Functions ---
def normalize_text(text: str) -> str:
    """
    Normalize text for embedding (strip, lower, remove extra spaces).
    """
    return re.sub(r"\s+", " ", text.strip().lower())

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def get_best_match(claim_emb: np.ndarray, chunk_embs: np.ndarray) -> (int, float):
    """
    Return index and similarity of best matching chunk.
    """
    sims = np.dot(chunk_embs, claim_emb) / (np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(claim_emb) + 1e-8)
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])

# --- Main Verification Function ---
def verify_claims_semantic(
    claims: List[str],
    context_chunks: List[Dict],
    model: Optional[SentenceTransformer] = None
) -> List[Dict]:
    """
    Semantic claim verification using embeddings.
    Args:
        claims: List of claim strings.
        context_chunks: List of dicts with 'chunk_id', 'text', 'source'.
        model: Optional preloaded SentenceTransformer.
    Returns:
        List of dicts with claim, supported, confidence, source_chunk_id.
    """
    if not claims or not context_chunks:
        return [
            {
                "claim": c,
                "supported": False,
                "confidence": 0.0,
                "source_chunk_id": None
            } for c in claims
        ]
    # Normalize and deduplicate claims
    seen_claims = set()
    norm_claims = []
    for c in claims:
        norm = normalize_text(c)
        if norm not in seen_claims and norm:
            norm_claims.append(c)
            seen_claims.add(norm)
    # Truncate long chunks
    def truncate(text, max_words=256):
        words = text.split()
        return ' '.join(words[:max_words])
    chunk_texts = [truncate(c['text']) for c in context_chunks]
    chunk_ids = [c['chunk_id'] for c in context_chunks]
    # Model
    model = model or get_model()
    chunk_embs = model.encode(chunk_texts, show_progress_bar=False, normalize_embeddings=True)
    claim_embs = model.encode(norm_claims, show_progress_bar=False, normalize_embeddings=True)
    results = []
    for i, claim in enumerate(norm_claims):
        claim_emb = claim_embs[i]
        idx, sim = get_best_match(claim_emb, chunk_embs)
        if sim >= 0.75:
            supported = True
        elif 0.60 <= sim < 0.75:
            supported = True  # weak support, but still mark as supported (low confidence)
        else:
            supported = False
        results.append({
            "claim": claim,
            "supported": supported,
            "confidence": round(sim, 3),
            "source_chunk_id": chunk_ids[idx] if supported else None
        })
    return results
