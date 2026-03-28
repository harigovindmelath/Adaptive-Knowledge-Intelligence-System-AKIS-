
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import threading
import re

SUPPORTED_THRESHOLD = 0.65
WEAK_THRESHOLD = 0.60
TOP_N = 3

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
    Normalize text for embedding: lower, strip, remove extra spaces, symbols, and line breaks.
    """
    text = text.lower().strip()
    text = re.sub(r"cid:\d+", "", text)  # remove PDF artifact tags
    text = re.sub(r"[^\w\s.,-]", "", text)  # remove most non-alphanum except .,-
    text = re.sub(r"\s+", " ", text)
    return text

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def get_top_n_matches(claim_emb: np.ndarray, chunk_embs: np.ndarray, n: int = TOP_N) -> List[tuple]:
    """
    Return list of (idx, similarity) for top-n most similar chunks.
    """
    sims = np.dot(chunk_embs, claim_emb) / (np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(claim_emb) + 1e-8)
    top_indices = np.argsort(sims)[-n:][::-1]
    return [(int(idx), float(sims[idx])) for idx in top_indices]

# --- Main Verification Function ---

def verify_claims_semantic(
    claims: List[str],
    context_chunks: List[Dict],
    model: Optional[SentenceTransformer] = None
) -> List[Dict]:
    """
    Semantic claim verification using embeddings. Handles paraphrasing, noise, and partial support.
    Args:
        claims: List of claim strings.
        context_chunks: List of dicts with 'chunk_id', 'text', 'source'.
        model: Optional preloaded SentenceTransformer.
    Returns:
        List of dicts with claim, supported, confidence, source_chunk_id, support_level.
    """
    if not claims or not context_chunks:
        return [
            {
                "claim": c,
                "supported": False,
                "confidence": 0.0,
                "source_chunk_id": None,
                "support_level": "none"
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
    # Truncate and normalize chunk texts
    def truncate(text, max_words=256):
        words = text.split()
        return ' '.join(words[:max_words])
    chunk_texts = [normalize_text(truncate(c['text'])) for c in context_chunks]
    chunk_ids = [c['chunk_id'] for c in context_chunks]
    # Model
    model = model or get_model()
    chunk_embs = model.encode(chunk_texts, show_progress_bar=False, normalize_embeddings=True)
    claim_embs = model.encode([normalize_text(c) for c in norm_claims], show_progress_bar=False, normalize_embeddings=True)
    results = []
    for i, claim in enumerate(norm_claims):
        claim_emb = claim_embs[i]
        top_matches = get_top_n_matches(claim_emb, chunk_embs, n=TOP_N)
        # Find best above threshold
        best_idx, best_sim = -1, 0.0
        support_level = "none"
        supported = False
        for idx, sim in top_matches:
            if sim >= SUPPORTED_THRESHOLD:
                supported = True
                best_idx, best_sim = idx, sim
                support_level = "strong"
                break
            elif WEAK_THRESHOLD <= sim < SUPPORTED_THRESHOLD and not supported:
                supported = True
                best_idx, best_sim = idx, sim
                support_level = "weak"
        results.append({
            "claim": claim,
            "supported": supported,
            "confidence": round(best_sim, 3),
            "source_chunk_id": chunk_ids[best_idx] if supported and best_idx >= 0 else None,
            "support_level": support_level
        })
    return results
