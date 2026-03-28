import os
import logging
from typing import List, Dict, Callable, Any
import requests

# --- Helper Functions ---
def build_context_string(chunks: List[Dict], max_tokens: int = 2048) -> str:
    """
    Combine chunk texts into a single context string, preserving chunk_id references.
    Truncate safely if context is too long.
    """
    context = ""
    token_count = 0
    for chunk in chunks:
        chunk_text = f"[chunk_id: {chunk.get('chunk_id', '-')}] {chunk.get('text', '')}\n"
        tokens = len(chunk_text.split())
        if token_count + tokens > max_tokens:
            break
        context += chunk_text
        token_count += tokens
    return context.strip()

def should_fallback(confidence: float, claims: List[Dict]) -> bool:
    """
    Decide if fallback is needed based on confidence and claim support.
    """
    if confidence < 50:
        return True
    unsupported = sum(1 for c in claims if not c.get('supported'))
    if unsupported > len(claims) // 2:
        return True
    return False

def safe_truncate_context(text: str, max_tokens: int = 2048) -> str:
    """
    Truncate context string to max_tokens words.
    """
    words = text.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens])
    return text

# --- LLM Call Abstractions ---
def generate_with_ollama(query: str, context: str) -> str:
    """
    Generate answer using local Ollama model.
    """
    model = os.getenv('OLLAMA_MODEL', 'llama3')
    try:
        # Example: POST to Ollama API (adjust as needed)
        payload = {
            "model": model,
            "prompt": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        }
        response = requests.post('http://localhost:11434/api/generate', json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except Exception as e:
        logging.error(f"Ollama call failed: {e}")
        return ""

def generate_with_gemini(query: str, context: str, strict: bool = False) -> str:
    """
    Generate answer using Gemini API.
    """
    api_key = os.getenv('AIzaSyA4C83L2kZLFBAoOt5cyFQJpADecuVlGl4')
    if not api_key:
        logging.error("GEMINI_API_KEY not set in environment.")
        return ""
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    if strict:
        prompt = ("Only use explicitly supported information from the context. Do not assume or infer.\n" + prompt)
    try:
        # Example Gemini API call (adjust endpoint as needed)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"prompt": prompt}
        response = requests.post('https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
                                 json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json().get('answer', '').strip()
    except Exception as e:
        logging.error(f"Gemini call failed: {e}")
        return ""

# --- Main Routing Function ---
def route_and_generate(
    query: str,
    context_chunks: List[Dict],
    retriever_output: str,
    validator_fn: Callable[[List[str], List[Dict]], List[Dict]],
    claim_splitter_fn: Callable[[str], List[str]],
    scorer_fn: Callable[[List[Dict]], float]
) -> Dict[str, Any]:
    """
    Route between Ollama and Gemini based on answer reliability.
    """
    logging.info("[LLM_ROUTER] Building context string...")
    context = build_context_string(context_chunks)
    context = safe_truncate_context(context)

    # --- Step 2: Primary Generation (Ollama) ---
    logging.info("[LLM_ROUTER] Generating answer with Ollama...")
    answer = generate_with_ollama(query, context)
    if not answer:
        logging.warning("[LLM_ROUTER] Ollama returned empty answer. Triggering fallback.")
        status = "FALLBACK_USED"
        model_used = "gemini"
        # Fallback immediately
        answer = generate_with_gemini(query, context, strict=True)
        if not answer:
            return {
                "answer": "Insufficient information in provided documents",
                "confidence": 0.0,
                "claims": [],
                "model_used": "gemini",
                "status": "FAILED"
            }
        claims = claim_splitter_fn(answer)
        if not claims:
            return {
                "answer": "Insufficient information in provided documents",
                "confidence": 0.0,
                "claims": [],
                "model_used": "gemini",
                "status": "FAILED"
            }
        results = validator_fn(claims, context_chunks)
        confidence = scorer_fn(results)
        logging.info(f"[LLM_ROUTER] Gemini confidence: {confidence}")
        if confidence < 50:
            return {
                "answer": "Insufficient information in provided documents",
                "confidence": confidence,
                "claims": results,
                "model_used": "gemini",
                "status": "FAILED"
            }
        return {
            "answer": answer,
            "confidence": confidence,
            "claims": results,
            "model_used": "gemini",
            "status": "FALLBACK_USED"
        }

    # --- Step 3: Validation Pipeline ---
    claims = claim_splitter_fn(answer)
    if not claims:
        logging.warning("[LLM_ROUTER] No claims extracted from Ollama answer. Triggering fallback.")
        status = "FALLBACK_USED"
        model_used = "gemini"
        answer = generate_with_gemini(query, context, strict=True)
        if not answer:
            return {
                "answer": "Insufficient information in provided documents",
                "confidence": 0.0,
                "claims": [],
                "model_used": "gemini",
                "status": "FAILED"
            }
        claims = claim_splitter_fn(answer)
        if not claims:
            return {
                "answer": "Insufficient information in provided documents",
                "confidence": 0.0,
                "claims": [],
                "model_used": "gemini",
                "status": "FAILED"
            }
        results = validator_fn(claims, context_chunks)
        confidence = scorer_fn(results)
        logging.info(f"[LLM_ROUTER] Gemini confidence: {confidence}")
        if confidence < 50:
            return {
                "answer": "Insufficient information in provided documents",
                "confidence": confidence,
                "claims": results,
                "model_used": "gemini",
                "status": "FAILED"
            }
        return {
            "answer": answer,
            "confidence": confidence,
            "claims": results,
            "model_used": "gemini",
            "status": "FALLBACK_USED"
        }

    results = validator_fn(claims, context_chunks)
    confidence = scorer_fn(results)
    logging.info(f"[LLM_ROUTER] Ollama confidence: {confidence}")

    # --- Step 4: Decision Logic ---
    if should_fallback(confidence, results):
        logging.info(f"[LLM_ROUTER] Fallback triggered. Confidence: {confidence}")
        answer2 = generate_with_gemini(query, context, strict=True)
        if not answer2:
            return {
                "answer": "Insufficient information in provided documents",
                "confidence": confidence,
                "claims": results,
                "model_used": "gemini",
                "status": "FAILED"
            }
        claims2 = claim_splitter_fn(answer2)
        if not claims2:
            return {
                "answer": "Insufficient information in provided documents",
                "confidence": 0.0,
                "claims": [],
                "model_used": "gemini",
                "status": "FAILED"
            }
        results2 = validator_fn(claims2, context_chunks)
        confidence2 = scorer_fn(results2)
        logging.info(f"[LLM_ROUTER] Gemini confidence: {confidence2}")
        if confidence2 < 50:
            return {
                "answer": "Insufficient information in provided documents",
                "confidence": confidence2,
                "claims": results2,
                "model_used": "gemini",
                "status": "FAILED"
            }
        return {
            "answer": answer2,
            "confidence": confidence2,
            "claims": results2,
            "model_used": "gemini",
            "status": "FALLBACK_USED"
        }
    # --- Step 7: Final Decision ---
    status = "SUCCESS" if confidence >= 80 else "FALLBACK_USED"
    return {
        "answer": answer,
        "confidence": confidence,
        "claims": results,
        "model_used": "ollama",
        "status": status
    }
