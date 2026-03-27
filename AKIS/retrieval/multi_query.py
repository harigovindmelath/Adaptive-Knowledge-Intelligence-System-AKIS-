import logging
from .retriever import Retriever

def expand_queries(query, llm_expand_fn=None, n=3):
    """
    Generate alternative queries using an LLM or fallback heuristic.
    """
    if llm_expand_fn:
        try:
            return llm_expand_fn(query, n)
        except Exception as e:
            logging.warning(f"LLM expansion failed: {e}")
    # Fallback: simple heuristic
    return [query] + [f"{query} (alt {i})" for i in range(1, n)]

def multi_query_retrieve(query, retriever: Retriever, top_k=5, llm_expand_fn=None):
    """
    Retrieve top-k chunks for each expanded query, deduplicate, preserve metadata.
    """
    queries = expand_queries(query, llm_expand_fn)
    all_chunks = []
    seen_ids = set()
    for q in queries:
        try:
            chunks = retriever.retrieve(q, top_k=top_k)
            for chunk in chunks:
                if chunk['chunk_id'] not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk['chunk_id'])
        except Exception as e:
            logging.error(f"Retrieval failed for query '{q}': {e}")
    return all_chunks
