import re
import logging

def split_into_claims(answer_text):
    """
    Split answer into atomic claims (simple heuristic: split by sentences).
    """
    try:
        claims = re.split(r'(?<=[.!?])\s+', answer_text.strip())
        return [c for c in claims if c]
    except Exception as e:
        logging.error(f"Claim splitting failed: {e}")
        return []
