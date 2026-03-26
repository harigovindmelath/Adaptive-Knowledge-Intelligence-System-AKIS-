"""
Semantic chunker for AKIS
"""
from typing import List, Dict
import re
import uuid

def chunk_text(text: str, source_file: str, target_words: int = 350) -> List[Dict]:
    """
    Split text into coherent chunks (200-500 words) by paragraphs/sentences.
    Attach metadata: chunk_id, source_file.
    """
    # Split by paragraphs, fallback to sentences if needed
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    current_chunk = ""
    current_word_count = 0
    for para in paragraphs:
        words = para.split()
        if current_word_count + len(words) > target_words and current_chunk:
            chunk_id = str(uuid.uuid4())
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
                "source_file": source_file
            })
            current_chunk = ""
            current_word_count = 0
        current_chunk += " " + para
        current_word_count += len(words)
    if current_chunk.strip():
        chunk_id = str(uuid.uuid4())
        chunks.append({
            "chunk_id": chunk_id,
            "text": current_chunk.strip(),
            "source_file": source_file
        })
    return chunks
