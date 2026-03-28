"""
PDF Loader module for AKIS
"""
from typing import List
import re
import pdfplumber



def clean_text(text: str) -> str:
    """
    Clean noisy extracted PDF text for chunking and embedding.
    Steps:
    1. Remove OCR artifacts (cid:1234)
    2. Fix broken words (lowercase→Uppercase)
    3. Normalize whitespace
    4. Remove special noise (weird symbols, excessive punctuation, non-ASCII)
    5. Lowercase normalization
    6. Preserve structure (avoid over-cleaning)
    """
    # 1. Remove OCR artifacts like (cid:1234)
    text = re.sub(r'\(cid:\d+\)', '', text)
    # 2. Fix broken words: insert space between lowercase-uppercase transitions
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # 3. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # 4. Remove special noise: weird symbols, excessive punctuation, non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII
    text = re.sub(r'[\u200b\ufeff]', '', text)  # invisible unicode
    text = re.sub(r'[\!\?\^\*\~\=\_\|\[\]<>\$%#@`\\]', '', text)  # excessive punctuation/symbols
    text = re.sub(r'\.{4,}', '...', text)  # collapse long runs of dots
    # 5. Lowercase normalization
    text = text.lower()
    # 6. Strip leading/trailing whitespace
    text = text.strip()
    # 7. Debug output
    print("Cleaned text sample:", text[:500])
    return text

def load_pdf(file_path: str) -> str:
    """
    Load a PDF file and return cleaned, normalized text.
    """
    with pdfplumber.open(file_path) as pdf:
        text = " ".join(page.extract_text() or "" for page in pdf.pages)
    text = clean_text(text)
    return text
