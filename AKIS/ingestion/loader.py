"""
PDF Loader module for AKIS
"""
from typing import List
import re
import pdfplumber


def load_pdf(file_path: str) -> str:
    """
    Load a PDF file and return cleaned, normalized text.
    """
    with pdfplumber.open(file_path) as pdf:
        text = " ".join(page.extract_text() or "" for page in pdf.pages)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
