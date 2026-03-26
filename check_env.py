import sys

print("\n=== AKIS Environment Validation ===")

try:
    import pdfplumber
    print("[OK] pdfplumber import: SUCCESS")
except Exception as e:
    print(f"[FAIL] pdfplumber import: {e}")

try:
    import faiss
    print("[OK] faiss import: SUCCESS")
except Exception as e:
    print(f"[FAIL] faiss import: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("[OK] sentence-transformers import: SUCCESS")
except Exception as e:
    print(f"[FAIL] sentence-transformers import: {e}")

try:
    import torch
    print(f"[OK] torch import: SUCCESS (version: {torch.__version__})")
except Exception as e:
    print(f"[FAIL] torch import: {e}")

try:
    import requests
    print("[OK] requests import: SUCCESS")
except Exception as e:
    print(f"[FAIL] requests import: {e}")

print("\n=== Validation Complete ===\n")
