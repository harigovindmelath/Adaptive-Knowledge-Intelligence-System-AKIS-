"""
Answer generator for AKIS
"""
from typing import List, Dict, Tuple
import requests

class Generator:
    def __init__(self, api_url: str, api_key: str = None):
        self.api_url = api_url
        self.api_key = api_key

    def build_prompt(self, context_chunks: List[Tuple[str, Dict, float]], query: str) -> str:
        context = "\n\n".join([chunk[0] for chunk in context_chunks])
        prompt = (
            "You are an expert assistant. Use ONLY the provided context to answer the question. "
            "If the answer is not in the context, reply: 'Insufficient information.'\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        return prompt

    def generate(self, context_chunks: List[Tuple[str, Dict, float]], query: str) -> str:
        prompt = self.build_prompt(context_chunks, query)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"prompt": prompt}
        response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json().get("answer", "Insufficient information.")
