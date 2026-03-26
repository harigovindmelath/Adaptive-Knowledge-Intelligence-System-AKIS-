"""
Main CLI for AKIS
"""
import argparse
import os
from AKIS.ingestion.loader import load_pdf
from AKIS.chunking.chunker import chunk_text
from AKIS.embeddings.vector_store import VectorStore
from AKIS.retrieval.retriever import Retriever
from AKIS.generation.generator import Generator


def main():
    parser = argparse.ArgumentParser(description="Adaptive Knowledge Intelligence System (AKIS)")
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--llm_api_url', required=True, help='LLM API endpoint URL')
    parser.add_argument('--llm_api_key', default=None, help='LLM API key (if needed)')
    args = parser.parse_args()

    print(f"[AKIS] Loading PDF: {args.pdf}")
    text = load_pdf(args.pdf)
    print(f"[AKIS] Chunking text...")
    chunks = chunk_text(text, os.path.basename(args.pdf))
    print(f"[AKIS] {len(chunks)} chunks created.")

    print(f"[AKIS] Generating embeddings and building vector store...")
    vs = VectorStore()
    vs.add_chunks(chunks)
    retriever = Retriever(vs)
    generator = Generator(args.llm_api_url, args.llm_api_key)

    while True:
        query = input("\n[AKIS] Enter your query (or 'exit' to quit): ").strip()
        if query.lower() in ("exit", "quit"): break
        print(f"[AKIS] Retrieving relevant chunks...")
        retrieved = retriever.retrieve(query)
        print("[AKIS] Retrieved Chunks:")
        for i, (text, meta, score) in enumerate(retrieved, 1):
            print(f"  Chunk {i}: id={meta['chunk_id']} score={score:.4f}")
            print(f"    {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"[AKIS] Generating answer...")
        try:
            answer = generator.generate(retrieved, query)
        except Exception as e:
            print(f"[AKIS] Error from LLM: {e}")
            answer = "Insufficient information."
        print(f"\n[AKIS] Answer: {answer}\n")

if __name__ == "__main__":
    main()
