"""
Main CLI for AKIS
"""
import argparse
import os
from AKIS.ingestion.loader import load_pdf
from AKIS.chunking.chunker import chunk_text
from AKIS.embeddings.vector_store import VectorStore
from AKIS.retrieval.retriever import Retriever
from AKIS.retrieval.multi_query import multi_query_retrieve
from AKIS.validation.claim_splitter import split_into_claims
from AKIS.validation.verifier import verify_claims
from AKIS.validation.scorer import compute_confidence
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
        print(f"[AKIS] Retrieving relevant chunks (multi-query)...")
        # Use multi-query retrieval, returns list of dicts with chunk_id, text, source_file
        retrieved_chunks = multi_query_retrieve(query, retriever, top_k=5)
        print("[AKIS] Retrieved Chunks:")
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"  Chunk {i}: id={chunk['chunk_id']}")
            print(f"    {chunk['text'][:200]}{'...' if len(chunk['text']) > 200 else ''}")
        print(f"[AKIS] Generating answer...")
        # For backward compatibility, convert to (text, meta, score) tuples with dummy score
        answer_input = [(c['text'], c, 0.0) for c in retrieved_chunks]
        try:
            answer = generator.generate(answer_input, query)
        except Exception as e:
            print(f"[AKIS] Error from LLM: {e}")
            answer = "Insufficient information."
        print(f"\n[AKIS] Answer: {answer}")

        # Analyze answer: split into claims, verify, score
        claims = split_into_claims(answer)
        verifications = verify_claims(claims, retrieved_chunks)
        confidence = compute_confidence(verifications)
        print(f"[AKIS] Confidence: {confidence}%")
        print("[AKIS] Claim Analysis:")
        for v in verifications:
            status = "SUPPORTED" if v['supported'] else "UNSUPPORTED"
            src = v['source'] if v['source'] else "-"
            print(f"  - {status}: {v['claim']} (source: {src})")

if __name__ == "__main__":
    main()
