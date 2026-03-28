

import logging
logging.basicConfig(level=logging.INFO)
logging.info("Starting AKIS API...")

try:
    import fastapi, uvicorn, numpy, requests
    logging.info("All core dependencies loaded.")
except ImportError as e:
    logging.error(f"Dependency import failed: {e}")
    raise

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from AKIS.retrieval.retriever import Retriever
from AKIS.retrieval.multi_query import multi_query_retrieve
from AKIS.llm_router import route_and_generate
from AKIS.validation.claim_splitter import split_into_claims
from AKIS.validation.semantic_verifier import verify_claims_semantic
from AKIS.validation.scorer import compute_confidence
from AKIS.embeddings.vector_store import VectorStore
from AKIS.chunking.chunker import chunk_text
from AKIS.ingestion.loader import load_pdf

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class ClaimResult(BaseModel):
    claim: str
    supported: bool
    confidence: float
    source_chunk_id: Optional[str]

class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    source: str

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    model_used: str
    status: str
    claims: List[ClaimResult]
    sources: List[SourceChunk]


PDF_PATH = os.getenv("AKIS_DEFAULT_PDF", "./default.pdf")

pipeline = None




def build_akis_pipeline():
    """
    Build and return a callable pipeline object with a .run(query) method.
    Ingests and indexes the default.pdf at startup.
    """
    pdf_path = "default.pdf"  # use the correct file in project root
    if not os.path.exists(pdf_path):
        raise RuntimeError(f"Default PDF not found: {pdf_path}")
    text = load_pdf(pdf_path)
    print(f"[DEBUG] Extracted text length: {len(text)}")
    print(f"[DEBUG] First 300 chars: {text[:300]}")
    chunks = chunk_text(text, source_file=pdf_path)
    print(f"Chunks created: {len(chunks)}")
    if chunks:
        print(f"Sample chunk: {chunks[0]}")
    else:
        print("No chunks created!")
    vector_store = VectorStore()
    vector_store.add_chunks(chunks)
    print(f"FAISS index size: {vector_store.index.ntotal}")
    if vector_store.index.ntotal == 0:
        raise RuntimeError("Vector store is empty after indexing")
    retriever = Retriever(vector_store)
    # Test retrieval
    results = retriever.retrieve("Harigovind")
    print(f"Test retrieval for 'Harigovind': {results}")

    class AKISPipeline:
        def __init__(self, retriever, chunks, vector_store):
            self.retriever = retriever
            self.chunks = chunks
            self.vector_store = vector_store
        def run(self, query: str):
            retrieved_chunks = multi_query_retrieve(query, self.retriever, top_k=5)
            if not retrieved_chunks:
                return {
                    "answer": "No relevant context found.",
                    "confidence": 0.0,
                    "model_used": "-",
                    "status": "FAILED",
                    "claims": [],
                    "sources": []
                }
            def validator_fn(claims, context_chunks):
                return verify_claims_semantic(claims, context_chunks)
            def scorer_fn(results):
                return compute_confidence(results)
            result = route_and_generate(
                query=query,
                context_chunks=retrieved_chunks,
                retriever_output="",
                validator_fn=validator_fn,
                claim_splitter_fn=split_into_claims,
                scorer_fn=scorer_fn
            )
            sources = [
                {
                    "chunk_id": c.get("chunk_id", "-"),
                    "text": c.get("text", ""),
                    "source": c.get("source_file", c.get("source", "-"))
                } for c in retrieved_chunks
            ]
            result["sources"] = sources
            return result
    return AKISPipeline(retriever, chunks, vector_store)



@app.on_event("startup")
def initialize_system():
    global pipeline
    try:
        logging.info("Initializing AKIS pipeline...")
        pipeline = build_akis_pipeline()
        logging.info("AKIS pipeline initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize system: {e}")
        pipeline = None

# --- Health Check Endpoint ---
@app.get("/")

def health():
    global pipeline
    return {
        "status": "running",
        "pipeline_initialized": pipeline is not None
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    global pipeline
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if pipeline is None:
        raise HTTPException(status_code=500, detail="System not initialized.")
    result = pipeline.run(request.query)
    claims = [ClaimResult(**cl) for cl in result.get("claims", [])]
    sources = [SourceChunk(**src) for src in result.get("sources", [])]
    return QueryResponse(
        answer=result.get("answer", ""),
        confidence=result.get("confidence", 0.0),
        model_used=result.get("model_used", "-"),
        status=result.get("status", "FAILED"),
        claims=claims,
        sources=sources
    )
