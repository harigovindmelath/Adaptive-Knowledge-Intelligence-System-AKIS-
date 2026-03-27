from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from AKIS.retrieval.retriever import Retriever
from AKIS.retrieval.multi_query import multi_query_retrieve
from AKIS.llm_router import route_and_generate
from AKIS.validation.claim_splitter import split_into_claims
from AKIS.validation.semantic_verifier import verify_claims_semantic
from AKIS.validation.scorer import compute_confidence
from AKIS.embeddings.vector_store import VectorStore
from AKIS.chunking.chunker import chunk_text
from AKIS.ingestion.loader import load_pdf
import os
import logging

app = FastAPI()

# --- Pydantic Models ---
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

# --- Load/Cache VectorStore (for demo: load once from a default PDF) ---
# In production, this should be persistent or multi-document aware
PDF_PATH = os.getenv("AKIS_DEFAULT_PDF", "./default.pdf")
vector_store = None
chunks = []
retriever = None

@app.on_event("startup")
def startup_event():
    global vector_store, chunks, retriever
    if not os.path.exists(PDF_PATH):
        logging.error(f"Default PDF not found: {PDF_PATH}")
        return
    text = load_pdf(PDF_PATH)
    chunks = chunk_text(text, os.path.basename(PDF_PATH))
    vector_store = VectorStore()
    vector_store.add_chunks(chunks)
    retriever = Retriever(vector_store)

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if retriever is None or not chunks:
        raise HTTPException(status_code=500, detail="System not initialized.")
    # --- Retrieval ---
    retrieved_chunks = multi_query_retrieve(request.query, retriever, top_k=5)
    if not retrieved_chunks:
        return QueryResponse(
            answer="No relevant context found.",
            confidence=0.0,
            model_used="-",
            status="FAILED",
            claims=[],
            sources=[]
        )
    # --- Routing + Generation + Validation ---
    def validator_fn(claims, context_chunks):
        return verify_claims_semantic(claims, context_chunks)
    def scorer_fn(results):
        return compute_confidence(results)
    result = route_and_generate(
        query=request.query,
        context_chunks=retrieved_chunks,
        retriever_output="",
        validator_fn=validator_fn,
        claim_splitter_fn=split_into_claims,
        scorer_fn=scorer_fn
    )
    # --- Format sources ---
    sources = [SourceChunk(
        chunk_id=c.get("chunk_id", "-"),
        text=c.get("text", ""),
        source=c.get("source_file", c.get("source", "-"))
    ) for c in retrieved_chunks]
    # --- Format claims ---
    claims = [ClaimResult(**cl) for cl in result.get("claims", [])]
    return QueryResponse(
        answer=result.get("answer", ""),
        confidence=result.get("confidence", 0.0),
        model_used=result.get("model_used", "-"),
        status=result.get("status", "FAILED"),
        claims=claims,
        sources=sources
    )
