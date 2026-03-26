# Adaptive Knowledge Intelligence System (AKIS)

A modular, production-grade Retrieval-Augmented Generation (RAG) pipeline for local, CLI-based document QA.

## Features
- PDF ingestion and cleaning
- Semantic chunking (200–500 words, paragraph/sentence-based)
- Embedding with sentence-transformers (bge-small)
- Vector storage with FAISS
- Top-k retrieval with metadata and similarity scores
- LLM answer generation via external API (context-constrained)
- CLI interface, no UI

## Project Structure
```
AKIS/
├── ingestion/loader.py
├── chunking/chunker.py
├── embeddings/vector_store.py
├── retrieval/retriever.py
├── generation/generator.py
└── main.py
```

## Setup & Run Instructions

1. **Clone the repo and enter the directory**
2. **Create a virtual environment:**
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # Or: source .venv/bin/activate  # On Linux/Mac
   ```
3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
4. **Run the pipeline:**
   ```
   python AKIS/main.py --pdf <path_to_pdf> --llm_api_url <API_URL> [--llm_api_key <API_KEY>]
   ```
   - Example: `python AKIS/main.py --pdf docs/sample.pdf --llm_api_url http://localhost:8000/generate`

5. **Interact via CLI:**
   - Enter queries at the prompt. Type `exit` to quit.

## Extensibility
- Modular code for easy extension (multi-query, validation, new chunkers, etc.)
- No hardcoded paths; all config via CLI

## Requirements
- Python 3.8+
- See `requirements.txt`

## Notes
- LLM API must accept a JSON payload: `{ "prompt": <prompt> }` and return `{ "answer": <answer> }`
- Answers are strictly limited to retrieved context; if not found, returns "Insufficient information."
