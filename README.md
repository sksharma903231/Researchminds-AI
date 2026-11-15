[![ResearchMinds tests](https://github.com/sksharma903231/Researchminds-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/sksharma903231/Researchminds-AI/actions/workflows/tests.yml)

# ResearchMinds AI

ResearchMinds AI is a lightweight Retrieval-Augmented Generation (RAG) research assistant prototype for indexing, searching, and synthesizing insights from academic PDF collections.

It provides grounded question answering with document and page-level source references, guards against hallucination on unsupported questions, and supports both local (Ollama) and cloud (Groq) LLM inference.

---

## Architecture Overview

```
                      ┌──────────────────────┐
                      │   Academic PDFs      │
                      │   (data/papers)      │
                      └──────────┬───────────┘
                                 │
                     PyMuPDF Ingestion Pipeline
                                 │
                      ┌──────────▼───────────┐
                      │  Page-Aware Text &   │
                      │  Metadata Registry   │
                      └──────────┬───────────┘
                                 │
                     Recursive Chunking (per page)
                                 │
                      ┌──────────▼───────────┐
                      │  Chunks Registry     │
                      │  (with page numbers) │
                      └──────────┬───────────┘
                                 │
                BAAI/bge-small-en-v1.5 Embeddings
                                 │
                      ┌──────────▼───────────┐
                      │  FAISS Vector Index  │
                      │  & Mapping (Cosine)  │
                      └──────────┬───────────┘
                                 │
            Semantic Retrieval (Top-k + Page Metadata)
                                 │
         Strictly Grounded Prompt (Guards vs Hallucination)
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            Local Inference            Cloud Inference
           (Ollama / llama3)      (Groq / llama-3.1-8b-instant)
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                     Grounded Answer + Citations
                     (Streamlit UI / CLI Runner)
```

### Key Components

- **Ingestion (`src/ingestion/`)**:
  - `pdf_loader.py`: Extracts text while preserving page demarcations (`extract_pages_from_pdf`).
  - `metadata_extractor.py`: Safely extracts document metadata (title, author, page count, filename, paper ID).
  - `ingest_pipeline.py`: Orchestrates PDF ingestion, writing page-delimited texts and `papers_metadata.json`.
  - `chunking_pipeline.py`: Splits text by page boundaries with configurable overlap, ensuring every chunk retains its document title, paper ID, and exact page number.
- **Vector Store (`src/vectorstore/`)**:
  - `build_index.py`: Encodes text chunks with normalized embeddings using `BAAI/bge-small-en-v1.5` and persists a FAISS `IndexFlatIP` index and `index_mapping.json`. Supports dependency injection of embedding models for testability.
- **Retrieval (`src/retrieval/`)**:
  - `retriever.py`: Computes cosine similarity across indexed vectors, returning top-k matching passages with document titles, page numbers, match scores, and text.
- **Generation (`src/generation/`)**:
  - `generator.py`: Strictly grounds responses in the retrieved context using Ollama (`llama3`) or Groq (`llama-3.1-8b-instant`). Empty context or unsupported questions immediately return an explicit disclosure instead of fabricating facts.
- **User Interface (`src/ui/`, `app.py`, `app_public.py`, `main.py`)**:
  - `app.py`: Streamlit application with interactive model selector (Local Ollama vs Cloud Groq) and collapsible evidence viewer.
  - `app_public.py`: Cloud-focused Streamlit deployment.
  - `main.py`: Command-line interface for testing retrieval and generation directly.

---

## Prerequisites

- **Python**: 3.10 to 3.14
- **Local Mode (Optional)**: [Ollama](https://ollama.ai) installed and running with `ollama pull llama3`
- **Cloud Mode (Optional)**: Groq API key (`export GROQ_API_KEY="your-api-key"`)

---

## Installation & Setup

1. **Clone and enter the repository**:
   ```bash
   git clone <repo-url>
   cd Researchminds-AI
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   - For standard CPU environments:
     ```bash
     pip install -r requirements.txt
     pip install pytest pytest-cov
     ```
   - For lightweight testing (without GPU wheels):
     ```bash
     pip install PyMuPDF langchain-core langchain-text-splitters langchain-groq langchain-ollama streamlit faiss-cpu pytest pytest-cov
     ```

4. **Configure environment variables (for Cloud Groq mode)**:
   ```bash
   export GROQ_API_KEY="gsk_..."
   # Optional overrides:
   # export GROQ_MODEL="llama-3.1-8b-instant"
   # export OLLAMA_MODEL="llama3"
   ```

---

## Ingesting Papers & Building the Index

The repository includes a pre-indexed demo corpus of three academic papers in `data/papers/`. To re-index or add new PDFs:

1. Place PDF files into `data/papers/`.
2. Run the ingestion pipeline:
   ```bash
   python -m src.ingestion.ingest_pipeline
   ```
3. Run the chunking pipeline:
   ```bash
   python -m src.ingestion.chunking_pipeline
   ```
4. Build the FAISS vector index:
   ```bash
   python -m src.vectorstore.build_index
   ```

---

## Running the Application

### 1. Interactive Streamlit Interface
Run the dual-mode application (switchable between Local Ollama and Cloud Groq):
```bash
streamlit run app.py
```

Run the cloud-only application:
```bash
streamlit run app_public.py
```

### 2. Command-Line Runner
Run an end-to-end query from the terminal:
```bash
python main.py
```

---

## Automated Test Suite

ResearchMinds AI includes an automated unit test suite using lightweight mocks and fixtures. Tests execute completely offline without requiring paid API keys, GPU drivers, or large model downloads.

Run all tests:
```bash
pytest -v
```

Run tests with test coverage:
```bash
pytest --cov=src -v
```

---

## Scope & Limitations

This project is an exploratory research prototype focused on grounded RAG mechanics:

- **Corpus Scale**: Uses a local, in-memory FAISS flat index (`IndexFlatIP`). Ideal for small-to-medium corpora (tens to hundreds of research papers). Large-scale production deployments would require persistent vector databases (e.g. PostgreSQL with pgvector, Milvus, or Qdrant).
- **PDF Extraction**: Text extraction relies on PyMuPDF's layout text stream. Scanned documents without OCR text layers or complex multi-column figures are not processed with OCR.
- **Session State**: The Streamlit interface maintains ephemeral session history; there is no persistent multi-user database or chat session storage.
- **Citations**: Citations reference the document title/filename and page number extracted during ingestion. Full bibliographic parsing (BibTeX, DOI resolution) is not implemented.
