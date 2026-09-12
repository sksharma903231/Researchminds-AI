# ResearchMinds AI

ResearchMinds AI is a V1 Retrieval-Augmented Generation (RAG) prototype for
querying a local collection of academic PDFs.

## Architecture

- Streamlit user interface
- PyMuPDF PDF text extraction
- RecursiveCharacterTextSplitter chunking
- `BAAI/bge-small-en-v1.5` embeddings through SentenceTransformers
- FAISS local vector index
- LangChain generation with Ollama (`llama3`) or Groq (`openai/gpt-oss-20b`)

## Prerequisites

- Python 3.10 or later
- Ollama with the `llama3` model for local mode
- `GROQ_API_KEY` for cloud mode

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For cloud mode, set the API key in your shell without committing it:

```bash
export GROQ_API_KEY="..."
```

## Ingest papers and build the index

Place PDFs in `data/papers`, then run these commands from the repository root:

```bash
python -m src.ingestion.ingest_pipeline
python -m src.ingestion.chunking_pipeline
python -m src.vectorstore.build_index
```

The pipeline writes extracted text and registries to `data/processed` and writes
the FAISS index and its mapping to `data/vectorstore`.

## Run the application

Run the selectable local/cloud application:

```bash
streamlit run app.py
```

Run the cloud-only application:

```bash
streamlit run app_public.py
```

## Current limitations

- Retrieval uses a local FAISS index built from the PDFs in `data/papers`.
- Answers are constrained to the retrieved chunks and show paper IDs rather than
  formal citations.
- The project has no automated test suite.

The tracked papers, extracted text, and vector index are the current demo
corpus. Newly generated logs and processed/index artifacts are ignored by Git.
