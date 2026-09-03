# ResearchMinds AI 📚

An end-to-end Retrieval-Augmented Generation (RAG) architecture designed to ingest, vectorize, and query academic research papers. 

This system extracts semantic context using dense vector retrieval and synthesizes highly accurate, source-cited answers using Llama 3. It features a dual-mode generation pipeline, allowing execution entirely offline via local hardware or through high-speed cloud inference.

## 🏗️ Architecture & Tech Stack

*   **Frontend Interface:** Streamlit
*   **Vector Database:** FAISS (Local dense vector indexing)
*   **Embedding Model:** `BAAI/bge-small-en-v1.5` (via SentenceTransformers)
*   **LLM Orchestration:** LangChain
*   **Generation Engine (Local):** Ollama (Llama 3 8B)
*   **Generation Engine (Cloud):** Groq API (`llama-3.1-8b-instant`)

## ⚙️ How It Works

1.  **Ingestion & Chunking:** PyMuPDF extracts text from academic PDFs, which is recursively split into semantic chunks while maintaining paragraph overlap.
2.  **Vectorization:** Text chunks are mapped into 384-dimensional mathematical space and indexed into FAISS.
3.  **Retrieval:** User queries are vectorized and compared against the database to extract the top-K matching semantic contexts.
4.  **Synthesis:** The retrieved context blocks (along with their original source IDs) are injected into a strict prompt template and fed to the LLM to generate a grounded answer.

## 🚀 Quickstart (Local Deployment)

### 1. Prerequisites
*   Ubuntu 24.04 LTS (Tested environment)
*   Python 3.10+
*   [Ollama](https://ollama.com/) installed and running locally.

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/yourusername/Researchminds-AI.git](https://github.com/yourusername/Researchminds-AI.git)
cd Researchminds-AI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt