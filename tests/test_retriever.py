import json
from pathlib import Path
import pytest
import faiss
import numpy as np

from src.retrieval.retriever import DocumentRetriever


def test_retriever_search_preserves_metadata(tmp_path: Path, mock_embedding_model):
    vectorstore_dir = tmp_path / "vectorstore"
    vectorstore_dir.mkdir()

    dim = mock_embedding_model.dimension
    index = faiss.IndexFlatIP(dim)

    # Create two test chunks
    texts = [
        "LightRAG uses graph structures for low-level and high-level retrieval.",
        "Precise zero-shot dense retrieval without relevance labels.",
    ]
    vectors = mock_embedding_model.encode(texts)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    faiss.write_index(index, str(vectorstore_dir / "papers.index"))

    mapping = [
        {
            "chunk_id": "p1_c0",
            "paper_id": "p_lightrag",
            "title": "LightRAG: Simple and Fast RAG",
            "filename": "lightrag.pdf",
            "author": "Guo et al.",
            "page": 1,
            "text": texts[0],
        },
        {
            "chunk_id": "p2_c0",
            "paper_id": "p_zeroshot",
            "title": "Precise Zero-Shot Dense Retrieval",
            "filename": "zeroshot.pdf",
            "author": "Gao et al.",
            "page": 1,
            "text": texts[1],
        },
    ]
    with open(vectorstore_dir / "index_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f)

    retriever = DocumentRetriever(
        vectorstore_dir=str(vectorstore_dir),
        model=mock_embedding_model,
    )

    results = retriever.search("What is LightRAG?", top_k=2)

    assert len(results) == 2
    top = results[0]
    assert "score" in top
    assert "paper_id" in top
    assert "title" in top
    assert "page" in top
    assert "filename" in top
    assert "text" in top
    assert top["page"] == 1


def test_retriever_search_empty_query(tmp_path: Path, mock_embedding_model):
    dim = mock_embedding_model.dimension
    index = faiss.IndexFlatIP(dim)
    retriever = DocumentRetriever(
        vectorstore_dir=str(tmp_path),
        model=mock_embedding_model,
        index=index,
        mapping=[],
    )
    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_retriever_missing_files_raises(tmp_path: Path, mock_embedding_model):
    with pytest.raises(FileNotFoundError):
        DocumentRetriever(
            vectorstore_dir=str(tmp_path / "nonexistent_vectorstore"),
            model=mock_embedding_model,
        )
