import json
from pathlib import Path

import faiss

from src.vectorstore.build_index import build_faiss_index


def test_build_faiss_index_with_mock_model(tmp_path: Path, mock_embedding_model):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    vectorstore_dir = tmp_path / "vectorstore"

    chunks = [
        {
            "chunk_id": "p1_c0",
            "paper_id": "p1",
            "title": "Paper One",
            "filename": "p1.pdf",
            "page": 1,
            "text": "Vector database indexing with FAISS is fast and efficient.",
        },
        {
            "chunk_id": "p1_c1",
            "paper_id": "p1",
            "title": "Paper One",
            "filename": "p1.pdf",
            "page": 2,
            "text": "Dense retrieval models map queries and documents to a shared space.",
        },
    ]

    chunks_file = processed_dir / "chunks_registry.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    success = build_faiss_index(
        processed_dir=str(processed_dir),
        vectorstore_dir=str(vectorstore_dir),
        model=mock_embedding_model,
    )

    assert success is True
    index_file = vectorstore_dir / "papers.index"
    mapping_file = vectorstore_dir / "index_mapping.json"

    assert index_file.exists()
    assert mapping_file.exists()

    loaded_index = faiss.read_index(str(index_file))
    assert loaded_index.ntotal == 2

    with open(mapping_file, "r", encoding="utf-8") as f:
        loaded_mapping = json.load(f)
    assert len(loaded_mapping) == 2
    assert loaded_mapping[0]["page"] == 1
    assert loaded_mapping[1]["page"] == 2


def test_build_faiss_index_missing_registry(tmp_path: Path, mock_embedding_model):
    empty_dir = tmp_path / "nonexistent"
    success = build_faiss_index(
        processed_dir=str(empty_dir),
        vectorstore_dir=str(tmp_path / "out"),
        model=mock_embedding_model,
    )
    assert success is False
