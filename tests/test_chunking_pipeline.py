import json
from pathlib import Path

from src.ingestion.chunking_pipeline import parse_pages_from_text, run_chunking


def test_parse_pages_from_text():
    sample_text = """
--- PAGE 1 ---
Introduction to the paper. Here is page one content.
--- PAGE 2 ---
Methodology and results. Here is page two content.
"""
    pages = parse_pages_from_text(sample_text)
    assert len(pages) == 2
    assert pages[0][0] == 1
    assert "Introduction to the paper" in pages[0][1]
    assert pages[1][0] == 2
    assert "Methodology and results" in pages[1][1]


def test_parse_pages_without_headers():
    plain_text = "Just a raw string with no page delimiter."
    pages = parse_pages_from_text(plain_text)
    assert len(pages) == 1
    assert pages[0][0] == 1
    assert pages[0][1] == plain_text


def test_run_chunking_preserves_page_and_metadata(tmp_path: Path):
    # Setup temporary processed directory
    processed_dir = tmp_path / "processed"
    texts_dir = processed_dir / "texts"
    texts_dir.mkdir(parents=True)

    text_file = texts_dir / "p_test123.txt"
    text_content = """--- PAGE 1 ---
LightRAG graph based indexing enables retrieval over complex entities.
--- PAGE 2 ---
Evaluation shows LightRAG improves contextual recall and precision.
"""
    text_file.write_text(text_content, encoding="utf-8")

    registry = [
        {
            "paper_id": "p_test123",
            "title": "LightRAG Paper",
            "filename": "lightrag.pdf",
            "author": "Guo et al.",
            "pages": 2,
            "text_file_path": str(text_file),
        }
    ]
    reg_path = processed_dir / "papers_metadata.json"
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f)

    chunks = run_chunking(str(processed_dir), chunk_size=500, chunk_overlap=50)

    assert len(chunks) >= 2
    chunk1 = chunks[0]
    assert chunk1["paper_id"] == "p_test123"
    assert chunk1["title"] == "LightRAG Paper"
    assert chunk1["page"] == 1
    assert "LightRAG graph based indexing" in chunk1["text"]

    chunk2 = chunks[1]
    assert chunk2["page"] == 2
    assert "Evaluation shows" in chunk2["text"]

    # Verify saved chunks_registry.json
    output_registry_path = processed_dir / "chunks_registry.json"
    assert output_registry_path.exists()
    with open(output_registry_path, "r", encoding="utf-8") as f:
        saved_chunks = json.load(f)
    assert len(saved_chunks) == len(chunks)


def test_run_chunking_missing_registry(tmp_path: Path):
    empty_dir = tmp_path / "empty_processed"
    empty_dir.mkdir()
    result = run_chunking(str(empty_dir))
    assert result == []
