import json
from pathlib import Path
from unittest.mock import patch

from src.ingestion.ingest_pipeline import run_ingestion


def test_run_ingestion_nonexistent_dir(tmp_path: Path):
    registry = run_ingestion(str(tmp_path / "nonexistent"), str(tmp_path / "processed"))
    assert registry == []


def test_run_ingestion_empty_dir(tmp_path: Path):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    processed_dir = tmp_path / "processed"

    registry = run_ingestion(str(papers_dir), str(processed_dir))
    assert registry == []


def test_run_ingestion_with_mocked_extractor(tmp_path: Path):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_file = papers_dir / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy content")

    processed_dir = tmp_path / "processed"

    mock_metadata = {
        "paper_id": "p_sample1",
        "title": "Sample Paper",
        "author": "Alice",
        "pages": 1,
        "filename": "sample.pdf",
    }
    mock_text = "--- PAGE 1 ---\nSample extracted text content."

    with patch("src.ingestion.ingest_pipeline.extract_metadata", return_value=mock_metadata), \
         patch("src.ingestion.ingest_pipeline.extract_text_from_pdf", return_value=mock_text):
        registry = run_ingestion(str(papers_dir), str(processed_dir))

        assert len(registry) == 1
        assert registry[0]["paper_id"] == "p_sample1"
        assert registry[0]["title"] == "Sample Paper"

        # Check saved files
        reg_file = processed_dir / "papers_metadata.json"
        assert reg_file.exists()
        with open(reg_file, "r", encoding="utf-8") as f:
            saved_reg = json.load(f)
        assert len(saved_reg) == 1

        text_file = processed_dir / "texts" / "p_sample1.txt"
        assert text_file.exists()
        assert "Sample extracted text content" in text_file.read_text(encoding="utf-8")
