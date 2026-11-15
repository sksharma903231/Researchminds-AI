from unittest.mock import MagicMock, patch

from src.ingestion.metadata_extractor import extract_metadata


def test_extract_metadata_nonexistent():
    meta = extract_metadata("missing_file.pdf")
    assert meta == {}


def test_extract_metadata_with_explicit_fields():
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "title": "A Novel RAG Architecture",
        "author": "Dr. Alice & Dr. Bob",
    }
    mock_doc.page_count = 12
    mock_doc.__enter__.return_value = mock_doc
    mock_doc.__exit__.return_value = None

    with patch("pathlib.Path.exists", return_value=True), patch("pymupdf.open", return_value=mock_doc):
        meta = extract_metadata("sample_paper.pdf")
        assert meta["title"] == "A Novel RAG Architecture"
        assert meta["author"] == "Dr. Alice & Dr. Bob"
        assert meta["pages"] == 12
        assert meta["filename"] == "sample_paper.pdf"
        assert meta["paper_id"].startswith("p_")


def test_extract_metadata_fallback_to_stem():
    mock_doc = MagicMock()
    mock_doc.metadata = {"title": "", "author": ""}
    mock_doc.page_count = 5
    mock_doc.__enter__.return_value = mock_doc
    mock_doc.__exit__.return_value = None

    with patch("pathlib.Path.exists", return_value=True), patch("pymupdf.open", return_value=mock_doc):
        meta = extract_metadata("arxiv_2401.12345.pdf")
        assert meta["title"] == "arxiv_2401.12345"
        assert meta["author"] == "Unknown Author"
        assert meta["pages"] == 5
