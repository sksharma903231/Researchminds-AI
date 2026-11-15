from unittest.mock import MagicMock, patch
from pathlib import Path

from src.ingestion.pdf_loader import extract_pages_from_pdf, extract_text_from_pdf


def test_extract_from_nonexistent_pdf():
    pages = extract_pages_from_pdf("nonexistent_file_xyz.pdf")
    assert pages == []

    text = extract_text_from_pdf("nonexistent_file_xyz.pdf")
    assert text == ""


def test_extract_pages_from_pdf_with_mock():
    mock_doc = MagicMock()
    page1 = MagicMock()
    page1.get_text.return_value = "Content of page 1"
    page2 = MagicMock()
    page2.get_text.return_value = ""  # empty page
    page3 = MagicMock()
    page3.get_text.return_value = "Content of page 3"
    mock_doc.__iter__.return_value = [page1, page2, page3]
    mock_doc.__enter__.return_value = mock_doc
    mock_doc.__exit__.return_value = None

    with patch("pathlib.Path.exists", return_value=True), patch("pymupdf.open", return_value=mock_doc):
        pages = extract_pages_from_pdf("dummy.pdf")
        assert len(pages) == 2
        assert pages[0]["page_number"] == 1
        assert pages[0]["text"] == "Content of page 1"
        assert pages[1]["page_number"] == 3
        assert pages[1]["text"] == "Content of page 3"


def test_extract_text_from_pdf_formats_headers():
    mock_doc = MagicMock()
    page1 = MagicMock()
    page1.get_text.return_value = "First page text"
    mock_doc.__iter__.return_value = [page1]
    mock_doc.__enter__.return_value = mock_doc
    mock_doc.__exit__.return_value = None

    with patch("pathlib.Path.exists", return_value=True), patch("pymupdf.open", return_value=mock_doc):
        text = extract_text_from_pdf("dummy.pdf")
        assert "--- PAGE 1 ---" in text
        assert "First page text" in text
