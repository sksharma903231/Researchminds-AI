import logging
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

def extract_pages_from_pdf(pdf_path: str | Path) -> list[dict[str, int | str]]:
    """Extract readable text from each non-empty page of a PDF as structured records."""
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        logger.error("PDF file not found at %s", path_obj)
        return []

    try:
        pages: list[dict[str, int | str]] = []
        with pymupdf.open(path_obj) as document:
            for page_num, page in enumerate(document):
                text = page.get_text("text")
                if text and text.strip():
                    pages.append({
                        "page_number": page_num + 1,
                        "text": text.strip(),
                    })
        return pages
    except (OSError, RuntimeError, ValueError) as error:
        logger.error("Error reading %s: %s", path_obj, error)
        return []


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract readable text from each non-empty page of a PDF with page headers."""
    pages = extract_pages_from_pdf(pdf_path)
    if not pages:
        return ""

    text_blocks = [
        f"\n--- PAGE {page['page_number']} ---\n{page['text']}"
        for page in pages
    ]
    return "".join(text_blocks)

