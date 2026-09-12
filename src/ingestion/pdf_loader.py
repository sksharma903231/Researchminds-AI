import logging
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract readable text from each non-empty page of a PDF."""
    try:
        text_content = []
        with pymupdf.open(pdf_path) as document:
            for page_num, page in enumerate(document):
                text = page.get_text("text")
                if text.strip():
                    text_content.append(f"\n--- PAGE {page_num + 1} ---\n{text}")

        return "".join(text_content)
    except (OSError, RuntimeError, ValueError) as error:
        logger.error("Error reading %s: %s", pdf_path, error)
        return ""
