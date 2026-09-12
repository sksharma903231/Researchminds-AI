import hashlib
import logging
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)


def extract_metadata(pdf_path: str | Path) -> dict[str, str | int]:
    """Extract the metadata required by the ingestion registry."""
    path_obj = Path(pdf_path)
    
    try:
        with pymupdf.open(path_obj) as document:
            metadata = document.metadata or {}
            file_hash = hashlib.md5(path_obj.name.encode()).hexdigest()[:8]
            title = metadata.get("title", "").strip() or path_obj.stem
            author = metadata.get("author", "").strip() or "Unknown Author"

            return {
                "paper_id": f"p_{file_hash}",
                "title": title,
                "author": author,
                "pages": document.page_count,
                "filename": path_obj.name,
            }
    except (OSError, RuntimeError, ValueError) as error:
        logger.error("Error extracting metadata from %s: %s", pdf_path, error)
        return {}
