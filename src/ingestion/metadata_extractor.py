import hashlib
import logging
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)


def extract_metadata(pdf_path: str | Path) -> dict[str, str | int]:
    """Extract the metadata required by the ingestion registry."""
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        logger.error("File does not exist: %s", path_obj)
        return {}

    try:
        with pymupdf.open(path_obj) as document:
            metadata = document.metadata or {}
            file_hash = hashlib.md5(path_obj.name.encode()).hexdigest()[:8]
            
            raw_title = metadata.get("title") or ""
            raw_title = raw_title.replace("\x00", "").strip()
            title = raw_title if raw_title else path_obj.stem

            raw_author = metadata.get("author") or ""
            raw_author = raw_author.replace("\x00", "").strip()
            author = raw_author if raw_author else "Unknown Author"

            return {
                "paper_id": f"p_{file_hash}",
                "title": title,
                "author": author,
                "pages": document.page_count,
                "filename": path_obj.name,
            }
    except (OSError, RuntimeError, ValueError) as error:
        logger.error("Error extracting metadata from %s: %s", path_obj, error)
        return {}

