import json
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

import re

def parse_pages_from_text(text_content: str) -> list[tuple[int, str]]:
    """Parse text into page number and content pairs using page delimiter headers."""
    pattern = re.compile(r"(?:^|\n)--- PAGE (\d+) ---\n")
    matches = list(pattern.finditer(text_content))

    if not matches:
        stripped = text_content.strip()
        return [(1, stripped)] if stripped else []

    pages: list[tuple[int, str]] = []
    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text_content)
        page_text = text_content[start_idx:end_idx].strip()
        if page_text:
            pages.append((page_num, page_text))

    return pages


def run_chunking(
    processed_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, str | int | None]]:
    """Split extracted papers into overlapping chunks preserving page and paper metadata."""
    processed_path = Path(processed_dir)
    registry_path = processed_path / "papers_metadata.json"

    if not registry_path.exists():
        logger.error("Metadata registry not found at %s. Run ingestion first.", registry_path)
        return []

    with open(registry_path, "r", encoding="utf-8") as f:
        metadata_registry = json.load(f)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    all_chunks: list[dict[str, str | int | None]] = []

    for paper in metadata_registry:
        paper_id = paper.get("paper_id")
        title = paper.get("title", paper_id)
        filename = paper.get("filename", "")
        author = paper.get("author", "Unknown Author")

        raw_path = Path(paper.get("text_file_path", ""))
        if raw_path.exists():
            text_file_path = raw_path
        else:
            alt_path = processed_path / "texts" / f"{paper_id}.txt"
            if alt_path.exists():
                text_file_path = alt_path
            else:
                logger.warning("Source text file missing for %s. Skipping.", paper_id)
                continue

        try:
            with open(text_file_path, "r", encoding="utf-8") as f:
                text_content = f.read()

            pages = parse_pages_from_text(text_content)
            chunk_counter = 0

            for page_num, page_text in pages:
                sub_chunks = text_splitter.split_text(page_text)
                for chunk_text in sub_chunks:
                    clean_text = chunk_text.strip()
                    if not clean_text:
                        continue
                    chunk_record = {
                        "chunk_id": f"{paper_id}_c{chunk_counter}",
                        "paper_id": paper_id,
                        "title": title,
                        "filename": filename,
                        "author": author,
                        "page": page_num,
                        "text": clean_text,
                    }
                    all_chunks.append(chunk_record)
                    chunk_counter += 1

            logger.info("Split %s into %d chunks.", paper_id, chunk_counter)

        except OSError as error:
            logger.error("Error processing text for %s: %s", paper_id, error)

    chunks_output_path = processed_path / "chunks_registry.json"
    try:
        with open(chunks_output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=4)
        logger.info(
            "Chunking complete. %d chunks saved to %s",
            len(all_chunks),
            chunks_output_path,
        )
    except OSError as error:
        logger.error("Failed to save chunks registry: %s", error)

    return all_chunks


if __name__ == "__main__":
    from src.logging_config import configure_logging

    configure_logging()
    PROCESSED_DIR = "data/processed"
    run_chunking(PROCESSED_DIR)
