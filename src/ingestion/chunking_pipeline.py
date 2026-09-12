import json
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def run_chunking(
    processed_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> None:
    """Split extracted papers into overlapping chunks and save the registry."""
    processed_path = Path(processed_dir)
    registry_path = processed_path / "papers_metadata.json"
    
    if not registry_path.exists():
        logger.error("Metadata registry not found at %s. Run ingestion first.", registry_path)
        return

    with open(registry_path, "r", encoding="utf-8") as f:
        metadata_registry = json.load(f)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    all_chunks = []
    
    for paper in metadata_registry:
        paper_id = paper.get("paper_id")
        text_file_path = Path(paper.get("text_file_path"))
        
        if not text_file_path.exists():
            logger.warning("Source text file missing for %s. Skipping.", paper_id)
            continue
            
        try:
            with open(text_file_path, "r", encoding="utf-8") as f:
                text_content = f.read()

            chunks = text_splitter.split_text(text_content)
            logger.info("Split %s into %d chunks.", paper_id, len(chunks))
            
            for index, chunk_text in enumerate(chunks):
                chunk_record = {
                    "chunk_id": f"{paper_id}_c{index}",
                    "paper_id": paper_id,
                    "text": chunk_text
                }
                all_chunks.append(chunk_record)
                
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

if __name__ == "__main__":
    from src.logging_config import configure_logging

    configure_logging()
    PROCESSED_DIR = "data/processed"
    run_chunking(PROCESSED_DIR)
