import json
import logging
from pathlib import Path

from src.ingestion.metadata_extractor import extract_metadata
from src.ingestion.pdf_loader import extract_text_from_pdf

logger = logging.getLogger(__name__)

def run_ingestion(papers_dir: str, processed_dir: str) -> None:
    """Extract text and metadata from PDFs into the processed-data directory."""
    papers_path = Path(papers_dir)
    processed_path = Path(processed_dir)
    texts_dir = processed_path / "texts"
    
    texts_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_registry = []
    pdf_files = list(papers_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDFs found in %s", papers_dir)
        return

    logger.info("Starting ingestion for %d files...", len(pdf_files))

    for pdf_file in pdf_files:
        logger.info("Processing: %s", pdf_file.name)
        
        try:
            metadata = extract_metadata(pdf_file)
            if not metadata:
                logger.error("Failed to extract metadata for %s. Skipping.", pdf_file.name)
                continue
                
            paper_id = metadata["paper_id"]
            
            text_content = extract_text_from_pdf(pdf_file)
            if not text_content:
                logger.error("Failed to extract text for %s. Skipping.", pdf_file.name)
                continue

            text_file_path = texts_dir / f"{paper_id}.txt"
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(text_content)
                
            metadata["text_file_path"] = str(text_file_path)
            metadata_registry.append(metadata)
            
            logger.info("Successfully processed %s", paper_id)

        except (KeyError, OSError, RuntimeError, ValueError):
            logger.exception("Error processing %s.", pdf_file.name)

    registry_path = processed_path / "papers_metadata.json"
    try:
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(metadata_registry, f, indent=4)
        logger.info("Ingestion complete. Registry saved to %s", registry_path)
    except OSError as error:
        logger.error("Failed to save metadata registry: %s", error)

if __name__ == "__main__":
    from src.logging_config import configure_logging

    configure_logging()
    PAPERS_DIR = "data/papers"
    PROCESSED_DIR = "data/processed"
    
    run_ingestion(PAPERS_DIR, PROCESSED_DIR)
