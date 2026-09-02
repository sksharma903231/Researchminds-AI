import json
import logging
import traceback
from pathlib import Path

from pdf_loader import extract_text_from_pdf
from metadata_extractor import extract_metadata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ingestion.log"), 
        logging.StreamHandler()               
    ]
)

def run_ingestion(papers_dir: str, processed_dir: str):
    papers_path = Path(papers_dir)
    processed_path = Path(processed_dir)
    texts_dir = processed_path / "texts"
    
    texts_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_registry = []
    pdf_files = list(papers_path.glob("*.pdf"))
    
    if not pdf_files:
        logging.warning(f"No PDFs found in {papers_dir}")
        return

    logging.info(f"Starting ingestion for {len(pdf_files)} files...")

    for pdf_file in pdf_files:
        logging.info(f"Processing: {pdf_file.name}")
        
        try:
            metadata = extract_metadata(pdf_file)
            if not metadata:
                logging.error(f"Failed to extract metadata for {pdf_file.name}. Skipping.")
                continue
                
            paper_id = metadata["paper_id"]
            
            text_content = extract_text_from_pdf(pdf_file)
            if not text_content:
                logging.error(f"Failed to extract text for {pdf_file.name}. Skipping.")
                continue

            text_file_path = texts_dir / f"{paper_id}.txt"
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(text_content)
                
            metadata["text_file_path"] = str(text_file_path)
            metadata_registry.append(metadata)
            
            logging.info(f"Successfully processed {paper_id}")

        except Exception as e:
            logging.error(f"Critical error processing {pdf_file.name}: {e}")
            logging.debug(traceback.format_exc())

    registry_path = processed_path / "papers_metadata.json"
    try:
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(metadata_registry, f, indent=4)
        logging.info(f"Ingestion complete. Registry saved to {registry_path}")
    except Exception as e:
        logging.error(f"Failed to save metadata registry: {e}")

if __name__ == "__main__":
    PAPERS_DIR = "data/papers"
    PROCESSED_DIR = "data/processed"
    
    run_ingestion(PAPERS_DIR, PROCESSED_DIR)