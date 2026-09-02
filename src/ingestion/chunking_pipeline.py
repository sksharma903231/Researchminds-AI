import json
import logging
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_chunking(processed_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Reads extracted texts based on the metadata registry, 
    splits them into overlapping chunks, and serializes them to disk.
    """
    processed_path = Path(processed_dir)
    registry_path = processed_path / "papers_metadata.json"
    
    if not registry_path.exists():
        logging.error(f"Metadata registry not found at {registry_path}. Run ingestion pipeline first.")
        return

    with open(registry_path, "r", encoding="utf-8") as f:
        metadata_registry = json.load(f)

    # Industry standard starting parameters for RAG
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
            logging.warning(f"Source text file missing for {paper_id}. Skipping.")
            continue
            
        try:
            with open(text_file_path, "r", encoding="utf-8") as f:
                text_content = f.read()

            chunks = text_splitter.split_text(text_content)
            logging.info(f"Split {paper_id} into {len(chunks)} chunks.")
            
            for index, chunk_text in enumerate(chunks):
                chunk_record = {
                    "chunk_id": f"{paper_id}_c{index}",
                    "paper_id": paper_id,
                    "text": chunk_text
                }
                all_chunks.append(chunk_record)
                
        except Exception as e:
            logging.error(f"Error processing text for {paper_id}: {e}")

    # Serialize the flattened list of all chunks to disk
    chunks_output_path = processed_path / "chunks_registry.json"
    try:
        with open(chunks_output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=4)
        logging.info(f"Chunking complete. {len(all_chunks)} total chunks saved to {chunks_output_path}")
    except Exception as e:
        logging.error(f"Failed to save chunks registry: {e}")

if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    run_chunking(PROCESSED_DIR)