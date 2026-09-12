import json
import logging
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

def build_faiss_index(
    processed_dir: str,
    vectorstore_dir: str = "data/vectorstore",
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
) -> None:
    """Build a FAISS index from the processed chunk registry."""
    processed_path = Path(processed_dir)
    chunks_path = processed_path / "chunks_registry.json"
    
    if not chunks_path.exists():
        logger.error("Chunks registry not found at %s.", chunks_path)
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_registry = json.load(f)

    if not chunks_registry:
        logger.error("Chunks registry is empty.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading embedding model '%s' on %s...", model_name, device.upper())
    model = SentenceTransformer(model_name, device=device)
    
    texts = [chunk["text"] for chunk in chunks_registry]
    
    logger.info("Generating embeddings for %d chunks in batches of %d...", len(texts), batch_size)
    embeddings = model.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    logger.info("Building FAISS index with dimension %d...", dimension)
    index.add(embeddings)
    
    vectorstore_path = Path(vectorstore_dir)
    vectorstore_path.mkdir(parents=True, exist_ok=True)
    
    index_path = vectorstore_path / "papers.index"
    faiss.write_index(index, str(index_path))
    
    mapping_path = vectorstore_path / "index_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(chunks_registry, f, indent=4)
        
    logger.info("Index successfully saved to %s", index_path)
    logger.info("Mapping successfully saved to %s", mapping_path)

if __name__ == "__main__":
    from src.logging_config import configure_logging

    configure_logging()
    PROCESSED_DIR = "data/processed"
    build_faiss_index(PROCESSED_DIR)
