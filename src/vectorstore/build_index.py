import json
import logging
import numpy as np
import faiss
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def build_faiss_index(processed_dir: str, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
    """
    Loads chunks, generates dense vectors using a local GPU-accelerated model, 
    and builds a FAISS index for high-speed retrieval.
    """
    processed_path = Path(processed_dir)
    chunks_path = processed_path / "chunks_registry.json"
    
    if not chunks_path.exists():
        logging.error(f"Chunks registry not found at {chunks_path}.")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_registry = json.load(f)

    if not chunks_registry:
        logging.error("Chunks registry is empty.")
        return

    # Hardware acceleration check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading embedding model '{model_name}' on {device.upper()}...")
    
    # BGE-small is a highly optimized production model for semantic search
    model = SentenceTransformer(model_name, device=device)
    
    texts = [chunk["text"] for chunk in chunks_registry]
    
    logging.info(f"Generating embeddings for {len(texts)} chunks in batches of {batch_size}...")
    
    # Batch encoding ensures we don't overflow the VRAM
    embeddings = model.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True,
        normalize_embeddings=True # Normalizing maps vectors to a unit sphere, improving cosine similarity accuracy
    )
    
    # Initialize FAISS Index (L2 distance or Inner Product)
    # Since embeddings are normalized, Inner Product (IndexFlatIP) is exactly equivalent to Cosine Similarity
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    logging.info(f"Building FAISS index with dimension {dimension}...")
    index.add(embeddings)
    
    # Ensure vectorstore directory exists
    vectorstore_dir = Path("data/vectorstore")
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    
    # Serialize the index and mapping to disk
    index_path = vectorstore_dir / "papers.index"
    faiss.write_index(index, str(index_path))
    
    # We must save a mapping file so FAISS ID (integer) maps back to our original chunk JSON
    mapping_path = vectorstore_dir / "index_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(chunks_registry, f, indent=4)
        
    logging.info(f"Index successfully saved to {index_path}")
    logging.info(f"Mapping successfully saved to {mapping_path}")

if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    build_faiss_index(PROCESSED_DIR)