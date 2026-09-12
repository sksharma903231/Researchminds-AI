import json
import logging
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Retrieve the highest-scoring chunks from the local FAISS index."""

    def __init__(
        self,
        vectorstore_dir: str = "data/vectorstore",
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.vectorstore_path = Path(vectorstore_dir)
        self.index_path = self.vectorstore_path / "papers.index"
        self.mapping_path = self.vectorstore_path / "index_mapping.json"
        
        if not self.index_path.exists() or not self.mapping_path.exists():
            raise FileNotFoundError("FAISS index or mapping missing. Run build_index.py first.")
            
        logger.info("Loading FAISS index into memory...")
        self.index = faiss.read_index(str(self.index_path))
        
        logger.info("Loading chunk mappings...")
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading embedding model on %s...", device.upper())
        self.model = SentenceTransformer(model_name, device=device)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, float | str]]:
        query_vector = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_vector, top_k)
        
        results: list[dict[str, float | str]] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            chunk_data = self.mapping[idx]
            results.append(
                {
                    "score": float(distances[0][i]),
                    "paper_id": chunk_data["paper_id"],
                    "text": chunk_data["text"],
                }
            )

        return results
