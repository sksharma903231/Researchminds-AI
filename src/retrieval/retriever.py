import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Retrieve the highest-scoring chunks from the local FAISS index."""

    def __init__(
        self,
        vectorstore_dir: str = "data/vectorstore",
        model_name: str = "BAAI/bge-small-en-v1.5",
        model: Any = None,
        index: Any = None,
        mapping: list[dict[str, Any]] | None = None,
    ):
        self.vectorstore_path = Path(vectorstore_dir)
        self.index_path = self.vectorstore_path / "papers.index"
        self.mapping_path = self.vectorstore_path / "index_mapping.json"

        if index is not None and mapping is not None:
            self.index = index
            self.mapping = mapping
        else:
            if not self.index_path.exists() or not self.mapping_path.exists():
                raise FileNotFoundError(
                    f"FAISS index or mapping missing in '{vectorstore_dir}'. Run build_index.py first."
                )

            logger.info("Loading FAISS index into memory...")
            self.index = faiss.read_index(str(self.index_path))

            logger.info("Loading chunk mappings...")
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)

        if len(self.mapping) != self.index.ntotal:
            logger.warning(
                "Index size (%d) and mapping size (%d) differ.",
                self.index.ntotal,
                len(self.mapping),
            )

        if model is not None:
            self.model = model
        else:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading embedding model on %s...", device.upper())
            self.model = SentenceTransformer(model_name, device=device)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search the FAISS vector index and return matching chunks with metadata."""
        if not query or not query.strip():
            logger.warning("Empty search query provided.")
            return []

        if self.index is None or self.index.ntotal == 0 or not self.mapping:
            logger.warning("Search attempted on empty vectorstore.")
            return []

        k = max(1, min(top_k, self.index.ntotal))

        query_vector = self.model.encode([query.strip()], normalize_embeddings=True)
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

        distances, indices = self.index.search(query_vector, k)

        results: list[dict[str, Any]] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.mapping):
                continue

            score = float(distances[0][i])
            if min_score is not None and score < min_score:
                continue

            chunk_data = self.mapping[idx]
            results.append(
                {
                    "score": round(score, 4),
                    "paper_id": chunk_data.get("paper_id", ""),
                    "chunk_id": chunk_data.get("chunk_id", ""),
                    "title": chunk_data.get("title", chunk_data.get("paper_id", "")),
                    "filename": chunk_data.get("filename", ""),
                    "author": chunk_data.get("author", ""),
                    "page": chunk_data.get("page", None),
                    "text": chunk_data.get("text", ""),
                }
            )

        return results

