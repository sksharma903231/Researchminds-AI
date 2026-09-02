import json
import logging
import faiss
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentRetriever:
    def __init__(self, vectorstore_dir: str = "data/vectorstore", model_name: str = "BAAI/bge-small-en-v1.5"):
        self.vectorstore_path = Path(vectorstore_dir)
        self.index_path = self.vectorstore_path / "papers.index"
        self.mapping_path = self.vectorstore_path / "index_mapping.json"
        
        if not self.index_path.exists() or not self.mapping_path.exists():
            raise FileNotFoundError("FAISS index or mapping missing. Run build_index.py first.")
            
        logging.info("Loading FAISS index into memory...")
        self.index = faiss.read_index(str(self.index_path))
        
        logging.info("Loading chunk mappings...")
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading embedding model on {device.upper()}...")
        self.model = SentenceTransformer(model_name, device=device)

    def search(self, query: str, top_k: int = 5) -> list:
        # Embed the query with the exact same model and normalization
        query_vector = self.model.encode([query], normalize_embeddings=True)
        
        # Search the FAISS index
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: 
                continue
            
            chunk_data = self.mapping[idx]
            results.append({
                "score": float(distances[0][i]),
                "paper_id": chunk_data["paper_id"],
                "text": chunk_data["text"]
            })
            
        return results

# Verification Block
if __name__ == "__main__":
    retriever = DocumentRetriever()
    
    test_query = "What is GraphRAG?"
    print(f"\n--- Executing Semantic Search For: '{test_query}' ---\n")
    
    results = retriever.search(test_query, top_k=3)
    
    for rank, res in enumerate(results):
        print(f"Rank {rank + 1} | Similarity Score: {res['score']:.4f} | Source: {res['paper_id']}")
        print(f"Text: {res['text'][:150]}...\n")