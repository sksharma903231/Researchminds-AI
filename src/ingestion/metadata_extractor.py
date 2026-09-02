import pymupdf
import hashlib
from pathlib import Path

def extract_metadata(pdf_path: str | Path) -> dict:
    path_obj = Path(pdf_path)
    
    try:
        doc = pymupdf.open(path_obj)
        meta = doc.metadata or {}
        
        file_hash = hashlib.md5(path_obj.name.encode()).hexdigest()[:8]
        
        title = meta.get("title", "").strip()
        if not title:
            title = path_obj.stem
            
        author = meta.get("author", "").strip() or "Unknown Author"

        return {
            "paper_id": f"p_{file_hash}",
            "title": title,
            "author": author,
            "pages": doc.page_count,
            "filename": path_obj.name
        }
    except Exception as e:
        print(f"Error extracting metadata from {pdf_path}: {e}")
        return {}