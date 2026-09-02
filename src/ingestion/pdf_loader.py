import pymupdf
from pathlib import Path

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    try:
        doc = pymupdf.open(pdf_path)
        text_content = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                text_content.append(f"\n--- PAGE {page_num + 1} ---\n{text}")

        return "".join(text_content)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""