import os
from pypdf import PdfReader

def load_pdf_text(pdf_path: str) -> str:
    """
    Reads a single PDF file and extracts all text from it.
    Returns the full text as one string.
    """
    reader = PdfReader(pdf_path)

    pages = []
    for page in reader.pages:
        text = page.extract_text() or "" # Get text from each page
        text = text.replace("\n", " ").strip() # Remove newlines and extra spaces
        if text:
            pages.append(text)

    return "\n".join(pages) # Join all pages into a single string

def load_all_pdfs_from_folder(folder_path: str):
    """
    Returns list of dictionaries:
    [
      {"doc_name": "paper1.pdf", "text": "..."},
      {"doc_name": "paper2.pdf", "text": "..."}
    ]
    """
    pdfs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            text = load_pdf_text(full_path)
            if text.strip():
                pdfs.append({
                    "doc_name": filename, 
                    "text": text,
                    "source": filename # Store source
                })
    return pdfs
