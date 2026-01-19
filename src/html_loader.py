import os
from bs4 import BeautifulSoup

def load_html_text(file_path: str) -> str:
    """
    Reads a single HTML file and extracts text from it.
    Prioritizes text inside <div id="main">, falls back to <body>.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        soup = BeautifulSoup(content, "html.parser")
        
        # Try to find the main content area
        main_content = soup.find("div", id="main")
        
        if main_content:
            # Extract text from the main div
            text = main_content.get_text(separator="\n")
        elif soup.body:
            # Fallback to body if no main div found
            text = soup.body.get_text(separator="\n")
        else:
            text = soup.get_text(separator="\n")
            
        # Clean up text logic similar to pdf loader
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def load_all_html_from_folder(folder_path: str):
    """
    Returns list of dictionaries:
    [
      {"doc_name": "page.htm", "text": "..."},
      ...
    ]
    """
    docs = []
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        return docs
        
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".html", ".htm")):
            full_path = os.path.join(folder_path, filename)
            text = load_html_text(full_path)
            if text.strip():
                docs.append({
                    "doc_name": filename, 
                    "text": text, 
                    "source": filename  # Store source filename
                })
    return docs
