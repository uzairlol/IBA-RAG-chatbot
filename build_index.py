"""
Run this script ONCE to process all PDFs and build your vector database.
Run it again whenever you add new PDFs to the data folder.
"""

from dotenv import load_dotenv
load_dotenv()

from src.config import (
    DATA_DIR, VECTOR_DB_DIR, FAISS_INDEX_PATH, CHUNKS_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL_NAME
)

import os
import uuid
from src.config import (
    DATA_DIR, VECTOR_DB_DIR, FAISS_INDEX_PATH, CHUNKS_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL_NAME
)

from src.pdf_loader import load_all_pdfs_from_folder
from src.html_loader import load_all_html_from_folder
from src.chunker import chunk_text
from src.embedder import get_embedder, embed_texts
from src.vector_store import build_faiss_index, build_bm25_index, save_vector_db


def main():
    """
    Reads all PDFs and HTML files from the data folder, chunks them,
    creates embeddings, and saves the FAISS index to disk.
    """
    print("Building local vector DB from ALL files (PDF + HTML) in folder...")
    print(f"Data folder: {DATA_DIR}")

    # 1. Load PDFs from the root 'data' folder
    print(f"Scanning for PDFs in {DATA_DIR}...")
    pdf_docs = load_all_pdfs_from_folder(DATA_DIR)
    print(f"Found {len(pdf_docs)} PDFs.")

    # 2. Load HTMLs from 'data/html files'
    html_dir = os.path.join(DATA_DIR, "html files")
    print(f"Scanning for HTML files in {html_dir}...")
    html_docs = load_all_html_from_folder(html_dir)
    print(f"Found {len(html_docs)} HTML files.")

    # Combine docs
    docs = pdf_docs + html_docs
    
    if not docs:
        print("No documents found! Check your data directory.")
        return
    
    # Chunk all documents
    all_chunks = []
    for doc in docs:
        doc_name = doc["doc_name"]
        text = doc["text"]
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        doc_source = doc.get("source", doc_name) # Fallback to doc_name if source missing
        for i, ch in enumerate(chunks):
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "doc_name": doc_name,
                "source": doc_source,
                "chunk_id": i,
                "text": ch
            })

    # Create Embeddings
    print("Creating embeddings (this may take a while)...")
    embedder = get_embedder(EMBED_MODEL_NAME)
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(embedder, chunk_texts)

    # Build FAISS Index
    print("Building FAISS index...")
    faiss_index = build_faiss_index(embeddings)

    # Build BM25 Index
    print("Building BM25 index...")
    bm25_index = build_bm25_index(all_chunks)

    # Save Vector DB
    print(f"Saving combined index to {VECTOR_DB_DIR}...")
    save_vector_db(faiss_index, bm25_index, all_chunks, VECTOR_DB_DIR, FAISS_INDEX_PATH, CHUNKS_PATH)
    
    print("\nDone! Hybrid Index created successfully.")
    print(f"Files indexed: {len(docs)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved FAISS index: {FAISS_INDEX_PATH}")
    print(f"Saved chunks JSON: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
