import os
import json
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

def ensure_dir(path: str):
    """Creates a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def build_faiss_index(embeddings: np.ndarray):
    """
    Creates a FAISS index from embeddings.
    """
    embeddings = embeddings.astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def build_bm25_index(chunks: list):
    """
    Creates a BM25 index from text chunks.
    """
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def save_vector_db(index, bm25, chunks, vector_db_dir: str, faiss_path: str, chunks_path: str):
    """
    Saves FAISS index, BM25 index, and chunks to disk.
    """
    ensure_dir(vector_db_dir)
    
    # Save FAISS
    faiss.write_index(index, faiss_path)
    
    # Save BM25
    bm25_path = os.path.join(vector_db_dir, "bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # Save Chunks
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def load_vector_db(faiss_path: str, chunks_path: str):
    """
    Loads FAISS index, BM25 index, and chunks from disk.
    """
    vector_db_dir = os.path.dirname(faiss_path)
    bm25_path = os.path.join(vector_db_dir, "bm25.pkl")

    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS index not found at: {faiss_path}")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file not found at: {chunks_path}")
    
    # Load FAISS
    index = faiss.read_index(faiss_path)
    
    # Load BM25 (optional, for backward compatibility)
    bm25 = None
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
    
    # Load Chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, bm25, chunks
