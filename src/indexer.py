from .pdf_loader import load_all_pdfs_from_folder
from .chunker import chunk_text
from .embedder import get_embedder, embed_texts
from .vector_store import build_faiss_index, save_vector_db

def build_and_save_index_from_folder(
    pdf_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_model_name: str,
    vector_db_dir: str,
    faiss_index_path: str,
    chunks_path: str
):
    """
   Run this once to build your vector database.
    
    Steps:
    1. Load all PDFs from the folder
    2. Split each document into overlapping chunks
    3. Convert all chunks into embeddings
    4. Build a FAISS index for fast similarity search
    5. Save the index and chunk metadata to disk
    
    Returns: (total_chunks, total_docs)
    """
    docs = load_all_pdfs_from_folder(pdf_dir)

    all_chunks = []
    for doc in docs:
        doc_name = doc["doc_name"]
        text = doc["text"]

        chunks = chunk_text(text, chunk_size, chunk_overlap)

        for i, ch in enumerate(chunks):
            all_chunks.append({
                "doc_name": doc_name,
                "chunk_id": i,
                "text": ch
            })

    # Embed only the text field
    embedder = get_embedder(embed_model_name)
    chunk_texts = [c["text"] for c in all_chunks]

    embeddings = embed_texts(embedder, chunk_texts)
    index = build_faiss_index(embeddings)

    save_vector_db(index, all_chunks, vector_db_dir, faiss_index_path, chunks_path)

    return len(all_chunks), len(docs)
