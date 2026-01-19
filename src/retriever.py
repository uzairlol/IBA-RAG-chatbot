import numpy as np

def retrieve_top_k(query: str, embedder, index, chunks, top_k: int, bm25=None):
    """
    Finds the most similar chunks using Hybrid Search (FAISS + BM25).
    Combines scores using Reciprocal Rank Fusion (RRF).
    """
    # 1. FAISS Search (Semantic)
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, faiss_indices = index.search(q_emb, top_k * 2) # Fetch more for fusion
    
    faiss_results = []
    for rank, idx in enumerate(faiss_indices[0]):
        if idx != -1 and idx < len(chunks):
            faiss_results.append((chunks[idx], rank))

    if not bm25:
        # Fallback to just FAISS if BM25 isn't available
        return [res[0] for res in faiss_results[:top_k]]

    # 2. BM25 Search (Keyword)
    tokenized_query = query.lower().split()
    bm25_results = bm25.get_top_n(tokenized_query, chunks, n=top_k * 2)
    
    # Map chunks to their BM25 rank
    # Note: BM25 returns chunk objects directly. We need to find their index in the 'chunks' list or just use object identity.
    # Since 'chunks' is a list of dicts, let's create a map for quick lookup.
    chunk_id_to_rank = {chunk["id"]: i for i, chunk in enumerate(bm25_results)}

    # 3. Reciprocal Rank Fusion
    # Score = 1 / (k + rank)
    k = 60
    scores = {}

    # Score FAISS results
    for chunk, rank in faiss_results:
        chunk_id = chunk["id"]
        if chunk_id not in scores: scores[chunk_id] = 0
        scores[chunk_id] += 1 / (k + rank)

    # Score BM25 results
    for rank, chunk in enumerate(bm25_results):
        chunk_id = chunk["id"]
        if chunk_id not in scores: scores[chunk_id] = 0
        scores[chunk_id] += 1 / (k + rank)

    # Sort by combined score
    sorted_chunk_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Retrieve final top_k chunks
    # Create a lookup for all chunks by ID to fetching
    all_chunks_map = {c["id"]: c for c in chunks}
    
    final_results = []
    for cid in sorted_chunk_ids[:top_k]:
        final_results.append(all_chunks_map[cid])
        
    return final_results