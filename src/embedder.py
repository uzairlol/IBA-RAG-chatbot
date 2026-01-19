from sentence_transformers import SentenceTransformer

def get_embedder(model_name: str):
    """
    The model converts text into numerical vectors (embeddings).
    Similar texts will have similar vectors.
    """
    return SentenceTransformer(model_name)

def embed_texts(embedder, texts):
    """
    Takes a list of text strings and converts them into vectors.
    Returns a numpy array where each row is an embedding for one text.
    """
    return embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
