import os

# Path of folder containing data
DATA_DIR = "data"

# Local Vector DB paths
VECTOR_DB_DIR = "vectordb"
FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(VECTOR_DB_DIR, "chunks.json")

# Chunking
CHUNK_SIZE = 900 #characters per chunk about 150 to 200 words
CHUNK_OVERLAP = 150 #characters overlap between chunks

# Retrieval
TOP_K = 4

# Embeddings
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Groq model
GROQ_MODEL = "llama-3.3-70b-versatile"
