"""
Interactive RAG chatbot.
Loads the pre-built vector database and lets you ask questions about the University Website.
"""

from dotenv import load_dotenv
load_dotenv()

from src.config import FAISS_INDEX_PATH, CHUNKS_PATH, TOP_K, EMBED_MODEL_NAME, GROQ_MODEL
from src.vector_store import load_vector_db
from src.embedder import get_embedder
from src.groq_llm import get_groq_client
from src.rag_chain import rag_query


def main():
    """
    Starts the RAG chatbot:
    1. Loads the FAISS index and chunk metadata
    2. Loads the embedding model
    3. Connects to Groq API
    4. Runs an interactive Q&A loop
    """
    # Load RAG components
    print("Loading local vector DB...")
    try:
        index, bm25, chunks = load_vector_db(FAISS_INDEX_PATH, CHUNKS_PATH)
    except Exception as e:
        print(f"Error loading database: {e}")
        print("Did you run build_index.py?")
        return

    print("Loading embedder...")
    embedder = get_embedder(EMBED_MODEL_NAME)
    
    # Initialize Groq client
    groq_client = get_groq_client()

    print("\n" + "="*50)
    print("🎓 University Chatbot Initialized")
    print(" Ask about admissions, fees, or programs.")
    print(" Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Run RAG pipeline
        print("Thinking...")
        answer, sources = rag_query(
            question=user_input, 
            embedder=embedder, 
            index=index, 
            chunks=chunks, 
            top_k=TOP_K,
            groq_client=groq_client,
            groq_model=GROQ_MODEL,
            bm25=bm25
        )
        
        print(f"Bot: {answer}")
        
        # Optional: Print sources in CLI
        print("\n[Sources referenced:]")
        for chunk in sources:
            src = chunk.get("source", chunk["doc_name"])
            print(f"- {src}")
        print("-" * 30 + "\n")


if __name__ == "__main__":
    main()
