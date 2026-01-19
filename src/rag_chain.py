from src.retriever import retrieve_top_k
from src.groq_llm import groq_answer

def rag_query(question: str, embedder, index, chunks, top_k: int, groq_client, groq_model, bm25=None):
    """
    End-to-end RAG pipeline:
    1. Retrieve relevant chunks using Hybrid Search.
    2. Generate answer using Groq.
    
    Returns: (answer_text, list_of_retrieved_chunks)
    """
    
    # Retrieve relevant chunks
    relevant_chunks = retrieve_top_k(question, embedder, index, chunks, top_k, bm25=bm25)
    
    if not relevant_chunks:
        return "I couldn't find any information about that in the university documents.", []
    
    # Step 2: Extract text from chunk objects for the LLM
    context_texts = [chunk["text"] for chunk in relevant_chunks]
    
    # Step 3: Send to LLM with context
    answer = groq_answer(groq_client, groq_model, question, context_texts)
    
    # Step 4: Return answer and context
    return answer, relevant_chunks
