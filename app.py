"""
Simple Streamlit UI for the RAG chatbot.
"""

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.config import FAISS_INDEX_PATH, CHUNKS_PATH, TOP_K, EMBED_MODEL_NAME, GROQ_MODEL
from src.vector_store import load_vector_db
from src.embedder import get_embedder
from src.groq_llm import get_groq_client
from src.rag_chain import rag_query

# Page config
st.set_page_config(page_title="IBA Chatbot", layout="centered")

st.title("University AI Chatbot")
st.caption("Ask questions about IBA, extracted largely from the Admissions Policy and Schedule.")

# Load resources once (cached)
@st.cache_resource
def load_rag_components():
    index, bm25, chunks = load_vector_db(FAISS_INDEX_PATH, CHUNKS_PATH)
    embedder = get_embedder(EMBED_MODEL_NAME)
    groq_client = get_groq_client()
    return index, bm25, chunks, embedder, groq_client

# Load components
with st.spinner("Loading AI system..."):
    index, bm25, chunks, embedder, groq_client = load_rag_components()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Ask about admissions, deadlines, or fees..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = rag_query(
                question=prompt,
                embedder=embedder,
                index=index,
                chunks=chunks,
                top_k=TOP_K,
                groq_client=groq_client,
                groq_model=GROQ_MODEL,
                bm25=bm25
            )
            st.write(answer)
            
            with st.expander("View Sources"):
                for chunk in sources:
                    source = chunk.get("source", chunk["doc_name"])
                    st.markdown(f"- **{source}**")
                    st.caption(chunk["text"][:200] + "...")
                    st.divider()
    
    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
