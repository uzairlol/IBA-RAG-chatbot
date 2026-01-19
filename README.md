# RAG System

A simple Retrieval-Augmented Generation (RAG) system that lets you ask questions about your PDF documents using local FAISS indexing and Groq's LLM API.

## How It Works

1. **Indexing**: PDFs are loaded, split into chunks, and converted to embeddings stored in a FAISS vector database.
2. **Querying**: Your question is embedded, similar chunks are retrieved, and sent to an LLM to generate an answer.

```
PDFs --> Chunk --> Embed --> FAISS Index
                                  |
Question --> Embed --> Search --> Retrieved Chunks --> LLM --> Answer
```

## Key Features
- **Hybrid Search**: Combines BM25 (Keyword) and FAISS (Semantic) for high-accuracy retrieval.
- **Multi-Format Support**: Indexes both PDF documents and HTML files.
- **Smart Chunking**: Uses recursive splitting to respect paragraph boundaries.
- **Source Attribution**: Clearly cites which document or webpage the answer came from.

## Project Structure

```
RAG/
├── .env                    # API keys
├── requirements.txt        # Python dependencies
├── build_index.py          # Run once to index PDFs + HTML
├── chat.py                 # Interactive Q&A chatbot (CLI)
├── app.py                  # Streamlit Web Interface
├── data/                   # Put your PDFs and HTML files here
├── vectordb/               # Generated vector database (FAISS + BM25)
└── src/
    ├── config.py           # Configuration settings
    ├── pdf_loader.py       # PDF text extraction
    ├── html_loader.py      # HTML text extraction
    ├── chunker.py          # Smart text chunking
    ├── embedder.py         # Sentence embeddings
    ├── vector_store.py     # FAISS & BM25 index management
    ├── retriever.py        # Hybrid search logic (RRF)
    ├── groq_llm.py         # Groq API integration
    └── rag_chain.py        # RAG query pipeline
```

## Setup

### Prerequisites

> [!IMPORTANT]
> **Python 3.10.x** is recommended.

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key**
   Create a `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Add Data**
   - Place PDFs in `data/`
   - Place HTML files in `data/html files/`

## Usage

### 1. Build the Index
Run this whenever you add new files:
```bash
python build_index.py
```

### 2. Run the Chatbot

**Option A: Web Interface (Recommended)**
```bash
streamlit run app.py
```

**Option B: Terminal Chat**
```bash
python chat.py
```

## Configuration
Edit `src/config.py` to customize settings like `CHUNK_SIZE` or `TOP_K`.

## Tech Stack
- **LangChain**: Text splitting
- **FAISS**: Vector search
- **Rank_BM25**: Keyword search
- **Streamlit**: Web UI
- **Groq**: Fast LLM inference

## Tech Stack

- **pypdf**: PDF text extraction
- **sentence-transformers**: Text embeddings
- **FAISS**: Vector similarity search
- **Groq**: LLM inference API
