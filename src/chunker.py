from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int, overlap: int):
    """
    Splits a long text into smaller overlapping pieces using RecursiveCharacterTextSplitter.
    This respects paragraph and sentence boundaries better than simple splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    
    return splitter.split_text(text)
