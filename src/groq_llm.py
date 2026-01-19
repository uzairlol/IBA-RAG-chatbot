import os
from groq import Groq

def get_groq_client():
    """
    Creates a Groq API client using the API key from .env file.
    Call this once at startup and reuse the client.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in .env")
    return Groq(api_key=api_key)

def groq_answer(client, model: str, question: str, context_chunks: list[str]):
    """
    Sends retrieved context + user question to Groq's LLM.
    Returns the generated answer as a string.
    """
    context_text = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are a helpful assistant answering questions from a research paper.\n"
        "Use ONLY the provided context. If the answer is not in the context, say:\n"
        "'I could not find this in the provided paper context.'"
    )

    user_prompt = f"""
Context from paper:
{context_text}

Question:
{question}

Answer clearly:
"""

    completion = client.chat.completions.create(
        model=model,  # "llama-3.3-70b-versatile"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # Low = more factual, less creative
    )

    return completion.choices[0].message.content
