# pdf_summarizer.py

import os
import pdfplumber
from IPython.display import display, Markdown
from dotenv import load_dotenv
from llm_api import generate_and_format_response

# Load environment variables from .env
load_dotenv()

# Maximum token limit for chunk processing
MAX_TOKENS = 8192 // 2  # Adjust based on model's context length

def pdf_to_text(pdf_path):
    """
    Extracts text from a PDF file.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def split_text_into_chunks(text, max_tokens):
    """
    Splits text into chunks to fit within the token limit.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for space
        if current_length >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_chunk(provider, chunk, model="gpt-3.5-turbo", temperature=0.7, max_tokens=200):
    """
    Summarizes a single chunk of text using the specified provider.
    """
    prompt = f"Summarize the following text in one sentence:\n\n{chunk}"
    messages = [{"role": "user", "content": prompt}]
    
    return generate_and_format_response(
        provider=provider,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

def summarize_pdf_with_provider(provider, pdf_path, model="gpt-3.5-turbo", temperature=0.7, max_tokens=200):
    """
    Summarizes a PDF by processing it in chunks.
    """
    # Convert PDF to text
    text = pdf_to_text(pdf_path)

    # Split text into chunks
    chunks = split_text_into_chunks(text, MAX_TOKENS)

    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        summary = summarize_chunk(provider, chunk, model=model, temperature=temperature, max_tokens=max_tokens)
        summaries.append(summary)
        print(f"Chunk {i+1} summarized.")

    # Combine chunk summaries to generate final summary
    combined_summary = " ".join([s.split("**Assistant:**\n\n")[1].strip() for s in summaries if "**Assistant:**\n\n" in s])
    final_summary_prompt = f"Summarize the following text in a concise summary:\n\n{combined_summary}"
    final_summary = generate_and_format_response(
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": final_summary_prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Display the final summary in Markdown format
    display(Markdown(final_summary))
