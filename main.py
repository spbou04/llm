# main.py

from chatbot import ChatBot
from retrieval import generate_response_with_rag
from pdf_summarizer import summarize_pdf_with_provider

def chat_example():
    """
    Example usage of the ChatBot class.
    """
    chatbot = ChatBot(provider="openai", model="gpt-3.5-turbo")
    user_input = "Hello, how are you?"
    response = chatbot.get_response(user_input)
    print(response)

def rag_example():
    """
    Example usage of Retrieval-Augmented Generation.
    """
    conversation_history = []
    current_message = "Tell me about quantum computing."
    documents = [
        "Quantum computing uses quantum bits, known as qubits.",
        "Classical computers use bits to process information.",
        "Quantum computers can solve certain problems faster."
    ]
    response = generate_response_with_rag(
        provider="openai",
        model="gpt-3.5-turbo",
        conversation_history=conversation_history,
        current_message=current_message,
        documents=documents
    )
    print(response)

def pdf_summary_example():
    """
    Example usage of PDF summarization.
    """
    provider = "openai"
    pdf_path = "path/to/your/document.pdf"
    summarize_pdf_with_provider(provider, pdf_path)

if __name__ == "__main__":
    # Uncomment the function you want to run
    # chat_example()
    # rag_example()
    # pdf_summary_example()
    pass
