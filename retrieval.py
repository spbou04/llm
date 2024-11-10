# retrieval.py

from IPython.display import display, Markdown
from llm_api import generate_and_format_response

def retrieve_relevant_docs(query, documents):
    """
    Simple keyword-based function to retrieve relevant documents.
    """
    relevant_docs = []
    for doc in documents:
        if any(keyword.lower() in doc.lower() for keyword in query.split()):
            relevant_docs.append(doc)
    return relevant_docs

def generate_response_with_rag(
    provider,
    model,
    conversation_history,
    current_message,
    documents,
    temperature=0.7,
    max_tokens=1500,
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    min_tokens=None,
    stream=False,
    stop=None,
    random_seed=None,
    response_format=None,
    tools=None,
    tool_choice="auto",
    safe_prompt=False
):
    """
    Wrapper function to perform RAG by retrieving relevant documents,
    combining them into a context, and generating a response.
    """
    relevant_docs = retrieve_relevant_docs(current_message, documents)
    context = "\n\n".join(relevant_docs)
    augmented_message = f"Context: {context}\n\n{current_message}"
    messages = conversation_history + [{"role": "user", "content": augmented_message}]

    response = generate_and_format_response(
        provider=provider,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        min_tokens=min_tokens,
        stream=stream,
        stop=stop,
        random_seed=random_seed,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        safe_prompt=safe_prompt
    )

    # Display formatted response as Markdown
    display(Markdown(response))
    return response
