# llm_api.py

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def generate_and_format_response(
    provider,
    model,
    messages,
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
    Generates and formats a response from the specified LLM provider (OpenAI, Anthropic, or Mistral).
    """
    api_key = None
    url = None
    headers = {"Content-Type": "application/json"}

    # Determine API key, endpoint, and headers based on provider
    if provider.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }

    elif provider.lower() == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        url = "https://api.anthropic.com/v1/messages"
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "messages": messages
        }

    elif provider.lower() == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        url = "https://api.mistral.ai/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "tool_choice": tool_choice,
            "safe_prompt": safe_prompt
        }
        # Add optional parameters for Mistral
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if min_tokens is not None:
            payload["min_tokens"] = min_tokens
        if stop is not None:
            payload["stop"] = stop
        if random_seed is not None:
            payload["random_seed"] = random_seed
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
    else:
        return "Invalid provider. Please choose from 'openai', 'anthropic', or 'mistral'."

    # Check if API key is available
    if not api_key:
        raise ValueError(f"{provider.capitalize()} API key not found. Please set it in the .env file.")
    
    try:
        # Send the POST request to the provider's API
        response = requests.post(url, headers=headers, json=payload)
        
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()
        
        # Parse the JSON response
        response_data = response.json()
        
        # Extract the assistant's message and token information based on the provider
        input_tokens = len(" ".join(msg["content"] for msg in messages).split())  # Estimate of input tokens
        if provider.lower() == "openai":
            assistant_message = response_data["choices"][0]["message"]["content"].strip()
            output_tokens = response_data["usage"]["completion_tokens"]

        elif provider.lower() == "anthropic":
            assistant_message = response_data.get("content", [{}])[0].get("text", "No reply found.")
            output_tokens = len(assistant_message.split())  # Estimate of output tokens

        elif provider.lower() == "mistral":
            assistant_message = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No reply found.")
            output_tokens = len(assistant_message.split())  # Estimate of output tokens
        
        # Format the response including provider, model, and token information
        formatted_text = (
            f"**Provider:** {provider.capitalize()} | **Model:** {model}  \n"
            f"**Tokens Used (Input/Output):** {input_tokens}/{output_tokens}  \n\n"
            f"**Assistant:**\n\n{assistant_message}\n"
        )
        return formatted_text

    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
