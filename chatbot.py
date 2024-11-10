# chatbot.py

import os
from dotenv import load_dotenv
from llm_api import generate_and_format_response

# Load environment variables from .env
load_dotenv()

class ChatBot:
    def __init__(self, provider, model, api_key=None, temperature=0.7, max_tokens=1500, top_p=0.9,
                 frequency_penalty=0.0, presence_penalty=0.0, min_tokens=None, stream=False, stop=None,
                 random_seed=None, response_format=None, tools=None, tool_choice="auto", safe_prompt=False):
        """
        Initialize the ChatBot with API key, provider, and parameters.
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv(f"{self.provider.upper()}_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.min_tokens = min_tokens
        self.stream = stream
        self.stop = stop
        self.random_seed = random_seed
        self.response_format = response_format
        self.tools = tools
        self.tool_choice = tool_choice
        self.safe_prompt = safe_prompt
        self.conversation_history = []

    def add_message(self, role, content):
        """
        Add a message to the conversation history.
        """
        self.conversation_history.append({"role": role, "content": content})

    def get_response(self, user_input):
        """
        Get a response from the selected provider's API based on the user input.
        """
        # Add user input to conversation history
        self.add_message("user", user_input)

        # Generate response using the llm_api module
        response = generate_and_format_response(
            provider=self.provider,
            model=self.model,
            messages=self.conversation_history,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            min_tokens=self.min_tokens,
            stream=self.stream,
            stop=self.stop,
            random_seed=self.random_seed,
            response_format=self.response_format,
            tools=self.tools,
            tool_choice=self.tool_choice,
            safe_prompt=self.safe_prompt
        )

        # Extract assistant's reply (assuming the response is formatted as per generate_and_format_response)
        try:
            assistant_reply = response.split("**Assistant:**\n\n")[1].strip()
        except IndexError:
            assistant_reply = "No reply found."

        # Add assistant's reply to conversation history
        self.add_message("assistant", assistant_reply)

        return response
