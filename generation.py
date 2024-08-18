#!/usr/bin/env python3
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

# generation.py
import os
import requests
import json
import time
from config import *
import anthropic
import google.generativeai as genai

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=google_api_key)
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

def process_openai_response(response_data):
    return response_data['choices'][0]['message']['content'].strip()

def process_mistral_response(response_data):
    return response_data['choices'][0]['message']['content'].strip()

def process_claude_response(response_data):
    # Assuming the response_data is the JSON-decoded response
    if "content" in response_data and response_data["content"]:
        # Assuming the first item in the content list is the one we're interested in
        content_item = response_data["content"][0]
        if "text" in content_item:
            return content_item["text"]
    return "No response generated."

def process_google_response(response_data):
    # Placeholder for Google-specific response processing
    return response_data.get('text', '')

response_processors = {
    "gpt-4": process_openai_response,
    "gpt-4o": process_openai_response,
    "gpt-3.5-turbo": process_openai_response,
    "gpt-4o-mini": process_openai_response,
    "mistral": process_mistral_response,
    "claude": process_claude_response,
    "google": process_google_response,
}

def generate(model, prompt, context=None, keep_alive='30s'):
    start_time = time.time()

    headers = {
        "Authorization": f"Bearer {os.getenv(model.upper() + '_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    api_url = ""

    if model in ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]:
#        print(f"Using API Key: {OPENAI_API_KEY}")  # temp DEBUG
        # Use OpenAI API for ChatGPT models
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data.update({
            "model": model,
            "max_tokens": openai_max_tokens,
            "temperature": openai_temperature,
            "messages": [
                {"role": "system", "content": openai_system_prompt},
                {"role": "user", "content": prompt}
            ],
        })
        api_url = "https://api.openai.com/v1/chat/completions"
    elif model.startswith("claude"):
        # Claude (Anthropic) API request setup
        headers = {
            "Authorization": f"Bearer {CLAUDE_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": claude_model,  # Example model, adjust as needed
            "max_tokens": claude_max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        api_url = "https://api.anthropic.com/v1/messages"
    elif model.startswith("google"):
        # Initialize the Google model
        google_model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Example without streaming
        response = google_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,  # Currently, only one candidate is supported
                stop_sequences=["\n"],  # Adjust based on your stopping criteria
                max_output_tokens=google_max_tokens,  # Adjust based on your needs
                temperature=google_temperature,  # Adjust for creativity level
            ),
        )
        first_choice_content = response.text
    elif model == "mistral":
        data.update({
            "model": mistral_model,
            "temperature": mistral_temperature,
            "top_p": 1,
            "max_tokens": mistral_max_tokens,
            "min_tokens": 0,
            "stream": False,
            "stop": "\n",
            "random_seed": None,
            "response_format": {"type": "text"},
            "tools": [],
            "tool_choice": "auto",
            "safe_prompt": False
        })
        api_url = "https://api.mistral.ai/v1/chat/completions"
    else:
        # Handle Ollama local models or any other models not specifically mentioned
        # Assuming a local API endpoint for Ollama models
        response = requests.post('http://localhost:11434/api/generate',
                                 json={
                                     'model': model,
                                     'prompt': prompt,
                                     'keep_alive': keep_alive
                                 },
                                 stream=True)
        response.raise_for_status()

        response_parts = []
        for line in response.iter_lines():
            if line:
                body = json.loads(line)
                response_part = body.get('response', '')
                response_parts.append(response_part)
                print(f"\033[32m{response_part}\033[0m", end='', flush=True)

                if 'error' in body:
                    raise Exception(body['error'])

                if body.get('done', False):
                    print()
                    break

        full_response = ''.join(response_parts)
        response_time = time.time() - start_time
        char_count = len(full_response)
        word_count = len(full_response.split())

        return None, full_response, response_time, char_count, word_count

    # Make the API request for external models
    if api_url:  # This check ensures we only proceed for external API models
        response = requests.post(api_url, json=data, headers=headers)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print("Error Response:", response.text)
            raise e

        # Process the response for external API models
        response_data = response.json()
        response_processor = response_processors.get(model)
        if response_processor:
            first_choice_content = response_processor(response_data)
        else:
            first_choice_content = "Response processing not implemented for this model."

        response_time = time.time() - start_time
        print(f"\033[32m{first_choice_content}\033[0m", flush=True)

        return None, first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())