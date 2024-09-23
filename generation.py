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
import logging
from user_messages import *
import spacy

# Suppress INFO logs from the `requests` library
logging.basicConfig(level=logging.WARNING)

def process_openai_response(response_data):
    return response_data['choices'][0]['message']['content'].strip()

def process_claude_response(response_data):
    # Assuming the response_data is the JSON-decoded response
    if "content" in response_data and response_data["content"]:
        # Assuming the first item in the content list is the one we're interested in
        content_item = response_data["content"][0]
        if "text" in content_item:
            return content_item["text"]
    return msg_invalid_response()

def process_google_response(response_data):
    # Placeholder for Google-specific response processing
    return response_data.get('text', '')

response_processors = {
    "gpt-4": process_openai_response,
    "gpt-4o": process_openai_response,
    "gpt-4o-mini": process_openai_response,
    "perplexity": process_openai_response,
    "mistral-nemo": process_openai_response,
    "mistral-small": process_openai_response,
    "mistral-large": process_openai_response,
    "claude-3.5-sonnet": process_claude_response,
    "gemini-1.5-flash": process_google_response,
}

# Load models.json for model routing
with open('models.json', 'r') as f:
    models_config = json.load(f)

def generate(model, prompt, context=None, keep_alive='30s'):
    start_time = time.time()

    # Find model in models.json config
    model_info = next((m for m in models_config['models'] if m['name'] == model), None)
    
    if model_info is None:
        raise ValueError(f"Model '{model}' not found in models.json")

    # Determine location (droplet, GPU, or API)
    model_location = model_info['location']

    # Set headers and data for Ollama models (local or GPU)
    if model_location == "droplet":
        ollama_url = ollama_droplet_url  # Assuming droplet_ollama_url is defined in config.py
    elif model_location == "gpu":
        ollama_url = ollama_gpu_url  # GPU Ollama server
    elif model_location == "api":
        # Handle API-based models
        if model in ["gpt-4", "gpt-4o", "claude-3.5-sonnet", "gemini-1.5-flash"]:
            return external_api_call(model, prompt)  # External API handling for GPT, Claude, Google
        else:
            raise ValueError(f"Unsupported API model: {model}")
    else:
        raise ValueError(f"Unsupported location for model '{model}'")

    # For Ollama models, make the request to the appropriate URL
    response = requests.post(ollama_url,
                             json={
                                 'model': model,
                                 'prompt': prompt,
                                 'keep_alive': keep_alive,
                                 'num_predict': num_predict,  # Assuming num_predict is defined
                                 'temperature': temperature,  # Assuming temperature is defined
                                 'top_p': top_p  # Assuming top_p is defined
                             },
                             stream=True)
    response.raise_for_status()

    response_parts = []
    for line in response.iter_lines():
        if line:
            body = json.loads(line)
            response_part = body.get('response', '')
            response_parts.append(response_part)
            print(f"{RESPONSE_COLOR}{response_part}{RESET_STYLE}", end='', flush=True)

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

def external_api_call(model, prompt):
    headers = {
        "Authorization": f"Bearer {os.getenv(model.upper() + '_API_KEY')}",
        "Content-Type": "application/json"
    }
    # Handle API-specific models like GPT-4, Claude, Gemini, etc.
    # Implement logic as already present for each of these APIs.
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    api_url = ""

    if model in ["gpt-4", "gpt-4o"]:
        # Handle GPT-4 specific logic (similar to your original code)
        api_url = openai_url  # Define this in your config.py
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        data.update({
            "max_tokens": openai_max_tokens,
            "temperature": openai_temperature,
            "messages": [{"role": "system", "content": openai_system_prompt},
                         {"role": "user", "content": prompt}]
        })

    elif model.startswith("claude"):
        # Claude API logic
        pass

    elif model.startswith("gemini"):
        # Google Gemini API logic
        pass

    # Make the API request for external models
    response = requests.post(api_url, json=data, headers=headers)
    response.raise_for_status()

    response_data = response.json()
    return process_response(model, response_data)  # Assuming process_response handles the response

def process_response(model, response_data):
    response_processor = response_processors.get(model)
    if response_processor:
        return response_processor(response_data)
    else:
        return msg_no_resp_processing()