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
    if "completion" in response_data and response_data["completion"]:
        return response_data["completion"].strip()
    return msg_invalid_response()

def process_google_response(response_data):
    # Google response processing
    return response_data.get('text', '')

response_processors = {
    "gpt-4": process_openai_response,
    "gpt-4-turbo": process_openai_response,
    "gpt-4o": process_openai_response,
    "gpt-4o-mini": process_openai_response,
    "o1-mini": process_openai_response,
    "o1-preview": process_openai_response,
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

def external_api_call(model, prompt):
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

    if model in ["gpt-4", "gpt-4-turbo", "gpt-4o", "o1-preview", "o1-mini", "gpt-4o-mini"]:
        # Use OpenAI API for ChatGPT models
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        data.update({
            "model": model,
            "max_tokens": openai_max_tokens,
            "temperature": openai_temperature,
            "messages": [
                {"role": "system", "content": openai_system_prompt},
                {"role": "user", "content": prompt}
            ],
        })
        api_url = openai_url
    
    elif model.startswith("claude"):
        # Initialize the Anthropics client with your API key
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.completions.create(
            model=claude_model,
            max_tokens=claude_max_tokens,
            temperature=claude_temperature,
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}"
        )
        first_choice_content = message.completion.strip()
        response_time = time.time() - start_time
        return None, first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())
     
    elif model.startswith("gemini"):
        genai.configure(api_key=google_api_key)
        response = genai.generate_text(prompt=prompt, model=google_model, max_tokens=google_max_tokens, temperature=google_temperature)
        if response.candidates:
            first_choice_content = response.candidates[0].text.strip()
            response_time = time.time() - start_time
            return None, first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())
        else:
            return None, msg_invalid_response(), 0, 0, 0

    elif model in ["mistral-nemo", "mistral-large", "mistral-small"]:
        headers["Authorization"] = f"Bearer {MISTRAL_API_KEY}"
        if model == "mistral-nemo":
            mistral_model_name = mistral_nemo_model
        elif model == "mistral-large":
            mistral_model_name = mistral_large_model
        elif model == "mistral-small":
            mistral_model_name = mistral_small_model
        data.update({
            "model": mistral_model_name,
            "temperature": mistral_temperature,
            "top_p": 1,
            "max_tokens": mistral_max_tokens,
            "min_tokens": mistral_min_tokens,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        })
        api_url = mistral_url

    elif model.startswith("perplexity"):
        headers["Authorization"] = f"Bearer {PPLX_API_KEY}"
        data.update({
            "model": perplexity_model,
            "max_tokens": perplexity_max_tokens,
            "temperature": perplexity_temperature,
            "messages": [{"role": "user", "content": prompt}]
        })
        api_url = perplexity_url

    try:
        response = requests.post(api_url, json=data, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        response_processor = response_processors.get(model)
        if response_processor:
            first_choice_content = response_processor(response_data)
            response_time = time.time() - start_time
            return None, first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())
    except requests.exceptions.RequestException as e:
        print(f"HTTPError for model {model}: {str(e)}")
        return None, msg_error_simple(e), 0, 0, 0

def generate(model, prompt, context=None, keep_alive='30s'):
    start_time = time.time()

    # Find model in models.json config
    model_info = next((m for m in models_config['models'] if m['name'] == model), None)
    
    if model_info is None:
        raise ValueError(f"Model '{model}' not found in models.json")

    # Determine location (droplet, GPU, or API)
    model_location = model_info['location']

    if model_location == "droplet":
        ollama_url = ollama_droplet_url  # Assuming droplet_ollama_url is defined in config.py
    elif model_location == "gpu":
        ollama_url = ollama_gpu_url  # GPU Ollama server
    elif model_location == "api":
        return external_api_call(model, prompt)
    else:
        raise ValueError(f"Unsupported location for model '{model}'")

    # Handle Ollama server models
    response = requests.post(ollama_url,
                                json={
                                    'model': model,
                                    'prompt': prompt,
                                    'keep_alive': keep_alive,
                                    'num_predict': num_predict,  # Limiting the token output
                                    'temperature': temperature,
                                    'top_p': top_p
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