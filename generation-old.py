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

def process_api_response(response_data, model):
    # Placeholder values for demonstration; adjust as necessary
    response_time = 0  # Placeholder, calculate or pass the actual response time if available
    processed_text = ""

    # Check if the model is from OpenAI or Mistral and adjust the extraction accordingly
    if model in ["gpt-4", "gpt-4o", "gpt-4o-mini", "mistral-nemo", "mistral-large", "mistral-small"]:
        if 'choices' in response_data and len(response_data['choices']) > 0 and 'text' in response_data['choices'][0]:
            processed_text = response_data['choices'][0]['text'].strip()
    else:
        # Handle other models or return a default message
        processed_text = msg_invalid_response()

    char_count = len(processed_text)
    word_count = len(processed_text.split())

    return None, processed_text, response_time, char_count, word_count

def process_claude_response(response_data):
    # Assuming the response_data is the JSON-decoded response
    if "content" in response_data and response_data["content"]:
        # Assuming the first item in the content list is the one we're interested in
        content_item = response_data["content"][0]
        if "text" in content_item:
            processed_text = content_item["text"]
            # Placeholder values for demonstration; adjust as necessary
            response_time = 0  # Placeholder, calculate or pass the actual response time if available
            char_count = len(processed_text)
            word_count = len(processed_text.split())
            return None, processed_text, response_time, char_count, word_count
    # Adjust the return value to match the expected tuple format
    return None, msg_invalid_response(), 0, 0, 0

response_processors = {
    "gpt-4": process_api_response,
    "gpt-4o": process_api_response,
    "gpt-4o-mini": process_api_response,
    "o1-mini": process_api_response,
    "o1-preview": process_api_response,
    "perplexity": process_api_response,
    "mistral-nemo": process_api_response,
    "mistral-small": process_api_response,
    "mistral-large": process_api_response,
    "claude-3.5-sonnet": process_claude_response,
    "gemini-1.5-flash": process_api_response,
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
        if model in ["gpt-4", "gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview", "mistral-nemo", "mistral-small", "mistral-large", "perplexity", "claude-3.5-sonnet", "gemini-1.5-flash"]:
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
    start_time = time.time()
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

    if model in ["gpt-4", "gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview"]:
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
        # Initialize the Anthropics client with your API key
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Construct the message payload according to the Claude API requirements
        message = client.messages.create(
            model=claude_model,  # Use the model specified in your config
            max_tokens=claude_max_tokens,
            temperature=claude_temperature,  # Ensure you have this variable defined in your config
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        # Process the response to extract text from each TextBlock
        if message.content:  # Check if content is not empty
            first_choice_content = ' '.join([block.text for block in message.content if block.type == 'text'])
        else:
            first_choice_content = msg_invalid_response()

        response_time = time.time() - start_time
        print(msg_content(first_choice_content), flush=True)

        # Calculate character and word counts
        char_count = len(first_choice_content)
        word_count = len(first_choice_content.split())

        return None, first_choice_content, response_time, char_count, word_count
     
    elif model.startswith("gemini"):
        # Configure the Google API with the API key
        genai.configure(api_key=google_api_key)

        # Initialize the Google model
        google_model_instance = genai.GenerativeModel(google_model)

        # Generate content without streaming
        response = google_model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,  # Currently, only one candidate is supported
                max_output_tokens=google_max_tokens,  # Adjust based on your needs
                temperature=google_temperature,  # Adjust for creativity level
            ),
        )
        # Process the response
        if response.candidates:
            candidate = response.candidates[0]  # Assuming you're interested in the first candidate
            if candidate.content and candidate.content.parts:
                first_choice_content = ''.join(part.text for part in candidate.content.parts if part.text)
                print(msg_content(first_choice_content), flush=True)
            else:
                print(msg_invalid_response())
                first_choice_content = msg_invalid_response()
        else:
            print(msg_invalid_response())
            first_choice_content = msg_invalid_response()

        response_time = time.time() - start_time
        return None, first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())

    elif model in ["mistral-nemo", "mistral-large", "mistral-small"]:
        # Mistral API request setup for specific models
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        # Select the correct model name based on the input
        if model == "mistral-nemo":
            mistral_model_name = mistral_nemo_model
        elif model == "mistral-large":
            mistral_model_name = mistral_large_model
        elif model == "mistral-small":
            mistral_model_name = mistral_small_model

        data = {
            "model": mistral_model_name,
            "temperature": mistral_temperature,
            "top_p": 1,
            "max_tokens": mistral_max_tokens,
            "min_tokens": mistral_min_tokens,
            "stream": False,
            #"stop": "string",  # Adjust based on your needs
            "random_seed": None,  # Optional, for deterministic results
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "text"},  # Assuming text responses
            "tools": [],  # Adjust if using any tools
            "tool_choice": "auto",
            "safe_prompt": False
        }
        api_url = mistral_url

    elif model.startswith("perplexity"):
        headers = {
            "Authorization": f"Bearer {PPLX_API_KEY}",
            "Content-Type": "application/json",
            "accept": "application/json"  # Ensure headers are correctly set
        }
        data = {
            "model": perplexity_model,
            "max_tokens": perplexity_max_tokens,
            "temperature": perplexity_temperature,
            "messages": [
                {"role": "system", "content": perplexity_system_prompt},  # Adjust based on your system prompt needs
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(perplexity_url, json=data, headers=headers)
        try:
            response.raise_for_status()  # Check for HTTP request errors
            response_data = response.json()
            # Assuming the response structure is similar to OpenAI's, adjust as needed
            if 'choices' in response_data and response_data['choices']:
                first_choice = response_data['choices'][0]
                if 'message' in first_choice and 'content' in first_choice['message']:
                    first_choice_content = first_choice['message']['content']
                else:
                    first_choice_content = msg_invalid_response()
            else:
                first_choice_content = msg_invalid_response()
        except requests.exceptions.HTTPError as e:
            print(msg_error_simple(e))
            first_choice_content = msg_error_simple(e)

        response_time = time.time() - start_time
        print(msg_content(first_choice_content), flush=True)
        response_data = response.json()
        return process_api_response(response_data, model)

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
        # Ensure a consistent return format even for this case
        return None, msg_no_resp_processing(), 0, 0, 0  # Adjust placeholder values as necessary