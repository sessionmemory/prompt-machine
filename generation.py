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
import cohere
import logging
from user_messages import *

# Suppress INFO logs from the `requests` library
logging.basicConfig(level=logging.WARNING)

def process_openai_response(response_data):
    return response_data['choices'][0]['message']['content'].strip()

def process_claude_response(response_data):
    if "content" in response_data and response_data["content"]:
        content_item = response_data["content"][0]
        if "text" in content_item:
            return content_item["text"]
    return msg_invalid_response()

def process_google_response(response_data):
    if 'candidates' in response_data and response_data['candidates']:
        candidate = response_data['candidates'][0]
        return candidate.get('content', '').strip()
    return msg_invalid_response()

def process_cohere_response(response):
    # Assuming response is an object returned by the Cohere SDK's chat method
    if hasattr(response, 'message') and hasattr(response.message, 'content'):
        # Iterate through the content list in the message
        for item in response.message.content:
            if item.type == "text":
                return item.text
    return msg_invalid_response()

response_processors = {
    "gpt-4o": process_openai_response,
    "gpt-4o-mini": process_openai_response,
    "perplexity": process_openai_response,
    "mistral-nemo": process_openai_response,
    "mistral-small": process_openai_response,
    "mistral-large": process_openai_response,
    "claude-3.5-sonnet": process_claude_response,
    "gemini-1.5-flash": process_google_response,
    "cohere_command_r_plus": process_cohere_response,
    "cohere_command_r": process_cohere_response,
    "cohere_command": process_cohere_response,
    "cohere_aya_35b": process_cohere_response,
    "cohere_aya_8b": process_cohere_response,
}

def generate(model, prompt, current_mode, keep_alive='30s'):
    start_time = time.time()

    # Fetch the pre-prompt based on the current mode
    pre_prompt = preprompt_modes.get(current_mode, "")
    # Combine pre-prompt with the actual prompt
    full_prompt = f"{pre_prompt}: {prompt}"

    headers = {
        "Authorization": f"Bearer {os.getenv(model.upper() + '_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    api_url = ""

    # Load models.json for model routing
    with open('models.json', 'r') as f:
        models_config = json.load(f)
    
    # Determine the model's location and set the appropriate URL or handle accordingly
    model_info = next((item for item in models_config['models'] if item["name"] == model), None)  # Adjusted for list of dicts
    if model_info is None:
        raise ValueError(f"Model '{model}' not found in models.json")

    model_location = model_info['location']

    # Set URLs based on model location
    if model_location == "droplet":
        ollama_url = ollama_droplet_url  # Assuming ollama_droplet_url is defined in config.py
    elif model_location == "gpu":
        ollama_url = ollama_gpu_url  # GPU Ollama server URL, also defined in config.py
    else:
        ollama_url = None  # For API models, we'll just pass through to the API

    # OpenAI / ChatGPT API calls
    if model in ["gpt-4", "gpt-4o-mini"]:
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
                {"role": "user", "content": full_prompt}
            ],
        })
        api_url = openai_url
    
    elif model.startswith("claude"):
        # Initialize the Anthropic's client with your API key
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Construct the message payload according to the Claude API requirements
        message = client.messages.create(
            model=claude_model,  # Use the model specified in your config
            max_tokens=claude_max_tokens,
            temperature=claude_temperature,  # Ensure you have this variable defined in your config
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
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

        return first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())

    elif model.startswith("cohere_"):
        # Initialize the Cohere client
        co = cohere.ClientV2(COHERE_API_KEY)  # Ensure COHERE_API_KEY is correctly set

        # Prepare the prompt for the Cohere API
        messages = [{
            "role": "user",
            "content": full_prompt
        }]

        # Use the correct model name from your config
        model_name = globals().get(model, "command-r-plus")  # Fallback to a default model if not found

        # Make the API call using the Cohere SDK
        response = co.chat(
            model=model_name,
            messages=messages,
            max_tokens=cohere_max_tokens,
            temperature=cohere_temperature
        )

        # Process the response directly without converting to JSON
        response_content = process_cohere_response(response)

        response_time = time.time() - start_time
        
        # Format the response content with color before printing
        formatted_response_content = msg_content(response_content)
        print(formatted_response_content, flush=True)
        
        return response_content, response_time, len(response_content), len(response_content.split())

    elif model.startswith("gemini"):
        # Gemini-specific logic
        try:
            # Configure the Google API with the API key
            genai.configure(api_key=google_api_key)

            # Initialize the Google model instance
            google_model_instance = genai.GenerativeModel(google_model)

            # Generate content without streaming
            response = google_model_instance.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,  # Currently, only one candidate is supported
                    max_output_tokens=google_max_tokens,  # Adjust this based on your needs
                    temperature=google_temperature,  # Adjust for creativity level
                ),
            )

            # Process the response
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    # Join text parts to form the full response
                    first_choice_content = ''.join(part.text for part in candidate.content.parts if part.text)
                    print(msg_content(first_choice_content), flush=True)
                else:
                    print(msg_invalid_response())
                    first_choice_content = msg_invalid_response()
            else:
                print(msg_invalid_response())
                first_choice_content = msg_invalid_response()

            response_time = time.time() - start_time
            return first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())

        except Exception as e:
            print(f"❌ Error with Gemini model: {e}")
            return "No valid response", time.time() - start_time, 0, 0

    elif model in ["mistral-nemo", "mistral-large", "mistral-small"]:
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
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
            "messages": [{"role": "user", "content": full_prompt}],
            "response_format": {"type": "text"},
            "safe_prompt": False
        }
        api_url = mistral_url

        response = requests.post(api_url, json=data, headers=headers)
        response.raise_for_status()
        response_data = response.json()

    elif model.startswith("perplexity"):
        headers = {
            "Authorization": f"Bearer {PPLX_API_KEY}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        # Dynamically select the Perplexity model based on the provided model name
        if model == "perplexity-sonar-small":
            perplexity_model_name = perplexity_small
        elif model == "perplexity-sonar-large":
            perplexity_model_name = perplexity_large
        elif model == "perplexity-sonar-chat":
            perplexity_model_name = perplexity_chat
        elif model == "perplexity-sonar-huge":
            perplexity_model_name = perplexity_huge
        else:
            print("Invalid Perplexity model specified.")
            return None, "Invalid Perplexity model specified.", 0, 0, 0

        data = {
            "model": perplexity_model_name,
            "max_tokens": perplexity_max_tokens,
            "temperature": perplexity_temperature,
            "messages": [
                {"role": "system", "content": perplexity_system_prompt},
                {"role": "user", "content": full_prompt}
            ]
        }
        response = requests.post(perplexity_url, json=data, headers=headers)
        response.raise_for_status()
        response_data = response.json()

        if 'choices' in response_data and response_data['choices']:
            first_choice = response_data['choices'][0]
            if 'message' in first_choice and 'content' in first_choice['message']:
                first_choice_content = first_choice['message']['content']
            else:
                first_choice_content = msg_invalid_response()
        else:
            first_choice_content = msg_invalid_response()

        response_time = time.time() - start_time
        print(msg_content(first_choice_content), flush=True)

        return first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())

    # Handle Ollama server models
    if ollama_url:
        response = requests.post(ollama_url, json={
            'model': model,
            'prompt': full_prompt,
            'keep_alive': keep_alive,
            'num_predict': num_predict,
            'temperature': temperature,
            'top_p': top_p
        }, stream=True)
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

        return full_response, response_time, len(full_response), len(full_response.split())

    # API request for external models
    if api_url:
            response = requests.post(api_url, json=data, headers=headers)
            response.raise_for_status()
            response_data = response.json()

            response_processor = response_processors.get(model)
            first_choice_content = response_processor(response_data) if response_processor else None
            
            # Ensure first_choice_content is a string before further processing
            if not isinstance(first_choice_content, str):
                first_choice_content = str(first_choice_content) if first_choice_content else "No valid response"

            response_time = time.time() - start_time
            
            print(msg_content(first_choice_content), flush=True)
            
            # Return safe values
            return first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())