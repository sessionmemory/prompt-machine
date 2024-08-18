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

def process_openai_response(response_data):
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
    "perplexity": process_openai_response,
    "mistral-nemo": process_openai_response,
    "claude-3.5-sonnet": process_claude_response,
    "gemini-1.5-flash": process_google_response,
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
        api_url = openai_url

    elif model.startswith("perplexity"):
        #print(f"Using API Key: {PPLX_API_KEY}")  # temp DEBUG
        # Use OpenAI API Client & Format for ChatGPT models
        headers = {
            "Authorization": f"Bearer {PPLX_API_KEY}",
            "Content-Type": "application/json"
        }
        data.update({
            "model": perplexity_model,
            "max_tokens": perplexity_max_tokens,
            "temperature": perplexity_temperature,
            "messages": [
                {"role": "system", "content": perplexity_system_prompt},
                {"role": "user", "content": prompt}
            ],
        })
        api_url = perplexity_url

    elif model.startswith("claude"):
            # Initialize the Anthropics client with your API key
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

            # Construct the message payload according to the Claude API requirements
            message = client.messages.create(
                model=claude_model,  # Use the model specified in your config
                max_tokens=claude_max_tokens,
                temperature=claude_temperature,  # Ensure you have this variable defined in your config
                system="You are a Claude, a helpful assistant.",  # Adjust this system prompt as needed
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            # Check if the message has content and process it
            if hasattr(message, 'content'):
                first_choice_content = ""
                for block in message.content:
                    if block.type == 'text':
                        first_choice_content += block.text
                response_time = time.time() - start_time
                print(f"{RESPONSE_COLOR}{first_choice_content}{RESET_STYLE}", flush=True)
                return None, first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())
            else:
                print("Error: No content in response")
                return None, "No content in response", time.time() - start_time, 0, 0
            
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
                print(f"{RESPONSE_COLOR}{first_choice_content}{RESET_STYLE}", flush=True)
            else:
                print("No valid parts in response.")
                first_choice_content = "No valid response generated."
        else:
            print("No candidates in response.")
            first_choice_content = "No valid response generated."

        response_time = time.time() - start_time
        return None, first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())

    elif model.startswith("mistral"):
        # Mistral API request setup
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": mistral_model,
            "temperature": mistral_temperature,
            "top_p": 1,
            "max_tokens": mistral_max_tokens,
            "min_tokens": mistral_min_tokens,
            "stream": False,
            "stop": "string",  # Adjust based on your needs
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

    else:
        # Handle Ollama server models
        # Assuming a local API endpoint for Ollama models
        response = requests.post(ollama_url,
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
        print(f"{RESPONSE_COLOR}{first_choice_content}{RESET_STYLE}", flush=True)

        return None, first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())