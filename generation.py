#!/usr/bin/env python3
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

# generation.py
import requests
import json
import time
from config import *

def generate(model, prompt, context=None, keep_alive='30s'):
    start_time = time.time()

    if model in ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]:
        # Use OpenAI API for ChatGPT models
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "max_tokens": 1000,
            "stream": "true",
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()

        response_data = response.json()
        first_choice_content = response_data['choices'][0]['message']['content'].strip()

        response_time = time.time() - start_time  # Calculate response time

        # Print the response content in a manner similar to streaming
        print(f"\033[32m{first_choice_content}\033[0m", flush=True)

        return None, first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())
    else:
        # Handle other models or local API calls
        r = requests.post('http://localhost:11434/api/generate',
                        json={
                            'model': model,
                            'prompt': prompt,
                            'keep_alive': keep_alive
                        },
                        stream=True)
        r.raise_for_status()

        response_parts = []
        for line in r.iter_lines():
            body = json.loads(line)
            response_part = body.get('response', '')
            response_parts.append(response_part)
            print(f"\033[32m{response_part}\033[0m", end='', flush=True)

            if 'error' in body:
                raise Exception(body['error'])

            if body.get('done', False):
                print()
                break

        response_time = time.time() - start_time  # Calculate response time
        full_response = ''.join(response_parts)
        char_count = len(full_response)
        word_count = len(full_response.split())

        return body.get('context', None), full_response, response_time, char_count, word_count