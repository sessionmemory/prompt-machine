#!/usr/bin/env python3
# test-ollama.py
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.1.0"
__license__ = "MIT"

import json
import requests
import time
import socket

# Function to check if the server is ready
def is_server_ready(port, host='localhost', timeout=2):
    print(f"Checking if server is ready on {host}:{port}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        if result == 0:
            print("Server is ready!")
            return True
        else:
            print(f"Server not ready, connect_ex returned {result}")
            return False

def get_response_from_ollama(model, prompt):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}
    try:
        print(f"Sending request to model {model}...")
        response = requests.post(url, json=payload, timeout=30)  # Added timeout of 30 seconds
        response.raise_for_status()  # Raises an HTTPError if the response was an error
        print(f"Response received from model {model}")
        return response.json()
    except requests.Timeout:
        print(f"Request to model {model} timed out.")
        return None
    except requests.RequestException as e:
        print(f"Error fetching response from model {model}: {e}")
        return None

# List of top 5 models
models = ["phi3:mini", "orca-mini", "qwen2:1.5b", "tinyllama", "gemma:2b"]

if __name__ == "__main__":
    # Check if the Ollama server is ready
    for _ in range(120):  # Try for up to 2 minutes
        if is_server_ready(11434):  # Adjust the port if necessary
            break
        time.sleep(1)
    else:
        print("Ollama server failed to start in time.")
        exit(1)
    
    prompt = "Can you explain the significance of the Kyoto Protocol in addressing climate change and list three key mechanisms it introduced?"
    for model in models:
        print(f"Testing model: {model}")
        response = get_response_from_ollama(model, prompt)
        if response:
            print(f"Response from {model}: {response}\n")
        else:
            print(f"No response from model {model}")