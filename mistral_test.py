import requests
import os
from config import mistral_url, mistral_model, mistral_max_tokens, mistral_temperature

# Fetch the API key from environment variables
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

def test_mistral():
    print("Testing Mistral API...")
    print(f"Using API Key: {MISTRAL_API_KEY}")  # Debug print for the API key
    
    # Set up the headers with the Authorization token
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    
    # Adjust the payload to match the API's expected format
    payload = {
        "model": mistral_model,
        "temperature": mistral_temperature,
        "top_p": 1,
        "max_tokens": mistral_max_tokens,
        "min_tokens": 10,
        "stream": False,
        "stop": "\n",  # Adjust based on your needs
        "random_seed": None,  # Optional, set this if you want deterministic results
        "messages": [
            {
                "role": "user",
                "content": "Hello, world!"
            }
        ],
        "response_format": {"type": "text"},  # Assuming you want text responses
        "tools": [],  # Adjust if you're using any tools
        "tool_choice": "auto",
        "safe_prompt": False
    }

    print(f"Sending request to Mistral with payload: {payload}")
    response = requests.post(mistral_url, json=payload, headers=headers)
    if response.status_code == 200:
        print("Mistral Response:", response.json())
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    test_mistral()