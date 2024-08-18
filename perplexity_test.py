import requests
import os
from config import perplexity_url, perplexity_model, perplexity_max_tokens, perplexity_temperature

# Fetch the API key from environment variables
PPLX_API_KEY = os.getenv('PPLX_API_KEY')

def test_perplexity():
    print("Testing Perplexity API...")
    print(f"Using API Key: {PPLX_API_KEY}")  # Debug print for the API key
    
    # Set up the headers with the Authorization token
    headers = {"Authorization": f"Bearer {PPLX_API_KEY}"}
    
    # Adjust the payload to match the API's expected format
    payload = {
        "model": perplexity_model,
        "temperature": perplexity_temperature,
        "top_p": 1,
        "max_tokens": perplexity_max_tokens,
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

    print(f"Sending request to Perplexity with payload: {payload}")
    response = requests.post(perplexity_url, json=payload, headers=headers)
    if response.status_code == 200:
        print("Mistral Response:", response.json())
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    test_perplexity()