import cohere
import os

# Fetch the API key from environment variables
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

def test_cohere():
    print("Testing Cohere API...")
    print(f"Using API Key: {COHERE_API_KEY}")  # Debug print for the API key



if __name__ == "__main__":
    test_cohere()