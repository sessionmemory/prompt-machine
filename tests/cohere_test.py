import cohere
import os

# Fetch the API key from environment variables
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

def test_cohere():
    print("Testing Cohere API...")
    print(f"Using API Key: {COHERE_API_KEY}")  # Debug print for the API key
    co = cohere.ClientV2(COHERE_API_KEY)
    response = co.chat(
        model="command-r-plus",
        messages=[
            {
                "role": "user",
                "content": "hello world!"
            }
        ]
    )
    print(response)

if __name__ == "__main__":
    test_cohere()