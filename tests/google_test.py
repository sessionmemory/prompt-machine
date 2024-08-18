import os
import google.generativeai as genai
from config import google_model, google_max_tokens, google_temperature
google_api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=google_api_key)

def test_google():
    print("Testing Google Gemini API...")
    print(f"Using API Key: {google_api_key}")  # Print the API key for debugging purposes
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel(google_model)

    print(f"Generating content with model {google_model} using prompt 'Hello, world!'")
    response = model.generate_content(
        "Hello, world!",
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=google_max_tokens,
            temperature=google_temperature
        )
    )

    print("Google Response:", response.text)

if __name__ == "__main__":
    test_google()