import anthropic
import os

# Fetch the API key from environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

def test_claude():
    print("Testing Claude API...")
    print(f"Using API Key: {ANTHROPIC_API_KEY}")  # Debug print for the API key

    # Initialize the Anthropics client with your API key
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Construct the message payload
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are a world-class poet. Respond only with short poems.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Why is the ocean salty?"
                    }
                ]
            }
        ]
    )

    # Print the response content
    if hasattr(message, 'content'):
        for block in message.content:
            if block.type == 'text':
                print(block.text)
    else:
        print("Error: No content in response")

if __name__ == "__main__":
    test_claude()