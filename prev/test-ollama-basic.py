#!/usr/bin/env python3
# test-ollama-adapted.py
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.1.0"
__license__ = "MIT"

import json
import requests

# List of top 5 models
models = ["phi3:mini", "orca-mini", "qwen2:1.5b", "tinyllama", "gemma:2b"]

def generate(model, prompt, context):
    r = requests.post('http://localhost:11434/api/generate',
                      json={
                          'model': model,
                          'prompt': prompt,
                          'context': context,
                      },
                      stream=True)
    r.raise_for_status()

    response_parts = []
    for line in r.iter_lines():
        body = json.loads(line)
        response_part = body.get('response', '')
        response_parts.append(response_part)
        # the response streams one token at a time, print that as we receive it
        print(response_part, end='', flush=True)

        if 'error' in body:
            raise Exception(body['error'])

        if body.get('done', False):
            print()  # Print a newline after done
            return body['context'], ''.join(response_parts)

def main():
    context = []  # the context stores a conversation history, you can use this to make the model more context aware

    while True:
        # Display model options
        print("\nSelect a model to use:")
        for idx, model in enumerate(models):
            print(f"{idx + 1}. {model}")
        print("Enter 'exit' to stop the program.")

        # Get user model selection
        model_input = input("Enter the number of the model you want to use: ")
        if model_input.lower() == 'exit':
            break

        try:
            model_idx = int(model_input) - 1
            if model_idx < 0 or model_idx >= len(models):
                print("Invalid model number, please try again.")
                continue
        except ValueError:
            print("Invalid input, please enter a number.")
            continue

        selected_model = models[model_idx]

        # Get user prompt input
        prompt = input("Enter your prompt: ")
        if not prompt:
            print("Prompt cannot be empty, please try again.")
            continue

        print(f"\nResponse from model {selected_model}:")
        context, _ = generate(selected_model, prompt, context)

        while True:
            # Ask if user wants to use the same model or another one
            use_same_model = input("\nDo you want to use the same model? (yes/no): ").strip().lower()
            if use_same_model not in ['yes', 'no']:
                print("Invalid input, please enter 'yes' or 'no'.")
                continue

            if use_same_model == 'no':
                break

            # Ask if user wants to use the same prompt or another one
            use_same_prompt = input("Do you want to use the same prompt? (yes/no): ").strip().lower()
            if use_same_prompt not in ['yes', 'no']:
                print("Invalid input, please enter 'yes' or 'no'.")
                continue

            if use_same_prompt == 'no':
                prompt = input("Enter your new prompt: ")
                if not prompt:
                    print("Prompt cannot be empty, please try again.")
                    continue

            print(f"\nResponse from model {selected_model}:")
            context, _ = generate(selected_model, prompt, context)

if __name__ == "__main__":
    main()