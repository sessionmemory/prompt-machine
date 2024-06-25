#!/usr/bin/env python3
# ollama-prompt-rate.py
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.2.0"
__license__ = "MIT"

import json
import requests
import os
import time

# List of top 5 models
models = ["phi3:mini", "orca-mini", "qwen2:1.5b", "tinyllama", "gemma:2b"]

def load_prompts(filename):
    if not os.path.exists(filename):
        return {}
    with open(filename, 'r') as f:
        data = json.load(f)
        return data.get('categories', {})

def save_prompts(filename, prompts):
    with open(filename, 'w') as f:
        json.dump({"categories": prompts}, f, indent=4)

def select_model(models):
    print("\n\033[97m→ Select a model to use:\033[0m")
    for idx, model in enumerate(models):
        print(f"{idx + 1}. \033[1m{model}\033[0m")
    print("Enter 'exit' to stop the program.")

    while True:
        model_input = input("\033[97m→ Enter the number of the model you want to use: \033[0m").strip()
        if model_input.lower() == 'exit':
            return None
        try:
            model_idx = int(model_input) - 1
            if 0 <= model_idx < len(models):
                return models[model_idx]
            else:
                print("Invalid model number, please try again.")
        except ValueError:
            print("Invalid input, please enter a number.")

def select_category(categories):
    print("\nSelect a category:")
    for idx, category in enumerate(categories):
        print(f"{idx + 1}. \033[1m{category}\033[0m")
    print("Enter '0' to enter a custom prompt.")
    print("Enter 'exit' to stop the program.")

    while True:
        category_input = input("\033[97m→ Enter the number of the category you want to use: \033[0m").strip()
        if category_input.lower() == 'exit':
            return None
        try:
            category_idx = int(category_input) - 1
            if category_idx == -1:
                return 'custom'
            elif 0 <= category_idx < len(categories):
                return categories[category_idx]
            else:
                print("Invalid category number, please try again.")
        except ValueError:
            print("Invalid input, please enter a number.")

def handle_custom_prompt(prompts, prompts_file):
    prompt = input("Enter your custom prompt: ")
    if not prompt:
        print("Prompt cannot be empty, please try again.")
        return None

    # Display existing categories with numerical options
    print("\nSelect a category to add your prompt to:")
    categories = list(prompts.keys())
    for idx, category in enumerate(categories):
        print(f"{idx + 1}. {category}")
    print("Or enter a new category name.")

    category_input = input("Enter the number or name of the category: ").strip()

    # Check if input is a number and corresponds to an existing category
    try:
        category_idx = int(category_input) - 1
        if 0 <= category_idx < len(categories):
            selected_category = categories[category_idx]
        else:
            print("Invalid category number, creating a new category.")
            selected_category = category_input
    except ValueError:
        # Input is not a number, treat it as a new category name
        selected_category = category_input

    # Add the prompt to the selected or new category
    if selected_category in prompts:
        prompts[selected_category].append(prompt)
    else:
        prompts[selected_category] = [prompt]

    save_prompts(prompts_file, prompts)
    return prompt

def save_response(model_name, prompt, response, rating, response_time, char_count, word_count):
    filename = f"{model_name.replace(':', '-')}.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = []
    
    data.append({
        "prompt": prompt,
        "response": response,
        "rating": rating,
        "response_time": response_time,
        "char_count": char_count,
        "word_count": word_count
    })
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def generate(model, prompt, context=None, keep_alive='30s'):
    start_time = time.time()  # Record the start time
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
            print()  # Print a newline after done
            break

    response_time = time.time() - start_time  # Calculate the response time
    full_response = ''.join(response_parts)
    char_count = len(full_response)
    word_count = len(full_response.split())

    return body['context'], full_response, response_time, char_count, word_count

def get_user_rating():
    while True:
        rating = input("Rate the response on a scale of 1-5 (5 being the best): ").strip()
        try:
            rating = int(rating)
            if 1 <= rating <= 5:
                return rating
            else:
                print("Invalid rating, please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input, please enter a number.")

def ask_to_save_response():
    save_response_input = input("\033[97m→ Do you want to save this response? (y/n): \033[0m").strip().lower()
    return save_response_input == 'y'

def get_yes_or_no_input(prompt):
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in ['y', 'n']:
            return user_input
        # If the input is not valid, re-prompt without a message

def main():
    context = []  # the context stores a conversation history, you can use this to make the model more context aware
    prompts_file = 'prompts.json'
    prompts = load_prompts(prompts_file)

    while True:
        selected_model = select_model(models)
        if selected_model is None:
            break  # Exit if the user chooses to exit during model selection

        categories = list(prompts.keys())
        selected_category = select_category(categories)
        if selected_category is None:
            break  # Exit if the user chooses to exit during category selection

        if selected_category == 'custom':
            prompt = handle_custom_prompt(prompts, prompts_file)
            if prompt is None:
                continue  # Skip the rest of the loop if no custom prompt is provided
        else:
            # Display prompt options within the selected category
            print("\nSelect a prompt:")
            category_prompts = prompts[selected_category]
            for idx, prompt in enumerate(category_prompts):
                print(f"{idx + 1}. \033[1m{prompt}\033[0m")
            print("Enter '0' to enter a custom prompt.")
            print("Enter 'exit' to stop the program.")

            prompt_input = input("\033[97m→ Enter the number of the prompt you want to use: \033[0m")
            if prompt_input.lower() == 'exit':
                break

            try:
                prompt_idx = int(prompt_input) - 1
                if prompt_idx == -1:
                    prompt = handle_custom_prompt(prompts, prompts_file)
                    if prompt is None:
                        continue  # Skip the rest of the loop if no custom prompt is provided
                elif 0 <= prompt_idx < len(category_prompts):
                    prompt = category_prompts[prompt_idx]
                else:
                    print("Invalid prompt number, please try again.")
                    continue
            except ValueError:
                print("Invalid input, please enter a number.")
                continue

        print(f"\nResponse from model \033[1m{selected_model}\033[0m:")
        context, response, response_time, char_count, word_count = generate(selected_model, prompt, context)

        # Format response time to display in minutes and seconds if it exceeds 60 seconds
        if response_time > 60:
            minutes = int(response_time // 60)
            seconds = int(response_time % 60)
            formatted_response_time = f"{minutes} minutes and {seconds} seconds"
        else:
            formatted_response_time = f"{response_time:.2f} seconds"

        print(f"\n\033[1mResponse Time:\033[0m {formatted_response_time}")
        print(f"\033[1mCharacter Count:\033[0m {char_count}")
        print(f"\033[1mWord Count:\033[0m {word_count}")
        # Calculate Character Rate (chars per second) and Word Rate (words per second)
        character_rate = char_count / response_time if response_time > 0 else 0
        word_rate = word_count / response_time if response_time > 0 else 0
        print(f"\n\033[1mCharacter Rate:\033[0m {character_rate:.2f} characters per second")
        print(f"\n\033[1mWord Rate:\033[0m {word_rate:.2f} words per second")
        
        # Ask if want to save, get rating then save
        if ask_to_save_response():
            rating = get_user_rating()
            save_response(selected_model, prompt, response, rating, response_time, char_count, word_count)

        # Ask if user wants to use the same model or another one, including the model name in the question
        use_same_model = get_yes_or_no_input(f"\n\033[97m→ Do you want to continue with\033[0m \033[1m{selected_model}\033[0m \033[97mor select a different model? (y/n): \033[0m")
        if use_same_model == 'n':
            continue  # This will restart the loop, allowing the user to select a new model

        use_same_prompt = get_yes_or_no_input("\033[97m→ Do you want to use the same prompt? (y/n): \033[0m")
        if use_same_prompt == 'n':
            continue  # This will restart the loop but keep the same model selected

if __name__ == "__main__":
    main()