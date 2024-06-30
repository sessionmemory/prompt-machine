#!/usr/bin/env python3
# ollama-prompt-tester.py
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

import json
import requests
import os
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Updated list of ollama models with sizes
models = [
    {"name": "dolphin-mistral", "size": "7b"},
    {"name": "gemma:2b", "size": "2b"},
    {"name": "llama3", "size": "8b"},
    {"name": "mistral", "size": "7b"},
    {"name": "phi3:mini", "size": "3b"},
    {"name": "qwen2", "size": "7b"},
    {"name": "samantha-mistral", "size": "7b"},
    {"name": "zephyr", "size": "7b"},
    {"name": "starling-lm", "size": "7b"},
    {"name": "wizardlm2", "size": "7b"},
    {"name": "dolphin-llama3", "size": "8b"}
]

def load_prompts(filename):
    if not os.path.exists(filename):
        return {}
    with open(filename, 'r') as f:
        data = json.load(f)
        return data.get('categories', {})

def load_all_prompts(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    all_prompts = []
    for category in data.get('categories', {}):
        all_prompts.extend(data['categories'][category])
    return all_prompts

def save_prompts(filename, prompts):
    with open(filename, 'w') as f:
        json.dump({"categories": prompts}, f, indent=4)

def select_model(models):
    print("\n\033[97m→ Select a model to use:\033[0m")
    for idx, model in enumerate(models):
        print(f"{idx + 1}. \033[1m{model['name']}\033[0m - {model['size']}")
    print("Enter 'exit' to stop the program.")

    while True:
        model_input = input("\033[97m→ Enter the number of the model you want to use: \033[0m").strip()
        if model_input.lower() == 'exit':
            return None
        try:
            model_idx = int(model_input) - 1
            if 0 <= model_idx < len(models):
                return models[model_idx]['name']  # Return the model name
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

def main_single_prompt_all_models():
    prompts_file = 'prompts.json'
    prompts = load_prompts(prompts_file)
    categories = list(prompts.keys())

    print("\nSelect a prompt category or enter a custom prompt:")
    for idx, category in enumerate(categories):
        print(f"{idx + 1}. {category}")
    print(f"{len(categories) + 1}. Enter a custom prompt.")

    category_input = input("\033[97m→ Enter your choice: \033[0m").strip()
    if category_input.lower() == 'exit':
        return

    try:
        category_idx = int(category_input) - 1
        if 0 <= category_idx < len(categories):
            selected_category = categories[category_idx]
            print("\nSelect a prompt:")
            category_prompts = prompts[selected_category]
            for idx, prompt_option in enumerate(category_prompts):
                print(f"{idx + 1}. {prompt_option}")
            prompt_input = input("\033[97m→ Enter the number of the prompt you want to use: \033[0m").strip()
            prompt_idx = int(prompt_input) - 1
            prompt = category_prompts[prompt_idx]
        elif category_idx == len(categories):  # Custom prompt option
            prompt = input("Enter your custom prompt: ").strip()
        else:
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input, please enter a number.")
        return

    for model in models:
        model_name = model['name']
        logging.info(f"Generating response for model \033[1m{model_name}\033[0m with prompt: {prompt}")
        print(f"\nResponse from model \033[1m{model_name}\033[0m:")
        try:
            context, response, response_time, char_count, word_count = generate(model_name, prompt)
            # Display and save the response as before
            print_response_stats(response, response_time, char_count, word_count)
            save_response(model_name, prompt, response, "", response_time, char_count, word_count)
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            print(f"Error generating response: {e}")
        time.sleep(10)  # 10-second break between models

def handle_custom_prompt(prompts, prompts_file):
    prompt = input("Enter your custom prompt: ")
    if not prompt:
        print("Prompt cannot be empty, please try again.")
        return None

    print("\nSelect a category to add your prompt to:")
    categories = list(prompts.keys())
    for idx, category in enumerate(categories):
        print(f"{idx + 1}. {category}")
    print("Or enter a new category name.")

    category_input = input("Enter the number or name of the category: ").strip()

    try:
        category_idx = int(category_input) - 1
        if 0 <= category_idx < len(categories):
            selected_category = categories[category_idx]
        else:
            print("Invalid category number, creating a new category.")
            selected_category = category_input
    except ValueError:
        selected_category = category_input

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
        "rating": rating,  # This will be an empty string if not provided
        "response_time": response_time,
        "char_count": char_count,
        "word_count": word_count
    })
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def generate(model, prompt, context=None, keep_alive='30s'):
    start_time = time.time()
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

    response_time = time.time() - start_time
    full_response = ''.join(response_parts)
    char_count = len(full_response)
    word_count = len(full_response.split())

    return body['context'], full_response, response_time, char_count, word_count

def get_user_rating():
    while True:
        rating = input("\033[97m→ Rate the response on a scale of 1-5 (5 being the best): \033[0m").strip()
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

def print_response_stats(response, response_time, char_count, word_count):
    # Similar to the existing code for displaying stats
    print(f"\n\033[1mResponse Time:\033[0m {response_time:.2f} seconds")
    print(f"\033[1mCharacter Count:\033[0m {char_count}")
    print(f"\033[1mWord Count:\033[0m {word_count}")
    character_rate = char_count / response_time if response_time > 0 else 0
    word_rate = word_count / response_time if response_time > 0 else 0
    print(f"\033[1mCharacter Rate:\033[0m {character_rate:.2f} characters per second")
    print(f"\033[1mWord Rate:\033[0m {word_rate:.2f} words per second\n")

def main_userselect():
    context = []
    prompts_file = 'prompts.json'
    prompts = load_prompts(prompts_file)
    selected_model = None
    prompt = None  # Initialize prompt to None to ensure it's selected in the first iteration

    while True:
        if not selected_model:
            selected_model = select_model(models)
            if selected_model is None:
                break  # Exit if the user chooses to exit during model selection

        # Move the prompt selection logic here, outside of the 'if not prompt' condition
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
            for idx, prompt_option in enumerate(category_prompts):
                print(f"{idx + 1}. \033[1m{prompt_option}\033[0m")
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

        logging.info(f"Generating response for model \033[1m{selected_model}\033[0m with prompt: {prompt}")
        print(f"\nResponse from model \033[1m{selected_model}\033[0m:")
        try:
            context, response, response_time, char_count, word_count = generate(selected_model, prompt, context)
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            print(f"Error generating response: {e}")
            continue

        if response_time > 60:
            minutes = int(response_time // 60)
            seconds = int(response_time % 60)
            formatted_response_time = f"{minutes} minutes and {seconds} seconds"
        else:
            formatted_response_time = f"{response_time:.2f} seconds"

        print(f"\n\033[1mResponse Time:\033[0m {formatted_response_time}")
        print(f"\033[1mCharacter Count:\033[0m {char_count}")
        print(f"\033[1mWord Count:\033[0m {word_count}")
        character_rate = char_count / response_time if response_time > 0 else 0
        word_rate = word_count / response_time if response_time > 0 else 0
        print(f"\033[1mCharacter Rate:\033[0m {character_rate:.2f} characters per second")
        print(f"\033[1mWord Rate:\033[0m {word_rate:.2f} words per second\n")
        
        if ask_to_save_response():
            rating = get_user_rating()
            save_response(selected_model, prompt, response, rating, response_time, char_count, word_count)

        # Ask if user wants to continue with the same model
        use_same_model = get_yes_or_no_input(f"\n\033[97m→ Do you want to continue with\033[0m \033[1m{selected_model}\033[0m \033[97mor select a different model? (y/n): \033[0m")
        if use_same_model == 'n':
            selected_model = None  # Reset selected_model to allow model selection
            # No need to ask if they want to use the same prompt since they're changing models

        # If 'y', continue with the same model but prompt will be re-selected in the next iteration

def main_queue():
    context = []
    prompts_file = 'prompts.json'
    all_prompts = load_all_prompts(prompts_file)
    selected_model = select_model(models)
    if selected_model is None:
        return  # Exit if the user chooses to exit during model selection

    for prompt in all_prompts:
        logging.info(f"Generating response for model \033[1m{selected_model}\033[0m with prompt: {prompt}")
        print(f"\nPrompt: {prompt}")
        try:
            context, response, response_time, char_count, word_count = generate(selected_model, prompt, context)
            # Display the response and stats
            print(f"\nResponse from model \033[1m{selected_model}\033[0m:")
            print(f"\033[32m{response}\033[0m")  # Assuming the response is not too large to print
            print(f"\n\033[1mResponse Time:\033[0m {response_time:.2f} seconds")
            print(f"\033[1mCharacter Count:\033[0m {char_count}")
            print(f"\033[1mWord Count:\033[0m {word_count}")
            character_rate = char_count / response_time if response_time > 0 else 0
            word_rate = word_count / response_time if response_time > 0 else 0
            print(f"\033[1mCharacter Rate:\033[0m {character_rate:.2f} characters per second")
            print(f"\033[1mWord Rate:\033[0m {word_rate:.2f} words per second\n")

            # Save the response without a rating
            save_response(selected_model, prompt, response, "", response_time, char_count, word_count)
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            print(f"Error generating response: {e}")
            continue

        time.sleep(10)  # 10-second break between prompts

def main():
    print("Select the mode you want to run:")
    print("1. User Select Mode")
    print("2. Queue Mode (Automatic)")
    print("3. Single Prompt Against All Models")
    mode_selection = input("Enter your choice (1, 2, or 3): ").strip()

    if mode_selection == '1':
        main_userselect()
    elif mode_selection == '2':
        main_queue()
    elif mode_selection == '3':
        main_single_prompt_all_models()
    else:
        print("Invalid selection. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
