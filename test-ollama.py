#!/usr/bin/env python3
# test-ollama.py
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

def multi_selection_input(prompt, items):
    while True:
        print(prompt)
        for idx, item in enumerate(items, start=1):
            print(f"{idx}. {item}")
        selection_input = input("Enter your choices (e.g., 1-3,5,7-8,10): ").strip()

        # Process the input to support ranges
        selected_indices = []
        for part in selection_input.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                selected_indices.extend(range(start, end + 1))
            else:
                selected_indices.append(int(part))

        # Deduplicate and sort the indices
        selected_indices = sorted(set(selected_indices))

        # Validate selection
        try:
            selected_items = [items[idx - 1] for idx in selected_indices]
            print("You have selected: ")
            for item in selected_items:
                print(f"- {item}")
            if confirm_selection():
                return selected_items
        except (ValueError, IndexError):
            print("Invalid selection, please try again.")

def confirm_selection(message="Confirm your selection? (y/n): "):
    while True:
        confirm = input(message).strip().lower()
        if confirm == 'y':
            return True
        elif confirm == 'n':
            return False
        else:
            print("Please enter 'y' or 'n'.")

def load_models(filename='models.json'):
    if not os.path.exists(filename):
        logging.error(f"Models file {filename} not found.")
        return []
    with open(filename, 'r') as f:
        data = json.load(f)
        return data.get('models', [])

# Use the load_models function to load the models
models = load_models()

def load_prompts(filename, flat=False):
    if not os.path.exists(filename):
        return {} if not flat else []
    with open(filename, 'r') as f:
        data = json.load(f)
    if flat:
        all_prompts = []
        for category in data.get('categories', {}):
            all_prompts.extend(data['categories'][category])
        return all_prompts
    else:
        return data.get('categories', {})

def save_prompts(filename, prompts):
    with open(filename, 'w') as f:
        json.dump({"categories": prompts}, f, indent=4)

def select_model(models, allow_multiple=False):
    while True:
        if allow_multiple:
            print("\n→ Select the model(s) to use (e.g., 1-4, 5, 7), or type 'exit' to quit:")
            print("You can select multiple models by separating numbers with commas or specifying ranges (e.g., 1-3,5).")
        else:
            print("\n→ Select the model to use (e.g., 4), or type 'exit' to quit:")
        
        for idx, model in enumerate(models, start=1):
            print(f"{idx}. {model['name']}")
        model_selection = input("→ Enter your model selection: ").strip()
        if model_selection.lower() == 'exit':
            return None  # User chose to exit

        if not allow_multiple and ("," in model_selection or "-" in model_selection):
            print("Invalid input, please enter a single number without commas or dashes.")
            continue

        try:
            selected_indices = []
            if allow_multiple:
                for part in model_selection.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        selected_indices.extend(range(start - 1, end))  # Convert to 0-based index
                    else:
                        selected_indices.append(int(part) - 1)  # Convert to 0-based index
            else:
                # For single selection, directly convert the input to an integer
                selected_index = int(model_selection) - 1
                if 0 <= selected_index < len(models):
                    selected_indices.append(selected_index)
                else:
                    raise ValueError("Selection out of range.")

            # Validate and deduplicate selected indices
            selected_indices = list(set(selected_indices))  # Remove duplicates
            selected_models = [models[i]['name'] for i in selected_indices]
            print(f"You have selected: {', '.join(selected_models)}")
            if confirm_selection():
                return selected_models
        except ValueError:
            print("Invalid input, please enter a valid selection or type 'exit'.")

def select_prompts(prompts):
    print("\n→ Select prompt(s):")
    selected_prompts = multi_selection_input("→ Enter your choices: ", prompts)
    if not selected_prompts:
        print("No prompts selected, exiting.")
        return None
    return selected_prompts

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
        elif category_input == '':
            return [category['name'] for category in categories]  # Select all categories
        try:
            category_idx = int(category_input) - 1
            if category_idx == -1:
                selected_category = 'custom'
            elif 0 <= category_idx < len(categories):
                selected_category = categories[category_idx]
                # Confirmation step
                if not confirm_selection(f"Confirm your category selection '{selected_category}'? (y/n): "):
                    print("Selection not confirmed. Please try again.")
                    return select_category(categories)  # Re-select if not confirmed
            else:
                print("Invalid category number, please try again.")
                return select_category(categories)
        except ValueError:
            print("Invalid input, please enter a number.")
            return select_category(categories)
        return selected_category

def handle_custom_prompt(prompts, prompts_file):
    while True:
        prompt = input("Enter your custom prompt: ").strip()
        if prompt:
            # Proceed with adding the prompt to a category
            break
        else:
            print("Prompt cannot be empty, please try again.")

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

def generate(model, prompt, context=None, keep_alive='30s'):
    start_time = time.time()

    if model in ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]:
        # Use OpenAI API for ChatGPT models
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()

        response_data = response.json()
        first_choice_content = response_data['choices'][0]['message']['content'].strip()

        response_time = time.time() - start_time  # Calculate response time

        # Print the response content in a manner similar to streaming
        print(f"\033[32m{first_choice_content}\033[0m", flush=True)

        return None, first_choice_content, response_time, len(first_choice_content), len(first_choice_content.split())
    else:
        # Handle other models or local API calls
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

        response_time = time.time() - start_time  # Calculate response time
        full_response = ''.join(response_parts)
        char_count = len(full_response)
        word_count = len(full_response.split())

        return body.get('context', None), full_response, response_time, char_count, word_count

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
    return confirm_selection("\033[97m→ Do you want to save this response? (y/n): \033[0m")

def save_response(model_name, prompt, response, rating, response_time, char_count, word_count):
    # Replace slashes in the model name with hyphens
    filename_safe_model_name = model_name.replace('/', '-')
    filename = f"{filename_safe_model_name}.json"
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
            selected_model_names = select_model(models, allow_multiple=False)
            if selected_model_names is None:
                print("Exiting.")
                break  # Exit the loop and end the program
            selected_model = selected_model_names[0]  # Assuming only one model is selected for this option, take the first item

        # Use select_category function for consistent category selection
        selected_category = select_category(list(prompts.keys()))
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
            for idx, prompt_option in enumerate(category_prompts, start=1):
                print(f"{idx}. {prompt_option}")
            prompt_selection = input("→ Enter the number of the prompt you want to use: ").strip()
            try:
                prompt_idx = int(prompt_selection) - 1
                prompt = category_prompts[prompt_idx]
                print(f"You have selected:\n- {prompt}")
                if not confirm_selection():
                    continue
            except (ValueError, IndexError):
                print("Invalid selection, please try again.")
                continue
            
        logging.info(f"Generating response for model \033[1m{selected_model}\033[0m with prompt: {prompt}")
        print(f"\nResponse from model \033[1m{selected_model}\033[0m:")
        try:
            # Ensure selected_model is a string, not a list
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
        use_same_model = confirm_selection(f"\n\033[97m→ Do you want to continue with\033[0m \033[1m{selected_model}\033[0m \033[97mor select a different model? (y/n): \033[0m")
        if use_same_model:
            # If 'y', continue with the same model but prompt will be re-selected in the next iteration
            continue
        else:
            # If 'n', reset selected_model to allow model selection
            selected_model = None

def main_model_prompt_selection_sequence():
    prompts = load_prompts('prompts.json')  # Loads prompts categorized
    selected_models = select_model(models, allow_multiple=True)
    if not selected_models:
        print("No models selected, exiting.")
        return

    # New Step: Select a prompt category first
    categories = list(prompts.keys())
    print("\nSelect a prompt category:")
    selected_category = select_category(categories)
    if not selected_category:
        print("Exiting.")
        return

    # Display prompts within the selected category
    category_prompts = prompts[selected_category]
    print(f"\nSelect prompts from the category '{selected_category}':")
    selected_prompts = multi_selection_input("→ Enter your choices: ", category_prompts)
    if not selected_prompts:
        print("No prompts selected, exiting.")
        return

    for model_name in selected_models:
        for prompt in selected_prompts:
            print(f"\nGenerating response for model {model_name} with prompt: {prompt}")
            try:
                context, response, response_time, char_count, word_count = generate(model_name, prompt, None)
                print_response_stats(response, response_time, char_count, word_count)
            except Exception as e:
                logging.error(f"Error generating response: {e}")
                print(f"Error generating response: {e}")
            time.sleep(1)  # Adjust sleep time as needed

def main_model_category_selection_sequence():
    prompts = load_prompts('prompts.json')  # Loads prompts categorized
    selected_models = select_model(models, allow_multiple=True)
    if not selected_models:
        print("No models selected, exiting.")
        return

    categories = list(prompts.keys())
    selected_category = select_category(categories)
    if selected_category is None:
        print("Exiting.")
        return
    elif selected_category == 'custom':
        # Handle custom prompt logic here
        print("Custom prompt logic not shown for brevity.")
        return
    else:
        # Use the selected category name directly to access prompts
        category_prompts = prompts[selected_category]
        for model_name in selected_models:
            for prompt in category_prompts:
                print(f"\nGenerating response for model {model_name} with prompt: {prompt}")
                try:
                    context, response, response_time, char_count, word_count = generate(model_name, prompt, None)
                    print_response_stats(response, response_time, char_count, word_count)
                except Exception as e:
                    logging.error(f"Error generating response: {e}")
                    print(f"Error generating response: {e}")
                time.sleep(1)  # Adjust sleep time as needed

def main():
    print("Select the mode you want to run:")
    print("1. Single Prompt, Model, and Rate")
    print("2. Model & Prompt Selection (Sequence)")
    print("3. Model & Category Selection (Sequence)")
    mode_selection = input("Enter your choice (1, 2, 3): ").strip()

    if mode_selection == '1':
        main_userselect()
    elif mode_selection == '2':
        main_model_prompt_selection_sequence()
    elif mode_selection == '3':
        main_model_category_selection_sequence()
    else:
        print("Invalid selection. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
