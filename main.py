#!/usr/bin/env python3
# main.py
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

import os
import time
import logging
from config import prompts_file, responses_dir, sleep_time, summary_input_xls
from models import load_models, select_model, ask_to_save_response, save_response
from prompts import load_prompts, handle_custom_prompt, find_missing_prompts, load_model_responses
from generation import generate
from utils import multi_selection_input, confirm_selection, select_category, print_response_stats, get_user_rating, process_excel_file

# Check and add responses folder for saving model output
responses_dir = responses_dir
if not os.path.exists(responses_dir):
    os.makedirs(responses_dir)

prompts_file = prompts_file

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use the load_models function to load the models
models = load_models()

def main_1_userselect():
    context = []
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
                print(f"{idx}. \033[35m{prompt_option}\033[0m")
            prompt_selection = input("→ Enter the number of the prompt you want to use: ").strip()
            try:
                prompt_idx = int(prompt_selection) - 1
                prompt = category_prompts[prompt_idx]
                print(f"You have selected:\n- \033[35m{prompt}\033[0m")
                if not confirm_selection():
                    continue
            except (ValueError, IndexError):
                print("Invalid selection, please try again.")
                continue
            
        logging.info(f"Generating response for model \033[1m\033[34m{selected_model}\033[0m with prompt: \033[35m{prompt}\033[0m")
        print(f"\nResponse from model \033[1m{selected_model}\033[0m:")
        try:
            # Ensure selected_model is a string, not a list
            context, response, response_time, char_count, word_count = generate(selected_model, prompt, context)
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            print(f"Error generating response: {e}")
            continue

        print_response_stats(response, response_time, char_count, word_count)
        
        if ask_to_save_response():
            rating = get_user_rating()
            save_response(selected_model, prompt, response, rating, response_time, char_count, word_count)

        # Ask if user wants to continue with the same model
        use_same_model = confirm_selection(f"\n\033[97m→ Do you want to continue with\033[0m \033[1m\033[34m{selected_model}\033[0m \033[97mor select a different model? (y/n): \033[0m")
        if use_same_model:
            # If 'y', continue with the same model but prompt will be re-selected in the next iteration
            continue
        else:
            # If 'n', reset selected_model to allow model selection
            selected_model = None

def main_2_model_prompt_selection_sequence():
    prompts = load_prompts(prompts_file)  # Loads prompts categorized
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
    print(f"\nSelect prompts from the category '\033[1;38;5;208m{selected_category}\033[0m':")
    selected_prompts = multi_selection_input("→ Enter your choices: ", category_prompts)
    if not selected_prompts:
        print("No prompts selected, exiting.")
        return

    # Ask for the quantity of times to send each prompt
    while True:
        try:
            quantity = int(input("Enter the number of times to send each prompt (1-10): ").strip())
            if 1 <= quantity <= 10:
                break
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    for model_name in selected_models:
        for prompt in selected_prompts:
            for _ in range(quantity):
                print(f"\nGenerating response for model \033[1m\033[34m{model_name}\033[0m with prompt: \033[35m{prompt}\033[0m")
                try:
                    context, response, response_time, char_count, word_count = generate(model_name, prompt, None)
                    print_response_stats(response, response_time, char_count, word_count)
                    # Directly save the response without user confirmation
                    save_response(model_name, prompt, response, "", response_time, char_count, word_count)
                except Exception as e:
                    logging.error(f"Error generating response: {e}")
                    print(f"Error generating response: {e}")
                time.sleep(sleep_time)  # Adjust sleep time as needed

def main_3_model_category_selection_sequence():
    prompts = load_prompts(prompts_file)  # Loads prompts categorized
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
                print(f"\nGenerating response for model \033[1m\033[34m{model_name}\033[0m with prompt: \033[35m{prompt}\033[0m")
                try:
                    context, response, response_time, char_count, word_count = generate(model_name, prompt, None)
                    print_response_stats(response, response_time, char_count, word_count)
                    # Directly save the response without user confirmation
                    save_response(model_name, prompt, response, "", response_time, char_count, word_count)
                except Exception as e:
                    logging.error(f"Error generating response: {e}")
                    print(f"Error generating response: {e}")
                time.sleep(sleep_time)  # Adjust sleep time as needed

def main_4_all_prompts_to_single_model():
    print("\nOption 4: All Prompts to Single Model")
    selected_model_names = select_model(models, allow_multiple=False)
    if selected_model_names is None:
        print("Exiting.")
        return
    selected_model = selected_model_names[0]  # Assuming only one model is selected for this option, take the first item

    model_name = selected_model  # Capture the selected model's name

    prompts = load_prompts(prompts_file, flat=True)  # Load all prompts, assuming a flat structure
    if not prompts:
        print("No prompts found.")
        return

    for prompt in prompts:
        print(f"\nSending prompt to model \033[1m\033[34m{model_name}\033[0m: \033[35m{prompt}\033[0m")
        try:
            context, response, response_time, char_count, word_count = generate(model_name, prompt, None)
            print_response_stats(response, response_time, char_count, word_count)
            # Directly save the response without user confirmation
            save_response(model_name, prompt, response, "", response_time, char_count, word_count)
        except Exception as e:
            logging.error(f"Error generating response for prompt '{prompt}': {e}")
            print(f"Error generating response for prompt '{prompt}': {e}")
        time.sleep(sleep_time)  # Throttle requests to avoid overwhelming the model

def main_5_review_missing_prompts():
    print("\nOption 5: View Unsent Prompts to Model")
    selected_model_names = select_model(models, allow_multiple=False)
    if selected_model_names is None:
        print("Exiting.")
        return
    selected_model = selected_model_names[0]

    model_name = selected_model  # Capture the selected model's name

    missing_prompts = find_missing_prompts(model_name)
    if not missing_prompts:
        print("No missing prompts for this model.")
        return

    print(f"\nFound {len(missing_prompts)} missing prompts for model \033[1m\033[34m{model_name}\033[0m.")
    selected_prompts = multi_selection_input("Select prompts to send (or hit enter to send all): ", missing_prompts)
    if not selected_prompts:
        selected_prompts = missing_prompts  # If user hits enter without selecting, use all missing prompts

    for prompt in selected_prompts:
        print(f"\nSending prompt to model \033[1m\033[34m{model_name}\033[0m: \033[35m{prompt}\033[0m")
        try:
            context, response, response_time, char_count, word_count = generate(model_name, prompt, None)
            print_response_stats(response, response_time, char_count, word_count)
            # Directly save the response without user confirmation
            save_response(model_name, prompt, response, "", response_time, char_count, word_count)
        except Exception as e:
            logging.error(f"Error generating response for prompt '{prompt}': {e}")
            print(f"Error generating response for prompt '{prompt}': {e}")
        time.sleep(sleep_time)  # Throttle requests to avoid overwhelming the model

def main_6_iterate_summary():
    print("\nOption 6: Summarize Content from Excel File Using Selected Prompt")

    # Select a single model
    selected_model_names = select_model(models, allow_multiple=False)
    if selected_model_names is None:
        print("Exiting.")
        return
    selected_model = selected_model_names[0]

    # Automatically select the "Comprehension and Summarization" category
    prompts = load_prompts(prompts_file)
    category_prompts = prompts.get("Comprehension and Summarization", [])
    if not category_prompts:
        print("No prompts found for 'Comprehension and Summarization'. Please check your prompts.json file.")
        return

    # Let the user select a prompt from the "Comprehension and Summarization" category
    print("\nSelect a summarization prompt:")
    for idx, prompt_option in enumerate(category_prompts, start=1):
        print(f"{idx}. \033[35m{prompt_option}\033[0m")
    prompt_selection = input("→ Enter the number of the prompt you want to use: ").strip()
    try:
        prompt_idx = int(prompt_selection) - 1
        prompt = category_prompts[prompt_idx]
    except (ValueError, IndexError):
        print("Invalid selection, please try again.")
        return  # Optionally, you could loop back to prompt selection instead of returning

    print(f"You have selected:\n- \033[35m{prompt}\033[0m")

    # Use the predefined Excel file path from config.py
    excel_path = summary_input_xls

    # Process the Excel file
    process_excel_file(selected_model, prompt, excel_path)

def main_7_query_responses():
    print("\nOption 7: Query Existing Responses")
    selected_models = select_model(models, allow_multiple=True)
    if not selected_models:
        print("No models selected, exiting.")
        return

    prompts = load_prompts(prompts_file)  # Loads prompts categorized
    categories = list(prompts.keys())
    selected_category = select_category(categories)
    if not selected_category:
        print("Exiting.")
        return

    category_prompts = prompts[selected_category]
    print(f"\nSelect prompts from the category '\033[1;38;5;208m{selected_category}\033[0m':")
    selected_prompts = multi_selection_input("→ Enter your choices: ", category_prompts)
    if not selected_prompts:
        print("No prompts selected, exiting.")
        return

    for model_name in selected_models:
        for prompt in selected_prompts:
            print(f"\nSearching for responses from model \033[1m\033[34m{model_name}\033[0m to prompt: \033[35m{prompt}\033[0m")
            responses = load_model_responses(model_name)
            matching_responses = [response for response in responses if response['prompt'] == prompt]
            if matching_responses:
                for response in matching_responses:
                    print(f"\nResponse from model \033[1m\033[34m{model_name}\033[0m to prompt '\033[35m{prompt}\033[0m':\n{response['response']}")
            else:
                print("No matching responses found.")

def main():
    print("Welcome to the Prompt Machine!\n\nSelect the mode you want to run:")
    # The menu option numbers will be magenta, the option name will be bold, and the parentheses will be regular
    print("\033[35m1.\033[0m \033[1mSingle Prompt, Model, and Rate\033[0m (Manual)")
    print("\033[35m2.\033[0m \033[1mModel & Prompt Selection\033[0m (Sequence)")
    print("\033[35m3.\033[0m \033[1mModel & Category Selection\033[0m (Sequence)")
    print("\033[35m4.\033[0m \033[1mAll Prompts to Single Model\033[0m (Sequence)")
    print("\033[35m5.\033[0m \033[1mUnsent Prompts for Model\033[0m (Sequence)")
    print("\033[35m6.\033[0m \033[1mIterate Summary Prompt on Excel\033[0m")
    print("\033[35m7.\033[0m \033[1mQuery Completed Responses\033[0m")

    mode_selection = input("Enter your choice (\033[35m1, 2, 3, 4, 5, 6,\033[0m or \033[35m7\033[0m): ").strip()
    if mode_selection == '1':
        main_1_userselect()
    elif mode_selection == '2':
        main_2_model_prompt_selection_sequence()
    elif mode_selection == '3':
        main_3_model_category_selection_sequence()
    elif mode_selection == '4':  # Handle Option 4
        main_4_all_prompts_to_single_model()
    elif mode_selection == '5':  # Handle Option 5
        main_5_review_missing_prompts()
    elif mode_selection == '6':  # Handle Option 6
        main_6_iterate_summary()
    elif mode_selection == '7':
        main_7_query_responses()
    else:
        print("Invalid selection. Please enter \033[35m1, 2, 3, 4, 5, 6,\033[0m or \033[35m7\033[0m.")  # Update this line

if __name__ == "__main__":
    main()
