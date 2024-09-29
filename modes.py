#!/usr/bin/env python3
# modes.py
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
from config import *
from models import *
from prompts import *
from generation import *
from utils import *
from user_messages import *
from analyze_excel import *
import random
import json
import openai
from config import OPENAI_API_KEY  # Ensure you have this in your config

# Check and add responses folder for saving model output
responses_dir = responses_dir
if not os.path.exists(responses_dir):
    os.makedirs(responses_dir)

prompts_file = prompts_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use the load_models function to load the models
models = load_models()

def main_1_userselect():
    context = []
    prompts = load_prompts(prompts_file)
    selected_model = None
    prompt = None  # Initialize prompt to None to ensure it's selected in the first iteration
    print(menu1_title())
    while True:
        if not selected_model:
            selected_model_names = select_model(models, allow_multiple=False)
            if selected_model_names is None:
                break  # Exit the loop and end the program
            selected_model = selected_model_names[0]  # Assuming only one model is selected for this option, take the first item

        # Use select_category function for consistent category selection
        categories = list(prompts.keys())
        # Exclude the "Summarization" category from the list
        categories = [category for category in categories if category != summary_category_name]
        
        selected_category = select_category(categories)
        if selected_category is None:
            break  # Exit if the user chooses to exit during category selection

        if selected_category == 'custom':
            prompt = handle_custom_prompt(prompts, prompts_file)
            if prompt is None:
                continue  # Skip the rest of the loop if no custom prompt is provided
        else:
            # Display prompt options within the selected category
            print(msg_select_prompt_single())
            category_prompts = prompts[selected_category]
            for idx, prompt_option in enumerate(category_prompts, start=1):
                print(f"{idx}. {PROMPT_COLOR}{prompt_option}{RESET_STYLE}")
            prompt_selection = input(msg_enter_prompt_selection()).strip()
            try:
                prompt_idx = int(prompt_selection) - 1
                prompt = category_prompts[prompt_idx]
                print(msg_prompt_confirm(prompt))
                if not confirm_selection():
                    continue
            except (ValueError, IndexError):
                print(msg_invalid_retry())
                continue
            
        print(msg_generating_selected(selected_model, prompt))
        try:
            # Ensure selected_model is a string, not a list
            context, response, response_time, char_count, word_count = generate(selected_model, prompt, context)
        except Exception as e:
            logging.error(msg_word_error() + msg_error_simple(e))
            print(msg_word_error() + msg_error_simple(e))
            continue

        print_response_stats(response, response_time, char_count, word_count)
        
        if ask_to_save_response():
            rating = get_user_rating()
            save_response(selected_model, prompt, response, rating, response_time, char_count, word_count)

        # Ask if user wants to continue with the same model
        use_same_model = confirm_selection(msg_use_same_model(selected_model))
        if use_same_model:
            # If 'y', continue with the same model but prompt will be re-selected in the next iteration
            continue
        else:
            # If 'n', reset selected_model to allow model selection
            selected_model = None

def main_2_model_prompt_selection_sequence():
    prompts = load_prompts(prompts_file)  # Loads prompts categorized
    selected_models = select_model(models, allow_multiple=True)
    print(menu2_title())
    if not selected_models:
        print(msg_no_models())
        return

    # Use select_category function for consistent category selection
    categories = list(prompts.keys())
    # Exclude the "Summarization" category from the list
    categories = [category for category in categories if category != summary_category_name]

    selected_category = select_category(categories)
    if not selected_category:
        print(msg_farewell())
        return

    # Display prompts within the selected category
    category_prompts = prompts[selected_category]
    print(msg_prompts_from_category(selected_category))
    selected_prompts = multi_selection_input("\n", category_prompts)
    if not selected_prompts:
        print(msg_no_prompts())
        return

    # Ask for the quantity of times to send each prompt
    while True:
        try:
            # Display the prompt and capture the user's input
            quantity_input = input(msg_prompt_quantity()).strip()
            # Check if input is empty and default to 1
            if quantity_input == "":
                quantity = 1
            else:
                # Convert the user's input into an integer
                quantity = int(quantity_input)
            break  # Exit the loop if the input is successfully converted or empty
        except ValueError:
            # Handle the case where the conversion fails
            print(msg_invalid_number())

    for model_name in selected_models:
        for prompt in selected_prompts:
            for _ in range(quantity):
                print(msg_generating_msg(model_name, prompt))
                try:
                    context, response, response_time, char_count, word_count = generate(model_name, prompt, None)
                    print_response_stats(response, response_time, char_count, word_count)
                    # Directly save the response without user confirmation
                    save_response(model_name, prompt, response, "", response_time, char_count, word_count)
                except Exception as e:
                    logging.error(msg_error_simple(e))
                    print(msg_error_simple(e))
                time.sleep(sleep_time)  # Adjust sleep time as needed

def main_3_model_category_selection_sequence():
    prompts = load_prompts(prompts_file)  # Loads prompts categorized
    selected_models = select_model(models, allow_multiple=True)
    print(menu3_title())
    if not selected_models:
        print(msg_no_models())
        return

    # Use select_category function for consistent category selection
    categories = list(prompts.keys())
    # Exclude the "Summarization" category from the list
    categories = [category for category in categories if category != summary_category_name]

    selected_category = select_category(categories)
    if selected_category is None:
        print(msg_farewell())
        return
    elif selected_category == 'custom':
        # Handle custom prompt logic here
        print("Custom " + msg_word_prompt() + " logic not shown for brevity.")
        return
    else:
        # Use the selected category name directly to access prompts
        category_prompts = prompts[selected_category]
        for model_name in selected_models:
            for prompt in category_prompts:
                print(msg_generating_msg(model_name, prompt))
                try:
                    context, response, response_time, char_count, word_count = generate(model_name, prompt, None)
                    print_response_stats(response, response_time, char_count, word_count)
                    # Directly save the response without user confirmation
                    save_response(model_name, prompt, response, "", response_time, char_count, word_count)
                except Exception as e:
                    logging.error(msg_error_simple(e))
                    print(msg_error_simple(e))
                time.sleep(sleep_time)  # Adjust sleep time as needed

def main_4_all_prompts_to_single_model():
    print(menu4_title())
    selected_model_names = select_model(models, allow_multiple=False)
    if selected_model_names is None:
        print(msg_farewell())
        return
    selected_model = selected_model_names[0]  # Assuming only one model is selected for this option, take the first item

    model_name = selected_model  # Capture the selected model's name

    prompts = load_prompts(prompts_file, flat=True)  # Load all prompts, assuming a flat structure
    if not prompts:
        print(msg_no_prompts())
        return

    for prompt in prompts:
        print(msg_sending_prompt(model_name, prompt))  # Use the function here
        try:
            context, response, response_time, char_count, word_count = generate(model_name, prompt, None)
            print_response_stats(response, response_time, char_count, word_count)
            # Directly save the response without user confirmation
            save_response(model_name, prompt, response, "", response_time, char_count, word_count)
        except Exception as e:
            logging.error(msg_error_response(prompt, e))  # Ensure msg_error_response is defined to handle this
            print(msg_error_response(prompt, e))  # And here as well
        time.sleep(sleep_time)  # Throttle requests to avoid overwhelming the model

def main_5_review_missing_prompts():
    print(menu5_title())
    selected_model_names = select_model(models, allow_multiple=False)
    if selected_model_names is None:
        print(msg_farewell())
        return
    selected_model = selected_model_names[0]

    model_name = selected_model  # Capture the selected model's name

    missing_prompts = find_missing_prompts(model_name)
    if not missing_prompts:
        print(msg_no_missing_prompts())
        return

    selected_prompts = multi_selection_input(msg_unsent_prompts(), missing_prompts)
    if not selected_prompts:
        selected_prompts = missing_prompts  # If user hits enter without selecting, use all missing prompts

    for prompt in selected_prompts:
        print(msg_sending_to_model(model_name, prompt))
        try:
            context, response, response_time, char_count, word_count = generate(model_name, prompt, None)
            print_response_stats(response, response_time, char_count, word_count)
            # Directly save the response without user confirmation
            save_response(model_name, prompt, response, "", response_time, char_count, word_count)
        except Exception as e:
            logging.error(msg_error_response(prompt, str(e)))  # Ensure to convert exception to string
            print(msg_error_response(prompt, str(e)))
        time.sleep(sleep_time)  # Throttle requests to avoid overwhelming the model

def main_6_iterate_summary():
    print(menu6_title())

    # Select a single model
    selected_model_names = select_model(models, allow_multiple=False)
    if selected_model_names is None:
        print(msg_farewell())
        return
    selected_model = selected_model_names[0]

    # Automatically select the "Comprehension and Summarization" category
    prompts = load_prompts(prompts_file)
    category_prompts = prompts.get(summary_category_name, [])
    if not category_prompts:
        print(msg_summary_prompt_missing())
        return

    # Let the user select a prompt from the "Comprehension and Summarization" category
    print(msg_select_summary_prompt())
    for idx, prompt_option in enumerate(category_prompts, start=1):
        print(f"{idx}. {PROMPT_COLOR}{prompt_option}{RESET_STYLE}")
    prompt_selection = input(msg_enter_prompt_num()).strip()
    try:
        prompt_idx = int(prompt_selection) - 1
        prompt = category_prompts[prompt_idx]
    except (ValueError, IndexError):
        print(msg_invalid_retry())
        return  # Optionally, you could loop back to prompt selection instead of returning

    print(msg_prompt_confirm(prompt))

    # Use the predefined Excel file path from config.py
    excel_path = summary_input_xls

    # Process the Excel file
    process_excel_file(selected_model, prompt, excel_path)

def main_7_query_responses():
    print(menu7_title())
    selected_models = select_model(models, allow_multiple=True)
    if not selected_models:
        print(msg_no_models())
        return

    prompts = load_prompts(prompts_file)  # Loads prompts categorized

    # Use select_category function for consistent category selection
    categories = list(prompts.keys())
    # Exclude the "Summarization" category from the list
    categories = [category for category in categories if category != summary_category_name] 

    selected_category = select_category(categories)
    if not selected_category:
        print(msg_farewell())
        return

    category_prompts = prompts[selected_category]
    # Use msg_select_prompt_from_cat to display the selected category
    print(msg_select_prompt_from_cat(selected_category))

    selected_prompts = multi_selection_input("\n" + msg_user_nudge() + msg_word_enter() + " your choices: ", category_prompts)
    if not selected_prompts:
        print(msg_no_prompts())
        return

    for model_name in selected_models:
        for prompt in selected_prompts:
            print(msg_search_query(model_name, prompt))
            responses = load_model_responses(model_name)
            matching_responses = [response for response in responses if response['prompt'] == prompt]
            if matching_responses:
                for response in matching_responses:
                    print(msg_query_display_results(model_name, prompt, response))
            else:
                print(msg_no_matching())

def main_8_random_model_prompt():
    print(menu8_title())

    while True:  # This loop allows re-rolling
        # Load models
        models = load_models()  # Assuming this function returns a list of model dictionaries

        # Load prompts from Excel instead of JSON
        try:
            prompts_df = pd.read_excel('prompts_benchmarks.xlsx', engine='openpyxl')  # Adjust the path if necessary
            all_prompts = prompts_df['Prompt_Text'].tolist()  # Assuming your Excel has a column named 'Prompt'
        except FileNotFoundError:
            print("File 'prompts_benchmarks.xlsx' not found.")
            return
        except Exception as e:
            print(f"Failed to load prompts from Excel: {e}")
            return

        if not all_prompts:
            print("No prompts available.")
            return

        # Randomly select a model
        selected_model = random.choice(models)
        model_name = selected_model['name']  # Extract the model name

        # Randomly select a prompt
        prompt = random.choice(all_prompts)

        # Display the randomized selection to the user
        print(f"\nRandomly selected {msg_word_model()}: {MODELNAME_COLOR}{BOLD_EFFECT}{model_name}{RESET_STYLE}")
        print(f"Randomly selected {msg_word_prompt()}: {PROMPT_COLOR}{prompt}{RESET_STYLE}")

        # Ask user to proceed or re-roll
        proceed = confirm_selection(msg_confirm_selection())
        if proceed:
            # Proceed with generation using the selected model and prompt
            try:
                print(msg_generating_msg(model_name, prompt))
                context, response, response_time, char_count, word_count = generate(model_name, prompt, None)
                print_response_stats(response, response_time, char_count, word_count)
                # Optionally save the response without user confirmation
                save_response(model_name, prompt, response, "", response_time, char_count, word_count)
            except Exception as e:
                logging.error(msg_error_simple(e))
                print(msg_error_simple(e))

            # After generation, ask if the user wants to roll again
            roll_again = confirm_selection(msg_roll_dice_again())
            if not roll_again:
                break  # Correctly placed to exit the while True loop if the user does not want to roll again
        else:
            # If the user chooses not to proceed, ask immediately if they want to roll again
            roll_again = confirm_selection(msg_roll_dice_again())
            if not roll_again:
                break  # Exit the loop if the user does not want to roll again

def main_9_export_to_excel():
    export_all_responses()

def main_10_response_evaluation():
    print("🔄 Starting Response Evaluation 🔄")
    
    # Define the modes of analysis (you can add more modes here if needed)
    modes_of_analysis = [
        "Count Sentences",
        "Count Tokens",
        "Count Characters",
        "Count Words",
        "Extract Named Entities",
        "Detect URLs and Code",
        "Cosine Similarity Analysis",
        "Sentiment Polarity Analysis",
        "Sentiment Subjectivity Analysis",
        "Word Frequency Check",
        "Spelling Error Check",
        "BERTScore Analysis",
        "Token Matching Analysis",
        "Semantic Similarity Analysis",
        "Noun-Phrase Extraction",
        "Gemini 1.5 Flash - AI Evaluation (6 aspects)"
    ]

    # Display menu to select modes
    selected_modes = multi_selection_input("Select the analysis modes to run:", modes_of_analysis)
    if not selected_modes:
        print("No modes selected. Exiting.")
        return

    # Load responses from Excel
    file_path = 'prompt_responses.xlsx'
    output_file_path = 'prompt_responses.xlsx'
    
    # Run the analyses on the selected modes
    process_selected_analysis_modes(file_path, output_file_path, selected_modes)
    print(f"✅ All selected analyses completed and saved to {output_file_path}.\n")
