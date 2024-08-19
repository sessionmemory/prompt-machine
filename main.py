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
from config import *
from models import load_models, select_model, ask_to_save_response, save_response
from prompts import load_prompts, handle_custom_prompt, find_missing_prompts, load_model_responses
from generation import generate
from utils import *
from user_messages import *

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

    while True:
        if not selected_model:
            selected_model_names = select_model(models, allow_multiple=False)
            if selected_model_names is None:
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
    if not selected_models:
        print(msg_no_models())
        return

    # New Step: Select a prompt category first
    categories = list(prompts.keys())
#    print(msg_select_category())
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
            quantity_input = input(msg_prompt_quantity())
            # Convert the user's input into an integer
            quantity = int(quantity_input.strip())
            break  # Exit the loop if the input is successfully converted
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
    if not selected_models:
        print(msg_no_models())
        return

    categories = list(prompts.keys())
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
    print("\nOption 4: All Prompts to Single Model")
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
    print("\nOption 5: View Unsent Prompts to Model")
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

    # Use msg_found_unsent to display the number of unsent prompts found
    print(msg_found_unsent(len(missing_prompts), model_name))
    
    # Simplify the multi-selection input message
    print(msg_multi_selection_input("send", missing_prompts, "or hit Enter to send all."))

    selected_prompts = multi_selection_input("\n" + msg_user_nudge() + msg_word_enter() + " your choices: ", missing_prompts)
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
    print("\nOption 6: Summarize Content from Excel File Using Selected Prompt")

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
    print("\nOption 7: Query Existing Responses")
    selected_models = select_model(models, allow_multiple=True)
    if not selected_models:
        print(msg_no_models())
        return

    prompts = load_prompts(prompts_file)  # Loads prompts categorized
    categories = list(prompts.keys())
    selected_category = select_category(categories)
    if not selected_category:
        print(msg_farewell())
        return

    category_prompts = prompts[selected_category]
    # Use msg_select_prompt_from_cat to display the selected category
    print(msg_select_prompt_from_cat(selected_category))
    
    # Simplify the multi-selection input message
    print(msg_multi_selection_input("query", category_prompts, "or hit Enter to query all."))

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

def task_complete_msg():
    """Displays the message for next steps after a task completes."""
    print(msg_msg_whats_next())
    print(menu_option_return_main())
    print(menu_option_go_back())
    print(menu_option_quit_application())

def main():
    last_action = None

    while True:
        print(welcome_message)
        print(msg_initial_mode())

        print(menu_option_single_prompt())
        print(menu_option_model_prompt_selection())
        print(menu_option_model_category_selection())
        print(menu_option_all_prompts_single_model())
        print(menu_option_unsent_prompts())
        print(menu_option_summary_prompts_excel())
        print(menu_option_query_completed_responses())
        print(menu_option_quit())

        if last_action:
            print(msg_go_back())

        choice = input(enter_your_choice()).strip().lower()

        if choice == 'q':
            print(msg_farewell())
            break
        elif choice == 'b' and last_action:
            choice = last_action
        else:
            last_action = choice

        if choice == '1':
            main_1_userselect()
        elif choice == '2':
            main_2_model_prompt_selection_sequence()
        elif choice == '3':
            main_3_model_category_selection_sequence()
        elif choice == '4':
            main_4_all_prompts_to_single_model()
        elif choice == '5':
            main_5_review_missing_prompts()
        elif choice == '6':
            main_6_iterate_summary()
        elif choice == '7':
            main_7_query_responses()
        else:
            print(msg_invalid_retry())

        task_complete_msg()
        next_action = input(msg_your_choice()).strip().lower()
        if next_action == 'q':
            print(msg_farewell())
            break
        elif next_action == 'm':
            continue  # This will restart the loop, showing the main menu
        elif next_action == 'b' and last_action:
            # Set choice to last_action to repeat it in the next iteration
            choice = last_action
        else:
            print(msg_invalid_returning())
            continue

if __name__ == "__main__":
    main()