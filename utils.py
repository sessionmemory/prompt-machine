#!/usr/bin/env python3
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

# utils.py
import os
import pandas as pd
import logging
from generation import generate
from config import *
from user_messages import *
import uuid
import json
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from pathlib import Path
from datetime import datetime
import shutil
import time

def is_non_standard_text(text):
    """
    Simple check to see if the text is made up mostly of non-alphanumeric characters
    (e.g., ASCII art, emojis, symbols).
    """
    return all(not c.isalnum() for c in text)

def strip_formatting(text):
    # Remove all style formatting (like bold and color codes)
    return text.replace(BOLD_EFFECT, "").replace(MENU_OPTION_COLOR, "").replace(RESET_STYLE, "")

def format_preprompt_mode_with_emoji(mode):
    emoji = preprompt_mode_emojis.get(mode, "")
    return f"{emoji} {mode}"

def get_mode_with_emoji(mode):
    emoji = preprompt_mode_emojis.get(mode, "")
    return f"{emoji} {mode}"

def multi_selection_input(prompt, items):
    while True:
        print(prompt)
        for idx, item in enumerate(items, start=1):
            print(f"{idx}. {PROMPT_COLOR}{item}{RESET_STYLE}")
        selection_input = input(msg_enter_prompt_selection_multiple()).strip()

        if not selection_input:  # Handle empty input
            print(msg_enter_prompt_selection_multiple())
            continue

        selected_indices = []
        for part in selection_input.split(','):
            try:
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected_indices.extend(range(start, end + 1))
                else:
                    selected_indices.append(int(part))
            except ValueError:
                print(msg_invalid_number())
                break  # Break out of the for loop, continue while loop for new input
        else:  # This else corresponds to the for loop
            # Deduplicate and sort the indices
            selected_indices = sorted(set(selected_indices))

            # Validate selection
            try:
                selected_items = [items[idx - 1] for idx in selected_indices]
                print(msg_prompt_confirm_multi())
                for item in selected_items:
                    print(msg_list_selected_prompts(item))
                if confirm_selection():
                    return selected_items
            except IndexError:
                print(msg_invalid_retry())

def confirm_selection(message=msg_confirm_selection()):
    while True:
        confirm = input(message).strip().lower()
        # Treat empty input as 'yes'
        if confirm in ['y', 'yes', '']:
            return True
        elif confirm in ['n', 'no']:
            return False
        else:
            print(msg_select_y_n())

def select_category(categories):
    print(msg_select_category())
    for idx, category in enumerate(categories):
        print(f"{idx + 1}. {CATEGORY_COLOR}{category}{RESET_STYLE}")
    print(msg_custom_prompt_instr())
    print(msg_q_to_quit())

    while True:
        category_input = input(msg_enter_category_num()).strip()
        if category_input.lower() == 'q':
            return None
        elif category_input == '':
            print(msg_enter_category_num())
            continue

        try:
            category_idx = int(category_input) - 1
            if category_idx == -1:
                selected_category = 'custom'
            elif 0 <= category_idx < len(categories):
                selected_category = categories[category_idx]
                if not confirm_selection(msg_confirm_custom_cat(selected_category)):
                    print(msg_invalid_retry())
                    continue  # Stay in the loop for a new selection
            else:
                raise ValueError  # Treat out-of-range numbers as invalid
        except ValueError:
            print(msg_invalid_number())
            continue  # Stay in the loop for a new selection

        return selected_category

def print_response_stats(response, response_time, char_count, word_count):
    if response_time > 60:
        minutes = int(response_time // 60)
        seconds = int(response_time % 60)
        formatted_response_time = f"{minutes} minutes and {seconds} seconds"
    else:
        formatted_response_time = f"{response_time:.2f} seconds"

    print(f"\n{BOLD_EFFECT}{STATS_COLOR}Response Time:{RESET_STYLE} {BOLD_EFFECT}{formatted_response_time}{RESET_STYLE}")
    print(f"{BOLD_EFFECT}{STATS_COLOR}Character Count:{RESET_STYLE} {BOLD_EFFECT}{char_count}{RESET_STYLE}")
    print(f"{BOLD_EFFECT}{STATS_COLOR}Word Count:{RESET_STYLE} {BOLD_EFFECT}{word_count}{RESET_STYLE}")
    character_rate = char_count / response_time if response_time > 0 else 0
    word_rate = word_count / response_time if response_time > 0 else 0
    print(f"{BOLD_EFFECT}{STATS_COLOR}Character Rate:{RESET_STYLE} {BOLD_EFFECT}{character_rate:.2f}{RESET_STYLE} characters per second")
    print(f"{BOLD_EFFECT}{STATS_COLOR}Word Rate:{RESET_STYLE} {BOLD_EFFECT}{word_rate:.2f}{RESET_STYLE} words per second")

def get_user_rating():
    while True:
        rating = input(msg_get_response_rating()).strip()
        try:
            rating = int(rating)
            if 1 <= rating <= 5:
                return rating
            else:
                print(msg_invalid_rating_num())
        except ValueError:
            print(msg_invalid_number())

def list_response_files():
    response_dir = 'responses'  # Adjust path as necessary
    files = [f for f in os.listdir(response_dir) if f.endswith('.json')]
    files.sort()  # Alphabetize the list
    return files

def select_response_files():
    files = list_response_files()
    print(f"{msg_word_select()} the {msg_word_model()} response file(s) to export:")
    selected_files = multi_selection_input("Enter the numbers (e.g., 1,2,3): ", files)
    return selected_files

def process_json_files(files):
    # Generate the timestamp once, to use for all entries
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prompts_df = pd.read_excel(prompts_file, engine='openpyxl')
    data = []

    for file in files:
        with open(f'responses/{file}', 'r') as f:
            responses = json.load(f)
            for response in responses:
                prompt_text = response['prompt']
                # Find the matching Prompt_ID, if available
                prompt_uuid = prompts_df[prompts_df['prompt_text'] == prompt_text]['prompt_id'].values[0] if not prompts_df[prompts_df['prompt_text'] == prompt_text].empty else ''
                data.append({
                    'response_message_id': str(uuid.uuid4()),
                    # Include the timestamp in the Conv_ID
                    'response_conv_id': f"test-{file.replace('.json', '')}-{timestamp}",
                    'responses_prompt_id': prompt_uuid,
                    'prompt_response_cat_emoji': '',
                    'response_prompt_category': '',
                    'responses_prompt_text': prompt_text,
                    'response_llm_name': file.replace('.json', ''),
                    'response_msg_content': response['response'],
                    'Msg_Month': '(10) October',
                    'Msg_Year': '2024',
                    'Msg_AuthorRole': 'assistant',
                    'Response_Dur': response['response_time'],
                    'Msg_Status': 'exported',
                })
    return pd.DataFrame(data)

def export_to_excel(df, json_filename):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Correctly format the Excel filename using the json_filename parameter
    base_filename = json_filename[:-5]  # Remove the '.json' extension
    excel_filename = f"{base_filename}-{timestamp}.xlsx"
    excel_path = os.path.join('responses', 'excel', excel_filename)  # Adjust path as necessary

    # Ensure the 'excel' directory exists
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    # Export DataFrame to Excel
    df.to_excel(excel_path, index=False)

    return timestamp, excel_path  # Return both the timestamp and the excel_path

def file_selection_input(prompt, options):
    """
    Display a prompt and list of file options, and allow the user to select one option.
    Returns the index of the selected file (0-indexed).
    """
    print(prompt)
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")

    while True:
        try:
            selection = int(input("Enter your choice: "))
            if 1 <= selection <= len(options):
                return selection - 1  # Return 0-indexed position
            else:
                print(f"Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def single_selection_input(prompt, options):
    """
    Display a prompt and list of options, and allow the user to select one option.
    """
    print(prompt)
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")

    while True:
        try:
            selection = int(input("Enter your choice: "))
            if 1 <= selection <= len(options):
                return options[selection - 1]
            else:
                print(f"Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def merge_evaluations():
    # Load all three evaluation files
    compute_df = pd.read_excel('prompt_responses_compute.xlsx')
    gemini_df = pd.read_excel('prompt_responses_gemini.xlsx')
    cohere_df = pd.read_excel('prompt_responses_cohere.xlsx')

    # Rename columns for Gemini and Cohere evaluations (to differentiate during merging)
    gemini_df = gemini_df.rename(columns=lambda col: f"gemini-1.5-flash_{col}" if col not in ['Prompt_Text'] else col)
    cohere_df = cohere_df.rename(columns=lambda col: f"cohere_command_r_{col}" if col not in ['Prompt_Text'] else col)

    # Merge compute, Gemini, and Cohere dataframes
    merged_df = compute_df.merge(gemini_df, how='left', on='Prompt_Text')
    merged_df = merged_df.merge(cohere_df, how='left', on='Prompt_Text')

    # Iterate through columns and handle the merging logic
    for col in compute_df.columns:
        if col != "Prompt_Text":  # Skip the common key column
            gemini_col = f"gemini-1.5-flash_{col}"
            cohere_col = f"cohere_command_r_{col}"
            
            # Check if the columns exist in both dataframes
            if gemini_col in merged_df.columns and cohere_col in merged_df.columns:
                if pd.api.types.is_numeric_dtype(merged_df[gemini_col]) and pd.api.types.is_numeric_dtype(merged_df[cohere_col]):
                    # For numeric columns, take the average if both have data
                    merged_df[col] = merged_df[[gemini_col, cohere_col]].mean(axis=1)
                else:
                    # For non-numeric columns, take the first non-null value
                    merged_df[col] = merged_df[gemini_col].combine_first(merged_df[cohere_col])
            
            # Drop the individual Gemini and Cohere columns after merging
            merged_df = merged_df.drop([gemini_col, cohere_col], axis=1)

    # Save the final merged file
    output_file = 'prompt_responses_eval_complete.xlsx'
    merged_df.to_excel(output_file, index=False)
    print(f"✅ All evaluations merged and saved to {output_file}")

    # Move the original files to the /evaluated folder with timestamps
    move_files_with_timestamp()

def move_files_with_timestamp():
    # Define the folder where files will be moved
    eval_folder = 'evaluated/'
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)
    
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Define file names and their new locations
    files_to_move = {
        'prompt_responses_compute.xlsx': f'{eval_folder}prompt_responses_compute_{timestamp}.xlsx',
        'prompt_responses_gemini.xlsx': f'{eval_folder}prompt_responses_gemini_{timestamp}.xlsx',
        'prompt_responses_cohere.xlsx': f'{eval_folder}prompt_responses_cohere_{timestamp}.xlsx'
    }

    # Move the files
    for old_file, new_file in files_to_move.items():
        if os.path.exists(old_file):
            os.rename(old_file, new_file)
            print(f"✅ Moved {old_file} to {new_file}")
        else:
            print(f"⚠️ {old_file} does not exist, skipping.")

def export_all_responses():
    selected_files = select_response_files()
    if not selected_files:
        print("No files selected.")
        return
    
    for json_file in selected_files:
        df = process_json_files([json_file])  # Process one file at a time
        timestamp, excel_path = export_to_excel(df, json_file)  # Capture both returned values
        
        # Move and rename the JSON file
        original_path = os.path.join('responses', json_file)
        new_filename = f"{json_file[:-5]}-{timestamp}.json"  # Remove .json extension and add timestamp
        new_path = os.path.join('responses', 'exported', new_filename)
        
        # Ensure the exported directory exists
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        
        # Move and rename the file
        shutil.move(original_path, new_path)
        
        print(f"Exported {BOLD_EFFECT}{json_file}{RESET_STYLE} to {BOLD_EFFECT}{excel_path}{RESET_STYLE}, and moved it to {BOLD_EFFECT}{new_path}{RESET_STYLE}.")