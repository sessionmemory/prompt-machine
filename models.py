#!/usr/bin/env python3
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

# models.py
import json
import logging
import os
from config import *
from utils import confirm_selection
from user_messages import *

def load_models(filename=models_file):
    if not os.path.exists(filename):
        logging.error(f"Models file {filename} not found.")
        return []
    with open(filename, 'r') as f:
        data = json.load(f)
        return data.get('models', [])

def select_model(models, allow_multiple=False):
    while True:
        if allow_multiple:
            print("\n" + msg_user_nudge() + msg_word_select() + f" the model(s) to use (e.g., 1-4, 5, 7), or type {STATS_COLOR}'q'{RESET_STYLE} to quit:")
            print("You can select multiple " + msg_word_model() + "s by separating " + msg_word_number() + f"s with commas or specifying ranges (e.g., {BOLD_EFFECT}1-3,5{RESET_STYLE}).")
        else:
            print("\n" + msg_user_nudge() + msg_word_select() + f" the model to use (e.g., 4), or type {STATS_COLOR}'q'{RESET_STYLE} to quit:")
        
        for idx, model in enumerate(models, start=1):
            print(f"{idx}. {BOLD_EFFECT}{MODELNAME_COLOR}{model['name']}{RESET_STYLE} - {BOLD_EFFECT}{STATS_COLOR}{model['size']}{RESET_STYLE}")
        model_selection = input("\n" + msg_user_nudge() + msg_word_enter() + " your " + msg_word_model() + " selection: ").strip()
        if model_selection.lower() == 'q':
            return None  # User chose to exit

        if not allow_multiple and ("," in model_selection or "-" in model_selection):
            print(msg_word_invalid() + " input, please " + msg_word_select() + " a single " + msg_word_number() + " without commas or dashes.")
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
            print(f"You have selected: {BOLD_EFFECT}{MODELNAME_COLOR}{', '.join(selected_models)}{RESET_STYLE}")
            if confirm_selection():
                return selected_models
        except ValueError:
            print(msg_word_invalid() + " input, please " + msg_word_select() + f" a valid selection or type {STATS_COLOR}'q'{RESET_STYLE}.")

def ask_to_save_response():
    return confirm_selection("\n" + msg_user_nudge() + f"Do you want to {BOLD_EFFECT}Save{RESET_STYLE} this response? " + yes_or_no() + ":")

def save_response(model_name, prompt, response, rating, response_time, char_count, word_count):
    # Replace slashes in the model name with hyphens
    filename_safe_model_name = model_name.replace('/', '-')
    filename = f"{responses_dir}/{filename_safe_model_name}.json"  # Adjusted path
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