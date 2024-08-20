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
        logging.error(msg_models_file_error(filename))
        return []
    with open(filename, 'r') as f:
        data = json.load(f)
        return data.get('models', [])

def select_model(models, allow_multiple=False):
    while True:
        filter_input = input(msg_enter_model_filter()).strip().lower()
        filtered_models = []

        # Filter models based on the input
        if filter_input:
            for model in models:
                if filter_input in model['name'].lower() or filter_input in model['size'].lower():
                    filtered_models.append(model)
            if not filtered_models:
                print(msg_filter_not_found())
                continue
        else:
            filtered_models = models

        if allow_multiple:
            print(msg_select_model_multiple())
        else:
            print(msg_select_model_single())

        for idx, model in enumerate(filtered_models, start=1):
            print(f"{idx}. {BOLD_EFFECT}{MODELNAME_COLOR}{model['name']}{RESET_STYLE} - {BOLD_EFFECT}{STATS_COLOR}{model['size']}{RESET_STYLE}")

        model_selection = input(msg_enter_model_selection()).strip()

        if model_selection.lower() == 'q':
            return None  # User chose to exit

        if not allow_multiple and ("," in model_selection or "-" in model_selection):
            print(msg_invalid_number2())
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
                if 0 <= selected_index < len(filtered_models):
                    selected_indices.append(selected_index)
                else:
                    raise ValueError(msg_valueerror)

            # Validate and deduplicate selected indices
            selected_indices = list(set(selected_indices))  # Remove duplicates
            selected_models = [filtered_models[i]['name'] for i in selected_indices]
            print(msg_model_confirm(selected_models))
            if confirm_selection():
                return selected_models
        except ValueError:
            print(msg_invalid_retry())

def ask_to_save_response():
    return confirm_selection(msg_save_response())

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