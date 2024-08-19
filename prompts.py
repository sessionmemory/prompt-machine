#!/usr/bin/env python3
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

# prompts.py
import json
import os
from utils import multi_selection_input
from config import *
from user_messages import *

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

def select_prompts(prompts):
    print("\n" + msg_user_nudge() + msg_word_select() + " " + msg_word_prompt() + "(s):")
    selected_prompts = multi_selection_input("\n" + msg_user_nudge() + msg_word_enter() + " your choices: ", prompts)
    if not selected_prompts:
        print("No " + msg_word_prompt() + "s selected, exiting.")
        return None
    return selected_prompts

def handle_custom_prompt(prompts, prompts_file):
    while True:
        prompt = input(msg_word_enter() + " your custom " + msg_word_prompt() + ": ").strip()
        if prompt:
            # Proceed with adding the prompt to a category
            break
        else:
            print(msg_word_prompt() + " cannot be empty, please try again.")

    print("\n" + msg_word_select() + " a " + msg_word_category() + " to add your " + msg_word_prompt() + " to:")
    categories = list(prompts.keys())
    for idx, category in enumerate(categories):
        print(f"{idx + 1}. {CATEGORY_COLOR}{category}{RESET_STYLE}")
    print("Or " + msg_word_select() + " a new " + msg_word_category() + " name.")

    category_input = input(msg_word_enter() + " the " + msg_word_number() + " or name of the " + msg_word_category() + ": ").strip()

    try:
        category_idx = int(category_input) - 1
        if 0 <= category_idx < len(categories):
            selected_category = categories[category_idx]
        else:
            print(msg_word_invalid() + " " + msg_word_category() + " " + msg_word_number() + ", creating a new " + msg_word_category() + ".")
            selected_category = category_input
    except ValueError:
        selected_category = category_input

    if selected_category in prompts:
        prompts[selected_category].append(prompt)
    else:
        prompts[selected_category] = [prompt]

    save_prompts(prompts_file, prompts)
    return prompt

def save_prompts(filename, prompts):
    with open(filename, 'w') as f:
        json.dump({"categories": prompts}, f, indent=4)

def load_model_responses(model_name):
    # Replace slashes in the model name with hyphens to match the JSON filename
    filename_safe_model_name = model_name.replace('/', '-')
    filename = f"{responses_dir}/{filename_safe_model_name}.json"
    if not os.path.exists(filename):
        print(f"No response file found for " + msg_word_model() + " {model_name}.")
        return []
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def load_model_prompts(model_name):
    filename = f"{responses_dir}/{model_name}.json"  # Adjusted path
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as file:
        data = json.load(file)
    return [item['prompt'] for item in data]

def find_missing_prompts(model_name):
    all_prompts = load_prompts(prompts_file, flat=True)  # Assuming this loads all prompts as a flat list
    used_prompts = load_model_prompts(model_name)
    missing_prompts = [prompt for prompt in all_prompts if prompt not in used_prompts]
    return missing_prompts