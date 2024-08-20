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
    instruction, input_prompt = msg_select_prompt_multiple()
    print(instruction)
    selected_prompts = multi_selection_input(input_prompt, prompts)
    if not selected_prompts:
        print(msg_no_prompts())
        return None
    return selected_prompts

def handle_custom_prompt(prompts, prompts_file):
    while True:
        prompt = input(msg_enter_custom_prompt()).strip()
        if prompt:
            # Proceed with adding the prompt to a category
            break
        else:
            print(msg_invalid_empty())

    print(msg_add_custom_prompt_cat())
    categories = list(prompts.keys())
    for idx, category in enumerate(categories):
        print(f"{idx + 1}. {CATEGORY_COLOR}{category}{RESET_STYLE}")
    print(msg_add_category_name())

    category_input = input(msg_enter_cat_name_num()).strip()

    try:
        category_idx = int(category_input) - 1
        if 0 <= category_idx < len(categories):
            selected_category = categories[category_idx]
        else:
            print(msg_invalid_category())
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
        print(msg_no_response_file())
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
    all_prompts = load_prompts(prompts_file, flat=True)
    used_prompts = load_model_prompts(model_name)
    missing_prompts = [prompt for prompt in all_prompts if prompt not in used_prompts]
    return missing_prompts