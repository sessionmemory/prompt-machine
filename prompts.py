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
import pandas as pd
import uuid


def load_prompts(filename, flat=False):
    # Check if the Excel file exists
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return {} if not flat else []
    
    # Read the Excel file
    df = pd.read_excel(filename, engine='openpyxl')
    
    if flat:
        # Return a flat list of all prompts
        return df['Prompt_Text'].tolist()
    else:
        # Return a dictionary of categories with their prompts
        categories = {}
        for _, row in df.iterrows():
            category = row['Prompt_Category']
            prompt = row['Prompt_Text']
            if category in categories:
                categories[category].append(prompt)
            else:
                categories[category] = [prompt]
        return categories

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
    # Load existing prompts if the file exists
    if os.path.exists(filename):
        df_existing = pd.read_excel(filename, engine='openpyxl')
    else:
        df_existing = pd.DataFrame(columns=['Prompt_ID', 'Prompt_Category', 'Prompt_Text'])
    
    # Convert the updated prompts dictionary back to a DataFrame
    data = []
    for category, prompts_list in prompts.items():
        for prompt in prompts_list:
            data.append({
                'Prompt_ID': str(uuid.uuid4()),  # Generate a new UUID for each prompt
                'Prompt_Category': category,
                'Prompt_Text': prompt
            })
    df_new = pd.DataFrame(data)
    
    # Append new prompts to existing DataFrame
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Save the updated DataFrame back to the Excel file
    df_final.to_excel(filename, index=False, engine='openpyxl')

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