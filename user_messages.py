#!/usr/bin/env python3
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

# user_messages.py
import os
from config import *

# Basic Commands and Words
def msg_word_enter():
    return f"{BOLD_EFFECT}{STATS_COLOR}Enter{RESET_STYLE}"
def msg_word_select():
    return f"{BOLD_EFFECT}{STATS_COLOR}Select{RESET_STYLE}"
def msg_word_error():
    return f"{BOLD_EFFECT}{ERROR_COLOR}Error{RESET_STYLE}"
def msg_word_invalid():
    return f"{BOLD_EFFECT}{ERROR_COLOR}Invalid{RESET_STYLE}"
def msg_word_model():
    return f"{BOLD_EFFECT}{MODEL_COLOR}model{RESET_STYLE}"
def msg_word_category():
    return f"{BOLD_EFFECT}{CATEGORY_COLOR}category{RESET_STYLE}"
def msg_word_prompt():
    return f"{BOLD_EFFECT}{PROMPT_COLOR}prompt{RESET_STYLE}"
def msg_word_number():
    return f"{BOLD_EFFECT}number{RESET_STYLE}"
def yes_or_no():
    return f"{BOLD_EFFECT}{STATS_COLOR}(y/n){RESET_STYLE}"

# User Interaction and Prompts
def msg_user_nudge():
    return f"{RESPONSE_COLOR}{BOLD_EFFECT}{emoji_user_nudge}{RESET_STYLE}"
def msg_model_confirm(selected_models):
    return f"\n{msg_user_nudge()}{STATS_COLOR}You have selected[new]: {BOLD_EFFECT}{MODELNAME_COLOR}{', '.join(selected_models)}{RESET_STYLE}"
def msg_prompt_confirm(prompt):
    return f"\n{msg_user_nudge()}{STATS_COLOR}You have selected[new]:{RESET_STYLE}\n- {PROMPT_SELECT_COLOR}{prompt}{RESET_STYLE}"
def msg_confirm_selection():
    return f"\n{msg_user_nudge()}{STATS_COLOR}Please confirm your selection[new]:{RESET_STYLE}" + yes_or_no() + ": "
def msg_select_model_multiple():
    return f"\n{msg_user_nudge()}{msg_word_select()} [multi]the model(s) to use (e.g., 1-4, 5, 7), or type {STATS_COLOR}'q'{RESET_STYLE} to quit:\nYou can select multiple {msg_word_model()}s by separating {msg_word_number()}s with commas or specifying ranges (e.g., {BOLD_EFFECT}1-3,5{RESET_STYLE})."
def msg_select_model_single():
    return f"\n{msg_user_nudge()}{msg_word_select()} [single]the model to use (e.g., 4), or type {STATS_COLOR}'q'{RESET_STYLE} to quit:"
def msg_enter_model_selection():
    return f"\n{msg_user_nudge()}{msg_word_enter()} [new]your {msg_word_model()} selection: "
def msg_select_prompt_multiple():
    instruction = f"\n{msg_user_nudge()}{msg_word_select()} {msg_word_prompt()}(s):"
    input_prompt = f"\n{msg_user_nudge()}{msg_word_enter()} your choices: "
    return instruction, input_prompt
def msg_enter_prompt_selection_multiple():
    return f"\n{msg_user_nudge()}{msg_word_enter()} your choices (e.g., 1-3,5,7-8,10): "
def msg_enter_custom_prompt():
    return msg_word_enter() + "[new] your custom " + msg_word_prompt() + ":"
def msg_select_category():
    return f"\n" + msg_user_nudge() + msg_word_select() + " a " + msg_word_prompt() + " " + msg_word_category() + ":"
def msg_select_category_prompts(category_prompts):
    return f"\n" + msg_user_nudge() + msg_word_enter() + " your choices: ", category_prompts
def msg_content(first_choice_content):
    return f"{RESPONSE_COLOR}{first_choice_content}[new]{RESET_STYLE}"
def msg_select_summary_prompt():
    return f"\n{msg_user_nudge()}{STATS_COLOR}{msg_word_select()} a summarization[new] {RESET_STYLE}{msg_word_prompt()}:"
def msg_summary_prompt_missing():
    return f"No " + msg_word_prompt() + f"s found for 'Comprehension and Summarization'. Please check your {BOLD_EFFECT}prompts.json{RESET_STYLE} file."
def msg_no_resp_processing():
    return f"\n{emoji_alert}Response processing was not implemented for this {msg_word_model()}."
def msg_use_same_model(selected_model):
    return f"\n{msg_user_nudge()}Do you want to continue[new] with{RESET_STYLE} {BOLD_EFFECT}{MODELNAME_COLOR}{selected_model}{RESET_STYLE} {CONFIRM_COLOR}or select a different model? " + yes_or_no() + f": {RESET_STYLE}"
def msg_enter_prompt_selection_single():
    return f"\n" + msg_user_nudge() + msg_word_enter() + " the " + msg_word_number() + " of the " + msg_word_prompt() + " you want to use: "
def msg_farewell():
    return f"Bye now!" + f"{emoji_bye} {RESET_STYLE}"
def msg_error_response(prompt, e):
    return f"{emoji_error}{msg_word_error} generating response for {msg_word_prompt} '{prompt}': {e}"
def msg_error_simple(e):
    return msg_word_error() + f" generating response: {str(e)}"
def msg_invalid_response():
    return f"{emoji_error}No valid response generated."
def msg_generating_msg(model_name, prompt):
    return f"\n{emoji_generating} Generating response for " + msg_word_model() + f": {BOLD_EFFECT}{MODELNAME_COLOR}{model_name}{RESET_STYLE} with " + msg_word_prompt() + f": {PROMPT_COLOR}{prompt}{RESET_STYLE}"
def msg_generating_selected(selected_model, prompt):
    return f"\n{emoji_generating} Generating response for " + msg_word_model() + f": {BOLD_EFFECT}{MODELNAME_COLOR}{selected_model}{RESET_STYLE} with " + msg_word_prompt() + f": {PROMPT_COLOR}{prompt}{RESET_STYLE}"
def msg_invalid_retry():
    return f"{emoji_error} Invalid entry. Please enter a valid selection or type {STATS_COLOR}'q'{RESET_STYLE}. {msg_user_nudge}"


# ? def msg_select_prompt_single():

def msg_no_prompts():
    return f"{emoji_alert}No {msg_word_prompt()}s selected."

def msg_no_category():
    return f"{emoji_alert}No {msg_word_category()} selected."

def msg_no_models():
    return f"{emoji_alert}No {msg_word_model()}s selected."

def msg_no_missing_prompts():
    return f"{emoji_alert}No missing {msg_word_prompt()}s for this {msg_word_model()}."

def msg_prompt_quantity():
    return f"{msg_user_nudge()} {emoji_number} {msg_word_enter()} the {msg_word_number()} of times to send each {msg_word_prompt()} (1-10): "

def msg_number_1_10():
    return f"{msg_user_nudge()} {emoji_number} Please enter a {msg_word_number()} between 1 and 10."

def msg_invalid_returning():
    return f"{emoji_error}{msg_word_invalid()} option {emoji_error} Returning to the Main Menu!  {emoji_menu_main}"

def msg_invalid_number():
    return f"{emoji_alert}{msg_word_invalid()} input {emoji_error} Please enter a {msg_word_number()} {emoji_number}."

def msg_no_matching():
    return f"{emoji_alert}No matching responses found. 🙁"





