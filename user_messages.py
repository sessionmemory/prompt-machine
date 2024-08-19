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


def msg_enter_selection():
    return f"\n{msg_user_nudge()}{STATS_COLOR}Please {msg_word_enter()} your selection[new]: {RESET_STYLE}"


def msg_select_summary_prompt():
    return f"\n{msg_user_nudge()}{STATS_COLOR}{msg_word_select()} a summarization[new] {RESET_STYLE}{msg_word_prompt()}:"

def msg_use_same_model(selected_model):
    return f"\n{msg_user_nudge()}Do you want to continue[new] with{RESET_STYLE} {BOLD_EFFECT}{MODELNAME_COLOR}{selected_model}{RESET_STYLE} {CONFIRM_COLOR}or select a different model? " + yes_or_no() + ": {RESET_STYLE}"

#

def msg_generating_msg(model_name, prompt):
    return f"\n{emoji_generating} Generating response for {msg_word_model()} {BOLD_EFFECT}{MODELNAME_COLOR}{model_name}{RESET_STYLE} with {msg_word_prompt()}: {PROMPT_COLOR}{prompt}{RESET_STYLE}"

def msg_content(first_choice_content):
    return f"{RESPONSE_COLOR}{first_choice_content}{RESET_STYLE}"

def msg_continue_model(selected_model):
    return f"\n{msg_user_nudge()}{CONFIRM_COLOR}Do you want to continue with{RESET_STYLE} {BOLD_EFFECT}{MODELNAME_COLOR}{selected_model}{RESET_STYLE} {CONFIRM_COLOR}or select a different {msg_word_model()}? {yes_or_no()}:" 

def msg_no_resp_processing():
    return f"\n{emoji_alert}Response processing was not implemented for this {msg_word_model()}."

# Model and Processing Related Messages
msg_generating_msg = "\n{emoji_generating} Generating response for {msg_word_model} {BOLD_EFFECT}{MODELNAME_COLOR}{model_name}{RESET_STYLE} with {msg_word_prompt}: {PROMPT_COLOR}{prompt}{RESET_STYLE}"
msg_content = "{RESPONSE_COLOR}{first_choice_content}{RESET_STYLE}"
msg_continue_model = "\n{msg_user_nudge}{CONFIRM_COLOR}Do you want to continue with{RESET_STYLE} {BOLD_EFFECT}{MODELNAME_COLOR}{selected_model}{RESET_STYLE} {CONFIRM_COLOR}or select a different {msg_word_model}? {yes_or_no}:"
msg_no_resp_processing = "\n{emoji_alert}Response processing was not implemented for this {msg_word_model}."

# Error and Invalid Inputs
msg_error_response = "{emoji_error}{msg_word_error} generating response for {msg_word_prompt} '{prompt}': {e}"
msg_invalid_response = "{emoji_error}No valid response generated."
msg_invalid_returning = "{emoji_error}{msg_word_invalid} option {emoji_error} Returning to the Main Menu!  {emoji_menu_main}"
msg_invalid_number = "{emoji_alert}{msg_word_invalid} input {emoji_error} Please enter a {msg_word_number} {emoji_number}."
msg_invalid_retry = "{emoji_alert}{msg_word_invalid} selection {emoji_error} Please enter a valid selection or type {STATS_COLOR}'q'{RESET_STYLE}. {msg_user_nudge}"
msg_no_matching = "{emoji_alert}No matching responses found. 🙁"

# No Selections or Errors Related to Selections
msg_no_prompts = "{emoji_alert}No {msg_word_prompt}s selected."
msg_no_category = "{emoji_alert}No {msg_word_category} selected."
msg_no_models = "{emoji_alert}No {msg_word_model}s selected."
msg_no_missing_prompts = "{emoji_alert}No missing {msg_word_prompt}s for this {msg_word_model}."
msg_no_summary_prompts = "{emoji_alert}No {msg_word_prompt}s found in {CATEGORY_COLOR}'Comprehension and Summarization'{RESET_STYLE}. Please check your {BOLD_EFFECT}{prompts_file}{RESET_STYLE} file."

# Farewell and Miscellaneous
msg_prompt_quantity = "{msg_user_nudge} {emoji_number} {msg_word_enter} the {msg_word_number} of times to send each {msg_word_prompt} (1-10): "
msg_number_1_10 = "{msg_user_nudge} {emoji_number} Please enter a {msg_word_number} between 1 and 10."
msg_farewell = "Bye now! {emoji_bye} {RESET_STYLE}"

