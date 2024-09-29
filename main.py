#!/usr/bin/env python3
# main.py
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

from user_messages import *
from modes import *

def main():
    last_action = None
    global current_mode  # Track the pre-prompt mode here
    while True:
        print(welcome_message)
        print(f"🌟 Current Pre-Prompt Mode: {current_mode}")  # Show current mode

        print(msg_initial_mode())
        current_mode = "normal"  # Default mode unless changed
        print(menu_option_single_prompt())
        print(menu_option_model_prompt_selection())
        print(menu_option_model_category_selection())
        print(menu_option_all_prompts_single_model())
        print(menu_option_unsent_prompts())
        print(menu_option_summary_prompts_excel())
        print(menu_option_query_completed_responses())
        print(menu_option_random_model_prompt())
        print(menu_option_export_excel())
        print(menu_option_response_evaluation())
        print(menu_option_preprompt_mode())
        print(menu_option_quit() + "\n")

        choice = input(enter_your_choice()).strip().lower()

        if choice == 'q':
            print(msg_farewell())
            break
            
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
        elif choice == '8':
            main_8_random_model_prompt()
        elif choice == '9':
            main_9_export_to_excel()
        elif choice == '10':
            main_10_response_evaluation()
        elif choice == '11':
            main_11_preprompt_mode()   
        else:
            print(msg_invalid_retry())

        task_complete_msg()
        next_action = input(msg_your_choice()).strip().lower()
        if next_action == 'q':
            print(msg_farewell())
            break
        elif next_action == 'm':
            continue  # This will restart the loop, showing the main menu
        else:
            print(msg_invalid_returning())
            continue

if __name__ == "__main__":
    main()