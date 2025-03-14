#!/usr/bin/env python3
# main.py
# for testing AI models responses to a diverse set of prompts

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

from user_messages import *
from utils import *
from modes import get_current_mode, main_1_model_prompt_selection_sequence, main_2_model_category_selection_sequence, main_3_all_prompts_to_single_model, main_4_review_missing_prompts, main_5_iterate_summary, main_6_query_responses, main_7_random_model_prompt, main_8_export_to_excel, main_9_response_evaluation, main_10_preprompt_mode, main_11_chat_mode

def rerun_last_action(last_action):
    """Reruns the last selected action based on the stored 'last_action' variable."""

    if last_action == '1':
        main_1_model_prompt_selection_sequence()
    elif last_action == '2':
        main_2_model_category_selection_sequence()
    elif last_action == '3':
        main_3_all_prompts_to_single_model()
    elif last_action == '4':
        main_4_review_missing_prompts()
    elif last_action == '5':
        main_5_iterate_summary()
    elif last_action == '6':
        main_6_query_responses()
    elif last_action == '7':
        main_7_random_model_prompt()
    elif last_action == '8':
        main_8_export_to_excel()
    elif last_action == '9':
        main_9_response_evaluation()
    elif last_action == '10':
        main_10_preprompt_mode()
    elif last_action == '11':
        main_11_chat_mode()
    else:
        print("No valid last action to rerun.")

def main():
    last_action = None

    while True:
        print(welcome_message)
        
        # Get current mode with emoji for display
        current_mode_with_emoji = get_mode_with_emoji(get_current_mode())
        print(f"🎭 Current Persona: {current_mode_with_emoji}")  # Show current mode with emoji

        print(msg_initial_mode())
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
        print(menu_option_chat_mode())
        print(menu_option_quit() + "\n")

        choice = input(enter_your_choice()).strip().lower()

        if choice == 'q':
            print(msg_farewell())
            break

        if choice == '1':
            last_action = '1'
            main_1_model_prompt_selection_sequence()
        elif choice == '2':
            last_action = '2'
            main_2_model_category_selection_sequence()
        elif choice == '3':
            last_action = '3'
            main_3_all_prompts_to_single_model()
        elif choice == '4':
            last_action = '4'
            main_4_review_missing_prompts()
        elif choice == '5':
            last_action = '5'
            main_5_iterate_summary()
        elif choice == '6':
            last_action = '6'
            main_6_query_responses()
        elif choice == '7':
            last_action = '7'
            main_7_random_model_prompt()
        elif choice == '8':
            last_action = '8'
            main_8_export_to_excel()
        elif choice == '9':
            last_action = '9'
            main_9_response_evaluation()
        elif choice == '10':
            last_action = '10'
            main_10_preprompt_mode()
        elif choice == '11':
            last_action = '11'
            main_11_chat_mode()
        else:
            print(msg_invalid_retry())

        # Display next action options
        task_complete_msg(last_action)  # Pass the last action to show rerun option

        next_action = input(msg_your_choice()).strip().lower()
        if next_action == 'q':
            print(msg_farewell())
            break
        elif next_action == 'm':
            continue  # Restart the loop to show the main menu
        elif next_action == 'r' and last_action:  # Option to rerun the last task
            rerun_last_action(last_action)  # Call the rerun function
        
if __name__ == "__main__":
    main()