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
import json

def multi_selection_input(prompt, items):
    while True:
        print(prompt)
        for idx, item in enumerate(items, start=1):
            print(f"{idx}. {item}")
        selection_input = input("Enter your choices (e.g., 1-3,5,7-8,10): ").strip()

        # Process the input to support ranges
        selected_indices = []
        for part in selection_input.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                selected_indices.extend(range(start, end + 1))
            else:
                selected_indices.append(int(part))

        # Deduplicate and sort the indices
        selected_indices = sorted(set(selected_indices))

        # Validate selection
        try:
            selected_items = [items[idx - 1] for idx in selected_indices]
            print("You have selected: ")
            for item in selected_items:
                print(f"- {item}")
            if confirm_selection():
                return selected_items
        except (ValueError, IndexError):
            print("Invalid selection, please try again.")

def confirm_selection(message="Confirm your selection? (y/n): "):
    while True:
        confirm = input(message).strip().lower()
        if confirm == 'y':
            return True
        elif confirm == 'n':
            return False
        else:
            print("Please enter 'y' or 'n'.")

def select_category(categories):
    print("\nSelect a category:")
    for idx, category in enumerate(categories):
        print(f"{idx + 1}. \033[1m{category}\033[0m")
    print("Enter '0' to enter a custom prompt.")
    print("Enter 'exit' to stop the program.")

    while True:
        category_input = input("\033[97m→ Enter the number of the category you want to use: \033[0m").strip()
        if category_input.lower() == 'exit':
            return None
        elif category_input == '':
            return [category['name'] for category in categories]  # Select all categories
        try:
            category_idx = int(category_input) - 1
            if category_idx == -1:
                selected_category = 'custom'
            elif 0 <= category_idx < len(categories):
                selected_category = categories[category_idx]
                # Confirmation step
                if not confirm_selection(f"Confirm your category selection '{selected_category}'? (y/n): "):
                    print("Selection not confirmed. Please try again.")
                    return select_category(categories)  # Re-select if not confirmed
            else:
                print("Invalid category number, please try again.")
                return select_category(categories)
        except ValueError:
            print("Invalid input, please enter a number.")
            return select_category(categories)
        return selected_category
    
def print_response_stats(response, response_time, char_count, word_count):
    # Similar to the existing code for displaying stats
    print(f"\n\033[1mResponse Time:\033[0m {response_time:.2f} seconds")
    print(f"\033[1mCharacter Count:\033[0m {char_count}")
    print(f"\033[1mWord Count:\033[0m {word_count}")
    character_rate = char_count / response_time if response_time > 0 else 0
    word_rate = word_count / response_time if response_time > 0 else 0
    print(f"\033[1mCharacter Rate:\033[0m {character_rate:.2f} characters per second")
    print(f"\033[1mWord Rate:\033[0m {word_rate:.2f} words per second\n")

def get_user_rating():
    while True:
        rating = input("\033[97m→ Rate the response on a scale of 1-5 (5 being the best): \033[0m").strip()
        try:
            rating = int(rating)
            if 1 <= rating <= 5:
                return rating
            else:
                print("Invalid rating, please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input, please enter a number.")