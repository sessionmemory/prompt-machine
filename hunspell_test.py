# hunspell_test.py

import json
import re
import hunspell
import spacy
from process_text import preprocess_text_for_spellcheck, load_hunspell_dictionaries, update_custom_dictionary
from process_eval import filter_spelling_errors_with_ai

# Load the Spacy NLP model
nlp = spacy.load('en_core_web_sm')  # Adjust model as needed

def main():
    print("\nGreetings! Welcome to the Hunspell Spellchecker.")
    hunspell_obj = load_hunspell_dictionaries()

    while True:
        print("\nPlease enter some text to check for spelling errors (or type 'exit' to quit):")
        user_input = input()

        if user_input.lower() == 'exit':
            print("\nFarewell!")
            break

        # Preprocess and check spelling
        preprocessed_text = preprocess_text_for_spellcheck(user_input)
        misspelled = [word for word in preprocessed_text.split() if not hunspell_obj.spell(word)]
        
        if misspelled:
            print("\nInitial Misspelled words:", ', '.join(misspelled))

            # Send to AI for further filtering
            filtered_words = filter_spelling_errors_with_ai(misspelled)
            print("\nWords validated by AI (not errors):", ', '.join(filtered_words) if filtered_words else "None")

            # Filter out the non-errors
            final_misspelled = [word for word in misspelled if word not in filtered_words]
            print("\nFinal list of misspelled words:", ', '.join(final_misspelled) if final_misspelled else "None")

            # Update the custom dictionary with new valid words
            if filtered_words:
                update_custom_dictionary(hunspell_obj, filtered_words)
                print("\nUpdated Hunspell custom dictionary with new valid terms.")
        else:
            print("\nNo spelling errors found.")

        print("\nWould you like to check spelling again? Y/N")
        if input().strip().upper() != 'Y':
            print("\nFarewell!")
            break

if __name__ == "__main__":
    main()