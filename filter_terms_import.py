import json
from hunspell import HunSpell

# Load the Hunspell dictionary with main and custom dictionaries
print("Initializing Hunspell with main and custom dictionaries.")
hunspell = HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
custom_dict_path = 'filter_words.dic'
hunspell.add_dic(custom_dict_path)  # Add current custom dictionary
print(f"Custom dictionary loaded from: {custom_dict_path}")

# Load the filter terms from JSON
print("Loading filter terms from filter_terms.json.")
with open("filter_terms.json", "r") as json_file:
    data = json.load(json_file)
    filter_terms = data["filter_terms"]
print(f"Filter terms loaded: {len(filter_terms)} words")

# Track the words to add to the custom dictionary
new_words = []
print("Checking each word in the filter terms list...")

# Check each word in the filter terms list
for word in filter_terms:
    print(f"Checking word: '{word}'")
    if not hunspell.spell(word):  # If the word is not recognized
        print(f"'{word}' is not recognized. Adding to new words list.")
        new_words.append(word)  # Add it to the list of words to add
    else:
        print(f"'{word}' is already recognized by Hunspell.")

# Update the custom dictionary (.dic file) with new words
if new_words:
    print("Reading the current custom dictionary to update word count and add new words.")
    with open(custom_dict_path, "r") as dic_file:
        lines = dic_file.readlines()
    
    # Update the word count on the first line
    current_count = int(lines[0].strip())
    updated_count = current_count + len(new_words)
    lines[0] = f"{updated_count}\n"  # Update the word count

    # Append new words to the dictionary
    lines.extend(f"{word}\n" for word in new_words)

    print(f"Adding {len(new_words)} new words to {custom_dict_path}.")
    with open(custom_dict_path, "w") as dic_file:
        dic_file.writelines(lines)  # Write updated count and existing words
else:
    print("No new words to add. The custom dictionary remains unchanged.")

print(f"Completed. Added {len(new_words)} new words to {custom_dict_path}.")