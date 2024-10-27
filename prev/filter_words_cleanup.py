def clean_text_file(file_path):
    """
    Reads a .txt file with one word per line, converts all words to lowercase,
    removes duplicates, and alphabetizes the list, then saves back to the file.
    """
    try:
        # Read file and process each line
        with open(file_path, 'r') as file:
            words = {line.strip().lower() for line in file if line.strip()}  # Set for uniqueness & lowercase

        # Sort words alphabetically
        sorted_words = sorted(words)

        # Write back to the file
        with open(file_path, 'w') as file:
            for word in sorted_words:
                file.write(f"{word}\n")

        print(f"✅ {file_path} has been cleaned: lowercase, deduplicated, and alphabetized.")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Specify the path to your custom dictionary file
clean_text_file('filter_words.dic')