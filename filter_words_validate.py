import google.generativeai as genai
from config import *
from hunspell import HunSpell
from process_text import load_hunspell_dictionaries_validation, validate_with_getty_vocabularies

def validate_filter_terms_with_hunspell_and_ai(file_path):
    """
    Validates filter terms by first checking them with Hunspell, then optionally with Getty Vocabularies,
    and finally allowing review before sending remaining unrecognized terms to Gemini.
    """
    try:
        # Step 1: Load filter terms from file, skipping the first line (word count)
        print(f"🔄 Loading terms from {file_path} for Hunspell and Getty validation...")
        with open(file_path, 'r') as file:
            lines = file.readlines()

        count_line = lines[0].strip()
        words = {line.strip().lower() for line in lines[1:] if line.strip()}  # lowercase for uniformity

        # Initialize Hunspell without loading the custom dictionary itself
        print("🔍 Initializing Hunspell with multiple dictionaries...")
        hunspell = load_hunspell_dictionaries_validation()

        # Step 2: Check words with Hunspell and display results
        unrecognized_words = set()
        print("\nChecking each word in custom dictionary with Hunspell:")
        for word in words:
            if hunspell.spell(word):
                print(f"✅ '{word}' found valid by Hunspell. Removing from custom dictionary.")
            else:
                print(f"🚫 '{word}' not recognized by Hunspell. Keeping for further validation.")
                unrecognized_words.add(word)

        # Update dictionary immediately with terms validated by Hunspell
        print("💾 Updating custom dictionary after Hunspell validation...")
        with open(file_path, 'w') as file:
            file.write(f"{len(unrecognized_words)}\n")
            for term in unrecognized_words:
                file.write(f"{term}\n")

        print(f"\n✅ {len(unrecognized_words)} unrecognized terms after Hunspell check.\n")

        # Step 3: Optionally proceed to Getty validation
        proceed_getty = input("Proceed with Getty validation for remaining terms? (Y/N): ").strip().lower()
        if proceed_getty == 'y':
            remaining_words = set()
            for word in unrecognized_words:
                sanitized_word = word.replace("/", " ")
                try:
                    print(f"⏳ Validating '{sanitized_word}' with Getty Vocabularies...")
                    if validate_with_getty_vocabularies(sanitized_word):
                        print(f"✅ '{sanitized_word}' found in Getty Vocabularies. Adding to validated list.")
                    else:
                        print(f"🚫 '{sanitized_word}' not found in Getty Vocabularies. Adding to list for Gemini.")
                        remaining_words.add(word)
                except Exception as e:
                    print(f"❌ Getty API error for '{sanitized_word}': {e}")
                    print("Skipping Getty validation for this term.")
                    remaining_words.add(word)
        else:
            remaining_words = unrecognized_words

        # Step 4: Review list before sending to Gemini
        print("\n📋 Remaining terms after Hunspell and optional Getty validation:")
        for term in remaining_words:
            print(term)

        proceed_gemini = input("\nProceed with Gemini validation for these terms? (Y/N): ").strip().lower()
        if proceed_gemini != 'y':
            print("❌ Gemini validation canceled. Exiting...")
            return

        # Step 5: Gemini validation
        validated_terms = set()
        print(f"🧠 Starting Gemini validation for {len(remaining_words)} remaining terms...")
        for term in remaining_words:
            print(f"⏳ Checking '{term}' with Gemini...")
            is_valid = check_word_with_gemini(term)
            if is_valid:
                print(f"✅ '{term}' is valid. Adding to validated list.")
                validated_terms.add(term)
            else:
                print(f"🚫 '{term}' flagged as invalid by Gemini. Removing it.")

        # Step 6: Write the final validated terms back to the dictionary file
        final_terms = sorted(validated_terms)
        updated_count = len(final_terms)

        print("💾 Writing final validated terms back to the dictionary file...")
        with open(file_path, 'w') as file:
            if updated_count > 0:
                file.write(f"{updated_count}\n")
                for term in final_terms:
                    file.write(f"{term}\n")
            else:
                file.writelines(lines)  # If no new terms validated, retain the original file content

        print(f"✅ {file_path} has been validated and updated with {updated_count} valid terms.")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def check_word_with_gemini(term):
    """
    Sends a term to Gemini AI to confirm if it's a valid word in any context.
    """
    # Create a prompt for validation
    prompt = f"""
    ROLE:
    You are a spelling expert with comprehensive knowledge of accurate word spelling across languages, domains, and contexts.
    ... (the rest of the prompt remains the same)
    """
    try:
        genai.configure(api_key=google_api_key)
        google_model_instance = genai.GenerativeModel(google_model)
        response = google_model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=5,
                temperature=0
            ),
        )

        if response.candidates:
            candidate_text = response.candidates[0].content.strip().lower()
            return candidate_text == "yes"

    except Exception as e:
        print(f"❌ Error validating term '{term}' with Gemini API: {e}")
        return False

    return False

# Run the filter validation
validate_filter_terms_with_hunspell_and_ai('filter_words.dic')