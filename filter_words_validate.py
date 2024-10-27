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
            validated_by_getty = set()  # Keep track of words validated by Getty
            
            for word in unrecognized_words:
                sanitized_word = word.replace("/", " ")
                try:
                    print(f"⏳ Validating '{sanitized_word}' with Getty Vocabularies...")
                    if validate_with_getty_vocabularies(sanitized_word):
                        print(f"✅ '{sanitized_word}' found in Getty Vocabularies. Keeping in custom dictionary.")
                        validated_by_getty.add(word)
                    else:
                        print(f"🚫 '{sanitized_word}' not found in Getty Vocabularies. Adding to list for Gemini.")
                        remaining_words.add(word)  # Only words not validated by Getty go to Gemini
                except Exception as e:
                    print(f"❌ Getty API error for '{sanitized_word}': {e}")
                    print("Skipping Getty validation for this term.")
                    remaining_words.add(word)
        else:
            remaining_words = unrecognized_words

        # Step 4: Gemini validation for terms not validated by Getty
        proceed_gemini = input("\nProceed with Gemini validation for terms not found by Getty? (Y/N): ").strip().lower()
        if proceed_gemini == 'y':
            validated_by_gemini = set()
            print(f"🧠 Starting Gemini validation for {len(remaining_words)} remaining terms...")
            for term in remaining_words:
                print(f"⏳ Checking '{term}' with Gemini...")
                is_valid = check_word_with_gemini(term)
                if is_valid:
                    print(f"✅ '{term}' is valid. Adding to validated list.")
                    validated_by_gemini.add(term)
                else:
                    print(f"🚫 '{term}' flagged as invalid by Gemini. Removing it.")
            # Merge validated sets from both Getty and Gemini
            validated_terms = validated_by_getty.union(validated_by_gemini)
        else:
            validated_terms = validated_by_getty  # Only Getty-validated terms if Gemini is skipped

        # Step 5: Write the final validated terms back to the dictionary file
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

    TASK:
    Determine if the term '{term}' is spelled correctly and is a valid word in any context. If the word exists as a proper noun, technical term, or in any specialized language, confirm its validity.
    
    RESPONSE CRITERIA:
    Respond "Yes" for a word that is valid in any context, such as:
    - Proper nouns (people’s names, place names, brand names)
    - Technical terms (scientific, medical, or technical jargon)
    - Software or coding-related terms (e.g., Python, SQL, Kubernetes)
    - Foreign words (e.g., Spanish, German, Romanized Japanese, Pinyin, etc.)
    - Common abbreviations or acronyms across any field
    - Widely recognized terms in daily use or contemporary culture that may not appear in standard dictionaries

    EXCLUDE:
    Respond "No" for:
    - Slangs or casual words (e.g., "flocka") or casual abbreviations (e.g., "lol").
    - "Squashed words" i.e. multiple valid words without spaces in between (e.g., "nobelprize", "workfromhome", "blacklivesmatter")
    
    ANSWER FORMAT:
    Respond with only 'Yes' if the term is valid and correctly spelled in any language or domain.
    Respond with only 'No' if the term is invalid or does not exist in any language or domain.
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

        # Process the response using the 'parts' attribute of content
        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate.content, 'parts'):
                # Join parts and strip any leading/trailing whitespace
                processed_response = ''.join(part.text for part in candidate.content.parts if part.text).strip().lower()
                return processed_response == "yes"
        return False

    except Exception as e:
        print(f"❌ Error validating term '{term}' with Gemini API: {e}")
        return False

    return False

# Run the filter validation
validate_filter_terms_with_hunspell_and_ai('filter_words.dic')