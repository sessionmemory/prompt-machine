import google.generativeai as genai
from config import *

def validate_filter_terms_with_ai(file_path):
    """
    Reads filter terms from a .dic file, sorts, deduplicates, and allows for review before
    sending each term (skipping the first line) to Gemini AI to confirm if they are valid.
    Removes any terms Gemini flags as errors, keeping only valid terms.
    """
    try:
        # Step 1: Load filter terms from file, skipping the first line (word count)
        print(f"🔄 Loading terms from {file_path} and preparing for cleanup...")
        with open(file_path, 'r') as file:
            lines = file.readlines()

        count_line = lines[0].strip()
        words = {line.strip().lower() for line in lines[1:] if line.strip()}  # lowercase for uniformity

        # Step 2: Sort and deduplicate terms
        print("🔍 Sorting and deduplicating terms...")
        sorted_terms = sorted(words)

        # Step 3: Display the cleaned terms for review before Gemini validation
        print("\n📋 Cleaned and sorted terms (before Gemini validation):")
        for term in sorted_terms:
            print(term)
        
        # Step 4: Prompt for proceeding with Gemini validation
        proceed = input("\nProceed with Gemini validation for these terms? (Y/N): ").strip().lower()
        if proceed != 'y':
            print("❌ Gemini validation canceled. Exiting...")
            return
        
        # Step 5: Initialize an empty set for valid terms and validate each term with Gemini
        print("🧠 Starting Gemini validation...")
        validated_terms = set()
        
        for term in sorted_terms:
            print(f"⏳ Checking '{term}' with Gemini...")
            is_valid = check_word_with_gemini(term)
            if is_valid:
                print(f"✅ '{term}' is valid. Keeping it in the list.")
                validated_terms.add(term)
            else:
                print(f"🚫 '{term}' flagged as invalid by Gemini. Removing it.")

        # Step 6: Update the count and alphabetize the validated terms
        final_terms = sorted(validated_terms)
        updated_count = len(final_terms)

        # Step 7: Write the cleaned, validated list back to the file
        print("💾 Writing validated terms back to the dictionary file...")
        with open(file_path, 'w') as file:
            file.write(f"{updated_count}\n")
            for term in final_terms:
                file.write(f"{term}\n")

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
    The following word was flagged as misspelled by a spell-checker using en_US and en_GB OpenOffice dictionaries, then converted to lowercase.

    QUESTION:
    Is the word or abbreviation "{term}" valid in any known domain or context?

    RESPONSE OPTION 1 of 2:
    Respond "Yes" if the word is correctly spelled or valid in any context. Examples:
    - Proper nouns (names of people, places, brands)
    - Technical terms (scientific, medical, technological jargon)
    - Software or coding terms (e.g., Python, SQL, Kubernetes)
    - Foreign words (e.g., Spanish, German, Romanized Japanese, etc.)
    - Common abbreviations or acronyms across any field (e.g., "llm" for Large Language Model)
    - Common terms from contemporary culture, even if not typically found in standard dictionaries.

    RESPONSE OPTION 2 of 2:
    Respond "No" if the word is likely erroneous, improper, or misspelled. Examples:
    - Slang (e.g., "flocka")
    - Casual abbreviations (e.g., "lol")
    - "Squashed words" i.e. multiple valid words without spaces in between (e.g., "nobelprize", "workfromhome", "blacklivesmatter")

    Respond strictly with "Yes" or "No" only.

    YOUR RESPONSE:
    """

    try:
        # Configure the Google API with your API key
        genai.configure(api_key=google_api_key)
        google_model_instance = genai.GenerativeModel(google_model)

        # Generate the response using the Gemini model
        response = google_model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=5,  # Only expecting 'Yes' or 'No'
                temperature=0
            ),
        )

        # Process the response, similar to your working code
        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                # Extract text parts and combine them
                candidate_text = ''.join(part.text for part in candidate.content.parts if part.text).strip().lower()
                return candidate_text == "yes"

    except Exception as e:
        print(f"❌ Error validating term '{term}' with Gemini API: {e}")
        return False

    return False

# Run the filter validation
validate_filter_terms_with_ai('filter_words.dic')