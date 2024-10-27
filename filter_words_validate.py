import google.generativeai as genai
from config import *
from hunspell import HunSpell
from process_text import load_hunspell_dictionaries, validate_with_getty_vocabularies

def validate_filter_terms_with_hunspell_and_ai(file_path):
    """
    Validates filter terms by first checking them with Hunspell, then Getty Vocabularies, 
    and finally sending unrecognized terms to Gemini for additional validation.
    """
    try:
        # Step 1: Load filter terms from file, skipping the first line (word count)
        print(f"🔄 Loading terms from {file_path} for Hunspell and Getty validation...")
        with open(file_path, 'r') as file:
            lines = file.readlines()

        count_line = lines[0].strip()
        words = {line.strip().lower() for line in lines[1:] if line.strip()}  # lowercase for uniformity

        # Initialize Hunspell using the provided function from process_text
        print("🔍 Initializing Hunspell with multiple dictionaries...")
        hunspell = load_hunspell_dictionaries()  # Load Hunspell with extended dictionaries

        # Step 2: Filter words with Hunspell
        unrecognized_words = {word for word in words if not hunspell.spell(word)}

        print(f"✅ {len(unrecognized_words)} unrecognized terms after Hunspell check. Proceeding with Getty validation...\n")

        # Step 3: Validate with Getty Vocabularies
        remaining_words = set()
        for word in unrecognized_words:
            print(f"⏳ Validating '{word}' with Getty Vocabularies...")
            if validate_with_getty_vocabularies(word):
                print(f"✅ '{word}' found in Getty Vocabularies. Keeping it in the list.")
                remaining_words.add(word)
            else:
                print(f"🚫 '{word}' not found in Getty Vocabularies. Sending to Gemini.")

        # Step 4: Prepare remaining words for Gemini validation
        print(f"🧠 Starting Gemini validation for {len(remaining_words)} remaining terms...")

        # Step 5: Validate with Gemini
        validated_terms = set()
        for term in remaining_words:
            print(f"⏳ Checking '{term}' with Gemini...")
            is_valid = check_word_with_gemini(term)
            if is_valid:
                print(f"✅ '{term}' is valid. Keeping it in the list.")
                validated_terms.add(term)
            else:
                print(f"🚫 '{term}' flagged as invalid by Gemini. Removing it.")

        # Step 6: Update the count and alphabetize validated terms
        final_terms = sorted(validated_terms)
        updated_count = len(final_terms)

        # Step 7: Write validated list back to the dictionary file
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
                max_output_tokens=5,
                temperature=0
            ),
        )

        # Process the response
        if response.candidates:
            candidate_text = response.candidates[0].content.strip().lower()
            return candidate_text == "yes"

    except Exception as e:
        print(f"❌ Error validating term '{term}' with Gemini API: {e}")
        return False

    return False

# Run the filter validation
validate_filter_terms_with_hunspell_and_ai('filter_words.dic')