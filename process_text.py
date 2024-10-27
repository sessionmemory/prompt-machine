#!/usr/bin/env python3

"""
process_text.py
"""

__author__ = "Alex Bishop"
__version__ = "0.1.0"
__license__ = "MIT"

from textblob import TextBlob
import spacy
from transformers import GPT2Tokenizer, BertModel, BertTokenizer, BartTokenizer, BartForConditionalGeneration, PegasusTokenizer, PegasusForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nlp = spacy.load("en_core_web_sm")
import numpy as np
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import hunspell
import torch
import re
import json
import pandas as pd
from openpyxl import load_workbook
import google.generativeai as genai
from generation import generate
import string
from config import FLAGGED_WORDS, FLAGGED_PHRASES
import requests
import os
from config import *
import time
import xml.etree.ElementTree as ET
# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def preprocess_text_for_spellcheck(text):
    """Perform preprocessing to clean and normalize text, skipping LaTeX notations and hashtags."""
    if not isinstance(text, str):
        return ""  # Return an empty string if the input is invalid

    # Normalize apostrophes to standard single quote
    text = text.replace("’`", "'")

    # Ignore content within double dollar signs ($$...$$) used in LaTeX notations
    text = re.sub(r'\$\$.+?\$\$', ' ', text, flags=re.DOTALL)

    # Remove hashtags and their content
    text = re.sub(r'#\S+', '', text)

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r'\.\.+', ' ', text)  # Replace multiple periods with space
    text = re.sub(r'[\*\@\&\$\(\)\[\]\{\}\<\>\:;,\!\?\"]+', ' ', text)  # Replace special chars with space

    # Replace commas with spaces to prevent word merging
    text = re.sub(r',', ' ', text)

    # Expand common abbreviations
    for abbr, full_form in abbreviations.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep letters and spaces only
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    return text

def load_hunspell_dictionaries_validation():
    """Load Hunspell dictionaries with extended language support and custom dictionary."""
    hunspell_obj = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
    
    # Load each additional dictionary separately to avoid the argument error
    additional_dictionaries = [
        '/usr/share/hunspell/en_GB.dic', '/usr/share/hunspell/en_GB.aff',
        '/usr/share/hunspell/en_AU.dic', '/usr/share/hunspell/en_AU.aff',
        '/usr/share/hunspell/en_CA.dic', '/usr/share/hunspell/en_CA.aff',
        '/usr/share/hunspell/en_ZA.dic', '/usr/share/hunspell/en_ZA.aff',
        '/usr/share/hunspell/en_USNames.dic', '/usr/share/hunspell/en_USNames.aff',
        '/usr/share/hunspell/fr_FR.dic', '/usr/share/hunspell/fr_FR.aff',
        '/usr/share/hunspell/de_DE.dic', '/usr/share/hunspell/de_DE.aff',
        '/usr/share/hunspell/es_ES.dic', '/usr/share/hunspell/es_ES.aff',
        '/usr/share/hunspell/es_MX.dic', '/usr/share/hunspell/es_MX.aff'
    ]

    for path in additional_dictionaries:
        hunspell_obj.add_dic(path)  # Add each additional dictionary individually
    
    return hunspell_obj

def load_hunspell_dictionaries():
    """Load Hunspell dictionaries with extended language support and custom dictionary."""
    hunspell_obj = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
    
    # Load each additional dictionary separately to avoid the argument error
    additional_dictionaries = [
        '/usr/share/hunspell/en_GB.dic', '/usr/share/hunspell/en_GB.aff',
        '/usr/share/hunspell/en_AU.dic', '/usr/share/hunspell/en_AU.aff',
        '/usr/share/hunspell/en_CA.dic', '/usr/share/hunspell/en_CA.aff',
        '/usr/share/hunspell/en_ZA.dic', '/usr/share/hunspell/en_ZA.aff',
        '/usr/share/hunspell/en_USNames.dic', '/usr/share/hunspell/en_USNames.aff',
        '/usr/share/hunspell/fr_FR.dic', '/usr/share/hunspell/fr_FR.aff',
        '/usr/share/hunspell/de_DE.dic', '/usr/share/hunspell/de_DE.aff',
        '/usr/share/hunspell/es_ES.dic', '/usr/share/hunspell/es_ES.aff',
        '/usr/share/hunspell/es_MX.dic', '/usr/share/hunspell/es_MX.aff'
    ]

    for path in additional_dictionaries:
        hunspell_obj.add_dic(path)  # Add each additional dictionary individually
    
    # Add custom dictionary last
    hunspell_obj.add_dic('filter_words.dic')
    
    return hunspell_obj

def update_custom_dictionary(hunspell_obj, new_terms):
    """
    Update the custom dictionary with new terms that should not be flagged as spelling errors.
    Ensure the dictionary's word count header is correctly updated.
    """
    dictionary_path = 'filter_words.dic'
    try:
        # Read the entire dictionary to update
        with open(dictionary_path, 'r') as file:
            lines = file.readlines()

        # First line contains the count of words in the dictionary
        if lines:
            current_count = int(lines[0].strip())
        else:
            current_count = 0
            lines.append('0\n')  # Initialize count if file was empty

        # Convert all terms to lowercase to avoid case sensitivity issues
        existing_terms = set(line.strip().lower() for line in lines[1:])
        new_unique_terms = [term.lower() for term in new_terms if term.lower() not in existing_terms]

        # Update count
        updated_count = current_count + len(new_unique_terms)
        lines[0] = f"{updated_count}\n"

        # Append new unique terms to the dictionary without additional newlines
        lines.extend(f"{term}\n" for term in new_unique_terms)

        # Rewrite the updated dictionary with a single final newline
        with open(dictionary_path, 'w') as file:
            file.writelines(lines)

    except Exception as e:
        print(f"Error updating the custom dictionary: {e}")

def get_term_match(vocabulary, term):
    """
    Queries the Getty Vocabulary Web Service for a term match within a specific vocabulary (AAT, ULAN, TGN).
    Returns detailed information if a match is found, otherwise returns None.
    """
    # Set up the endpoint URL
    url = f"{BASE_URLS[vocabulary]}/{vocabulary}GetTermMatch"
    
    # Configure parameters based on vocabulary type
    params = {}
    if vocabulary == "AAT":
        params = {"term": term, "logop": "and", "notes": ""}  # Empty "notes" for AAT
    elif vocabulary == "ULAN":
        params = {"name": term, "roleid": "", "nationid": ""}  # Empty "roleid" and "nationid" for ULAN
    elif vocabulary == "TGN":
        params = {"name": term, "placetypeid": "", "nationid": ""}  # Empty "placetypeid" and "nationid" for TGN

    # Construct the full request URL for debugging
    request_url = requests.Request("GET", url, params=params).prepare().url
    # print(f"\n🔍 Debug: Generated request URL:\n  curl -X GET \"{request_url}\"\n")

    try:
        # Perform the request
        print(f"🔄 Querying {vocabulary} for term '{term}'...")
        response = requests.get(request_url)
        
        # Print the raw response content for debugging
        # print("\n🔍 Debug: Raw response content:\n", response.text, "\n")
        
        response.raise_for_status()  # Raise an error for non-200 status codes

        # Parse the XML response and structure data
        root = ET.fromstring(response.content)
        subjects = []
        for subject in root.findall(".//Subject"):
            term_name = subject.find(".//Preferred_Term").text if subject.find(".//Preferred_Term") is not None else "N/A"
            subject_id = subject.find(".//Subject_ID").text if subject.find(".//Subject_ID") is not None else "N/A"
            hierarchy = subject.find(".//Preferred_Parent").text if subject.find(".//Preferred_Parent") is not None else "N/A"
            
            subjects.append({"term": term_name, "subject_id": subject_id, "hierarchy": hierarchy})

        if subjects:
            print(f"✅ Match found for '{term}' in {vocabulary}.")
            return subjects
        else:
            print(f"🚫 No match found for '{term}' in {vocabulary}.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Error querying {vocabulary} for term '{term}': {e}")
        return None

def validate_with_getty_vocabularies(term):
    """
    Validates a term across AAT, ULAN, and TGN vocabularies.
    Returns True if found in any of them, False otherwise.
    """
    # Check in each vocabulary and return True if a match is found
    for vocab in ["AAT", "ULAN", "TGN"]:
        result = get_term_match(vocab, term)
        if result:
            return True
    return False

def spelling_check(text, hunspell_obj):
    """
    Check spelling using Hunspell and Getty Vocabularies, then return unique misspelled words.
    """
    # Preprocess the text
    cleaned_text = preprocess_text_for_spellcheck(text)
    
    # Tokenize and lowercase the preprocessed text
    words = cleaned_text.lower().split()
    
    # Step 1: Identify misspelled words using Hunspell
    initial_misspelled_words = set(word for word in words if not hunspell_obj.spell(word))
    
    # Display initial misspelled words found by Hunspell
    if initial_misspelled_words:
        print(f"🚩 Initial Spelling Errors by Hunspell: {initial_misspelled_words} - Sending to Getty for validation.")
    
    # Step 2: Validate with Getty vocabularies and remove any recognized words
    misspelled_words_after_getty = set()
    for word in initial_misspelled_words:
        print(f"🔄 Querying Getty for term '{word}'...")
        if not validate_with_getty_vocabularies(word):  # Keep only words not recognized by Getty
            print(f"🚫 '{word}' not found in Getty. Adding to misspelled list.")
            misspelled_words_after_getty.add(word)
        else:
            print(f"✅ '{word}' found in Getty. Keeping in custom dictionary.")
    
    # Return the number of unique misspelled words and the specific errors after Getty validation
    return len(misspelled_words_after_getty), list(misspelled_words_after_getty)

def filter_spelling_errors_with_ai(spelling_errors_list):
    """
    Filters the spelling errors list by sending it to Gemini 1.5 Flash AI for review,
    ensuring that only clear and contextually irrelevant misspellings are flagged.
    """
    if not spelling_errors_list:
        return []

    # Convert the list into a lowercase string to send to the model
    spelling_errors_string = ', '.join(spelling_errors_list).lower()

    # Create the prompt for the AI to review
    prompt = f"""
    ROLE:
    You are a spelling expert with comprehensive knowledge of accurate word spelling across languages, domains, and contexts.

    TASK:
    1. Review the following list of flagged words, all lowercase, from various contexts.
    2. Return a list of only words that should NOT be considered misspelled in any context or domain.

    Words: {spelling_errors_string}

    RESPONSE CRITERIA:
    Include words that are valid in any context, such as:
    - Proper nouns (people's names, place names, brand names)
    - Technical terms (scientific, medical, or technical jargon)
    - Software or coding-related terms (e.g., Python, SQL, Kubernetes)
    - Foreign words (e.g., Spanish, German, Romanized Japanese, etc.)
    - Common abbreviations or acronyms across any field
    - Widely recognized terms in daily use or contemporary culture that may not appear in standard dictionaries

    EXCLUDE:
    - Do not include slang (e.g., "flocka") or casual abbreviations (e.g., "lol").
    - Do not include "squashed words" i.e. multiple valid words without spaces in between (e.g., "nobelprize", "workfromhome", "blacklivesmatter")

    YOUR RESPONSE:
    Return the list of words that should NOT be treated as misspelled, separated by commas, with NO explanation whatsoever. If none, respond with "None"
    """

    try:
        # Configure the Google API with your API key
        genai.configure(api_key=google_api_key)

        # Initialize the Google model instance
        google_model_instance = genai.GenerativeModel(google_model)

        # Generate the response using the Gemini model
        response = google_model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,  # Currently, only one candidate is supported
                max_output_tokens=150,  # Adjust this based on your needs
                temperature=0  # Adjust for creativity level
            ),
        )

        # Process the response
        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                # Extract text parts and form the full response, in lowercase
                filtered_terms = ''.join(part.text for part in candidate.content.parts if part.text).strip().lower()

                if filtered_terms.lower() == 'none':
                    return []

                # Convert the filtered terms back into a list format
                return [term.strip() for term in filtered_terms.split(',') if term.strip()]
        return []

    except Exception as e:
        print(f"❌ Error during AI review with Gemini API: {e}")
        return []

def is_word_valid_in_context(word, context_text):
    """
    Sends a single word along with its context to Gemini for validation. Returns True if Gemini deems it correct, else False.
    """
    prompt = f"""
    ROLE:
    You are a spelling expert with comprehensive knowledge of accurate word spelling across languages, domains, and contexts.

    TASK:
    The following word was flagged as potentially misspelled in the given context. Determine if the word is valid in the provided context.

    WORD: "{word}"
    CONTEXT: "{context_text}"

    RESPONSE CRITERIA:
    Respond "Yes" if the word is correctly spelled or valid in this context. Examples:
    - Proper nouns (names of people, places, brands)
    - Technical terms (scientific, medical, technological jargon)
    - Software or coding terms (e.g., Python, SQL, Kubernetes)
    - Foreign words (Spanish, German, etc.)
    - Common abbreviations or acronyms

    Respond "No" if the word is likely misspelled or out of context.

    YOUR RESPONSE (Yes/No only):
    """

    try:
        # Configure the Google API with your API key
        genai.configure(api_key=google_api_key)

        # Initialize the Google model instance
        google_model_instance = genai.GenerativeModel(google_model)

        # Send the prompt to Gemini
        response = google_model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=5,  # Only expecting 'Yes' or 'No'
                temperature=0
            ),
        )
        
        # Process the response to extract "Yes" or "No"
        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and isinstance(candidate.content, str):
                # Safely strip the content and check if it matches "yes"
                return candidate.content.strip().lower() == "yes"
            elif hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                # For structured content, join parts if needed
                full_content = ''.join(part.text for part in candidate.content.parts if part.text).strip().lower()
                return full_content == "yes"
    
    except Exception as e:
        print(f"❌ Error validating '{word}' in context with Gemini API: {e}")
        return False

    return False

def lemmatize_text(text):
    # Ensure input is a string, if not convert to an empty string
    if not isinstance(text, str):
        text = ""
    doc = nlp(text)
    lemmatized = " ".join([token.lemma_ for token in doc])
    return lemmatized

def extract_noun_phrases(text):
    if isinstance(text, str):
        doc = nlp(text)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        return noun_phrases
    else:
        # Return an empty list if text is not a valid string
        return []

def count_sentences(text):
    # Check if text is a valid string
    if not isinstance(text, str) or pd.isna(text):
        return 0  # Return 0 sentences for invalid or missing content
    blob = TextBlob(text)
    sentences_total = len(blob.sentences)
    return sentences_total

def count_chars(text):
    if isinstance(text, str):
        return len(text)
    return 0  # Return 0 for non-string values

def count_words(text):
    if isinstance(text, str):
        blob = TextBlob(text)
        return len(blob.words)
    return 0  # Return 0 for non-string values

def count_tokens(text):
    if isinstance(text, str):
        return len(tokenizer.tokenize(text))
    return 0  # Return 0 for non-string values

def extract_named_entities(text):
    if isinstance(text, str):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    else:
        # Return an empty list if text is not a valid string
        return []

def analyze_polarity(text):
    # Ensure the input is a valid string
    if not isinstance(text, str):
        text = ""

    # Use TextBlob to calculate the polarity
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    return sentiment_polarity

def analyze_subjectivity(text):
    # Ensure the input is a valid string
    if not isinstance(text, str):
        text = ""

    # Use TextBlob to calculate the subjectivity
    blob = TextBlob(text)
    sentiment_subjectivity = blob.sentiment.subjectivity
    return sentiment_subjectivity

def token_level_matching(text1, text2):
    # Ensure both inputs are valid strings
    if not isinstance(text1, str):
        text1 = ""
    if not isinstance(text2, str):
        text2 = ""

    # Tokenize the texts
    vectorizer = CountVectorizer().fit([text1, text2])
    tokens1 = set(vectorizer.build_analyzer()(text1))
    tokens2 = set(vectorizer.build_analyzer()(text2))
    
    # Calculate matching tokens
    matching_tokens = tokens1.intersection(tokens2)
    
    # Calculate the token-level match percentage
    if len(tokens1) == 0:
        return 0.0  # Handle edge case for empty text
    token_match_percentage = len(matching_tokens) / len(tokens1)
    return token_match_percentage

# Compute Semantic Similarity
def compute_semantic_similarity(text1, text2, tokenizer, model):
    # Ensure both inputs are valid strings
    if not isinstance(text1, str):
        text1 = ""
    if not isinstance(text2, str):
        text2 = ""

    # Compute embeddings for both texts
    embedding1 = get_embedding(text1, tokenizer, model)
    embedding2 = get_embedding(text2, tokenizer, model)

    # Compute cosine similarity between the embeddings
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
    return similarity[0][0]

# Convert Text to Embeddings
def get_embedding(text, tokenizer, model):
    # Ensure the input is a valid string
    if not isinstance(text, str):
        text = ""

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def check_word_frequency(text):
    # Ensure the input is a valid string
    if not isinstance(text, str):
        text = ""

    # Convert text to lowercase and remove punctuation to ensure consistency
    text_lower = text.lower()
    text_clean = text_lower.translate(str.maketrans('', '', string.punctuation))
    
    word_counts = {}
    
    # Count individual flagged words
    for word in FLAGGED_WORDS:
        word_counts[word] = text_clean.count(word)
    
    # Count flagged phrases
    for phrase in FLAGGED_PHRASES:
        phrase_clean = phrase.lower().translate(str.maketrans('', '', string.punctuation))
        word_counts[phrase] = text_clean.count(phrase_clean)

    return word_counts

def calculate_total_flagged_words(flagged_words_str):
    total_count = 0
    
    # Ensure the input is a valid string
    if not isinstance(flagged_words_str, str) or not flagged_words_str:
        return total_count  # Return 0 if the input is invalid or empty
    
    flagged_words = flagged_words_str.split(", ")  # Split by commas
    for word_count in flagged_words:
        try:
            word, count = word_count.split(": ")  # Split each word and count
            total_count += int(count)  # Add the count to the total
        except ValueError:
            # Handle cases where splitting or conversion fails (e.g., malformed strings)
            print(f"Error processing flagged word: {word_count}")
            continue
    
    return total_count

# Compute Cosine Similarity
def compute_cosine_similarity(text1, text2):
    try:
        # Ensure both inputs are valid strings, if not convert to empty strings
        if not isinstance(text1, str):
            text1 = ""
        if not isinstance(text2, str):
            text2 = ""

        # Use the TfidfVectorizer to convert text to vectors
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2])

        # Compute the cosine similarity between the vectors
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        return similarity
    except Exception as e:
        print(f"Error processing texts: {text1}, {text2} - Error: {e}")
        return None  # Returning None to handle errors

# Load transformer model for summarization (local models)
def load_summarization_model(model_name='facebook/bart-large-cnn'):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def load_summarization_model2(model_name='google/pegasus-large'):
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def load_summarization_model3(model_name='t5-large'):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def summarize_text(text, tokenizer, model, max_length=150):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Summarize using Gemini (for longer texts > 512 tokens)
def summarize_with_gemini(text):
    try:
        # Configure the Google API with the API key
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

        # Prepare the model configuration
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=512,
            temperature=0.5,  # Adjust temperature if needed
        )

        # Send the prompt to Gemini
        response = genai.GenerativeModel(model="gemini-1.5-flash").generate_content(
            text, generation_config=generation_config
        )
        if not response.candidates:
            print(f"No candidates received from {model_name}. Check the input or API call.")
            return []
        if response.candidates:
            candidate = response.candidates[0]
            return candidate.get("content", "").strip()
        else:
            return "No valid response"

    except Exception as e:
        print(f"Error with Gemini summarization: {e}")
        return "Error"

def summarize_based_on_token_count(text, tokenizer, model, max_token_count=500):
    """
    Summarize the text based on token count. If token count is within the limit, 
    use the local transformer model. Otherwise, use Gemini for summarization.
    """
    token_count = count_tokens(text)

    if token_count <= max_token_count:
        # If token count is within the local model's limit
        print(f"Using local transformer model for summarization (token count: {token_count}).")
        summary = summarize_text(text, tokenizer, model)
    else:
        # If token count exceeds the limit, use Gemini
        print(f"Text exceeds {max_token_count} tokens (actual: {token_count}). Using Gemini for summarization.")
        summary = summarize_with_gemini(text)

    return summary

# Function to summarize the text based on token count
def evaluate_and_summarize_response(response, tokenizer, model):
    # Call the generalized summarization function
    summary = summarize_based_on_token_count(response, tokenizer, model)

    return summary

# API-based AI evaluation logic for Gemini
def evaluate_response_with_model(response, prompt, eval_type, model_name, current_mode, benchmark_response1=None, benchmark_response2=None):
    """
    Sends a specific evaluation prompt (Accuracy, Clarity, etc.) to the specified model's API 
    (Gemini or Cohere) and returns the evaluation rating and explanation. For Variance evaluation, 
    it combines the prompt from eval_prompts.json with the Msg_Content_Variance.
    """
    with open('eval_prompts.json', 'r') as f:
        eval_prompts = json.load(f)

    # Get the appropriate evaluation prompt for the given type (e.g., Accuracy)
    eval_prompt_template = next((ep['prompt'] for ep in eval_prompts['evaluations'] if ep['name'] == eval_type), None)

    if not eval_prompt_template:
        raise ValueError(f"Evaluation type '{eval_type}' not found in eval_prompts.json")

    # Handle Variance evaluation by combining the JSON prompt template with the Msg_Content_Variance
    if eval_type == "Variance":
        # Combine the evaluation prompt (from JSON) and response (from Excel's Msg_Content_Variance)
        eval_prompt = eval_prompt_template + response  # 'response' here is the content from Msg_Content_Variance
        print(f"✅ Variance evaluation prompt prepared with benchmarks for {model_name}")
    else:
        # For other types, combine the prompt and response as usual
        eval_prompt = eval_prompt_template.replace("<prompt>", prompt).replace("<response>", response)

    # Print the full payload before sending to the API for debugging purposes
    print(f"🚀 Sending evaluation to {model_name} API for prompt: \n{prompt}\n")
    # Send evaluation prompt to model API
    try:
        first_choice_content, response_time, content_length, word_count = generate(model_name, eval_prompt, current_mode)

        if not first_choice_content or not isinstance(first_choice_content, str):
            raise ValueError(f"No valid response generated for {model_name} on {eval_type}")

        # Process the result for Variance or other evaluation types
        if eval_type == "Variance":
            rating1, explanation1, rating2, explanation2 = extract_double_variance(first_choice_content)
            return rating1, explanation1, rating2, explanation2
        else:
            rating, explanation = extract_standard_evaluation(first_choice_content)
            return rating, explanation

    except Exception as e:
        print(f"❌ Error evaluating {eval_type} for {model_name}: {e}")
        return None, f"Error: {e}"

def extract_standard_evaluation(evaluation_response):
    """
    Extract the rating and explanation from the evaluation response for non-variance evaluations.
    This assumes the response has a structure that includes a numeric rating and a text explanation.
    """
    # Example logic to parse the response. Adjust based on how the response is formatted.
    try:
        # Assuming the format is something like "Rating: ###9### - Explanation text"
        rating_part = evaluation_response.split("###")[1]  # Get the rating between "###"
        explanation_part = evaluation_response.split("###")[2].strip()  # Get the explanation after the rating
        rating = int(rating_part)  # Convert the rating to an integer
        explanation = explanation_part  # The explanation follows the rating
    except (IndexError, ValueError) as e:
        # Handle cases where the format might not match expectations
        raise ValueError(f"Error parsing evaluation response: {evaluation_response}") from e
    
    return rating, explanation

def extract_rating(evaluation_response):
    """
    Extracts the numerical rating or special values ('-' or 'N/A') from the evaluation response.
    Assumes the format '###<rating>###', where <rating> can be a number (0-10), '-' or 'N/A'.
    Any other values will be ignored.
    """
    import re
    # Match for numbers (0-10), '-', or 'N/A' between the ### ###
    match = re.search(r'###(\d{1,2}|N/A|-)###', evaluation_response)

    if match:
        rating = match.group(1)
        # Ensure numbers are within the 0-10 range if it's a digit
        if rating.isdigit():
            rating = int(rating)
            if 0 <= rating <= 10:
                return rating
            else:
                return None  # Ignore invalid number ratings outside 0-10 range
        elif rating in ['-', 'N/A']:
            return rating
    return None  # Return None for any invalid format

def extract_explanation(evaluation_response):
    """
    Extracts the explanation part from the evaluation response.
    Assumes it comes after the rating, in the format:
    'Clarity: ###<rating>### - <explanation>'
    """
    return evaluation_response.split(' - ')[1].strip() if ' - ' in evaluation_response else None

def extract_double_variance(evaluation_response):
    """
    Extracts two numerical ratings and explanations from the evaluation response for variance.
    Assumes the format:
    'Variance 1: ###<rating1>### - <explanation1>'
    'Variance 2: ###<rating2>### - <explanation2>'
    """
    import re
    # Match for Variance 1
    match1 = re.search(r'Variance 1: ###(\d+)### - (.*?)(?=Variance 2|$)', evaluation_response, re.DOTALL)
    # Match for Variance 2
    match2 = re.search(r'Variance 2: ###(\d+)### - (.*)', evaluation_response, re.DOTALL)
    
    if match1 and match2:
        rating1 = int(match1.group(1))
        explanation1 = match1.group(2).strip()
        rating2 = int(match2.group(1))
        explanation2 = match2.group(2).strip()
        return rating1, explanation1, rating2, explanation2
    else:
        # If parsing fails, return None values
        return None, "Variance 1 extraction failed", None, "Variance 2 extraction failed"