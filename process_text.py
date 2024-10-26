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
from spellchecker import SpellChecker
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
FILTER_TERMS_FILE = "filter_terms.json"
from config import *

def load_filter_terms():
    """
    Load the filter terms from the JSON file. If the file doesn't exist, return an empty list.
    """
    if os.path.exists(FILTER_TERMS_FILE):
        with open(FILTER_TERMS_FILE, 'r') as f:
            data = json.load(f)
        return data.get('filter_terms', [])
    else:
        return []

def update_filter_terms(new_terms):
    """
    Update the filter terms JSON file with new terms.
    """
    filter_terms = load_filter_terms()  # Load existing terms

    # Add new terms, avoiding duplicates
    updated_terms = set(filter_terms).union(set(new_terms))

    # Write back to the JSON file
    with open(FILTER_TERMS_FILE, 'w') as f:
        json.dump({"filter_terms": list(updated_terms)}, f, indent=4)
    
    print(f"🔄 {len(new_terms)} new terms added to filter list.")

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

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

def preprocess_text_for_spellcheck(text):
    # Check if the input is valid
    if not isinstance(text, str):
        return ""  # Return an empty string if the input is invalid

    # Load filter terms    
    filter_terms = load_filter_terms()

    # Apply NER to remove entities
    doc = nlp(text)
    entities = {ent.text for ent in doc.ents}

    # Lowercase the text
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Replace ellipses and periods with spaces to avoid merging words
    text = re.sub(r'\.\.+', ' ', text)  # Replace ellipses
    text = re.sub(r'\.', ' ', text)      # Replace single periods

    # Remove other punctuation, except for periods already replaced with spaces
    text = re.sub(r'[\*\#\@\&\$\(\)\[\]\{\}\<\>\:;,\!\?\"]+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove common technical terms, abbreviations, proper nouns, and code snippets
    filter_pattern = r'\b(' + '|'.join(filter_terms) + r')\b'
    text = re.sub(filter_pattern, '', text)

    # Expand common abbreviations (example dictionary)
    abbreviations = {
        "don't": "do not",
        "can't": "cannot",
        "i'm": "i am",
        "he's": "he is",
        "let's": "let us",
        "they're": "they are"
    }
    
    # Replace non-alphanumeric characters (except spaces) with space, like emojis
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    for abbr, full_form in abbreviations.items():
        text = re.sub(r'\b' + abbr + r'\b', full_form, text)

    # Remove named entities identified by NER
    for entity in entities:
        text = text.replace(entity.lower(), "")

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def spelling_check(text):
    spell = SpellChecker()
    
    # Preprocess the text
    cleaned_text = preprocess_text_for_spellcheck(text)
    
    # Tokenize the preprocessed text
    words = cleaned_text.split()
    misspelled = spell.unknown(words)
    
    # Calculate the number of spelling errors
    spelling_errors = len(misspelled)
    
    # Return both the number of spelling errors and the specific errors
    return spelling_errors, misspelled

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

def filter_spelling_errors_with_ai(spelling_errors_list):
    """
    Filters the spelling errors list by sending it to Gemini 1.5 Flash AI for review.
    """
    # Convert the list into a string to send to the model
    spelling_errors_string = ', '.join(spelling_errors_list)

    # Create the prompt for the Gemini AI to review
    prompt = f"""
    Review the following list of flagged words for potential spelling errors:

    Words: {spelling_errors_string}

    Your task:
    - Identify only the words that should NOT be considered misspelled.
    - These may include:
        - Proper nouns (people's names, place names)
        - Technical terms (e.g., scientific, medical, or technical jargon)
        - Foreign words such as Spanish, German, Sanskrit, Latin, French, Romanized Japanese, Pinyin, etc.
        - Common abbreviations or acronyms.

    Return the list of words that should NOT be treated as misspelled, separated by commas, with no other details. If there are none, say 'None'.
    """

    try:
        # Configure the Google API with your API key (set in environment)
        genai.configure(api_key=google_api_key)

        # Initialize the Google model instance
        google_model_instance = genai.GenerativeModel(google_model)

        # Generate the response using the Gemini model
        response = google_model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,  # Currently, only one candidate is supported
                max_output_tokens=google_max_tokens,  # Adjust this based on your needs
                temperature=google_temperature,  # Adjust for creativity level
            ),
        )

        # Process the response
        if response.candidates:
            candidate = response.candidates[0]

            # Check if 'content' is a structured object with 'parts'
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                # Join text parts to form the full response
                filtered_terms = ''.join(part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text)
            else:
                # If 'content' is not structured, assume it's a string and strip whitespace
                filtered_terms = candidate.content.strip() if isinstance(candidate.content, str) else ''

            # Convert the filtered terms back into a list format
            return [term.strip() for term in filtered_terms.split(',') if term]
        else:
            return []

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return []

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