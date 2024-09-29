#!/usr/bin/env python3

"""
text_processing.py
"""

__author__ = "Alex Bishop"
__version__ = "0.1.0"
__license__ = "MIT"

from textblob import TextBlob
import spacy
from transformers import GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nlp = spacy.load("en_core_web_sm")
import numpy as np
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
from spellchecker import SpellChecker
import torch
from transformers import BertModel, BertTokenizer
import re
import json
import pandas as pd
from openpyxl import load_workbook
import google.generativeai as genai
from generation import generate
import string
from config import FLAGGED_WORDS, FLAGGED_PHRASES

# Function to load tech terms from JSON
def load_filter_terms(file_path="filter_terms.json"):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data.get("filter_terms", [])

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_content = " ".join([token.lemma_ for token in doc])
    return lemmatized_content

def extract_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    return noun_phrases

def count_chars(text):
    blob = TextBlob(text)
    chars_total = len(text)
    return chars_total

def count_words(text):
    blob = TextBlob(text)
    words_total = len(blob.words)
    return words_total

def count_sentences(text):
    blob = TextBlob(text)
    sentences_total = len(blob.sentences)
    return sentences_total

def count_tokens(text):
    blob = TextBlob(text)
    tokens_total = len(tokenizer.tokenize(text))
    return tokens_total

def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def analyze_polarity(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    return sentiment_polarity

def analyze_subjectivity(text):
    blob = TextBlob(text)
    sentiment_subjectivity = blob.sentiment.subjectivity
    return sentiment_subjectivity

def preprocess_text_for_spellcheck(text):
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

    # Remove punctuation
    text = re.sub(r'[\*\#\@\&\$\(\)\[\]\{\}\<\>\:;,\.\!\?\"]+', '', text)

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
    
    # Replace non-alphanumeric characters (except spaces) with space, like emoji's
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

# Convert Text to Embeddings
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def check_word_frequency(text):
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
    
    if flagged_words_str:
        flagged_words = flagged_words_str.split(", ")  # Split by commas
        for word_count in flagged_words:
            word, count = word_count.split(": ")  # Split each word and count
            total_count += int(count)  # Add the count to the total
    
    return total_count

# Compute Semantic Similarity
def compute_semantic_similarity(text1, text2, tokenizer, model):
    embedding1 = get_embedding(text1, tokenizer, model)
    embedding2 = get_embedding(text2, tokenizer, model)
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
    return similarity[0][0]

#Compute Cosine Similarity
def compute_cosine_similarity(text1, text2):
    try:
        # Ensure both inputs are strings
        if not isinstance(text1, str):
            text1 = str(text1)
        if not isinstance(text2, str):
            text2 = str(text2)

        # Use the TfidfVectorizer to convert text to vectors
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2])

        # Compute the cosine similarity between the vectors
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        return similarity
    except Exception as e:
        print(f"Error processing texts: {text1}, {text2} - Error: {e}")
        return None

# API-based AI evaluation logic
def evaluate_response_with_gemini(response, prompt, eval_type, benchmark_response=None):
    """
    Sends a specific evaluation prompt (Accuracy, Clarity, etc.) to the Gemini API 
    and returns the evaluation rating and explanation. For Variance evaluation, 
    it compares the response with a benchmark.
    """
    # Load the evaluation prompts from eval_prompts.json
    with open('eval_prompts.json', 'r') as f:
        eval_prompts = json.load(f)

    # Get the appropriate evaluation prompt for the given type (e.g., Accuracy)
    eval_prompt_template = next((ep['prompt'] for ep in eval_prompts['evaluations'] if ep['name'] == eval_type), None)
    
    if not eval_prompt_template:
        raise ValueError(f"Evaluation type '{eval_type}' not found in eval_prompts.json")
    
    # Replace placeholders in the template with the actual prompt and response
    eval_prompt = eval_prompt_template.replace("<prompt>", prompt).replace("<response>", response)

    # Handle the special case for Variance where benchmark response is needed
    if eval_type == "Variance" and benchmark_response:
        eval_prompt = eval_prompt.replace("<benchmark_response>", benchmark_response)

    # Send the evaluation prompt to the Gemini API
    model = "gemini-1.5-flash"
    _, evaluation_response, response_time, _, _ = generate(model, eval_prompt)

    # Extract rating and explanation from the evaluation response using regex or predefined parsing logic
    rating = extract_rating(evaluation_response)
    explanation = extract_explanation(evaluation_response)

    return rating, explanation

# API-based AI evaluation logic
def evaluate_response_with_mistral(response, prompt, eval_type, benchmark_response=None):
    """
    Sends a specific evaluation prompt (Accuracy, Clarity, etc.) to the Mistral API 
    and returns the evaluation rating and explanation. For Variance evaluation, 
    it compares the response with a benchmark.
    """
    # Load the evaluation prompts from eval_prompts.json
    with open('eval_prompts.json', 'r') as f:
        eval_prompts = json.load(f)

    # Get the appropriate evaluation prompt for the given type (e.g., Accuracy)
    eval_prompt_template = next((ep['prompt'] for ep in eval_prompts['evaluations'] if ep['name'] == eval_type), None)
    
    if not eval_prompt_template:
        raise ValueError(f"Evaluation type '{eval_type}' not found in eval_prompts.json")
    
    # Replace placeholders in the template with the actual prompt and response
    eval_prompt = eval_prompt_template.replace("<prompt>", prompt).replace("<response>", response)

    # Handle the special case for Variance where benchmark response is needed
    if eval_type == "Variance" and benchmark_response:
        eval_prompt = eval_prompt.replace("<benchmark_response>", benchmark_response)

    # Send the evaluation prompt to the Mistral API
    model = "mistral-large"
    _, evaluation_response, response_time, _, _ = generate(model, eval_prompt)

    # Extract rating and explanation from the evaluation response using regex or predefined parsing logic
    rating = extract_rating(evaluation_response)
    explanation = extract_explanation(evaluation_response)

    return rating, explanation

def extract_rating(evaluation_response):
    """
    Extracts the numerical rating from the evaluation response.
    Assumes the format '###<rating>###'.
    """
    import re
    match = re.search(r'###(\d+)###', evaluation_response)
    if match:
        return int(match.group(1))
    else:
        return None

def extract_explanation(evaluation_response):
    """
    Extracts the explanation part from the evaluation response.
    Assumes it comes after the rating.
    """
    return evaluation_response.split(' - ')[1].strip() if ' - ' in evaluation_response else None