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

# Function to extract URLs from text
def extract_urls(text):
    url_pattern = r'(https?://\S+|www\.\S+)'
    urls = re.findall(url_pattern, text)
    return urls

# Function to detect if text is code-related
def detect_code_related(text):
    # Define common programming-related keywords and patterns
    code_keywords = [
        'def', 'class', 'import', 'return', 'from', 'if', 'else', 'elif', 
        'for', 'while', 'try', 'except', 'with', 'lambda', 'async', 'await',
        'static', 'int', 'string', 'kubectl',
        'System.out', 'console.log', 'function', 'var', 'let', 'const'
    ]
    
    # Check if any code-related keywords or patterns are in the text
    if any(keyword in text for keyword in code_keywords):
        return 'Y'
    else:
        return ''

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

# Compute Cosine Similarity on Embeddings
def compute_semantic_similarity(text1, text2, tokenizer, model):
    embedding1 = get_embedding(text1, tokenizer, model)
    embedding2 = get_embedding(text2, tokenizer, model)
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
    return similarity[0][0]

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