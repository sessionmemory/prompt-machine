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
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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

def analyze_polarity(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    return sentiment_polarity

def analyze_subjectivity(text):
    blob = TextBlob(text)
    sentiment_subjectivity = blob.sentiment.subjectivity
    return sentiment_subjectivity

def spelling_check(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    # Count how many words were corrected
    total_words = len(text.split())
    corrections = sum(1 for original, corrected in zip(text.split(), corrected_text.split()) if original != corrected)
    # Calculate the spelling accuracy
    if total_words == 0:
        return 1.0  # Handle edge case for empty text
    spelling_accuracy = 1 - (corrections / total_words)
    return spelling_accuracy

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