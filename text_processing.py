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