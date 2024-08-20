#!/usr/bin/env python3
# analyze_excel.py

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

import pandas as pd
from text_processing import *
import nltk
nltk.download('punkt_tab')

# Function to calculate and update Cosine Similarity
def process_cosine_similarity(df):
    for index, row in df.iterrows():
        if pd.isna(row['Cosine_Similarity']):
            similarity = compute_cosine_similarity(row['Msg_Content'], row['Benchmark_Response'])
            df.at[index, 'Cosine_Similarity'] = similarity
            print(f"Row {index+1}: Cosine Similarity between model response and benchmark response: {similarity}")

# Function to calculate and update Polarity Sentiment
def process_polarity_sentiment(df):
    for index, row in df.iterrows():
        if pd.isna(row['Polarity_Sentiment']):
            polarity = analyze_polarity(row['Msg_Content'])
            df.at[index, 'Polarity_Sentiment'] = polarity
            print(f"Row {index+1}: Polarity Sentiment: {polarity}")

# Function to calculate and update Subjective Sentiment
def process_subjective_sentiment(df):
    for index, row in df.iterrows():
        if pd.isna(row['Subjective_Sentiment']):
            subjectivity = analyze_subjectivity(row['Msg_Content'])
            df.at[index, 'Subjective_Sentiment'] = subjectivity
            print(f"Row {index+1}: Subjective Sentiment: {subjectivity}")

# Function to calculate and update Sentence Count
def process_sentence_count(df):
    for index, row in df.iterrows():
        if pd.isna(row['Sentences_Total']):
            sentence_count = count_sentences(row['Msg_Content'])
            df.at[index, 'Sentences_Total'] = sentence_count
            print(f"Row {index+1}: Sentence Count: {sentence_count}")

# Function to calculate and update Token Count
def process_token_count(df):
    for index, row in df.iterrows():
        if pd.isna(row['Tokens_Total']):
            token_count = count_tokens(row['Msg_Content'])
            df.at[index, 'Tokens_Total'] = token_count
            print(f"Row {index+1}: Token Count: {token_count}")

# Function to calculate and update Character Count
def process_char_count(df):
    for index, row in df.iterrows():
        if pd.isna(row['Chars_Total']):
            char_count = count_chars(row['Msg_Content'])
            df.at[index, 'Chars_Total'] = char_count
            print(f"Row {index+1}: Character Count: {char_count}")

# Function to calculate and update Word Count
def process_word_count(df):
    for index, row in df.iterrows():
        if pd.isna(row['Words_Total']):
            word_count = count_words(row['Msg_Content'])
            df.at[index, 'Words_Total'] = word_count
            print(f"Row {index+1}: Word Count: {word_count}")

# Function to extract and update Noun Phrases
def process_noun_phrases(df):
    for index, row in df.iterrows():
        if pd.isna(row['Noun_Phrases']):
            noun_phrases = extract_noun_phrases(row['Msg_Content'])
            df.at[index, 'Noun_Phrases'] = ', '.join(noun_phrases)
            print(f"Row {index+1}: Noun Phrases: {noun_phrases}")

# Main processing function to run all analyses
def process_excel(file_path, sheet_name="Model_Responses", last_row=2144):
    # Load the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    
    # Ensure the dataframe is truncated at the last row of interest
    df = df.iloc[:last_row]
    
    print("🔄 Running analyses on the Excel sheet...\n")

    # Run each processing function
    process_cosine_similarity(df)
    process_polarity_sentiment(df)
    process_subjective_sentiment(df)
    process_sentence_count(df)
    process_token_count(df)
    process_char_count(df)
    process_word_count(df)
    process_noun_phrases(df)

    # Save the modified dataframe back to Excel
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

    print("All analyses have been calculated and saved to the Excel file. ✅")

# Run the process on the specific Excel file
process_excel('prompt_responses.xlsx')