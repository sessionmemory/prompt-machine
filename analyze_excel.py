#!/usr/bin/env python3
# analyze_excel.py

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"
from bert_score import score as bert_score
import pandas as pd
from text_processing import *
import nltk
import os
import warnings
import time
from config import sleep_time_api

# Suppress all warnings
warnings.filterwarnings("ignore")

# Function to calculate and update Cosine Similarity
def process_cosine_similarity(df):
    for index, row in df.iterrows():
        if pd.isna(row['Cosine_Similarity']):
            similarity = compute_cosine_similarity(row['Msg_Content'], row['Benchmark_Response-Import'])
            df.at[index, 'Cosine_Similarity'] = similarity
            print(f"Row {index+1}: Cosine Similarity between model response and benchmark response: {similarity}")

# Function to calculate and update Polarity Sentiment
def process_polarity_sentiment(df):
    for index, row in df.iterrows():
        if pd.isna(row['Sentiment_Polarity']):
            polarity = analyze_polarity(row['Msg_Content'])
            df.at[index, 'Sentiment_Polarity'] = polarity
            print(f"Row {index+1}: Sentiment Polarity: {polarity}")

# Function to calculate and update Subjective Sentiment
def process_subjective_sentiment(df):
    for index, row in df.iterrows():
        if pd.isna(row['Sentiment_Subjectivity']):
            subjectivity = analyze_subjectivity(row['Msg_Content'])
            df.at[index, 'Sentiment_Subjectivity'] = subjectivity
            print(f"Row {index+1}: Sentiment Subjectivity: {subjectivity}")

def calculate_bertscore(text1, text2):
    # Calculate Precision, Recall, F1 using BERTScore
    P, R, F1 = bert_score([text1], [text2], lang="en", rescale_with_baseline=True)
    return P.mean().item(), R.mean().item(), F1.mean().item()

def process_bertscore(df, file_path, sheet_name):
    # Ensure the columns exist
    if 'BERT_Precision' not in df.columns:
        df['BERT_Precision'] = pd.NA
    if 'BERT_Recall' not in df.columns:
        df['BERT_Recall'] = pd.NA
    if 'BERT_F1' not in df.columns:
        df['BERT_F1'] = pd.NA

    for index, row in df.iterrows():
        if pd.isna(row['BERT_F1']):
            if 'Benchmark_Response-Import' in row:
                P, R, F1 = calculate_bertscore(row['Msg_Content'], row['Benchmark_Response-Import'])
                df.at[index, 'BERT_Precision'] = P
                df.at[index, 'BERT_Recall'] = R
                df.at[index, 'BERT_F1'] = F1
                
                print(f"Row {index+1}: BERTScore - Precision: {P}, Recall: {R}, F1: {F1}")
    
    # Save results
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

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

def process_named_entities(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['Named_Entities']):
            entities = extract_named_entities(row['Msg_Content'])
            df.at[index, 'Named_Entities'] = str(entities)
            print(f"Row {index+1}: Named Entities: {entities}")
    # Save results
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

# Function to check spelling and process the dataframe
def process_spelling(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['Spelling_Errors']):
            spelling_errors, misspelled_words = spelling_check(row['Msg_Content'])
            df.at[index, 'Spelling_Error_Qty'] = spelling_errors
            df.at[index, 'Spelling_Errors'] = ', '.join(misspelled_words)
            print(f"Row {index+1}: Spelling Errors: {spelling_errors} - Misspelled Words: {misspelled_words}")
    
    # Save results
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

def process_flagged_words(df):
    for index, row in df.iterrows():
        # If the flagged words haven't been processed yet, proceed
        if pd.isna(row['Flagged_Words']):
            flagged_word_counts = check_word_frequency(row['Msg_Content'])
            # Create a string that summarizes the flagged words and their counts
            flagged_summary = ', '.join([f"{word}: {count}" for word, count in flagged_word_counts.items() if count > 0])
            
            # Store the summary in the "Flagged_Words" column
            df.at[index, 'Flagged_Words'] = flagged_summary
            print(f"Row {index+1}: Flagged Words & Phrases: {flagged_summary}")
        
        # Calculate the total number of flagged words/phrases
        flagged_penalty = calculate_total_flagged_words(df.at[index, 'Flagged_Words'])
        
        # Store the total count in the "Flagged_Penalty" column
        df.at[index, 'Flagged_Penalty'] = flagged_penalty
        print(f"Row {index+1}: Flagged Penalty: {flagged_penalty}")
    
    # Save changes back to Excel (this can be done later when all analyses are completed)
    df.to_excel("prompt_responses_rated.xlsx", sheet_name="Model_Responses", index=False)

def process_cosine_similarity_with_lemmatization(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['Cosine_Similarity']):
            lemmatized_msg_content = lemmatize_text(row['Msg_Content'])
            lemmatized_benchmark_response = lemmatize_text(row['Benchmark_Response-Import'])
            similarity = compute_cosine_similarity(lemmatized_msg_content, lemmatized_benchmark_response)
            df.at[index, 'Cosine_Similarity'] = similarity
            print(f"Row {index+1}: Cosine Similarity after Lemmatization: {similarity}")

    # Save results back to Excel
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

# Same can be done for token matching
def process_token_matching_with_lemmatization(df):
    for index, row in df.iterrows():
        if pd.isna(row['Token_Matching']):
            lemmatized_msg_content = lemmatize_text(row['Msg_Content'])
            lemmatized_benchmark_response = lemmatize_text(row['Benchmark_Response-Import'])
            match_score = token_level_matching(lemmatized_msg_content, lemmatized_benchmark_response)
            df.at[index, 'Token_Matching'] = match_score
            print(f"Row {index+1}: Token Matching after Lemmatization: {match_score}")

# Function to check token-level matching and process the dataframe
def process_token_matching(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['Token_Match']):
            token_match = token_level_matching(row['Msg_Content'], row['Benchmark_Response-Import'])
            df.at[index, 'Token_Match'] = token_match
            print(f"Row {index+1}: Token Match: {token_match:.2f}")
    # Save results
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

def process_semantic_similarity(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['Semantic_Similarity']):
            similarity = compute_semantic_similarity(row['Msg_Content'], row['Benchmark_Response-Import'], tokenizer, model)
            df.at[index, 'Semantic_Similarity'] = similarity
            print(f"Row {index+1}: Semantic Similarity: {similarity}")
    # Save results
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

def process_gemini_evaluations(df, output_file):
    """
    Process the DataFrame, evaluate each response with Gemini, 
    and store the results in the new columns for each evaluation aspect.
    """
    # Loop through each row (response) in the DataFrame
    for index, row in df.iterrows():
        prompt = row['Prompt_Text']  # Assuming this is the column name for prompts
        response = row['Msg_Content']  # Assuming this is the column name for responses
        
        # Perform evaluations for each aspect
        print("🔄 'Gemini 1.5 Flash' evaluating Accuracy...\n")
        accuracy_rating, accuracy_explanation = evaluate_response_with_gemini(response, prompt, "Accuracy")
        time.sleep(sleep_time_api)

        print("🔄 'Gemini 1.5 Flash' evaluating Clarity...\n")
        clarity_rating, clarity_explanation = evaluate_response_with_gemini(response, prompt, "Clarity")
        time.sleep(sleep_time_api)
        
        print("🔄 'Gemini 1.5 Flash' evaluating Relevance...\n")
        relevance_rating, relevance_explanation = evaluate_response_with_gemini(response, prompt, "Relevance")
        time.sleep(sleep_time_api)
        
        print("🔄 'Gemini 1.5 Flash' evaluating Adherence...\n")
        adherence_rating, adherence_explanation = evaluate_response_with_gemini(response, prompt, "Adherence")
        time.sleep(sleep_time_api)
        
        print("🔄 'Gemini 1.5 Flash' evaluating Insight...\n")
        insight_rating, insight_explanation = evaluate_response_with_gemini(response, prompt, "Insight")
        time.sleep(sleep_time_api)
        
        # Get the benchmark response for variance evaluation
        benchmark_response = row.get('Benchmark_Response-Import', None)  # Safely get benchmark response, defaults to None if missing
        print("🔄 Checking for Benchmark response...\n")
        if benchmark_response:
            # If benchmark response exists, proceed with variance evaluation
            print("🔄 'Gemini 1.5 Flash' evaluating Variance...\n")
            time.sleep(sleep_time_api)
            variance_rating, variance_explanation = evaluate_response_with_gemini(response, prompt, "Variance", benchmark_response)
        else:
            # If no benchmark response, set default values
            variance_rating, variance_explanation = None, "No benchmark response provided."
        
        # Update the DataFrame with the evaluation results
        df.at[index, 'Gemini_Accuracy_Rating'] = accuracy_rating
        df.at[index, 'Gemini_Accuracy_Explain'] = accuracy_explanation
        df.at[index, 'Gemini_Clarity_Rating'] = clarity_rating
        df.at[index, 'Gemini_Clarity_Explain'] = clarity_explanation
        df.at[index, 'Gemini_Relevance_Rating'] = relevance_rating
        df.at[index, 'Gemini_Relevance_Explain'] = relevance_explanation
        df.at[index, 'Gemini_Adherence_Rating'] = adherence_rating
        df.at[index, 'Gemini_Adherence_Explain'] = adherence_explanation
        df.at[index, 'Gemini_Insight_Rating'] = insight_rating
        df.at[index, 'Gemini_Insight_Explain'] = insight_explanation
        df.at[index, 'Gemini_Variance_Rating'] = variance_rating
        df.at[index, 'Gemini_Variance_Explain'] = variance_explanation

    # Save the updated DataFrame back to the Excel file
    print("🔄 Updating Excel file...\n")
    df.to_excel(output_file, index=False)

# Main processing function to run analyses
def process_selected_analysis_modes(input_file_path, output_file_path, selected_modes, sheet_name="Model_Responses", last_row=62):
    # Check if the output file already exists
    if os.path.exists(output_file_path):
        # Load the existing rated file
        df = pd.read_excel(output_file_path, sheet_name=sheet_name, engine='openpyxl')
        print(f"🔄 Existing rated file {output_file_path} loaded.")
    else:
        # Load the original file if the rated one doesn't exist yet
        df = pd.read_excel(input_file_path, sheet_name=sheet_name, engine='openpyxl')
        print(f"🔄 No rated file found, loading original file {input_file_path}.")

    # Ensure the dataframe is truncated at the last row of interest
    df = df.iloc[:last_row]
    
    print("🔄 Initiating selected analyses of the messages...\n")
    
    # Run the selected analysis modes
    if "Count Sentences" in selected_modes:
        print("🔄 Counting sentences...\n")
        process_sentence_count(df)
        print("✅ Done!\n")

    if "Count Tokens" in selected_modes:
        print("🔄 Counting tokens...\n")
        process_token_count(df)
        print("✅ Done!\n")

    if "Count Characters" in selected_modes:
        print("🔄 Counting characters...\n")
        process_char_count(df)
        print("✅ Done!\n")

    if "Count Words" in selected_modes:
        print("🔄 Counting words...\n")
        process_word_count(df)
        print("✅ Done!\n")

    if "Extract Named Entities" in selected_modes:
        print("🔄 Extracting named entities...\n")
        process_named_entities(df, input_file_path, sheet_name)
        print("✅ Done!\n")

    if "Cosine Similarity Analysis with Lemmatization" in selected_modes:
        print("🔄 Running Cosine similarity analysis with lemmatization...\n")
        process_cosine_similarity_with_lemmatization(df, input_file_path, sheet_name)
        print("✅ Done!\n")

    if "Sentiment Polarity Analysis" in selected_modes:
        print("🔄 Running Sentiment Polarity analysis...\n")
        process_polarity_sentiment(df)
        print("✅ Done!\n")

    if "Sentiment Subjectivity Analysis" in selected_modes:
        print("🔄 Running Sentiment Subjectivity analysis...\n")
        process_subjective_sentiment(df)
        print("✅ Done!\n")

    if "Flagged Words and Phrases Analysis" in selected_modes:
        print("🔄 Checking for flagged words...\n")
        process_flagged_words(df)
        print("✅ Done!\n")

    if "Spelling Error Check" in selected_modes:
        print("🔄 Checking for spelling errors...\n")
        process_spelling(df, input_file_path, sheet_name)
        print("✅ Done!\n")

    if "BERTScore Analysis" in selected_modes:
        print("🔄 Analyzing BERTScore...\n")
        process_bertscore(df, input_file_path, sheet_name)
        print("✅ Done!\n")

    if "Token Matching Analysis" in selected_modes:
        print("🔄 Running Token Matching analysis...\n")
        process_token_matching_with_lemmatization(df)
        print("✅ Done!\n")

    if "Semantic Similarity Analysis" in selected_modes:
        print("🔄 Running Semantic similarity analysis...\n")
        process_semantic_similarity(df, input_file_path, sheet_name)
        print("✅ Done!\n")

    if "Noun-Phrase Extraction" in selected_modes:
        print("🔄 Running Noun-Phrase extraction...\n")
        process_noun_phrases(df)
        print("✅ Done!\n")

    if "Gemini 1.5 Flash - AI Evaluation (6 aspects)" in selected_modes:
        print("🔄 Running 'Gemini 1.5 Flash' evaluations...\n")
        process_gemini_evaluations(df, output_file_path)
        print("✅ Done!\n")

    # Save the modified dataframe back to the rated Excel file
    print(f"🔄 Saving to {output_file_path}...\n")
    df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
    print(f"✅ File saved as {output_file_path}\n")
