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
from utils import *

# Suppress all warnings
warnings.filterwarnings("ignore")

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
def process_noun_phrases(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['Noun_Phrases']):
            noun_phrases = extract_noun_phrases(row['Msg_Content'])
            df.at[index, 'Noun_Phrases'] = ', '.join(noun_phrases)
            print(f"Row {index+1}: Noun Phrases: {noun_phrases}")
    # Save results
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

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
        
        # Debug print statement to ensure this part is processed
        print(f"Row {index+1}: Completed Spelling Error Check.")

    # Save results after spelling
    df.to_excel(file_path, sheet_name=sheet_name, index=False)
    print("🔄 Spelling Check Completed. Moving to next analysis...\n")

def process_flagged_words(df, file_path, sheet_name):
    for index, row in df.iterrows():
        # If the flagged words haven't been processed yet, proceed
        if pd.isna(row['Flagged_Words']):
            flagged_word_counts = check_word_frequency(row['Msg_Content'])
            
            # Create a string that summarizes the flagged words and their counts
            flagged_summary = ', '.join([f"{word}: {count}" for word, count in flagged_word_counts.items() if count > 0])
            
            # If no flagged words, set 'None'
            if not flagged_summary:
                flagged_summary = 'None'
            
            # Store the summary in the "Flagged_Words" column
            df.at[index, 'Flagged_Words'] = flagged_summary
        else:
            # If Flagged_Words already exist in the DataFrame, use that
            flagged_summary = df.at[index, 'Flagged_Words']

        # Print flagged words and phrases
        print(f"Row {index+1}: Flagged Words & Phrases: {flagged_summary}")
        
        # Calculate the total number of flagged words/phrases
        if flagged_summary != 'None':
            flagged_penalty = calculate_total_flagged_words(flagged_summary)
        else:
            flagged_penalty = 0

        # Store the total count in the "Flagged_Penalty" column
        df.at[index, 'Flagged_Penalty'] = flagged_penalty
        print(f"Row {index+1}: Flagged Penalty: {flagged_penalty}")
    
    # Save changes back to Excel (this can be done later when all analyses are completed)
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

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
            precision_scores = []
            recall_scores = []
            f1_scores = []

            response = row['Msg_Content']

            # Skip non-standard texts (ASCII art, emojis, etc.)
            if is_non_standard_text(response):
                print(f"Row {index+1}: Skipping BERTScore for non-standard text.")
                df.at[index, 'BERT_Precision'] = 0  # Assign default value or 'N/A'
                df.at[index, 'BERT_Recall'] = 0
                df.at[index, 'BERT_F1'] = 0
                continue

            # Calculate BERTScore for ChatGPT benchmark
            if 'Benchmark_ChatGPT' in row and pd.notna(row['Benchmark_ChatGPT']):
                try:
                    P, R, F1 = calculate_bertscore(response, row['Benchmark_ChatGPT'])
                    precision_scores.append(P)
                    recall_scores.append(R)
                    f1_scores.append(F1)
                    print(f"Row {index+1}: BERTScore (ChatGPT) - Precision: {P}, Recall: {R}, F1: {F1}")
                except Exception as e:
                    print(f"Error processing BERTScore for ChatGPT: {e}")
                    df.at[index, 'BERT_Precision'] = 'N/A'
                    df.at[index, 'BERT_Recall'] = 'N/A'
                    df.at[index, 'BERT_F1'] = 'N/A'

            # Calculate BERTScore for Claude benchmark
            if 'Benchmark_Claude' in row and pd.notna(row['Benchmark_Claude']):
                try:
                    P, R, F1 = calculate_bertscore(response, row['Benchmark_Claude'])
                    precision_scores.append(P)
                    recall_scores.append(R)
                    f1_scores.append(F1)
                    print(f"Row {index+1}: BERTScore (Claude) - Precision: {P}, Recall: {R}, F1: {F1}")
                except Exception as e:
                    print(f"Error processing BERTScore for Claude: {e}")
                    df.at[index, 'BERT_Precision'] = 'N/A'
                    df.at[index, 'BERT_Recall'] = 'N/A'
                    df.at[index, 'BERT_F1'] = 'N/A'

            # If both exist, average the results
            if precision_scores and recall_scores and f1_scores:
                df.at[index, 'BERT_Precision'] = sum(precision_scores) / len(precision_scores)
                df.at[index, 'BERT_Recall'] = sum(recall_scores) / len(recall_scores)
                df.at[index, 'BERT_F1'] = sum(f1_scores) / len(f1_scores)

    # Save results
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

# Function to calculate and update Cosine Similarity
def process_cosine_similarity_with_lemmatization(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['Cosine_Similarity']):
            similarities = []
            
            # Cosine Similarity for ChatGPT benchmark (with lemmatization)
            if 'Benchmark_ChatGPT' in row and pd.notna(row['Benchmark_ChatGPT']):
                lemmatized_msg_content = lemmatize_text(row['Msg_Content'])
                lemmatized_benchmark_response = lemmatize_text(row['Benchmark_ChatGPT'])
                similarity = compute_cosine_similarity(lemmatized_msg_content, lemmatized_benchmark_response)
                # Append 0 if similarity is None (error case)
                similarities.append(similarity if similarity is not None else 0)
                print(f"Row {index+1}: Cosine Similarity (ChatGPT) after Lemmatization: {similarity}")
            
            # Cosine Similarity for Claude benchmark (with lemmatization)
            if 'Benchmark_Claude' in row and pd.notna(row['Benchmark_Claude']):
                lemmatized_msg_content = lemmatize_text(row['Msg_Content'])
                lemmatized_benchmark_response = lemmatize_text(row['Benchmark_Claude'])
                similarity = compute_cosine_similarity(lemmatized_msg_content, lemmatized_benchmark_response)
                # Append 0 if similarity is None (error case)
                similarities.append(similarity if similarity is not None else 0)
                print(f"Row {index+1}: Cosine Similarity (Claude) after Lemmatization: {similarity}")

            # Average Cosine Similarity
            if similarities:
                df.at[index, 'Cosine_Similarity'] = sum(similarities) / len(similarities)

    # Save results back to Excel
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

# Function to calculate and update Token Matching Similarity
def process_token_matching_with_lemmatization(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['Token_Matching']):
            token_matches = []
            
            # Token Matching for ChatGPT benchmark (with lemmatization)
            if 'Benchmark_ChatGPT' in row and pd.notna(row['Benchmark_ChatGPT']):
                lemmatized_msg_content = lemmatize_text(row['Msg_Content'])
                lemmatized_benchmark_response = lemmatize_text(row['Benchmark_ChatGPT'])
                
                try:
                    match_score = token_level_matching(lemmatized_msg_content, lemmatized_benchmark_response)
                    token_matches.append(match_score)
                    print(f"Row {index+1}: Token Matching (ChatGPT) after Lemmatization: {match_score}")
                except ValueError as e:
                    print(f"Error processing Token Matching (ChatGPT) for Row {index+1}: {e}")
                    token_matches.append(0)  # Assign a score of 0 if there's an error
            
            # Token Matching for Claude benchmark (with lemmatization)
            if 'Benchmark_Claude' in row and pd.notna(row['Benchmark_Claude']):
                lemmatized_msg_content = lemmatize_text(row['Msg_Content'])
                lemmatized_benchmark_response = lemmatize_text(row['Benchmark_Claude'])
                
                try:
                    match_score = token_level_matching(lemmatized_msg_content, lemmatized_benchmark_response)
                    token_matches.append(match_score)
                    print(f"Row {index+1}: Token Matching (Claude) after Lemmatization: {match_score}")
                except ValueError as e:
                    print(f"Error processing Token Matching (Claude) for Row {index+1}: {e}")
                    token_matches.append(0)  # Assign a score of 0 if there's an error

            # Average Token Matching
            if token_matches:
                df.at[index, 'Token_Matching'] = sum(token_matches) / len(token_matches)

    # Save results back to Excel
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

def process_semantic_similarity(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['Semantic_Similarity']):
            similarities = []
            
            # Semantic Similarity for ChatGPT benchmark
            if 'Benchmark_ChatGPT' in row and pd.notna(row['Benchmark_ChatGPT']):
                try:
                    similarity = compute_semantic_similarity(row['Msg_Content'], row['Benchmark_ChatGPT'], tokenizer, model)
                    similarities.append(similarity)
                    print(f"Row {index+1}: Semantic Similarity (ChatGPT): {similarity}")
                except Exception as e:
                    print(f"Error processing Semantic Similarity (ChatGPT) for Row {index+1}: {e}")
                    similarities.append(0)  # Assign a default score of 0 in case of an error
            
            # Semantic Similarity for Claude benchmark
            if 'Benchmark_Claude' in row and pd.notna(row['Benchmark_Claude']):
                try:
                    similarity = compute_semantic_similarity(row['Msg_Content'], row['Benchmark_Claude'], tokenizer, model)
                    similarities.append(similarity)
                    print(f"Row {index+1}: Semantic Similarity (Claude): {similarity}")
                except Exception as e:
                    print(f"Error processing Semantic Similarity (Claude) for Row {index+1}: {e}")
                    similarities.append(0)  # Assign a default score of 0 in case of an error

            # Average Semantic Similarity
            if similarities:
                df.at[index, 'Semantic_Similarity'] = sum(similarities) / len(similarities)

    # Save results back to Excel
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

def process_model_evaluations(df, output_file, model_name, eval_function, current_mode):
    """
    Generalized function to process Gemini or Mistral evaluations.
    """
    # Loop through each row (response) in the DataFrame
    for index, row in df.iterrows():
        prompt = row['Prompt_Text']
        response = row['Msg_Content']

        # Set the current mode to "Normal" for evaluations
        current_mode = "Normal"

        # Perform evaluations for each aspect
        eval_aspects = ["Accuracy", "Clarity", "Relevance", "Adherence", "Insight"]
        
        for aspect in eval_aspects:
            # Check if the aspect has already been evaluated
            if pd.isna(row.get(f'{model_name}_{aspect}_Rating')):
                try:
                    print(f"🔄 '{model_name}' evaluating {aspect}...\n")
                    rating, explanation = eval_function(response, prompt, aspect, model_name, current_mode)
                    
                    # Use specific sleep times for each model
                    if model_name == "mistral-large":
                        time.sleep(sleep_time_mistral)
                    elif model_name == "gemini-1.5-flash":
                        time.sleep(sleep_time_api)  # Default API sleep time for Gemini
                    else:
                        time.sleep(sleep_time_api)  # Fallback in case new models are added

                    # Dynamically set the column names based on the model (Gemini or Mistral)
                    df.at[index, f'{model_name}_{aspect}_Rating'] = rating
                    df.at[index, f'{model_name}_{aspect}_Explain'] = explanation
                except Exception as e:
                    print(f"❌ No valid {aspect} response generated for {model_name}. Error: {str(e)}")
                    df.at[index, f'{model_name}_{aspect}_Rating'] = "N/A"
                    df.at[index, f'{model_name}_{aspect}_Explain'] = "No valid response generated."
            else:
                print(f"🔄 Skipping {aspect} for {model_name}, already evaluated.\n")

        # Handle the Variance evaluation separately
        try:
            benchmark_response_chatgpt = row.get('Benchmark_ChatGPT', None)
            '''benchmark_response_claude = row.get('Benchmark_Claude', None)'''

            print(f"🔄 Checking for Benchmark responses for '{model_name}'...\n")

            # Skip variance evaluation if both benchmarks are missing
            #if not benchmark_response_chatgpt and not benchmark_response_claude:
            if not benchmark_response_chatgpt:
                print("No benchmark response available, skipping variance evaluation.\n")
                variance_chatgpt_rating, variance_chatgpt_explanation = "N/A", "No benchmark response provided."
                '''variance_claude_rating, variance_claude_explanation = "N/A", "No benchmark response provided."'''
            else:
                # Check if variance has already been evaluated
                #if pd.isna(row.get(f'{model_name}_Variance_ChatGPT')) or pd.isna(row.get(f'{model_name}_Variance_Claude')):
                if pd.isna(row.get(f'{model_name}_Variance_ChatGPT')):
                    print(f"🔄 '{model_name}' evaluating Variance against benchmark...\n")
                    variance_chatgpt_rating, variance_chatgpt_explanation = eval_function(
                        response, prompt, "Variance", model_name, benchmark_response_chatgpt
                    )
                    '''variance_chatgpt_rating, variance_chatgpt_explanation, variance_claude_rating, variance_claude_explanation = eval_function(
                        response, prompt, "Variance", model_name, benchmark_response_chatgpt, benchmark_response_claude
                    )'''

                    # Update the DataFrame with the variance results
                    df.at[index, f'{model_name}_Variance_ChatGPT'] = variance_chatgpt_rating
                    df.at[index, f'{model_name}_Variance_ChatGPT_Explain'] = variance_chatgpt_explanation
                    '''df.at[index, f'{model_name}_Variance_Claude'] = variance_claude_rating
                    df.at[index, f'{model_name}_Variance_Claude_Explain'] = variance_claude_explanation'''
                else:
                    print(f"🔄 Skipping Variance for {model_name}, already evaluated.\n")
        except Exception as e:
            print(f"❌ No valid Variance response generated for {model_name}. Error: {str(e)}")
            df.at[index, f'{model_name}_Variance_ChatGPT'] = "N/A"
            df.at[index, f'{model_name}_Variance_ChatGPT_Explain'] = "No valid response generated."
            '''df.at[index, f'{model_name}_Variance_Claude'] = "N/A"
            df.at[index, f'{model_name}_Variance_Claude_Explain'] = "No valid response generated."'''
        # Save the updated DataFrame back to the Excel file after every row, for safety
        print("🔄 Updating Excel file...\n")
        df.to_excel(output_file, index=False)

# Main processing function to run analyses
def process_selected_analysis_modes(input_file_path, output_file_path, selected_mode, sheet_name="Model_Responses", last_row=1500):
    """
    Process selected analysis modes: handle 'Compute Evaluations (All)', 'Gemini Evaluations (6 Aspects)', 'Mistral Evaluations (6 Aspects)', and 'Merge Excel Evaluation Results'.
    """
    print(f"Selected mode: '{selected_mode}'\n")
    
    # For merging, no need to load or work with a DataFrame
    if selected_mode == "Merge Excel Evaluation Results":
        print("🔄 Merging the 3 evaluation results...\n")
        merge_evaluations()  # Call the merge function directly
        print("✅ Excel Results Merge Completed!\n")
        return  # No need to save anything, just exit after merging

    # Always load from the input file (remove the check for existing output file)
    df = pd.read_excel(input_file_path, sheet_name=sheet_name, engine='openpyxl')
    print(f"🔄 Loaded file {input_file_path}.")

    # Ensure the dataframe is truncated at the last row of interest
    df = df.iloc[:last_row]

    print("🔄 Initiating analysis...\n")

    # Handle the 'Compute Evaluations (All)' option
    if selected_mode == "Compute Evaluations (All)":
        print("🔄 Running all evaluations...\n")
        
        # Add debug print statements between each analysis
        process_sentence_count(df)
        print("🔄 Completed Sentence Count...\n")
        
        process_token_count(df)
        print("🔄 Completed Token Count...\n")
        
        process_char_count(df)
        print("🔄 Completed Character Count...\n")
        
        process_word_count(df)
        print("🔄 Completed Word Count...\n")
        
        process_named_entities(df, input_file_path, sheet_name)
        print("🔄 Completed Named Entities...\n")
        
        process_cosine_similarity_with_lemmatization(df, input_file_path, sheet_name)
        print("🔄 Completed Cosine Similarity...\n")
        
        process_polarity_sentiment(df)
        print("🔄 Completed Sentiment Polarity...\n")
        
        process_subjective_sentiment(df)
        print("🔄 Completed Sentiment Subjectivity...\n")
        
        process_flagged_words(df, input_file_path, sheet_name)
        print("🔄 Completed Flagged Words...\n")
        
        process_spelling(df, input_file_path, sheet_name)
        print("🔄 Completed Spelling Errors...\n")
        
        process_bertscore(df, input_file_path, sheet_name)
        print("🔄 Completed BERTScore...\n")
        
        process_token_matching_with_lemmatization(df, input_file_path, sheet_name)
        print("🔄 Completed Token Matching...\n")
        
        process_semantic_similarity(df, input_file_path, sheet_name)
        print("🔄 Completed Semantic Similarity...\n")
        
        process_noun_phrases(df, input_file_path, sheet_name)
        print("🔄 Completed Noun Phrases...\n")

        print("✅ Compute-level Evaluations Completed!\n")

    # Handle the 'Gemini Evaluations (6 Aspects)' option
    elif selected_mode == "Gemini Evaluations (6 Aspects)":
        print("🔄 Running 'Gemini 1.5 Flash' evaluations...\n")
        current_mode = "Normal"
        process_model_evaluations(df, output_file_path, "gemini-1.5-flash", evaluate_response_with_model, current_mode)
        print("✅ Gemini AI Evaluations Completed!\n")

    # Handle the 'Mistral Evaluations (6 Aspects)' option
    elif selected_mode == "Mistral Evaluations (6 Aspects)":
        print("🔄 Running 'Mistral-Large' evaluations...\n")
        current_mode = "Normal"
        process_model_evaluations(df, output_file_path, "mistral-large", evaluate_response_with_model, current_mode)
        print("✅ Mistral AI Evaluations Completed!\n")

    # Save the modified dataframe back to the rated Excel file
    print(f"🔄 Saving to {output_file_path}...\n")
    df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
    print(f"✅ File saved as {output_file_path}\n")