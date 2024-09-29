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
            
            # Calculate BERTScore for ChatGPT benchmark
            if 'Benchmark_ChatGPT' in row and pd.notna(row['Benchmark_ChatGPT']):
                P, R, F1 = calculate_bertscore(row['Msg_Content'], row['Benchmark_ChatGPT'])
                precision_scores.append(P)
                recall_scores.append(R)
                f1_scores.append(F1)
                print(f"Row {index+1}: BERTScore (ChatGPT) - Precision: {P}, Recall: {R}, F1: {F1}")
            
            # Calculate BERTScore for Claude benchmark
            if 'Benchmark_Claude' in row and pd.notna(row['Benchmark_Claude']):
                P, R, F1 = calculate_bertscore(row['Msg_Content'], row['Benchmark_Claude'])
                precision_scores.append(P)
                recall_scores.append(R)
                f1_scores.append(F1)
                print(f"Row {index+1}: BERTScore (Claude) - Precision: {P}, Recall: {R}, F1: {F1}")

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
                similarities.append(similarity)
                print(f"Row {index+1}: Cosine Similarity (ChatGPT) after Lemmatization: {similarity}")
            
            # Cosine Similarity for Claude benchmark (with lemmatization)
            if 'Benchmark_Claude' in row and pd.notna(row['Benchmark_Claude']):
                lemmatized_msg_content = lemmatize_text(row['Msg_Content'])
                lemmatized_benchmark_response = lemmatize_text(row['Benchmark_Claude'])
                similarity = compute_cosine_similarity(lemmatized_msg_content, lemmatized_benchmark_response)
                similarities.append(similarity)
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
                match_score = token_level_matching(lemmatized_msg_content, lemmatized_benchmark_response)
                token_matches.append(match_score)
                print(f"Row {index+1}: Token Matching (ChatGPT) after Lemmatization: {match_score}")
            
            # Token Matching for Claude benchmark (with lemmatization)
            if 'Benchmark_Claude' in row and pd.notna(row['Benchmark_Claude']):
                lemmatized_msg_content = lemmatize_text(row['Msg_Content'])
                lemmatized_benchmark_response = lemmatize_text(row['Benchmark_Claude'])
                match_score = token_level_matching(lemmatized_msg_content, lemmatized_benchmark_response)
                token_matches.append(match_score)
                print(f"Row {index+1}: Token Matching (Claude) after Lemmatization: {match_score}")

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
                similarity = compute_semantic_similarity(row['Msg_Content'], row['Benchmark_ChatGPT'], tokenizer, model)
                similarities.append(similarity)
                print(f"Row {index+1}: Semantic Similarity (ChatGPT): {similarity}")
            
            # Semantic Similarity for Claude benchmark
            if 'Benchmark_Claude' in row and pd.notna(row['Benchmark_Claude']):
                similarity = compute_semantic_similarity(row['Msg_Content'], row['Benchmark_Claude'], tokenizer, model)
                similarities.append(similarity)
                print(f"Row {index+1}: Semantic Similarity (Claude): {similarity}")

            # Average Semantic Similarity
            if similarities:
                df.at[index, 'Semantic_Similarity'] = sum(similarities) / len(similarities)

    # Save results back to Excel
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
        
        # Get the benchmark responses for variance evaluation
        benchmark_response_chatgpt = row.get('Benchmark_ChatGPT', None)  # Safely get ChatGPT benchmark response
        benchmark_response_claude = row.get('Benchmark_Claude', None)  # Safely get Claude benchmark response

        print("🔄 Checking for Benchmark responses...\n")

        # Check if both benchmarks are missing
        if not benchmark_response_chatgpt and not benchmark_response_claude:
            print("No benchmark responses provided, skipping variance evaluation.\n")
            variance_chatgpt_rating, variance_chatgpt_explanation = None, "No benchmark response provided."
            variance_claude_rating, variance_claude_explanation = None, "No benchmark response provided."
        else:
            # Prepare placeholders for missing benchmarks
            if not benchmark_response_chatgpt:
                benchmark_response_chatgpt = "Message B not available"
                print("ChatGPT benchmark missing, inserting placeholder...\n")
            if not benchmark_response_claude:
                benchmark_response_claude = "Message C not available"
                print("Claude benchmark missing, inserting placeholder...\n")
            
            # Run the combined variance evaluation prompt
            print("🔄 'Gemini 1.5 Flash' evaluating Variance against both benchmarks...\n")
            time.sleep(sleep_time_api)
            
            # Pass both benchmark responses to the evaluator
            variance_chatgpt_rating, variance_chatgpt_explanation = evaluate_response_with_gemini(response, prompt, "Variance", benchmark_response_chatgpt)
            variance_claude_rating, variance_claude_explanation = evaluate_response_with_gemini(response, prompt, "Variance", benchmark_response_claude)

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
        df.at[index, 'Gemini_Variance_ChatGPT'] = variance_chatgpt_rating
        df.at[index, 'Gemini_Variance_ChatGPT_Explain'] = variance_chatgpt_explanation
        df.at[index, 'Gemini_Variance_Claude'] = variance_claude_rating
        df.at[index, 'Gemini_Variance_Claude_Explain'] = variance_claude_explanation

        # Save the updated DataFrame back to the Excel file
        print("🔄 Updating Excel file...\n")
        df.to_excel(output_file, index=False)

def process_mistral_evaluations(df, output_file):
    """
    Process the DataFrame, evaluate each response with Mistral, 
    and store the results in the new columns for each evaluation aspect.
    """
    # Loop through each row (response) in the DataFrame
    for index, row in df.iterrows():
        prompt = row['Prompt_Text']  # Assuming this is the column name for prompts
        response = row['Msg_Content']  # Assuming this is the column name for responses
        
        # Perform evaluations for each aspect
        print("🔄 'Mistral-Large' evaluating Accuracy...\n")
        accuracy_rating, accuracy_explanation = evaluate_response_with_mistral(response, prompt, "Accuracy")
        time.sleep(sleep_time_api)

        print("🔄 'Mistral-Large' evaluating Clarity...\n")
        clarity_rating, clarity_explanation = evaluate_response_with_mistral(response, prompt, "Clarity")
        time.sleep(sleep_time_api)
        
        print("🔄 'Mistral-Large' evaluating Relevance...\n")
        relevance_rating, relevance_explanation = evaluate_response_with_mistral(response, prompt, "Relevance")
        time.sleep(sleep_time_api)
        
        print("🔄 'Mistral-Large' evaluating Adherence...\n")
        adherence_rating, adherence_explanation = evaluate_response_with_mistral(response, prompt, "Adherence")
        time.sleep(sleep_time_api)
        
        print("🔄 'Mistral-Large' evaluating Insight...\n")
        insight_rating, insight_explanation = evaluate_response_with_mistral(response, prompt, "Insight")
        time.sleep(sleep_time_api)
        
        # Get the benchmark responses for variance evaluation
        benchmark_response_chatgpt = row.get('Benchmark_ChatGPT', None)  # Safely get ChatGPT benchmark response
        benchmark_response_claude = row.get('Benchmark_Claude', None)  # Safely get Claude benchmark response

        print("🔄 Checking for Benchmark responses...\n")

        # Check if both benchmarks are missing
        if not benchmark_response_chatgpt and not benchmark_response_claude:
            print("No benchmark responses provided, skipping variance evaluation.\n")
            variance_chatgpt_rating, variance_chatgpt_explanation = None, "No benchmark response provided."
            variance_claude_rating, variance_claude_explanation = None, "No benchmark response provided."
        else:
            # Prepare placeholders for missing benchmarks
            if not benchmark_response_chatgpt:
                benchmark_response_chatgpt = "Message B not available"
                print("ChatGPT benchmark missing, inserting placeholder...\n")
            if not benchmark_response_claude:
                benchmark_response_claude = "Message C not available"
                print("Claude benchmark missing, inserting placeholder...\n")
            
            # Run the combined variance evaluation prompt
            print("🔄 'Mistral-Large' evaluating Variance against both benchmarks...\n")
            time.sleep(sleep_time_api)
            
            # Pass both benchmark responses to the evaluator
            variance_chatgpt_rating, variance_chatgpt_explanation = evaluate_response_with_mistral(response, prompt, "Variance", benchmark_response_chatgpt)
            variance_claude_rating, variance_claude_explanation = evaluate_response_with_mistral(response, prompt, "Variance", benchmark_response_claude)

        # Update the DataFrame with the evaluation results
        df.at[index, 'Mistral_Accuracy_Rating'] = accuracy_rating
        df.at[index, 'Mistral_Accuracy_Explain'] = accuracy_explanation
        df.at[index, 'Mistral_Clarity_Rating'] = clarity_rating
        df.at[index, 'Mistral_Clarity_Explain'] = clarity_explanation
        df.at[index, 'Mistral_Relevance_Rating'] = relevance_rating
        df.at[index, 'Mistral_Relevance_Explain'] = relevance_explanation
        df.at[index, 'Mistral_Adherence_Rating'] = adherence_rating
        df.at[index, 'Mistral_Adherence_Explain'] = adherence_explanation
        df.at[index, 'Mistral_Insight_Rating'] = insight_rating
        df.at[index, 'Mistral_Insight_Explain'] = insight_explanation
        df.at[index, 'Mistral_Variance_ChatGPT'] = variance_chatgpt_rating
        df.at[index, 'Mistral_Variance_ChatGPT_Explain'] = variance_chatgpt_explanation
        df.at[index, 'Mistral_Variance_Claude'] = variance_claude_rating
        df.at[index, 'Mistral_Variance_Claude_Explain'] = variance_claude_explanation

    # Save the updated DataFrame back to the Excel file
    print("🔄 Updating Excel file...\n")
    df.to_excel(output_file, index=False)

# Main processing function to run analyses
def process_selected_analysis_modes(input_file_path, output_file_path, selected_mode, sheet_name="Model_Responses", last_row=62):
    """
    Process selected analysis modes: 'Compute Evaluations (All)', 'Gemini Evaluations (6 Aspects)', 'Mistral Evaluations (6 Aspects)'.
    """

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
    
    print("🔄 Initiating selected analysis...\n")
    
    # Handle the 'Compute Evaluations (All)' option
    if selected_mode == "Compute Evaluations (All)":
        print("🔄 Running all evaluations...\n")
        process_sentence_count(df)
        process_token_count(df)
        process_char_count(df)
        process_word_count(df)
        process_named_entities(df, input_file_path, sheet_name)
        process_cosine_similarity_with_lemmatization(df, input_file_path, sheet_name)
        process_polarity_sentiment(df)
        process_subjective_sentiment(df)
        process_flagged_words(df)
        process_spelling(df, input_file_path, sheet_name)
        process_bertscore(df, input_file_path, sheet_name)
        process_token_matching_with_lemmatization(df)
        process_semantic_similarity(df, input_file_path, sheet_name)
        process_noun_phrases(df)
        print("✅ Compute-level Evaluations Completed!\n")

    # Handle the 'Gemini Evaluations (6 Aspects)' option
    elif selected_mode == "Gemini Evaluations (6 Aspects)":
        print("🔄 Running 'Gemini 1.5 Flash' evaluations...\n")
        process_gemini_evaluations(df, output_file_path)
        print("✅ Gemini AI Evaluations Completed!\n")

    # Handle the 'Mistral Evaluations (6 Aspects)' option
    elif selected_mode == "Mistral Evaluations (6 Aspects)":
        print("🔄 Running 'Mistral-Large' evaluations...\n")
        process_mistral_evaluations(df, output_file_path)
        print("✅ Mistral AI Evaluations Completed!\n")

    # Save the modified dataframe back to the rated Excel file
    print(f"🔄 Saving to {output_file_path}...\n")
    df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
    print(f"✅ File saved as {output_file_path}\n")