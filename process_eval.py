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
from process_text import *
import nltk
import os
import warnings
import time
import subprocess
from config import sleep_time_api, row_save_frequency
from utils import *

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load Hunspell dictionaries once
hunspell_obj = load_hunspell_dictionaries()

# Function to calculate and update Sentence Count
def process_sentence_count(df):
    for index, row in df.iterrows():
        if pd.isna(row['response_total_sentences']):
            # Ensure the 'response_msg_content' is valid before counting sentences
            sentence_count = count_sentences(row['response_msg_content'])
            df.at[index, 'response_total_sentences'] = sentence_count
            print(f"Row {index+1}: Sentence Count: {sentence_count}")

# Function to calculate and update Token Count
def process_token_count(df):
    for index, row in df.iterrows():
        if pd.isna(row['response_total_tokens']):
            token_count = count_tokens(row.get('response_msg_content', ''))
            df.at[index, 'response_total_tokens'] = token_count
            print(f"Row {index+1}: Token Count: {token_count}")

# Function to calculate and update Character Count
def process_char_count(df):
    for index, row in df.iterrows():
        if pd.isna(row['response_total_chars']):
            char_count = count_chars(row.get('response_msg_content', ''))
            df.at[index, 'response_total_chars'] = char_count
            print(f"Row {index+1}: Character Count: {char_count}")

# Function to calculate and update Word Count
def process_word_count(df):
    for index, row in df.iterrows():
        if pd.isna(row['response_total_words']):
            word_count = count_words(row.get('response_msg_content', ''))
            df.at[index, 'response_total_words'] = word_count
            print(f"Row {index+1}: Word Count: {word_count}")

# Function to calculate and update list of Named Entities
def process_named_entities(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['eval_named_entities']):
            entities = extract_named_entities(row['response_msg_content'])
            df.at[index, 'eval_named_entities'] = str(entities)
            print(f"Row {index+1}: Named Entities: {entities}")
    # Save results
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

# Function to calculate and update Polarity Sentiment
def process_polarity_sentiment(df):
    for index, row in df.iterrows():
        if pd.isna(row['eval_sentiment_polarity']):
            polarity = analyze_polarity(row['response_msg_content'])
            df.at[index, 'eval_sentiment_polarity'] = polarity
            print(f"Row {index+1}: Sentiment Polarity: {polarity}")

# Function to calculate and update Subjective Sentiment
def process_subjective_sentiment(df):
    for index, row in df.iterrows():
        if pd.isna(row['eval_sentiment_subjectivity']):
            subjectivity = analyze_subjectivity(row['response_msg_content'])
            df.at[index, 'eval_sentiment_subjectivity'] = subjectivity
            print(f"Row {index+1}: Sentiment Subjectivity: {subjectivity}")

# Function to extract and update Noun Phrases
def process_noun_phrases(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['eval_noun_phrases']):
            noun_phrases = extract_noun_phrases(row['response_msg_content'])
            df.at[index, 'eval_noun_phrases'] = ', '.join(noun_phrases)
            print(f"Row {index+1}: {len(noun_phrases)} Noun Phrases extracted.")
    # Save results
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

# def process_spelling(df, file_path, sheet_name, hunspell_obj): # NOT CURRENTLY USED - replaced with 'process_spelling_with_ai'
    """
    Check spelling and update the list of errors.
    """
    for index, row in df.iterrows():
        # Ensure response_msg_content is a valid string before processing
        if pd.notna(row['response_msg_content']) and isinstance(row['response_msg_content'], str):
            # Check if both spelling fields are already filled
            if pd.isna(row['eval_spelling_error_qty']):
                spelling_errors, misspelled_words = spellcheck_hunspell_getty(row['response_msg_content'], hunspell_obj)
                df.at[index, 'eval_spelling_error_qty'] = spelling_errors
                df.at[index, 'eval_spelling_errors'] = ', '.join(misspelled_words)
                print(f"Row {index+1}: Spelling Errors: {spelling_errors} - Misspelled Words: {misspelled_words}")
            else:
                print(f"Row {index+1}: Skipping Spelling Check - Already Evaluated.")
        else:
            print(f"Row {index+1}: Skipping Spelling Check - Invalid response_msg_content.")
    
    # Save results after spelling
    df.to_excel(file_path, sheet_name=sheet_name, index=False)
    print("🔄 Spelling Check Completed. Moving to next analysis...\n")

def process_spelling_with_ai(df, file_path, sheet_name, hunspell_obj, save_interval=row_save_frequency):
    """
    Process the spelling check, filter out words using AI for review, and update the Hunspell custom dictionary file automatically.
    """
    terms_added = False  # Track if new terms are added

    for index, row in df.iterrows():
        if pd.notna(row['response_msg_content']) and pd.isna(row['eval_spelling_errors']) and pd.isna(row['eval_spelling_error_qty']):
            # Perform the initial spelling check
            spelling_errors, misspelled_words = spellcheck_hunspell_getty(row['response_msg_content'], hunspell_obj)
            total_initial_errors = len(misspelled_words)
            print(f"🚩 Row {index+1}: Hunspell identified {total_initial_errors} spelling errors: {misspelled_words} - Sending to Getty for validation.")
            
            if misspelled_words:
                print(f"🚩 Row {index+1}: Spelling Errors remaining after Getty: {misspelled_words} - Sending to Gemini for validation.")

                # Gemini validation pass without context
                filtered_by_ai = filter_spelling_errors_with_ai(misspelled_words)

                # Final review step: Send each remaining flagged word along with the full context if needed
                final_spelling_errors = []
                for word in misspelled_words:
                    if word not in filtered_by_ai:
                        # Resend word with full context if it remains flagged
                        print(f"🚩 Row {index+1}: No match for '{word}'. Re-sending to Gemini with context.")
                        if not is_word_valid_in_context(word, row['response_msg_content']):
                            print(f"🚩 Row {index+1}: Word '{word}' confirmed as misspelled by Gemini, with context.")
                            final_spelling_errors.append(word)

                final_error_count = len(final_spelling_errors)

                # Add any newly AI-validated terms to the custom dictionary
                if filtered_by_ai:
                    filtered_by_ai_lowercase = [term.lower() for term in filtered_by_ai]
                    update_custom_dictionary(hunspell_obj, filtered_by_ai_lowercase)
                    terms_added = True
                    print(f"⬆️ Row {index+1}: Gemini validated, added to dictionary: {filtered_by_ai_lowercase} - Total New Words: {len(filtered_by_ai_lowercase)}")
                    hunspell_obj = load_hunspell_dictionaries()
                else:
                    print(f"⚖️  Row {index+1}: Gemini did not add any new words to the custom dictionary.")

                # Update the DataFrame with final errors after filtering
                df.at[index, 'eval_spelling_errors'] = ', '.join(final_spelling_errors) if final_spelling_errors else ''
                df.at[index, 'eval_spelling_error_qty'] = final_error_count

                # Conditional message based on whether any final spelling errors remain
                if final_error_count == 0:
                    print(f"✅ Row {index+1}: No spelling errors after final filtering.")
                else:
                    print(f"🚩 Row {index+1}: Final Spelling Errors after all validations: {final_spelling_errors} - Total: {final_error_count}")
            else:
                # No spelling errors detected
                df.at[index, 'eval_spelling_errors'] = ''
                df.at[index, 'eval_spelling_error_qty'] = 0
                print(f"✅ Row {index+1}: No spelling errors detected.")

        # Save progress every `save_interval` rows
        if (index + 1) % save_interval == 0:
            df.to_excel(file_path, sheet_name=sheet_name, index=False)
            print(f"💾 Progress saved after {index+1} rows.\n")

    # Final save after processing all rows
    df.to_excel(file_path, sheet_name=sheet_name, index=False)
    print("🔄 Spelling Check with AI Filter and Filter Term Update Completed. All data saved.\n")

def process_flagged_words(df, file_path, sheet_name):
    for index, row in df.iterrows():
        # If the flagged words haven't been processed yet, proceed
        if pd.isna(row['eval_flagged_penalty']):
            flagged_word_counts = check_word_frequency(row['response_msg_content'])
            
            # Create a string that summarizes the flagged words and their counts
            flagged_summary = ', '.join([f"{word}: {count}" for word, count in flagged_word_counts.items() if count > 0])
            
            # If no flagged words, set 'None'
            if not flagged_summary:
                flagged_summary = 'None'
            
            # Store the summary in the "Flagged_Words" column
            df.at[index, 'eval_flagged_words'] = flagged_summary
        else:
            # If Flagged_Words already exist in the DataFrame, use that
            flagged_summary = df.at[index, 'eval_flagged_words']

        # Print flagged words and phrases
        print(f"Row {index+1}: Flagged Words & Phrases: {flagged_summary}")
        
        # Calculate the total number of flagged words/phrases
        if flagged_summary != 'None':
            flagged_penalty = calculate_total_flagged_words(flagged_summary)
        else:
            flagged_penalty = 0

        # Store the total count in the "Flagged_Penalty" column
        df.at[index, 'eval_flagged_penalty'] = flagged_penalty
        print(f"Row {index+1}: Flagged Penalty: {flagged_penalty}")
    
    # Save changes back to Excel
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

def calculate_bertscore(text1, text2):
    # Calculate Precision, Recall, F1 using BERTScore
    P, R, F1 = bert_score([text1], [text2], lang="en", rescale_with_baseline=True)
    return P.mean().item(), R.mean().item(), F1.mean().item()

def process_bertscore(df, file_path, sheet_name):
    # Ensure the columns exist
    if 'eval_bert_precision' not in df.columns:
        df['eval_bert_precision'] = pd.NA
    if 'eval_bert_recall' not in df.columns:
        df['eval_bert_recall'] = pd.NA
    if 'eval_bert_f1' not in df.columns:
        df['eval_bert_f1'] = pd.NA

    for index, row in df.iterrows():
        if pd.isna(row['eval_bert_f1']):
            precision_scores = []
            recall_scores = []
            f1_scores = []

            response = row['response_msg_content']

            # Ensure response_msg_content is a valid string before processing
            if not isinstance(response, str):
                print(f"Row {index+1}: Skipping BERTScore - Invalid response_msg_content.")
                df.at[index, 'eval_bert_precision'] = 0  # Assign default value or 'N/A'
                df.at[index, 'eval_bert_recall'] = 0
                df.at[index, 'eval_bert_f1'] = 0
                continue

            # Skip non-standard texts (ASCII art, emojis, etc.)
            if is_non_standard_text(response):
                print(f"Row {index+1}: Skipping BERTScore for non-standard text.")
                df.at[index, 'eval_bert_precision'] = 0  # Assign default value or 'N/A'
                df.at[index, 'eval_bert_recall'] = 0
                df.at[index, 'eval_bert_f1'] = 0
                continue

            # Calculate BERTScore for ChatGPT benchmark
            if 'prompt_benchmark_chatgpt' in row and pd.notna(row['prompt_benchmark_chatgpt']):
                try:
                    P, R, F1 = calculate_bertscore(response, row['prompt_benchmark_chatgpt'])
                    precision_scores.append(P)
                    recall_scores.append(R)
                    f1_scores.append(F1)
                    print(f"Row {index+1}: BERTScore (ChatGPT) - Precision: {P}, Recall: {R}, F1: {F1}")
                except Exception as e:
                    print(f"Error processing BERTScore for ChatGPT: {e}")
                    df.at[index, 'eval_bert_precision'] = 'N/A'
                    df.at[index, 'eval_bert_recall'] = 'N/A'
                    df.at[index, 'eval_bert_f1'] = 'N/A'

            # Calculate BERTScore for Claude benchmark
            if 'prompt_benchmark_claude' in row and pd.notna(row['prompt_benchmark_claude']):
                try:
                    P, R, F1 = calculate_bertscore(response, row['prompt_benchmark_claude'])
                    precision_scores.append(P)
                    recall_scores.append(R)
                    f1_scores.append(F1)
                    print(f"Row {index+1}: BERTScore (Claude) - Precision: {P}, Recall: {R}, F1: {F1}")
                except Exception as e:
                    print(f"Error processing BERTScore for Claude: {e}")
                    df.at[index, 'eval_bert_precision'] = 'N/A'
                    df.at[index, 'eval_bert_recall'] = 'N/A'
                    df.at[index, 'eval_bert_f1'] = 'N/A'

            # If both exist, average the results
            if precision_scores and recall_scores and f1_scores:
                df.at[index, 'eval_bert_precision'] = sum(precision_scores) / len(precision_scores)
                df.at[index, 'eval_bert_recall'] = sum(recall_scores) / len(recall_scores)
                df.at[index, 'eval_bert_f1'] = sum(f1_scores) / len(f1_scores)

    # Save results
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

# Function to calculate and update Cosine Similarity
def process_cosine_similarity_with_lemmatization(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['eval_cosine_similarity']):
            similarities = []
            
            # Cosine Similarity for ChatGPT benchmark (with lemmatization)
            if 'prompt_benchmark_chatgpt' in row and pd.notna(row['prompt_benchmark_chatgpt']):
                lemmatized_msg_content = lemmatize_text(row['response_msg_content'])
                lemmatized_benchmark_response = lemmatize_text(row['prompt_benchmark_chatgpt'])
                similarity = compute_cosine_similarity(lemmatized_msg_content, lemmatized_benchmark_response)
                # Append 0 if similarity is None (error case)
                similarities.append(similarity if similarity is not None else 0)
                print(f"Row {index+1}: Cosine Similarity (ChatGPT) after Lemmatization: {similarity}")
            
            # Cosine Similarity for Claude benchmark (with lemmatization)
            if 'prompt_benchmark_claude' in row and pd.notna(row['prompt_benchmark_claude']):
                lemmatized_msg_content = lemmatize_text(row['response_msg_content'])
                lemmatized_benchmark_response = lemmatize_text(row['prompt_benchmark_claude'])
                similarity = compute_cosine_similarity(lemmatized_msg_content, lemmatized_benchmark_response)
                # Append 0 if similarity is None (error case)
                similarities.append(similarity if similarity is not None else 0)
                print(f"Row {index+1}: Cosine Similarity (Claude) after Lemmatization: {similarity}")

            # Average Cosine Similarity
            if similarities:
                df.at[index, 'eval_cosine_similarity'] = sum(similarities) / len(similarities)

    # Save results back to Excel
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

# Function to calculate and update Token Matching Similarity
def process_token_matching_with_lemmatization(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['eval_token_matching']):
            token_matches = []
            
            # Token Matching for ChatGPT benchmark (with lemmatization)
            if 'prompt_benchmark_chatgpt' in row and pd.notna(row['prompt_benchmark_chatgpt']):
                lemmatized_msg_content = lemmatize_text(row['response_msg_content'])
                lemmatized_benchmark_response = lemmatize_text(row['prompt_benchmark_chatgpt'])
                
                try:
                    match_score = token_level_matching(lemmatized_msg_content, lemmatized_benchmark_response)
                    token_matches.append(match_score)
                    print(f"Row {index+1}: Token Matching (ChatGPT) after Lemmatization: {match_score}")
                except ValueError as e:
                    print(f"Error processing Token Matching (ChatGPT) for Row {index+1}: {e}")
                    token_matches.append(0)  # Assign a score of 0 if there's an error
            
            # Token Matching for Claude benchmark (with lemmatization)
            if 'prompt_benchmark_claude' in row and pd.notna(row['prompt_benchmark_claude']):
                lemmatized_msg_content = lemmatize_text(row['response_msg_content'])
                lemmatized_benchmark_response = lemmatize_text(row['prompt_benchmark_claude'])
                
                try:
                    match_score = token_level_matching(lemmatized_msg_content, lemmatized_benchmark_response)
                    token_matches.append(match_score)
                    print(f"Row {index+1}: Token Matching (Claude) after Lemmatization: {match_score}")
                except ValueError as e:
                    print(f"Error processing Token Matching (Claude) for Row {index+1}: {e}")
                    token_matches.append(0)  # Assign a score of 0 if there's an error

            # Average Token Matching
            if token_matches:
                df.at[index, 'eval_token_matching'] = sum(token_matches) / len(token_matches)

    # Save results back to Excel
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

# Function to calculate and update Semantic Similarity
def process_semantic_similarity(df, file_path, sheet_name):
    for index, row in df.iterrows():
        if pd.isna(row['eval_semantic_similarity']):
            similarities = []
            
            # Semantic Similarity for ChatGPT benchmark
            if 'prompt_benchmark_chatgpt' in row and pd.notna(row['prompt_benchmark_chatgpt']):
                try:
                    similarity = compute_semantic_similarity(row['response_msg_content'], row['prompt_benchmark_chatgpt'], tokenizer, model)
                    similarities.append(similarity)
                    print(f"Row {index+1}: Semantic Similarity (ChatGPT): {similarity}")
                except Exception as e:
                    print(f"Error processing Semantic Similarity (ChatGPT) for Row {index+1}: {e}")
                    similarities.append(0)  # Assign a default score of 0 in case of an error
            
            # Semantic Similarity for Claude benchmark
            if 'prompt_benchmark_claude' in row and pd.notna(row['prompt_benchmark_claude']):
                try:
                    similarity = compute_semantic_similarity(row['response_msg_content'], row['prompt_benchmark_claude'], tokenizer, model)
                    similarities.append(similarity)
                    print(f"Row {index+1}: Semantic Similarity (Claude): {similarity}")
                except Exception as e:
                    print(f"Error processing Semantic Similarity (Claude) for Row {index+1}: {e}")
                    similarities.append(0)  # Assign a default score of 0 in case of an error

            # Average Semantic Similarity
            if similarities:
                df.at[index, 'eval_semantic_similarity'] = sum(similarities) / len(similarities)

    # Save results back to Excel
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

# Summarize and update responses in dataframe
def process_summaries(df, file_path, sheet_name, tokenizer, model):
    if 'eval_summary' not in df.columns:
        df['eval_summary'] = pd.NA

    for index, row in df.iterrows():
        if pd.isna(row['eval_summary']):
            try:
                response = row['response_msg_content']

                # Call the generalized summarization function
                summary = summarize_based_on_token_count(response, tokenizer, model)

                df.at[index, 'eval_summary'] = summary

            except Exception as e:
                print(f"Error generating summary for row {index+1}: {e}")
                df.at[index, 'eval_summary'] = "Error"
        
    # Save results back to Excel
    df.to_excel(file_path, sheet_name=sheet_name, index=False)
    print(f"💾 Summaries saved to {file_path}")

def process_model_evaluations(df, output_file, model_name, eval_function, current_mode):
    """
    Generalized function to process Gemini or Cohere evaluations.
    """
    # Track the row progress and save frequency
    save_frequency = row_save_frequency

    # Loop through each row (response) in the DataFrame
    for index, row in df.iterrows():
        print(f"🔍 Processing row {index+1}/{len(df)}")
        prompt = row['responses_prompt_text']
        response = row['response_msg_content']

        # Set the current mode to "Normal" for evaluations
        current_mode = "Normal"

        # Perform evaluations for each aspect (non-variance aspects)
        eval_aspects = ["Accuracy", "Clarity", "Relevance", "Adherence", "Insight"]
        
        for aspect in eval_aspects:
            # Check if the aspect has already been evaluated
            if pd.isna(row.get(f'{model_name}_{aspect}_rating')):
                try:
                    print(f"🤖 '{model_name}' evaluating {aspect} for row {index+1}...\n")
                    rating, explanation = eval_function(response, prompt, aspect, model_name, current_mode)
                    
                    # Use specific sleep times for each model
                    if model_name == "cohere_command_r":
                        time.sleep(sleep_time_api)
                    elif model_name == "gemini-1.5-flash":
                        time.sleep(sleep_time_api)  # Default API sleep time for Gemini
                    else:
                        time.sleep(sleep_time_api)  # Fallback in case new models are added

                    # Dynamically set the column names based on the model (Gemini or Cohere)
                    df.at[index, f'{model_name}_{aspect}_rating'] = rating
                    df.at[index, f'{model_name}_{aspect}_explain'] = explanation
                except Exception as e:
                    print(f"❗ No valid {aspect} response generated for {model_name}. Error: {str(e)}")
                    df.at[index, f'{model_name}_{aspect}_rating'] = "N/A"
                    df.at[index, f'{model_name}_{aspect}_explain'] = "No valid response generated."
            else:
                print(f"🦘 Skipping {aspect} for {model_name}, already evaluated.\n")

        # Handle the Variance evaluation using the pre-constructed eval_response_variance_content
        try:
            msg_content_variance = row.get('eval_response_variance_content', None)

            if not msg_content_variance:
                print(f"❗ Missing eval_response_variance_content for row {index+1}, skipping variance evaluation.\n")
                variance_chatgpt_rating, variance_chatgpt_explanation = "N/A", "No valid eval_response_variance_content provided."
                variance_claude_rating, variance_claude_explanation = "N/A", "No valid eval_response_variance_content provided."
            else:
                # Check if variance has already been evaluated
                if pd.isna(row.get(f'{model_name}_variance_chatgpt')) or pd.isna(row.get(f'{model_name}_variance_claude')):
                    print(f"🤖 '{model_name}' evaluating Variance for row {index+1}...\n")
                    # Pass the full eval_response_variance_content to the eval function
                    variance_chatgpt_rating, variance_chatgpt_explanation, variance_claude_rating, variance_claude_explanation = eval_function(
                        msg_content_variance, prompt, "Variance", model_name, current_mode
                    )

                    # Update the DataFrame with the variance results
                    df.at[index, f'{model_name}_variance_chatgpt'] = variance_chatgpt_rating
                    df.at[index, f'{model_name}_variance_chatgpt_explain'] = variance_chatgpt_explanation
                    df.at[index, f'{model_name}_variance_claude'] = variance_claude_rating
                    df.at[index, f'{model_name}_variance_claude_explain'] = variance_claude_explanation
                else:
                    print(f"🦘 Skipping Variance for {model_name}, already evaluated.\n")
        except Exception as e:
            print(f"❗ No valid Variance response generated for {model_name}. Error: {str(e)}")
            df.at[index, f'{model_name}_variance_chatgpt'] = "N/A"
            df.at[index, f'{model_name}_variance_chatgpt_explain'] = "No valid response generated."
            df.at[index, f'{model_name}_variance_claude'] = "N/A"
            df.at[index, f'{model_name}_variance_claude_explain'] = "No valid response generated."

        # Save the updated DataFrame to the Excel file every 500 rows
        if (index + 1) % save_frequency == 0:
            print(f"💾 Saving progress at row {index+1} to {output_file}\n")
            df.to_excel(output_file, index=False)

    # Final save at the end of the process
    print(f"💾 Final save for {model_name} evaluations completed!\n")
    df.to_excel(output_file, index=False)

# Main processing function to run analyses
def process_selected_analysis_modes(input_file_path, output_file_path, selected_mode, sheet_name=eval_sheet_name, last_row=last_row_value, first_row=first_row_value):
    """
    Process selected analysis modes: handle 'Compute Evaluations (All)', 'Gemini Evaluations (6 Aspects)', 'Cohere Evaluations (6 Aspects)', and 'Merge Excel Evaluation Results'.
    """
    print(f"Selected mode: '{selected_mode}'\n")
    
    # For merging, no need to load or work with a DataFrame
    if selected_mode == "Merge Excel Evaluation Results":
        print("↣ Merging the 3 evaluation results...\n")
        # merge_evaluations()  # Call the merge function directly
        
        # Run the external Python script to merge the Excel files
        try:
            subprocess.run(["python3", "process_excel_combine.py"], check=True)
            print("✅ Excel Results Merge Completed!\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error while running process_excel_combine.py: {e}")
        return  # No need to save anything, just exit after merging

    # Always load from the input file (remove the check for existing output file)
    print(f"🔄 Loading file {input_file_path}...")
    df = pd.read_excel(input_file_path, sheet_name=sheet_name, engine='openpyxl')
    print(f"☑️  Loaded file {input_file_path}.")

    # Load the summarization model once and pass it to the processing function
    tokenizer, model = load_summarization_model()  # Default is 'facebook/bart-large-cnn', or change if needed

    # Ensure the dataframe is truncated between the first and last row of interest
    df = df.iloc[first_row:last_row]

    print("🔄 Initiating analysis...\n")

    # Handle the 'Compute Evaluations (All)' option
    if selected_mode == "Compute Evaluations (All)":
        print("🔄 Running all evaluations...\n")
        
        # Add debug print statements between each analysis
        '''print("🔄 Running Sentence Count...\n")
        process_sentence_count(df)
        print("✅ Completed Sentence Count...\n")
        # Save progress after sentence count
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Sentence Count to {output_file_path}.\n")

        print("🔄 Running Token Count...\n")
        process_token_count(df)
        print("✅ Completed Token Count...\n")
        # Save progress after token count
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Token Count to {output_file_path}.\n")

        print("🔄 Running Character Count...\n")
        process_char_count(df)
        print("✅ Completed Character Count...\n")
        # Save progress after character count
        print("🔄 Saving progress to Excel...\n")        
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Character Count to {output_file_path}.\n")

        print("🔄 Running Word Count...\n")
        process_word_count(df)
        print("✅ Completed Word Count...\n")
        # Save progress after word count
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Word Count to {output_file_path}.\n")

        print("🔄 Running Named Entities...\n")
        process_named_entities(df, input_file_path, sheet_name)
        print("✅ Completed Named Entities...\n")
        # Save progress after named entities
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Named Entities to {output_file_path}.\n")

        print("🔄 Running Sentiment Polarity...\n")
        process_polarity_sentiment(df)
        print("✅ Completed Sentiment Polarity...\n")
        # Save progress after polarity sentiment
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Sentiment Polarity to {output_file_path}.\n")

        print("🔄 Running Sentiment Subjectivity...\n")
        process_subjective_sentiment(df)
        print("✅ Completed Sentiment Subjectivity...\n")
        # Save progress after subjective sentiment
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Sentiment Subjectivity to {output_file_path}.\n")

        print("🔄 Running Flagged Words...\n")
        process_flagged_words(df, input_file_path, sheet_name)
        print("✅ Completed Flagged Words...\n")
        # Save progress after flagged words
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Flagged Words to {output_file_path}.\n")'''

        print("🔄 Running Hunspell Spell Check - Now with Getty & Gemini filtering!\n")
        process_spelling_with_ai(df, input_file_path, sheet_name, hunspell_obj)
        print("✅ Completed Spelling Errors...\n")
        # Save progress after spelling check
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Spelling Check to {output_file_path}.\n")
 
        '''print("🔄 Running Noun Phrases...\n")
        process_noun_phrases(df, input_file_path, sheet_name)
        print("✅ Completed Noun Phrases...\n")
        # Save progress after noun phrases
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Noun Phrases to {output_file_path}.\n")

        print("🔄 Running Cosine Similarity...\n")
        process_cosine_similarity_with_lemmatization(df, input_file_path, sheet_name)
        print("✅ Completed Cosine Similarity...\n")
        # Save progress after cosine similarity
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Cosine Similarity to {output_file_path}.\n")

        print("🔄 Running Token Matching...\n")
        process_token_matching_with_lemmatization(df, input_file_path, sheet_name)
        print("✅ Completed Token Matching...\n")
        # Save progress after token matching
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Token Matching to {output_file_path}.\n")

        print("🔄 Running Semantic Similarity...\n")
        process_semantic_similarity(df, input_file_path, sheet_name)
        print("✅ Completed Semantic Similarity...\n")
        # Save progress after semantic similarity
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Semantic Similarity to {output_file_path}.\n")

        print("🔄 Running BERTScore...\n")
        process_bertscore(df, input_file_path, sheet_name)
        print("✅ Completed BERTScore...\n")
        # Save progress after BERTScore
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after BERTScore to {output_file_path}.\n")

        print("🔄 Running Summarization...\n")
        process_summaries(df, input_file_path, sheet_name, tokenizer, model)
        print("✅ Completed Summarization...\n")
        # Save progress after noun phrases
        print("🔄 Saving progress to Excel...\n")
        df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        print(f"💾 Saved progress after Summarization to {output_file_path}.\n")'''

        print("✅ Compute-level Evaluations Completed!\n")

    # Handle the 'Gemini Evaluations (6 Aspects)' option
    elif selected_mode == "Gemini Evaluations (6 Aspects)":
        print("🏃🏻‍♂️‍➡️ Running 'Gemini 1.5 Flash' evaluations...\n")
        current_mode = "Normal"
        process_model_evaluations(df, output_file_path, "gemini-1.5-flash", evaluate_response_with_model, current_mode)
        print("✅ Gemini AI Evaluations Completed!\n")

    # Handle the 'Cohere Evaluations (6 Aspects)' option
    elif selected_mode == "Cohere Evaluations (6 Aspects)":
        print("🏃🏻‍♂️‍➡️ Running 'Cohere - Command-R' evaluations...\n")
        current_mode = "Normal"
        process_model_evaluations(df, output_file_path, "cohere_command_r", evaluate_response_with_model, current_mode)
        print("✅ Cohere AI Evaluations Completed!\n")

    # Final save after all evaluations
    print(f"💾 Saving to {output_file_path}...\n")
    df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
    print(f"✅ File saved as {output_file_path}\n")