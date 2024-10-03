#!/usr/bin/env python3
# process_excel_apitable.py

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"
import pandas as pd
import os
import warnings
import time
from datetime import datetime
import subprocess
import json
import numpy as np
import re

# Suppress all warnings
warnings.filterwarnings("ignore")

# Function to check if a message exists in APITable using cURL
def message_exists_in_apitable(message_id):
    try:
        # cURL command to get all records from APITable
        curl_command = f"""
        curl -X GET "http://localhost:8080/fusion/v1/datasheets/dst7ueHm0SoLJMcFCL/records?fieldKey=name" \
        -H "Authorization: Bearer uskMfwX3Y4Ard86eEXY6ynf"
        """

        # Execute the cURL command and capture the output
        result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, text=True)

        # Convert the result from JSON to Python dictionary
        response_data = json.loads(result.stdout)

        # Iterate through the records to check for the message_id
        for record in response_data['data']['records']:
            if record['fields'].get('Message_ID') == message_id:
                print(f"Message ID {message_id} exists.")
                return True

        print(f"Message ID {message_id} does not exist.")
        return False

    except Exception as e:
        print(f"Error while checking for message existence: {e}")
        return False

# Function to safely handle NaN values
def safe_value(value):
    if pd.isna(value) or value == np.nan or value == "":
        return None
    return value

# Function to escape special characters in a string for shell commands
def escape_special_characters(text):
    if isinstance(text, str):
        # Escape quotes and backslashes
        return text.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
    return text

# Helper function to process each field
def process_field(value):
    return escape_special_characters(safe_value(value))

def execute_curl_command(index, record_json):
    curl_command = f"""
    curl -X POST "http://localhost:8080/fusion/v1/datasheets/dst7ueHm0SoLJMcFCL/records?viewId=viwjcjDfKpPH0&fieldKey=name" \
    -H "Authorization: Bearer uskMfwX3Y4Ard86eEXY6ynf" \
    -H "Content-Type: application/json" \
    --data '{record_json}'
    """
    print(f"Executing cURL command for row {index + 1}: {curl_command}")
    try:
        result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, text=True)
        result.check_returncode()  # This will raise an error if return code is non-zero
    except subprocess.CalledProcessError as e:
        print(f"❌ cURL command failed for row {index + 1}: {e}")
        print(f"Record data: {record_json}")  # Log the record data for troubleshooting
    
    if result.returncode == 0:
        print(f"✅ Successfully imported row {index + 1} to APITable.")
    else:
        print(f"❌ Failed to import row {index + 1}: {result.stdout}")

# Function to import data from Excel to APITable using cURL
def import_data_to_apitable(excel_file, sheet_name, start_row=0, end_row=None):
    # Load the Excel file
    print(f"📂 Loading {excel_file} for import to APITable...")
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Load all rows starting from start_row to end_row
    df = df.iloc[start_row:end_row]
    
    # Ensure the DataFrame isn’t empty, as this could cause issues if no rows are loaded.
    if df.empty:
        print(f"No rows to import from {excel_file}.")
        return

    # Loop through each row in the dataframe
    for index, row in df.iterrows():
        # Log progress every 1000 rows
        if index % 1000 == 0:
            print(f"Processed {index + 1} rows so far...")

        # Check if the Message_ID already exists or Msg_Status is "imported"
        if message_exists_in_apitable(row['Message_ID']):
            print(f"🔄 Skipping row {index + 1}: Message_ID '{row['Message_ID']}' already exists in APITable.")
            continue
        
        if row['Msg_Status'] == "imported":
            print(f"🔄 Skipping row {index + 1}: Msg_Status is already 'imported'.")
            continue

        # Prepare the data for the cURL request, using the helper function to process fields
        record_data = {
            "records": [
                {
                    "fields": {
                        "Message_ID": process_field(row['Message_ID']),
                        "Conv_ID": process_field(row['Conv_ID']),
                        "Prompt_ID_Raw": process_field(row['Prompt_ID_Raw']),
                        "LLM_Name_Raw": process_field(row['LLM_Name_Raw']),
                        "Msg_Content_Raw": process_field(row['Msg_Content_Raw']),
                        "gemini-1.5-flash_Accuracy_Rating": safe_value(row['gemini-1.5-flash_Accuracy_Rating']),
                        "gemini-1.5-flash_Accuracy_Explain": process_field(row['gemini-1.5-flash_Accuracy_Explain']),
                        "gemini-1.5-flash_Clarity_Rating": safe_value(row['gemini-1.5-flash_Clarity_Rating']),
                        "gemini-1.5-flash_Clarity_Explain": process_field(row['gemini-1.5-flash_Clarity_Explain']),
                        "gemini-1.5-flash_Relevance_Rating": safe_value(row['gemini-1.5-flash_Relevance_Rating']),
                        "gemini-1.5-flash_Relevance_Explain": process_field(row['gemini-1.5-flash_Relevance_Explain']),
                        "gemini-1.5-flash_Adherence_Rating": safe_value(row['gemini-1.5-flash_Adherence_Rating']),
                        "gemini-1.5-flash_Adherence_Explain": process_field(row['gemini-1.5-flash_Adherence_Explain']),
                        "gemini-1.5-flash_Insight_Rating": safe_value(row['gemini-1.5-flash_Insight_Rating']),
                        "gemini-1.5-flash_Insight_Explain": process_field(row['gemini-1.5-flash_Insight_Explain']),
                        "cohere_command_r_Accuracy_Rating": safe_value(row['cohere_command_r_Accuracy_Rating']),
                        "cohere_command_r_Accuracy_Explain": process_field(row['cohere_command_r_Accuracy_Explain']),
                        "cohere_command_r_Clarity_Rating": safe_value(row['cohere_command_r_Clarity_Rating']),
                        "cohere_command_r_Clarity_Explain": process_field(row['cohere_command_r_Clarity_Explain']),
                        "cohere_command_r_Relevance_Rating": safe_value(row['cohere_command_r_Relevance_Rating']),
                        "cohere_command_r_Relevance_Explain": process_field(row['cohere_command_r_Relevance_Explain']),
                        "cohere_command_r_Adherence_Rating": safe_value(row['cohere_command_r_Adherence_Rating']),
                        "cohere_command_r_Adherence_Explain": process_field(row['cohere_command_r_Adherence_Explain']),
                        "cohere_command_r_Insight_Rating": safe_value(row['cohere_command_r_Insight_Rating']),
                        "cohere_command_r_Insight_Explain": process_field(row['cohere_command_r_Insight_Explain']),
                        "Cosine_Similarity": safe_value(row['Cosine_Similarity']),
                        "Token_Matching": safe_value(row['Token_Matching']),
                        "Semantic_Similarity": safe_value(row['Semantic_Similarity']),
                        "Noun_Phrases": process_field(row['Noun_Phrases']),
                        "Spelling_Errors": process_field(row['Spelling_Errors']),
                        "Spelling_Error_Qty": safe_value(row['Spelling_Error_Qty']),
                        "BERT_Precision": safe_value(row['BERT_Precision']),
                        "BERT_Recall": safe_value(row['BERT_Recall']),
                        "BERT_F1": safe_value(row['BERT_F1']),
                        "Named_Entities": process_field(row['Named_Entities']),
                        "Flagged_Words": process_field(row['Flagged_Words']),
                        "Flagged_Penalty": safe_value(row['Flagged_Penalty']),
                        "Sentiment_Polarity": safe_value(row['Sentiment_Polarity']),
                        "Sentiment_Subjectivity": safe_value(row['Sentiment_Subjectivity']),
                        "Chars_Total": safe_value(row['Chars_Total']),
                        "Sentences_Total": safe_value(row['Sentences_Total']),
                        "Words_Total": safe_value(row['Words_Total']),
                        "Tokens_Total": safe_value(row['Tokens_Total']),
                        "Msg_Month": process_field(row['Msg_Month']),
                        "Msg_Year": safe_value(row['Msg_Year']),
                        "Msg_AuthorRole": process_field(row['Msg_AuthorRole']),
                        "Response_Dur": safe_value(row['Response_Dur']),
                        "Msg_Status": "imported",  # Set the status to imported after successful import
                        "Import_Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Add current timestamp
                    }
                }
            ],
            "fieldKey": "name"
        }

        try:
            # Convert the record data to JSON format for the cURL request
            record_json = json.dumps(record_data)
        except Exception as e:
            print(f"Error converting row {index + 1} to JSON: {e}")
            continue  # Skip this row if there's an issue

        # Execute the cURL command and print the command on screen
        execute_curl_command(index, record_json)

    print(f"🎉 All rows from {excel_file} have been imported to APITable.")

# Example usage of the function (you can adjust the path to the excel and sheet name as needed)
if __name__ == "__main__":
    excel_file_path = "MASTER_PROMPT_RESPONSES_EVALS.xlsx"
    sheet_name = "Sheet1"
    
    # Specify which rows to import (0-indexed, e.g., from row 100 to 200)
    start_row = 1
    end_row = 49000
    
    # Call the import function
    import_data_to_apitable(excel_file_path, sheet_name, start_row, end_row)