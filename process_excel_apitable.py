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

import numpy as np

# Function to safely handle NaN values
def safe_value(value):
    if pd.isna(value) or value == np.nan:
        return None
    return value

# Function to import data from Excel to APITable using cURL
def import_data_to_apitable(excel_file, sheet_name, start_row=0, end_row=None):
    # Load the Excel file
    print(f"📂 Loading {excel_file} for import to APITable...")
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # No need to limit the rows, just load everything from start_row onward
    df = df.iloc[start_row:]  # This will load all rows starting from start_row

    # Loop through each row in the dataframe
    for index, row in df.iterrows():
        # Check if the Message_ID already exists or Msg_Status is "imported"
        if message_exists_in_apitable(row['Message_ID']):
            print(f"🔄 Skipping row {index + 1}: Message_ID '{row['Message_ID']}' already exists in APITable.")
            continue
        
        if row['Msg_Status'] == "imported":
            print(f"🔄 Skipping row {index + 1}: Msg_Status is already 'imported'.")
            continue

        # Prepare the data for the cURL request, using safe_value() to handle NaNs
        record_data = {
            "records": [
                {
                    "fields": {
                        "Message_ID": safe_value(row['Message_ID']),
                        "Conv_ID": safe_value(row['Conv_ID']),
                        "Prompt_ID_Raw": safe_value(row['Prompt_ID_Raw']),
                        "LLM_Name_Raw": safe_value(row['LLM_Name_Raw']),
                        "Msg_Content_Raw": safe_value(row['Msg_Content_Raw']),
                        "gemini-1.5-flash_Accuracy_Rating": safe_value(row['gemini-1.5-flash_Accuracy_Rating']),
                        "gemini-1.5-flash_Accuracy_Explain": safe_value(row['gemini-1.5-flash_Accuracy_Explain']),
                        "gemini-1.5-flash_Clarity_Rating": safe_value(row['gemini-1.5-flash_Clarity_Rating']),
                        "gemini-1.5-flash_Clarity_Explain": safe_value(row['gemini-1.5-flash_Clarity_Explain']),
                        "gemini-1.5-flash_Relevance_Rating": safe_value(row['gemini-1.5-flash_Relevance_Rating']),
                        "gemini-1.5-flash_Relevance_Explain": safe_value(row['gemini-1.5-flash_Relevance_Explain']),
                        "gemini-1.5-flash_Adherence_Rating": safe_value(row['gemini-1.5-flash_Adherence_Rating']),
                        "gemini-1.5-flash_Adherence_Explain": safe_value(row['gemini-1.5-flash_Adherence_Explain']),
                        "gemini-1.5-flash_Insight_Rating": safe_value(row['gemini-1.5-flash_Insight_Rating']),
                        "gemini-1.5-flash_Insight_Explain": safe_value(row['gemini-1.5-flash_Insight_Explain']),
                        "cohere_command_r_Accuracy_Rating": safe_value(row['cohere_command_r_Accuracy_Rating']),
                        "cohere_command_r_Accuracy_Explain": safe_value(row['cohere_command_r_Accuracy_Explain']),
                        "cohere_command_r_Clarity_Rating": safe_value(row['cohere_command_r_Clarity_Rating']),
                        "cohere_command_r_Clarity_Explain": safe_value(row['cohere_command_r_Clarity_Explain']),
                        "cohere_command_r_Relevance_Rating": safe_value(row['cohere_command_r_Relevance_Rating']),
                        "cohere_command_r_Relevance_Explain": safe_value(row['cohere_command_r_Relevance_Explain']),
                        "cohere_command_r_Adherence_Rating": safe_value(row['cohere_command_r_Adherence_Rating']),
                        "cohere_command_r_Adherence_Explain": safe_value(row['cohere_command_r_Adherence_Explain']),
                        "cohere_command_r_Insight_Rating": safe_value(row['cohere_command_r_Insight_Rating']),
                        "cohere_command_r_Insight_Explain": safe_value(row['cohere_command_r_Insight_Explain']),
                        "Cosine_Similarity": safe_value(row['Cosine_Similarity']),
                        "Token_Matching": safe_value(row['Token_Matching']),
                        "Semantic_Similarity": safe_value(row['Semantic_Similarity']),
                        "Noun_Phrases": safe_value(row['Noun_Phrases']),
                        "Spelling_Errors": safe_value(row['Spelling_Errors']),
                        "Spelling_Error_Qty": safe_value(row['Spelling_Error_Qty']),
                        "BERT_Precision": safe_value(row['BERT_Precision']),
                        "BERT_Recall": safe_value(row['BERT_Recall']),
                        "BERT_F1": safe_value(row['BERT_F1']),
                        "Named_Entities": safe_value(row['Named_Entities']),
                        "Flagged_Words": safe_value(row['Flagged_Words']),
                        "Flagged_Penalty": safe_value(row['Flagged_Penalty']),
                        "Sentiment_Polarity": safe_value(row['Sentiment_Polarity']),
                        "Sentiment_Subjectivity": safe_value(row['Sentiment_Subjectivity']),
                        "Chars_Total": safe_value(row['Chars_Total']),
                        "Sentences_Total": safe_value(row['Sentences_Total']),
                        "Words_Total": safe_value(row['Words_Total']),
                        "Tokens_Total": safe_value(row['Tokens_Total']),
                        "Msg_Month": safe_value(row['Msg_Month']),
                        "Msg_Year": safe_value(row['Msg_Year']),
                        "Msg_AuthorRole": safe_value(row['Msg_AuthorRole']),
                        "Response_Dur": safe_value(row['Response_Dur']),
                        "Msg_Status": "imported",  # Set the status to imported after successful import
                        "Import_Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Add current timestamp
                    }
                }
            ],
            "fieldKey": "name"
        }

        # Convert the record data to JSON format for the cURL request
        record_json = json.dumps(record_data)

        # Print the cURL command for debugging purposes
        print(f"\nExecuting cURL command for row {index + 1}:\n{record_json}\n")

        # Execute the cURL command
        curl_command = f"""
        curl -X POST "http://localhost:8080/fusion/v1/datasheets/dst7ueHm0SoLJMcFCL/records?viewId=viwjcjDfKpPH0&fieldKey=name" \
        -H "Authorization: Bearer uskMfwX3Y4Ard86eEXY6ynf" \
        -H "Content-Type: application/json" \
        --data '{record_json}'
        """

        # Run the cURL command and capture the response
        result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Print the full response details
        print(f"cURL Response (row {index + 1}):")
        print(f"Status Code: {result.returncode}")
        print(f"Response Output: {result.stdout}")
        print(f"Response Error: {result.stderr}")

        # Check if the request was successful
        if result.returncode == 0:
            print(f"✅ Successfully imported row {index + 1} to APITable.")
        else:
            print(f"❌ Failed to import row {index + 1}: {result.stdout}")

        # Optional sleep to avoid overloading API
        time.sleep(0.1)

    print(f"🎉 All rows from {excel_file} have been imported to APITable.")
# Example usage of the function (you can adjust the path to the excel and sheet name as needed)
if __name__ == "__main__":
    excel_file_path = "MASTER_PROMPT_RESPONSES_EVALS.xlsx"
    sheet_name = "Sheet1"
    
    # Specify which rows to import (0-indexed, e.g., from row 100 to 200)
    start_row = 1
    end_row = 10
    
    # Call the import function
    import_data_to_apitable(excel_file_path, sheet_name, start_row, end_row)