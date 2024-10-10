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

# Helper function to safely handle NaN values
def safe_value(value):
    if pd.isna(value):
        return None
    return value

# Function to get the recordId and existing field values from APITable using Message_ID
def get_record_data(message_id):
    try:
        curl_command = f"""
        curl -X GET "http://localhost:8080/fusion/v1/datasheets/dst7ueHm0SoLJMcFCL/records?fieldKey=name" \
        -H "Authorization: Bearer uskMfwX3Y4Ard86eEXY6ynf"
        """

        result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, text=True)
        response_data = json.loads(result.stdout)

        for record in response_data['data']['records']:
            if record['fields'].get('Message_ID') == message_id:
                # Return both the recordId and existing field values
                return record['recordId'], record['fields']

        return None, {}
    except Exception as e:
        print(f"Error fetching recordId: {e}")
        return None, {}

# Function to get the recordId from APITable using Message_ID
def get_record_id(message_id):
    try:
        curl_command = f"""
        curl -X GET "http://localhost:8080/fusion/v1/datasheets/dst7ueHm0SoLJMcFCL/records?fieldKey=name" \
        -H "Authorization: Bearer uskMfwX3Y4Ard86eEXY6ynf"
        """

        result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, text=True)
        response_data = json.loads(result.stdout)

        for record in response_data['data']['records']:
            if record['fields'].get('Message_ID') == message_id:
                return record['recordId']  # Return the recordId associated with this Message_ID

        return None
    except Exception as e:
        print(f"Error fetching recordId: {e}")
        return None

# Function to escape special characters for shell commands
def escape_special_characters(text):
    if isinstance(text, str):
        return text.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
    return text

# Function to process and update records in APITable
def add_data_to_apitable(excel_file, sheet_name, start_row=0, end_row=None):
    # Load the Excel data
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df = df.iloc[start_row:end_row]  # Load only the specified range of rows

    if df.empty:
        print("No data to process.")
        return

    for index, row in df.iterrows():
        message_id = row['Message_ID']
        record_id = get_record_id(message_id)

        if not record_id:
            print(f"Record for Message_ID {message_id} not found, skipping row {index + 1}.")
            continue

        record_data = {
            "records": [
                {
                    "recordId": record_id,
                    "fields": {
                        "Message_ID": (row['Message_ID']),
                        "Conv_ID": (row['Conv_ID']),
                        "Prompt_ID_Raw": (row['Prompt_ID_Raw']),
                        "LLM_Name_Raw": (row['LLM_Name_Raw']),
                        "Msg_Content_Raw": (row['Msg_Content_Raw']),
                        "gemini-1.5-flash_Accuracy_Rating": safe_value(row['gemini-1.5-flash_Accuracy_Rating']),
                        "gemini-1.5-flash_Accuracy_Explain": (row['gemini-1.5-flash_Accuracy_Explain']),
                        "gemini-1.5-flash_Clarity_Rating": safe_value(row['gemini-1.5-flash_Clarity_Rating']),
                        "gemini-1.5-flash_Clarity_Explain": (row['gemini-1.5-flash_Clarity_Explain']),
                        "gemini-1.5-flash_Relevance_Rating": safe_value(row['gemini-1.5-flash_Relevance_Rating']),
                        "gemini-1.5-flash_Relevance_Explain": (row['gemini-1.5-flash_Relevance_Explain']),
                        "gemini-1.5-flash_Adherence_Rating": safe_value(row['gemini-1.5-flash_Adherence_Rating']),
                        "gemini-1.5-flash_Adherence_Explain": (row['gemini-1.5-flash_Adherence_Explain']),
                        "gemini-1.5-flash_Insight_Rating": safe_value(row['gemini-1.5-flash_Insight_Rating']),
                        "gemini-1.5-flash_Insight_Explain": (row['gemini-1.5-flash_Insight_Explain']),
                        "cohere_command_r_Accuracy_Rating": safe_value(row['cohere_command_r_Accuracy_Rating']),
                        "cohere_command_r_Accuracy_Explain": (row['cohere_command_r_Accuracy_Explain']),
                        "cohere_command_r_Clarity_Rating": safe_value(row['cohere_command_r_Clarity_Rating']),
                        "cohere_command_r_Clarity_Explain": (row['cohere_command_r_Clarity_Explain']),
                        "cohere_command_r_Relevance_Rating": safe_value(row['cohere_command_r_Relevance_Rating']),
                        "cohere_command_r_Relevance_Explain": (row['cohere_command_r_Relevance_Explain']),
                        "cohere_command_r_Adherence_Rating": safe_value(row['cohere_command_r_Adherence_Rating']),
                        "cohere_command_r_Adherence_Explain": (row['cohere_command_r_Adherence_Explain']),
                        "cohere_command_r_Insight_Rating": safe_value(row['cohere_command_r_Insight_Rating']),
                        "cohere_command_r_Insight_Explain": (row['cohere_command_r_Insight_Explain']),
                        "Cosine_Similarity": safe_value(row['Cosine_Similarity']),
                        "Token_Matching": safe_value(row['Token_Matching']),
                        "Semantic_Similarity": safe_value(row['Semantic_Similarity']),
                        "Noun_Phrases": (row['Noun_Phrases']),
                        "Spelling_Errors": (row['Spelling_Errors']),
                        "Spelling_Error_Qty": safe_value(row['Spelling_Error_Qty']),
                        "BERT_Precision": safe_value(row['BERT_Precision']),
                        "BERT_Recall": safe_value(row['BERT_Recall']),
                        "BERT_F1": safe_value(row['BERT_F1']),
                        "Named_Entities": (row['Named_Entities']),
                        "Flagged_Words": (row['Flagged_Words']),
                        "Flagged_Penalty": safe_value(row['Flagged_Penalty']),
                        "Sentiment_Polarity": safe_value(row['Sentiment_Polarity']),
                        "Sentiment_Subjectivity": safe_value(row['Sentiment_Subjectivity']),
                        "Chars_Total": safe_value(row['Chars_Total']),
                        "Sentences_Total": safe_value(row['Sentences_Total']),
                        "Words_Total": safe_value(row['Words_Total']),
                        "Tokens_Total": safe_value(row['Tokens_Total']),
                        "Msg_Month": (row['Msg_Month']),
                        "Msg_Year": safe_value(row['Msg_Year']),
                        "Msg_AuthorRole": (row['Msg_AuthorRole']),
                        "Response_Dur": safe_value(row['Response_Dur']),
                        "Msg_Status": "imported",  # Set the status to imported after successful import
                        "Import_Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Add current timestamp
                    }
                }
            ],
            "fieldKey": "name"
        }

        try:
            # Convert record data to JSON
            record_json = json.dumps(record_data)
            curl_command = f"""
            curl -X PATCH "http://localhost:8080/fusion/v1/datasheets/dst7ueHm0SoLJMcFCL/records?viewId=viwNwRMHNfgtL&fieldKey=name" \
            -H "Authorization: Bearer uskMfwX3Y4Ard86eEXY6ynf" \
            -H "Content-Type: application/json" \
            --data '{record_json}'
            """
            result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, text=True)
            result.check_returncode()  # Raise error if the return code is non-zero

            print(f"✅ Row {index + 1} updated successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to update row {index + 1}: {e}")
        except Exception as e:
            print(f"❌ Error processing row {index + 1}: {e}")

# Function to process and update records in APITable
def update_data_in_apitable(excel_file, sheet_name, start_row=0, end_row=None):
    # Load the Excel data
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df = df.iloc[start_row:end_row]  # Load only the specified range of rows

    if df.empty:
        print("No data to process.")
        return

    for index, row in df.iterrows():
        message_id = row['Message_ID']
        record_id, existing_fields = get_record_data(message_id)  # Get record data

        if not record_id:
            print(f"Record for Message_ID {message_id} not found, skipping row {index + 1}.")
            continue

        # Build the field update payload, but only for fields that are empty
        fields_to_update = {}

        def update_if_empty(field_name, new_value):
            if not existing_fields.get(field_name) and safe_value(new_value) is not None:
                fields_to_update[field_name] = safe_value(new_value)

        # Update only if the field is empty
        update_if_empty("gemini-1.5-flash_Accuracy_Rating", row['gemini-1.5-flash_Accuracy_Rating'])
        update_if_empty("gemini-1.5-flash_Accuracy_Explain", row['gemini-1.5-flash_Accuracy_Explain'])
        update_if_empty("gemini-1.5-flash_Clarity_Rating", row['gemini-1.5-flash_Clarity_Rating'])
        update_if_empty("gemini-1.5-flash_Clarity_Explain", row['gemini-1.5-flash_Clarity_Explain'])
        update_if_empty("gemini-1.5-flash_Relevance_Rating", row['gemini-1.5-flash_Relevance_Rating'])
        update_if_empty("gemini-1.5-flash_Relevance_Explain", row['gemini-1.5-flash_Relevance_Explain'])
        update_if_empty("gemini-1.5-flash_Adherence_Rating", row['gemini-1.5-flash_Adherence_Rating'])
        update_if_empty("gemini-1.5-flash_Adherence_Explain", row['gemini-1.5-flash_Adherence_Explain'])
        update_if_empty("gemini-1.5-flash_Insight_Rating", row['gemini-1.5-flash_Insight_Rating'])
        update_if_empty("gemini-1.5-flash_Insight_Explain", row['gemini-1.5-flash_Insight_Explain'])
        update_if_empty("gemini-1.5-flash_Variance_ChatGPT", row['gemini-1.5-flash_Variance_ChatGPT'])
        update_if_empty("gemini-1.5-flash_Variance_ChatGPT_Explain", row['gemini-1.5-flash_Variance_ChatGPT_Explain'])
        update_if_empty("gemini-1.5-flash_Variance_Claude", row['gemini-1.5-flash_Variance_Claude'])
        update_if_empty("gemini-1.5-flash_Variance_Claude_Explain", row['gemini-1.5-flash_Variance_Claude_Explain'])

        update_if_empty("cohere_command_r_Accuracy_Rating", row['cohere_command_r_Accuracy_Rating'])
        update_if_empty("cohere_command_r_Accuracy_Explain", row['cohere_command_r_Accuracy_Explain'])
        update_if_empty("cohere_command_r_Clarity_Rating", row['cohere_command_r_Clarity_Rating'])
        update_if_empty("cohere_command_r_Clarity_Explain", row['cohere_command_r_Clarity_Explain'])
        update_if_empty("cohere_command_r_Relevance_Rating", row['cohere_command_r_Relevance_Rating'])
        update_if_empty("cohere_command_r_Relevance_Explain", row['cohere_command_r_Relevance_Explain'])
        update_if_empty("cohere_command_r_Adherence_Rating", row['cohere_command_r_Adherence_Rating'])
        update_if_empty("cohere_command_r_Adherence_Explain", row['cohere_command_r_Adherence_Explain'])
        update_if_empty("cohere_command_r_Insight_Rating", row['cohere_command_r_Insight_Rating'])
        update_if_empty("cohere_command_r_Insight_Explain", row['cohere_command_r_Insight_Explain'])
        update_if_empty("cohere_command_r_Variance_ChatGPT", row['cohere_command_r_Variance_ChatGPT'])
        update_if_empty("cohere_command_r_Variance_ChatGPT_Explain", row['cohere_command_r_Variance_ChatGPT_Explain'])
        update_if_empty("cohere_command_r_Variance_Claude", row['cohere_command_r_Variance_Claude'])
        update_if_empty("cohere_command_r_Variance_Claude_Explain", row['cohere_command_r_Variance_Claude_Explain'])

        update_if_empty("Cosine_Similarity", row['Cosine_Similarity'])
        update_if_empty("Token_Matching", row['Token_Matching'])
        update_if_empty("Semantic_Similarity", row['Semantic_Similarity'])
        update_if_empty("Noun_Phrases", row['Noun_Phrases'])
        update_if_empty("Spelling_Errors", row['Spelling_Errors'])
        update_if_empty("Spelling_Error_Qty", row['Spelling_Error_Qty'])
        update_if_empty("BERT_Precision", row['BERT_Precision'])
        update_if_empty("BERT_Recall", row['BERT_Recall'])
        update_if_empty("BERT_F1", row['BERT_F1'])
        update_if_empty("Named_Entities", row['Named_Entities'])
        update_if_empty("Flagged_Words", row['Flagged_Words'])
        update_if_empty("Flagged_Penalty", row['Flagged_Penalty'])
        update_if_empty("Sentiment_Polarity", row['Sentiment_Polarity'])
        update_if_empty("Sentiment_Subjectivity", row['Sentiment_Subjectivity'])
        update_if_empty("Chars_Total", row['Chars_Total'])
        update_if_empty("Sentences_Total", row['Sentences_Total'])
        update_if_empty("Words_Total", row['Words_Total'])
        update_if_empty("Tokens_Total", row['Tokens_Total'])

        # Fields to update regardless of whether they're empty or not
        fields_to_update["Msg_Status"] = "api_updated"
        fields_to_update["Import_Timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # If there are no fields to update, skip this row
        if not fields_to_update:
            print(f"🔄 Skipping row {index + 1}: No fields to update.")
            continue

        # Prepare the record update data
        record_data = {
            "records": [
                {
                    "recordId": record_id,
                    "fields": fields_to_update
                }
            ],
            "fieldKey": "name"
        }

        try:
            # Convert record data to JSON
            record_json = json.dumps(record_data)
            curl_command = f"""
            curl -X PATCH "http://localhost:8080/fusion/v1/datasheets/dst7ueHm0SoLJMcFCL/records?viewId=viwNwRMHNfgtL&fieldKey=name" \
            -H "Authorization: Bearer uskMfwX3Y4Ard86eEXY6ynf" \
            -H "Content-Type: application/json" \
            --data '{record_json}'
            """
            result = subprocess.run(curl_command, shell=True, stdout=subprocess.PIPE, text=True)
            result.check_returncode()  # Raise error if the return code is non-zero

            print(f"✅ Row {index + 1} updated successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to update row {index + 1}: {e}")
        except Exception as e:
            print(f"❌ Error processing row {index + 1}: {e}")

# Example usage
if __name__ == "__main__":
    excel_file_path = "MASTER_PROMPT_RESPONSES_EVALS.xlsx"
    sheet_name = "Sheet1"
    start_row = 1
    end_row = 500  # Adjust this as necessary

    update_data_in_apitable(excel_file_path, sheet_name, start_row, end_row)