#!/usr/bin/env python3
# main.py
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

import pandas as pd
from text_processing import *




def process_excel(file_path):
    df = pd.read_excel(file_path, sheet_name='Model_Responses', engine='openpyxl')
    last_row = 2144
    print("🔄 Cosine similarity analysis in progress...\n")
    for index, row in df.iterrows():
        if index >= last_row:
            break

        # Check if the 'Cosine_Similarity' column is empty
        if pd.isna(row['Cosine_Similarity']):
            # Check for missing or invalid data before processing
            if pd.isna(row['Msg_Content']) or pd.isna(row['Benchmark_Response']):
                print(f"Skipping row {index} due to missing data.")
                continue

            # Compute cosine similarity safely
            similarity = compute_cosine_similarity(row['Msg_Content'], row['Benchmark_Response'])
            
            if similarity is not None:
                print(f"Row {index}: Cosine Similarity between model response and benchmark response: {similarity}")
                df.at[index, 'Cosine_Similarity'] = similarity
            else:
                print(f"Skipping row {index} due to an error in similarity computation.")
        else:
            print(f"Skipping row {index} because Cosine_Similarity is already calculated.")

    df.to_excel(file_path, sheet_name='Model_Responses', index=False)
    print("ℹ️ Cosine similarity has been calculated and saved to the Excel file! ✅")

# Run the process on the specific Excel file
process_excel('prompt_responses.xlsx')