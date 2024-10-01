import pandas as pd
import os
import glob
from datetime import datetime
import shutil

def get_directory(default_directory):
    directory = input(f"Enter the directory (default is {default_directory}): ")
    return directory if directory else default_directory

def move_merged_files(files, input_directory):
    merged_directory = os.path.join(input_directory, 'merged')
    os.makedirs(merged_directory, exist_ok=True)
    for file in files:
        shutil.move(file, merged_directory)

def combine_excel_files(input_directory='responses/excel'):
    input_directory = get_directory(input_directory)
    all_files = glob.glob(os.path.join(input_directory, "*.xlsx"))
    num_files = len(all_files)
    
    # Display file names and sizes, and ask user to select files to combine
    print(f"Found {num_files} Excel files in directory: {input_directory}")
    for i, file in enumerate(all_files, start=1):
        size = os.path.getsize(file) / 1024  # size in KB
        if size > 1024:
            size /= 1024  # convert to MB if size > 1024 KB
            print(f"{i}. {os.path.basename(file)} ({size:.2f} MB)")
        else:
            print(f"{i}. {os.path.basename(file)} ({size:.2f} KB)")
    
    selection = input("Enter the indices of the files to combine, separated by commas, or press Enter to select all: ")
    if selection:
        selected_indices = [int(index) - 1 for index in selection.split(',') if index.isdigit()]
        all_files = [all_files[i] for i in selected_indices]
    
    print(f"Selected {len(all_files)} files to combine.")
    for file in all_files:
        print(os.path.basename(file))
    
    confirm = input("Do you want to proceed with combining these files? Y/N: ")
    if confirm.upper() != 'Y':
        print("Operation cancelled.")
        return
    
    # Combine selected Excel files
    all_dfs = (pd.read_excel(f) for f in all_files)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Add a timestamp to the output file name
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_file = os.path.join(input_directory, f'combined_{timestamp}.xlsx')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    combined_df.to_excel(output_file, index=False)
    print(f"Combined file saved as {output_file}")

    # Move the merged files to a subfolder
    move_merged_files(all_files, input_directory)

def main():
    print("Please select the file type to merge:")
    print("1: Excel")
    choice = input("Enter your choice (1): ")
    if choice == '1':
        combine_excel_files()
    else:
        print("Invalid choice. Please enter 1.")

# Usage
if __name__ == "__main__":
    main()