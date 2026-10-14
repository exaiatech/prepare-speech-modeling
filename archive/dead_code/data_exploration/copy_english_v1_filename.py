# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 7 (0-based index in the original notebook).
# What it does: Copies English recordings using language == 'en' and a 'filename' column.
# Why archived: Superseded by cell 8. Uses the wrong column value ('en' vs 'english') and a 'filename' column that the metadata does not appear to have.
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

import os
import shutil
import pandas as pd

# 1. Define folder and file paths
metadata_file = r"C:\Users\arday\Downloads\metadata.csv"

source_train_dir = r"C:\Users\arday\Downloads\train_audios"
source_test_dir = r"C:\Users\arday\Downloads\test_audios"

output_train_dir = r"C:\Users\arday\Desktop\aaaa\train"
output_test_dir = r"C:\Users\arday\Desktop\aaaa\test"

# 2. Create the new folders if they do not exist
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# 3. Read metadata and get a list of English filenames
# Note: Update 'language' and 'filename' to match your actual CSV column names
df = pd.read_csv(metadata_file)
english_data = df[df['language'] == 'en'] 
english_filenames = english_data['filename'].tolist()

# 4. Create a function to copy files
def filter_and_copy(source_dir, output_dir, valid_files):
    # Check every file in the source folder
    for current_file in os.listdir(source_dir):
        # If the file is in our English list, copy it
        if current_file in valid_files:
            source_path = os.path.join(source_dir, current_file)
            output_path = os.path.join(output_dir, current_file)
            
            shutil.copy2(source_path, output_path)

# 5. Run the function for both train and test folders
print("Processing train files...")
filter_and_copy(source_train_dir, output_train_dir, english_filenames)

print("Processing test files...")
filter_and_copy(source_test_dir, output_test_dir, english_filenames)

print("All English files copied successfully!")
