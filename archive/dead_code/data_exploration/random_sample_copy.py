# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 2 (0-based index in the original notebook).
# What it does: Copies a fixed number of random train recordings per corpus for listening.
# Why archived: Manual listening sample. NOTE: .sample() has no random_state, so the selection is not reproducible. Left unchanged as archived code.
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

import pandas as pd
import os
import shutil

# --- FILE PATHS ---
# Update these paths to match your computer
METADATA_PATH = r"C:\Users\arday\Downloads\metadata.csv"
TRAIN_AUDIO_DIR = r"C:\Users\arday\Downloads\train_audios"
OUTPUT_DIR = r"C:\Users\arday\Desktop\UROP\Random_Audio_Samples"

# Create the output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- SELECTION RULES ---
# How many files we need from each corpus
target_counts = {
    'wls': 5,
    'pitt': 5,
    'vas': 3,
    'delaware': 3,
    'lu': 3,
    'hopkins': 3,
    'baycrest': 2,
    'depaul': 1
}

def select_random_audios():
    # Load metadata
    print("Reading metadata...")
    df = pd.read_csv(METADATA_PATH)
    
    # Keep only the 'train' files
    df_train = df[df['split'].str.lower() == 'train']
    
    print("Selecting files and copying to the new folder...")
    
    # Loop through our rules
    for corpus_name, needed_count in target_counts.items():
        # Find all train files for this specific corpus
        corpus_data = df_train[df_train['corpus'].str.lower() == corpus_name]
        
        # Check how many files actually exist
        available_files = len(corpus_data)
        
        # Pick the needed count, or all of them if we don't have enough
        pick_count = min(needed_count, available_files)
        
        if pick_count == 0:
            print(f"No files found for corpus: {corpus_name}")
            continue
            
        # Select random rows
        random_picks = corpus_data.sample(n=pick_count)
        
        # Copy each selected file to the new folder
        for index, row in random_picks.iterrows():
            uid = str(row['uid'])
            # Assuming your files are .mp3. Change to .wav if necessary.
            filename = f"{uid}.mp3" 
            
            source_path = os.path.join(TRAIN_AUDIO_DIR, filename)
            # Add corpus name to the file so you know what you are listening to
            new_filename = f"{corpus_name}_{filename}"
            target_path = os.path.join(OUTPUT_DIR, new_filename)
            
            # Check if file exists on your computer before copying
            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                print(f"Copied: {new_filename}")
            else:
                print(f"Missing audio file: {source_path}")

    print(f"\nSuccess! All random samples are ready inside: {OUTPUT_DIR}")

# Run the script
if __name__ == "__main__":
    select_random_audios()
