# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 0 (0-based index in the original notebook).
# What it does: Per-recording MP3 duration / sample rate / channel inventory (mutagen).
# Why archived: One-off data quality check. Output is not consumed by any pipeline step.
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

import os
import pandas as pd
from mutagen.mp3 import MP3

# --- CONFIGURATION ---
METADATA_PATH = r"C:\Users\arday\Downloads\metadata.csv"
TRAIN_AUDIO_DIR = r"C:\Users\arday\Downloads\train_audios"
TEST_AUDIO_DIR = r"C:\Users\arday\Downloads\test_audios"
OUTPUT_FILE = r"C:\Users\arday\Desktop\UROP\english_audio_summary.csv"

def get_audio_features(mp3_path):
    """
    Reads an .mp3 file and extracts duration, sample rate, and channels.
    """
    if not os.path.exists(mp3_path):
        return None, None, None
        
    try:
        audio = MP3(mp3_path)
        duration_seconds = audio.info.length
        rate = audio.info.sample_rate
        channels = audio.info.channels
        
        return duration_seconds, rate, channels
    except Exception as e:
        print(f"Error reading audio {mp3_path}: {e}")
        return None, None, None

def main():
    """
    Main function to process English metadata and extract MP3 audio features.
    """
    print("Loading metadata...")
    if not os.path.exists(METADATA_PATH):
        print(f"ERROR: {METADATA_PATH} not found.")
        return

    df = pd.read_csv(METADATA_PATH)
    
    # Filter only English recordings
    df_english = df[df['language'].str.lower() == 'english'].copy()
    print(f"Found {len(df_english)} English records. Starting analysis...\n")
    
    results = []
    found_count = 0
    missing_count = 0
    
    # Process each English record
    for index, row in df_english.iterrows():
        uid = str(row['uid'])
        split_type = str(row.get('split', '')).lower()
        
        # Look for .mp3 files instead of .wav
        if split_type == 'test':
            audio_path = os.path.join(TEST_AUDIO_DIR, f"{uid}.mp3")
        else:
            audio_path = os.path.join(TRAIN_AUDIO_DIR, f"{uid}.mp3")
            
        # Extract Features
        duration, rate, channels = get_audio_features(audio_path)
        
        if duration is not None:
            found_count += 1
        else:
            missing_count += 1
            
        # Save results
        results.append({
            'UID': uid,
            'Diagnosis': row.get('diagnosis', ''),
            'Split': row.get('split', ''),
            'Audio_Duration_Seconds': duration,
            'Sample_Rate_Hz': rate,
            'Channels': channels
        })
        
    # Export to a new CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n--- ANALYSIS COMPLETE ---")
    print(f"Total English records processed: {len(df_english)}")
    print(f"Audio files successfully read: {found_count}")
    print(f"Audio files missing/not found: {missing_count}")
    print(f"\nSaved results to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
