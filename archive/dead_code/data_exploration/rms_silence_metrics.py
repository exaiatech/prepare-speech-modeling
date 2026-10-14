# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 1 (0-based index in the original notebook).
# What it does: RMS loudness and silence proportion via librosa.effects.split(top_db=20).
# Why archived: Early exploration. This silence heuristic was replaced by WebRTC VAD.
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

import os
import pandas as pd
import librosa
import numpy as np

# --- CONFIGURATION ---
METADATA_PATH = r"C:\Users\arday\Downloads\metadata.csv"
TRAIN_DIR = r"C:\Users\arday\Downloads\train_audios"
TEST_DIR = r"C:\Users\arday\Downloads\test_audios"
OUTPUT_FILE = r"C:\Users\arday\Desktop\UROP\advanced_audio_metrics.csv"

def analyze_audio(file_path):
    """
    Calculates RMS loudness and Silence proportion for an audio file.
    """
    if not os.path.exists(file_path):
        return None, None

    try:
        # Load audio (y is the audio array, sr is sample rate)
        y, sr = librosa.load(file_path, sr=None)
        
        # 1. Calculate RMS Loudness
        # This finds the average energy/volume of the track
        rms_array = librosa.feature.rms(y=y)
        mean_rms = float(np.mean(rms_array))
        
        # 2. Calculate Silence Proportion
        # librosa.effects.split finds the non-silent parts. 
        # top_db=20 means any sound 20dB below the peak is considered silence.
        non_silent_intervals = librosa.effects.split(y, top_db=20)
        
        non_silent_samples = 0
        for start, end in non_silent_intervals:
            non_silent_samples += (end - start)
            
        total_samples = len(y)
        silent_samples = total_samples - non_silent_samples
        
        # Calculate the percentage of silence
        silence_proportion = float(silent_samples / total_samples) if total_samples > 0 else 0.0
        
        return mean_rms, silence_proportion
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def main():
    print("Loading metadata...")
    df = pd.read_csv(METADATA_PATH)
    
    # Filter only English records
    df_english = df[df['language'].str.lower() == 'english'].copy()
    print(f"Found {len(df_english)} English files to process.")
    
    results = []
    
    for index, row in df_english.iterrows():
        uid = str(row['uid'])
        split_type = str(row.get('split', '')).lower()
        
        # Determine the correct folder based on split
        if split_type == 'test':
            audio_path = os.path.join(TEST_DIR, f"{uid}.mp3")
        else:
            audio_path = os.path.join(TRAIN_DIR, f"{uid}.mp3")
            
        print(f"Analyzing: {uid}.mp3")
        rms, silence = analyze_audio(audio_path)
        
        if rms is not None:
            results.append({
                'UID': uid,
                'RMS_Loudness': round(rms, 4),
                'Silence_Proportion': round(silence, 4)
            })
            
    # Export results to a new CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Analysis Complete! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
