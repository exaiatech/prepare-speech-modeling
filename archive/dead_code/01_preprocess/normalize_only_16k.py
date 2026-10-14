# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 24 (0-based index in the original notebook).
# What it does: Resample to 16 kHz + peak normalization only (no filtering).
# Why archived: Competing preprocessing variant. See bandpass_only_16k.py.
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

import os
import glob
import librosa
import numpy as np
import soundfile as sf

# --- CONFIGURATION (Dynamic Relative Paths) ---

# 1. Find the exact folder where this Python script is currently located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Set Input and Output paths relative to the script's location
INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "train_audios")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "processed_audios")

TARGET_SR = 16000

def preprocess_audio(file_path, output_path):
    """
    Applies ONLY Peak Normalization to the audio file.
    No filtering, no trimming, no DC offset removal.
    Scales the audio so its maximum amplitude reaches 1.0.
    """
    # 1. Load audio and resample to target frequency
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    
    # 2. Apply Peak Normalization
    max_amplitude = np.max(np.abs(y))
    if max_amplitude > 0:
        y_normalized = y / max_amplitude
    else:
        y_normalized = y  # Skip if the audio file is completely silent
    
    # Save the normalized file
    sf.write(output_path, y_normalized, TARGET_SR)

def main():
    """
    Main pipeline function to process all audio files in the input directory.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    mp3_files = glob.glob(os.path.join(INPUT_DIR, "*.mp3"))
    
    audio_files = wav_files + mp3_files

    if not audio_files:
        print(f"No audio files (.wav or .mp3) found in: {INPUT_DIR}")
        return

    print(f"Starting normalization for {len(audio_files)} files...\n")

    for file_path in audio_files:
        file_name = os.path.basename(file_path)
        file_base_name = os.path.splitext(file_name)[0]
        output_file_name = f"{file_base_name}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_file_name)

        try:
            preprocess_audio(file_path, output_path)
            print(f"SUCCESS - {file_name} -> normalized & saved")
        except Exception as e:
            print(f"ERROR - {file_name} failed: {e}")

    print("\nAll audio files normalized successfully.")

if __name__ == "__main__":
    main()
