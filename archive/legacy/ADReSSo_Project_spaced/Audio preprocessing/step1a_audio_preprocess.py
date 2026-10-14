import os
import glob
import librosa
import numpy as np
import soundfile as sf

# --- CONFIGURATION (Dynamic Relative Paths) ---

# 1. Find the exact folder where this Python script is currently located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Set Input and Output paths relative to the script's location
# ".." means "go up one main folder"
INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "train_audios")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "processed_audios")

TARGET_SR = 16000
TRIM_TOP_DB = 30


def preprocess_audio(file_path, output_path):
    """
    Cleans and standardizes a single audio file.
    Steps: Resample, remove DC offset, and trim silence.
    """
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    y_centered = y - np.mean(y)
    y_trimmed, _ = librosa.effects.trim(y_centered, top_db=TRIM_TOP_DB)
    sf.write(output_path, y_trimmed, TARGET_SR)


def main():
    """
    Main pipeline function to process all audio files in the input directory.
    """
    # Create output folder automatically if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Search for all .wav files in the dynamically found INPUT_DIR
    audio_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))

    if not audio_files:
        print(f"No audio files found in: {INPUT_DIR}")
        return

    print(f"Starting preprocessing for {len(audio_files)} files...")

    for file_path in audio_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_DIR, file_name)

        try:
            preprocess_audio(file_path, output_path)
            print(f"SUCCESS - {file_name}")
        except Exception as e:
            print(f"ERROR - {file_name} failed: {e}")

    print("All audio files processed successfully.")


if __name__ == "__main__":
    main()

