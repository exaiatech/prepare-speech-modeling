import os
import glob
import librosa
import numpy as np
import soundfile as sf

# --- CONFIGURATION (Finalized Settings) ---
INPUT_DIR = r"C:\Users\arday\Desktop\Old_Audio"
OUTPUT_DIR = r"C:\Users\arday\Desktop\New_Audio"
TARGET_SR = 16000
TRIM_TOP_DB = 30


def preprocess_audio(file_path, output_path):
    """
    Cleans and standardizes a single audio file.
    Steps: Resample, remove DC offset, and trim silence.
    """
    # 1. Loading and Resampling
    y, sr = librosa.load(file_path, sr=TARGET_SR)

    # 2. DC Offset Removal (Centers the waveform)
    y_centered = y - np.mean(y)

    # 3. Silence Trimming (Removes dead silence from start and end)
    y_trimmed, _ = librosa.effects.trim(y_centered, top_db=TRIM_TOP_DB)

    # 4. Save processed audio
    sf.write(output_path, y_trimmed, TARGET_SR)


def main():
    """
    Main pipeline function to process all audio files in the input directory.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audio_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))

    if not audio_files:
        print("No audio files found in the input directory.")
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
print("All voices processed")


