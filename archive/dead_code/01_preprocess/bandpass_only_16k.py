# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 23 (0-based index in the original notebook).
# What it does: Resample to 16 kHz + Butterworth band-pass only (no normalization).
# Why archived: Competing preprocessing variant. Cell 22 (band-pass + peak normalization, original sample rate) was kept as acoustic_pipeline/src/01_preprocess.py.
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

import os
import glob
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter

# --- CONFIGURATION (Dynamic Relative Paths) ---

# 1. Find the exact folder where this Python script is currently located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Set Input and Output paths relative to the script's location
INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "train_audios")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "processed_audios")

TARGET_SR = 16000

# --- FILTER SETTINGS ---
LOWCUT_HZ = 75.0      # Eliminates low-frequency mechanical rumble
HIGHCUT_HZ = 8000.0   # Eliminates high-frequency electronic hiss
FILTER_ORDER = 4      # Order of the Butterworth filter (steepness)


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Creates the Butterworth band-pass filter coefficients.
    """
    nyq = 0.5 * fs # Nyquist frequency
    low = lowcut / nyq
    
    # Ensure highcut doesn't exceed Nyquist frequency
    high = min(highcut / nyq, 0.999) 
    
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, fs, lowcut, highcut, order=4):
    """
    Applies the band-pass filter to the audio data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y_filtered = lfilter(b, a, data)
    return y_filtered


def preprocess_audio(file_path, output_path):
    """
    Applies ONLY the Band-Pass filter to the audio file.
    No trimming, no DC offset removal.
    """
    # 1. Load audio and resample to target frequency
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    
    # 2. Apply ONLY Butterworth Band-Pass Filter (75 Hz - 8 kHz)
    y_filtered = apply_bandpass_filter(y, TARGET_SR, LOWCUT_HZ, HIGHCUT_HZ, order=FILTER_ORDER)
    
    # Save the filtered file
    sf.write(output_path, y_filtered, TARGET_SR)


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

    print(f"Starting filtering for {len(audio_files)} files...")
    print(f"Applying ONLY Band-Pass Filter: {LOWCUT_HZ}Hz - {HIGHCUT_HZ}Hz\n")

    for file_path in audio_files:
        file_name = os.path.basename(file_path)
        file_base_name = os.path.splitext(file_name)[0]
        output_file_name = f"{file_base_name}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_file_name)

        try:
            preprocess_audio(file_path, output_path)
            print(f"SUCCESS - {file_name} -> filtered & saved")
        except Exception as e:
            print(f"ERROR - {file_name} failed: {e}")

    print("\nAll audio files filtered successfully.")


if __name__ == "__main__":
    main()
