print("Loading libraries and Silero VAD model... Please wait.")

import os
import glob
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "..", "..", "data", "raw", "train_audios")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "..", "data", "interim", "processed_audios")

TARGET_SR = 16000

# Silero VAD Settings (Faster-Whisper defaults)
VAD_THRESHOLD = 0.4
MIN_SILENCE_MS = 2000
SPEECH_PAD_MS = 400

# Filter settings
LOWCUT_HZ = 75.0
HIGHCUT_HZ = 8000.0
FILTER_ORDER = 4


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs 
    low = lowcut / nyq
    high = min(highcut / nyq, 0.999) 
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, fs, lowcut, highcut, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def apply_silero_vad_silence(audio_data, sr, model, get_timestamps, threshold, min_silence_ms, speech_pad_ms):
    # Convert numpy array to PyTorch tensor
    tensor = torch.FloatTensor(audio_data)
    
    # Get speech timestamps using Silero AI
    speech_timestamps = get_timestamps(
        tensor, 
        model, 
        sampling_rate=sr,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms
    )
    
    # Create a completely silent array
    silenced_audio = np.zeros_like(audio_data)
    
    # Copy only the detected speech segments
    for segment in speech_timestamps:
        start = segment['start']
        end = segment['end']
        silenced_audio[start:end] = audio_data[start:end]
        
    return silenced_audio


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Silero VAD from PyTorch Hub
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad', 
        model='silero_vad', 
        force_reload=False,
        trust_repo=True
    )
    get_speech_timestamps, _, _, _, _ = utils
    
    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    mp3_files = glob.glob(os.path.join(INPUT_DIR, "*.mp3"))
    audio_files = wav_files + mp3_files

    if not audio_files:
        print(f"No audio files found in: {INPUT_DIR}")
        return

    print(f"Found {len(audio_files)} files. Starting processing...")

    for i, file_path in enumerate(audio_files, 1):
        file_name = os.path.basename(file_path)
        file_base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{file_base_name}.wav")
        
        print(f"[{i}/{len(audio_files)}] Processing: {file_name} ...")
        
        try:
            # 1. Load audio
            y, sr = librosa.load(file_path, sr=TARGET_SR)
            
            # 2. Filter
            y_filtered = apply_bandpass_filter(y, TARGET_SR, LOWCUT_HZ, HIGHCUT_HZ, order=FILTER_ORDER)
            
            # 3. DC offset
            y_centered = y_filtered - np.mean(y_filtered)
            
            # 4. Silero VAD Silence
            y_vad_silenced = apply_silero_vad_silence(
                y_centered, 
                TARGET_SR, 
                model, 
                get_speech_timestamps,
                threshold=VAD_THRESHOLD,
                min_silence_ms=MIN_SILENCE_MS,
                speech_pad_ms=SPEECH_PAD_MS
            )
            
            # 5. Save
            sf.write(output_path, y_vad_silenced, TARGET_SR)
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print("All files processed successfully!")

if __name__ == "__main__":
    main()