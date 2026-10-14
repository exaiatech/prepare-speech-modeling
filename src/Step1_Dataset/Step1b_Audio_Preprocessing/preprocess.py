"""
Step 1 - Audio preprocessing: band-pass filtering and peak normalization.

STATUS AFTER THE SUPERVISOR'S REVIEW: acoustic-feature preprocessing only. This is
not a universal preparation step and must not be applied to every encoder. Each
encoder needs its own documented input, following its published method, with the
resampling, filtering, padding and truncation recorded. See
docs/encoders/encoder_input_preparation.md.

Derived from notebook cell 22 (features.ipynb).
Superseded variants live in archive/dead_code/01_preprocess/
(cell 23 = band-pass only + 16 kHz resample, cell 24 = normalization only + 16 kHz resample).

INPUTS
    paths.preprocess.input_dir   directory of .wav / .mp3 recordings
    config.yaml                  filter cutoffs, order, normalization switch

OUTPUTS
    paths.preprocess.output_dir  one .wav per input file, filtered and normalized

PROCESSING
    1. Load audio at its ORIGINAL sample rate (no resampling; preprocess.resample_to is null)
    2. Butterworth band-pass filter, 75 Hz - 8 kHz, order 4
    3. Peak normalization to a maximum absolute amplitude of 1.0
    4. Write as .wav at the original sample rate

HOW TO RUN
    python src/Step1_Dataset/Step1b_Audio_Preprocessing/preprocess.py
    python src/Step1_Dataset/Step1b_Audio_Preprocessing/preprocess.py --config /path/to/config.yaml

NOTE
    extract_recording_level.py does not currently read this stage's output; it reads
    the raw audio directories. See docs/open_questions.md (Q11).
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import DEFAULT_CONFIG, load_config, resolve_path


def butter_bandpass(lowcut, highcut, fs, order=4, max_normalized_freq=0.999):
    """
    Creates the Butterworth band-pass filter coefficients.
    """
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq

    # Ensure highcut doesn't exceed Nyquist frequency for low-quality audios
    high = min(highcut / nyq, max_normalized_freq)

    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(data, fs, lowcut, highcut, order=4, max_normalized_freq=0.999):
    """
    Applies the band-pass filter to the audio data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order,
                           max_normalized_freq=max_normalized_freq)
    y_filtered = lfilter(b, a, data)
    return y_filtered


def preprocess_audio(file_path, output_path, cfg):
    """
    Applies Band-Pass filter and then Peak Normalization.
    NO Resampling is done (keeps original sample rate).
    """
    # 1. Load audio with its ORIGINAL sample rate (sr=None)
    y, sr = librosa.load(file_path, sr=cfg["resample_to"])

    # 2. Apply Butterworth Band-Pass Filter (using the original sr)
    y_filtered = apply_bandpass_filter(
        y, sr, cfg["lowcut_hz"], cfg["highcut_hz"],
        order=cfg["filter_order"],
        max_normalized_freq=cfg["max_normalized_freq"],
    )

    # 3. Apply Peak Normalization
    if cfg["peak_normalize"]:
        max_amplitude = np.max(np.abs(y_filtered))
        if max_amplitude > 0:
            y_normalized = y_filtered / max_amplitude
        else:
            y_normalized = y_filtered  # Skip if the audio file is completely silent
    else:
        y_normalized = y_filtered

    # 4. Save the final processed file using the ORIGINAL sample rate
    sf.write(output_path, y_normalized, sr)


def collect_audio_files(input_dir, patterns):
    """Gathers every input file matching the configured glob patterns."""
    audio_files = []
    for pattern in patterns:
        audio_files.extend(glob.glob(os.path.join(input_dir, pattern)))
    return audio_files


def main(config_path=DEFAULT_CONFIG):
    """
    Main pipeline function to process all audio files.
    """
    config = load_config(config_path)
    cfg = config["preprocess"]
    input_dir = config["paths"]["preprocess"]["input_dir"]
    output_dir = config["paths"]["preprocess"]["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    audio_files = collect_audio_files(input_dir, cfg["input_globs"])

    if not audio_files:
        print(f"No audio files {cfg['input_globs']} found in: {input_dir}")
        return

    print(f"Starting processing for {len(audio_files)} files...")
    print("Steps: 1. Keep Original SR -> 2. Band-Pass Filter -> 3. Peak Normalization\n")

    for file_path in audio_files:
        file_name = os.path.basename(file_path)
        file_base_name = os.path.splitext(file_name)[0]
        output_file_name = f"{file_base_name}{cfg['output_ext']}"
        output_path = os.path.join(output_dir, output_file_name)

        try:
            preprocess_audio(file_path, output_path, cfg)
            print(f"SUCCESS - {file_name} -> filtered & normalized (Original SR)")
        except Exception as e:
            print(f"ERROR - {file_name} failed: {e}")

    print("\nAll audio files processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
