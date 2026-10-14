# Step 1A: Audio Preprocessing Script

### Purpose
This script handles the initial cleaning and standardization of raw clinical audio files. It sits at the very beginning of the pipeline (Step 1). Its primary goal is to ensure that the Automatic Speech Recognition (ASR) system receives consistent, reproducible, and clean acoustic data by removing dead silence and centering the audio signal.

### Inputs
* **Required Files:** Raw audio recordings.
* **File Format:** `.wav` files.
* **Parameters:** The script automatically scans the designated input directory for any `.wav` files. 

### Outputs
* **Generated Outputs:** Cleaned and standardized audio files.
* **File Format:** `.wav` files (16kHz, mono).
* **Structure:** The processed files are saved into the specified output directory using their exact original filenames to maintain consistency for downstream alignment.

### Configuration
The script uses constant variables at the top of the file for easy adjustment. Default settings include:
* `INPUT_DIR`: Path to the raw audio files (Default: `C:\Users\arday\Desktop\Old_Audio`).
* `OUTPUT_DIR`: Path to save the processed files (Default: `C:\Users\arday\Desktop\New_Audio`).
* `TARGET_SR`: The target sample rate required by the Whisper ASR model (Default: `16000` Hz).
* `TRIM_TOP_DB`: The decibel threshold used to detect and trim silence at the beginning and end of the audio (Default: `30`).

### Dependencies
The following Python libraries are required to run this script:
* `librosa` (for audio loading, resampling, and silence trimming)
* `soundfile` (for exporting the processed audio)
* `numpy` (for mathematical operations like DC offset removal)
* Built-in Python modules: `os`, `glob`

### Execution Instructions
To run the script, open your terminal or command prompt, navigate to the directory containing the script, and execute the following command:
```bash
python step1a_audio_preprocess.py