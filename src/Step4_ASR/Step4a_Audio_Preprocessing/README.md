# Step 4a: Audio Preprocessing for ASR

### Purpose
This script handles the initial cleaning and standardization of raw clinical audio files. It is the first stage of the ASR branch (Step 4). Its primary goal is to ensure that the Automatic Speech Recognition (ASR) system receives consistent, reproducible, and clean acoustic data by removing dead silence and centering the audio signal.

### Inputs
* **Required Files:** Raw audio recordings.
* **File Format:** `.wav` and `.mp3` files.
* **Parameters:** The script scans `data/raw/train_audios/` for any `.wav` or `.mp3` file.

### Outputs
* **Generated Outputs:** Cleaned and standardized audio files.
* **File Format:** `.wav` files (16kHz, mono).
* **Structure:** The processed files are saved into the specified output directory using their exact original filenames to maintain consistency for downstream alignment.

### Configuration
The script uses constant variables at the top of the file for easy adjustment. Paths are derived from the script's own location, so they work from any clone:
* `INPUT_DIR`: `data/raw/train_audios/`, resolved relative to this script.
* `OUTPUT_DIR`: `data/interim/processed_audios/`, resolved relative to this script.
* `TARGET_SR`: The target sample rate required by the Whisper ASR model (Default: `16000` Hz).
* **Band-pass filter:** `LOWCUT_HZ = 75.0`, `HIGHCUT_HZ = 8000.0`, `FILTER_ORDER = 4`.
* **Silero VAD:** `VAD_THRESHOLD = 0.4`, `MIN_SILENCE_MS = 2000`, `SPEECH_PAD_MS = 400`. These match the faster-whisper defaults used in Step 4b, so the two stages agree on what counts as speech.

> Silence is **zeroed in place, not cut out.** Non-speech regions are replaced with silence while the timeline is preserved, so the word timestamps Step 4b produces still line up with the original recording. This matters because Step 4c aligns on those timestamps.

### Dependencies
The following Python libraries are required to run this script:
* `librosa` (for audio loading and resampling)
* `soundfile` (for exporting the processed audio)
* `numpy` (for mathematical operations like DC offset removal)
* `scipy` (for the Butterworth band-pass filter)
* `torch` (Silero VAD is fetched from `torch.hub` on first run, which needs network access once)
* Built-in Python modules: `os`, `glob`

### Execution Instructions
To run the script, open your terminal or command prompt, navigate to the directory containing the script, and execute the following command:
```bash
python src/Step4_ASR/Step4a_Audio_Preprocessing/step1a_audio_preprocess.py
```

> Not to be confused with `Step1_Dataset/Step1b_Audio_Preprocessing/`. That one belongs to the acoustic branch, uses a different method, and is not on any current path. This script is the one the ASR branch depends on.