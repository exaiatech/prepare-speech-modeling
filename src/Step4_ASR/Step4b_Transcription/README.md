# Step 4b: ASR Transcription

### Purpose
This script serves as the core transcription engine of the pipeline. It processes the cleaned audio files from Step 4a and converts the clinical speech into text using the `faster-whisper` Automatic Speech Recognition (ASR) system. It strictly applies the finalized experimental methodology to capture hesitations and cognitive pauses accurately without hallucinating text.

### Inputs
* **Required Files:** Cleaned and preprocessed audio recordings from Step 4a, in `data/interim/processed_audios/`.
* **File Format:** `.wav` files.
* **Parameters:** A locally stored Whisper model (`large-v3`). The script automatically scans the input directory for all audio files.

### Outputs
* **Generated Outputs:** Standardized transcription files containing both the full text and word-level data.
* **File Format:** `.json` files.
* **Structure:** Each JSON file contains the original file name, the complete transcribed text string, and a list of individual words with their exact `start` and `end` timestamps (measured in seconds).

### Configuration
The script uses finalized baseline settings determined from previous qualitative and quantitative experiments. These are hardcoded at the top of the file:
* **VAD Settings:** `MIN_SILENCE_MS = 2000` and `SPEECH_PAD_MS = 400` to protect cognitive pauses and word edges.
* **Acoustic Filters:** `NO_SPEECH_THRESHOLD = 0.4` and `CONDITION_ON_PREVIOUS_TEXT = True` to suppress noise-induced hallucinations.
* **Prompt Engineering:** Uses a finalized "Simple Prompt" designed to force the model to transcribe verbatim speech (including stutters and hesitations) for the Cookie Theft picture description task.

### Dependencies
The following Python libraries are required:
* `faster-whisper` (for optimized and fast ASR inference)
* Built-in Python modules: `os`, `json`, `logging`, `pathlib`

> **The Whisper checkpoint is not in the repository.** `LOCAL_MODEL_PATH` points at `models/large-v3-local/`, which is gitignored because the checkpoint is 2.9 GB. Download it before running this step; see [`models/README.md`](../../../models/README.md). The script does not fall back to downloading at run time, so it fails clearly if the checkpoint is missing.

### Execution Instructions
To run the script, open your terminal, navigate to the directory containing the script, and execute:
```bash
python src/Step4_ASR/Step4b_Transcription/step2_speech_to_text.py