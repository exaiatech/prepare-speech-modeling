# Step 4c: Data Preparation Script for Error Analysis

### Purpose
This script prepares the ASR output data for error analysis by aligning the recognized Whisper output with the human-transcribed gold standard reference. It acts as the critical bridge between ASR generation (Step 4b) and NLP evaluation (Step 4d/4e), ensuring that corresponding words are matched perfectly, regardless of punctuation or casing differences.

### Inputs
* **Required Files:** 1. The ASR output files containing word-level timestamps.
  2. The Gold Standard reference transcripts.
* **File Formats:** `.json` (Whisper outputs) and `.txt` (Gold standard).
* **Parameters:** Automatically scans the `DIR_JSON` directory and pairs files with the corresponding `.txt` file in `DIR_TXT` based on matching filenames.

### Outputs
* **Generated Outputs:** Aligned dataset reports.
* **File Format:** `.csv` files.
* **Structure:** A standard four-column format (`Gold_Word`, `Whisper_Word`, `Start_Time`, `End_Time`). Each row aligns a spoken word with its reference, maintaining the original text casing and punctuation for qualitative review.

### Configuration
* Directory paths (`DIR_JSON`, `DIR_TXT`, `DIR_OUT`) are defined as constant variables at the top of the script.
* **Matching Algorithm Logic:** The script uses an internal `clean_for_matching` function to strip punctuation and normalize casing before using `difflib.SequenceMatcher`. This prevents false-positive errors (e.g., failing to match "cookie," with "cookie") while still preserving the original text in the final CSV output.

### Dependencies
* Built-in Python modules: `os`, `json`, `difflib`, `csv`, `string`. (No external libraries required).

### Execution Instructions
To run the script, open your terminal, navigate to the directory containing the script, and execute:
```bash
python src/Step4_ASR/Step4c_Alignment/step2_text_alignment.py