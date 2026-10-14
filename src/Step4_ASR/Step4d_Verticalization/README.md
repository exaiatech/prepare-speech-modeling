# Step 4d: Verticalization Script

### Purpose
This script bridges the gap between raw data alignment (Step 4c) and advanced error analysis (Step 4e). It converts the aligned word-pairs into a structured, row-wise vertical format. Critically, it evaluates each row to assign basic alignment metadata (MATCH, DELETION, INSERTION, SUBSTITUTION) without performing complex linguistic analysis, creating a clean dataset perfectly formatted for the final evaluation stage.

### Inputs
* **Required Files:** Aligned transcripts generated from Step 4c.
* **File Format:** `.csv` files.
* **Parameters:** Automatically scans the defined `INPUT_DIR` (including subfolders) for any aligned CSV files.

### Outputs
* **Generated Outputs:** Verticalized datasets ready for NLP and WER analysis.
* **File Format:** `.csv` files (`[audio_name]_verticalized.csv`).
* **Structure:** Each row represents a single word event. The columns strictly include:
  1. `Gold_Word` (The reference word)
  2. `Whisper_Word` (The recognized word)
  3. `Start_Time` & `End_Time` (Timestamps)
  4. `Error_Type` (Alignment metadata)

### Configuration
* Input and output directories are defined at the top of the script.
* **Metadata Logic:** Uses a robust text-cleaning helper function (`clean_text_for_comparison`) that temporarily strips punctuation and normalizes casing to accurately assign error types (e.g., ensuring "cookie," and "cookie" register as a `MATCH` rather than a `SUBSTITUTION`).

### Dependencies
* `pandas` (for fast and efficient data frame manipulation)
* Built-in Python modules: `os`, `glob`, `string`.

### Execution Instructions
To run the script, open your terminal, navigate to the directory containing the script, and execute:
```bash
python src/Step4_ASR/Step4d_Verticalization/step2_verticalization.py