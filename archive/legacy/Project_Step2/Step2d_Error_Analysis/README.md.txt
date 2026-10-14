# Step 2d: Error Analysis & Reporting Script

### Purpose
This is the final stage of the evaluation pipeline. It takes the verticalized datasets from Step 2c and applies advanced Natural Language Processing (NLP) to categorize the clinical and linguistic impact of the Automatic Speech Recognition (ASR) errors. It calculates key performance metrics (such as Word Error Rate) and compiles everything into a comprehensive, production-ready Master Excel Report for researchers.

### Inputs
* **Required Files:** Verticalized CSV files generated from Step 2c.
* **File Format:** `.csv` files.
* **Parameters:** Automatically iterates through all CSVs located in the `INPUT_DIR`.

### Outputs
* **Generated Outputs:** A multi-sheet Master Excel Report.
* **File Format:** `.xlsx` file (`Master_Report_Final.xlsx`).
* **Structure:**
  - **1_EXECUTIVE_SUMMARY:** A high-level overview detailing Total Errors, Word Error Rate (WER), Content Word Loss, and an automated Clinical Conclusion for each audio file.
  - **2_TERMINOLOGY_LEGEND:** Definitions for the error types and linguistic categories.
  - **Detailed Sheets:** Individual sheets for every audio file, listing *every single word* alongside its timestamps, alignment error type, and NLP linguistic category.

### Configuration
* **NLP Categorization Logic:** The script uses `spacy` to perform Part-of-Speech (POS) tagging and lemmatization. It relies on hardcoded lists (`HESITATIONS`, `FUNCTION_TAGS`, `CONTENT_TAGS`) to flag specific clinical biomarkers (like pauses/stutters) and severe errors (missing nouns/verbs).

### Dependencies
* `spacy` (Requires the `en_core_web_sm` language model).
* `pandas` (for dataset analysis and manipulation).
* `xlsxwriter` (for formatting the final Excel output).
* Built-in Python modules: `os`, `glob`, `math`.

### Execution Instructions
1. Ensure the spaCy language model is installed:
   ```bash
   python -m spacy download en_core_web_sm