# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 5 (0-based index in the original notebook).
# What it does: spaCy-based linguistic categorisation of ASR errors + WER summary.
# Why archived: Belongs to the separate ASR work package (asr_pipeline).
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

import os
import glob
import pandas as pd
import spacy
import warnings

# Hide unnecessary warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "..", "data", "verticalized_data")
OUTPUT_CSV = os.path.join(BASE_DIR, "..", "..", "data", "Ablation_Summary.csv")

# --- NLP Settings ---
HESITATIONS = ["uh", "um", "ah", "er", "hmm", "well", "hes"]
FUNCTION_TAGS = ["DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ", "PART"]
CONTENT_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]


def get_linguistic_category(row, nlp_engine):
    """
    Applies NLP to categorize the linguistic impact of the word.
    """
    error_type = row.get('Error_Type', '')
    gold = str(row.get('Gold_Word', '')).lower()
    whisper = str(row.get('Whisper_Word', '')).lower()
    
    if error_type == "MATCH":
        return "Correctly_Recognized"
        
    target = gold if error_type in ["DELETION", "SUBSTITUTION"] else whisper
    
    if not target or target == "nan": 
        return "Other"
        
    if target in HESITATIONS or whisper in HESITATIONS: 
        return "Hesitation_Marker"
        
    doc = nlp_engine(target)
    target_lemma = doc[0].lemma_
    target_pos = doc[0].pos_
    
    if error_type == "SUBSTITUTION":
        doc_w = nlp_engine(whisper)
        whisper_lemma = doc_w[0].lemma_ if len(doc_w) > 0 else ""
        if target_lemma == whisper_lemma and target_lemma != "":
            return "Morphological_Deviation"
            
    if target_pos in FUNCTION_TAGS: 
        return "Function_Word"
    if target_pos in CONTENT_TAGS: 
        return "Content_Word"
        
    return "Other"


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Cannot find input directory at {INPUT_DIR}")
        return

    print("Loading spaCy NLP engine (en_core_web_sm)...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("ERROR: spaCy model not found. Please run: python -m spacy download en_core_web_sm")
        return

    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        print(f"No verticalized CSV files found in {INPUT_DIR}.")
        return

    print(f"Starting Error Analysis for {len(csv_files)} files...\n")
    
    summary_data = [] 
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        audio_name = file_name.replace("_verticalized.csv", "")
        
        # 1. Skip completely empty files to prevent EmptyDataError
        if os.path.getsize(file_path) == 0:
            print(f"Skipping {file_name}: File is completely empty.")
            continue
            
        try:
            # 2. Read the valid CSV file
            df = pd.read_csv(file_path)
            
            # 3. Check if dataframe is empty or missing required columns
            if df.empty or 'Error_Type' not in df.columns:
                print(f"Skipping {file_name}: Invalid format or no data rows.")
                continue
            
            df['Linguistic_Category'] = df.apply(lambda row: get_linguistic_category(row, nlp), axis=1)
            
            sub_count = len(df[df['Error_Type'] == 'SUBSTITUTION'])
            del_count = len(df[df['Error_Type'] == 'DELETION'])
            ins_count = len(df[df['Error_Type'] == 'INSERTION'])
            
            reference_words = len(df[df['Error_Type'].isin(['MATCH', 'DELETION', 'SUBSTITUTION'])])
            total_errors = sub_count + del_count + ins_count
            
            wer = (total_errors / reference_words) * 100 if reference_words > 0 else 0
            
            summary_data.append({
                "File_Name": audio_name,
                "WER (%)": round(wer, 2),
                "Substitutions": sub_count,
                "Deletions": del_count,
                "Insertions": ins_count
            })
            
        # Catch any unexpected pandas empty data errors just in case
        except pd.errors.EmptyDataError:
            print(f"Skipping {file_name}: pandas found no columns to parse.")
        except Exception as e:
            print(f"File {file_name} could not be processed. Error: {e}")

    if not summary_data:
        print("No valid data found to summarize.")
        return

    # Create the final dataframe and save to CSV
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n[SUCCESS] Pipeline Complete!")
    print(f"Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
