import os
import glob
import pandas as pd
import string

# --- CONFIGURATION ---
INPUT_DIR = r"C:\Users\arday\Desktop\UROP\TEST\Second\alignment_reports"
OUTPUT_DIR = r"C:\Users\arday\Desktop\UROP\TEST\Second\verticalized_data"


def clean_text_for_comparison(word):
    """
    Removes punctuation and normalizes casing to accurately determine error types.
    """
    if pd.isna(word): 
        return ""
    text = str(word).lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    if text == "unk": 
        return ""
    return text


def get_alignment_metadata(row):
    """
    Compares the Gold Standard and Whisper words to assign a basic Error Type.
    """
    gold = clean_text_for_comparison(row.get('Gold_Word', ''))
    whisper = clean_text_for_comparison(row.get('Whisper_Word', ''))
    
    if gold == "" and whisper != "": 
        return "INSERTION"
    elif gold != "" and whisper == "": 
        return "DELETION"
    elif gold == whisper: 
        return "MATCH"
    else: 
        return "SUBSTITUTION"


def main():
    """
    Main pipeline function for Step 4.
    Reads aligned CSVs, adds alignment metadata, and creates verticalized datasets.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Use recursive=True if your files are inside experiment subfolders
    csv_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.csv"), recursive=True)
    print(f"Found {len(csv_files)} aligned files. Starting verticalization...\n")
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        audio_name = file_name.split("_")[0]
        
        try:
            df = pd.read_csv(file_path)
            
            # Clean up: Remove audio ID from the first row if present (From your original code)
            first_gold = str(df.loc[0, 'Gold_Word']) if 'Gold_Word' in df.columns else ""
            if audio_name in first_gold or "adrs" in first_gold:
                df = df.iloc[1:].reset_index(drop=True)
            
            # Add Alignment Metadata (MATCH, DELETION, INSERTION, SUBSTITUTION)
            df['Error_Type'] = df.apply(get_alignment_metadata, axis=1)
            
            # Reorder columns to match Daniel's requirement
            cols = ['Gold_Word', 'Whisper_Word', 'Start_Time', 'End_Time', 'Error_Type']
            df_vertical = df[[c for c in cols if c in df.columns]]
            
            # Save the verticalized dataset
            out_path = os.path.join(OUTPUT_DIR, f"{audio_name}_verticalized.csv")
            df_vertical.to_csv(out_path, index=False, encoding='utf-8')
            
            print(f"SUCCESS - Verticalized: {audio_name}_verticalized.csv")
            
        except Exception as e:
            print(f"ERROR - Could not process {file_name}: {e}")

    print("\nVerticalization pipeline finished. Data is ready for Step 5 (Error Analysis).")


if __name__ == "__main__":
    main()