# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 6 (0-based index in the original notebook).
# What it does: Simplified WER/SUB/DEL/INS counter without the NLP step.
# Why archived: Regressed copy of cell 5. Belongs to the separate ASR work package.
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

import os
import glob
import pandas as pd

def process_csv_files():
    """
    Reads all CSV files from a user-provided folder, counts the errors, 
    and saves a summary table in the same folder.
    """
    # 1. Ask the user to paste the folder path
    print("Example path: C:\\Users\\Name\\Desktop\\verticalized_data")
    INPUT_DIR = r"C:\Users\arday\Desktop\UROP\Project_Codes\ADReSSo_Project\data\verticalized_data"
    
    # Check if the provided folder actually exists
    if not os.path.isdir(INPUT_DIR):
        print("\nError: The folder path you provided does not exist.")
        print("Please check the path and try again.")
        return

    # 2. Set the output file to be saved in the exact same folder
    OUTPUT_CSV = os.path.join(INPUT_DIR, "Ablation_Summary.csv")
    
    # Find all .csv files in the input folder
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    # Prevent the script from reading its own summary file if you run it twice
    csv_files = [f for f in csv_files if "Ablation_Summary.csv" not in f]
    
    if not csv_files:
        print("\nNo CSV files found in:", INPUT_DIR)
        return
        
    summary_list = []
    print(f"\nFound {len(csv_files)} CSV files. Processing...")
    
    # 3. Process each file
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        clean_name = file_name.replace("_verticalized.csv", "").replace(".csv", "")
        
        # Default values
        sub = 0
        del_count = 0
        ins = 0
        wer = 0.0
        
        # Check if the file is NOT empty
        if os.path.getsize(file_path) > 0:
            try:
                df = pd.read_csv(file_path)
                
                if 'Error_Type' in df.columns:
                    sub = len(df[df['Error_Type'] == 'SUBSTITUTION'])
                    del_count = len(df[df['Error_Type'] == 'DELETION'])
                    ins = len(df[df['Error_Type'] == 'INSERTION'])
                    match_count = len(df[df['Error_Type'] == 'MATCH'])
                    
                    reference_words = match_count + sub + del_count
                    total_errors = sub + del_count + ins
                    
                    if reference_words > 0:
                        wer = (total_errors / reference_words) * 100
                        
            except pd.errors.EmptyDataError:
                pass
            except Exception as e:
                print(f"Could not read {file_name}. Error: {e}")
        
        # Append data for this file
        summary_list.append({
            "File_Name": clean_name,
            "WER (%)": round(wer, 2),
            "Substitutions": sub,
            "Deletions": del_count,
            "Insertions": ins
        })
        
    # 4. Save the final table
    final_table = pd.DataFrame(summary_list)
    final_table.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nSuccess! Processed {len(summary_list)} files.")
    print(f"The summary table is saved at: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_csv_files()
