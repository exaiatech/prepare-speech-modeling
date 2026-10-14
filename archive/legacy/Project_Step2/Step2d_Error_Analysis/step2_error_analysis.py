import os
import glob
import pandas as pd
import spacy
import math

# --- CONFIGURATION ---
# Input is the output directory of Step 4
INPUT_DIR = r"C:\Users\arday\Desktop\UROP\TEST\Second\verticalized_data"
OUTPUT_EXCEL = r"C:\Users\arday\Desktop\UROP\TEST\Second\Master_Report_Final.xlsx"

# NLP Settings
HESITATIONS = ["uh", "um", "ah", "er", "hmm", "well", "hes"]
FUNCTION_TAGS = ["DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ", "PART"]
CONTENT_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]


def get_linguistic_category(row, nlp_engine):
    """
    Applies NLP to categorize the linguistic impact of the word or error.
    """
    error_type = row.get('Error_Type', '')
    gold = str(row.get('Gold_Word', '')).lower()
    whisper = str(row.get('Whisper_Word', '')).lower()
    
    # If it's a perfect match, no need for error categorization
    if error_type == "MATCH":
        return "Correctly_Recognized"
        
    # Determine which word to analyze based on the error type
    target = gold if error_type in ["DELETION", "SUBSTITUTION"] else whisper
    
    if not target or target == "nan": 
        return "Other"
        
    # Check for clinical biomarkers (hesitations)
    if target in HESITATIONS or whisper in HESITATIONS: 
        return "Hesitation_Marker"
        
    # Apply spaCy NLP
    doc = nlp_engine(target)
    target_lemma = doc[0].lemma_
    target_pos = doc[0].pos_
    
    # Check for minor morphological errors (e.g., boy vs boys)
    if error_type == "SUBSTITUTION":
        doc_w = nlp_engine(whisper)
        whisper_lemma = doc_w[0].lemma_ if len(doc_w) > 0 else ""
        if target_lemma == whisper_lemma and target_lemma != "":
            return "Morphological_Deviation"
            
    # Check word importance
    if target_pos in FUNCTION_TAGS: 
        return "Function_Word"
    if target_pos in CONTENT_TAGS: 
        return "Content_Word"
        
    return "Other"


def main():
    """
    Main pipeline function for Step 5.
    Calculates performance metrics and generates the final Master Excel Report.
    """
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
    sheet_data = {}
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        audio_name = file_name.replace("_verticalized.csv", "")
        
        df = pd.read_csv(file_path)
        
        # Apply NLP categorization to all rows
        df['Linguistic_Category'] = df.apply(lambda row: get_linguistic_category(row, nlp), axis=1)
        
        # --- METRICS CALCULATION ---
        # Total reference words = Matches + Deletions + Substitutions (Insertions don't count in base reference)
        reference_words = len(df[df['Error_Type'].isin(['MATCH', 'DELETION', 'SUBSTITUTION'])])
        
        total_errors = len(df[df['Error_Type'] != 'MATCH'])
        content_errors = len(df[df['Linguistic_Category'] == 'Content_Word'])
        
        # Word Error Rate (WER) Calculation
        wer = (total_errors / reference_words) * 100 if reference_words > 0 else 0
        accuracy = max(0, 100 - wer) # Simple accuracy representation
        
        # Automatic Clinical Conclusion
        if content_errors > 15:
            conclusion = "⚠️ HIGH IMPACT: Many core words lost. Meaning may be altered."
        elif content_errors > 5:
            conclusion = "Moderate Impact: Acceptable, but some content words missed."
        else:
            conclusion = "✅ CLINICALLY SAFE: Errors are mostly harmless grammar/hesitations."
            
        summary_data.append({
            "Audio File": audio_name,
            "Total Ref Words": reference_words,
            "Total Errors": total_errors,
            "Content Word Errors": content_errors,
            "WER (%)": round(wer, 2),
            "Accuracy (%)": round(accuracy, 2),
            "Clinical Conclusion": conclusion
        })
        
        # Save sheet data (keep name under 31 chars for Excel)
        sheet_name = audio_name[:30]
        sheet_data[sheet_name] = df

    # --- GENERATE EXCEL REPORT ---
    print("Writing Master Report to Excel...")
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='xlsxwriter') as writer:
        
        # 1. Executive Summary
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='1_EXECUTIVE_SUMMARY', index=False)
        worksheet = writer.sheets['1_EXECUTIVE_SUMMARY']
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:F', 18)
        worksheet.set_column('G:G', 60)
        
        # 2. Terminology Legend
        legend_data = {
            "Term": ["MATCH / Correctly_Recognized", "DELETION", "INSERTION", "SUBSTITUTION", "Function_Word", "Content_Word", "Hesitation_Marker", "Morphological_Deviation"],
            "Meaning": [
                "Word was correctly transcribed.",
                "Word was missed/ignored by Whisper.",
                "Whisper hallucinated/invented a word that wasn't there.",
                "Whisper heard the word incorrectly (replaced it).",
                "Harmless grammar words (the, a, is, on). Doesn't alter meaning.",
                "CRITICAL: Core nouns, verbs, adjectives. Changes clinical meaning!",
                "Biomarker: Patient stutters or pauses (um, uh). Very important for dementia.",
                "Minor error: Root is correct, but suffix is wrong (boy -> boys)."
            ]
        }
        pd.DataFrame(legend_data).to_excel(writer, sheet_name='2_TERMINOLOGY_LEGEND', index=False)
        worksheet_legend = writer.sheets['2_TERMINOLOGY_LEGEND']
        worksheet_legend.set_column('A:A', 30)
        worksheet_legend.set_column('B:B', 80)
        
        # 3. Detailed Audio Sheets
        for sheet_name, df_detail in sheet_data.items():
            df_detail.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet_det = writer.sheets[sheet_name]
            worksheet_det.set_column('A:B', 15) # Words
            worksheet_det.set_column('C:D', 12) # Timestamps
            worksheet_det.set_column('E:F', 25) # Categories
            
    print(f"\n[SUCCESS] Pipeline Complete! Final Master Report generated at: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()