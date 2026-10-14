# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 3 (0-based index in the original notebook).
# What it does: WER report comparing Whisper JSON output against gold-standard transcripts.
# Why archived: Belongs to the separate ASR work package (asr_pipeline), not to the acoustic pipeline. WARNING: the source is TRUNCATED mid-statement in the notebook and will not run as-is.
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

import os
import glob
import json
import re
import string
import difflib
import spacy

# --- CONFIGURATION ---
# Sadece klasör yollarını veriyoruz, arama ifadelerini sildik.
GOLD_DIR = r"C:\Users\arday\Desktop\UROP\Project_Codes\ADReSSo Project\data\gold_standard"
WHISPER_DIR = r"C:\Users\arday\Desktop\UROP\Project_Codes\ADReSSo Project\data\transcripts"
OUTPUT_REPORT = r"C:\Users\arday\Desktop\UROP\Project_Codes\ADReSSo Project\data\Wiki_Report.md"

HESITATIONS = ["uh", "um", "ah", "er", "hmm", "well", "hes", "eee", "aaa", "iii", "ehm"]
CONTENT_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]

def clean_word(word):
    """Removes punctuation and makes text lowercase."""
    return word.translate(str.maketrans('', '', string.punctuation)).lower().strip()

def main():
    print("Loading spaCy NLP engine...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("ERROR: Run 'python -m spacy download en_core_web_sm' first.")
        return

    # Iterate through Whisper outputs (JSONs) instead of Gold Standard (TXTs)
    json_files = glob.glob(os.path.join(WHISPER_DIR, "*.json"))
    if not json_files:
        print("No Whisper JSON files found.")
        return

    results = []

    for json_path in json_files:
        file_name = os.path.basename(json_path)
        base_name = os.path.splitext(file_name)[0]

        # 1. Read Whisper JSON
        whisper_words = []
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data.get("words", []):
                whisper_words.append(item.get("word", ""))

        # 2. Try to Read Gold Standard TXT
        txt_path = os.path.join(GOLD_DIR, f"{base_name}.txt")
        gold_words = []
        has_gold = False
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    has_gold = True
                    parts = re.split(r'\t+', content, maxsplit=1)
                    gold_text = parts[1].strip() if len(parts) == 2 else content
                    gold_words = gold_text.split()

        # 3. Clean Words
        gold_clean = [clean_word(w) for w in gold_words]
        whisper_clean = [clean_word(w) for w in whisper_words]

        # 4. Handle Missing/Empty Gold Standard Files
        if not has_gold:
            if len(whisper_clean) == 0:
                results.append({
                    "name": base_name,
                    "wer": 0.0,
                    "status": "No Gold Standard - Model correctly stayed silent."
