# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 9 (0-based index in the original notebook).
# What it does: Age/gender-matched pilot cohort (25 pairs per gender, Hungarian assignment).
# Why archived: Superseded by 03_extract_features.py, which uses the full English subset from metadata.csv instead of a matched pilot cohort.
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

import os
import shutil
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

# ── Paths ──────────────────────────────────────────────
METADATA     = r"C:\Users\arday\Downloads\metadata.csv"
TRAIN_FOLDER = r"C:\Users\arday\Downloads\train_audios"
OUTPUT       = r"C:\Users\arday\Desktop\pilot_dataset_2"
# ───────────────────────────────────────────────────────

df = pd.read_csv(METADATA)
df['age'] = pd.to_numeric(df['age'], errors='coerce')

eng = df[(df['language'] == 'english') & (df['split'] == 'train') & df['age'].notna()]
ctrl = eng[eng['diagnosis'] == 'Control'].reset_index(drop=True)
ad   = eng[eng['diagnosis'].isin(['ProbableAD', 'AD'])].reset_index(drop=True)

print(f"Pool — Control: {len(ctrl)}  AD: {len(ad)}")

sel_ctrl, sel_ad, pairs = [], [], []

for gender in ['female', 'male']:
    cg = ctrl[ctrl['gender'] == gender].reset_index(drop=True)
    ag = ad[ad['gender'] == gender].reset_index(drop=True)
    if len(cg) == 0 or len(ag) == 0:
        continue
    cost = np.abs(cg['age'].values[:, None] - ag['age'].values[None, :])
    ri, ci = linear_sum_assignment(cost)
    ranked = sorted(zip(ri, ci, cost[ri, ci]), key=lambda x: x[2])
    for r, c, diff in ranked[:25]:
        sel_ctrl.append(cg.iloc[r].to_dict())
        sel_ad.append(ag.iloc[c].to_dict())
        pairs.append({'ctrl': cg.iloc[r]['uid'], 'ctrl_age': int(cg.iloc[r]['age']),
                      'ad':   ag.iloc[c]['uid'], 'ad_age':   int(ag.iloc[c]['age']),
                      'diff': int(diff), 'gender': gender})

df_ctrl = pd.DataFrame(sel_ctrl)
df_ad   = pd.DataFrame(sel_ad)
df_pairs = pd.DataFrame(pairs)

print(f"\nSelected — Control: {len(df_ctrl)}  AD: {len(df_ad)}")
print(f"Control  age: {df_ctrl['age'].mean():.1f} ± {df_ctrl['age'].std():.1f}  gender: {df_ctrl['gender'].value_counts().to_dict()}")
print(f"AD       age: {df_ad['age'].mean():.1f} ± {df_ad['age'].std():.1f}  gender: {df_ad['gender'].value_counts().to_dict()}")
print(f"Mean age diff per pair: {df_pairs['diff'].mean():.1f} yrs")
print(f"\nPairs:\n{df_pairs.to_string(index=False)}")

os.makedirs(os.path.join(OUTPUT, 'control'),   exist_ok=True)
os.makedirs(os.path.join(OUTPUT, 'alzheimer'), exist_ok=True)

def copy_audio(uid, dest):
    for root, _, files in os.walk(TRAIN_FOLDER):
        for f in files:
            if f.startswith(uid) and f.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                shutil.copy2(os.path.join(root, f), os.path.join(dest, f))
                return True
    print(f"  [!] Not found: {uid}")
    return False

for row in sel_ctrl: copy_audio(row['uid'], os.path.join(OUTPUT, 'control'))
for row in sel_ad:   copy_audio(row['uid'], os.path.join(OUTPUT, 'alzheimer'))

df_ctrl.to_csv(os.path.join(OUTPUT, 'control_metadata.csv'),   index=False)
df_ad.to_csv(os.path.join(OUTPUT,   'alzheimer_metadata.csv'), index=False)
df_pairs.to_csv(os.path.join(OUTPUT, 'pairs.csv'),             index=False)

print(f"\nDone → {OUTPUT}/")
