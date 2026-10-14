"""
Stage 1, step 3: evaluation.

Takes ONE predictions file for ONE model and computes every metric Daniel
asked for. Same script for every model (encoder baselines, interpretable
models, hybrids later), so results are computed identically across models.

Expected input CSV, one row per recording per split (for CV) or one row per
recording (for the fixed test evaluation):

    ID, split_id, role, y_true, p_control, p_mci, p_adrd

  - split_id: which of the 50 CV splits this row belongs to (blank/omit for
    a single fixed-partition evaluation, e.g. the final test run)
  - role: "val" for a cross-validation row, "test" for the held-out test run
  - y_true: one of "Control", "MCI", "AD/ADRD"
  - p_control, p_mci, p_adrd: predicted probabilities (p_mci = 0 for
    binary-only models; the script detects binary vs 3-class automatically)

Set MODEL_NAME and TASK below; the predictions file path and the names used
for saved result files are built from these automatically, matching the
folder and file naming used by the model scripts (05_whisper_train_eval.py,
07_speechcare_train_eval.py), so nothing else needs editing.
"""

MODEL_NAME = "whisper_medium"   # base model name, e.g. "whisper_medium", "speechcare"
TASK       = "binary"           # "binary" or "multiclass" - must match the TASK used when the predictions were made

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import load_config, resolve_path
_cfg = load_config()   # config.yaml, merged with config.local.yaml when present
PREDICTIONS_CSV = str(resolve_path(_cfg["stage1"]["predictions_dir"]) / f"{MODEL_NAME}_{TASK}_predictions.csv")
OUT_DIR         = str(resolve_path(_cfg["stage1"]["results_dir"]))
CORPUS_LOOKUP   = str(resolve_path(_cfg["stage1"]["inclusion_list"]))

# internal: results are saved under MODEL_NAME_TASK so binary/multiclass
# runs, or different models, never overwrite each other
RUN_NAME = f"{MODEL_NAME}_{TASK}"

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (balanced_accuracy_score, f1_score, recall_score,
                              precision_score, confusion_matrix, roc_auc_score,
                              accuracy_score)

out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(PREDICTIONS_CSV)

CLASSES = ["Control", "MCI", "AD/ADRD"]
present = [c for c in CLASSES if c in set(df["y_true"])]
is_binary = "MCI" not in present
labels = ["Control", "AD/ADRD"] if is_binary else CLASSES

COL_FOR_CLASS = {"Control": "p_control", "MCI": "p_mci", "AD/ADRD": "p_adrd"}

def predicted_label(row):
    probs = {c: row.get(COL_FOR_CLASS[c], np.nan) for c in labels}
    probs = {k: v for k, v in probs.items() if pd.notna(v)}
    return max(probs, key=probs.get)

df["y_pred"] = df.apply(predicted_label, axis=1)

def metrics_block(sub, tag):
    y_true, y_pred = sub["y_true"], sub["y_pred"]
    out_rows = {
        "n": len(sub),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }
    if not is_binary:
        out_rows["mci_recall"] = recall_score(y_true, y_pred, labels=["MCI"],
                                               average=None, zero_division=0)[0]
    for cls in labels:
        out_rows[f"precision_{cls}"] = precision_score(y_true, y_pred, labels=[cls],
                                                        average=None, zero_division=0)[0]
        out_rows[f"recall_{cls}"] = recall_score(y_true, y_pred, labels=[cls],
                                                 average=None, zero_division=0)[0]
    if is_binary and "p_adrd" in sub.columns:
        out_rows["auc"] = roc_auc_score((y_true == "AD/ADRD").astype(int), sub["p_adrd"])
    elif not is_binary:
        # One-vs-rest, matching the metric choice confirmed in SpeechCARE's own
        # source (utils.py: roc_auc_score(..., multi_class='ovr')), for
        # comparability with the published Li et al. figures.
        # sklearn requires `labels` sorted when passed explicitly to
        # roc_auc_score(multi_class='ovr') - sort here and reorder the
        # probability columns to match, independent of CLASSES' own order.
        sorted_labels = sorted(labels)
        prob_cols = [COL_FOR_CLASS[c] for c in sorted_labels]
        if all(c in sub.columns for c in prob_cols):
            out_rows["auc_ovr"] = roc_auc_score(y_true, sub[prob_cols].values,
                                                 multi_class="ovr", labels=sorted_labels)
    return out_rows

results = []

# ── Per-split metrics (if this is a CV predictions file) ──────────────────
if "split_id" in df.columns and df["split_id"].notna().any():
    cv = df[df["role"] == "val"]
    for sid, sub in cv.groupby("split_id"):
        r = metrics_block(sub, f"split_{sid}")
        r["split_id"] = sid
        results.append(r)
    per_split = pd.DataFrame(results)
    per_split.to_csv(out / f"{RUN_NAME}_per_split.csv", index=False)

    summary = per_split.drop(columns=["split_id", "n"]).agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    summary.to_csv(out / f"{RUN_NAME}_cv_summary.csv")
    print(f"=== {RUN_NAME}: cross-validation summary ({len(per_split)} splits) ===")
    print(summary.round(3).to_string())

    cm = confusion_matrix(cv["y_true"], cv["y_pred"], labels=labels)
    print(f"\nPooled confusion matrix over all val rows (rows=true, cols=pred), labels={labels}")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

# ── Fixed test-partition evaluation (single run, no repeats) ──────────────
if "test" in set(df.get("role", [])):
    test = df[df["role"] == "test"]
    r = metrics_block(test, "test")
    pd.Series(r).to_csv(out / f"{RUN_NAME}_test_result.csv")
    print(f"\n=== {RUN_NAME}: held-out test result (n={len(test)}, evaluated once) ===")
    for k, v in r.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    cm = confusion_matrix(test["y_true"], test["y_pred"], labels=labels)
    print(f"\nConfusion matrix (rows=true, cols=pred), labels={labels}")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

# ── Corpus-wise breakdown, for comparison against the corpus-only baseline ─
try:
    lookup = pd.read_csv(CORPUS_LOOKUP)[["ID", "corpus"]]
    df2 = df.merge(lookup, on="ID", how="left")
    src = df2[df2["role"].isin(["val", "test"])]
    if src["corpus"].isna().all():
        print("\nWARNING: no IDs in the predictions file matched the corpus lookup; "
              "skipping the corpus breakdown.")
    else:
        print(f"\n=== Balanced accuracy by corpus ===")
        for corpus, sub in src.groupby("corpus"):
            if sub["y_true"].nunique() > 1:
                print(f"  {corpus:10s} n={len(sub):4d}  "
                      f"balanced_acc={balanced_accuracy_score(sub['y_true'], sub['y_pred']):.3f}")
except FileNotFoundError:
    pass

print(f"\nSaved to {out}")
