"""
Stage 1, step 1: Cookie Theft cohort verification.

Builds the inclusion list from Daniel's CSV (COOKIE_THEFT == 1), matches each
ID to its audio file and to `uid` in metadata.csv, and runs the checks
requested before any model is trained.

Paths come from config.yaml (paths.* and stage1.*), not from this file.
"""

# ── Paths ────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import load_config, resolve_path
_cfg = load_config()   # config.yaml, merged with config.local.yaml when present
COOKIE_CSV   = str(resolve_path(_cfg["paths"]["cohort"]))
METADATA_CSV = str(resolve_path(_cfg["paths"]["metadata"]))
AUDIO_DIRS   = [str(resolve_path(_cfg["paths"]["raw_audio"][k])) for k in ("train", "test")]
OUT_DIR      = str(resolve_path(_cfg["stage1"]["data_check_dir"]))

# If the diagnosis column in Daniel's CSV is not detected automatically,
# put its exact name here (the script prints candidate columns).
DIAGNOSIS_COL = None
# ─────────────────────────────────────────────────────────────────────────

import hashlib
from pathlib import Path
import pandas as pd

# Diagnosis mapping (case-insensitive). Stated explicitly; confirm with Daniel.
AD_ADRD  = {"probablead", "ad", "adrd"}
CONTROL  = {"control", "cn", "hc"}
MCI      = {"mci"}
EXCLUDED = {"ppa"}          # distinct disease mechanism, part of neither task

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a"}
out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
report = []
def log(line=""):
    print(line); report.append(str(line))


def find_col(df, *candidates):
    """Case-insensitive exact column lookup."""
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


# ── 1. Load and filter ───────────────────────────────────────────────────
ct   = pd.read_csv(COOKIE_CSV)
meta = pd.read_csv(METADATA_CSV)

c_id    = find_col(ct, "ID", "uid")
c_ct    = find_col(ct, "COOKIE_THEFT")
c_set   = find_col(ct, "set")
c_split = find_col(ct, "split")
c_diag  = DIAGNOSIS_COL or find_col(
    ct, "diagnosis_class", "diagnosis", "dx", "diag", "group", "class", "label",
    "cohort", "condition", "cognitive_status", "status")

log("# Stage 1 data check\n")
log(f"Rows in shared CSV: {len(ct)}   Columns: {ct.shape[1]}")
log(f"Columns used: ID={c_id}, COOKIE_THEFT={c_ct}, diagnosis={c_diag}, "
    f"set={c_set}, split={c_split}")
hint = [c for c in ct.columns if any(k in c.lower() for k in
        ("diag", "dx", "group", "class", "label", "status", "cond"))]
log(f"Columns that could hold the diagnosis: {hint}")
log(f"First 25 columns: {list(ct.columns[:25])}")

ct = ct[ct[c_ct] == 1].copy()
log(f"\nRows with COOKIE_THEFT == 1: {len(ct)}")

# ── 2. Duplicate IDs ─────────────────────────────────────────────────────
dup_ids = ct[ct[c_id].duplicated(keep=False)]
log(f"\n## Duplicate IDs\n{dup_ids[c_id].nunique()} IDs appear more than once")
ct = ct.drop_duplicates(subset=c_id)

# ── 3. Match to audio files ──────────────────────────────────────────────
audio_index = {}
for d in AUDIO_DIRS:
    for p in Path(d).rglob("*"):
        if p.suffix.lower() in AUDIO_EXT:
            audio_index.setdefault(p.stem, []).append(p)

ct["audio_matches"] = ct[c_id].map(lambda u: len(audio_index.get(str(u), [])))
ct["audio_file"] = ct[c_id].map(
    lambda u: audio_index[str(u)][0].name if len(audio_index.get(str(u), [])) == 1 else "")
ct["audio_path"] = ct[c_id].map(
    lambda u: str(audio_index[str(u)][0]) if len(audio_index.get(str(u), [])) == 1 else "")

log(f"\n## Audio matching\nUnique match: {(ct['audio_matches'] == 1).sum()}")
log(f"Missing audio: {(ct['audio_matches'] == 0).sum()}")
log(f"Multiple audio files for one ID: {(ct['audio_matches'] > 1).sum()}")

# ── 4. Duplicate audio content (hash) ────────────────────────────────────
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

ct["audio_md5"] = ct["audio_path"].map(lambda p: md5(p) if p else "")
has_audio = ct[ct["audio_md5"] != ""]
dup_audio = has_audio[has_audio["audio_md5"].duplicated(keep=False)]
log(f"\n## Duplicate audio content\n{len(dup_audio)} recordings share an identical "
    f"audio file with another ID ({dup_audio['audio_md5'].nunique()} distinct files)")

# ── 4b. `hash` column vs metadata hash ──────────────────────────────────
# (filled in after the metadata merge below)

# ── 5. Match to metadata uid ─────────────────────────────────────────────
m_uid    = find_col(meta, "uid")
m_split  = find_col(meta, "split")
m_diag   = find_col(meta, "diagnosis")
m_corpus = find_col(meta, "corpus")
m_hash = find_col(meta, "hash")
keep = {m_uid: "_uid", m_split: "meta_split", m_diag: "meta_diagnosis"}
if m_hash:
    keep[m_hash] = "meta_hash"
if m_corpus:
    keep[m_corpus] = "corpus"
ct = ct.merge(meta[list(keep)].rename(columns=keep),
              left_on=c_id, right_on="_uid", how="left")
log(f"\n## Metadata match\nIDs not found in metadata.csv: {ct['_uid'].isna().sum()}")
c_hash = find_col(ct, "hash")
if c_hash and "meta_hash" in ct.columns:
    log(f"`hash` matches metadata hash: {(ct[c_hash] == ct['meta_hash']).sum()} of {len(ct)}")
    log(f"`hash` values unique: {ct[c_hash].nunique()} of {len(ct)}")

# ── 6. Partition columns: values first, then agreement ───────────────────
log("\n## Partition columns")
if c_set:
    log(f"`set` values:   {ct[c_set].value_counts(dropna=False).to_dict()}")
if c_split:
    log(f"`split` values: {ct[c_split].value_counts(dropna=False).to_dict()}")
log(f"metadata split: {ct['meta_split'].value_counts(dropna=False).to_dict()}")

if c_set and c_split:
    log("\n`set` (rows) x `split` (columns), both inside the shared CSV:")
    log(pd.crosstab(ct[c_set], ct[c_split], dropna=False).to_string())
if c_set:
    log("\n`set` (rows) x metadata split (columns):")
    log(pd.crosstab(ct[c_set], ct["meta_split"], dropna=False).to_string())
if c_split:
    n = (ct[c_split].astype(str).str.lower() != ct["meta_split"].astype(str).str.lower()).sum()
    log(f"\n`split` vs metadata split, disagreements: {n}")
if c_set:
    n = (ct[c_set].astype(str).str.lower() != ct["meta_split"].astype(str).str.lower()).sum()
    log(f"`set` vs metadata split, disagreements: {n}")
if c_set and c_split:
    n = (ct[c_set].astype(str).str.lower() != ct[c_split].astype(str).str.lower()).sum()
    log(f"`set` vs `split` inside the shared CSV, disagreements: {n}")

if c_set and c_split:
    log("\nNote: if `split` is 'train' for every row, all recordings come from the "
        "official training split and `set` is a re-partition of them "
        "(the official test recordings are not in this file).")
part_col = c_set or c_split
ct["partition"] = ct[part_col].astype(str).str.lower() if part_col else "unknown"

# ── 7. Diagnosis and class assignment ────────────────────────────────────
if c_diag:
    ct["dx"] = ct[c_diag]
    log(f"\n## Diagnosis\nSource: column `{c_diag}` in the shared CSV")
    def _coarse(label):
        l = str(label).lower()
        l = __import__("re").sub(r"^class_\d+_", "", l)
        return ("adrd" if l in AD_ADRD else "control" if l in CONTROL
                else "mci" if l in MCI else l)
    cmp = ct[ct["meta_diagnosis"].notna()]
    differ = cmp[cmp["dx"].map(_coarse) != cmp["meta_diagnosis"].map(_coarse)]
    log(f"Disagreements with metadata diagnosis (compared by meaning): {len(differ)}")
    if len(differ):
        log(differ[[c_id, "dx", "meta_diagnosis"]].to_string(index=False))
else:
    ct["dx"] = ct["meta_diagnosis"]
    log("\n## Diagnosis\nNo diagnosis column found in the shared CSV; "
        "using the diagnosis from metadata.csv instead.")

log(f"Raw labels: {ct['dx'].value_counts(dropna=False).to_dict()}")
if c_diag:
    log("\nShared-CSV class x metadata diagnosis:")
    log(pd.crosstab(ct["dx"], ct["meta_diagnosis"]).to_string())
# Daniel's labels look like class_0_control, class_1_mci, class_2_adrd
dxl = (ct["dx"].astype(str).str.lower()
       .str.replace(r"^class_\d+_", "", regex=True))
ct["dx_norm"] = dxl

def task_a(d): return "AD/ADRD" if d in AD_ADRD else "Control" if d in CONTROL else ""
def task_b(d): return ("AD/ADRD" if d in AD_ADRD else "Control" if d in CONTROL
                       else "MCI" if d in MCI else "")
ct["task_A_binary"]     = dxl.map(task_a)
ct["task_B_multiclass"] = dxl.map(task_b)

log(f"AD/ADRD = {sorted(AD_ADRD)} | Control = {sorted(CONTROL)} | "
    f"MCI = {sorted(MCI)} | Excluded = {sorted(EXCLUDED)}")
unmapped = set(dxl.dropna()) - AD_ADRD - CONTROL - MCI - EXCLUDED
if unmapped:
    log(f"WARNING: unmapped diagnosis labels: {sorted(unmapped)}")

# ── 8. Counts ────────────────────────────────────────────────────────────
usable = ct[ct["audio_matches"] == 1]
log("\n## Counts by diagnosis x partition (usable recordings)")
log(pd.crosstab(usable["dx"], usable["partition"], margins=True).to_string())
log("\nTask A (binary, MCI excluded):")
log(usable["task_A_binary"].replace("", pd.NA).value_counts().to_string())
log("\nTask B (multiclass):")
log(usable["task_B_multiclass"].replace("", pd.NA).value_counts().to_string())
if "corpus" in usable.columns:
    log("\n## Corpus composition of the Cookie Theft set")
    log(pd.crosstab(usable["corpus"], usable["dx"], margins=True).to_string())

# ── 9. Participant identity ──────────────────────────────────────────────
pid_candidates = [c for c in ct.columns
                  if any(k in c.lower() for k in
                         ("participant", "subject", "speaker", "patient", "pid"))]
log("\n## Participant identity")
if pid_candidates:
    log(f"Possible participant columns: {pid_candidates}")
    for c in pid_candidates:
        per = usable.groupby(c)[c_id].nunique()
        log(f"  {c}: {per.gt(1).sum()} participants with more than one recording")
else:
    log("No participant identifier found. Participant-level (grouped) splitting "
        "cannot be verified; each recording would have to be treated as one "
        "participant, which is only valid if one recording per person holds.")

# ── 10. Save ─────────────────────────────────────────────────────────────
cols = [c_id, "audio_file", "audio_matches", "dx", "task_A_binary",
        "task_B_multiclass", "partition", "meta_split", "audio_md5"] + pid_candidates
if "corpus" in ct.columns:
    cols.append("corpus")
ct[[c for c in cols if c in ct.columns]].to_csv(
    out / "cookie_theft_inclusion_list.csv", index=False)
(out / "data_check_report.md").write_text("\n".join(report), encoding="utf-8")
print(f"\nSaved to {out}")
