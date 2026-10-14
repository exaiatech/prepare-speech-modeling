"""
SpeechCARE-AGF, script 06: prepare segment tensors, demography and text lookups.

This REPLACES the repo's own `preprocessing/` pipeline, which we could not access
(GitHub's directory listing for that folder is blocked). Everything here follows
SPEECHCARE_SPEC.md Section B/B2 (confirmed from model.py/dataset.py/utils.py) except
for two points flagged explicitly below, which are our own reconstruction:

  ASSUMPTION 1 (segmentation): 5 s windows, 20% overlap (hop = 4.0 s), 16 kHz, raw
  waveform saved per segment. The 20% figure is confirmed from `preprocess_audio`'s
  default in model.py; the exact tensor CONTENT (raw waveform vs. pre-extracted
  features) is not confirmed, since the real preprocessing/ code is unseen. We save
  raw waveform and apply Wav2Vec2FeatureExtractor at collate time in 07_speechcare_train_eval.py, since
  main.py's Trainer loads the feature_extractor separately from the dataset and
  passes it into get_dataloaders - consistent with feature extraction happening at
  batch-construction time rather than being baked into saved tensors.

  ASSUMPTION 2 (age_bin): the literal code only tells us `config.demography` is a
  column name, not how it's computed. We compute it ourselves using the 3-category
  scheme stated in the companion paper (arXiv:2601.16989): 46-65 -> 0, 66-80 -> 1,
  80+ -> 2.

Both are documented as reconstructions, not confirmed identical to the original.
"""

# ── Paths ───────────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import load_config, resolve_path
_cfg = load_config()   # config.yaml, merged with config.local.yaml when present
COOKIE_CSV      = str(resolve_path(_cfg["paths"]["cohort"]))
INCLUSION_LIST  = str(resolve_path(_cfg["stage1"]["inclusion_list"]))
AUDIO_DIRS      = [str(resolve_path(_cfg["paths"]["raw_audio"][k])) for k in ("train", "test")]
OUT_DIR         = str(resolve_path(_cfg["stage1"]["speechcare_data_dir"]))

SAMPLE_RATE      = 16000
SEGMENT_LENGTH_S = 5.0
OVERLAP          = 0.20          # confirmed default from model.py's preprocess_audio
MIN_TAIL_S       = 5.0           # trailing remainder kept only if >= this long (code default)
# ──────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch

out = Path(OUT_DIR)
(out / "segments").mkdir(parents=True, exist_ok=True)

# ── 1. Cohort + demography + text ─────────────────────────────────────────
incl = pd.read_csv(INCLUSION_LIST)
incl = incl[incl["audio_matches"] == 1].reset_index(drop=True)

cookie = pd.read_csv(COOKIE_CSV, low_memory=False)
age_col = "age" if "age" in cookie.columns else None
assert age_col, "age column not found in Cookie Theft CSV"
text_col = "text" if "text" in cookie.columns else None
assert text_col, "text/transcript column not found in Cookie Theft CSV"

demo = cookie[["ID", age_col]].rename(columns={"ID": "uid", age_col: "age"})

def age_to_bin(age):
    if pd.isna(age):
        return np.nan
    if age <= 65:
        return 0
    if age <= 80:
        return 1
    return 2

demo["age_bin"] = demo["age"].apply(age_to_bin)
n_missing = demo["age_bin"].isna().sum()
if n_missing:
    print(f"WARNING: {n_missing} recordings have no age; age_bin left as NaN, "
          f"these rows should be excluded or imputed before training.")
demo.to_csv(out / "demography.csv", index=False)
print(f"demography.csv: {len(demo)} rows, bins: "
      f"{demo['age_bin'].value_counts(dropna=False).to_dict()}")

text_df = cookie[["ID", text_col]].rename(columns={"ID": "uid", text_col: "transcription"})
text_df.to_csv(out / "text.csv", index=False)
print(f"text.csv: {len(text_df)} rows")

# ── 2. Audio segmentation ──────────────────────────────────────────────────
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a"}
audio_index = {}
for d in AUDIO_DIRS:
    for p in Path(d).rglob("*"):
        if p.suffix.lower() in AUDIO_EXT:
            audio_index[p.stem] = p

done_log = out / "_segments_done.json"
done = set(json.loads(done_log.read_text())) if done_log.exists() else set()

seg_step = SEGMENT_LENGTH_S * (1 - OVERLAP)   # 4.0 s hop at 20% overlap


def segment_audio(path):
    """Mirrors model.py's calculate_num_segments / preprocess_audio logic
    (confirmed defaults: 5 s, 20% overlap, 16 kHz), reimplemented with librosa
    instead of torchaudio (consistent with the rest of this project; avoids
    the torchcodec dependency issue hit earlier)."""
    wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    duration = len(wav) / SAMPLE_RATE

    seg_samples = int(SEGMENT_LENGTH_S * SAMPLE_RATE)
    step_samples = int(seg_step * SAMPLE_RATE)

    segments = []
    start = 0
    while start + seg_samples <= len(wav):
        segments.append(wav[start:start + seg_samples])
        start += step_samples

    # trailing remainder, kept only if long enough (code's min_acceptable=5s default)
    remainder = wav[start:]
    if len(remainder) / SAMPLE_RATE >= MIN_TAIL_S:
        segments.append(remainder)

    if not segments and duration > 0:
        # recording shorter than one segment: use the whole thing
        segments = [wav]

    return segments


rows = []
uids = incl["ID"].tolist()
print(f"{len(uids)} recordings to segment; {len(done)} already done (resuming)")

for i, uid in enumerate(uids):
    if uid in done:
        continue
    path = audio_index.get(uid)
    if path is None:
        print(f"  [!] no audio for {uid}, skipping")
        continue
    try:
        segs = segment_audio(path)
        for seg_idx, seg in enumerate(segs):
            tensor_path = out / "segments" / f"{uid}_{seg_idx}.pt"
            torch.save(torch.tensor(seg, dtype=torch.float32), tensor_path)
            rows.append({"uid": uid, "segment": seg_idx, "path": str(tensor_path),
                         "n_segments_total": len(segs)})
        done.add(uid)
        if (i + 1) % 25 == 0:
            done_log.write_text(json.dumps(sorted(done)))
            print(f"  {i+1}/{len(uids)}  ({uid}: {len(segs)} segments)")
    except Exception as e:
        print(f"  [!] failed on {uid}: {e}")

done_log.write_text(json.dumps(sorted(done)))

# rebuild the full segments index from whatever is on disk (handles resumed runs
# where `rows` only has this session's new segments)
all_rows = []
for p in sorted((out / "segments").glob("*.pt")):
    stem = p.stem
    uid, seg_idx = stem.rsplit("_", 1)
    all_rows.append({"uid": uid, "segment": int(seg_idx), "path": str(p)})

segments_df = pd.DataFrame(all_rows)
if segments_df.empty:
    raise SystemExit(
        "No segments were produced for any recording (segments.csv would be empty). "
        "Check the [!] failed/skipped lines above for the actual cause before rerunning."
    )
segments_df.to_csv(out / "segments.csv", index=False)

counts = segments_df.groupby("uid").size()
print(f"\nsegments.csv: {len(segments_df)} segments from {segments_df['uid'].nunique()} recordings")
print(f"segments per recording: min {counts.min()}, max {counts.max()}, "
      f"mean {counts.mean():.2f}")
print(f"max_num_segments observed = {counts.max()}  "
      f"(use this for the model's positional-encoding size in script 07)")

print(f"\nAll outputs -> {out}")
