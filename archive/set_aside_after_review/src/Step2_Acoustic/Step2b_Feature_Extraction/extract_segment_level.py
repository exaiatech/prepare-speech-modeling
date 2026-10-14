"""
Segment step 3 - Extraction of the 135 acoustic features per 5-second segment.

Derived from notebook cell F1 of features2.ipynb ("Step 0b - Segment-Based
Feature Extraction"). The nine feature-computing functions in F1 are identical
to those of notebook cell 12 (whole-recording path); both paths now import them
from ../feature_functions.py. Only the orchestration - segmentation,
per-segment extraction, parallelism - lives here.

INPUTS
    paths.metadata                  metadata.csv
    paths.raw_audio.train / .test   raw recordings
    config.yaml                     segments.segmentation.*, features.*, vad.*

OUTPUTS
    <segments.paths.features_dir>/train/cn/features_segments.csv
    <segments.paths.features_dir>/train/ad/features_segments.csv
    <segments.paths.features_dir>/test/cn/features_segments.csv
    <segments.paths.features_dir>/test/ad/features_segments.csv

    One row per segment: the 135 features plus uid, segment_idx, group.

SEGMENTATION
    Fixed-length windows: 5.0 s long, 3.75 s hop (25% overlap, matching
    SpeechCARE's segmentation on the same PREPARE dataset). Trailing windows
    shorter than 3.0 s are discarded. Segmentation is NOT voice-activity
    based - the VAD in common/vad.py is applied INSIDE each segment to
    produce the 18 pause/fluency features, not to place segment boundaries.

DIFFERENCE FROM THE WHOLE-RECORDING PATH
    This path calls extract_all_segment(), which builds its parselmouth.Sound
    from the already-resampled 16 kHz array, while extract_all() builds one
    from the file at its original sample rate. The two pipelines therefore do
    not compute their Praat features from identical signal representations.
    See docs/open_questions.md (Q15).

HOW TO RUN
    python Step2_Acoustic/Step2b_Feature_Extraction/extract_segment_level.py
    python Step2_Acoustic/Step2b_Feature_Extraction/extract_segment_level.py --config /path/to/config.yaml
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import DEFAULT_CONFIG, load_config, resolve_path
from common.feature_functions import build_index, extract_all_segment, find_audio


def segment_starts(total_dur, seg_cfg):
    """Start times (seconds) of the fixed-length analysis windows."""
    return np.arange(0,
                     max(total_dur - seg_cfg['min_segment_len_s'], 0) + 1e-9,
                     seg_cfg['hop_len_s'])


def extract_file_segments(filepath, uid, group_label, config):
    """Load one file, split into overlapping segments, extract features
    from each. Returns a list of row-dicts (one per segment)."""
    seg_cfg = config['segments']['segmentation']
    sr = config['features']['target_sr']

    y, _ = librosa.load(filepath, sr=sr, mono=True)
    total_dur = len(y) / sr

    rows = []
    for seg_idx, start in enumerate(segment_starts(total_dur, seg_cfg)):
        end = min(start + seg_cfg['segment_len_s'], total_dur)
        if end - start < seg_cfg['min_segment_len_s']:
            continue
        y_seg = y[int(start * sr):int(end * sr)]
        try:
            feat = extract_all_segment(y_seg, config)
            feat['uid'] = uid
            feat['segment_idx'] = seg_idx
            feat['group'] = group_label
            rows.append(feat)
        except Exception as e:
            print(f"    ERROR ({uid} seg{seg_idx}): {e}")
    return rows


def process_group(df_rows, exact, prefix_list, split, group_label, config):
    """Segments and extracts every recording of one split/diagnosis group."""
    seg_cfg = config['segments']['segmentation']
    out_dir = os.path.join(resolve_path(config['segments']['paths']['features_dir']),
                           split, group_label.lower())
    os.makedirs(out_dir, exist_ok=True)

    tasks = []
    for _, r in df_rows.iterrows():
        uid = r['uid']
        path = find_audio(uid, exact, prefix_list)
        if not path:
            print(f"  [!] {split}/{group_label} - not found: {uid}")
            continue
        tasks.append((uid, path))

    print(f"  Segmenting {len(tasks)} files in parallel...")
    results = Parallel(n_jobs=seg_cfg['n_jobs'], backend=seg_cfg['backend'],
                       verbose=seg_cfg['verbose'])(
        delayed(extract_file_segments)(path, uid, group_label, config)
        for uid, path in tasks
    )
    rows = [row for file_rows in results for row in file_rows]  # flatten

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, 'features_segments.csv')
    df_out.to_csv(out_path, index=False)
    n_recordings = df_out['uid'].nunique() if len(df_out) else 0
    print(f"  Saved {len(df_out)} segments from {n_recordings} recordings -> {out_path}\n")
    return df_out


def main(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    fcfg = config['features']
    filters = fcfg['metadata_filters']
    audio_exts = tuple(fcfg['audio_exts'])

    warnings.filterwarnings("ignore",
                            message=config['segments']['segmentation']['suppress_warning'])

    folders = {
        'train': resolve_path(config['paths']['raw_audio']['train']),
        'test': resolve_path(config['paths']['raw_audio']['test']),
    }

    meta = pd.read_csv(resolve_path(config['paths']['metadata']))
    meta_eng = meta[meta['language'] == filters['language']]

    for split in filters['splits']:
        folder = folders[split]
        print(f"\n{'='*50}\n{split.upper()}\n{'='*50}")
        split_df = meta_eng[meta_eng['split'] == split]

        cn_df = split_df[split_df['diagnosis'].isin(filters['control_labels'])]
        ad_df = split_df[split_df['diagnosis'].isin(filters['ad_labels'])]

        print(f"Pool - CN: {len(cn_df)}  AD: {len(ad_df)}")
        print(f"Indexing {folder} ...")
        exact, prefix_list = build_index(folder, audio_exts)
        print(f"  {len(prefix_list)} audio files found")

        process_group(cn_df, exact, prefix_list, split, 'CN', config)
        process_group(ad_df, exact, prefix_list, split, 'AD', config)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
