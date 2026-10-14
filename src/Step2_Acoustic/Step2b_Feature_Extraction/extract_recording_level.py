"""
Step 3 - Extraction of the 135 acoustic features (train/test, CN vs AD).

Derived from notebook cell 12 (features.ipynb). This script now holds only the
orchestration: metadata filtering, audio discovery and writing the four output
tables. The 135 feature definitions live in feature_functions.py, shared with
the segment path; the pause/fluency block lives in vad.py. Pilot
predecessors (43 and 100 features, cells 10 and 11) are in
archive/dead_code/03_extract_features/.

INPUTS
    paths.metadata        metadata.csv (uid, language, split, diagnosis, ...)
    paths.raw_audio.train raw training recordings
    paths.raw_audio.test  raw test recordings
    config.yaml           every extraction parameter (pitch floor/ceiling, MFCC
                          count, STFT sizes, spectral bands, ...)

OUTPUTS
    <paths.features_dir>/train/cn/features.csv
    <paths.features_dir>/train/ad/features.csv
    <paths.features_dir>/test/cn/features.csv
    <paths.features_dir>/test/ad/features.csv

    One row per recording: the 135 features plus uid and group.

FEATURES
    See feature_functions.py for the feature breakdown, the library choices and
    the sample-rate caveat (docs/open_questions.md Q15).

HOW TO RUN
    python Step2_Acoustic/Step2b_Feature_Extraction/extract_recording_level.py
    python Step2_Acoustic/Step2b_Feature_Extraction/extract_recording_level.py --config /path/to/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import DEFAULT_CONFIG, load_config, resolve_path
from common.feature_functions import build_index, extract_all, find_audio


def process_group(df_rows, exact, prefix_list, split, group_label, config):
    out_dir = os.path.join(resolve_path(config['paths']['features_dir']), split, group_label.lower())
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for _, r in df_rows.iterrows():
        uid = r['uid']
        path = find_audio(uid, exact, prefix_list)
        if not path:
            print(f"  [!] {split}/{group_label} - not found: {uid}")
            continue
        print(f"  [{split}/{group_label}] {uid}")
        try:
            feat = extract_all(path, config)
            feat['uid'] = uid
            feat['group'] = group_label
            rows.append(feat)
        except Exception as e:
            print(f"    ERROR ({uid}): {e}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(out_dir, 'features.csv'), index=False)
    print(f"  Saved {len(df_out)} rows -> {out_dir}/features.csv\n")
    return df_out


def main(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    fcfg = config['features']
    filters = fcfg['metadata_filters']
    audio_exts = tuple(fcfg['audio_exts'])

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
