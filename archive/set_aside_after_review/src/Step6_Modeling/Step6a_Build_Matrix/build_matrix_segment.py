"""
Segment step 3b - Merge, clean and scale the segment-level feature tables.

Derived from notebook cell F2 of features2.ipynb ("Step 1 (segments)").
Same processing as ../build_matrix_recording.py, but each row is a segment and the
grouping keys uid and segment_idx are carried through so that later steps can
group by recording and aggregate predictions back to recording level.

INPUTS
    <segments.paths.features_dir>/{train,test}/{cn,ad}/features_segments.csv
    config.yaml   segments.matrix.*

OUTPUTS
    <segments.paths.processed_dir>/train_segments_processed.csv
    <segments.paths.processed_dir>/test_segments_processed.csv
    <segments.paths.processed_dir>/scaler_segments.pkl
    <segments.paths.processed_dir>/feature_cols_segments.pkl

PROCESSING
    Label CN=0 / AD=1, drop features with more than
    segments.matrix.max_missing_fraction missing in train, fill remaining NaNs
    with the TRAIN median, apply RobustScaler fitted on train only.

    In the poster run no feature was dropped here: all 135 survived, whereas
    the whole-recording path ends up with a different set. See
    docs/open_questions.md (Q19).

HOW TO RUN
    python Step6_Modeling/Step6a_Build_Matrix/build_matrix_segment.py
    python Step6_Modeling/Step6a_Build_Matrix/build_matrix_segment.py --config /path/to/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import DEFAULT_CONFIG, load_config, resolve_path


SCALERS = {'robust': RobustScaler}


def load_labeled(features_dir, split, label_map):
    """Loads the CN and AD segment tables of one split and labels them."""
    cn = pd.read_csv(os.path.join(features_dir, split, 'cn', 'features_segments.csv'))
    cn['label'] = label_map['cn']
    ad = pd.read_csv(os.path.join(features_dir, split, 'ad', 'features_segments.csv'))
    ad['label'] = label_map['ad']
    return pd.concat([cn, ad], ignore_index=True)


def describe_split(name, df):
    """Prints segment and recording counts of one split."""
    n_cn = df['uid'][df['label'] == 0].nunique()
    n_ad = df['uid'][df['label'] == 1].nunique()
    print(f"{name}: {len(df)} segments from {df['uid'].nunique()} recordings "
          f"(CN={n_cn}, AD={n_ad})")


def select_feature_columns(train_df, cfg):
    """Drops metadata columns and features with too much missing data in train."""
    meta_cols = ['uid', cfg['segment_column'], 'group', 'label']
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    missing_frac = train_df[feature_cols].isna().mean()
    drop_cols = missing_frac[missing_frac > cfg['max_missing_fraction']].index.tolist()
    if drop_cols:
        pct = int(cfg['max_missing_fraction'] * 100)
        print(f"\nDropping {len(drop_cols)} features with >{pct}% missing in train:")
        print(f"  {drop_cols}")
    return [c for c in feature_cols if c not in drop_cols]


def main(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    cfg = config['segments']['matrix']
    features_dir = resolve_path(config['segments']['paths']['features_dir'])
    output_dir = resolve_path(config['segments']['paths']['processed_dir'])
    seg_col = cfg['segment_column']

    os.makedirs(output_dir, exist_ok=True)

    train_df = load_labeled(features_dir, 'train', cfg['label_map'])
    test_df = load_labeled(features_dir, 'test', cfg['label_map'])
    describe_split('Train', train_df)
    describe_split('Test ', test_df)

    feature_cols = select_feature_columns(train_df, cfg)

    train_median = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(train_median)
    test_df[feature_cols] = test_df[feature_cols].fillna(train_median)

    scaler = SCALERS[cfg['scaler']]()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])

    train_out = pd.DataFrame(train_scaled, columns=feature_cols)
    train_out['uid'] = train_df['uid'].values
    train_out[seg_col] = train_df[seg_col].values
    train_out['label'] = train_df['label'].values

    test_out = pd.DataFrame(test_scaled, columns=feature_cols)
    test_out['uid'] = test_df['uid'].values
    test_out[seg_col] = test_df[seg_col].values
    test_out['label'] = test_df['label'].values

    train_path = os.path.join(output_dir, 'train_segments_processed.csv')
    test_path = os.path.join(output_dir, 'test_segments_processed.csv')
    scaler_path = os.path.join(output_dir, 'scaler_segments.pkl')
    cols_path = os.path.join(output_dir, 'feature_cols_segments.pkl')

    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, cols_path)

    print(f"\nFinal feature count: {len(feature_cols)}")
    for p in (train_path, test_path, scaler_path, cols_path):
        print(f"Saved -> {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
