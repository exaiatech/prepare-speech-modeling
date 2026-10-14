"""
Step 3b - Merge, clean and scale the feature tables into model-ready matrices.

Derived from notebook cell 13 (features.ipynb), "Step 1 - Merge, Clean, and
Scale Feature Data". It sits between feature extraction (03) and feature
selection (04); the target structure had no slot for it, so it is numbered 03b.

INPUTS
    <paths.features_dir>/train/cn/features.csv
    <paths.features_dir>/train/ad/features.csv
    <paths.features_dir>/test/cn/features.csv
    <paths.features_dir>/test/ad/features.csv
    config.yaml   label map, missing-data threshold, scaler choice

OUTPUTS
    <paths.processed_dir>/train_processed.csv
    <paths.processed_dir>/test_processed.csv
    <paths.processed_dir>/scaler.pkl        fitted RobustScaler (reused downstream)
    <paths.processed_dir>/feature_cols.pkl  final feature name list

PROCESSING
    Merges train/cn + train/ad into one labeled train set and test/cn + test/ad
    into one labeled test set (CN=0, AD=1). Drops features with more than
    matrix.max_missing_fraction missing values in train, fills the remaining
    NaNs with the TRAIN median, and applies RobustScaler fitted on train only.

HOW TO RUN
    python Step6_Modeling/Step6a_Build_Matrix/build_matrix_recording.py
    python Step6_Modeling/Step6a_Build_Matrix/build_matrix_recording.py --config /path/to/config.yaml
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


def feature_csv_path(features_dir, split, group):
    """Locates one of the four features.csv files written by step 03."""
    return os.path.join(features_dir, split, group, 'features.csv')


def load_labeled(features_dir, split, label_map):
    """Loads the CN and AD tables of one split and concatenates them with labels."""
    cn = pd.read_csv(feature_csv_path(features_dir, split, 'cn'))
    cn['label'] = label_map['cn']
    ad = pd.read_csv(feature_csv_path(features_dir, split, 'ad'))
    ad['label'] = label_map['ad']
    merged = pd.concat([cn, ad], ignore_index=True)
    return merged, len(cn), len(ad)


def select_feature_columns(train_df, max_missing_fraction):
    """Drops metadata columns and any feature with too much missing data in train."""
    meta_cols = ['uid', 'group', 'label']
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    missing_frac = train_df[feature_cols].isna().mean()
    drop_cols = missing_frac[missing_frac > max_missing_fraction].index.tolist()
    if drop_cols:
        pct = int(max_missing_fraction * 100)
        print(f"\nDropping {len(drop_cols)} features with >{pct}% missing in train:")
        print(f"  {drop_cols}")
    return [c for c in feature_cols if c not in drop_cols]


def main(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    cfg = config['matrix']
    features_dir = resolve_path(config['paths']['features_dir'])
    output_dir = resolve_path(config['paths']['processed_dir'])

    os.makedirs(output_dir, exist_ok=True)

    # Load and label: CN=0, AD=1
    train_df, n_train_cn, n_train_ad = load_labeled(features_dir, 'train', cfg['label_map'])
    test_df, n_test_cn, n_test_ad = load_labeled(features_dir, 'test', cfg['label_map'])

    print(f"Train: {len(train_df)} rows  (CN={n_train_cn}, AD={n_train_ad})")
    print(f"Test:  {len(test_df)} rows  (CN={n_test_cn}, AD={n_test_ad})")

    feature_cols = select_feature_columns(train_df, cfg['max_missing_fraction'])

    # Fill remaining NaNs with the TRAIN median (applied to both sets)
    train_median = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(train_median)
    test_df[feature_cols] = test_df[feature_cols].fillna(train_median)

    # Robust scaling - fit on train only, apply to both
    scaler = SCALERS[cfg['scaler']]()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])

    train_out = pd.DataFrame(train_scaled, columns=feature_cols)
    train_out['uid'] = train_df['uid'].values
    train_out['label'] = train_df['label'].values

    test_out = pd.DataFrame(test_scaled, columns=feature_cols)
    test_out['uid'] = test_df['uid'].values
    test_out['label'] = test_df['label'].values

    train_out.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
    test_out.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(output_dir, 'feature_cols.pkl'))

    print(f"\nFinal feature count: {len(feature_cols)}")
    print(f"Saved -> {os.path.join(output_dir, 'train_processed.csv')}")
    print(f"Saved -> {os.path.join(output_dir, 'test_processed.csv')}")
    print(f"Saved -> {os.path.join(output_dir, 'scaler.pkl')}  (reuse this in later steps)")
    print(f"Saved -> {os.path.join(output_dir, 'feature_cols.pkl')}  (final feature list)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
