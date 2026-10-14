"""
Segment step 4 - Feature selection on the segment-level train set (135 -> 25).

Derived from notebook cells F3 and F4 of features2.ipynb:
  F3 = "Step 2 (segments) - Statistical Analysis on Segment-Level Train Data"
  F4 = "Step 3 (segments) - Feature Selection: Effect Size + Correlation Pruning"

The selection logic is identical to common/selection.py; only the input
table (segments instead of recordings) and the output file names differ. The
thresholds live under segments.selection in config.yaml and are independent of
the whole-recording ones.

INPUTS
    <segments.paths.processed_dir>/train_segments_processed.csv
    config.yaml   segments.selection.d_threshold, .corr_threshold

OUTPUTS
    <segments.paths.processed_dir>/feature_stats_segments.csv
    <segments.paths.processed_dir>/selected_features_segments.csv

PROCESSING
    1. Cohen's d (pooled SD) and Welch t-test per feature, CN vs AD, with each
       SEGMENT counting as one observation.
    2. Keep candidates with |d| >= segments.selection.d_threshold.
    3. Greedy correlation pruning from strongest |d| downwards, dropping any
       feature correlated above segments.selection.corr_threshold with a
       feature already kept.

CAVEATS
    * Segments from the same recording are correlated, so the p-values are
      optimistic through pseudo-replication - the original notebook says so
      itself. Selection uses |d| only, but the `significant` column inherits
      the problem. See docs/open_questions.md (Q17).
    * Selection runs once on the FULL train set, outside the cross-validation
      of Step 6c. Same issue as the whole-recording path (Q1 / Q17).

HOW TO RUN
    python Step6_Modeling/Step6b_Feature_Selection/feature_selection_segment.py
    python Step6_Modeling/Step6b_Feature_Selection/feature_selection_segment.py --config /path/to/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import DEFAULT_CONFIG, load_config, resolve_path
from common.selection import compute_feature_stats, prune_correlated, report_stats


def main(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    cfg = config['segments']['selection']
    seg_col = config['segments']['matrix']['segment_column']
    processed_dir = resolve_path(config['segments']['paths']['processed_dir'])

    train_csv = os.path.join(processed_dir, 'train_segments_processed.csv')
    train_df = pd.read_csv(train_csv)
    feature_cols = [c for c in train_df.columns if c not in ('uid', seg_col, 'label')]

    # -- Part 1: statistics on segment-level observations --------------------
    print(f"Segments - total: {len(train_df)}   "
          f"recordings: {train_df['uid'].nunique()}")
    res_df = compute_feature_stats(train_df, feature_cols, cfg['significance_alpha'])
    stats_path = os.path.join(processed_dir, 'feature_stats_segments.csv')
    res_df.to_csv(stats_path, index=False)
    report_stats(res_df, cfg)
    print(f"\nSaved -> {stats_path}")

    # -- Part 2: effect-size filter + correlation pruning ---------------------
    candidates = res_df[res_df['abs_d'] >= cfg['d_threshold']].sort_values('abs_d', ascending=False)
    print(f"\nCandidates with |d| >= {cfg['d_threshold']}: {len(candidates)}")

    kept, dropped = prune_correlated(train_df, candidates, cfg['corr_threshold'])

    print(f"\nFinal feature count after correlation pruning: {len(kept)}")
    print(f"Dropped as redundant: {len(dropped)}\n")

    d_lookup = dict(zip(candidates['feature'], candidates['abs_d']))
    print(f"{'feature':<32}{'d':>8}")
    print("=" * 40)
    for f in kept:
        print(f"{f:<32}{d_lookup[f]:>8.3f}")

    selected_path = os.path.join(processed_dir, 'selected_features_segments.csv')
    pd.Series(kept).to_csv(selected_path, index=False, header=['feature'])
    print(f"\nSaved -> {selected_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
