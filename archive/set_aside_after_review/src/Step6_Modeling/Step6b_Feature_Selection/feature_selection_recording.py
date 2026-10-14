"""
Step 4 - Feature selection: 135 features down to the final subset (29).

Derived from notebook cells 14 and 15 (features.ipynb):
  cell 14 = "Step 2 - Statistical Analysis: Cohen's d and p-value"
  cell 15 = "Step 3 - Feature Selection: Effect Size Filtering + Correlation Pruning"

INPUTS
    <paths.processed_dir>/train_processed.csv   from build_matrix_recording.py
    config.yaml    selection.d_threshold, selection.corr_threshold

OUTPUTS
    <paths.processed_dir>/feature_stats.csv       per-feature Cohen's d and p
    <paths.processed_dir>/selected_features.csv   the final feature list

PROCESSING
    1. For every feature, effect size (Cohen's d, pooled SD) and significance
       (Welch t-test) comparing CN vs AD on the train set.
    2. Keep candidates with |d| >= selection.d_threshold, sorted by |d|.
    3. Greedy correlation pruning: walking from the strongest d to the weakest,
       any later feature correlated above selection.corr_threshold with an
       already-kept feature is dropped.

HOW TO RUN
    python Step6_Modeling/Step6b_Feature_Selection/feature_selection_recording.py
    python Step6_Modeling/Step6b_Feature_Selection/feature_selection_recording.py --config /path/to/config.yaml

CAVEAT
    Selection runs once on the FULL train set, before any cross-validation in
    Step 6c. See docs/open_questions.md (Q1) - not changed here.
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
    cfg = config['selection']
    processed_dir = resolve_path(config['paths']['processed_dir'])

    train_csv = os.path.join(processed_dir, 'train_processed.csv')
    train_df = pd.read_csv(train_csv)
    feature_cols = [c for c in train_df.columns if c not in ('uid', 'label')]

    # -- Part 1: statistics -------------------------------------------------
    res_df = compute_feature_stats(train_df, feature_cols, cfg['significance_alpha'])
    stats_path = os.path.join(processed_dir, 'feature_stats.csv')
    res_df.to_csv(stats_path, index=False)
    report_stats(res_df, cfg)
    print(f"\nSaved -> {stats_path}")

    # -- Part 2: effect-size filter + correlation pruning --------------------
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

    selected_path = os.path.join(processed_dir, 'selected_features.csv')
    pd.Series(kept).to_csv(selected_path, index=False, header=['feature'])
    print(f"\nSaved -> {selected_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
