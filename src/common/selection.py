"""
Feature selection: Cohen's d effect sizes and greedy correlation pruning.

Derived from notebook cells 14 and 15 (features.ipynb):
  cell 14 = "Step 2 - Statistical Analysis: Cohen's d and p-value"
  cell 15 = "Step 3 - Feature Selection: Effect Size Filtering + Correlation Pruning"

Both selection steps import these three functions unchanged:

    Step6b_Feature_Selection/feature_selection_recording.py
    Step6b_Feature_Selection/feature_selection_segment.py

They previously lived in the recording-level script and were pulled into the
segment-level one with importlib.util.spec_from_file_location. The bodies are
unchanged; only their location moved.

METHOD
    compute_feature_stats  per feature, Cohen's d (pooled SD) and a Welch
                           t-test comparing CN vs AD on the training set
    prune_correlated       walking from the strongest |d| to the weakest, drops
                           any feature correlated above corr_threshold with a
                           feature already kept
    report_stats           prints the effect-size summary table

WHY p-VALUES ARE NOT A SELECTION CRITERION
    With n = 1183 they reach significance for trivially small effects, and at
    segment level they are inflated by pseudo-replication. They are computed
    and reported, but only |d| and the correlation filter decide the subset.

This module defines no CLI: it is imported, never run.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_feature_stats(df, feature_cols, alpha):
    """Cohen's d (pooled SD) and Welch t-test p-value per feature, CN vs AD."""
    cn = df[df['label'] == 0]
    ad = df[df['label'] == 1]
    print(f"CN: {len(cn)}   AD: {len(ad)}   Features: {len(feature_cols)}\n")

    results = []
    for f in feature_cols:
        c = cn[f].dropna().values
        a = ad[f].dropna().values
        if len(c) < 2 or len(a) < 2:
            continue

        pooled = np.sqrt((np.var(c) + np.var(a)) / 2)
        d = (np.mean(a) - np.mean(c)) / pooled if pooled > 0 else 0.0
        _, p = stats.ttest_ind(c, a, equal_var=False)

        results.append({
            'feature': f,
            'd': d,
            'abs_d': abs(d),
            'p': p,
            'significant': p < alpha,
            'mean_cn': np.mean(c),
            'mean_ad': np.mean(a),
        })

    return pd.DataFrame(results).sort_values('abs_d', ascending=False).reset_index(drop=True)


def prune_correlated(train_df, candidates, corr_threshold):
    """
    Greedy pruning: walk features from strongest d to weakest,
    drop any later feature that's too correlated with one already kept.
    """
    cand_features = candidates['feature'].tolist()
    corr = train_df[cand_features].corr().abs()

    kept, dropped = [], []
    for f in cand_features:  # already sorted by |d| descending
        if f in dropped:
            continue
        too_correlated_with_kept = any(corr.loc[f, k] > corr_threshold for k in kept)
        if too_correlated_with_kept:
            dropped.append(f)
            continue
        kept.append(f)
        # mark any remaining candidate too correlated with this one as dropped
        for other in cand_features:
            if other != f and other not in kept and other not in dropped:
                if corr.loc[f, other] > corr_threshold:
                    dropped.append(other)

    return kept, dropped


def report_stats(res_df, cfg):
    """Prints the top features and the effect-size summary of step 1."""
    top_n = cfg['report_top_n']
    print(f"{'feature':<32}{'d':>8}{'p':>10}{'sig':>6}")
    print("=" * 58)
    for _, r in res_df.head(top_n).iterrows():
        flag = '*' if r['significant'] else ''
        print(f"{r['feature']:<32}{r['d']:>8.2f}{r['p']:>10.4f}{flag:>6}")

    n_sig = res_df['significant'].sum()
    n_large = (res_df['abs_d'] >= cfg['effect_size_large']).sum()
    n_medium = ((res_df['abs_d'] >= cfg['effect_size_medium'])
                & (res_df['abs_d'] < cfg['effect_size_large'])).sum()

    print(f"\nTotal features:        {len(res_df)}")
    print(f"Significant (p<{cfg['significance_alpha']}):  {n_sig}")
    print(f"Large effect (|d|>={cfg['effect_size_large']}): {n_large}")
    print(f"Medium effect ({cfg['effect_size_medium']}-{cfg['effect_size_large']}): {n_medium}")
