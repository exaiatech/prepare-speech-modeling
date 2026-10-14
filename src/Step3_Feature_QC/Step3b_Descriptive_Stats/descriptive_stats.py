"""
Step 3b - Descriptive statistics of every feature, per diagnostic class.

Written for the supervisor request of 2026-07-30: mean, standard deviation,
minimum and maximum of every feature, reported separately for each diagnostic
class.

It is a NEW reporting script, not derived from either notebook, and it changes
no analysis. It reads the RAW, unscaled feature tables (before any scaling) so
that the numbers keep their physical units - Hz, dB, seconds - and can be read
against docs/feature_inventory.md.

AFTER THE SUPERVISOR'S REVIEW
    * Recording level only. The segment-level statistics, and their output file,
      were set aside (archive/set_aside_after_review/). One observation per
      recording.
    * Cohort membership comes from the ID list of the shared cohort CSV
      (paths.cohort, column cohort.id_column). Rows of the feature tables whose
      uid is not in that list are dropped, and the script reports how many.

INPUTS
    paths.cohort                                  the shared cohort CSV (ID list)
    <paths.features_dir>/{train,test}/{cn,ad}/features.csv
    <paths.docs_dir>/feature_inventory.csv        domain and unit per feature
    config.yaml   reporting.*, cohort.id_column

OUTPUTS
    <paths.results_dir>/descriptive_stats_recording.csv
    <paths.docs_dir>/descriptive_statistics.md

    One row per feature and per class, with n, n_missing, mean, std, min, max,
    plus the pooled difference in means and Cohen's d for orientation, across
    train and test combined and additionally per partition.

CLASS DEFINITION
    CN = rows of the control feature tables
    AD = rows of the AD feature tables
    The feature tables hold Control and AD recordings only (the extractor's
    metadata filter). Cohort members with another diagnosis, MCI in particular,
    have no row in them and therefore do not appear here. The script prints how
    many.

HOW TO RUN
    python src/Step3_Feature_QC/Step3b_Descriptive_Stats/descriptive_stats.py
    python src/Step3_Feature_QC/Step3b_Descriptive_Stats/descriptive_stats.py --config /path/to/config.yaml

    Requires build_inventory.py to have been run first (for units).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cohort import load_cohort_ids
from common.config import DEFAULT_CONFIG, load_config, resolve_path


META_COLUMNS = ('uid', 'group', 'label')


def load_raw_tables(features_dir, filename):
    """Loads the four raw feature tables and tags class and partition."""
    frames = []
    for split in ('train', 'test'):
        for group, cls in (('cn', 'CN'), ('ad', 'AD')):
            path = os.path.join(features_dir, split, group, filename)
            if not os.path.exists(path):
                print(f"  [!] missing, skipped: {path}")
                continue
            df = pd.read_csv(path)
            df = df.assign(diagnostic_class=cls, partition=split)
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def restrict_to_cohort(df, cohort_ids):
    """Keeps the rows whose uid is in the cohort list, and reports what it dropped.

    Returns the filtered table. Prints: rows kept, rows dropped because the uid is
    not in the list, and cohort IDs that have no row in the table.
    """
    uid = df['uid'].astype(str)
    keep = uid.isin(cohort_ids)
    absent = len(cohort_ids - set(uid))
    print(f"  cohort list: {len(cohort_ids)} IDs. Feature table: {len(df)} rows, "
          f"{int(keep.sum())} kept, {int((~keep).sum())} dropped (not in the list), "
          f"{absent} cohort IDs have no row")
    return df[keep].reset_index(drop=True)


def feature_columns(df):
    """Every column that is a feature rather than metadata."""
    return [c for c in df.columns
            if c not in META_COLUMNS and c not in ('diagnostic_class', 'partition')]


def describe_group(df, features, decimals, partition_label):
    """mean / std / min / max per feature and diagnostic class."""
    rows = []
    for cls in ('CN', 'AD'):
        sub = df[df['diagnostic_class'] == cls]
        for feat in features:
            vals = sub[feat]
            clean = vals.dropna()
            rows.append({
                'partition': partition_label,
                'feature': feat,
                'diagnostic_class': cls,
                'n': int(len(clean)),
                'n_missing': int(vals.isna().sum()),
                'mean': round(float(clean.mean()), decimals) if len(clean) else np.nan,
                'std': round(float(clean.std(ddof=0)), decimals) if len(clean) else np.nan,
                'min': round(float(clean.min()), decimals) if len(clean) else np.nan,
                'max': round(float(clean.max()), decimals) if len(clean) else np.nan,
            })
    return pd.DataFrame(rows)


def add_contrast(stats_df, decimals):
    """Adds AD-minus-CN mean difference and Cohen's d, repeated on both rows."""
    wide = stats_df.pivot_table(index=['partition', 'feature'],
                                columns='diagnostic_class',
                                values=['mean', 'std'])
    diff = wide[('mean', 'AD')] - wide[('mean', 'CN')]
    pooled = np.sqrt((wide[('std', 'CN')] ** 2 + wide[('std', 'AD')] ** 2) / 2)
    d = (diff / pooled).replace([np.inf, -np.inf], np.nan)

    contrast = pd.DataFrame({
        'mean_diff_ad_minus_cn': diff.round(decimals),
        'cohens_d': d.round(decimals),
    }).reset_index()
    return stats_df.merge(contrast, on=['partition', 'feature'], how='left')


def build_stats(df, decimals):
    """Overall statistics plus one block per partition."""
    features = feature_columns(df)
    blocks = [describe_group(df, features, decimals, 'all')]
    for split in ('train', 'test'):
        sub = df[df['partition'] == split]
        if len(sub):
            blocks.append(describe_group(sub, features, decimals, split))
    stats_df = pd.concat(blocks, ignore_index=True)
    return add_contrast(stats_df, decimals), features


def attach_inventory(stats_df, inventory_path):
    """Joins domain and unit from the generated feature inventory."""
    if not os.path.exists(inventory_path):
        print(f"  [!] inventory not found, domain/unit columns omitted: {inventory_path}")
        return stats_df
    inv = pd.read_csv(inventory_path, usecols=['feature', 'domain', 'unit', 'order'])
    merged = stats_df.merge(inv, on='feature', how='left')
    return merged.sort_values(['partition', 'order', 'diagnostic_class']).reset_index(drop=True)


def write_markdown(path, recording_stats, counts):
    """Writes the human-readable summary pointing at the CSV."""
    lines = [
        '# Descriptive Statistics',
        '',
        'Mean, standard deviation, minimum and maximum of every extracted feature,',
        'reported separately for each diagnostic class, on the **raw, unscaled**',
        'feature tables so that units remain interpretable',
        '(see `feature_inventory.md`).',
        '',
        'Generated by `Step3_Feature_QC/Step3b_Descriptive_Stats/descriptive_stats.py`.',
        'Recording level only: the segment-level statistics were set aside after the',
        "supervisor's review. Cohort membership comes from the ID list of the shared",
        'cohort CSV.',
        '',
        '## Class definition',
        '',
        '| Class | Diagnoses | Model label |',
        '| --- | --- | --- |',
        '| CN | Control | 0 |',
        '| AD | ProbableAD, AD | 1 |',
        '',
        'The feature tables hold Control and AD recordings only. Cohort members with another',
        'diagnosis, MCI in particular, have no row in them and do not appear here.',
        '',
        '## Full table',
        '',
        '| Unit of analysis | File | Rows |',
        '| --- | --- | --- |',
    ]
    if recording_stats is not None:
        lines.append(f'| recording | `../data/results/descriptive_stats_recording.csv` | '
                     f'{len(recording_stats)} |')
    lines += [
        '',
        'The CSV has one row per feature, diagnostic class and partition block',
        '(`all`, `train`, `test`), with columns:',
        '',
        '`partition, feature, diagnostic_class, n, n_missing, mean, std, min, max,`',
        '`mean_diff_ad_minus_cn, cohens_d, domain, unit`',
        '',
        '## Sample sizes',
        '',
        '| Unit of analysis | Partition | CN | AD |',
        '| --- | --- | --- | --- |',
    ]
    for label, block in counts:
        for partition, cn, ad in block:
            lines.append(f'| {label} | {partition} | {cn} | {ad} |')

    lines += [
        '',
        '## Reading notes',
        '',
        '- `std` is the population standard deviation (ddof=0).',
        "- `cohens_d` here is computed on RAW values for orientation only. It is not the effect size",
        '  used to select features; that analysis was set aside and will be redone inside each',
        '  training fold when interpretable modeling starts.',
        '- `n_missing` counts NaNs produced by the extractor before any imputation.',
        '',
    ]
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))


def partition_counts(df, unit):
    """Class counts per partition, in the unit of analysis."""
    out = []
    for partition in ('all', 'train', 'test'):
        sub = df if partition == 'all' else df[df['partition'] == partition]
        cn = (sub['diagnostic_class'] == 'CN').sum()
        ad = (sub['diagnostic_class'] == 'AD').sum()
        out.append((partition, f'{cn} {unit}', f'{ad} {unit}'))
    return out


def main(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    rcfg = config['reporting']
    decimals = rcfg['decimals']

    results_dir = resolve_path(config['paths']['results_dir'])
    docs_dir = resolve_path(config['paths']['docs_dir'])
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    inventory_path = docs_dir / rcfg['feature_inventory_csv']

    cohort_ids = load_cohort_ids(config)

    counts = []

    print("Recording-level tables...")
    rec_raw = load_raw_tables(resolve_path(config['paths']['features_dir']), 'features.csv')
    rec_stats = None
    if rec_raw is not None:
        rec_raw = restrict_to_cohort(rec_raw, cohort_ids)
        rec_stats, feats = build_stats(rec_raw, decimals)
        rec_stats = attach_inventory(rec_stats, inventory_path)
        path = results_dir / rcfg['descriptive_stats_recording']
        rec_stats.to_csv(path, index=False)
        counts.append(('recording', partition_counts(rec_raw, 'recordings')))
        print(f"  {len(feats)} features x 2 classes x 3 partition blocks "
              f"= {len(rec_stats)} rows -> {path}")

    md_path = docs_dir / rcfg['descriptive_stats_md']
    write_markdown(md_path, rec_stats, counts)
    print(f"\nSaved -> {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
