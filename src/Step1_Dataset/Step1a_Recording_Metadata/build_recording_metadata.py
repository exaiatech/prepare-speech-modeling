"""
Step 0 - Recording-level metadata table (one row per recording).

Written for the supervisor request of 2026-07-30: a table that makes explicit
how participants, recordings and segments relate, which elicitation task each
recording used, and how everything was allocated to the data partitions.

It is a NEW script, not derived from either notebook. It only joins and counts
existing information; it computes no features and changes no analysis.

INPUTS
    paths.cohort                                         the shared cohort CSV; its ID column
                                                         is the list of cohort members
    paths.metadata                                       metadata.csv
    <segments.paths.features_dir>/*/*/features_segments.csv   segment counts
    <paths.features_dir>/*/*/features.csv                whole-recording rows
    config.yaml   reporting.metadata_passthrough, reporting.participant_proxy_keys

OUTPUT
    <paths.data_dir>/recording_metadata.csv

    Columns
      uid                  recording identifier (PREPARE)
      corpus               source corpus (wls, pitt, ...)
      split                train / test, as distributed by PREPARE
      diagnosis            Control / ProbableAD / AD / MCI / PPA
      label                CN=0, AD=1, empty for diagnoses outside the cohort
      in_cohort            1 if the uid is in the ID list of the shared cohort CSV
                           (paths.cohort, column cohort.id_column), else 0
      age, gender, education, race, handedness, language, filesize_kb
      participant_id       ALWAYS EMPTY - see below
      elicitation_task     ALWAYS EMPTY - see below
      n_recordings_for_participant   ALWAYS EMPTY - see below
      in_recording_pipeline          1 if a row exists in the recording feature table
      n_segments           number of 5 s segments extracted from this recording
      partition_recording  partition used by the whole-recording pipeline
      partition_segment    partition used by the segment pipeline
      participant_proxy_group        heuristic collision key, NOT an identifier
      participant_proxy_group_size   how many recordings share that key

WHAT THIS TABLE CANNOT ANSWER
    metadata.csv has 12 columns and 2058 rows, one row per uid, and contains
    NO participant identifier and NO elicitation-task column. `hash` is unique
    per row (a file hash, not a person). Therefore:

      * "which recordings belong to the same participant" - NOT ANSWERABLE
      * "how many recordings did each participant contribute" - NOT ANSWERABLE
      * "which elicitation task was used" - NOT ANSWERABLE (only `corpus`,
        which constrains but does not determine the task)

    Those three columns are emitted empty rather than guessed. The
    participant_proxy_* columns group recordings that share
    reporting.participant_proxy_keys (corpus, age, gender, diagnosis). That is
    a COLLISION ESTIMATE showing how much room for repeated participants
    exists - it is not an identity claim and must not be used for grouping in
    cross-validation. See docs/dataset_description.md and
    docs/open_questions.md (Q23).

HOW TO RUN
    python src/Step1_Dataset/Step1a_Recording_Metadata/build_recording_metadata.py
    python src/Step1_Dataset/Step1a_Recording_Metadata/build_recording_metadata.py --config /path/to/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cohort import load_cohort_ids
from common.config import DEFAULT_CONFIG, load_config, resolve_path


UNKNOWN_COLUMNS = ['participant_id', 'elicitation_task', 'n_recordings_for_participant']


def read_optional(path, **kwargs):
    """Reads a CSV if it exists, otherwise returns None."""
    if os.path.exists(path):
        return pd.read_csv(path, **kwargs)
    print(f"  [!] missing, skipped: {path}")
    return None


def collect_segment_counts(features_dir):
    """Counts extracted segments per uid, and records their partition."""
    frames = []
    for split in ('train', 'test'):
        for group in ('cn', 'ad'):
            path = os.path.join(features_dir, split, group, 'features_segments.csv')
            df = read_optional(path, usecols=['uid'])
            if df is None:
                continue
            df = df.assign(partition_segment=split)
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['uid', 'n_segments', 'partition_segment'])

    allseg = pd.concat(frames, ignore_index=True)
    counts = (allseg.groupby('uid')
              .agg(n_segments=('uid', 'size'),
                   partition_segment=('partition_segment', 'first'))
              .reset_index())
    return counts


def collect_recording_rows(features_dir):
    """Marks which uids produced a row in the whole-recording feature table."""
    frames = []
    for split in ('train', 'test'):
        for group in ('cn', 'ad'):
            path = os.path.join(features_dir, split, group, 'features.csv')
            df = read_optional(path, usecols=['uid'])
            if df is None:
                continue
            frames.append(df.assign(partition_recording=split))
    if not frames:
        return pd.DataFrame(columns=['uid', 'in_recording_pipeline', 'partition_recording'])

    allrec = pd.concat(frames, ignore_index=True)
    allrec['in_recording_pipeline'] = 1
    return allrec.drop_duplicates('uid')[['uid', 'in_recording_pipeline',
                                          'partition_recording']]


def check_membership(table, cohort_ids):
    """Recording-level checks of the cohort list against the tables. Prints, never alters."""
    missing = sorted(cohort_ids - set(table['uid'].astype(str)))
    print(f"\nCohort list: {len(cohort_ids)} IDs, {len(cohort_ids) - len(missing)} found in the "
          f"metadata rows that pass the language filter, {len(missing)} not found")
    if missing:
        print(f"  not found, first five: {missing[:5]}")
    members = table[table['in_cohort'] == 1]
    no_features = members[members['in_recording_pipeline'] == 0]
    print(f"Cohort members without a row in the recording feature table: "
          f"{len(no_features)} of {len(members)}")
    if len(no_features):
        print(no_features['diagnosis'].value_counts().to_string())


def add_participant_proxy(df, keys):
    """Adds the heuristic collision key. NOT a participant identifier."""
    present = [k for k in keys if k in df.columns]
    proxy = df[present].astype(str).agg('|'.join, axis=1)
    df['participant_proxy_group'] = proxy
    df['participant_proxy_group_size'] = proxy.map(proxy.value_counts())
    return df


def build_table(config):
    """Assembles the recording-level metadata table."""
    fcfg = config['features']['metadata_filters']
    rcfg = config['reporting']

    meta = pd.read_csv(resolve_path(config['paths']['metadata']))
    meta = meta[meta['language'] == fcfg['language']].copy()

    keep = [c for c in rcfg['metadata_passthrough'] if c in meta.columns]
    table = meta[keep].copy()

    control = fcfg['control_labels']
    ad = fcfg['ad_labels']
    cohort_ids = load_cohort_ids(config)
    table['in_cohort'] = table['uid'].astype(str).isin(cohort_ids).astype(int)
    table['label'] = table['diagnosis'].map(
        {**{d: 0 for d in control}, **{d: 1 for d in ad}})

    for col in UNKNOWN_COLUMNS:
        table[col] = pd.NA

    rec = collect_recording_rows(resolve_path(config['paths']['features_dir']))
    table = table.merge(rec, on='uid', how='left')
    table['in_recording_pipeline'] = table['in_recording_pipeline'].fillna(0).astype(int)

    seg = collect_segment_counts(resolve_path(config['segments']['paths']['features_dir']))
    table = table.merge(seg, on='uid', how='left')
    table['n_segments'] = table['n_segments'].fillna(0).astype(int)

    table = add_participant_proxy(table, rcfg['participant_proxy_keys'])
    table = table.sort_values(['corpus', 'split', 'diagnosis', 'uid']).reset_index(drop=True)
    check_membership(table, cohort_ids)
    return table


def print_summary(table):
    """Prints the counts the supervisor asked to have documented."""
    cohort = table[table['in_cohort'] == 1]

    print("\nRecordings per diagnosis and partition (members of the shared cohort list):")
    print(pd.crosstab(cohort['split'], cohort['diagnosis'], margins=True))

    print("\nRecordings per corpus and partition:")
    print(pd.crosstab(cohort['corpus'], cohort['split'], margins=True))

    seg = cohort[cohort['n_segments'] > 0]
    print(f"\nSegments: {int(cohort['n_segments'].sum())} from {len(seg)} recordings")
    print(f"  recordings in cohort with zero segments: {(cohort['n_segments'] == 0).sum()}")
    if len(seg):
        print("  segments per recording:")
        print(seg['n_segments'].describe().to_string())
        print("\n  distribution:")
        print(seg['n_segments'].value_counts().sort_index().to_string())

    grp = cohort['participant_proxy_group_size']
    print(f"\nParticipant proxy (heuristic, NOT an identifier):")
    print(f"  distinct proxy groups:                 {cohort['participant_proxy_group'].nunique()}")
    print(f"  recordings sharing a key with another: {(grp > 1).sum()} of {len(cohort)}")
    print(f"  largest proxy group:                   {grp.max()}")
    print("  -> participant-level uniqueness CANNOT be confirmed from metadata.csv")


def main(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    table = build_table(config)

    out_dir = resolve_path(config['paths']['data_dir'])
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / config['reporting']['recording_metadata_file']
    table.to_csv(out_path, index=False)

    print(f"Recording-level metadata table: {len(table)} rows, {table.shape[1]} columns")
    print_summary(table)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
