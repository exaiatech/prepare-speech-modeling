# Data

This directory is where all data enters and leaves the pipeline. Scripts never
read from anywhere else, and no script contains a path pointing at one
person's machine. A fresh clone runs once the two inputs below are in place.

## What is and is not in the repository

| Directory | In git? | Contents |
| --- | --- | --- |
| `data/raw/train_audios/`, `test_audios/` | **No** | The PREPARE audio. You must supply these. |
| `data/raw/metadata.csv` | **No** | The PREPARE metadata table. You must supply this. |
| `data/raw/gold_transcripts/` | Yes | Hand-checked reference transcripts for the ASR evaluation. Small. |
| `data/interim/` | **No** | Everything regenerable: feature tables, preprocessed audio, transcripts, alignment reports, verticalized data. |
| `data/processed/` | **No** | Scaled matrices, fitted scalers, fitted models. Regenerable by Step 6. |
| `data/metadata/` | Yes | `recording_metadata.csv`, the cohort table. |
| `data/results/` | Yes | Descriptive statistics and the ASR master report. |

The audio, the metadata table and everything regenerable are listed in
`.gitignore`. Audio is never committed: the cohort is 1,031 recordings, and the
dataset is not ours to redistribute.

The Whisper checkpoint is not committed either, for the same reason of size.
See [`models/README.md`](../models/README.md).

## What flows where

The acoustic branch (Steps 2 and 6) and the ASR branch (Step 4) both start from
`data/raw/` and never overwrite each other's outputs.

```
data/raw/train_audios, test_audios      supplied by you
  |
  |-- Step2b  -> data/interim/features_recording, features_segment
  |               |
  |               Step6a -> data/processed/{recording,segment}
  |               Step6b/c/d read and write there
  |
  '-- Step4a  -> data/interim/processed_audios
        Step4b -> data/interim/transcripts          (needs models/, see below)
        Step4c -> data/interim/alignment_reports    (also reads data/raw/gold_transcripts)
        Step4d -> data/interim/verticalized_data
        Step4e -> data/results/Master_Report_Final.xlsx
```

Every directory above exists in a fresh clone, empty. Each step creates what it
needs, so you can run the branches independently and in any order.

## Getting the raw data

The PREPARE Challenge dataset comes from the NIA, aggregated from DementiaBank
corpora. Access is through the challenge organisers; it is not public.

Once you have it, arrange it like this:

```
data/raw/
  PREPARE_English_complete_COOKIE_THEFT.csv   the cohort file from the supervisor (config key paths.cohort)
  metadata.csv
  train_audios/
  test_audios/
  gold_transcripts/     already in the repository
```

The ASR branch additionally needs the Whisper checkpoint in `models/`, which is
also not committed. See [`models/README.md`](../models/README.md) for how to
download it.

The project's working copy lives on the shared Drive. Ask the project owner
for the link rather than copying audio between machines.

## If your data lives somewhere else

Do not edit `config.yaml` to point at your own machine: that path would then be
committed and break the pipeline for everyone else. Instead create
`config.local.yaml` in the repository root, which is gitignored, and override
only the keys you need:

```yaml
paths:
  metadata: 'D:\prepare\metadata.csv'
  raw_audio:
    train: 'D:\prepare\train_audios'
    test: 'D:\prepare\test_audios'
```

It is merged over `config.yaml` key by key at load time, so anything you leave
out keeps its committed value. Absolute paths are used as given; relative paths
resolve against the repository root, never against your current directory.

## Expected metadata columns

`metadata.csv` must contain at least `uid`, `language`, `split` and
`diagnosis`. The pipeline filters to `language == english`, then to Control
versus `ProbableAD` + `AD`, excluding MCI and PPA.

Audio files are matched to `uid` by filename stem, falling back to a prefix
match, so `abcd.wav` and `abcd_16k.wav` both match uid `abcd`.

> `metadata.csv` contains **no participant identifier** and no elicitation-task
> column. Participant-level independence therefore cannot be verified: if one
> person contributed several recordings across different partitions, that
> leakage is undetectable here. See `docs/open_questions.md` (Q23).

## Cohort at a glance

The current cohort is the Cookie Theft subset: **1,031 recordings, Control 635, MCI 65, AD/ADRD 331**, listed in
`PREPARE_English_complete_COOKIE_THEFT.csv`. The Stage 1 scripts write their outputs (inclusion list, frozen splits, embeddings,
predictions, metric tables) to `data/interim/stage1/`, which is gitignored.

The table below describes the **earlier full English cohort** and is **superseded, retained for reference**.

| | Recordings |
| --- | --- |
| Raw dataset | 2058 |
| English | 1655 |
| After excluding PPA (39) and MCI (135) | 1481 |
| Train / Validation / Test | 948 / 235 / 298 |
| Segments (5 s, 25% overlap) | 10 288 |

Class balance: 944 Control, 537 ProbableAD + AD.
Corpora: wls 825, pitt 507, vas 65, lu 51, delaware 30, baycrest 3.
