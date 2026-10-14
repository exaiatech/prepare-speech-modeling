# Step 1a: Recording-Level Metadata Table

### Purpose
Builds one row per recording describing the cohort: diagnosis, corpus, split, demographics, whether the recording is in the cohort, and how many segments it produced. This is the table every cohort count in the paper is read from, so that no figure is quoted from an ad hoc notebook cell.

> **After the supervisor's review.** `in_cohort` is now **1 if the recording's `uid` is in the ID list of the shared cohort CSV** (`paths.cohort`, column `cohort.id_column`), not a filter on diagnosis labels. The script reports how many cohort IDs it could not find in `metadata.csv`, and how many cohort members have no row in the recording feature table (MCI members, since the extractor covers Control and AD only). The segment counts (`n_segments`) are still produced from the segment feature tables if they exist; that part was not revisited after the segment path was set aside.

It also computes a **participant proxy** by grouping on `corpus|age|gender|diagnosis`. This is deliberately not a participant identifier. It exists only to estimate how often two recordings *could* belong to the same person, because `metadata.csv` ships no participant column. See `docs/open_questions.md` (Q23).

### Inputs
* **Required Files:**
  1. `data/raw/PREPARE_English_complete_COOKIE_THEFT.csv`, the shared cohort CSV (its `ID` column is the membership list).
  2. `data/raw/metadata.csv`, the PREPARE metadata table.
  3. Optional: the extracted feature tables, if Step 2b has already run. Missing feature tables are tolerated; the corresponding columns are left empty.
* **File Formats:** `.csv`
* **Parameters:** `reporting.metadata_passthrough` selects which metadata columns are copied through. `reporting.participant_proxy_keys` defines the proxy grouping.

### Outputs
* **Generated Outputs:** The recording-level metadata table.
* **File Format:** `.csv`
* **Location:** `data/metadata/recording_metadata.csv`
* **Structure:** One row per recording. Passthrough columns from `metadata.csv`, plus `in_cohort`, `label`, `participant_id` (proxy), `elicitation_task`, `n_recordings_for_participant`, `in_recording_pipeline`, `partition_record`.

### Configuration
* All settings come from `config.yaml` via `common/config.py`. Nothing is hardcoded.
* Keys used: `paths.cohort`, `cohort.id_column`, `paths.metadata`, `paths.data_dir`, `paths.features_dir`, `segments.paths.features_dir`, `reporting.metadata_passthrough`, `reporting.participant_proxy_keys`.

### Dependencies
* `pandas`, `numpy`, `pyyaml`. See the repository `requirements.txt`.

### Execution Instructions
Run from the repository root:
```bash
python src/Step1_Dataset/Step1a_Recording_Metadata/build_recording_metadata.py
```
