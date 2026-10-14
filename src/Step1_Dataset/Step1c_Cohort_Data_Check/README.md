# Step 1c: Cohort Data Check (Stage 1)

### Purpose
Verifies the Cookie Theft cohort before any model is trained and writes the **inclusion list** that Steps 7 and 8 read. It builds
the list from the shared cohort CSV (rows with `COOKIE_THEFT == 1`), matches every ID to its audio file and to `uid` in
`metadata.csv`, and runs the checks requested before modelling: audio matching, duplicate IDs and duplicate audio content, the
`hash` column against the metadata, the diagnosis mapping, and the corpus composition.

It overlaps in intent with Step 1a, which verifies the cohort against the agreed counts and builds the recording table. The two
produce different outputs: this one writes the inclusion list and a check report.

> **Status: run and verified by the author.**

### Inputs
* `paths.cohort` (the shared cohort CSV), `paths.metadata` (`metadata.csv`), `paths.raw_audio.train` and `.test`.

### Outputs
In `stage1.data_check_dir` (default `data/interim/stage1/`, gitignored):
* `cookie_theft_inclusion_list.csv`
* `data_check_report.md`

### Configuration
Paths come from `config.yaml` (`paths.*` and `stage1.*`). The script used to carry absolute Windows paths at the top; they were
moved out. Diagnosis mapping and the optional `DIAGNOSIS_COL` stay as constants in the script.

### Dependencies
`pandas`, `pyyaml`.

### Execution Instructions
```bash
python src/Step1_Dataset/Step1c_Cohort_Data_Check/01_data_check.py
```
Run it first. Step 7a reads its output.
