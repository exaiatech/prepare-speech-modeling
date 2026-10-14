# Step 3b: Descriptive Statistics

### Purpose
Reports mean, standard deviation, minimum and maximum per diagnostic class for every feature, together with the Control versus Alzheimer's contrast.

> **After the supervisor's review.** Recording level only: the segment-level statistics and their output file were set aside (`archive/set_aside_after_review/`). **Cohort membership comes from the ID list of the shared cohort CSV** (`paths.cohort`, column `cohort.id_column`). Feature-table rows whose `uid` is not in that list are dropped, and the script prints how many were kept, how many were dropped, and how many cohort IDs have no row. The feature tables hold Control and AD only, so cohort members with another diagnosis (MCI) do not appear in the statistics.

Statistics are computed on the **raw, unscaled** feature tables so that the units stay interpretable. This is why the script reads `paths.features_dir` (Step 2b output) rather than the scaled matrices produced in Step 6a.

### Inputs
* **Required Files:**
  1. `data/raw/PREPARE_English_complete_COOKIE_THEFT.csv`, the shared cohort CSV (its `ID` column is the membership list)
  2. `data/interim/features_recording/{train,test}/{cn,ad}/features.csv`
  3. `docs/feature_inventory.csv`, which supplies the domain and unit of each feature. **Run Step 3a first.**
* **File Formats:** `.csv`

### Outputs
* **Generated Outputs:** Per-class descriptive statistics, plus a written summary.
* **File Format:** `.csv` and `.md`
* **Location:** `data/results/descriptive_stats_recording.csv`, `docs/descriptive_statistics.md`
* **Structure:** One row per feature, with mean/std/min/max for each class and the between-class contrast, joined to the feature's domain and unit.

### Configuration
* `reporting.use_raw_features` (`true`), the decision to describe unscaled values.
* `reporting.decimals` (4), rounding in the output tables.
* `paths.cohort`, `cohort.id_column`, `paths.features_dir`, `paths.results_dir`, `paths.docs_dir`.

### Dependencies
* `pandas`, `numpy`, `pyyaml`.

### Execution Instructions
Requires Step 2b and Step 3a to have run.
```bash
python src/Step3_Feature_QC/Step3b_Descriptive_Stats/descriptive_stats.py
```
