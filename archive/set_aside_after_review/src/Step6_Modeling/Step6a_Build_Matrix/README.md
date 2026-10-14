# Step 6a: Build the Modelling Matrix

### Purpose
Turns the per-group feature tables from Step 2b into a single labelled, imputed and scaled matrix per split, ready for selection and training. Two scripts, one per unit of analysis.

Three things happen here:
1. **Merge and label.** The four `{split}/{cn,ad}` tables become one table per split, with `label` 0 for Control and 1 for Alzheimer's.
2. **Drop and impute.** A feature missing in more than `max_missing_fraction` (30%) of the training rows is dropped outright. Remaining gaps are filled with the training median.
3. **Scale.** `RobustScaler` (median and IQR), chosen over standardisation because several features are heavy-tailed.

> **The scaler and the medians are fitted on training data only, then applied to test.** This is the boundary that keeps the held-out result honest. Do not refit on the combined data.

### Inputs
* **Required Files:** Step 2b output.
  * Recording level: `data/interim/features_recording/{train,test}/{cn,ad}/features.csv`
  * Segment level: `data/interim/features_segment/{train,test}/{cn,ad}/features_segments.csv`
* **File Formats:** `.csv`

### Outputs
* **Generated Outputs:** Scaled train and test matrices, plus the fitted scaler.
* **File Format:** `.csv` for the matrices, joblib `.pkl` for the scaler.
* **Location:** `data/processed/recording/` and `data/processed/segment/`
* **Structure:** `train_processed.csv` and `test_processed.csv`, one row per observation: the surviving feature columns plus `label` and `uid`. The segment matrices additionally carry `segment_idx`, and keep `uid` as the **grouping column** that later steps use to stop segments from one recording crossing a fold boundary.

### Configuration
* `matrix.max_missing_fraction` (0.3), `matrix.scaler` (`robust`), `matrix.label_map` (`cn: 0`, `ad: 1`).
* Segment equivalents live under `segments.matrix:`, including `group_column` (`uid`) and `segment_column` (`segment_idx`).
* The key names are duplicated deliberately: the two levels are independent settings that happen to share values today.

### Dependencies
* `pandas`, `numpy`, `scikit-learn`, `joblib`, `pyyaml`.

### Execution Instructions
```bash
python Step6_Modeling/Step6a_Build_Matrix/build_matrix_recording.py
```
```bash
python Step6_Modeling/Step6a_Build_Matrix/build_matrix_segment.py
```
