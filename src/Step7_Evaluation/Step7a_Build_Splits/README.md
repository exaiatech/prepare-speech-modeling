# Step 7a: Build the Frozen Splits

### Purpose
Freezes the data splits, once, for every model:
1. the fixed `set` partition (train, dev, test) for both tasks;
2. repeated 5-fold cross-validation, 5 folds x 10 repeats = **50 splits**, built **only on the train+dev pool**, for both tasks.

The test partition never enters a CV split. Encoder baselines, interpretable models and hybrids all read these files, so their
predictions are comparable. `random_state` is fixed (42), but the files should not be regenerated once models use them.

> **Status: run and verified by the author.**

### Inputs
`stage1.inclusion_list` (from Step 1c).

### Outputs
In `stage1.splits_dir` (default `data/interim/stage1/splits/`, gitignored):
`fixed_split_taskA_binary.csv`, `fixed_split_taskB_multiclass.csv`, `cv_splits_taskA_binary.csv`, `cv_splits_taskB_multiclass.csv`.
Task A is Control against AD/ADRD (MCI excluded). Task B is the three-class task.

### Configuration
`stage1.inclusion_list`, `stage1.splits_dir`. `RANDOM_STATE`, `N_SPLITS` and `N_REPEATS` stay constants in the script.

### Execution Instructions
```bash
python src/Step7_Evaluation/Step7a_Build_Splits/02_build_splits.py
```
