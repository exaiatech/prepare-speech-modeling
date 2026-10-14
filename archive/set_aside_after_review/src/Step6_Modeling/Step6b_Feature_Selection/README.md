# Step 6b: Feature Selection

### Purpose
Reduces 135 features to the modelling subset: 29 at recording level, 25 at segment level. Both scripts share their statistics and pruning logic through `common/selection.py`.

Two stages:
1. **Effect size filter.** Cohen's d (pooled SD) comparing Control against Alzheimer's on the training set. Features with `|d| >= 0.2` enter the candidate pool, ordered by effect size.
2. **Greedy correlation pruning.** Walking from the strongest effect to the weakest, any feature correlated above `r > 0.85` with a feature already kept is dropped.

> **p-values are computed and reported but are NOT a selection criterion.** With n = 1183 they reach significance for trivially small effects, and at segment level they are further inflated by pseudo-replication, since segments from one recording are not independent observations. Only `|d|` and the correlation filter decide the subset.

> **Known leakage.** Selection runs **once on the full training set, before cross-validation**, rather than inside each fold. The cross-validation AUCs reported downstream are therefore optimistically biased. The held-out test result is unaffected, because the test set plays no part in selection. Fixing this means nesting selection inside each fold and re-running every model comparison. See `docs/open_questions.md` (Q1, Q17). It is not fixed here.

### Inputs
* **Required Files:** `train_processed.csv` from Step 6a, in `data/processed/recording/` or `data/processed/segment/`.
* **File Formats:** `.csv`

### Outputs
* **Generated Outputs:** The per-feature statistics table and the selected feature list.
* **File Format:** `.csv`
* **Location:** `data/processed/{recording,segment}/`
* **Structure:** `feature_stats.csv` has one row per feature with Cohen's d, the Welch t-test p-value, and per-class means. `selected_features.csv` is the ordered list of surviving feature names.

### Configuration
* `selection.d_threshold` (0.2), `selection.corr_threshold` (0.85), `selection.report_top_n` (30).
* `selection.significance_alpha`, `effect_size_large` and `effect_size_medium` affect the printed summary only, never which features survive.
* Segment equivalents under `segments.selection:`.

### Dependencies
* `pandas`, `numpy`, `scipy`, `pyyaml`.

### Execution Instructions
```bash
python Step6_Modeling/Step6b_Feature_Selection/feature_selection_recording.py
```
```bash
python Step6_Modeling/Step6b_Feature_Selection/feature_selection_segment.py
```
