# Step 6c: Train and Evaluate

### Purpose
Cross-validates the candidate classifiers, fixes the decision threshold, fits the final model, and evaluates it once on the held-out test set.

| | Recording level | Segment level |
| --- | --- | --- |
| Cross-validation | `StratifiedKFold` | `StratifiedGroupKFold`, grouped by `uid` |
| Final model | SVM (RBF) | Random Forest |
| Features | 29 | 25 |
| Prediction level | recording | segment, averaged to recording |
| **Test AUC** | **0.664** | **0.682** |
| Sensitivity | 0.535 | 0.614 |
| Specificity | 0.745 | 0.668 |
| Balanced accuracy | 0.640 | 0.641 |

The segment path shows near-perfect train-test consistency: train CV AUC 0.681 against test 0.682.

> **Grouping is not optional at segment level.** Segments from one recording are not independent. `StratifiedGroupKFold(groups=uid)` keeps every segment of a recording inside one fold. Without it the CV score is meaningless.

> **Threshold tuning.** The operating threshold comes from Youden's J computed on **out-of-fold training predictions**, never on the test set. The default 0.5 cutoff collapsed sensitivity in several models, which is why this is a standard part of training rather than a separate experiment.

> **The test set is evaluated exactly once**, after the model and threshold are fixed on training data.

### Inputs
* **Required Files:** `train_processed.csv`, `test_processed.csv` and `selected_features.csv` from Steps 6a and 6b.
* **File Formats:** `.csv`

### Outputs
* **Generated Outputs:** The cross-validation comparison table printed to stdout, and the fitted final model plus its threshold saved to disk.
* **File Format:** joblib `.pkl`
* **Location:** `data/processed/{recording,segment}/`
* **Structure:** The saved model is the fitted estimator; the companion file records the chosen threshold so Step 6d applies the same one.

### Configuration
* `train.n_folds` (5), `train.shuffle`, `train.final_model` (`svm_rbf` at recording level, `random_forest` at segment level), `train.default_threshold`.
* Per-model hyperparameters under `train.models:`. All use `class_weight: balanced`, since the cohort is 944 Control against 537 Alzheimer's.
* `segments.train.aggregation` (`mean`) controls how segment probabilities become a recording probability.
* `seed` (42) feeds fold shuffling and every estimator.

> Hyperparameter tuning (nested CV with XGBoost) and ensembling (averaging, stacking) were both tested and gave no meaningful improvement. They are archived as negative results in `archive/dead_code/05_train_evaluate/`. Do not re-run them without a specific reason.

### Dependencies
* `scikit-learn`, `pandas`, `numpy`, `joblib`, `pyyaml`.

### Execution Instructions
```bash
python Step6_Modeling/Step6c_Train_Evaluate/train_evaluate_recording.py
```
```bash
python Step6_Modeling/Step6c_Train_Evaluate/train_evaluate_segment.py
```
