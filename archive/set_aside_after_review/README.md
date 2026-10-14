# Set aside after the supervisor's review

Scripts, tables and READMEs that were taken off the active path after the supervisor reviewed the code. They were
moved with `git mv`, so their history is intact, and each sits under the path it had before, relative to the
repository root. **None of this is deleted and none of it is deprecated for good**: the recording-level modelling path
will be redesigned when interpretable modeling starts.

> To be redesigned when interpretable modeling starts, with fold-internal imputation, scaling and feature selection.

These files do not run from here. The scripts expect to live under `src/` and import from `src/common/`.

## Segment-level extraction and modelling

Supervisor's reason: one observation and one prediction per recording. SpeechCARE's 5 second windows are internal
encoder inputs, not a reason to create a separate window-level feature dataset.

| File | What it was |
| --- | --- |
| `src/Step2_Acoustic/Step2b_Feature_Extraction/extract_segment_level.py` | The 135 features per 5 s window with 25% overlap |
| `src/Step6_Modeling/Step6a_Build_Matrix/build_matrix_segment.py` | Merge, impute and scale the segment features |
| `src/Step6_Modeling/Step6b_Feature_Selection/feature_selection_segment.py` | Effect-size filter and correlation pruning on segments |
| `src/Step6_Modeling/Step6c_Train_Evaluate/train_evaluate_segment.py` | Grouped cross-validation, final model, recording-level aggregation |
| `src/Step6_Modeling/Step6d_Interpret/interpret_segment.py` | SHAP and permutation importance for the segment model |
| `data/results/descriptive_stats_segment.csv` | Per-class descriptive statistics over segments |

## Recording-level modelling path

Supervisor's reason: it does not implement the published encoders, and its cross-validation estimates are affected by
imputation, scaling and label-based feature selection performed before the folds.

| File | What it was |
| --- | --- |
| `src/Step6_Modeling/Step6a_Build_Matrix/build_matrix_recording.py` | Merge, impute and scale the recording features |
| `src/Step6_Modeling/Step6b_Feature_Selection/feature_selection_recording.py` | Effect-size filter and correlation pruning |
| `src/Step6_Modeling/Step6c_Train_Evaluate/train_evaluate_recording.py` | Cross-validation, final model, held-out test |

## READMEs that described the files above

| File | What it was |
| --- | --- |
| `src/Step6_Modeling/Step6a_Build_Matrix/README.md` | README of the matrix-building step, both variants |
| `src/Step6_Modeling/Step6b_Feature_Selection/README.md` | README of the feature-selection step, both variants |
| `src/Step6_Modeling/Step6c_Train_Evaluate/README.md` | README of the training step, both variants |

No matrices, selected-feature lists or model result tables were tracked by git, so none are archived here.

## Shared modules that stay in `src/common/`

`src/common/selection.py` is imported by the two feature-selection scripts above and by nothing else, and
`extract_all_segment()` in `src/common/feature_functions.py` is used only by the segment extractor. Both stay where
they are, because they are shared modules and the redesign may reuse them.
