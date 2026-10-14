# Step 7b: Evaluation

### Purpose
Computes every metric from **one predictions file for one model**. The same script serves every model, so results are computed
identically across them.

Binary: accuracy, balanced accuracy, macro F1, AUC, per-class precision and recall. Three-class: the same plus MCI recall and the
one-vs-rest AUC (`auc_ovr`). The CV summary is the mean and standard deviation over the 50 splits. The test result is a single
evaluation. A balanced-accuracy-by-corpus breakdown is printed for comparison with the corpus-only baseline.

> **Status: run and verified by the author.** Re-checked on a synthetic three-class predictions file during the cleanup: the
> output contains `accuracy`, `balanced_accuracy`, `macro_f1`, `mci_recall` and `auc_ovr`.

### Input format
One row per recording per split, with columns `ID, split_id, role, y_true, p_control, p_mci, p_adrd`. `role` is `val` for a CV
row and `test` for the held-out run. `y_true` is `Control`, `MCI` or `AD/ADRD`. `p_mci` may be absent for a binary model.

### Inputs and outputs
* Reads `<stage1.predictions_dir>/<MODEL_NAME>_<TASK>_predictions.csv` and the corpus from `stage1.inclusion_list`.
* Writes `<RUN>_per_split.csv`, `<RUN>_cv_summary.csv` and `<RUN>_test_result.csv` to `stage1.results_dir`.

### Configuration
`MODEL_NAME` and `TASK` are constants at the top of the script and must match the predictions file (for example
`whisper_medium`, `speechcare`, `pause_bert`, with `binary` or `multiclass`). Paths come from `stage1.*` in `config.yaml`.

### Execution Instructions
```bash
python src/Step7_Evaluation/Step7b_Evaluate/03_evaluate.py
```
