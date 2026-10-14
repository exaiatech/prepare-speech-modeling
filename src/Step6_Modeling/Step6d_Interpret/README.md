# Step 6d: Interpretability

> **Status after the supervisor's review.**
> * `interpret_segment.py` was **set aside**, with the other segment-level scripts. It is in `archive/set_aside_after_review/src/Step6_Modeling/Step6d_Interpret/`.
> * `interpret_recording.py` stays here but has nothing to read: it needs the model and matrices written by Step 6c and Step 6a, whose scripts were also set aside. The supervisor's table did not cover this script, so it was left as it is. Its future is an open question.
> * The description below is of the earlier analysis. The redesign (fold-internal imputation, scaling and feature selection) will replace it.

### Purpose
Explains what the fitted models actually use, through SHAP values and permutation importance.

`spectral_slope_0_500_mean` ranks **first by SHAP at both levels** and is the single most consistent biomarker across four independent analyses. Segment level additionally surfaces intensity distribution shape (skewness and kurtosis) as discriminative, which is invisible when averaging over a whole recording.

The two scripts use different explainers because the two final models differ:

| | Recording level | Segment level |
| --- | --- | --- |
| Model | SVM (RBF) | Random Forest |
| SHAP explainer | `KernelExplainer` | `TreeExplainer` |
| Permutation unit | recording | recording (aggregated from segments) |

`KernelExplainer` is model-agnostic and slow, which is why the recording-level script explains a sample of `interpret.n_shap_samples` (150) test rows against a k-means-compressed background of 50 points rather than the full matrix.

> **Caveat.** Both interpretability scripts reuse the test set that Step 6c describes as evaluated only once. Interpretation does not change the model or the reported metrics, but it does mean the test set has been looked at more than once. See `docs/open_questions.md` (Q3, Q18).

### Inputs
* **Required Files:** The fitted model and threshold from Step 6c, the processed matrices from Step 6a, and `selected_features.csv` from Step 6b.
* **File Formats:** `.csv` and joblib `.pkl`

### Outputs
* **Generated Outputs:** Feature importance rankings.
* **File Format:** `.csv`
* **Location:** `data/processed/{recording,segment}/`
* **Structure:** One row per selected feature with its mean absolute SHAP value and its permutation importance, ordered by importance.

### Configuration
* `interpret.n_shap_samples` (150), `interpret.kmeans_background` (50), `interpret.permutation_repeats` (30), `interpret.permutation_scoring` (`roc_auc`), `interpret.top_n_plot` (20).
* Segment equivalents under `segments.interpret:`, with fewer permutation repeats (15) because the segment matrix is much larger.
* `seed` (42) feeds the permutation shuffles and the SHAP sample draw.

### Dependencies
* `shap`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `matplotlib`, `pyyaml`.

### Execution Instructions
Requires Step 6c to have run and saved a model.
```bash
python src/Step6_Modeling/Step6d_Interpret/interpret_recording.py
```
The recording-level script is slow: `KernelExplainer` is model-agnostic and evaluates the SVM many times per explained row.
