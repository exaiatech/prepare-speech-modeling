# Train/Test and Cross-Validation Strategy

Precise account of the partitioning and cross-validation used by both pipeline
variants, as requested by the supervisor on 2026-07-30 — including the
requested confirmation that all recordings and segments of one participant
stayed within one partition.

**That confirmation cannot be given.** Section 4 explains exactly what can and
cannot be asserted, and why. Sections 1-3 describe what the code actually does.

---

## 1. The train/test split

The split is **not** created by this project. It is the `split` column of
`metadata.csv`, as distributed with the PREPARE Challenge. No recording is
reassigned, and no validation split exists.

| Partition | Recordings | Segments | Used for |
| --- | --- | --- | --- |
| train | 1183 | 8232 | feature selection, scaling, model fitting, threshold choice |
| test | 298 | 2056 | one final evaluation |

The test set is not touched by `03b_build_matrix.py` beyond having the
train-fitted median and scaler applied to it, and is not touched at all by
`04_feature_selection.py`.

---

## 2. Whole-recording pipeline (`../src/05_train_evaluate.py`)

**`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`** over the 1183
training recordings.

- One recording is one row, so there is no within-recording leakage by
  construction.
- Folds are stratified on the CN/AD label. **They are not grouped by anything.**
- Part A compares four models and reports the mean of the per-fold metrics at
  the fixed threshold 0.5.
- Part B refits the chosen model (SVM rbf) on all 1183 training recordings.
  Its decision threshold is taken from out-of-fold train predictions using
  Youden's J, then applied unchanged to the test set.

Reported: **test AUC 0.664.**

---

## 3. Segment pipeline (`../src/segments/05_segment_train_evaluate.py`)

**`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42, groups=uid)`**
over the 8232 training segments.

- `groups=uid` guarantees that **all segments of one recording land in the same
  fold**. A model is never validated on a recording it partially trained on.
  This is a genuine improvement over the whole-recording pipeline's plain
  `StratifiedKFold`, and it is the main methodological reason the segment
  variant is more defensible.
- Each segment receives exactly one out-of-fold probability.
- Segment probabilities are averaged per `uid` (arithmetic mean, unweighted) to
  give one probability per recording. **All reported metrics are computed at
  that aggregated level**, so they are directly comparable to the
  whole-recording numbers.
- The Youden threshold is derived from the aggregated out-of-fold train
  predictions, then applied unchanged to the aggregated test predictions.

Reported: **test AUC 0.682**, threshold 0.338, sensitivity 0.614,
specificity 0.668, balanced accuracy 0.641, over 298 test recordings.

> **Terminology.** The original notebook calls the aggregated level "person
> level". The refactored scripts call it **recording level**, because the
> grouping key `uid` identifies a recording. The two are only the same thing if
> every participant contributed exactly one recording, which is unverified.
> See section 4.

---

## 4. The participant-partition question

> *"Confirmation that all recordings and segments from the same participant
> were kept within the same data partition."*

### What can be confirmed

- **Segments never cross folds within a recording.** Guaranteed by
  `StratifiedGroupKFold(groups=uid)` in the segment pipeline. Verifiable from
  the code and reproducible.
- **Segments never cross the train/test boundary.** Segmentation is applied
  after the split, separately per partition, so a recording's segments are all
  on one side.
- **No recording appears in both train and test.** `split` is a per-uid
  attribute.

### What cannot be confirmed

**Whether one participant's recordings stayed together — in the train/test
split, or across cross-validation folds.**

`metadata.csv` has no participant identifier. `uid` is unique per recording,
`hash` is unique per recording. There is no key on which two recordings of the
same person could be joined. See
[`dataset_description.md`](dataset_description.md) section 5.

The consequence is asymmetric between the two pipelines but present in both:

| | Segment leakage | Participant leakage |
| --- | --- | --- |
| Whole-recording | not applicable (one row per recording) | **possible, unverifiable** |
| Segment | prevented by `groups=uid` | **possible, unverifiable** |

If a participant contributed several recordings, the models can be validated on
a person they were partly trained on, in both variants. The 0.682 result would
then be optimistic — and it is exactly the result the poster leads with.

The risk is not hypothetical. 507 of the 1481 cohort recordings come from the
Pitt corpus (DementiaBank), which is longitudinal by design and does contain
repeated recordings per participant in its original form.

### What would make the confirmation possible

Any participant-to-`uid` mapping would turn this from unverifiable into a
one-line check, and would let `StratifiedGroupKFold(groups=participant_id)`
replace `groups=uid` in the segment pipeline and be introduced in the
whole-recording pipeline. Concretely, one of:

1. A participant identifier from the PREPARE organisers.
2. Original per-corpus participant IDs (at minimum for the Pitt subset) plus
   the mapping to `uid`.
3. Written confirmation that `uid` is one-to-one with participant **and** that
   the challenge's train/test split was made participant-wise.

Until then, the honest formulation for the paper is that **recording-level and
segment-level independence are guaranteed, participant-level independence is
not established**.

Tracked as `open_questions.md` Q2, Q16 and Q23.

---

## 5. Where the test set is used more than once

Step 05 of both pipelines documents the test set as being evaluated exactly
once. The interpretability step then uses it again:

- `../src/06_interpret.py` runs permutation importance on the test set with 30
  repeats and draws its SHAP explanation sample from the test set.
- `../src/segments/06_segment_interpret.py` runs a recording-level permutation
  loop with 15 repeats and full TreeExplainer SHAP, both on the test set.

No model or threshold is selected on the basis of those runs, so the headline
AUCs are not invalidated. But the phrase "the test set is used exactly once" is
not accurate as written, and either the wording or the analysis should change
before publication. Tracked as `open_questions.md` Q3 and Q18.

---

## 6. Reproducibility

A single seed, `seed: 42` in `config.yaml`, feeds every stochastic component:
the fold shuffling of `StratifiedKFold` and `StratifiedGroupKFold`, every
estimator's `random_state`, the permutation shuffles, and the SHAP sample draw.
No other source of randomness exists in either pipeline.
