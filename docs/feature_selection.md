# Feature Selection Procedure

Precise account of how the 135 extracted features are reduced, as requested by
the supervisor on 2026-07-30: quality-control and missingness criteria, the
screening criteria, the redundancy threshold, the order of the steps, and
whether selection happened inside or outside cross-validation.

Both pipeline variants use the same procedure with independently configured
thresholds.

| | Whole-recording | Segment |
| --- | --- | --- |
| Implemented in | `../src/03b_build_matrix.py`, `../src/04_feature_selection.py` | `../src/segments/03b_build_segment_matrix.py`, `../src/segments/04_segment_feature_selection.py` |
| Config block | `matrix`, `selection` | `segments.matrix`, `segments.selection` |
| Observation unit | one recording | one 5 s segment |
| Started from | 135 | 135 |
| After missingness filter | **134** | **135** (none dropped) |
| Candidates after effect size | 40 | 34 |
| **Final** | **29** | **25** |

---

## The four steps, in the order they are applied

### Step 1 — Missingness filter (quality control)

*Where:* `03b_build_matrix.py`, before any statistics.

A feature is dropped if more than **30%** of its values are missing **in the
train set** (`matrix.max_missing_fraction: 0.3`). The test set has no vote:
a feature that is missing everywhere in test but present in train is kept.

Missing values arise inside the extractor, not from the data files:
`stats6()` returns `NaN` when fewer than two valid frames survive, formant
tracking returns `NaN` when Burg analysis fails, `f0_std_norm` is `NaN` when
`f0_mean <= 0`, and the voice-quality block yields `NaN` when no analysis
window contains at least four glottal periods.

In the poster run this dropped **one** feature in the recording pipeline and
**none** in the segment pipeline.

There is no other quality control. In particular there is **no** outlier
removal, no minimum-duration criterion on recordings, no signal-quality or SNR
screening, and no manual exclusion of recordings.

### Step 2 — Imputation and scaling

*Where:* `03b_build_matrix.py`, immediately after step 1.

1. Remaining `NaN`s are filled with the **train median** of that feature. The
   same train median is applied to the test set.
2. All surviving features are scaled with scikit-learn's `RobustScaler`
   (median centring, IQR scaling), **fitted on the train set only** and applied
   unchanged to test.

The fitted scaler and the surviving column list are persisted
(`scaler.pkl`, `feature_cols.pkl`) so later steps cannot re-derive them
inconsistently.

This step selects nothing, but it defines the values that steps 3 and 4 see.

### Step 3 — Effect-size and significance screening

*Where:* `04_feature_selection.py`, first half (notebook cells 14 / F3).

For every surviving feature, on the **scaled train set**, comparing CN vs AD:

- **Cohen's d** with a pooled standard deviation:
  `d = (mean_AD - mean_CN) / sqrt((var_CN + var_AD) / 2)`,
  using population variances (`ddof=0`). Sign indicates direction; screening
  uses `|d|`.
- **Welch's t-test** (`scipy.stats.ttest_ind(..., equal_var=False)`), and a
  boolean `significant` at α = 0.05.

Both are written to `feature_stats.csv` / `feature_stats_segments.csv`.

**The screening criterion is the effect size alone:** features with
`|d| >= 0.2` (`selection.d_threshold`) become candidates. The p-value is
recorded and reported but **does not gate anything** — no feature is included
or excluded because of it, and there is **no multiple-comparison correction**,
because none is needed for a criterion that is never applied.

This yielded 40 candidates (recording) and 34 candidates (segment).

> In the segment pipeline the p-values are additionally invalid through
> pseudo-replication: the ~7 segments of one recording are treated as
> independent observations, so n is inflated roughly sevenfold and every
> p-value collapses toward zero. The original notebook flags this itself.
> Since p gates nothing, the selection result is unaffected. See
> `open_questions.md` Q17.

### Step 4 — Correlation pruning (redundancy reduction)

*Where:* `04_feature_selection.py`, second half (notebook cells 15 / F4).

Threshold: **|r| > 0.85** (`selection.corr_threshold`), Pearson correlation on
the scaled train set, computed among the candidates only.

Greedy algorithm, walking the candidate list from the largest `|d|` to the
smallest:

1. Skip any feature already marked as dropped.
2. If the feature correlates above threshold with any feature already **kept**,
   drop it.
3. Otherwise keep it, then scan the remaining candidates and mark as dropped
   any that correlate above threshold with the feature just kept.

Because the walk is ordered by descending `|d|`, whenever two features are
redundant the one with the **smaller** effect size is the one discarded. The
result is order-dependent by construction, but the order is deterministic.

Final: **29** features (recording), **25** features (segment). Written to
`selected_features.csv` / `selected_features_segments.csv`.

---

## Was selection performed inside each fold, or on the complete dataset?

**On the complete training set, before cross-validation. In both pipelines.**

This is the answer to the project's first open question, and it is
unambiguous. The execution order is:

```
03b_build_matrix        train median + RobustScaler   fitted on all of train
04_feature_selection    Cohen's d, correlation pruning computed on all of train
                        -> selected_features.csv          a fixed list on disk
05_train_evaluate       StratifiedKFold / StratifiedGroupKFold reads that list
```

Every cross-validation fold in step 05 is therefore validated on samples that
already contributed to choosing the features being evaluated — and, through the
train median and the RobustScaler in step 03b, to the preprocessing as well.

### What this does and does not compromise

- **Reported cross-validation AUCs are optimistically biased.** The bias is not
  quantified. With 135 candidates screened down to 25-29 on 1183 recordings the
  effect is real but unlikely to be dramatic; only a nested run can say.
- **The final test-set numbers (0.664 and 0.682) are less affected but not
  clean.** The test set never entered the selection, the median, the scaler, or
  the threshold choice. But the feature list was chosen using all of the train
  data, which is the normal and accepted practice for a held-out test set. The
  test AUCs are the more trustworthy of the two families of numbers reported.
- **The model comparison in step 05 Part A is the weakest part.** All four
  models are compared on folds whose feature list was tuned on the full train
  set, so the winner may be the model that best exploits that shared advantage.

### What a corrected version would require

Moving steps 1-4 inside the outer CV loop: per fold, recompute the missingness
filter, the median, the scaler, the effect sizes and the correlation pruning on
that fold's training half only, then evaluate on its held-out half. The final
model would still be fitted on all of train with a list selected on all of
train, and evaluated once on test.

This has **not** been done. It would change every cross-validation number in
the poster, so it is a decision for the supervisor, tracked as
`open_questions.md` Q1 (recording) and Q17 (segment).
