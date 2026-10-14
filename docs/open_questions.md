# Open Questions

Issues found while restructuring `features.ipynb` into `acoustic_pipeline/`.

**Nothing in this list was fixed.** The refactor moved code and parameters only;
the analysis logic is byte-for-byte the logic that produced the poster numbers
(SVM RBF AUC 0.664). Every item below changes the reported results if acted on,
so each one is a deliberate decision for Arday and Daniel Wiechmann to make.

Recorded: 2026-07-30.

---

## Q1 - Feature selection happens outside cross-validation

`04_feature_selection.py` (notebook cells 14+15) computes Cohen's d and the
correlation pruning **once, on the entire train set**. `05_train_evaluate.py`
then cross-validates using that fixed list. Every CV fold's validation samples
therefore contributed to choosing the features being evaluated.

This is the project's first open question, and the answer is: selection was
done on the full dataset before CV, not inside each fold. The reported AUCs are
optimistically biased to an unknown degree.

**Fix would be:** move steps 04 into the outer CV loop (nested CV), recompute all
metrics. This changes every number in the poster.

**Affects:** `04_feature_selection.py`, `05_train_evaluate.py`.

---

## Q2 - No participant-level grouping in the CV splits

`05_train_evaluate.py` uses `StratifiedKFold`, which balances the CN/AD ratio but
does not keep a participant's recordings together. If any participant contributes
more than one recording to the English subset, their recordings can land in both
the training and the validation half of a fold.

This is the project's second open question. It cannot be answered from the
notebook alone — it depends on whether `metadata.csv` has repeated participants
per `uid`. For the segment-based analysis (which is not in this notebook at all,
see Q12) the risk is much higher, since all segments share a speaker.

**Fix would be:** `StratifiedGroupKFold` with a participant ID as the group.
Requires a participant column in the metadata.

**Affects:** `05_train_evaluate.py`.

---

## Q3 - The test set is used more than once

`05_train_evaluate.py` documents that the test set is touched exactly once, for
final reporting. But `06_interpret.py` (notebook cell 21) then runs
`permutation_importance` on the **test set** with 30 repeats, and draws the SHAP
explanation sample from the test set as well.

Either the "evaluated once" claim needs softening in the paper, or the
interpretability analysis should move to the train set / out-of-fold predictions.

**Affects:** `05_train_evaluate.py` docstring, `06_interpret.py`.

---

## Q4 - Residual leakage in the archived stacking ensemble

`archive/dead_code/05_train_evaluate/ensembles.py` (cell 19) builds an OOF
probability matrix over all samples, then cross-validates a meta logistic
regression on that same matrix. The cell's own comment argues there is no
leakage, but each OOF column was produced by models trained on the other folds
of the *whole* dataset, so label information from the meta-model's validation
samples has already influenced the columns it is validated on.

Archived and unused, so this is informational — flagged in case the ensemble is
revived for the paper.

---

## Q5 - Power ratios use `20*log10` instead of `10*log10`

In `03_extract_features.py`, `spectral_shape_features()` computes:

```python
power = S ** 2
alpha_ratio = 20 * np.log10(e_low / e_high)     # e_* are sums of `power`
hammarberg  = 20 * np.log10(peak1 / peak2)      # peaks of `power`
```

`20*log10` is the amplitude-to-dB conversion. These two ratios are already
**power** ratios (`S**2`), so the eGeMAPS definition is `10*log10`. As written,
`alpha_ratio_*` and `hammarberg_index_*` are exactly twice their intended dB
value.

This is a uniform scale factor, so it does not change feature rankings or any
correlation, and it is absorbed by the RobustScaler in `03b_build_matrix.py`.
It **does** make the absolute values non-comparable to published eGeMAPS numbers,
which matters if these features are reported in the paper.

Note that `band_slope()` in the same function correctly uses `20*log10` — it
operates on the amplitude spectrum `S`, not on `power`.

**Affects:** `03_extract_features.py`. 4 of the 135 features.

---

## Q6 - Preprocessing was written for `.py` files, not for the notebook

Cells 4, 5, 22, 23 and 24 all begin with:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
```

`__file__` is undefined in a Jupyter cell, so **these cells raise `NameError` and
cannot have run inside the notebook as they stand**. They were pasted in from
standalone scripts. It is therefore unclear which version of the preprocessing
actually produced any data on disk, and whether it was run at all.

**Affects:** `01_preprocess.py` and the two archived variants.

---

## Q7 - `find_audio` can match the wrong recording

`03_extract_features.py`:

```python
def find_audio(uid, exact, prefix_list):
    if uid in exact:
        return exact[uid]
    for fname, path in prefix_list:
        if fname.startswith(uid):
            return path
```

The prefix fallback returns the first filename that merely *starts with* the uid.
If one uid is a prefix of another (e.g. `abc` and `abcd`), the shorter uid can
silently be assigned the longer uid's audio — a label-swap that no error message
would reveal. Whether this can actually happen depends on the uid format in
`metadata.csv`; worth verifying once.

**Affects:** `03_extract_features.py`.

---

## Q8 - The VAD loop discards the final frame

`02_vad_segment.py`, `label_frames()`:

```python
for i in range(0, len(pcm) - frame_len, frame_len):
```

The stop bound should be `len(pcm)` (or `len(pcm) - frame_len + 1`) to cover the
last complete frame. As written, up to 40 ms at the end of every recording is
never classified. The effect on the pause statistics is tiny but systematic, and
it is larger for short recordings.

**Affects:** `02_vad_segment.py`, and through it all 18 pause features.

---

## Q9 - Empty TextGrid labels are counted as speech

`archive/dead_code/03_extract_features/pilot_43_features_praat_pauses.py`
(cell 10) classifies intervals with:

```python
(pauses if label == "silence" else speech_segs).append(dur)
```

Anything not literally labelled `"silence"` — including empty-label intervals —
becomes speech. This is part of the pause-detection approach that was already
found to be measuring background noise, and it is archived. Recorded only so the
reason for the switch to WebRTC VAD is documented.

Note that the surviving `articulation_features()` in `03_extract_features.py`
uses the safer form (`if label == "speech"`), so the active pipeline is fine.

---

## Q10 - Seeds that did not exist in the notebook

Per the repository rules, all seeds now come from `config.yaml` (`seed: 42`,
which is the notebook's `RANDOM_STATE`). Two random operations in the notebook
had **no** seed:

1. `archive/dead_code/data_exploration/random_sample_copy.py` (cell 2) —
   `corpus_data.sample(n=pick_count)` with no `random_state`. The listening
   sample it produced is not reproducible.
2. The boxplot jitter in the archived pilot extractors (cells 10, 11) —
   `np.random.uniform`, cosmetic only.

Both are in archived code, which was copied verbatim and deliberately left
unmodified. **No seed was added to them**, because adding one would change which
files `random_sample_copy.py` selects. If any of that code is ever revived, seed
it from the config at that point.

The active pipeline has no unseeded randomness.

---

## Q11 - Preprocessing is not wired into the pipeline

`01_preprocess.py` writes filtered/normalized audio to
`paths.preprocess.output_dir`, but `03_extract_features.py` reads
`paths.raw_audio.train` / `paths.raw_audio.test` — the **raw downloads**. Nothing
consumes the preprocessing output.

This is exactly how the notebook behaved, and it was preserved. So one of two
things is true, and only Arday can say which:

- the 135 features were extracted from **raw, unfiltered, un-normalized** audio,
  and the preprocessing stage is documentation of an intent that was never
  applied; or
- preprocessing was run manually at some point and the extractor's input paths
  were supposed to point at its output.

Note that `03_extract_features.py` does resample to 16 kHz itself, at load time
(`librosa.load(..., sr=16000)`), so resampling does happen regardless — but only
for the librosa-based features. The Parselmouth features (`f0`, intensity,
articulation, formants, voice quality) are computed from
`parselmouth.Sound(filepath)`, which reads the file at its **original** sample
rate. The two halves of the feature set therefore see different signals.

**Affects:** `01_preprocess.py`, `03_extract_features.py`, `config.yaml`.

---

## Q12 - The segment-based analysis is not in the notebook — CLOSED

**Closed 2026-07-30.** The missing code was in a second notebook,
`features2.ipynb`, now archived as `archive/features2_original.ipynb` and
refactored into `acoustic_pipeline/src/segments/` (cells F1-F7).

`segments/05_segment_train_evaluate.py` (cells F5+F6) is the script that
produces the reported result: test AUC **0.682**, threshold 0.338,
sensitivity 0.614, specificity 0.668, balanced accuracy 0.641, over 298 test
recordings. The notebook's stored outputs confirm this end to end, and its
cells were executed linearly.

Both results are now runnable from this repository: 0.664 from
`src/05_train_evaluate.py`, 0.682 from
`src/segments/05_segment_train_evaluate.py`.

The segment code raised its own questions in turn: Q15-Q23 below.

Note that `02_vad_segment.py` still does not perform segmentation — it computes
VAD-based pause statistics, and is called *inside* each 5 s segment. The actual
segmentation is fixed-window and lives in
`segments/03_extract_segment_features.py`. The file name is kept for continuity
with the agreed structure.

---

## Q13 - No Cookie Theft-only subset

The project's third open question: the dataset mixes elicitation tasks and a
Cookie Theft-only subset needs to be definable. `config.yaml`'s
`features.metadata_filters` currently filters on language, split and diagnosis
only. Adding a task filter requires knowing which `metadata.csv` column encodes
the elicitation task — it is not referenced anywhere in the notebook.

---

---

# Questions raised by the segment pipeline (`features2.ipynb`)

Added 2026-07-30, when `features2.ipynb` was integrated. Same rule as above:
nothing here was fixed.

---

## Q15 - The two pipelines do not see the same signal

The whole-recording extractor builds its Praat object from the file:

```python
y, _ = librosa.load(filepath, sr=16000, mono=True)   # librosa: 16 kHz
snd   = parselmouth.Sound(filepath)                  # Praat: ORIGINAL rate
```

The segment extractor builds it from the already-resampled array:

```python
y, _ = librosa.load(filepath, sr=16000, mono=True)
snd  = parselmouth.Sound(y_seg, sampling_frequency=16000)   # Praat: 16 kHz
```

So the segment pipeline is internally consistent and the whole-recording
pipeline is not (that inconsistency is Q11). But it also means the two
pipelines compute the *same feature definitions on different signal
representations*.

Consequence for the paper: **0.664 vs 0.682 is not a clean ablation of
segmentation.** The comparison confounds three changes at once — unit of
analysis (recording vs 5 s window), classifier (SVM rbf vs Random Forest), and
sample-rate handling of the Praat features. Any claim that segmentation
improves performance needs at least one of the confounds removed.

**Affects:** `03_extract_features.py`, `segments/03_extract_segment_features.py`.

---

## Q16 - "Person level" is really recording level

The segment pipeline groups and aggregates by `uid`:

```python
StratifiedGroupKFold(...).split(X, y, groups=df['uid'].values)
agg.groupby('uid').agg(y=('y', 'first'), proba=('proba', 'mean'))
```

`uid` identifies a **recording**. The original notebook labels this level
"person" throughout — in variable names (`person_auc`, `n_people`,
`to_person_level`), in printed output ("People: 1182", "People in test set:
298") and in the docstrings. The refactored scripts renamed it to
"recording level"; the numbers are unchanged.

The two are the same thing only if every participant contributed exactly one
recording. That is unverified and unverifiable from the current metadata
(Q23). Until it is established, the poster and paper should say
"recording-level AUC", not "person-level AUC", and should not describe 298 as a
number of people.

**Affects:** `segments/05_segment_train_evaluate.py`,
`segments/06_segment_interpret.py`, and the poster wording.

---

## Q17 - Feature selection outside CV, again — plus pseudo-replication

The segment pipeline repeats Q1: `segments/04_segment_feature_selection.py`
runs once on the full train segment set, and
`segments/05_segment_train_evaluate.py` then cross-validates using that fixed
list of 25 features.

It adds a second problem on top. Cohen's d and the Welch t-test treat each of
the 8232 segments as an independent observation, but they come from only 1182
recordings — roughly 7 correlated segments each. The effective sample size is
inflated about sevenfold, which is why every one of the top 30 features has
`p = 0.0000`. The notebook flags this itself.

Mitigating factor: the screening criterion is `|d| >= 0.2`, and `d` is a
standardised mean difference that does not grow with n. The p-values gate
nothing. So the selected feature set is not corrupted by the inflation — but
the `significant` column in `feature_stats_segments.csv` is meaningless and
must not be reported.

**Affects:** `segments/04_segment_feature_selection.py`. See also
`docs/feature_selection.md`.

---

## Q18 - The segment test set is also used twice

`segments/06_segment_interpret.py` runs both its recording-level permutation
loop (15 repeats over 25 features) and its TreeExplainer SHAP on the test set,
after `segments/05_segment_train_evaluate.py` has declared the test set
evaluated once. Identical in substance to Q3.

---

## Q19 - The two pipelines end on different feature spaces

| | Recording | Segment |
| --- | --- | --- |
| Dropped by the >30% missing filter | 1 | 0 |
| Entering selection | 134 | 135 |
| Candidates at \|d\| >= 0.2 | 40 | 34 |
| Final | **29** | **25** |

The final sets are not nested and were selected independently. Feature-level
statements ("spectral slope 0-500 Hz is the top SHAP feature") happen to hold
in both, but any other cross-pipeline feature comparison needs checking against
both `selected_features.csv` and `selected_features_segments.csv`.

The differing missingness outcome is itself informative: a feature that fails
on whole recordings can succeed on 5 s windows, and vice versa.

---

## Q20 - One executed cell is missing from `features2.ipynb`

The first cell of the notebook, F1, carries `execution_count = 2`. Cells F1-F7
then run 2, 3, 4, 5, 6, 7, 8 without a gap. Something was executed as cell 1 in
that kernel session and is no longer in the file.

F1 imports everything it needs, so the pipeline is very likely self-contained
and the missing cell was probably a scratch or install cell. But the notebook
does not represent its full execution history, and this should be confirmed
before the notebook is cited as the provenance of 0.682.

---

## Q21 - The segment pipeline also ignores `01_preprocess.py`

Like the whole-recording extractor (Q11),
`segments/03_extract_segment_features.py` reads `paths.raw_audio.*` directly.
The band-pass filtering and peak normalisation of `01_preprocess.py` are
applied to neither pipeline. Whatever is decided for Q11 applies here too.

---

## Q22 - A numerical warning is suppressed in the segment extractor

`segments/03_extract_segment_features.py` suppresses
`"Precision loss occurred in moment calculation"`, carried over from the
notebook (`segments.segmentation.suppress_warning` in `config.yaml`).

The warning comes from `scipy.stats.kurtosis` / `skew` when a distribution is
nearly degenerate — which is far more likely on a 5 s window than on a whole
recording, since some frame-level arrays become very short. It is suppressed
rather than investigated, and the whole-recording pipeline does not suppress
it.

It is worth checking how many segments actually trigger it and whether the
affected kurtosis/skewness features are stable, before those features are
reported. Removing the filter changes no numbers, only the log output.

---

## Q23 - No participant identifier and no elicitation task in the metadata

Raised by the supervisor request of 2026-07-30 and confirmed against the data:
`metadata.csv` has 2058 rows and 12 columns, one row per `uid`, and contains

- **no participant identifier** (`hash` is unique per row, i.e. per file);
- **no elicitation-task column**.

Therefore three of the requested deliverables cannot be produced: which
recordings belong to the same participant, how many recordings each participant
contributed, and which speech task each recording used. The corresponding
columns in `data/recording_metadata.csv` are emitted empty rather than guessed.

This subsumes Q2 (participant leakage, whole-recording) and Q16 (recording vs
person level, segment) — neither can be resolved without new information, and
both are resolved immediately if a participant mapping is obtained.

`00_build_recording_metadata.py` emits a heuristic collision diagnostic
(`corpus + age + gender + diagnosis`): 361 distinct keys for 1481 recordings,
1312 recordings sharing a key with at least one other, largest group 132. That
establishes only that repeats **cannot be ruled out**; it is not an identity
claim and must never be used as a CV grouping key.

Resolving it needs one of: a participant identifier from the PREPARE
organisers, the original per-corpus IDs (at minimum for the 507 Pitt
recordings) plus a `uid` mapping, or written confirmation that `uid` is
one-to-one with participant and that the challenge split was made
participant-wise.

**Affects:** `docs/dataset_description.md`, `docs/cv_strategy.md`, Q2, Q13, Q16.

---

## Q24 - One cohort recording produces no segments

The cohort holds 1481 recordings; the segment pipeline covers 1480. One
recording is shorter than `segments.segmentation.min_segment_len_s` (3.0 s), so
`extract_file_segments` returns an empty list and the recording disappears
silently — there is no warning.

The practical impact is negligible, but the two pipelines are evaluated on
cohorts that differ by one recording, and a silent drop is the kind of thing
that should be logged rather than inferred.

**Affects:** `segments/03_extract_segment_features.py`.
