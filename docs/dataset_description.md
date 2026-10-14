# Dataset Description

Answers the dataset questions raised by the supervisor on 2026-07-30. It
describes the data actually underlying the poster analysis, and states plainly
which of the requested facts **cannot** be derived from the material currently
in hand.

The machine-readable companion is
[`../data/recording_metadata.csv`](../data/recording_metadata.csv), one row per
recording, produced by `../src/00_build_recording_metadata.py`.

---

## 1. Summary: what can and cannot be documented

| Requested | Status |
| --- | --- |
| Recordings per diagnostic group and partition | **Available** |
| Number of segments generated per recording | **Available** |
| Allocation of recordings and segments to train / test | **Available** |
| Which recordings belong to the same participant | **NOT AVAILABLE** |
| How many recordings each participant contributed | **NOT AVAILABLE** |
| Which elicitation task was used for each recording | **NOT AVAILABLE** |
| Confirmation that one participant's data stayed in one partition | **NOT POSSIBLE** |

The three unavailable items all have the same cause, described in section 5.
They are emitted as empty columns in the metadata table rather than guessed.

---

## 2. Source metadata

`metadata.csv` as distributed with the PREPARE Challenge: **2058 rows, 12
columns**, exactly one row per `uid`.

```
uid, age, gender, split, hash, filesize_kb,
diagnosis, race, language, handedness, education, corpus
```

- `uid` — recording identifier, unique across all 2058 rows.
- `hash` — 32-character hex string, also unique per row. It is a per-file hash,
  **not** a participant key: grouping by it recovers the rows one-to-one.
- `split` — train / test, as distributed by the challenge. The project did not
  create this split.
- `corpus` — one of 10 source corpora (`wls`, `pitt`, `vas`, `delaware`, `lu`,
  `hopkins`, `baycrest`, `depaul`, `ivanova`, `ye`).
- No participant identifier. No session or visit number. No task or stimulus
  column.

---

## 3. The analysed cohort

Two filters are applied, both in `config.yaml` under
`features.metadata_filters`:

1. `language == 'english'` → 1655 of 2058 recordings.
2. `diagnosis in {Control, ProbableAD, AD}` → **1481 recordings**.

MCI (135) and PPA (39) recordings pass the language filter but are excluded
from the cohort. They are neither trained on nor evaluated.

### Recordings by diagnosis and partition

| Partition | Control | ProbableAD | AD | Total |
| --- | --- | --- | --- | --- |
| train | 760 | 346 | 77 | 1183 |
| test | 184 | 95 | 19 | 298 |
| **Total** | **944** | **441** | **96** | **1481** |

Model labels: CN = 0 (Control), AD = 1 (ProbableAD + AD). Class balance is
therefore 944 CN vs 537 AD, i.e. **36.3% AD**.

### Recordings by corpus and partition

| Corpus | train | test | Total |
| --- | --- | --- | --- |
| wls | 660 | 165 | 825 |
| pitt | 403 | 104 | 507 |
| vas | 56 | 9 | 65 |
| lu | 37 | 14 | 51 |
| delaware | 24 | 6 | 30 |
| baycrest | 3 | 0 | 3 |
| **Total** | **1183** | **298** | **1481** |

`hopkins`, `depaul`, `ivanova` and `ye` contribute no recordings to the cohort
(their English recordings are all MCI/PPA, or non-English).

Two corpora carry 90% of the data. WLS and Pitt use different protocols, so any
corpus effect is a confound; `corpus` is not currently used as a covariate or
as a stratification variable.

---

## 4. Segments

The segment pipeline cuts each recording into fixed 5.0 s windows with a 3.75 s
hop (25% overlap) and discards trailing windows shorter than 3.0 s. See
`../src/segments/03_extract_segment_features.py`.

**10 288 segments from 1480 recordings.**

| Partition | Class | Recordings | Segments |
| --- | --- | --- | --- |
| train | CN | 760 | 5349 |
| train | AD | 422 | 2883 |
| test | CN | 184 | 1310 |
| test | AD | 114 | 746 |
| **Total** | | **1480** | **10 288** |

One cohort recording produced **zero** segments: it is shorter than the 3.0 s
minimum, so it is silently absent from the segment pipeline while being present
in the whole-recording pipeline (which has all 1481). The two pipelines are
therefore evaluated on cohorts that differ by one recording.

### Segments per recording

| Segments | Recordings |
| --- | --- |
| 1 | 2 |
| 2 | 16 |
| 3 | 62 |
| 4 | 98 |
| 5 | 115 |
| 6 | 143 |
| 7 | 109 |
| 8 | 935 |

Mean 6.95, median 8, min 1, max 8. The ceiling at 8 is a property of the
recordings, not a cap in the code: a ninth window would require a recording of
at least ~33 s, and none of the cohort recordings reaches that. Per-recording
counts are in `../data/recording_metadata.csv` (`n_segments`).

Because segment counts differ per recording, **the mean-probability
aggregation weights every recording equally** regardless of how many segments
it contributed — but a recording with one segment gets a much noisier estimate
than one with eight.

---

## 5. Why participants and tasks cannot be documented

`metadata.csv` contains no participant identifier and no task column. That is
the whole reason, and no amount of processing of the current files changes it.

This matters because two of the source corpora are known to contain repeated
recordings per participant. In particular the Pitt corpus (DementiaBank),
507 recordings here, is longitudinal by design: participants were recorded at
multiple visits. If any of those participants appear more than once in this
cohort, then:

- the recording-level analysis silently treats one person as several
  independent observations;
- the segment pipeline's `StratifiedGroupKFold(groups=uid)` protects against
  **segment** leakage across folds but **not** against a participant appearing
  in both the training and the validation half;
- the PREPARE train/test split may or may not have kept participants together —
  the project has no way to verify it.

### The proxy diagnostic, and its limits

`00_build_recording_metadata.py` emits two columns,
`participant_proxy_group` and `participant_proxy_group_size`, that group
recordings sharing `corpus + age + gender + diagnosis`
(`reporting.participant_proxy_keys`).

Result on the cohort: **361 distinct keys for 1481 recordings; 1312 recordings
share their key with at least one other recording; the largest group holds
132 recordings.**

That is a *collision count*, not an identity claim. A 70-year-old female
Control from WLS is not a rare combination, so most collisions are certainly
different people. The number establishes only that the metadata is far too
coarse to rule repeated participants out — it cannot establish that repeats
exist. **These columns must never be used as a grouping key for
cross-validation.**

### What would resolve it

One of:

1. A participant identifier from the PREPARE organisers, joinable on `uid`.
2. The original per-corpus keys (e.g. DementiaBank participant IDs for the Pitt
   subset) plus a mapping to `uid`.
3. Written confirmation from the organisers that `uid` is one-to-one with
   participant, and that the train/test split was made participant-wise.

Option 3 is the cheapest and would close the question outright. Until one of
these arrives, every statement about participant-level independence in the
poster and the paper should be explicitly qualified. See `open_questions.md`
Q2, Q16 and Q23.

### Elicitation task

Same situation. `corpus` constrains the task — WLS and Pitt both include a
Cookie Theft picture description, others use different protocols — but a corpus
can contain several task types, and the metadata does not say which one a given
recording used. The cohort therefore mixes elicitation tasks in an unknown
proportion, and the Cookie Theft-only subset that the project wants to define
**cannot be constructed from the current metadata**. See `open_questions.md` Q13.

---

## 6. Allocation to partitions

There is no validation split. Both pipelines use:

- **train** — the PREPARE train split; all model fitting, feature selection and
  threshold selection happen here, via cross-validation.
- **test** — the PREPARE test split; used once, for the final reported numbers.

Cross-validation folds are drawn inside the train split only. Details, and the
precise sense in which "no leakage" does and does not hold, are in
[`cv_strategy.md`](cv_strategy.md).
