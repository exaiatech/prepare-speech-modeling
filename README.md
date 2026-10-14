<table>
<tr>
<td>

## PREPARE: Speech Preprocessing and Exploratory Biomarker Analysis

**Project overview · Exaia Technologies**

This project investigates **speech-derived indicators of cognitive impairment** using recordings from the **PREPARE Challenge**, an initiative of the **NIH National Institute on Aging** hosted by DrivenData. The challenge’s acoustic track examines computational approaches to identifying Alzheimer’s disease and related dementias from speech, with an emphasis on predictive performance and explainability. The underlying speech corpora are available through **[DementiaBank](https://talkbank.org/dementia/)**, subject to its research access requirements.

Developed in collaboration with **Ada Yazici**, a research student participating in **RWTH Aachen University’s UROP International programme**, the project establishes a documented Python workflow for dataset preparation, speech preprocessing, and initial analyses.

The current analysis cohort comprises **1,031 English Cookie Theft picture-description recordings**, spanning cognitively healthy controls, mild cognitive impairment, and Alzheimer’s disease or related dementias. The workflow integrates cohort verification, audio preprocessing, acoustic and fluency feature extraction, descriptive statistical analysis, Whisper-based transcription, transcription-error assessment, and initial speech-encoder baseline evaluation.

The broader research objective is to compare **interpretable acoustic and linguistic biomarkers**, **learned speech representations**, and **hybrid models** under a common evaluation framework, assessing their contributions to classification performance and model interpretability.

[PREPARE Challenge](https://www.drivendata.org/competitions/group/nih-nia-alzheimers-adrd-competition/) · [DementiaBank](https://talkbank.org/dementia/)

</td>
</tr>
</table>

---

## Technical Documentation

*The following documentation describes the implementation, workflow, and development status of the student research project.*



# Speech Biomarkers of Cognitive Impairment (PREPARE Cookie Theft cohort)

Research internship project (UROP International, Exaia Technologies). Supervisor: Daniel Wiechmann.

The study asks how well speech-based models detect cognitive impairment, and what interpretability costs. It uses the
English PREPARE **Cookie Theft** cohort: **1,031 recordings, Control 635, MCI 65, AD/ADRD 331**. Two tasks are run as
separate experiments: binary (Control against AD/ADRD, MCI excluded) and three-class.

Three families of models are compared under one evaluation framework:

1. published encoder baselines (Whisper-medium, SpeechCARE-AGF, pause-annotated BERT);
2. interpretable speech-biomarker models (linguistic, acoustic and fluency, combined);
3. hybrids of encoder representations and interpretable biomarkers.

This README says what each part of the repository **is and what state it is in**. It does not claim that anything is
validated beyond what its own status says.

---

## Where the code stands, by topic

| Topic | Code | State |
| --- | --- | --- |
| Cohort | `Step1a_Recording_Metadata`, `Step1c_Cohort_Data_Check` | Present. |
| Evaluation framework | `Step7_Evaluation` | Present, run and verified by the author. |
| Encoder baselines | `Step8_Encoders` | Whisper-medium run and verified by the author. SpeechCARE written, not run. Pause-annotated BERT is a scaffold. |
| Acoustic and fluency features | `Step2_Acoustic`, `Step3_Feature_QC` | The 135-feature extractor is retained, **not used for modeling yet**. |
| Forced alignment (pauses, fluency biomarkers) | `Step9_Forced_Alignment` | Written. Tried on 20 recordings (all aligned with a wide beam); the full cohort has **not been aligned**. Which fluency source is used is undecided. |
| Linguistic features | `Step5_Linguistic` | Placeholder, not written. |
| Interpretable models | `Step6_Modeling` | The earlier path is **set aside** after the supervisor's review, to be redesigned. |
| Hybrids, comparison, paper outputs | none | Not started. |

---

## Layout

```
config.yaml            every path, threshold and parameter. Paths are relative to the repository root.
config.local.yaml      optional and gitignored: personal overrides, for data that lives elsewhere
requirements.txt       pinned versions
script_migration.csv   where every script of the earlier layouts ended up, and its status

src/                   ALL code
  common/              shared modules, imported and never run
  Step1_Dataset/       cohort table (1a), acoustic-feature preprocessing (1b), Stage 1 cohort check (1c)
  Step2_Acoustic/      VAD inspection tool (2a), the 135 acoustic features (2b)
  Step3_Feature_QC/    feature inventory (3a), descriptive statistics (3b)
  Step4_ASR/           audio preprocessing, Whisper transcription, alignment, verticalization, error analysis
  Step5_Linguistic/    placeholder
  Step6_Modeling/      set aside, README placeholders only (6d keeps one script with nothing to read)
  Step7_Evaluation/    frozen splits (7a), evaluation metrics (7b)
  Step8_Encoders/      whisper_medium, speechcare, pause_bert
  Step9_Forced_Alignment/  MFA corpus (9a), alignment (9b), quality check (9c), fluency biomarkers (9d)

data/                  ALL data. Nothing inside src/ is written to.
  raw/                 gitignored: cohort CSV, metadata.csv, the audio. gold_transcripts/ is committed.
  interim/             gitignored: everything the scripts write, including data/interim/stage1/
  processed/           gitignored
  metadata/, results/  committed: small tables

docs/                  documentation. docs/encoders/ holds the encoder notes (see below).
models/                gitignored: model checkpoints. README only.

archive/
  legacy/                       six folders that used to sit at the repository root (ARCHIVE_INDEX.md)
  set_aside_after_review/       files taken off the active path after the supervisor's review
  dead_code/                    superseded notebook cells, with reasons
  notebook_original.ipynb, features2_original.ipynb
  ARCHIVE_INDEX.md
```

Numbering note: the step numbers are pipeline stages and mean nothing else. Step 5 is a placeholder and Steps 7 and 8 are the
next free numbers.

---

## Status of every step

Status words: **retained** (kept, logic untouched), **changed after review** (behaviour changed after the supervisor's review),
**set aside after review**, **added**, **planned**.

| Step | What it is | Status |
| --- | --- | --- |
| `common/` | config loading, WebRTC VAD, the 135 feature definitions, feature-selection statistics, cohort membership | `cohort.py` added, the rest retained |
| 1a `build_recording_metadata.py` | recording table and cohort membership | **changed after review**: `in_cohort` now comes from the ID list of the shared cohort CSV |
| 1b `preprocess.py` | band-pass and peak normalisation | retained. **Acoustic-feature preprocessing only, not a universal step for encoders.** |
| 1c `01_data_check.py` | verifies the cohort, writes the inclusion list | added, run and verified by the author |
| 2a `vad_segment.py` | prints VAD pause diagnostics | retained, **diagnostic tool, not part of the modeling path** |
| 2b `extract_recording_level.py` | 135 acoustic features per recording | retained, unchanged. Not used for modeling yet. Covers Control and AD only. |
| 2b `extract_segment_level.py` | the same per 5 s window | **set aside after review** |
| 3a `build_inventory.py` | feature inventory table | retained |
| 3b `descriptive_stats.py` | per-class statistics | **changed after review**: recording level only, membership from the cohort CSV |
| 4a to 4e | ASR branch | retained, unchanged by the cleanup. Parameters are constants in each file, not in `config.yaml`. |
| 5 | CYMO linguistic features | planned |
| 6a, 6b, 6c | matrix, selection, training | **set aside after review**, to be redesigned when interpretable modeling starts |
| 6d `interpret_recording.py` | SHAP and permutation importance | left as it is. It needs outputs of the set-aside scripts. |
| 7a `02_build_splits.py` | frozen splits | added, run and verified by the author |
| 7b `03_evaluate.py` | metrics for any model | added, run and verified by the author |
| 8 `whisper_medium/` | frozen encoder plus a small head | added, run and verified by the author |
| 8 `speechcare/` | SpeechCARE-AGF port | added, **written, not run** (needs a GPU) |
| 8 `pause_bert/` | pause-annotated BERT, pauses from Step 9 | added, **scaffold, not run, untested** |
| 9a `prepare_mfa_corpus.py` | wav and txt per recording for MFA | added, run on the cohort (1,031 recordings) |
| 9b `run_mfa.py` | wrapper around `mfa align` | added, tried on 20 recordings |
| 9c `alignment_qc.py` | per-recording alignment check and MFA scores | added, tried on 20 recordings |
| 9d `extract_mfa_fluency_biomarkers.py` | fluency biomarkers from the alignment, adapted from the supervisor's script | added, tried on 20 recordings |

What the supervisor set aside, and why, is in
[`archive/set_aside_after_review/README.md`](archive/set_aside_after_review/README.md).

---

## Encoders

Each encoder has its own documented input, prepared according to its published method. They read the **raw** recordings and
none of them reads the output of Step 1b.
[`docs/encoders/encoder_input_preparation.md`](docs/encoders/encoder_input_preparation.md) records, per encoder, the published
method, the file our script reads, and the resampling, filtering, padding and truncation.
[`docs/encoders/SPEECHCARE_SPEC.md`](docs/encoders/SPEECHCARE_SPEC.md) specifies SpeechCARE-AGF and has the review notes.

All encoder models are evaluated on the **same frozen splits** (the fixed `set` partition plus 5-fold x 10-repeat CV on
train+dev) and by the same script, so their numbers are comparable. The test partition is predicted once per model.

---

## Quick start

```bash
pip install -r requirements.txt
```

Place the inputs under `data/raw/` (see [`data/README.md`](data/README.md)): the cohort CSV
`PREPARE_English_complete_COOKIE_THEFT.csv`, `metadata.csv`, and the recordings in `train_audios/`. Paths are in `config.yaml`
and are relative to the repository root. To keep the data elsewhere, put the changed keys in `config.local.yaml`.

Stage 1, in order:

```bash
python src/Step1_Dataset/Step1c_Cohort_Data_Check/01_data_check.py
python src/Step7_Evaluation/Step7a_Build_Splits/02_build_splits.py      # once; never regenerate
python src/Step8_Encoders/whisper_medium/04_whisper_extract_embeddings.py
python src/Step8_Encoders/whisper_medium/05_whisper_train_eval.py        # set TASK in the script, run once per task
python src/Step7_Evaluation/Step7b_Evaluate/03_evaluate.py               # set MODEL_NAME and TASK in the script
```

SpeechCARE and the pause-annotated BERT follow the same pattern; see their READMEs for what is untested. The pause-annotated BERT and the
MFA fluency biomarkers both start from the forced alignment of `Step9_Forced_Alignment`, run once (its README gives the order). Everything the Stage 1
scripts write goes to `data/interim/stage1/` and is not committed.

The retained acoustic scripts (`Step1a`, `Step2b`, `Step3a`, `Step3b`) are documented in their own READMEs. They were **not run on
the real data** during the cleanup. The two that changed (`Step1a`, `Step3b`) were checked with small synthetic inputs.

---

## Reproducibility

* Every path and parameter is in `config.yaml`. The Stage 1 scripts used to carry absolute Windows paths; they were moved there.
* The splits are fixed once (`random_state` 42) and read by every model. Do not regenerate them after models have used them.
* `requirements.txt` pins the versions the analysis was run with.
* Not everything is reproducible run to run: the Whisper head script (`05`) defines a seed and never applies it, and the
  SpeechCARE runs share one global random stream. See `docs/encoders/encoder_input_preparation.md`.

---

## Known problems

None of these has been fixed unless it says so.

1. **Participant independence cannot be confirmed.** The data has no participant identifier, only a recording ID and an audio
   hash, so repeated participants across partitions cannot be ruled out (`docs/open_questions.md`, Q23).
2. **The retained feature extractor covers Control and AD only.** Cohort membership in the acoustic scripts now comes from the
   cohort CSV, but `extract_recording_level.py` still selects its recordings with `features.metadata_filters`, so MCI recordings
   have no feature row.
3. **Sample-rate mismatch in the retained extractor.** Parselmouth reads the native rate while librosa resamples to 16 kHz
   (`docs/open_questions.md`, Q15).
4. **The set-aside modeling path selected features outside cross-validation**, so its CV estimates are biased. It is set aside for
   that reason and will be redesigned with fold-internal imputation, scaling and selection.
5. **SpeechCARE-AGF has never been run with the real checkpoints.** One protocol bug (model selection on the evaluated set, the test
   partition in the final run) was fixed in the cleanup. Open points are in `SPEECHCARE_SPEC.md`, section E.
6. **The pause-annotated BERT and the MFA fluency biomarkers depend on forced alignment of disfluent speech**, which has been tried on
   20 recordings only. With MFA's default beam, 8 of those 20 could not be aligned; a wider beam aligned all 20 (`Step9b_Run_MFA/README.md`). The
   provenance of the transcripts in the cohort CSV is not documented here.
7. **The Whisper baseline** has no amplitude normalisation, includes 30 s padding in the mean pool, and its head script is not
   seeded. These are listed, not changed.
8. **The documents under `docs/` other than `docs/encoders/` describe the earlier full-cohort analysis** (1,481 recordings, with
   segment-level and recording-level models). They were not revised in the cleanup. `data/README.md` was updated for the cohort
   file, and its cohort table is labelled as the earlier cohort's.
9. **The ASR scripts** (Step 4) keep their parameters as constants in each file and do not read `config.yaml`.

### Earlier results, superseded

An earlier analysis of the full English cohort (1,481 recordings, Control against AD only) reported test AUCs of 0.664
(recording level) and 0.682 (segment level). They were produced on a different cohort by a path that is now set aside, and they
must not be compared with results on the Cookie Theft cohort. The code is retained in `archive/set_aside_after_review/`.

---

## Conventions

* Everything configurable goes in `config.yaml`. No hardcoded path points at one person's machine.
* All executable code lives under `src/`, all data under `data/`. No script writes inside `src/`.
* Audio, embeddings, model weights, predictions and archives are never committed (see `.gitignore`).
* The test partition is predicted once per model, after every development decision is frozen.
* Binary and three-class tasks are separate experiments. Selected features and tuned hyperparameters are never transferred.
* Analysis logic is not changed without discussion with the supervisor. Anything superseded is archived, not deleted, with the reason
  recorded.
