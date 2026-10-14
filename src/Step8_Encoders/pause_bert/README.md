# Pause-annotated BERT baseline

> **Status: scaffold, not run, untested.** The pause-text script and the plumbing of the training script were exercised on synthetic
> inputs during the cleanup. The real BERT checkpoint was **not loaded** and no model was trained. The forced alignment it depends on
> is now its own stage, `Step9_Forced_Alignment`; see that folder for what has been run.

### Purpose
The text baseline of Li et al. (arXiv:2506.11119): a transcript in which pauses are marked with punctuation (a comma, a period or
an ellipsis for short, medium and long pauses), embedded by a frozen `bert-base-uncased` and classified by a small head.
The paper's best text model is BERT with pause annotation (accuracy 0.662, AUC 0.744 on its own cohort).

The cohort CSV has plain transcripts and **no word timestamps**, so the pauses are derived with the **Montreal Forced
Aligner (MFA)**. Preparing the corpus, running MFA and checking the alignment are shared with the fluency biomarkers and live in
[`Step9_Forced_Alignment`](../../Step9_Forced_Alignment/README.md). **If the supervisor supplies word-level timestamps, Step 9 is
replaced by his file** and script 01 reads it instead of the TextGrids.

### Pipeline
| Step | File | What it does |
| --- | --- | --- |
| before | `Step9_Forced_Alignment` 9a to 9c | The corpus, the MFA alignment and its quality check. |
| 01 | `01_build_pause_annotated_text.py` | Reads the TextGrids, measures the gap between consecutive words and writes `ID, text, text_pause_annotated, n_pauses, alignment_ok`. |
| 02 | `02_train_eval_bert.py` | Frozen BERT embeddings, age and sex, a 128-unit head, the frozen 50 CV splits plus the fixed test run. Writes predictions in the format `03_evaluate.py` reads. |

### Pause scheme, as in the paper
A gap above **0.05 s** gets a marker. Short (under 0.5 s) is `,`, medium (0.5 to 2 s) is `.`, long (over 2 s) is `...`. Other
punctuation is removed and the text is lower-cased. Every threshold and marker is in `config.yaml` under `stage1.pause_bert`, each
labelled "paper" or "ours".

### Where this differs from the paper, or where the paper is silent
* **Timestamps.** The paper used Whisper's. We use MFA's word intervals.
* **Sentence pauses.** The paper also marked the silence between adjacent Whisper segments. There are no segments here, so only
  gaps between words are marked. The paper gives no separate threshold for sentence pauses.
* **Boundaries.** Whether a gap of exactly 0.5 s or 2 s is medium is not stated (`boundary_is_medium`, ours).
* **Edges.** Whether to mark a gap before the first or after the last word is not stated (both off, ours).
* **Embedding.** The paper gives the 768 dimensions, not which vector. We use `[CLS]` (`embedding`, ours). It also speaks of
  sentence-level embeddings mean-pooled to the clip level; we embed the whole transcript.
* **Scheduler.** `ReduceLROnPlateau` is named without parameters; PyTorch defaults are used.
* **Audio before alignment.** The paper normalised amplitudes, trimmed silence and denoised with DeepFilterNet before
  transcription. We feed MFA the 16 kHz mono audio and nothing more.
* **Protocol.** The project's frozen splits replace the paper's five random 80/20 splits. The inner validation set is carved from
  the training uids, so neither a CV fold nor the test partition takes part in choosing the epoch.

### Alignment quality must be checked first
Dementia speech is full of fillers and restarts, and a forced aligner assumes a word-for-word transcript. Script 01 sets
`alignment_ok = 0`, and leaves the annotated text empty, when the TextGrid is missing, when the aligned and transcript word
counts differ, or when too many words are `<unk>` (the check of `common/mfa.py`, the same as in Step 9c). A failed recording is never
filled in. By default script 02 **stops** if any recording failed (`on_alignment_failure: error`). Report the share that passed
before reading any model result.

### Run it
```bash
python src/Step9_Forced_Alignment/Step9a_Prepare_Corpus/prepare_mfa_corpus.py
python src/Step9_Forced_Alignment/Step9b_Run_MFA/run_mfa.py
python src/Step9_Forced_Alignment/Step9c_Alignment_QC/alignment_qc.py
python src/Step8_Encoders/pause_bert/01_build_pause_annotated_text.py
python src/Step8_Encoders/pause_bert/02_train_eval_bert.py --task binary --text pause
python src/Step8_Encoders/pause_bert/02_train_eval_bert.py --task binary --text plain      # the no-pause baseline
python src/Step8_Encoders/pause_bert/02_train_eval_bert.py --dry-run                        # plumbing check, random embeddings
```

### Dependencies
`pandas`, `numpy`, `librosa`, `soundfile`, `praatio`, `torch`, `transformers`, `scikit-learn`. MFA itself is installed separately with
conda and is not a Python dependency of this repository (see `Step9b_Run_MFA/README.md`).
