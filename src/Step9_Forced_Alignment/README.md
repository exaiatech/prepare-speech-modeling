# Step 9: Forced Alignment (Montreal Forced Aligner)

The cohort CSV has plain transcripts and **no word timestamps**. A forced aligner takes a recording and its transcript and
returns the start and end time of every word. From those times come the pauses, and from the pauses two things the project needs:
the pause markers of the pause-annotated BERT baseline, and a set of fluency biomarkers.

**One alignment of the cohort feeds both.** It is run once here, checked once here, and the two consumers read the same result.

Not to be confused with `Step4_ASR/Step4c_Alignment`, which aligns Whisper's text against hand-checked reference text to score
transcription errors. That is text against text. This step is audio against text.

## The chain

| Step | Folder | Reads | Writes |
| --- | --- | --- | --- |
| 9a | `Step9a_Prepare_Corpus` | cohort CSV, raw audio | `data/interim/mfa/corpus/` (`<ID>.wav`, `<ID>.txt`) |
| 9b | `Step9b_Run_MFA` | the corpus | `data/interim/mfa/textgrids/` (`<ID>.TextGrid`) |
| 9c | `Step9c_Alignment_QC` | corpus, TextGrids | `data/interim/mfa/alignment_qc.csv` and `.md` |
| 9d | `Step9d_Fluency_Biomarkers` | corpus, TextGrids | `data/interim/mfa/fluency/` |

The pause-annotated text for BERT is built by `Step8_Encoders/pause_bert/01_build_pause_annotated_text.py`, from the same TextGrids.
Run the steps in order. Settings are in the `mfa:` section of `config.yaml`. Nothing here is written to `src/`.

## What the checks guard against

Dementia speech is full of fillers and restarts, and a forced aligner assumes a word-for-word transcript. If the transcript omits a
filler that was spoken, or adds a word that was not, the neighbouring words are aligned to the wrong stretch of audio, and the gaps
between them, which are the pause durations, come out wrong. The provenance of the transcripts in the cohort file (which ASR,
whether hand-corrected) is not documented in this repository.

A recording counts as aligned (`alignment_ok = 1`) only when its TextGrid exists, the number of aligned words equals the number of
transcript words, and not more than `mfa.max_unk_fraction` of the words are `<unk>`. The check is one function, `common/mfa.py`
`check_alignment`, used by 9c, 9d and the BERT text step, so they never disagree. **A failed alignment is never repaired or filled in.**
Report the share that passed before reading any result built on the alignment.

## Where the fluency extractor comes from

`Step9d_Fluency_Biomarkers/extract_mfa_fluency_biomarkers.py` is adapted from a script the supervisor wrote for the CAADMI recordings.
The measures and the column names are his; what changed is listed at the top of the file and in its README. It is one of three
candidate sources of pause and fluency features, next to the WebRTC-VAD pause features of Step 2b and the ASR-derived pause measures
in the cohort CSV. Which one the project uses is not decided (see the open questions in the main README).

## Before you start

* **The audio and the cohort CSV** are not in the repository. See [`data/README.md`](../../data/README.md).
* **MFA** is installed separately, with conda, in its own environment. See `Step9b_Run_MFA/README.md`.
* **Beam.** MFA's default beam failed on 8 of 20 trial recordings; `config.yaml` sets a wider one. See `Step9b_Run_MFA/README.md`.
* **Disk.** The corpus holds one 16 kHz wav per recording, about 0.75 MB each, so around 800 MB for the 1,031 recordings.
