# Step 4: ASR Branch

Produces transcripts from the recordings, and the evidence that those
transcripts are good enough to build linguistic features on.

This branch is **independent of the acoustic branch.** Step 2 needs no
transcripts and Step 4 needs no acoustic features. The two meet only at
multimodal modelling, which is supervised work and not in this repository.

## What Step 4 is and what Step 5 will be

Step 4 asks **whether the text is right.** Step 5 will ask **what the text says
about the speaker.**

Whisper is not trusted blindly. Steps 4c to 4e align its output word by word
against hand-checked references, classify every mismatch, and report the word
error rate together with a linguistic breakdown of what was lost. That matters
because hesitations, stutters and pauses are the clinical signal here, so a
transcript that reads cleanly but silently drops them is a bad transcript, not
a good one. This is why the transcription prompt and the VAD settings are tuned
for verbatim disfluency capture rather than for readability.

Step 5 (linguistic features, CYMO) is not written yet. It will consume the
transcripts this branch validates.

## The chain

| Step | Reads | Writes |
| --- | --- | --- |
| `Step4a_Audio_Preprocessing` | `data/raw/train_audios/` | `data/interim/processed_audios/` |
| `Step4b_Transcription` | `data/interim/processed_audios/` + `models/` | `data/interim/transcripts/` |
| `Step4c_Alignment` | `transcripts/` + `data/raw/gold_transcripts/` | `data/interim/alignment_reports/` |
| `Step4d_Verticalization` | `alignment_reports/` | `data/interim/verticalized_data/` |
| `Step4e_Error_Analysis` | `verticalized_data/` | `data/results/Master_Report_Final.xlsx` |

Each folder has its own README with the full detail. Run them in order.

## Before you start

Two things are **not in the repository** and must be supplied:

1. **The audio.** `data/raw/train_audios/`. See [`data/README.md`](../../data/README.md).
2. **The Whisper checkpoint.** `models/large-v3-local/`, 2.9 GB. See [`models/README.md`](../../models/README.md).

The gold-standard transcripts in `data/raw/gold_transcripts/` **are** committed:
27 hand-checked references, which is what the evaluation scores against.

## Loose files in this folder

* `Alignment_App.html` is a standalone browser tool for reviewing alignments by
  hand. Open it directly; it is not part of the scripted chain.
* `Step4d_Verticalization/gathersamples.py` is a near-duplicate of
  `step2_verticalization.py` with an extra rule that counts a gold `HES`
  annotation as a match when Whisper produced any hesitation token. It writes to
  the same output directory, so run one or the other, not both.

## Configuration

Unlike the acoustic branch, these scripts keep their settings as constants at
the top of each file rather than in `config.yaml`. Paths are derived from each
script's own location, so a fresh clone runs without editing anything, but the
parameters are not yet centralised. Folding them into `config.yaml` would make
the whole repository consistent and is the obvious next cleanup.
