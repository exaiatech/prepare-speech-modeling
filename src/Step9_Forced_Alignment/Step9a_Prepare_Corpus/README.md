# Step 9a: Prepare the MFA corpus

### Purpose
Writes, for every recording of the cohort, one 16 kHz mono wav and one `.txt` transcript with the same base name. That is the
corpus layout the Montreal Forced Aligner reads.

### Inputs
| Path | What |
| --- | --- |
| `paths.cohort` | the cohort CSV: the ID column and the transcript column (`mfa.text_column`) |
| `paths.raw_audio.train`, `.test` | the raw recordings |

The raw audio is read, not the output of `Step1b_Audio_Preprocessing`, which is acoustic-feature preprocessing only.

### Outputs
| Path | What |
| --- | --- |
| `mfa.corpus_dir` | `<ID>.wav` and `<ID>.txt` per recording |
| `mfa.input_report` | recordings without audio, with an empty or very short transcript, and what the normalisation removed |

### Transcript normalisation
Conservative, and it keeps every word: lower-case; `&` becomes `and`; hyphens and slashes become spaces; every character other than a
letter, a digit or an apostrophe inside a word is removed. Nothing is spelled out, merged or dropped, so the fillers (`uh`, `um`) stay.
Words MFA does not know get a noise phone in the alignment; Step 9c counts them as `<unk>`.

### Run it
```bash
python src/Step9_Forced_Alignment/Step9a_Prepare_Corpus/prepare_mfa_corpus.py
python src/Step9_Forced_Alignment/Step9a_Prepare_Corpus/prepare_mfa_corpus.py --overwrite    # rewrite existing entries
```
Entries that already exist are skipped, so an interrupted run can be repeated.
