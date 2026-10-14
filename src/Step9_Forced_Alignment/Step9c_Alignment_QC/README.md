# Step 9c: Alignment quality control

### Purpose
Says, for every recording, whether its forced alignment can be trusted, before any pause or fluency measure is read from it.
It is the one place where this is decided; the BERT text step and the fluency extractor apply the same check.

### A recording passes (`alignment_ok = 1`) when
* its TextGrid exists, can be read and has the word tier `mfa.word_tier`;
* the number of aligned words equals the number of transcript words;
* no more than `mfa.max_unk_fraction` (default 0.20) of the aligned words are `<unk>`.

**The word count check is weak for MFA.** When MFA aligns a recording at all it returns every transcript word, so the counts nearly
always agree, even when the transcript does not match the audio. The real signal is MFA's own scores (`alignment_analysis.csv` in the
TextGrid folder): the log-likelihood of the alignment, the deviation of the phone durations and the signal-to-noise ratio. They are joined
into the table as `mfa_*` columns. **No threshold is applied on them**, because none has been validated on this data. Read their
distribution in the report and look at the lowest-scoring recordings in Praat before choosing one.

A recording MFA could not align at all has no TextGrid and fails with the reason `no TextGrid`.

### Inputs
`mfa.corpus_dir` (Step 9a) and `mfa.textgrid_dir` (Step 9b).

### Outputs
| Path | What |
| --- | --- |
| `mfa.qc_csv` | one row per recording: `ID, alignment_ok, reason, n_words, n_aligned, n_unk, unk_fraction, audio_duration_s, textgrid_duration_s, first_word_start_s, last_word_end_s, speech_span_fraction, mfa_overall_log_likelihood, mfa_speech_log_likelihood, mfa_phone_duration_deviation, mfa_snr` |
| `mfa.qc_report` | how many pass, why the others fail, summary statistics, and the words MFA could not find in its dictionary |

`speech_span_fraction` is the time from the first to the last aligned word divided by the audio duration. A value far from 1 means a
long stretch at the start or the end of the recording has no aligned word.

### Run it
```bash
python src/Step9_Forced_Alignment/Step9c_Alignment_QC/alignment_qc.py
```

### Reading the result
Report the share that passed. A failed recording is not repaired. The pause-annotated BERT step stops if any recording failed (its
`on_alignment_failure` setting), and the fluency extractor leaves the features of a failed recording empty. This does not prove that a
passing alignment is right: a transcript with the right number of words can still have a misplaced word. Spot-check a few
TextGrids in Praat.
