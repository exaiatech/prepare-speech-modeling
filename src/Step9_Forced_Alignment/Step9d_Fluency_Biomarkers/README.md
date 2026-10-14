# Step 9d: Fluency biomarkers from the forced alignment

### Purpose
One row per recording of hesitation, pause, speech-run and speech-rate measures, computed from the word times of the forced alignment.
The script is adapted from `extract_mfa_fluency_biomarkers.py`, which the supervisor wrote for the CAADMI recordings. **The measures and
the column names are his**, so the two tables can be compared. The changes are listed below.

### What is measured
| Group | Columns (the pause columns exist once per threshold, suffix `_ge_250ms`, `_ge_500ms`, `_ge_1000ms`, `_ge_2000ms`) |
| --- | --- |
| Counts | `n_words_total`, `n_lexical_words`, `n_hes_total`, `n_ae_hes`, `n_aehm_hes`, `n_hm_hes`, `n_mhm_hes` |
| Hesitation rates | `hes_per_100_words_total`, `hes_per_100_lexical_words`, `hes_per_min` |
| Time | `duration_sec`, `audio_duration_sec`, `speech_time_sec_word_intervals`, `all_interword_gap_time_sec` |
| Rates | `speech_rate_wpm_*`, `articulation_rate_wpm_*`, each for total words and for lexical words |
| Speech runs | `n_speech_runs`, `mean_length_of_run_words_total`, `..._lexical_words`, `..._sec`, `median_length_of_run_sec` |
| Pauses | `n_pauses_ge_*`, `pause_total_sec_ge_*`, `pause_mean_sec_ge_*`, `pause_median_sec_ge_*`, `pause_max_sec_ge_*`, `pause_p95_sec_ge_*`, `pause_rate_per_min_ge_*`, `pauses_per_100_words_ge_*`, `pause_time_ratio_ge_*`, `speech_to_pause_ratio_ge_*` |
| Status | `alignment_ok`, `reason`, `unk_fraction` |

Hesitation group names are his. In his German data `ae_hes` is the "äh" type; here it is the "uh" type, and `aehm_hes` the "um" type. The
tokens of each group are in `config.yaml` (`mfa.fluency.hesitations`), and a token can be added there.

Not measured here: spectral and voice-quality features (MFCC, jitter, shimmer, formants, CPP, LTAS). They stay in Step 2b.

### How a measure is counted (as in his script)
* A pause is any positive gap between the end of one word and the start of the next. Silence before the first word and after the last
  is not a pause.
* Speech time is the summed duration of the aligned word intervals, hesitations included. Total time is the length of the TextGrid.
* A speech run ends at a gap of `mfa.fluency.run_pause_threshold_s` (0.25 s) or more.

### What differs from his script
| His script | Here |
| --- | --- |
| Paths and parameters as constants with his folders | Read from `config.yaml` (`mfa:`) |
| Identifier column `TID` | `ID`, the cohort ID column |
| Word taken from the TextGrid | Word taken from the transcript, time from the TextGrid. A word missing from MFA's dictionary (`uhm`, `mm`) is read as `<unk>`, so the reader's label cannot identify a hesitation |
| German hesitation list with a few English entries | English list in the config; `uhm` and `erm` added; `ah` and `oh` not counted |
| Every TextGrid is trusted | Failed alignments (Step 9c check) get no feature values, only `alignment_ok = 0` and the reason |
| Hand-written TextGrid parser | `praatio` |
| Every file in the folder | The cohort corpus of Step 9a |

### Outputs (`mfa.fluency.output_dir`)
`mfa_fluency_biomarkers.csv` (one row per recording), `pause_events_long.csv`, `speech_runs_long.csv`, `word_intervals_long.csv`. The three long tables
hold only recordings with `alignment_ok = 1`.

### Run it
```bash
python src/Step9_Forced_Alignment/Step9d_Fluency_Biomarkers/extract_mfa_fluency_biomarkers.py
```

### Open decision
This is one of three candidate sources of pause and fluency features: the WebRTC-VAD pause features (Step 2b), the ASR-derived pause
measures in the cohort CSV, and this one. Which is used, or whether they are compared, is for the supervisor to decide. Until then none of them is
frozen. A fluency feature that is read from a forced alignment inherits the alignment's errors, so check the share that passed Step 9c first.
