# Archived code

Nothing here is part of the pipeline. Every file is a **verbatim** copy of one
cell of `features.ipynb` (also archived, as `../notebook_original.ipynb`), with a
header stating which cell it came from and why it was set aside. None of it was
refactored, none of it reads `config.yaml`, and the hardcoded paths still point
at the original machine.

Nothing was deleted.

Cell numbers are 0-based indices into the original notebook.

| File | Cell | Why it is here |
|---|---|---|
| `00_cohort/pilot_cohort_matching.py` | 9 | Age/gender-matched pilot cohort (25 pairs per gender). Superseded — the pipeline uses the full English subset. |
| `01_preprocess/bandpass_only_16k.py` | 23 | Preprocessing variant: 16 kHz + band-pass, no normalization. |
| `01_preprocess/normalize_only_16k.py` | 24 | Preprocessing variant: 16 kHz + normalization, no filter. |
| `03_extract_features/pilot_43_features_praat_pauses.py` | 10 | 43-feature pilot. Its Praat pause detection is the one that measured background noise. |
| `03_extract_features/pilot_100_features_webrtc_vad.py` | 11 | 100-feature pilot with WebRTC VAD. Superseded by the 135-feature extractor. |
| `05_train_evaluate/threshold_tuning.py` | 17 | Youden's J threshold comparison. The threshold logic itself survives in `05_train_evaluate.py`. |
| `05_train_evaluate/nested_cv_hyperparam_xgb.py` | 18 | Nested CV + XGBoost tuning. Its successor states tuning gave no improvement. |
| `05_train_evaluate/ensembles.py` | 19 | Averaging and stacking ensembles. The reported result is a single SVM (rbf). |
| `data_exploration/audio_duration_summary.py` | 0 | One-off duration / sample rate inventory. |
| `data_exploration/rms_silence_metrics.py` | 1 | Early RMS + silence exploration. |
| `data_exploration/random_sample_copy.py` | 2 | Listening sample selection. Unseeded `.sample()`. |
| `data_exploration/copy_english_v1_filename.py` | 7 | Broken first attempt at filtering English recordings. |
| `data_exploration/copy_english_v2_uid.py` | 8 | Corrected version, but never wired into the pipeline. |
| `asr/wer_report_truncated.py` | 3 | ASR work package. **Source is truncated in the notebook.** |
| `asr/verticalize_alignment.py` | 4 | ASR work package. |
| `asr/error_analysis_nlp.py` | 5 | ASR work package. |
| `asr/wer_summary_simple.py` | 6 | ASR work package, regressed copy of cell 5. |

## About `asr/`

ASR and transcript work is a separate work package, now `src/Step4_ASR/`,
not part of the acoustic branch. Those four cells were archived
rather than promoted, because reconstructing that pipeline was out of scope for
this restructuring — and cell 3 is incomplete in the notebook, so the package
cannot be made runnable from this source alone. Move them into `asr_pipeline/`
when that work package gets its own pass.
