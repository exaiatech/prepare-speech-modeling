# Step 2b: Acoustic Feature Extraction (135 features)

> **Status after the supervisor's review.**
> * `extract_recording_level.py` is **retained for the later interpretable-feature stage and not used for modeling yet.** For now the work is to check the feature definitions, the failed recordings and the missing values. Do not train models on these features yet.
> * It currently covers **Control and AD only**: membership is set by `features.metadata_filters` in `config.yaml` (Control, ProbableAD, AD). The frozen cohort also contains MCI. This is an open question.
> * `extract_segment_level.py` was **set aside** and moved to `archive/set_aside_after_review/`. One observation and one prediction per recording.

### Purpose
Extracts the 135 acoustic features, one row per recording. The feature definitions live in `src/common/feature_functions.py`; this script only decides which recordings to read and writes the tables.

The segment-level extractor, which computed the same definitions for 5 s windows with 25% overlap, was set aside after the supervisor's review and is in `archive/set_aside_after_review/`.

> **Known sample-rate difference.** `extract_recording_level.py` builds `parselmouth.Sound(filepath)`, which reads the file at its **original** sample rate, while librosa resamples to 16 kHz. So its Praat features (F0, formants, jitter, shimmer, HNR) and its librosa features (MFCC, spectral shape) are computed on different signals. This is unfixed; see `docs/open_questions.md` (Q15).

### Inputs
* **Required Files:**
  1. `data/raw/metadata.csv`
  2. The raw recordings under `data/raw/train_audios/` and `data/raw/test_audios/`
* **File Formats:** `.csv` for metadata; `.wav`, `.mp3`, `.flac`, `.m4a` for audio.
* **Parameters:** Metadata is filtered to `language == english`, then split into Control versus `ProbableAD` + `AD`. Audio files are matched to `uid` by exact filename stem, falling back to a prefix match.

### Outputs
* **Generated Outputs:** Feature tables, one per split and diagnosis group.
* **File Format:** `.csv`
* **Location:**
  * `data/interim/features_recording/{train,test}/{cn,ad}/features.csv`
* **Structure:** One row per recording with the 135 feature columns plus `uid` and `group`.

### Feature breakdown (135)
F0 7 | intensity 6 | energy 6 | pause/fluency 18 | articulation 3 | formant 3 | voice quality (jitter/shimmer/HNR) 30 | MFCC + delta-MFCC 52 | spectral shape 10

Full definitions, units and extraction parameters: `docs/feature_inventory.md`, generated from the running code by Step 3a.

### Configuration
* Extraction parameters live under `features:` in `config.yaml`: pitch floor/ceiling, intensity minimum pitch, energy frame sizes, articulation thresholds, formant settings, voice-quality windowing, MFCC count, STFT sizes and spectral bands.
* Parallelism: this extractor is sequential. If threading is ever added, use threads and never processes: the Praat C++ objects cannot be pickled.

### Dependencies
* `librosa`, `praat-parselmouth`, `webrtcvad`, `numpy`, `scipy`, `pandas`, `joblib`, `pyyaml`.

### Execution Instructions
It is slow: it runs Praat analyses over every recording. Not to be run as part of the cleanup, and not used for modeling yet.
```bash
python src/Step2_Acoustic/Step2b_Feature_Extraction/extract_recording_level.py
```
