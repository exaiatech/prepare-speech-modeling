# Step 1b: Audio Preprocessing

> **Scope after the supervisor's review: acoustic-feature preprocessing only. This is not a universal step for encoders.**
> Preserve the original recordings. Each encoder (Whisper-medium, SpeechCARE, the pause-annotated BERT) needs its own
> documented input prepared according to its published method, with the resampling, filtering, padding and truncation
> recorded. Do not apply this script to all models. The per-encoder table is `docs/encoders/encoder_input_preparation.md`.

### Purpose
Band-pass filters and peak-normalises the raw recordings: a 4th-order Butterworth filter between 75 Hz and 8000 Hz removes low-frequency mechanical rumble and high-frequency electronic hiss, then each file is peak-normalised.

> **This stage is currently NOT on either analysis path.** Neither feature extractor reads its output; both read `paths.raw_audio` directly, exactly as the source notebook did. The script is kept because the filtering is a reasonable thing to want and because removing it would hide the decision. Whether the pipeline should switch to preprocessed audio is unresolved: see `docs/open_questions.md` (Q11 and Q21).

### Inputs
* **Required Files:** Raw audio recordings.
* **File Formats:** `.wav`, `.mp3` (controlled by `preprocess.input_globs`).
* **Parameters:** Reads every file matching the input globs under `paths.preprocess.input_dir`.

### Outputs
* **Generated Outputs:** Filtered and normalised copies of the input audio.
* **File Format:** `.wav`
* **Location:** `data/interim/preprocessed_audio/`
* **Structure:** One output file per input file, same stem.

### Configuration
* `preprocess.lowcut_hz` (75.0), `preprocess.highcut_hz` (8000.0), `preprocess.filter_order` (4).
* `preprocess.resample_to` is `null`, meaning the original sample rate is kept.
* `preprocess.peak_normalize` toggles normalisation.
* `preprocess.max_normalized_freq` clamps the high cutoff for files whose Nyquist frequency sits below it, so low-sample-rate recordings do not raise a filter design error.

### Dependencies
* `librosa`, `soundfile`, `scipy`, `numpy`, `pyyaml`.

### Execution Instructions
```bash
python src/Step1_Dataset/Step1b_Audio_Preprocessing/preprocess.py
```
