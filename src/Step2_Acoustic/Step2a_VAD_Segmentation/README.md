# Step 2a: VAD Segmentation Inspection

> **Diagnostic tool, not part of the modeling path.** Retained, per the supervisor's review, for checking questionable
> pause results. It does not need to produce a modeling dataset.

### Purpose
A diagnostic entry point for looking at how WebRTC VAD segments a single recording into speech and silence. **Nothing downstream reads its output.** The VAD itself lives in `common/vad.py` and is applied inside the Step 2b extractors, where it produces the 18 temporal/fluency features of the 135-feature set.

Use this when a recording produces suspicious pause statistics. 103 recordings (7.0%) have at least one feature outside its plausible range, and the largest single cause is pause detection returning exactly zero.

> The VAD replaced an earlier Praat intensity-threshold method, which detected **zero** pauses in noisy recordings and produced a spurious significant result that vanished once corrected. Do not revert to the Praat method. The archived predecessor is in `archive/dead_code/03_extract_features/`.

### Inputs
* **Required Files:** One or more audio files, given as command-line arguments.
* **File Formats:** Anything `librosa` reads (`.wav`, `.mp3`, `.flac`, `.m4a`).
* **Parameters:** Audio is loaded mono at `vad.sample_rate` (16 kHz).

### Outputs
* **Generated Outputs:** A per-file segmentation summary printed to stdout. **No files are written.**
* **Structure:** The 18 pause and speech-segment features, one per line, for each input file.

### Configuration
* `vad.aggressiveness` (2, on the 0-3 WebRTC scale; moderate, chosen for noisy recordings).
* `vad.frame_ms` (20), `vad.min_silence_frames` (5, so 100 ms minimum to count as a pause), `vad.min_speech_frames` (2, so 40 ms minimum to count as speech).
* `vad.long_pause_threshold_s` (0.5) splits `long_pause_count` from `short_pause_count`.

### Dependencies
* `webrtcvad`, `librosa`, `numpy`, `scipy`, `pyyaml`.

### Execution Instructions
```bash
python src/Step2_Acoustic/Step2a_VAD_Segmentation/vad_segment.py data/raw/train_audios/example.wav
```
Several files at once:
```bash
python src/Step2_Acoustic/Step2a_VAD_Segmentation/vad_segment.py data/raw/train_audios/a.wav data/raw/train_audios/b.wav
```
