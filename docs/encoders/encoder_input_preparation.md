# Encoder Input Preparation

The supervisor's review of the audio preprocessing asked for three things: preserve the original recordings, prepare a
separate documented input for each encoder according to its published method, and record the resampling, filtering,
padding and truncation of each. `src/Step1_Dataset/Step1b_Audio_Preprocessing/preprocess.py` is **acoustic-feature
preprocessing only** and is not a universal step for encoders.

Everything about our pipeline below is read from the code. Everything about the published methods is read from the
source named in the table, and anything that could not be confirmed says **not verified**.

## Summary

| | Whisper-medium | SpeechCARE-AGF | Pause-annotated BERT |
| --- | --- | --- | --- |
| **Published method** | Li et al., arXiv:2506.11119. 48 kHz clip resampled to 16 kHz, amplitudes normalised, padded or truncated to 30 s, 80-channel log-Mel (25 ms window, 10 ms stride), frozen encoder, mean-pooled embedding. Classification head: one hidden layer of 128 units, age and sex concatenated. | Azadmaleki et al., repository `SpeechCARE/SpeechCARE-NIA-Phase2` and arXiv:2601.16989, as recorded in `SPEECHCARE_SPEC.md`. 5 s segments, 20% overlap, 16 kHz, per-segment waveform tensors, `Wav2Vec2FeatureExtractor` for mHuBERT-147. The released code reads `LPF_train_audios/`, which the spec describes as low-pass filtered audio. The `preprocessing/` code that builds the tensors was **not accessible**. | Li et al., arXiv:2506.11119. Transcripts from Whisper. Before transcription: resampled to 16 kHz, amplitudes normalised, leading and trailing silence trimmed, DeepFilterNet noise suppression. Pauses recoded as punctuation (see below). `bert-base-uncased`, padded or truncated to 512 tokens. |
| **File our script actually reads** | `04_whisper_extract_embeddings.py` reads the **raw** recording at `paths.raw_audio.train` and `paths.raw_audio.test` (default `data/raw/train_audios/<ID>.mp3`), IDs from `stage1.inclusion_list`. | `06_speechcare_prepare_data.py` reads the **raw** recording at `paths.raw_audio.train` and `.test`, IDs from `stage1.inclusion_list`, text and age from `paths.cohort`. | `pause_bert/01_prepare_mfa_inputs.py` reads the **raw** recording at `paths.raw_audio.train` and `.test` and the `text` column of `paths.cohort`. The training script reads the annotated CSV it produces. |
| **Resampling** | `librosa.load(path, sr=16000, mono=True)` from the native 48 kHz mp3. | `librosa.load(path, sr=16000, mono=True)`, then 5 s windows with a 4.0 s hop. | `librosa.load(path, sr=16000, mono=True)` for the MFA corpus, written as 16 kHz mono wav. |
| **Filtering** | None applied by us. Resampling to 16 kHz band-limits at 8 kHz as a side effect. | None applied by us. The original used low-pass filtered audio (cut-off as stated in the spec, not verified here). Whether our 16 kHz resample is equivalent is **not verified**. | None. The paper's DeepFilterNet step is **not reproduced**. |
| **Amplitude normalisation** | None by us. The paper normalises amplitudes. How is **not verified**. `WhisperFeatureExtractor` normalises the log-Mel but does not remove gain. | None by us. `Wav2Vec2FeatureExtractor` normalises each segment to zero mean and unit variance at collate time. | None. The paper normalises and trims silence before transcription. |
| **Padding** | `WhisperFeatureExtractor` pads every clip to 30 s with zeros. | Zero-filled segments are appended so every recording has `MAX_NUM_SEGMENTS` segments. The model has **no segment mask**, as in the original `forward()`. | Tokeniser pads to 512 tokens. |
| **Truncation** | `WhisperFeatureExtractor` truncates beyond 30 s. No cohort recording is longer than 30 s. | Segments beyond `MAX_NUM_SEGMENTS` are cut (none, by construction). A trailing remainder is kept only if at least 5 s long, which cannot happen (see the review notes in `SPEECHCARE_SPEC.md`). | Tokeniser truncates beyond 512 tokens. |
| **Status** | `04`, `05`: run and verified by the author. | `06`, `07`: written, not run (needs a GPU). Forward and backward pass checked on small random stand-in encoders only. | Scaffold, not run, untested. |

## Does any encoder read the output of `preprocess.py`?

**No.** `preprocess.py` writes to `paths.preprocess.output_dir`. None of `04`, `05`, `06` or `07` reads that key or any
path under it: they read `paths.raw_audio.*` and the `stage1.*` keys only. A search of the four scripts for
`preprocess`, `processed_audio`, `butter`, `lfilter`, `silero` and `LPF` finds only comments that describe the upstream
SpeechCARE `preprocess_audio` method. The same holds for the `pause_bert/` scripts and for `Step9_Forced_Alignment`, which writes its own 16 kHz mono corpus from the raw audio.

## Whisper-medium: what matters about the 30 s window

`WhisperFeatureExtractor` always produces a 30 s input, and the encoder output is mean-pooled over all 1,500 frames,
**padding included**. The paper pads to 30 s and mean-pools too, so this matches it. It still has a consequence in this
cohort, where recordings are at most 30 s:

* 401 of 1,031 recordings (38.9%) are shorter than 30 s, and 293 (28.4%) are shorter than 25 s.
* The padding share of the window averages 11.4% and reaches 82%.
* For those recordings the embedding is diluted by frames that carry no speech, in proportion to how short the
  recording is. Recording length is therefore partly encoded in the embedding.

## Known issues found while reading the code

| Script | Issue | Effect |
| --- | --- | --- |
| `04` | No amplitude normalisation before the feature extractor. The paper normalises amplitudes. | Recording gain can reach the embedding. |
| `05` | `RANDOM_STATE = 42` is defined and never used. Neither NumPy nor PyTorch is seeded, and the early-stopping hold-out uses `torch.randperm`. | Head training is not reproducible run to run. |
| `05` | Up to 200 epochs and no `ReduceLROnPlateau`. The paper trains up to 100 epochs with `ReduceLROnPlateau`. | A deviation from the paper's head. |
| `05` | The sex label encoder is fitted on all rows, including the test partition. | Harmless in effect (it only assigns integer codes), but it is fitted outside the training fold. |
| `06` | The branch that keeps a trailing remainder of at least 5 s can never run: after the last full window the remainder is shorter than 5 s. | Up to about 4 s at the end of a recording is dropped. |
| `06` | A recording shorter than 5 s would produce one short segment and break `torch.stack` in `07`. | None here: the shortest recording is 5.4 s. |
| `07` | Checkpoint and scheduler were driven by the loss on the very set that was then predicted, and in the final run that set was the test partition. | **Fixed** in this cleanup, see `SPEECHCARE_SPEC.md`, review notes. |

`05` and `04` were not changed in this cleanup. They are listed so the author can decide.

## Paper details for the pause-annotated BERT

From Li et al. (arXiv:2506.11119), read from the paper text:

* A word pause is a gap between consecutive words inside a Whisper segment longer than **0.05 s**. A sentence pause is
  the silence between adjacent Whisper segments. The sentence-pause threshold is not stated separately.
* Pauses are binned: short under 0.5 s, medium 0.5 to 2 s, long over 2 s. They are written as `,`, `.` and `...`
  respectively. Punctuation otherwise is removed and the text is lower-cased for the uncased model.
* `bert-base-uncased`, tokenised with the default tokeniser, padded or truncated to 512 tokens, a 768-dimensional
  embedding, a single feed-forward layer of 128 units, frozen encoder.
* Adam, learning rate 0.0005, batch size 32, up to 100 epochs, `ReduceLROnPlateau`, stop after five epochs without
  validation-loss improvement.
* The paper took the timestamps from Whisper. We have no timestamps, so `Step9_Forced_Alignment` derives them with the
  Montreal Forced Aligner. That is a deviation, and it is listed in the folder's README.
