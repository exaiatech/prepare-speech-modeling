# Step 8: Encoder Baselines

Published encoder baselines, evaluated under the same frozen splits and the same metric script as every other model.

| Folder | Model | Status |
| --- | --- | --- |
| `whisper_medium/` | Frozen Whisper-medium encoder plus the small head of Li et al. | Run and verified by the author |
| `speechcare/` | SpeechCARE-AGF (mHuBERT-147 + mGTE, gated fusion) | Written, not run (needs a GPU) |
| `pause_bert/` | Pause-annotated BERT of Li et al., with pauses from the forced alignment of `Step9_Forced_Alignment` | Scaffold, not run, untested |

**Each encoder has its own documented input.** None of them reads the output of `Step1b_Audio_Preprocessing`, which is
acoustic-feature preprocessing only. See [`docs/encoders/encoder_input_preparation.md`](../../docs/encoders/encoder_input_preparation.md)
for the per-encoder table of resampling, filtering, padding and truncation.

Order: Step 1c, then Step 7a (once), then a model folder, then Step 7b.
