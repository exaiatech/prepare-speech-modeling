# SpeechCARE-AGF Reproduction Spec

Source: github.com/SpeechCARE/SpeechCARE-NIA-Phase2 (main.py, config.yml, confirmed
directly from the files) + arXiv:2601.16989 "The Voice of Equity" (architecture
formulas, same authors, same model).

## A. Confirmed from source code (main.py, config.yml) - literal, not inferred

```yaml
seed: 133
batch_size: 4
epochs: 14
num_labels: 3                                    # Control, MCI, ADRD
txt_transformer_chp: Alibaba-NLP/gte-multilingual-base
speech_transformer_chp: utter-project/mHuBERT-147
segment_size: 5                                  # seconds
active_layers: 12                                # all mHuBERT-147 layers active (full fine-tune)
demography: age_bin                              # categorical, NOT raw age
demography_hidden_size: 128
hidden_size: 128
integration: SIMPLE_ATTENTION                    # meaning not yet verified, see Open Items
bias: False                                      # fairness-reweighting flag, off by default
```

**Optimizer** (`Trainer.init_optimizer`, non-Whisper branch - this is our path):
```python
optim.AdamW([
    {'params': decay,                        'lr': 1e-5, 'weight_decay': 0.005},
    {'params': no_decay,                     'lr': 1e-5, 'weight_decay': 0.0},
    {'params': model.txt_transformer.params, 'lr': 1e-6, 'weight_decay': 0.005},
])
```
mHuBERT and the fusion/classifier head train at 1e-5; mGTE trains at 1e-6. `no_decay`
= any parameter with 'bias', 'LayerNorm', 'norm', or 'positional_encoding' in its name.

**NOTE - discrepancy with config.yml:** `config.yml` states `wd: 1e-3`, but the optimizer
code hardcodes `weight_decay: 0.005` for the decay group. The code value governs actual
training; `wd: 1e-3` in the config does not appear to be read by this optimizer
construction. Document this as a known discrepancy, use 0.005 (literal code behaviour).

**Scheduler:**
```python
ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=4, min_lr=1e-6, threshold=0.01)
```
Not mentioned in either paper - code-only detail.

**Checkpoint / model selection criterion:**
```python
if val_loss <= self.minloss:
    self.minloss = val_loss
    save(model)
```
Selection is by **minimum validation loss**, evaluated every epoch.

**NOTE - discrepancy with the companion paper:** arXiv:2601.16989 states the checkpoint
with the "highest validation F1-score" is selected. The literal code selects on lowest
validation loss instead. Use the code's criterion (val loss) since it is literal and
verified; record the paper's claim as a documented discrepancy, same treatment as
SpeechCARE's 20%/25% segmentation-overlap inconsistency noted earlier.

**Preprocessing:** audio paths are `./LPF_train_audios/` and `./LPF_test_audios/` - LPF =
low-pass filtered. Confirms the 8 kHz low-pass filter (not neural denoising) is the
preprocessing actually used to produce the audio the model trains on.

**Training loop:** standard `CrossEntropyLoss`, optional per-sample reweighting via
`compute_sample_weights` (age-group-based) when `config.bias=True` - this is a fairness
mechanism from the later bias-mitigation paper, not part of the original Phase 2
submission; leave `bias=False` for a faithful reproduction of the original model.

**Prediction output:** per-recording class probabilities, keyed by `uid`:
```python
test_df.loc[test_df['uid'] == uid, 'C']    = p_control
test_df.loc[test_df['uid'] == uid, 'MCI']  = p_mci
test_df.loc[test_df['uid'] == uid, 'ADRD'] = p_adrd
```
Maps directly onto our `03_evaluate.py` format (`p_control`, `p_mci`, `p_adrd`).

## B. Confirmed from literal source (`model/model.py`, class `TBNet`)

This section previously relied on the companion paper's formulas. Those formulas turn
out to be a **simplified, two-modality description of a three-modality model**. The
literal code is authoritative; corrections from the paper version are marked.

**Acoustic branch (mHuBERT path, `config.is_whisper = False`):**
1. Input `input_values` arrives pre-segmented: shape `[batch, num_segments, seq_length]`
   (segmentation itself happens upstream, in the dataset code, not in `model.py`)
2. Reshaped to `[batch*num_segments, seq_length]`, passed through mHuBERT-147 whole:
   `self.speech_transformer(input_values).last_hidden_state`
3. **Correction:** the full frame-level `last_hidden_state` is kept, NOT pooled per
   segment. Reshaped to `[batch, num_segments, num_frames, dim]`, then flattened to
   `[batch, num_segments * num_frames, dim]` - every mHuBERT frame from every segment
   becomes one token in a single long sequence (roughly segments x ~250 frames)
4. A learnable `[CLS]` token (`nn.Parameter`, randomly initialised) is prepended
5. A learned absolute positional encoding (`nn.Parameter`, size driven by
   `config.max_num_segments` - see Open Items) is added
6. Passed through 2 `MultiHeadAttentionAddNorm` blocks: `nn.MultiheadAttention` (4
   heads, dropout 0.1) → residual add → `LayerNorm`. **Correction:** no feed-forward
   sublayer in either block - this is not a standard transformer encoder layer, just
   self-attention + add&norm, twice
7. The `[CLS]` position of the final output is the acoustic representation

**Linguistic branch (mGTE):** `self.txt_transformer(input_ids, attention_mask)`, take
`last_hidden_state[:, 0, :]` (the `[CLS]` position). As described before, unchanged.

**Demography branch:** a single scalar (shape becomes `[batch, 1]` via `unsqueeze(1)`)
through `nn.Linear(1, demography_hidden_size)` → `Tanh`. The scalar's exact meaning
(raw age vs. bin index) is not yet confirmed - see Open Items.

**Fusion - correction, three modalities each with their own classifier, not two
combined into one:**
```python
speech_x = tanh(W_speech · speech_cls + b)          # hidden_size
txt_x    = tanh(W_txt    · txt_cls    + b)          # hidden_size
demo_x   = tanh(W_demo   · demography + b)          # demography_hidden_size

gate_weights = Softmax(W_g · [speech_x; txt_x; demo_x] + b_g)   # 3 outputs

speech_out = W_speech_cls · speech_x + b     # num_labels logits
txt_out    = W_txt_cls    · txt_x    + b     # num_labels logits
demo_out   = W_demo_cls   · demo_x   + b     # num_labels logits

fused_output = gate_weights[0]*speech_out + gate_weights[1]*txt_out + gate_weights[2]*demo_out
# returned as raw logits; Softmax applied outside the model (in Trainer.evaluate)
```
Each modality is projected all the way to its own `num_labels`-dim logit vector before
the gated weighted sum - not fused as hidden representations. This matches the paper's
"weighted combination of modality-specific scores" wording, but the paper's explicit
equation only wrote out L and A, omitting demography from the formula even though it
is a full third branch in the deployed model.

**Segmentation, confirmed from `preprocess_audio`/`calculate_num_segments` (used at
single-file inference time; training-time segmentation presumably lives in a dataset
file we have not seen, but these defaults are the strongest evidence available):**
```python
def preprocess_audio(self, audio_path, segment_length=5, overlap=0.2, target_sr=16000):
```
**Overlap = 20%, resolving the ambiguity** between the main SpeechCARE paper (25% in
the main text, 20% in its own ablation section) in favour of 20%. Segments are fixed
`segment_length` with `step = segment_length * (1 - overlap)`; a trailing remainder is
kept as an extra (variable-length) segment only if it is at least 5 s long.

**Inference-time transcription note (not relevant to our pipeline):** `TBNet.inference()`
spins up a separate Whisper-large-v3 ASR pipeline purely to transcribe raw audio on the
fly when no transcript is supplied. We already have transcripts from Daniel's CSV, so
this path does not apply to us - mHuBERT remains the sole acoustic encoder either way.

## B2. Confirmed from `utils.py` and `dataset.py`

**Metrics (`Utils.save_result`) - correction:**
```python
roc_auc_score(true_labels, pred_probs, multi_class='ovr')   # one-vs-rest AUC
precision_score(..., average='weighted')                     # NOT macro
recall_score(..., average='weighted')
f1_score(..., average='weighted')
```
The companion paper's "macro-F1" claim does not match the code, which uses
`average='weighted'` throughout. Weighted averaging is dominated by the majority class
(Control); this matters for comparability since our own pipeline reports macro metrics
and per-class recall separately, as Daniel's training/evaluation spec requires.

**Data pipeline (`dataset.py`, class `AudioDataset`) - important structural finding:**
`__getitem__` does **not** process audio. It loads pre-computed, pre-saved per-segment
waveform tensors from disk (`torch.load(tensor_path)`), referenced by a dataframe with
columns including `uid`, `segment`, `path`, `label`. The actual segmentation and
tensor-saving step is a **separate offline preprocessing pipeline**, almost certainly
living in the repo's `preprocessing/` folder, which we have not yet seen. This means:
- The exact resample rate, windowing, and whether the saved tensors are raw waveform
  samples or already-feature-extracted `input_values` is not confirmed. `main.py`
  loads a `Wav2Vec2FeatureExtractor` and passes it into `get_dataloaders(...)`, which
  suggests feature extraction happens in a collate function at batch time rather than
  being baked into the saved tensors - but this is inference, not confirmed.
- For the Whisper path (`is_whisper=True`), one tensor per whole recording is loaded
  (no per-segment files), consistent with Whisper handling long audio internally.

**Demography - partially resolved:** `config.demography` ('age_bin') is literally a
**column name** read from a pre-existing demography dataframe, not an encoding
instruction. The column's actual values (bin index 0/1/2, or something else) are set
wherever that dataframe was built - not visible in `model.py`, `dataset.py`, or
`utils.py`. Our best-evidence assumption remains the companion paper's 3-category
scheme (46-65, 66-80, 80+) cast to float index.

**Text:** transcript is a single raw string per `uid` (column `transcription`), not
tokenized inside the `Dataset`. Tokenization happens elsewhere - a collate function,
not yet located (likely in a `data/` package, since `main.py` does
`from dataset import *` AND separately `from data import *`; the repo root listing
shows both a `dataset/` and a `data/` top-level folder, so these are two different
modules and `get_dataloaders`/`prepare_data` most likely live in `data/`, still unseen).

## C. Open items - still not verified

1. `config.max_num_segments` - referenced in the positional-encoding size calculation,
   not present in the config.yml excerpt seen so far
2. Exact encoding of the demography scalar - raw age, or the bin index (0/1/2) from the
   paper's 46-65/66-80/80+ scheme cast to float. The `Linear(1, ...)` shape is
   consistent with either; only `dataset.py` (not yet seen) can confirm which
3. What `integration: SIMPLE_ATTENTION` selects, given the gating mechanism in the code
   has no literal name matching this string - possibly a flag read elsewhere (dataset.py
   or utils.py) that is not exercised by `model.py` directly
4. Whether training-time segmentation (inside whatever `Dataset` class `main.py`
   constructs via `get_dataloaders`) uses the same 20% overlap default as the
   single-file `preprocess_audio` method above, or a separately configured value
5. Tokenizer padding/truncation settings for `input_ids`/`attention_mask` (max length,
   etc.) - not visible outside `dataset.py`
6. Contents of the `preprocessing/` folder - determines resample rate, exact
   segment-tensor construction, and whether feature extraction is pre-baked or applied
   at collate time
7. Contents of the `data/` package (distinct from the `dataset/` package we have seen)
   - likely holds `get_dataloaders`, `prepare_data`, and the collate function that
   tokenizes text and stacks variable-length segment lists into batches
8. The real values populating the `age_bin` demography column

## D. Practical constraint - compute

Unlike the Whisper-medium baseline (frozen encoder, linear probe only), this model
**jointly fine-tunes mHuBERT-147 (98M params) and mGTE (305M params) end to end**. This
is a real training job, not feature extraction. CPU training at this scale is
impractical; the Intel Arc GPU's training support via IPEX is far less mature than its
inference support. A CUDA GPU (even modest, e.g. a single consumer card with enough
VRAM) would be needed for this to run in reasonable time. This should be raised with
Daniel before committing effort to a training script.

## E. Review notes

Added during the repository cleanup. Sections A to D above are unchanged. The review covered
`06_speechcare_prepare_data.py` and `07_speechcare_train_eval.py` against this spec. The upstream
`model/model.py` could not be fetched (HTTP 404 on the raw file), so the comparison is against this
spec, not against the upstream source.

### E1. Checked and found consistent with the spec

* Optimizer groups, learning rates (1e-5 and 1e-6), weight decay 0.005, `ReduceLROnPlateau(factor 0.8,
  patience 4, min_lr 1e-6, threshold 0.01)`, 14 epochs, batch size 4, seed 133, `CrossEntropyLoss`.
* Fusion: three modality heads, each with its own classifier, and a softmax gate over their logits.
  The gate input width (128 + 128 + 128) matches.
* Segmentation: 5 s windows, 20% overlap (4.0 s hop), 16 kHz.
* `num_labels` follows the task (2 or 3). The predictions file has the columns that `03_evaluate.py` reads.

### E2. Forward and backward pass on stand-in encoders

`TBNet` was run with small random encoders that have the same output shapes as the real ones (a `HubertModel`
with the default convolutional stack and a small `BertModel`), with no download. Input was 2 recordings of
7 segments of 80,000 samples.

* One 5 s segment gives **249 frames**, so the sequence is 7 x 249 + 1 = **1,744** tokens. This equals the size of
  the learned positional table that the formula in `07` produces, for both `num_labels = 2` and `3`.
* Logits have the right shape, are finite, and the gate weights sum to 1.
* No parameter has a NaN or infinite gradient. Parameters without a gradient are the unused text pooler and, in
  some runs, layers that HuBERT's stochastic LayerDrop skipped in training mode. Neither is a defect of the port.

This does **not** establish that the real forward pass with the pretrained `mHuBERT-147` and `gte-multilingual-base`
checkpoints works. That has not been run.

### E3. Bug fixed

**Model selection used the evaluated set.** In `run_split`, `ReduceLROnPlateau` and the "minimum validation loss"
checkpoint were computed on `val_uids`, the same set the run then predicted. In a CV split that selects the epoch on
the held-out fold. In the final run `val_uids` was the **test partition**, so the test set was read at every epoch and
used to pick the model, contradicting "the test partition is evaluated once, after every decision is frozen".

The criterion is unchanged (minimum validation loss, as in the original code). Its data source changed: an inner
validation set is held out from the training uids (`INNER_VAL_FRACTION`, 0.10, stratified by label, seeded with
`SEED`). The held-out set is only predicted, after training. The inner split was tested for disjointness,
stratification and determinism. A full training run was not executed.

A side effect to be aware of: each run now trains on 90% of its training uids rather than all of them.

### E4. Left as it is, listed for a decision

1. **`INNER_VAL_FRACTION = 0.10`** is our choice. The original had a separate validation set, so there is no
   published value to match.
2. **No segment mask.** Recordings with fewer than 7 segments are padded with silent segments that the attention
   layers still see. This matches the original `forward()` as recorded in section B, but it means short recordings
   carry silent tokens. In this cohort 401 of 1,031 recordings are under 30 s.
3. **Trailing remainder.** The branch that keeps a remainder of at least 5 s cannot run (the remainder after the last
   full window is always shorter than 5 s), so up to about 4 s at the end of a recording is dropped. Whether the
   original behaves the same is not verified.
4. **Low-pass filtered audio.** The original read `LPF_train_audios/`. We resample the raw audio to 16 kHz, which
   band-limits at 8 kHz. Equivalence is not verified.
5. **Demography encoding, `integration: SIMPLE_ATTENTION`, tokeniser settings and the `preprocessing/` package** remain
   open items in section C.
6. **Seeding.** `07` seeds NumPy and PyTorch once at import. Each of the 51 runs continues the global random
   stream, so a single run cannot be reproduced in isolation.
7. **LayerDrop.** HuBERT's stochastic LayerDrop is active in training mode. The configured value of the real
   checkpoint was not inspected.

