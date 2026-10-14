# SpeechCARE-AGF baseline

> **Status: written, not run (needs a GPU). The real forward pass with the pretrained checkpoints is untested.**
> The model was run forward and backward on small random stand-in encoders with the same output shapes (no download); see
> `docs/encoders/SPEECHCARE_SPEC.md`, section E.

### Purpose
A port of SpeechCARE-AGF (`SpeechCARE/SpeechCARE-NIA-Phase2`): mHuBERT-147 over 5 s segments with 20% overlap, concatenated without
per-segment pooling, a learnable CLS token and positional encoding, two attention blocks; mGTE for the transcript; an age-bin branch;
and a gate over the three modality heads. It **jointly fine-tunes** about 400 million parameters for each of 51 runs, so
it needs a GPU. Roughly 30 minutes to 2 hours per run on a GPU, which was not measured.

### Run order
1. `06_speechcare_prepare_data.py`: writes the segment tensors, `demography.csv` and `text.csv`. Of the order of 2 GB of tensors (estimated from the segment count, not measured).
2. Time **one** run of `07_speechcare_train_eval.py` and check the forward pass before running all 51 unattended.
3. `07_speechcare_train_eval.py` (`TASK` constant: `binary` or `multiclass`). Writes `speechcare_<TASK>_predictions.csv`.
4. Evaluate with `Step7_Evaluation/Step7b_Evaluate/03_evaluate.py` (`MODEL_NAME = "speechcare"`).

### Changed in the cleanup
`07` selected the epoch (scheduler and best checkpoint) on the set it then predicted, which in the final run was the **test
partition**. It now holds out `INNER_VAL_FRACTION` (0.10) of the training uids for that. Details and the remaining open points
are in `docs/encoders/SPEECHCARE_SPEC.md`, section E.

### Inputs and outputs
Reads the **raw** recordings, `paths.cohort`, `stage1.inclusion_list` and the frozen splits. Writes under
`stage1.speechcare_data_dir` and `stage1.predictions_dir` (gitignored).

### Dependencies
`torch`, `transformers`, `librosa`, `scikit-learn`, `pandas`, `numpy`. Checkpoints `utter-project/mHuBERT-147` and
`Alibaba-NLP/gte-multilingual-base` are downloaded on first use.

```bash
python src/Step8_Encoders/speechcare/06_speechcare_prepare_data.py
python src/Step8_Encoders/speechcare/07_speechcare_train_eval.py
```
