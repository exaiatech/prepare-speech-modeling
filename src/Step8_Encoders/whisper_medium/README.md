# Whisper-medium baseline

> **Status: run and verified by the author.**

### Purpose
Reproduces the audio pipeline of Li et al. (arXiv:2506.11119) as closely as possible: a frozen Whisper-medium encoder, a 16 kHz
input, a clip-level embedding mean-pooled over time (1,024 dimensions). A small head (one hidden layer of 128 units, Adam,
learning rate 0.0005, batch 32, early stopping with patience 5) is trained on the embedding with age and sex concatenated.

### Run order
1. `04_whisper_extract_embeddings.py`: the expensive step, run once for all recordings, resumable.
2. `05_whisper_train_eval.py`: trains the head on all 50 CV splits and once on the fixed partition, for one task at a time
   (`TASK` constant at the top: `binary` or `multiclass`). Writes `whisper_medium_<TASK>_predictions.csv`.
3. Evaluate with `Step7_Evaluation/Step7b_Evaluate/03_evaluate.py` (`MODEL_NAME = "whisper_medium"`).

### Inputs and outputs
* Reads the **raw** recordings (`paths.raw_audio.*`), `stage1.inclusion_list`, the frozen splits and `paths.cohort`.
* Writes under `stage1.embeddings_dir` and `stage1.predictions_dir` (gitignored). Nothing here is committed.

### Deviation from the paper (documented)
Li et al. used five random 80/20 splits. The project uses 5-fold x 10-repeat CV on train+dev plus one fixed test run.

### Known issues (not changed in the cleanup)
See `docs/encoders/encoder_input_preparation.md`: no amplitude normalisation, 30 s padding included in the mean pool,
`RANDOM_STATE` defined but never applied, up to 200 epochs and no `ReduceLROnPlateau`, whereas the paper trains up to 100 epochs with it.

### Dependencies
`torch`, `transformers`, `librosa`, `scikit-learn`, `pandas`, `numpy`. The Whisper checkpoint is downloaded from the Hugging Face Hub on first use.

```bash
python src/Step8_Encoders/whisper_medium/04_whisper_extract_embeddings.py
python src/Step8_Encoders/whisper_medium/05_whisper_train_eval.py
```
