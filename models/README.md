# Models

**This directory is empty in the repository.** The Whisper checkpoint is 2.9 GB
and is not committed. You download it yourself before running Step 4b.

## What goes here

```
models/
  large-v3-local/
    config.json
    model.bin                 2.9 GB
    preprocessor_config.json
    tokenizer.json
    vocabulary.json
```

`Step4_ASR/Step4b_Transcription/step2_speech_to_text.py` loads this path
directly and does **not** fall back to downloading from the internet at run
time, so the transcription step fails with a clear error until the checkpoint
is present.

## How to get it

The checkpoint is the CTranslate2 conversion of Whisper large-v3 that
`faster-whisper` uses. The simplest way is to let `faster-whisper` fetch it
once from the Hugging Face Hub and then point the script at the cached copy:

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3')"
```

That downloads to the Hugging Face cache. Either copy the resulting folder to
`models/large-v3-local/`, or edit `LOCAL_MODEL_PATH` at the top of
`step2_speech_to_text.py` to point at the cache.

To download it explicitly instead:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-large-v3', local_dir='models/large-v3-local')"
```

## Why large-v3 and not a smaller model

The transcription settings are tuned to capture hesitations, stutters and
cognitive pauses verbatim rather than to produce clean readable text, because
those disfluencies are the clinical signal. Smaller Whisper variants normalise
them away far more aggressively. If you swap the model you must re-run the
Step 4c to 4e evaluation, since the reported word error rates no longer apply.

## Hardware

`device="auto"` picks a GPU when one is available and falls back to CPU. On CPU
the 1331-recording set takes many hours; on a recent GPU it is closer to one.
