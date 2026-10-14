"""
Stage 1, Whisper-medium, script 04: extract embeddings.

Reproduces Li et al. (2025, arXiv:2506.11119)'s audio pipeline as closely as
possible:
  - Whisper-medium encoder (frozen, not fine-tuned)
  - 16 kHz input
  - clip-level embedding = mean-pooled encoder hidden state (1024-dim)

This is the expensive step and is run ONCE for all 1,031 Cookie Theft
recordings, regardless of which task (binary/3-class) or which of the 50
CV splits will use them later. The small classifier that uses these
embeddings is trained separately per split in 05_whisper_train_eval.py.

Output: one row per recording, embedding dims + ID, saved as a single file
so 05_whisper_train_eval.py never has to touch audio again.
"""

# ── Paths ───────────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import load_config, resolve_path
_cfg = load_config()   # config.yaml, merged with config.local.yaml when present
INCLUSION_LIST = str(resolve_path(_cfg["stage1"]["inclusion_list"]))
AUDIO_DIRS     = [str(resolve_path(_cfg["paths"]["raw_audio"][k])) for k in ("train", "test")]
OUT_DIR        = str(resolve_path(_cfg["stage1"]["embeddings_dir"]))

WHISPER_CHECKPOINT = "openai/whisper-medium"   # 769M params, encoder only used
SAMPLE_RATE         = 16000                     # Whisper's required input rate
# ──────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
done_log = out / "_done.json"
done = set(json.loads(done_log.read_text())) if done_log.exists() else set()

# ── Device: CPU by default; use Intel XPU if available, never block on it ──
device = "cpu"
try:
    import intel_extension_for_pytorch as ipex  # noqa: F401
    if torch.xpu.is_available():
        device = "xpu"
except Exception:
    pass
print(f"Using device: {device}")

print(f"Loading {WHISPER_CHECKPOINT} (encoder only, frozen) ...")
feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_CHECKPOINT)
model = WhisperModel.from_pretrained(WHISPER_CHECKPOINT).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# ── Locate audio ──────────────────────────────────────────────────────────
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a"}
audio_index = {}
for d in AUDIO_DIRS:
    for p in Path(d).rglob("*"):
        if p.suffix.lower() in AUDIO_EXT:
            audio_index[p.stem] = p

df = pd.read_csv(INCLUSION_LIST)
df = df[df["audio_matches"] == 1].reset_index(drop=True)
print(f"{len(df)} recordings to embed; {len(done)} already done (resuming)")


@torch.no_grad()
def embed_one(path):
    # librosa.load handles mono conversion and resampling in one call, and
    # does not depend on torchaudio's I/O backend (torchcodec), which is not
    # installed. Kept consistent with the rest of the acoustic pipeline.
    wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)

    inputs = feat_extractor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # Encoder only - no decoder/transcription needed for embeddings.
    encoder_out = model.encoder(input_features).last_hidden_state  # (1, T, 1024)
    clip_embedding = encoder_out.mean(dim=1).squeeze(0)            # mean-pool over time
    return clip_embedding.float().cpu().numpy()


rows = []
for i, r in df.iterrows():
    uid = r["ID"]
    if uid in done:
        continue
    path = audio_index.get(uid)
    if path is None:
        print(f"  [!] no audio for {uid}, skipping")
        continue
    try:
        emb = embed_one(path)
        np.save(out / f"{uid}.npy", emb)
        done.add(uid)
        if (i + 1) % 25 == 0:
            done_log.write_text(json.dumps(sorted(done)))  # checkpoint
            print(f"  {i+1}/{len(df)}  ({uid})")
    except Exception as e:
        print(f"  [!] failed on {uid}: {e}")

done_log.write_text(json.dumps(sorted(done)))
print(f"\nDone: {len(done)} embeddings saved as individual .npy files in {out}")

# ── Pack everything into one file for convenience ──────────────────────────
ids, vecs = [], []
for uid in sorted(done):
    f = out / f"{uid}.npy"
    if f.exists():
        ids.append(uid)
        vecs.append(np.load(f))

packed = pd.DataFrame(np.vstack(vecs), columns=[f"dim_{i}" for i in range(vecs[0].shape[0])])
packed.insert(0, "ID", ids)
packed.to_csv(out / "whisper_medium_embeddings.csv", index=False)
print(f"Packed into {out / 'whisper_medium_embeddings.csv'}  "
      f"({packed.shape[0]} rows x {packed.shape[1]-1} dims)")
