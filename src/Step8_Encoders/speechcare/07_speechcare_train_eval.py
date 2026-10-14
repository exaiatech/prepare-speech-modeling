"""
SpeechCARE-AGF, script 07: model definition, training and evaluation across all splits.

The model classes below (MultiHeadAttentionAddNorm, GatingNetwork, TBNet) are a direct
port of the confirmed source in model/model.py, adapted only where our data pipeline
differs from the original (see SPEECHCARE_SPEC.md Sections B/B2/C for exactly what is
confirmed vs. reconstructed).

DEVIATION, confirmed necessary and documented: num_labels is set from TASK (2 for
binary, 3 for multiclass) rather than hardcoded to 3, since Daniel's brief asks for
both settings as separate experiments, not just the original 3-class task.

DEVIATION, practical: max_num_segments is computed from our own segmented data (06_speechcare_prepare_data.py)
rather than read from a config field we never located.

FIX AFTER REVIEW (protocol, not a design change): the scheduler and the "minimum validation loss"
checkpoint selection used to be computed on the very set the run then predicted. In a CV split that
selects the epoch on the held-out fold. In the final run it selected the epoch on the TEST partition,
which is then no longer evaluated once. The selection now uses an inner validation set holding out
INNER_VAL_FRACTION of the training uids, stratified by label and seeded. The held-out set is only
predicted, after training. The criterion itself (minimum validation loss, as in the original code) is
unchanged. See docs/encoders/SPEECHCARE_SPEC.md, "Review notes".

COMPUTE WARNING: unlike the Whisper-medium baseline (frozen encoder, linear probe
only), this jointly fine-tunes mHuBERT-147 (98M params) and mGTE (305M params) for
every split. This is a fundamentally heavier job (needs a GPU; roughly 30 min to 2 h per run),
so time one run and check the real forward pass before running the full 50-split protocol unattended.
"""

TASK = "binary"   # "binary" or "multiclass"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import load_config, resolve_path
_cfg = load_config()   # config.yaml, merged with config.local.yaml when present
SEGMENTS_CSV   = str(resolve_path(_cfg["stage1"]["speechcare_data_dir"]) / "segments.csv")
DEMOGRAPHY_CSV = str(resolve_path(_cfg["stage1"]["speechcare_data_dir"]) / "demography.csv")
TEXT_CSV       = str(resolve_path(_cfg["stage1"]["speechcare_data_dir"]) / "text.csv")
FIXED_SPLIT_CSV = str(resolve_path(_cfg["stage1"]["splits_dir"]) / "fixed_split_taskA_binary.csv")
CV_SPLITS_CSV   = str(resolve_path(_cfg["stage1"]["splits_dir"]) / "cv_splits_taskA_binary.csv")
OUT_DIR        = str(resolve_path(_cfg["stage1"]["predictions_dir"]))

if TASK == "multiclass":
    FIXED_SPLIT_CSV = FIXED_SPLIT_CSV.replace("taskA_binary", "taskB_multiclass")
    CV_SPLITS_CSV   = CV_SPLITS_CSV.replace("taskA_binary", "taskB_multiclass")

# ── Confirmed hyperparameters (config.yml + main.py) ───────────────────────
SEED           = 133
BATCH_SIZE     = 4
EPOCHS         = 14
HIDDEN_SIZE    = 128
DEMOGRAPHY_HIDDEN_SIZE = 128
SPEECH_CHP     = "utter-project/mHuBERT-147"
TEXT_CHP       = "Alibaba-NLP/gte-multilingual-base"
LR_MAIN        = 1e-5          # mHuBERT + CSE + fusion/classifier heads
LR_TEXT        = 1e-6          # mGTE
WD_DECAY       = 0.005         # literal code value; config.yml says 1e-3 but is not
                                # actually read by the optimizer construction - see spec
SCHED_FACTOR, SCHED_PATIENCE, SCHED_MIN_LR, SCHED_THRESH = 0.8, 4, 1e-6, 0.01

# Share of the TRAINING uids held out inside every run, used only for ReduceLROnPlateau and the
# minimum-validation-loss checkpoint. The original code had a separate validation set for this;
# a CV fold has none, so it is carved out of the training uids here. 0.10 is our choice, not
# the paper's (see the review notes).
INNER_VAL_FRACTION = 0.10

RUN_NAME = f"speechcare_{TASK}"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor

torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cpu"
try:
    import intel_extension_for_pytorch as ipex  # noqa: F401
    if torch.xpu.is_available():
        device = "xpu"
except Exception:
    pass
if torch.cuda.is_available():
    device = "cuda"
print(f"Using device: {device}")

out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)

CLASSES = ["Control", "AD/ADRD"] if TASK == "binary" else ["Control", "MCI", "AD/ADRD"]
NUM_LABELS = len(CLASSES)
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
COL_FOR_CLASS = {"Control": "p_control", "MCI": "p_mci", "AD/ADRD": "p_adrd"}

# ── Data ─────────────────────────────────────────────────────────────────
segments = pd.read_csv(SEGMENTS_CSV)
demo = pd.read_csv(DEMOGRAPHY_CSV)
text = pd.read_csv(TEXT_CSV)
labels_df = pd.read_csv(FIXED_SPLIT_CSV)[["ID", "label"]].rename(columns={"ID": "uid"})

MAX_NUM_SEGMENTS = int(segments.groupby("uid").size().max())
print(f"max_num_segments = {MAX_NUM_SEGMENTS} (from script 06's segmented data)")

print(f"Loading {SPEECH_CHP} and {TEXT_CHP} ...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SPEECH_CHP)
tokenizer = AutoTokenizer.from_pretrained(TEXT_CHP, trust_remote_code=True)


# ── Model (ported from model/model.py) ──────────────────────────────────
class MultiHeadAttentionAddNorm(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        return self.norm(x + self.dropout(attn_output))


class GatingNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 3)   # speech, text, demography
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.fc(x))


class TBNet(nn.Module):
    def __init__(self, num_labels, hidden_size, demography_hidden_size,
                 max_num_segments, speech_chp, text_chp):
        super().__init__()
        self.speech_transformer = AutoModel.from_pretrained(speech_chp)
        self.txt_transformer = AutoModel.from_pretrained(text_chp, trust_remote_code=True)

        speech_dim = self.speech_transformer.config.hidden_size
        txt_dim = self.txt_transformer.config.hidden_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, speech_dim))
        # max_seq_length: max_num_segments * (frames per 5s segment at mHuBERT's
        # ~20ms stride - 1) + 1 for CLS, matching model.py's formula
        max_seq_length = int(max_num_segments * ((5.0 / 0.02) - 1)) + 1
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, speech_dim))

        self.layers = nn.ModuleList([
            MultiHeadAttentionAddNorm(speech_dim, 4, 0.1) for _ in range(2)
        ])

        self.speech_head = nn.Sequential(nn.Linear(speech_dim, hidden_size), nn.Tanh())
        self.txt_head = nn.Sequential(nn.Linear(txt_dim, hidden_size), nn.Tanh())
        self.demography_head = nn.Sequential(nn.Linear(1, demography_hidden_size), nn.Tanh())

        self.speech_classifier = nn.Linear(hidden_size, num_labels)
        self.txt_classifier = nn.Linear(hidden_size, num_labels)
        self.demography_classifier = nn.Linear(demography_hidden_size, num_labels)

        self.weight_gate = GatingNetwork(hidden_size * 2 + demography_hidden_size)

    def forward(self, input_values, input_ids, demography, attention_mask):
        batch_size, num_segments, seq_length = input_values.size()
        input_values = input_values.view(batch_size * num_segments, seq_length)
        speech_embeddings = self.speech_transformer(input_values).last_hidden_state
        speech_embeddings = speech_embeddings.view(batch_size, num_segments, -1,
                                                     speech_embeddings.size(-1))
        speech_embeddings = speech_embeddings.view(batch_size,
                                                     num_segments * speech_embeddings.size(2), -1)

        txt_embeddings = self.txt_transformer(input_ids=input_ids, attention_mask=attention_mask)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        speech_embeddings = torch.cat((cls_tokens, speech_embeddings), dim=1)
        speech_embeddings = speech_embeddings + self.positional_encoding[:, :speech_embeddings.size(1), :]

        for layer in self.layers:
            speech_embeddings = layer(speech_embeddings)

        speech_cls = speech_embeddings[:, 0, :]
        txt_cls = txt_embeddings.last_hidden_state[:, 0, :]

        demography = demography.unsqueeze(1)
        demography_x = self.demography_head(demography)
        speech_x = self.speech_head(speech_cls)
        txt_x = self.txt_head(txt_cls)

        gate_weights = self.weight_gate(torch.cat([speech_x, txt_x, demography_x], dim=1))

        speech_out = self.speech_classifier(speech_x)
        txt_out = self.txt_classifier(txt_x)
        demography_out = self.demography_classifier(demography_x)

        w_speech, w_txt, w_demo = gate_weights[:, 0], gate_weights[:, 1], gate_weights[:, 2]
        fused = (w_speech.unsqueeze(1) * speech_out + w_txt.unsqueeze(1) * txt_out
                 + w_demo.unsqueeze(1) * demography_out)
        return fused


# ── Dataset + collate ────────────────────────────────────────────────────
class SpeechCAREDataset(Dataset):
    def __init__(self, uids):
        self.uids = uids

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        seg_rows = segments[segments["uid"] == uid].sort_values("segment")
        waveforms = [torch.load(p) for p in seg_rows["path"]]
        txt = text.loc[text["uid"] == uid, "transcription"].values[0]
        age_bin = demo.loc[demo["uid"] == uid, "age_bin"].values[0]
        label = labels_df.loc[labels_df["uid"] == uid, "label"].values[0]
        return waveforms, txt, float(age_bin), LABEL_TO_IDX[label], uid


def collate_fn(batch):
    waveforms_list, texts, demographies, labels, uids = zip(*batch)

    # pad/truncate every sample to MAX_NUM_SEGMENTS segments (zero-segments for
    # padding; the attention mechanism has no explicit segment-level mask here,
    # matching the fact that model.py's forward() does not take one either -
    # this is a known simplification)
    seg_len = waveforms_list[0][0].shape[-1] if waveforms_list[0] else int(5 * 16000)
    padded = []
    for wfs in waveforms_list:
        wfs = list(wfs)[:MAX_NUM_SEGMENTS]
        while len(wfs) < MAX_NUM_SEGMENTS:
            wfs.append(torch.zeros(seg_len))
        # feature_extractor normalises each segment independently, matching how
        # Wav2Vec2FeatureExtractor is meant to be used (per-waveform normalisation)
        normed = [feature_extractor(w.numpy(), sampling_rate=16000,
                                     return_tensors="pt").input_values.squeeze(0)
                  for w in wfs]
        padded.append(torch.stack(normed))
    input_values = torch.stack(padded)   # [batch, MAX_NUM_SEGMENTS, seq_len]

    tok = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)

    demography_t = torch.tensor(demographies, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    return (input_values, tok["input_ids"], demography_t, tok["attention_mask"],
            labels_t, list(uids))


# ── Train / eval for one split ──────────────────────────────────────────
def inner_split(train_uids):
    """Splits the training uids into (fit, selection). Stratified by label and seeded, so the
    same call always gives the same split. The selection part only drives the scheduler and the
    checkpoint choice. It never overlaps the set that is predicted at the end."""
    y = labels_df.set_index("uid").loc[list(train_uids), "label"].values
    fit, sel = train_test_split(list(train_uids), test_size=INNER_VAL_FRACTION,
                                stratify=y, random_state=SEED)
    return fit, sel


def run_split(train_uids, val_uids):
    """Trains on train_uids and returns predictions for val_uids.

    val_uids is the held-out set (a CV fold, or the test partition). It is PREDICTED once, after
    training, and is never used to choose the epoch or to step the scheduler."""
    fit_uids, sel_uids = inner_split(train_uids)
    model = TBNet(NUM_LABELS, HIDDEN_SIZE, DEMOGRAPHY_HIDDEN_SIZE,
                   MAX_NUM_SEGMENTS, SPEECH_CHP, TEXT_CHP).to(device)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if "txt_transformer" in name:
            continue
        if any(k in name for k in ("bias", "LayerNorm", "norm", "positional_encoding")):
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay, "lr": LR_MAIN, "weight_decay": WD_DECAY},
        {"params": no_decay, "lr": LR_MAIN, "weight_decay": 0.0},
        {"params": model.txt_transformer.parameters(), "lr": LR_TEXT, "weight_decay": WD_DECAY},
    ])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=SCHED_FACTOR,
                                   patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR,
                                   threshold=SCHED_THRESH)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(SpeechCAREDataset(fit_uids), batch_size=BATCH_SIZE,
                               shuffle=True, collate_fn=collate_fn)
    sel_loader = DataLoader(SpeechCAREDataset(sel_uids), batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(SpeechCAREDataset(val_uids), batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)   # predicted once, at the end

    best_val_loss, best_state = float("inf"), None
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            iv, ids, demo_b, mask, lab, _ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            optimizer.zero_grad()
            out_logits = model(iv, ids, demo_b, mask)
            loss = criterion(out_logits, lab)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in sel_loader:
                iv, ids, demo_b, mask, lab, _ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                out_logits = model(iv, ids, demo_b, mask)
                val_losses.append(criterion(out_logits, lab).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        scheduler.step(val_loss)
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"    epoch {epoch+1}/{EPOCHS}  val_loss={val_loss:.4f}")

    if best_state is None:
        raise RuntimeError("no epoch produced a finite selection loss; nothing to restore")
    model.load_state_dict(best_state)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in val_loader:
            iv, ids, demo_b, mask, lab, uids_b = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            proba = F.softmax(model(iv, ids, demo_b, mask), dim=1).cpu().numpy()
            for i, uid in enumerate(uids_b):
                row = {"ID": uid, "y_true": CLASSES[lab[i].item()]}
                for c_idx, c_name in enumerate(CLASSES):
                    row[COL_FOR_CLASS[c_name]] = proba[i, c_idx]
                rows.append(row)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


# ── Main: all CV splits + final fixed test ───────────────────────────────
all_rows = []
cv = pd.read_csv(CV_SPLITS_CSV)
split_ids = sorted(cv["split_id"].unique())
print(f"Running {len(split_ids)} CV splits (each is a full SpeechCARE-AGF training run) ...")

for sid in split_ids:
    tr_uids = cv[(cv["split_id"] == sid) & (cv["role"] == "train")]["ID"].tolist()
    va_uids = cv[(cv["split_id"] == sid) & (cv["role"] == "val")]["ID"].tolist()
    print(f"  split {sid}/{len(split_ids)}: train={len(tr_uids)} val={len(va_uids)}")
    rows = run_split(tr_uids, va_uids)
    for r in rows:
        r["split_id"] = sid
        r["role"] = "val"
    all_rows += rows
    # checkpoint predictions after every split, in case a later split fails
    pd.DataFrame(all_rows).to_csv(out / f"{RUN_NAME}_predictions.csv", index=False)

fixed = pd.read_csv(FIXED_SPLIT_CSV)
trdev_uids = fixed[fixed["partition"].isin(["train", "dev"])]["ID"].tolist()
test_uids = fixed[fixed["partition"] == "test"]["ID"].tolist()
print(f"\nFinal test run: train+dev={len(trdev_uids)}, test={len(test_uids)}")
test_rows = run_split(trdev_uids, test_uids)
for r in test_rows:
    r["split_id"] = None
    r["role"] = "test"
all_rows += test_rows

pred_df = pd.DataFrame(all_rows)
pred_df.to_csv(out / f"{RUN_NAME}_predictions.csv", index=False)
print(f"\nSaved {len(pred_df)} prediction rows -> {out / f'{RUN_NAME}_predictions.csv'}")
print("Feed this file into 03_evaluate.py with MODEL_NAME='speechcare'.")
