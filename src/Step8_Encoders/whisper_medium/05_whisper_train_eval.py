"""
Stage 1, Whisper-medium, script 05: train and evaluate.

Reads the embeddings from 04_whisper_extract_embeddings.py (one 1024-dim vector per recording),
concatenates age + sex, and trains Li et al.'s classification head:
  - single hidden layer, 128 units
  - Adam, lr 5e-4, batch 32, early stopping with patience 5

Runs this once per CV split (as produced by 02_build_splits.py) and once
on the fixed train+dev -> test partition, for ONE task at a time (binary
or 3-class - set TASK below). Output matches the format 03_evaluate.py
expects.

DEVIATION FROM THE PAPER (documented, not silent):
Li et al. used 5 repeated random 80/20 splits. Per Daniel's instruction we
use 5-fold x 10-repeat CV (50 splits) on the train+dev pool instead, plus
one run on the fixed test partition. The classifier architecture and
training hyperparameters follow the paper; the resampling scheme follows
Daniel's protocol.
"""

TASK = "binary"   # "binary" or "multiclass"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import load_config, resolve_path
_cfg = load_config()   # config.yaml, merged with config.local.yaml when present
EMBEDDINGS_CSV   = str(resolve_path(_cfg["stage1"]["embeddings_dir"]) / "whisper_medium_embeddings.csv")
INCLUSION_LIST   = str(resolve_path(_cfg["stage1"]["inclusion_list"]))
COOKIE_CSV       = str(resolve_path(_cfg["paths"]["cohort"]))
FIXED_SPLIT_CSV  = str(resolve_path(_cfg["stage1"]["splits_dir"]) / "fixed_split_taskA_binary.csv")
CV_SPLITS_CSV    = str(resolve_path(_cfg["stage1"]["splits_dir"]) / "cv_splits_taskA_binary.csv")
OUT_DIR          = str(resolve_path(_cfg["stage1"]["predictions_dir"]))

if TASK == "multiclass":
    FIXED_SPLIT_CSV = FIXED_SPLIT_CSV.replace("taskA_binary", "taskB_multiclass")
    CV_SPLITS_CSV   = CV_SPLITS_CSV.replace("taskA_binary", "taskB_multiclass")

RANDOM_STATE = 42
HIDDEN_UNITS = 128     # Li et al.
LEARNING_RATE = 5e-4   # Li et al.
BATCH_SIZE = 32        # Li et al.
PATIENCE = 5           # Li et al., early stopping
MAX_EPOCHS = 200

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn

out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)

device = "cpu"
try:
    import intel_extension_for_pytorch as ipex  # noqa: F401
    if torch.xpu.is_available():
        device = "xpu"
except Exception:
    pass

# ── Load and merge ─────────────────────────────────────────────────────
emb = pd.read_csv(EMBEDDINGS_CSV)
meta = pd.read_csv(INCLUSION_LIST)
dim_cols = [c for c in emb.columns if c.startswith("dim_")]
print(f"Embeddings: {emb.shape[0]} recordings x {len(dim_cols)} dims")

age_sex_source = pd.read_csv(FIXED_SPLIT_CSV)  # has ID, label, partition (not used further)
import csv
with open(COOKIE_CSV) as f:
    header = next(csv.reader(f))
age_col = "age" if "age" in header else None
sex_col = "gender" if "gender" in header else ("sex" if "sex" in header else None)
demo = pd.read_csv(COOKIE_CSV, usecols=[c for c in ["ID", age_col, sex_col] if c])
emb = emb.merge(demo, on="ID", how="left")

le_sex = LabelEncoder()
emb["sex_code"] = le_sex.fit_transform(emb[sex_col].astype(str))
feature_cols = dim_cols + [age_col, "sex_code"]

labels_df = pd.read_csv(FIXED_SPLIT_CSV)[["ID", "label"]]
emb = emb.merge(labels_df, on="ID", how="inner")

CLASSES = ["Control", "AD/ADRD"] if TASK == "binary" else ["Control", "MCI", "AD/ADRD"]
le_y = LabelEncoder()
le_y.fit(CLASSES)
print(f"Task: {TASK}  |  classes: {list(le_y.classes_)}  |  n={len(emb)}")


class Head(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_UNITS), nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, n_classes))

    def forward(self, x):
        return self.net(x)


def train_and_predict(train_df, val_df):
    scaler = StandardScaler().fit(train_df[feature_cols].values)
    X_tr = torch.tensor(scaler.transform(train_df[feature_cols].values), dtype=torch.float32)
    y_tr = torch.tensor(le_y.transform(train_df["label"].values), dtype=torch.long)
    X_val = torch.tensor(scaler.transform(val_df[feature_cols].values), dtype=torch.float32)

    model = Head(X_tr.shape[1], len(CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # simple internal holdout (10% of train) for early stopping, since the
    # paper uses early stopping but a CV fold has no separate dev set here
    n_hold = max(1, int(0.1 * len(X_tr)))
    perm = torch.randperm(len(X_tr))
    hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]

    best_loss, bad_epochs, best_state = float("inf"), 0, None
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = fit_idx[torch.randperm(len(fit_idx))]
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(X_tr[idx].to(device)), y_tr[idx].to(device))
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            hold_loss = loss_fn(model(X_tr[hold_idx].to(device)), y_tr[hold_idx].to(device)).item()
        if hold_loss < best_loss:
            best_loss, bad_epochs, best_state = hold_loss, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(X_val.to(device)), dim=1).cpu().numpy()
    return proba


COL_FOR_CLASS = {"Control": "p_control", "MCI": "p_mci", "AD/ADRD": "p_adrd"}

def rows_from_proba(ids, y_true, proba, split_id, role):
    rows = []
    for i, uid in enumerate(ids):
        row = {"ID": uid, "split_id": split_id, "role": role, "y_true": y_true[i]}
        for c_idx, c_name in enumerate(le_y.classes_):
            row[COL_FOR_CLASS[c_name]] = proba[i, c_idx]
        rows.append(row)
    return rows


all_rows = []

# ── A. Repeated CV on train+dev ─────────────────────────────────────────
cv = pd.read_csv(CV_SPLITS_CSV)
split_ids = sorted(cv["split_id"].unique())
print(f"Running {len(split_ids)} CV splits ...")
for sid in split_ids:
    tr_ids = cv[(cv["split_id"] == sid) & (cv["role"] == "train")]["ID"]
    va_ids = cv[(cv["split_id"] == sid) & (cv["role"] == "val")]["ID"]
    tr = emb[emb["ID"].isin(tr_ids)].reset_index(drop=True)
    va = emb[emb["ID"].isin(va_ids)].reset_index(drop=True)
    if tr["label"].nunique() < len(CLASSES):
        print(f"  [!] split {sid}: missing a class in train, skipped")
        continue
    proba = train_and_predict(tr, va)
    all_rows += rows_from_proba(va["ID"].tolist(), va["label"].tolist(), proba, sid, "val")
    if sid % 10 == 0:
        print(f"  split {sid}/{len(split_ids)} done")

# ── B. Fixed test partition: train on train+dev pool, test once ────────
fixed = pd.read_csv(FIXED_SPLIT_CSV)
trdev_ids = fixed[fixed["partition"].isin(["train", "dev"])]["ID"]
test_ids  = fixed[fixed["partition"] == "test"]["ID"]
tr = emb[emb["ID"].isin(trdev_ids)].reset_index(drop=True)
te = emb[emb["ID"].isin(test_ids)].reset_index(drop=True)
print(f"\nFinal test run: train+dev n={len(tr)}, test n={len(te)}")
proba = train_and_predict(tr, te)
all_rows += rows_from_proba(te["ID"].tolist(), te["label"].tolist(), proba, None, "test")

pred_df = pd.DataFrame(all_rows)
out_path = out / f"whisper_medium_{TASK}_predictions.csv"
pred_df.to_csv(out_path, index=False)
print(f"\nSaved {len(pred_df)} prediction rows -> {out_path}")
print("Feed this file into 03_evaluate.py (set MODEL_NAME accordingly).")
