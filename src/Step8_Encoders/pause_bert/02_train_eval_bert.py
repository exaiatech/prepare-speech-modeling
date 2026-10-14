"""
Pause-annotated BERT, script 02: train and evaluate.

SCAFFOLD, NOT RUN, UNTESTED. It has never been run on the real transcripts or with the real
BERT checkpoint. Only its plumbing (reading the frozen splits, building the features, writing the
predictions file) was exercised, with random embeddings, through --dry-run.

It follows Li et al. (arXiv:2506.11119):
    * frozen bert-base-uncased, the transcript padded or truncated to 512 tokens, a 768-dimensional
      embedding (paper); which vector, the [CLS] one or a mean over tokens, is not stated (ours,
      stage1.pause_bert.embedding);
    * age and sex concatenated to the embedding (paper);
    * one hidden layer of 128 units, Adam, learning rate 0.0005, batch size 32, up to 100 epochs,
      ReduceLROnPlateau, early stopping after 5 epochs without validation-loss improvement (paper);
    * 20% of the training set held out for validation (paper).

Deviation, the same as for the Whisper script: the paper used five random 80/20 splits. We use the
frozen protocol of the project, 50 repeated 5-fold CV splits on train+dev and one fixed test run,
read from stage1.splits_dir (written once by Step7a_Build_Splits and never regenerated).

THE TEST PARTITION IS PREDICTED ONCE. The validation set that drives the scheduler and early
stopping is carved out of the TRAINING uids of every run, so neither a CV fold nor the test
partition takes part in choosing the epoch.

TWO TEXT VARIANTS, SAME CODE
    --text pause   the pause-annotated transcript (model name pause_bert)
    --text plain   the same transcript without markers (model name bert_plain_text), the baseline
                   the paper compares with

ALIGNMENT FAILURES. Recordings with alignment_ok = 0 have no annotated text. By default
(stage1.pause_bert.on_alignment_failure = error) the script stops and lists them. 'exclude' drops
them and prints how many, which makes the evaluated cohort differ from the other models'. It never
substitutes the plain text silently.

OUTPUT
    <predictions_dir>/<model>_<task>_predictions.csv with columns
    ID, split_id, role, y_true, p_control, p_mci, p_adrd, the format 03_evaluate.py reads.

HOW TO RUN
    python src/Step8_Encoders/pause_bert/02_train_eval_bert.py --task binary --text pause
    python src/Step8_Encoders/pause_bert/02_train_eval_bert.py --task multiclass --text pause
    python src/Step8_Encoders/pause_bert/02_train_eval_bert.py --dry-run      # plumbing check only
Then evaluate with Step7b_Evaluate/03_evaluate.py (MODEL_NAME = "pause_bert").
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import load_config, resolve_path

COL_FOR_CLASS = {"Control": "p_control", "MCI": "p_mci", "AD/ADRD": "p_adrd"}


def sex_code(gender):
    g = str(gender).strip().lower()
    if g in ("female", "f"):
        return 1.0
    if g in ("male", "m"):
        return 0.0
    raise ValueError(f"unrecognised gender value {gender!r}")


def bert_embeddings(texts, pcfg, device):
    """Frozen BERT, one embedding per text. Not exercised by --dry-run."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pcfg["bert_checkpoint"])
    model = AutoModel.from_pretrained(pcfg["bert_checkpoint"]).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            enc = tok(texts[i:i + 16], padding="max_length", truncation=True,
                      max_length=pcfg["max_length"], return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            if pcfg["embedding"] == "cls":
                v = h[:, 0, :]
            else:
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
            out.append(v.cpu().numpy())
    return np.vstack(out)


class Head(nn.Module):
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def train_and_predict(train_df, eval_df, feature_cols, classes, pcfg, seed, device, max_epochs):
    """Fits on part of train_df, selects the epoch on the rest, predicts eval_df once."""
    fit_df, sel_df = train_test_split(train_df, test_size=pcfg["inner_val_fraction"],
                                      stratify=train_df["label"], random_state=seed)
    scaler = StandardScaler().fit(fit_df[feature_cols].values)

    def tensors(df, with_y=True):
        x = torch.tensor(scaler.transform(df[feature_cols].values), dtype=torch.float32)
        if not with_y:
            return x, None
        y = torch.tensor([classes.index(c) for c in df["label"]], dtype=torch.long)
        return x, y

    x_fit, y_fit = tensors(fit_df)
    x_sel, y_sel = tensors(sel_df)
    x_eval, _ = tensors(eval_df, with_y=False)

    torch.manual_seed(seed)
    model = Head(x_fit.shape[1], pcfg["hidden_units"], len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=pcfg["learning_rate"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=pcfg["scheduler_factor"], patience=pcfg["scheduler_patience"])
    loss_fn = nn.CrossEntropyLoss()

    best, bad, best_state = float("inf"), 0, None
    for _ in range(max_epochs):
        model.train()
        order = torch.randperm(len(x_fit))
        for i in range(0, len(order), pcfg["batch_size"]):
            idx = order[i:i + pcfg["batch_size"]]
            opt.zero_grad()
            loss_fn(model(x_fit[idx].to(device)), y_fit[idx].to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sel_loss = loss_fn(model(x_sel.to(device)), y_sel.to(device)).item()
        sched.step(sel_loss)
        if sel_loss < best:
            best, bad, best_state = sel_loss, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= pcfg["patience"]:
                break
    if best_state is None:
        raise RuntimeError("no epoch produced a finite validation loss")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x_eval.to(device)), dim=1).cpu().numpy()


def prediction_rows(ids, y_true, proba, classes, split_id, role):
    rows = []
    for i, uid in enumerate(ids):
        row = {"ID": uid, "split_id": split_id, "role": role, "y_true": y_true[i]}
        for j, c in enumerate(classes):
            row[COL_FOR_CLASS[c]] = float(proba[i, j])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--text", choices=["pause", "plain"], default="pause")
    ap.add_argument("--dry-run", action="store_true",
                    help="random embeddings, two CV splits, two epochs; checks the plumbing only")
    args = ap.parse_args()

    cfg = load_config()
    s1, pcfg = cfg["stage1"], cfg["stage1"]["pause_bert"]
    splits = resolve_path(s1["splits_dir"])
    suffix = "taskA_binary" if args.task == "binary" else "taskB_multiclass"
    classes = ["Control", "AD/ADRD"] if args.task == "binary" else ["Control", "MCI", "AD/ADRD"]
    name = "pause_bert" if args.text == "pause" else "bert_plain_text"
    text_col = "text_pause_annotated" if args.text == "pause" else "text"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fixed = pd.read_csv(splits / f"fixed_split_{suffix}.csv")            # ID, label, partition
    cv = pd.read_csv(splits / f"cv_splits_{suffix}.csv")                 # split_id, ID, role
    ann = pd.read_csv(resolve_path(pcfg["annotated_csv"]))
    ann = ann.merge(fixed[["ID"]], on="ID", how="inner")

    if args.text == "pause":
        failed = ann.loc[ann["alignment_ok"] == 0, "ID"].tolist()
        if failed:
            if pcfg["on_alignment_failure"] == "error":
                sys.exit(f"{len(failed)} recordings have no usable pause annotation, e.g. {failed[:5]}.\n"
                         "Fix the alignment, or set stage1.pause_bert.on_alignment_failure to 'exclude' "
                         "knowing that the evaluated cohort then differs from the other models'.")
            print(f"Excluding {len(failed)} recordings with a failed alignment (of {len(ann)}).")
            ann = ann[ann["alignment_ok"] == 1]
    ann = ann.dropna(subset=[text_col])

    demo = pd.read_csv(resolve_path(cfg["paths"]["cohort"]), usecols=[cfg["cohort"]["id_column"], "age", "gender"])
    demo = demo.rename(columns={cfg["cohort"]["id_column"]: "ID"})
    demo["sex_code"] = demo["gender"].map(sex_code)
    data = ann[["ID", text_col]].merge(demo[["ID", "age", "sex_code"]], on="ID").merge(fixed[["ID", "label"]], on="ID")

    if args.dry_run:
        emb = np.random.default_rng(0).normal(size=(len(data), 768))
        print("DRY RUN: random embeddings. The numbers below mean nothing.")
    else:
        cache = resolve_path(pcfg["embeddings_dir"]) / f"{name}_{pcfg['embedding']}.npy"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            emb = np.load(cache)
        else:
            emb = bert_embeddings(data[text_col].tolist(), pcfg, device)
            np.save(cache, emb)
        if len(emb) != len(data):
            sys.exit(f"cached embeddings have {len(emb)} rows, the data has {len(data)}. Delete {cache}.")
    dims = [f"dim_{i}" for i in range(emb.shape[1])]
    data = pd.concat([data.reset_index(drop=True), pd.DataFrame(emb, columns=dims)], axis=1)
    feature_cols = dims + ["age", "sex_code"]
    print(f"{name} | task {args.task} | classes {classes} | n = {len(data)}")

    seed, max_epochs = pcfg["seed"], (2 if args.dry_run else pcfg["max_epochs"])
    split_ids = sorted(cv["split_id"].unique())[: (2 if args.dry_run else None)]
    rows = []
    for sid in split_ids:
        tr = data[data["ID"].isin(cv[(cv["split_id"] == sid) & (cv["role"] == "train")]["ID"])].reset_index(drop=True)
        va = data[data["ID"].isin(cv[(cv["split_id"] == sid) & (cv["role"] == "val")]["ID"])].reset_index(drop=True)
        if tr["label"].nunique() < len(classes):
            print(f"  [!] split {sid}: a class is missing from the training part, skipped")
            continue
        proba = train_and_predict(tr, va, feature_cols, classes, pcfg, seed + int(sid), device, max_epochs)
        rows += prediction_rows(va["ID"].tolist(), va["label"].tolist(), proba, classes, int(sid), "val")
        print(f"  split {sid} done")

    pool = fixed[fixed["partition"].isin(["train", "dev"])]["ID"]
    test = fixed[fixed["partition"] == "test"]["ID"]
    tr = data[data["ID"].isin(pool)].reset_index(drop=True)
    te = data[data["ID"].isin(test)].reset_index(drop=True)
    print(f"Final run: train+dev n = {len(tr)}, test n = {len(te)} (predicted once)")
    proba = train_and_predict(tr, te, feature_cols, classes, pcfg, seed + 10_000, device, max_epochs)
    rows += prediction_rows(te["ID"].tolist(), te["label"].tolist(), proba, classes, None, "test")

    out_dir = resolve_path(s1["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}{'_dryrun' if args.dry_run else ''}_{args.task}_predictions.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} prediction rows -> {out}")


if __name__ == "__main__":
    main()
