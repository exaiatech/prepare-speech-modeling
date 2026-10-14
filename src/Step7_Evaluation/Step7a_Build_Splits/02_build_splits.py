"""
Stage 1, step 2: freeze the data splits.

Produces, from cookie_theft_inclusion_list.csv:
  1. The fixed `set` partition (train/dev/test), for both tasks.
  2. Repeated 5-fold CV assignments (5 splits x 10 repeats = 50), for both
     tasks, built only on the train+dev pool (test is never used to fit or
     select anything).

These files are written once and then used by every model (encoder,
interpretable, hybrid) so predictions are comparable across models. Do not
regenerate after models have started using them; the random_state is fixed
so the assignment is reproducible, but a fresh run should not be needed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import load_config, resolve_path
_cfg = load_config()   # config.yaml, merged with config.local.yaml when present
INCLUSION_LIST = str(resolve_path(_cfg["stage1"]["inclusion_list"]))
OUT_DIR        = str(resolve_path(_cfg["stage1"]["splits_dir"]))
RANDOM_STATE   = 42
N_SPLITS       = 5
N_REPEATS      = 10

import pandas as pd
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold

out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(INCLUSION_LIST)
df = df[df["audio_matches"] == 1].reset_index(drop=True)
# pandas reads an empty CSV cell back as NaN, not "" - normalise before filtering
df[["task_A_binary", "task_B_multiclass"]] = df[["task_A_binary", "task_B_multiclass"]].fillna("")

# ── 1. Fixed `set` partition ───────────────────────────────────────────
fixed_A = df[df["task_A_binary"] != ""][["ID", "task_A_binary", "partition"]].rename(
    columns={"task_A_binary": "label"})
fixed_B = df[["ID", "task_B_multiclass", "partition"]].rename(
    columns={"task_B_multiclass": "label"})

fixed_A.to_csv(out / "fixed_split_taskA_binary.csv", index=False)
fixed_B.to_csv(out / "fixed_split_taskB_multiclass.csv", index=False)

print("=== Fixed `set` partition ===")
for name, d in [("Task A (binary)", fixed_A), ("Task B (3-class)", fixed_B)]:
    print(f"\n{name}")
    print(pd.crosstab(d["label"], d["partition"], margins=True).to_string())

# ── 2. Repeated 5-fold CV, train+dev pool only ─────────────────────────
# Test partition is fully excluded here: it is reserved for the one-time
# final evaluation of each finished model, never for fitting or fold-based
# comparison.
def build_cv(task_df, label_col, task_name):
    pool = task_df[task_df["partition"].isin(["train", "dev"])].reset_index(drop=True)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                    random_state=RANDOM_STATE)
    rows = []
    for split_id, (tr_idx, val_idx) in enumerate(rskf.split(pool, pool[label_col])):
        for i in tr_idx:
            rows.append({"split_id": split_id, "ID": pool.loc[i, "ID"], "role": "train"})
        for i in val_idx:
            rows.append({"split_id": split_id, "ID": pool.loc[i, "ID"], "role": "val"})
    cv = pd.DataFrame(rows)
    cv.to_csv(out / f"cv_splits_{task_name}.csv", index=False)

    print(f"\n=== Repeated CV pool: {task_name} ===")
    print(f"Pool size (train+dev): {len(pool)}  |  {N_SPLITS} folds x {N_REPEATS} repeats "
          f"= {N_SPLITS * N_REPEATS} splits")
    print(f"Label distribution in pool:\n{pool[label_col].value_counts().to_string()}")
    return cv

cv_A = build_cv(fixed_A, "label", "taskA_binary")
cv_B = build_cv(fixed_B, "label", "taskB_multiclass")

print(f"\nAll split files written to {out}")
print("Files:")
for f in sorted(out.glob("*.csv")):
    print(" ", f.name)
