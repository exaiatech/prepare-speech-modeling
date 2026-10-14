# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 18 (0-based index in the original notebook).
# What it does: Step 4c - nested CV (outer 5 / inner 3 GridSearchCV) plus XGBoost.
# Why archived: Experiment. Its own successor (cell 19) states that tuning gave no real improvement and reverts to default hyperparameters.
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

"""
Step 4c — Hyperparameter Tuning + XGBoost (Nested Cross-Validation)
------------------------------------------------------------------------
pip install pandas numpy scikit-learn xgboost

Nested CV: an INNER GridSearchCV (3-fold) picks the best hyperparameters
within each OUTER fold (5-fold), and the resulting best estimator is
evaluated on the held-out outer fold. This avoids the optimistic bias
of picking hyperparameters and evaluating on the same folds. Out-of-fold
probabilities are then used to find the optimal threshold (Youden's J),
same approach as Step 4b.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, roc_curve
from xgboost import XGBClassifier

# ── Paths ───────────────────────────────────────────────
TRAIN_CSV    = r"C:\Users\arday\Desktop\full_features\processed\train_processed.csv"
FEATURES_CSV = r"C:\Users\arday\Desktop\full_features\processed\selected_features.csv"
N_OUTER      = 5
N_INNER      = 3
RANDOM_STATE = 42
# ────────────────────────────────────────────────────────

train_df = pd.read_csv(TRAIN_CSV)
features = pd.read_csv(FEATURES_CSV)['feature'].tolist()

X = train_df[features].values
y = train_df['label'].values

pos_weight = (y == 0).sum() / (y == 1).sum()  # for XGBoost class imbalance
print(f"Samples: {len(y)}  Features: {len(features)}  CN={sum(y==0)}  AD={sum(y==1)}  pos_weight={pos_weight:.2f}\n")

model_grids = {
    'Logistic Regression': (
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
        {'C': [0.01, 0.1, 1, 10]}
    ),
    'SVM (linear)': (
        SVC(kernel='linear', probability=True, class_weight='balanced', random_state=RANDOM_STATE),
        {'C': [0.01, 0.1, 1, 10]}
    ),
    'SVM (rbf)': (
        SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE),
        {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1]}
    ),
    'Random Forest': (
        RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
        {'n_estimators': [200, 400], 'max_depth': [None, 5, 10], 'min_samples_leaf': [1, 3, 5]}
    ),
    'XGBoost': (
        XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE),
        {'n_estimators': [200, 400], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1],
         'scale_pos_weight': [1, pos_weight]}
    ),
}


def metrics_at_threshold(y_true, proba, thresh):
    pred = (proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    bal  = balanced_accuracy_score(y_true, pred)
    return sens, spec, bal


outer_skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)

results = []
for name, (base_model, grid) in model_grids.items():
    print(f"Tuning {name} ...")
    oof_proba = np.zeros(len(y))

    for fold_i, (train_idx, val_idx) in enumerate(outer_skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        inner_skf = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)
        search = GridSearchCV(base_model, grid, scoring='roc_auc', cv=inner_skf, n_jobs=-1)
        search.fit(X_tr, y_tr)

        oof_proba[val_idx] = search.best_estimator_.predict_proba(X_val)[:, 1]
        print(f"  Fold {fold_i}: best params = {search.best_params_}")

    auc = roc_auc_score(y, oof_proba)
    fpr, tpr, thresholds = roc_curve(y, oof_proba)
    best_thresh = thresholds[np.argmax(tpr - fpr)]

    sens_05, spec_05, bal_05 = metrics_at_threshold(y, oof_proba, 0.5)
    sens_opt, spec_opt, bal_opt = metrics_at_threshold(y, oof_proba, best_thresh)

    results.append({
        'model': name, 'AUC': auc, 'best_thresh': best_thresh,
        'sens_0.5': sens_05, 'spec_0.5': spec_05, 'bal_0.5': bal_05,
        'sens_opt': sens_opt, 'spec_opt': spec_opt, 'bal_opt': bal_opt,
    })
    print(f"  → AUC={auc:.3f}  BalAcc(opt)={bal_opt:.3f}\n")

res_df = pd.DataFrame(results).sort_values('bal_opt', ascending=False)

print(f"\n{'Model':<22}{'AUC':>7}{'Thr':>7} | {'Sens.5':>8}{'Spec.5':>8}{'Bal.5':>8} | {'SensOpt':>8}{'SpecOpt':>8}{'BalOpt':>8}")
print("=" * 96)
for _, r in res_df.iterrows():
    print(f"{r['model']:<22}{r['AUC']:>7.3f}{r['best_thresh']:>7.3f} | "
          f"{r['sens_0.5']:>8.3f}{r['spec_0.5']:>8.3f}{r['bal_0.5']:>8.3f} | "
          f"{r['sens_opt']:>8.3f}{r['spec_opt']:>8.3f}{r['bal_opt']:>8.3f}")

out_path = os.path.join(os.path.dirname(TRAIN_CSV), 'tuned_model_results.csv')
res_df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
