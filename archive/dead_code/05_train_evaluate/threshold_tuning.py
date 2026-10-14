# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 17 (0-based index in the original notebook).
# What it does: Step 4b - Youden's J threshold tuning on out-of-fold probabilities.
# Why archived: Exploratory. The Youden's J threshold selection itself survives inside 05_train_evaluate.py (final model); this standalone comparison does not.
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

"""
Step 4b — Threshold Tuning (Youden's J)
----------------------------------------------------------
pip install pandas numpy scikit-learn

AUC measures ranking ability across ALL thresholds, but sensitivity/
specificity only look at one cutoff (0.5 by default). This script
collects out-of-fold probabilities across 5-fold CV, then finds the
threshold that maximizes Youden's J = sensitivity + specificity - 1
instead of using 0.5. Reports both the default and tuned results
side by side.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, roc_curve

# ── Paths ───────────────────────────────────────────────
TRAIN_CSV    = r"C:\Users\arday\Desktop\full_features\processed\train_processed.csv"
FEATURES_CSV = r"C:\Users\arday\Desktop\full_features\processed\selected_features.csv"
N_FOLDS      = 5
RANDOM_STATE = 42
# ────────────────────────────────────────────────────────

train_df = pd.read_csv(TRAIN_CSV)
features = pd.read_csv(FEATURES_CSV)['feature'].tolist()

X = train_df[features].values
y = train_df['label'].values

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
    'SVM (linear)':        SVC(kernel='linear', probability=True, class_weight='balanced', random_state=RANDOM_STATE),
    'SVM (rbf)':           SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE),
    'Random Forest':       RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE),
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def metrics_at_threshold(y_true, proba, thresh):
    pred = (proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    bal  = balanced_accuracy_score(y_true, pred)
    return sens, spec, bal


results = []
for name, model in models.items():
    # Out-of-fold probabilities: each sample predicted exactly once,
    # always by a model that never saw it during training.
    oof_proba = np.zeros(len(y))
    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        oof_proba[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    auc = roc_auc_score(y, oof_proba)

    fpr, tpr, thresholds = roc_curve(y, oof_proba)
    best_idx = np.argmax(tpr - fpr)          # Youden's J = tpr - fpr
    best_thresh = thresholds[best_idx]

    sens_05, spec_05, bal_05 = metrics_at_threshold(y, oof_proba, 0.5)
    sens_opt, spec_opt, bal_opt = metrics_at_threshold(y, oof_proba, best_thresh)

    results.append({
        'model': name, 'AUC': auc, 'best_thresh': best_thresh,
        'sens_0.5': sens_05, 'spec_0.5': spec_05, 'bal_0.5': bal_05,
        'sens_opt': sens_opt, 'spec_opt': spec_opt, 'bal_opt': bal_opt,
    })
    print(f"  Done: {name}  (optimal threshold = {best_thresh:.3f})")

res_df = pd.DataFrame(results).sort_values('bal_opt', ascending=False)

print(f"\n{'Model':<22}{'AUC':>7}{'Thr':>7} | {'Sens.5':>8}{'Spec.5':>8}{'Bal.5':>8} | {'SensOpt':>8}{'SpecOpt':>8}{'BalOpt':>8}")
print("=" * 96)
for _, r in res_df.iterrows():
    print(f"{r['model']:<22}{r['AUC']:>7.3f}{r['best_thresh']:>7.3f} | "
          f"{r['sens_0.5']:>8.3f}{r['spec_0.5']:>8.3f}{r['bal_0.5']:>8.3f} | "
          f"{r['sens_opt']:>8.3f}{r['spec_opt']:>8.3f}{r['bal_opt']:>8.3f}")
