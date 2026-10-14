"""
Step 5 - Model training, cross-validation and final test-set evaluation.

Derived from notebook cells 16 and 20 (features.ipynb):
  cell 16 = "Step 4  - Baseline Models with 5-Fold Cross-Validation"
  cell 20 = "Step 5  - Final Model: Train on Full Train Set, Evaluate Once on Test"

Experimental variants that were NOT part of the reported results are archived:
  archive/dead_code/05_train_evaluate/threshold_tuning.py            (cell 17)
  archive/dead_code/05_train_evaluate/nested_cv_hyperparam_xgb.py    (cell 18)
  archive/dead_code/05_train_evaluate/ensembles.py                   (cell 19)

INPUTS
    <paths.processed_dir>/train_processed.csv     from build_matrix_recording.py
    <paths.processed_dir>/test_processed.csv      from build_matrix_recording.py
    <paths.processed_dir>/selected_features.csv   from feature_selection_recording.py
    config.yaml    seed, fold count, model hyperparameters, final model choice

OUTPUTS
    <paths.processed_dir>/final_model_svm_rbf.pkl   fitted final estimator
    <paths.processed_dir>/final_threshold.pkl       decision threshold
    cross-validation and test metrics printed to stdout

PROCESSING
    Part A  5-fold stratified CV over Logistic Regression, SVM (linear), SVM
            (rbf) and Random Forest on the selected features; reports mean AUC,
            sensitivity, specificity and balanced accuracy at threshold 0.5.
    Part B  The final model (train.final_model) is refitted on the ENTIRE train
            set. Its threshold comes from out-of-fold predictions on the train
            set only, and is then applied as a FIXED threshold to the test set.
            This is the only point where the test set is used - for final
            reporting, not for model or threshold selection.

HOW TO RUN
    python Step6_Modeling/Step6c_Train_Evaluate/train_evaluate_recording.py
    python Step6_Modeling/Step6c_Train_Evaluate/train_evaluate_recording.py --config /path/to/config.yaml

CAVEAT
    The folds are stratified by label only, not grouped by participant, and the
    feature list was selected outside the CV loop (step 04). See
    docs/open_questions.md (Q1, Q2) - not changed here.
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import DEFAULT_CONFIG, load_config, resolve_path


MODEL_LABELS = {
    'logistic_regression': 'Logistic Regression',
    'svm_linear': 'SVM (linear)',
    'svm_rbf': 'SVM (rbf)',
    'random_forest': 'Random Forest',
}


def build_model(key, cfg, seed):
    """Instantiates one classifier from its config.yaml entry."""
    p = cfg['models'][key]
    if key == 'logistic_regression':
        return LogisticRegression(class_weight=p['class_weight'],
                                  max_iter=p['max_iter'], random_state=seed)
    if key == 'svm_linear':
        return SVC(kernel='linear', probability=p['probability'],
                   class_weight=p['class_weight'], random_state=seed)
    if key == 'svm_rbf':
        return SVC(kernel='rbf', probability=p['probability'],
                   class_weight=p['class_weight'], random_state=seed)
    if key == 'random_forest':
        return RandomForestClassifier(n_estimators=p['n_estimators'],
                                      class_weight=p['class_weight'], random_state=seed)
    raise ValueError(f"Unknown model key: {key}")


def build_all_models(cfg, seed):
    """Returns the display-name -> estimator mapping used by the CV comparison."""
    return {MODEL_LABELS[k]: build_model(k, cfg, seed) for k in cfg['models']}


def metrics_at_threshold(y_true, proba, thresh):
    """Sensitivity (AD recall), specificity (CN recall) and balanced accuracy."""
    pred = (proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    bal = balanced_accuracy_score(y_true, pred)
    return sens, spec, bal


def cross_validate_models(X, y, models, skf, threshold):
    """Per-fold AUC / sensitivity / specificity / balanced accuracy for each model."""
    results = []
    for name, model in models.items():
        aucs, sens, spec, bal_accs = [], [], [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_val)[:, 1]

            aucs.append(roc_auc_score(y_val, proba))
            s, sp, bal = metrics_at_threshold(y_val, proba, threshold)
            sens.append(s)
            spec.append(sp)
            bal_accs.append(bal)

        results.append({
            'model': name,
            'AUC': np.mean(aucs),
            'AUC_std': np.std(aucs),
            'Sensitivity': np.mean(sens),
            'Specificity': np.mean(spec),
            'Balanced Acc': np.mean(bal_accs),
        })
        print(f"  Done: {name}")

    return pd.DataFrame(results).sort_values('AUC', ascending=False)


def print_cv_table(res_df):
    """Renders the baseline cross-validation comparison table."""
    print(f"\n{'Model':<22}{'AUC':>8}{'±std':>7}{'Sens':>8}{'Spec':>8}{'BalAcc':>9}")
    print("=" * 62)
    for _, r in res_df.iterrows():
        print(f"{r['model']:<22}{r['AUC']:>8.3f}{r['AUC_std']:>7.3f}"
              f"{r['Sensitivity']:>8.3f}{r['Specificity']:>8.3f}{r['Balanced Acc']:>9.3f}")


def find_threshold_via_oof(X_train, y_train, model_key, cfg, skf, seed):
    """Youden's J threshold from out-of-fold train predictions; test untouched."""
    oof_proba = np.zeros(len(y_train))
    for train_idx, val_idx in skf.split(X_train, y_train):
        model = build_model(model_key, cfg, seed)
        model.fit(X_train[train_idx], y_train[train_idx])
        oof_proba[val_idx] = model.predict_proba(X_train[val_idx])[:, 1]

    fpr, tpr, thresholds = roc_curve(y_train, oof_proba)
    threshold = thresholds[np.argmax(tpr - fpr)]
    return threshold, roc_auc_score(y_train, oof_proba)


def main(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    cfg = config['train']
    seed = config['seed']
    processed_dir = resolve_path(config['paths']['processed_dir'])

    train_df = pd.read_csv(os.path.join(processed_dir, 'train_processed.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test_processed.csv'))
    features = pd.read_csv(os.path.join(processed_dir, 'selected_features.csv'))['feature'].tolist()

    X_train, y_train = train_df[features].values, train_df['label'].values
    X_test, y_test = test_df[features].values, test_df['label'].values

    skf = StratifiedKFold(n_splits=cfg['n_folds'], shuffle=cfg['shuffle'], random_state=seed)

    # -- Part A: baseline cross-validation ----------------------------------
    print(f"Samples: {len(y_train)}   Features: {len(features)}   "
          f"CN={sum(y_train == 0)}  AD={sum(y_train == 1)}\n")

    models = build_all_models(cfg, seed)
    cv_results = cross_validate_models(X_train, y_train, models, skf, cfg['default_threshold'])
    print_cv_table(cv_results)

    # -- Part B: final model, single test-set evaluation ---------------------
    print(f"\n\nTrain: {len(y_train)}  (CN={sum(y_train == 0)}, AD={sum(y_train == 1)})")
    print(f"Test:  {len(y_test)}  (CN={sum(y_test == 0)}, AD={sum(y_test == 1)})\n")

    model_key = cfg['final_model']
    print("Finding threshold via cross-validation on train set...")
    threshold, train_auc = find_threshold_via_oof(X_train, y_train, model_key, cfg, skf, seed)
    print(f"  Train CV AUC:     {train_auc:.3f}")
    print(f"  Chosen threshold: {threshold:.3f}  (from train CV only, test untouched)\n")

    print("Training final model on full train set...")
    final_model = build_model(model_key, cfg, seed)
    final_model.fit(X_train, y_train)

    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    sens, spec, bal = metrics_at_threshold(y_test, test_proba, threshold)

    print(f"\n{'='*50}")
    print("FINAL TEST SET RESULTS (evaluated once)")
    print(f"{'='*50}")
    print(f"  AUC:                {test_auc:.3f}")
    print(f"  Threshold used:     {threshold:.3f}")
    print(f"  Sensitivity:        {sens:.3f}")
    print(f"  Specificity:        {spec:.3f}")
    print(f"  Balanced Accuracy:  {bal:.3f}")

    model_path = os.path.join(processed_dir, f'final_model_{model_key}.pkl')
    threshold_path = os.path.join(processed_dir, 'final_threshold.pkl')
    joblib.dump(final_model, model_path)
    joblib.dump(threshold, threshold_path)
    print(f"\nSaved -> {model_path}")
    print(f"Saved -> {threshold_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
