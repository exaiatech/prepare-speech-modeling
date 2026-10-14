"""
Segment step 5 - Grouped cross-validation and final recording-level evaluation.

Derived from notebook cells F5 and F6 of features2.ipynb:
  F5 = "Step 4 (segments) - Baseline Models with Grouped Cross-Validation"
  F6 = "Step 5 (segments) - Final Model, Evaluate Once on Test (Person-Level)"

This is the script that produces the poster's best result:
    test AUC 0.682, threshold 0.338, sensitivity 0.614, specificity 0.668,
    balanced accuracy 0.641, over 298 test recordings.

INPUTS
    <segments.paths.processed_dir>/train_segments_processed.csv
    <segments.paths.processed_dir>/test_segments_processed.csv
    <segments.paths.processed_dir>/selected_features_segments.csv
    config.yaml   seed, segments.train.*

OUTPUTS
    <segments.paths.processed_dir>/final_model_random_forest_segments.pkl
    <segments.paths.processed_dir>/final_threshold_segments.pkl
    cross-validation and test metrics printed to stdout

PROCESSING
    Part A  StratifiedGroupKFold(groups=uid) over Logistic Regression, SVM
            (linear), SVM (rbf) and Random Forest. Every segment gets exactly
            one out-of-fold probability, produced by a model that never saw
            any segment of that recording. Segment probabilities are then
            averaged per uid, and AUC / Youden threshold / sensitivity /
            specificity / balanced accuracy are reported at that aggregated
            level, so the numbers are comparable to the whole-recording path.
    Part B  The final model (segments.train.final_model) is refitted on ALL
            train segments. Its threshold comes from the grouped out-of-fold
            predictions of the train set only, aggregated to recording level,
            and is then applied as a FIXED threshold to the test set, whose
            segment probabilities are likewise averaged per uid.

GROUPING CAVEAT
    The grouping key is `uid`, which identifies a RECORDING, not a participant.
    All segments of one recording are guaranteed to stay in the same fold. If a
    participant contributed more than one recording, their recordings can still
    be split across folds - metadata.csv contains no participant identifier, so
    this cannot be verified. The notebook calls this level "person"; this
    script calls it "recording". See docs/open_questions.md (Q16) and
    docs/cv_strategy.md.

HOW TO RUN
    python Step6_Modeling/Step6c_Train_Evaluate/train_evaluate_segment.py
    python Step6_Modeling/Step6c_Train_Evaluate/train_evaluate_segment.py --config /path/to/config.yaml
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
from sklearn.model_selection import StratifiedGroupKFold
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


def metrics_at_threshold(y_true, proba, thresh):
    """Sensitivity (AD recall), specificity (CN recall) and balanced accuracy."""
    pred = (proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    bal = balanced_accuracy_score(y_true, pred)
    return sens, spec, bal


def to_recording_level(uids, y, proba):
    """Averages segment probabilities within each recording (uid)."""
    agg = pd.DataFrame({'uid': uids, 'y': y, 'proba': proba})
    return agg.groupby('uid').agg(y=('y', 'first'), proba=('proba', 'mean')).reset_index()


def grouped_oof_probabilities(X, y, groups, model, sgkf):
    """One out-of-fold probability per segment, folds grouped by recording."""
    oof_proba = np.zeros(len(y))
    for train_idx, val_idx in sgkf.split(X, y, groups=groups):
        model.fit(X[train_idx], y[train_idx])
        oof_proba[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    return oof_proba


def youden_threshold(y_true, proba):
    """Threshold maximising Youden's J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    return thresholds[np.argmax(tpr - fpr)]


def cross_validate_models(X, y, groups, cfg, seed, sgkf):
    """Grouped CV for every configured model, scored at recording level."""
    results = []
    for key in cfg['models']:
        name = MODEL_LABELS[key]
        print(f"Running {name} ...")
        model = build_model(key, cfg, seed)
        oof_proba = grouped_oof_probabilities(X, y, groups, model, sgkf)

        # segment-level AUC, just for reference
        segment_auc = roc_auc_score(y, oof_proba)

        recording = to_recording_level(groups, y, oof_proba)
        recording_auc = roc_auc_score(recording['y'], recording['proba'])
        best_thresh = youden_threshold(recording['y'], recording['proba'])
        sens, spec, bal = metrics_at_threshold(recording['y'], recording['proba'], best_thresh)

        results.append({
            'model': name, 'segment_AUC': segment_auc, 'recording_AUC': recording_auc,
            'best_thresh': best_thresh, 'sens_opt': sens, 'spec_opt': spec, 'bal_opt': bal,
        })
        print(f"  segment AUC={segment_auc:.3f}   recording AUC={recording_auc:.3f}   "
              f"recording BalAcc={bal:.3f}\n")

    return pd.DataFrame(results).sort_values('bal_opt', ascending=False)


def print_cv_table(res_df):
    """Renders the grouped cross-validation comparison table."""
    print(f"\n{'Model':<22}{'SegAUC':>8}{'RecAUC':>9}{'Thr':>7}"
          f"{'SensOpt':>9}{'SpecOpt':>9}{'BalOpt':>9}")
    print("=" * 76)
    for _, r in res_df.iterrows():
        print(f"{r['model']:<22}{r['segment_AUC']:>8.3f}{r['recording_AUC']:>9.3f}"
              f"{r['best_thresh']:>7.3f}{r['sens_opt']:>9.3f}{r['spec_opt']:>9.3f}"
              f"{r['bal_opt']:>9.3f}")


def main(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    cfg = config['segments']['train']
    seed = config['seed']
    seg_col = config['segments']['matrix']['segment_column']
    processed_dir = resolve_path(config['segments']['paths']['processed_dir'])

    train_df = pd.read_csv(os.path.join(processed_dir, 'train_segments_processed.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test_segments_processed.csv'))
    features = pd.read_csv(
        os.path.join(processed_dir, 'selected_features_segments.csv'))['feature'].tolist()

    X_train = train_df[features].values
    y_train = train_df['label'].values
    groups_train = train_df['uid'].values

    X_test = test_df[features].values
    y_test = test_df['label'].values
    groups_test = test_df['uid'].values

    sgkf = StratifiedGroupKFold(n_splits=cfg['n_folds'], shuffle=cfg['shuffle'],
                                random_state=seed)

    # -- Part A: grouped cross-validation ------------------------------------
    print(f"Segments: {len(y_train)}   Recordings: {train_df['uid'].nunique()}   "
          f"Features: {len(features)}")
    print(f"Segment labels - CN={sum(y_train == 0)}, AD={sum(y_train == 1)}\n")

    cv_results = cross_validate_models(X_train, y_train, groups_train, cfg, seed, sgkf)
    print_cv_table(cv_results)

    # -- Part B: final model, single test-set evaluation ---------------------
    print(f"\n\nTrain: {len(y_train)} segments from {train_df['uid'].nunique()} recordings")
    print(f"Test:  {len(y_test)} segments from {test_df['uid'].nunique()} recordings\n")

    model_key = cfg['final_model']
    print("Finding threshold via grouped cross-validation on train segments...")
    oof_proba = grouped_oof_probabilities(X_train, y_train, groups_train,
                                          build_model(model_key, cfg, seed), sgkf)
    train_recording = to_recording_level(groups_train, y_train, oof_proba)
    train_auc = roc_auc_score(train_recording['y'], train_recording['proba'])
    threshold = youden_threshold(train_recording['y'], train_recording['proba'])
    print(f"  Train CV recording-level AUC: {train_auc:.3f}")
    print(f"  Chosen threshold:             {threshold:.3f}  "
          f"(from train CV only, test untouched)\n")

    print("Training final model on full train segments...")
    final_model = build_model(model_key, cfg, seed)
    final_model.fit(X_train, y_train)

    test_seg_proba = final_model.predict_proba(X_test)[:, 1]
    test_recording = to_recording_level(groups_test, y_test, test_seg_proba)
    test_auc = roc_auc_score(test_recording['y'], test_recording['proba'])
    sens, spec, bal = metrics_at_threshold(test_recording['y'], test_recording['proba'],
                                           threshold)

    print(f"\n{'='*55}")
    print("FINAL TEST SET RESULTS - recording level (evaluated once)")
    print(f"{'='*55}")
    print(f"  Recordings in test set: {len(test_recording)}")
    print(f"  AUC:                    {test_auc:.3f}")
    print(f"  Threshold used:         {threshold:.3f}")
    print(f"  Sensitivity:            {sens:.3f}")
    print(f"  Specificity:            {spec:.3f}")
    print(f"  Balanced Accuracy:      {bal:.3f}")

    model_path = os.path.join(processed_dir, f'final_model_{model_key}_segments.pkl')
    threshold_path = os.path.join(processed_dir, 'final_threshold_segments.pkl')
    joblib.dump(final_model, model_path)
    joblib.dump(threshold, threshold_path)
    print(f"\nSaved -> {model_path}")
    print(f"Saved -> {threshold_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
