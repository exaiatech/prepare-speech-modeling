"""
Segment step 6 - Interpretability: TreeExplainer SHAP + recording-level
permutation importance.

Derived from notebook cell F7 of features2.ipynb ("Step 7 (segments)").

Two deliberate differences from the whole-recording interpret_recording.py:
  * The final model is a Random Forest, so shap.TreeExplainer can be used -
    exact and fast, no sampling-based KernelExplainer needed.
  * Permutation importance is computed with a custom loop instead of sklearn's
    permutation_importance, because the evaluation metric is RECORDING-level
    AUC (segment probabilities averaged per uid). A feature's importance is
    measured by how much recording-level AUC drops when it is shuffled, not
    segment-level AUC, which could give a misleading ranking.

INPUTS
    <segments.paths.processed_dir>/test_segments_processed.csv
    <segments.paths.processed_dir>/selected_features_segments.csv
    <segments.paths.processed_dir>/final_model_<key>_segments.pkl
    config.yaml   seed, segments.interpret.*

OUTPUTS
    <segments.paths.processed_dir>/feature_importance_segments.csv
    <segments.paths.processed_dir>/shap_importance_segments.png
    <segments.paths.processed_dir>/permutation_importance_segments.png

HOW TO RUN
    python Step6_Modeling/Step6d_Interpret/interpret_segment.py
    python Step6_Modeling/Step6d_Interpret/interpret_segment.py --config /path/to/config.yaml

CAVEAT
    Both analyses run on the test set, which Step 6c declares to be evaluated
    only once. Same issue as the whole-recording path. See
    docs/open_questions.md (Q18).
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import DEFAULT_CONFIG, load_config, resolve_path


def to_recording_level(uids, y, proba):
    """Averages segment probabilities within each recording (uid)."""
    agg = pd.DataFrame({'uid': uids, 'y': y, 'proba': proba})
    return agg.groupby('uid').agg(y=('y', 'first'), proba=('proba', 'mean')).reset_index()


def make_recording_auc(model, uids, y_test):
    """Returns a function mapping a feature matrix to recording-level AUC."""
    def recording_auc(X):
        proba = model.predict_proba(X)[:, 1]
        recording = to_recording_level(uids, y_test, proba)
        return roc_auc_score(recording['y'], recording['proba'])
    return recording_auc


def recording_level_permutation_importance(X_test, features, auc_fn, n_repeats, seed):
    """Shuffles one feature at a time and measures the recording-level AUC drop."""
    rng = np.random.RandomState(seed)
    baseline_auc = auc_fn(X_test)
    print(f"  Baseline recording-level AUC: {baseline_auc:.3f}")

    perm_results = []
    for i, feat in enumerate(features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, i] = X_perm[rng.permutation(len(X_perm)), i]
            drops.append(baseline_auc - auc_fn(X_perm))
        perm_results.append({'feature': feat,
                             'perm_importance_mean': np.mean(drops),
                             'perm_importance_std': np.std(drops)})
        print(f"  [{i+1}/{len(features)}] {feat}: {np.mean(drops):.4f}")

    return pd.DataFrame(perm_results).sort_values('perm_importance_mean', ascending=False)


def tree_shap_importance(model, X_test, features):
    """Mean |SHAP value| per feature via the exact TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        sv = shap_values[1]            # class 1 = AD (older shap versions)
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]      # (n_samples, n_features, n_classes)
    else:
        sv = shap_values

    return pd.DataFrame({
        'feature': features,
        'mean_abs_shap': np.abs(sv).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)


def dark_barplot(df, value_col, title, out_path, plot_cfg, top_n=25):
    """Horizontal bar plot in the notebook's dark figure style."""
    top = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(plot_cfg['figure_facecolor'])
    ax.set_facecolor(plot_cfg['axes_facecolor'])
    ax.barh(top['feature'], top[value_col], color=plot_cfg['bar_color'])
    ax.set_title(title, color='white', fontsize=12)
    ax.tick_params(colors=plot_cfg['tick_color'], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(plot_cfg['spine_color'])
    plt.tight_layout()
    plt.savefig(out_path, dpi=plot_cfg['dpi'], facecolor=plot_cfg['figure_facecolor'])
    plt.close()


def main(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    cfg = config['segments']['interpret']
    plot_cfg = config['plots']
    seed = config['seed']
    processed_dir = resolve_path(config['segments']['paths']['processed_dir'])
    model_key = config['segments']['train']['final_model']

    test_df = pd.read_csv(os.path.join(processed_dir, 'test_segments_processed.csv'))
    features = pd.read_csv(
        os.path.join(processed_dir, 'selected_features_segments.csv'))['feature'].tolist()
    model = joblib.load(os.path.join(processed_dir,
                                     f'final_model_{model_key}_segments.pkl'))

    X_test = test_df[features].values
    y_test = test_df['label'].values
    uids = test_df['uid'].values

    print("Computing recording-level permutation importance...")
    auc_fn = make_recording_auc(model, uids, y_test)
    perm_df = recording_level_permutation_importance(
        X_test, features, auc_fn, cfg['permutation_repeats'], seed)

    print("\nComputing SHAP values...")
    shap_df = tree_shap_importance(model, X_test, features)

    combined = perm_df.merge(shap_df, on='feature').sort_values('mean_abs_shap',
                                                                ascending=False)

    print(f"\n{'feature':<28}{'SHAP':>10}{'PermImp':>10}")
    print("=" * 48)
    for _, r in combined.iterrows():
        print(f"{r['feature']:<28}{r['mean_abs_shap']:>10.4f}{r['perm_importance_mean']:>10.4f}")

    importance_path = os.path.join(processed_dir, 'feature_importance_segments.csv')
    combined.to_csv(importance_path, index=False)

    shap_png = os.path.join(processed_dir, 'shap_importance_segments.png')
    perm_png = os.path.join(processed_dir, 'permutation_importance_segments.png')
    dark_barplot(shap_df, 'mean_abs_shap',
                 'Top Features - Mean |SHAP value| (segments)',
                 shap_png, plot_cfg, cfg['top_n_plot'])
    dark_barplot(perm_df, 'perm_importance_mean',
                 'Top Features - Recording-Level Permutation Importance',
                 perm_png, plot_cfg, cfg['top_n_plot'])

    print(f"\nSaved -> {importance_path}")
    print(f"Saved -> {shap_png}")
    print(f"Saved -> {perm_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
