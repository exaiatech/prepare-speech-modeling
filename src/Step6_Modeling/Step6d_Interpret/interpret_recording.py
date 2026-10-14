"""
Step 6 - Interpretability: SHAP and permutation importance.

Derived from notebook cell 21 (features.ipynb), "Step 7 - Interpretability".
The target structure had no slot for it, so it is numbered 06 after training.

INPUTS
    <paths.processed_dir>/train_processed.csv       from build_matrix_recording.py
    <paths.processed_dir>/test_processed.csv        from build_matrix_recording.py
    <paths.processed_dir>/selected_features.csv     from feature_selection_recording.py
    <paths.processed_dir>/final_model_<key>.pkl     from train_evaluate_recording.py
    config.yaml    SHAP sample count, background size, permutation repeats

OUTPUTS
    <paths.processed_dir>/feature_importance.csv        merged SHAP + permutation
    <paths.processed_dir>/shap_importance.png
    <paths.processed_dir>/permutation_importance.png

PROCESSING
    Explains which of the selected features actually drive the model's
    predictions, using two independent methods that should broadly agree
    if the findings are trustworthy:

      1. Permutation importance - shuffle one feature at a time on the
         TEST set and measure how much AUC drops. Model-agnostic, exact.
      2. SHAP (KernelExplainer) - attributes each prediction's probability
         to individual feature contributions. Works for any model,
         including non-linear ones like SVM (rbf), unlike TreeExplainer
         or LinearExplainer which only work for specific model types.

HOW TO RUN
    python Step6_Modeling/Step6d_Interpret/interpret_recording.py
    python Step6_Modeling/Step6d_Interpret/interpret_recording.py --config /path/to/config.yaml

CAVEAT
    Permutation importance is computed on the test set, which Step 6c declares
    to be evaluated only once. See docs/open_questions.md (Q3) - not changed here.
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
from sklearn.inspection import permutation_importance  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import DEFAULT_CONFIG, load_config, resolve_path


def compute_permutation_importance(model, X_test, y_test, features, cfg, seed):
    """Permutation importance on the test set, scored by AUC drop."""
    print("Computing permutation importance...")
    perm = permutation_importance(
        model, X_test, y_test,
        scoring=cfg['permutation_scoring'], n_repeats=cfg['permutation_repeats'],
        random_state=seed, n_jobs=-1,
    )
    return pd.DataFrame({
        'feature': features,
        'perm_importance_mean': perm.importances_mean,
        'perm_importance_std': perm.importances_std,
    }).sort_values('perm_importance_mean', ascending=False)


def compute_shap_importance(model, X_train, X_test, features, cfg, seed):
    """Mean |SHAP value| per feature via the model-agnostic KernelExplainer."""
    print("Computing SHAP values (this can take a few minutes)...")
    background = shap.kmeans(X_train, cfg['kmeans_background'])  # compact reference set
    rng = np.random.RandomState(seed)
    explain_idx = rng.choice(len(X_test),
                             size=min(cfg['n_shap_samples'], len(X_test)), replace=False)
    X_explain = X_test[explain_idx]

    def predict_ad_proba(x):
        return model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(predict_ad_proba, background)
    shap_values = explainer.shap_values(X_explain, nsamples='auto')

    return pd.DataFrame({
        'feature': features,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)


def dark_barplot(df, value_col, title, out_path, plot_cfg, top_n=20):
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
    cfg = config['interpret']
    plot_cfg = config['plots']
    seed = config['seed']
    processed_dir = resolve_path(config['paths']['processed_dir'])
    model_key = config['train']['final_model']

    train_df = pd.read_csv(os.path.join(processed_dir, 'train_processed.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test_processed.csv'))
    features = pd.read_csv(os.path.join(processed_dir, 'selected_features.csv'))['feature'].tolist()
    model = joblib.load(os.path.join(processed_dir, f'final_model_{model_key}.pkl'))

    X_train = train_df[features].values
    X_test = test_df[features].values
    y_test = test_df['label'].values

    perm_df = compute_permutation_importance(model, X_test, y_test, features, cfg, seed)
    shap_df = compute_shap_importance(model, X_train, X_test, features, cfg, seed)

    combined = perm_df.merge(shap_df, on='feature').sort_values('mean_abs_shap', ascending=False)

    print(f"\n{'feature':<28}{'SHAP':>10}{'PermImp':>10}")
    print("=" * 48)
    for _, r in combined.head(cfg['top_n_plot']).iterrows():
        print(f"{r['feature']:<28}{r['mean_abs_shap']:>10.4f}{r['perm_importance_mean']:>10.4f}")

    importance_path = os.path.join(processed_dir, 'feature_importance.csv')
    combined.to_csv(importance_path, index=False)

    shap_png = os.path.join(processed_dir, 'shap_importance.png')
    perm_png = os.path.join(processed_dir, 'permutation_importance.png')
    dark_barplot(shap_df, 'mean_abs_shap', 'Top Features - Mean |SHAP value|',
                 shap_png, plot_cfg, cfg['top_n_plot'])
    dark_barplot(perm_df, 'perm_importance_mean',
                 'Top Features - Permutation Importance (AUC drop)',
                 perm_png, plot_cfg, cfg['top_n_plot'])

    print(f"\nSaved -> {importance_path}")
    print(f"Saved -> {shap_png}")
    print(f"Saved -> {perm_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
