# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 10 (0-based index in the original notebook).
# What it does: 43-feature pilot extractor with Praat "To TextGrid (silences)" pause detection.
# Why archived: Superseded: this Praat-based pause detection is what produced the apparent pause effect that turned out to be background noise. Replaced by WebRTC VAD.
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

"""
43 Acoustic Biomarker Extractor + Visualization
------------------------------------------------
pip install praat-parselmouth librosa scipy pandas matplotlib

Usage: python extract_features.py
"""

import os, glob
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ── Paths ───────────────────────────────────────────────
CONTROL_DIR   = r"C:\Users\arday\Desktop\pilot_dataset_2\control"
ALZHEIMER_DIR = r"C:\Users\arday\Desktop\pilot_dataset_2\alzheimer"
OUTPUT_CSV    = r"C:\Users\arday\Desktop\pilot_dataset_2\features_40.csv"
PLOTS_DIR     = r"C:\Users\arday\Desktop\pilot_dataset_2\feature_plots"
# ────────────────────────────────────────────────────────

AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.m4a')


def stats6(arr):
    """mean, std, min, max, kurtosis, skewness"""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return [np.nan] * 6
    return [np.mean(arr), np.std(arr), np.min(arr), np.max(arr),
            float(stats.kurtosis(arr)), float(stats.skew(arr))]


# ── Feature groups ──────────────────────────────────────

def f0_features(snd):
    pitch = snd.to_pitch_ac(pitch_floor=75, pitch_ceiling=600)
    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]
    if len(f0) < 2:
        return {k: np.nan for k in
                ['f0_mean','f0_std','f0_min','f0_max','f0_kurtosis','f0_skewness','f0_std_norm']}
    m, s, mn, mx, ku, sk = stats6(f0)
    return {'f0_mean': m, 'f0_std': s, 'f0_min': mn, 'f0_max': mx,
            'f0_kurtosis': ku, 'f0_skewness': sk,
            'f0_std_norm': s / m if m > 0 else np.nan}


def intensity_features(snd):
    intensity = snd.to_intensity(minimum_pitch=75)
    vals = intensity.values.flatten()
    m, s, mn, mx, ku, sk = stats6(vals)
    return {'intensity_mean': m, 'intensity_std': s, 'intensity_min': mn,
            'intensity_max': mx, 'intensity_kurtosis': ku, 'intensity_skewness': sk}


def energy_features(y):
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    m, s, mn, mx, ku, sk = stats6(rms)
    return {'energy_mean': m, 'energy_std': s, 'energy_min': mn,
            'energy_max': mx, 'energy_kurtosis': ku, 'energy_skewness': sk}


def pause_features(snd):
    total_dur = snd.get_total_duration()
    intensity = snd.to_intensity(minimum_pitch=75)
    tg = call(intensity, "To TextGrid (silences)", -25, 0.1, 0.05, "silence", "speech")

    n = call(tg, "Get number of intervals", 1)
    pauses, speech_segs = [], []
    for i in range(1, n + 1):
        label = call(tg, "Get label of interval", 1, i)
        dur = (call(tg, "Get end time of interval", 1, i) -
               call(tg, "Get start time of interval", 1, i))
        (pauses if label == "silence" else speech_segs).append(dur)

    p = np.array(pauses)   if pauses      else np.array([0.0])
    s = np.array(speech_segs) if speech_segs else np.array([0.0])
    sp = s.sum()
    ipi = s[1:-1] if len(s) > 2 else np.array([0.0])

    return {
        'mean_pause_duration':          float(np.mean(p)),
        'total_speech_time':            float(sp),
        'total_pause_time':             float(p.sum()),
        'pause_percentage':             float(p.sum() / total_dur * 100),
        'silence_to_speech_ratio':      float(p.sum() / sp) if sp > 0 else 0.0,
        'pause_variability':            float(np.std(p)),
        'number_of_pauses':             len(pauses),
        'number_of_speech_segments':    len(speech_segs),
        'max_pause_duration':           float(np.max(p)),
        'min_pause_duration':           float(np.min(p)),
        'pause_rate':                   len(pauses) / total_dur,
        'mean_speech_segment_duration': float(np.mean(s)),
        'std_speech_segment_duration':  float(np.std(s)),
        'mean_inter_pause_interval':    float(np.mean(ipi)),
        'std_inter_pause_interval':     float(np.std(ipi)),
        'long_pause_count':             int(np.sum(p > 0.5)),
        'short_pause_count':            int(np.sum(p <= 0.5)),
        'pause_duration_skewness':      float(stats.skew(p)) if len(p) > 2 else 0.0,
    }


def articulation_features(snd):
    total_dur = snd.get_total_duration()

    # Voiced duration → approximate syllable count
    pitch = snd.to_pitch_ac(pitch_floor=75, pitch_ceiling=600)
    f0 = pitch.selected_array['frequency']
    voiced_dur = np.sum(f0 > 0) * pitch.time_step
    n_syl = max(voiced_dur / 0.15, 1)  # ~150ms per syllable average

    # Speech time (excl. pauses)
    intensity = snd.to_intensity(minimum_pitch=75)
    tg = call(intensity, "To TextGrid (silences)", -35, 0.25, 0.1, "silence", "speech")
    n = call(tg, "Get number of intervals", 1)
    speech_time = sum(
        call(tg, "Get end time of interval", 1, i) - call(tg, "Get start time of interval", 1, i)
        for i in range(1, n + 1)
        if call(tg, "Get label of interval", 1, i) == "speech"
    )

    return {
        'speech_rate':           n_syl / total_dur,
        'articulation_rate':     n_syl / speech_time if speech_time > 0 else np.nan,
        'avg_syllable_duration': speech_time / n_syl if n_syl > 0 else np.nan,
    }


def formant_features(snd):
    fmt = snd.to_formant_burg(max_number_of_formants=5, maximum_formant=5500,
                               window_length=0.025, pre_emphasis_from=50)
    t = fmt.xs()
    f1 = np.array([call(fmt, "Get value at time", 1, ti, 'Hertz', 'Linear') for ti in t])
    f2 = np.array([call(fmt, "Get value at time", 2, ti, 'Hertz', 'Linear') for ti in t])
    f1, f2 = f1[~np.isnan(f1)], f2[~np.isnan(f2)]
    disp = np.mean(f2) - np.mean(f1) if len(f1) and len(f2) else np.nan
    vtl = 34300 / (2 * disp) if disp and disp > 0 else np.nan
    return {'f1_std': np.std(f1) if len(f1) > 1 else np.nan,
            'f2_std': np.std(f2) if len(f2) > 1 else np.nan,
            'vocal_tract_length': vtl}


# ── Main extraction ─────────────────────────────────────

def extract_all(filepath):
    y, _ = librosa.load(filepath, sr=16000, mono=True)
    snd   = parselmouth.Sound(filepath)
    row = {}
    row.update(f0_features(snd))
    row.update(intensity_features(snd))
    row.update(energy_features(y))
    row.update(pause_features(snd))
    row.update(articulation_features(snd))
    row.update(formant_features(snd))
    return row


def load_folder(folder, label):
    rows = []
    files = [f for f in sorted(glob.glob(os.path.join(folder, '*')))
             if f.endswith(AUDIO_EXTS)]
    for fp in files:
        uid = Path(fp).stem
        print(f"  [{label}] {uid}")
        try:
            row = extract_all(fp)
            row['uid'] = uid
            row['group'] = label
            rows.append(row)
        except Exception as e:
            print(f"    ERROR: {e}")
    return rows


print("Extracting features...")
rows = load_folder(CONTROL_DIR, 'Control') + load_folder(ALZHEIMER_DIR, 'Alzheimer')
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved → {OUTPUT_CSV}  ({len(df)} rows)")


# ── Visualization ───────────────────────────────────────

FEATURE_COLS = [c for c in df.columns if c not in ('uid', 'group')]
GROUPS  = ['Control', 'Alzheimer']
COLORS  = {'Control': '#60a5fa', 'Alzheimer': '#f87171'}
os.makedirs(PLOTS_DIR, exist_ok=True)

n_cols = 6
n_rows = -(-len(FEATURE_COLS) // n_cols)   # ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.2))
fig.patch.set_facecolor('#0d1117')
axes = axes.flatten()

for i, feat in enumerate(FEATURE_COLS):
    ax = axes[i]
    ax.set_facecolor('#111827')
    ax.spines[['top', 'right']].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')

    data = [df[df['group'] == g][feat].dropna().values for g in GROUPS]

    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='#475569'),
                    capprops=dict(color='#475569'),
                    flierprops=dict(marker='o', markerfacecolor='#475569', markersize=2))
    for patch, g in zip(bp['boxes'], GROUPS):
        patch.set_facecolor(COLORS[g])
        patch.set_alpha(0.65)

    # Individual points with jitter
    for j, (vals, g) in enumerate(zip(data, GROUPS)):
        jitter = np.random.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), j + 1) + jitter, vals,
                   color=COLORS[g], alpha=0.55, s=14, zorder=5)

    # Stats annotation
    if len(data[0]) > 1 and len(data[1]) > 1:
        pooled = np.sqrt((np.var(data[0]) + np.var(data[1])) / 2)
        d = (np.mean(data[1]) - np.mean(data[0])) / pooled if pooled > 0 else 0
        _, p = stats.ttest_ind(data[0], data[1], equal_var=False)
        col = '#fbbf24' if p < 0.05 else '#94a3b8'
        ax.set_title(f"{feat}\nd={d:.2f}  p={p:.3f}", fontsize=6.5, color=col, pad=4)
    else:
        ax.set_title(feat, fontsize=6.5, color='#94a3b8', pad=4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Ctrl', 'AD'], fontsize=7, color='#94a3b8')
    ax.tick_params(axis='y', labelsize=6, colors='#64748b')

for i in range(len(FEATURE_COLS), len(axes)):
    axes[i].set_visible(False)

legend = [Patch(facecolor=COLORS[g], label=g) for g in GROUPS]
fig.legend(handles=legend, loc='lower right', fontsize=9,
           facecolor='#111827', edgecolor='#1e293b', labelcolor='white')

fig.suptitle('Acoustic Biomarkers — Control vs Alzheimer (n=40)',
             fontsize=13, color='white', y=1.005)
plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'all_features.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"Plot saved → {out}")
print("\nDone.")
