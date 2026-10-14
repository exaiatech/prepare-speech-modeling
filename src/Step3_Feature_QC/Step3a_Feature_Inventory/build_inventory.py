"""
Step 7 - Inventory of all 135 extracted acoustic biomarkers.

Written for the supervisor request of 2026-07-30: a complete inventory of the
extracted biomarkers with domain classification, precise definitions, units and
computational implementation.

It is a NEW documentation generator, not derived from either notebook, and it
computes no features of its own. To guarantee that the documentation can never
drift from the code, the feature NAMES are obtained by running the real
extractor (common/feature_functions.py) on a synthetic waveform and reading back the
keys it produces. Those names are then joined against the definition table
below. If the code ever gains, loses or renames a feature, this script fails
loudly instead of emitting a stale document.

INPUTS
    common/feature_functions.py    the authoritative feature implementation
    config.yaml                 extraction parameters quoted in the definitions

OUTPUTS
    <paths.docs_dir>/feature_inventory.csv    machine-readable, one row per feature
    <paths.docs_dir>/feature_inventory.md     grouped by domain, human-readable

DOMAINS (the six of the project's biomarker table)
    temporal/fluency, prosodic, phonatory, articulatory,
    spectral/cepstral, intensity/energy

HOW TO RUN
    python src/Step3_Feature_QC/Step3a_Feature_Inventory/build_inventory.py
    python src/Step3_Feature_QC/Step3a_Feature_Inventory/build_inventory.py --config /path/to/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import DEFAULT_CONFIG, load_config, resolve_path
from common.feature_functions import extract_all


# Suffix semantics for every family computed through stats6().
STAT_SUFFIXES = {
    'mean': ('arithmetic mean over all frames/windows', 'same as base'),
    'std': ('population standard deviation over all frames/windows', 'same as base'),
    'min': ('minimum over all frames/windows', 'same as base'),
    'max': ('maximum over all frames/windows', 'same as base'),
    'kurtosis': ("Fisher kurtosis (scipy.stats.kurtosis, excess, normal = 0)",
                 'dimensionless'),
    'skewness': ('skewness (scipy.stats.skew)', 'dimensionless'),
}

# ---------------------------------------------------------------------------
# Definition table. One entry per feature FAMILY.
#   stats6 = True   -> expands into the six STAT_SUFFIXES above
#   exact  = {...}  -> individually named features
# ---------------------------------------------------------------------------
FAMILIES = [
    dict(
        base='f0', domain='prosodic', stats6=True, unit='Hz',
        quantity='Fundamental frequency (F0) of voiced frames',
        definition=(
            'Autocorrelation pitch track; only frames with a defined pitch '
            '(F0 > 0) are kept, unvoiced frames are excluded before the '
            'statistics are computed.'),
        implementation=(
            "parselmouth Sound.to_pitch_ac(pitch_floor=features.pitch.floor_hz, "
            "pitch_ceiling=features.pitch.ceiling_hz); "
            "pitch.selected_array['frequency'] filtered to > 0"),
        extra=[dict(
            name='f0_std_norm', unit='dimensionless',
            quantity='Normalised F0 variability (coefficient of variation)',
            definition='f0_std divided by f0_mean; NaN when f0_mean <= 0.',
            implementation='s / m from the same stats6() call')],
    ),
    dict(
        base='intensity', domain='intensity/energy', stats6=True, unit='dB SPL (Praat scale)',
        quantity='Short-term intensity contour',
        definition=(
            'Praat intensity contour of the whole signal, in decibels on '
            "Praat's internal SPL scale. All frames are used, including "
            'silent ones.'),
        implementation=(
            "parselmouth Sound.to_intensity("
            "minimum_pitch=features.intensity.minimum_pitch_hz); "
            "intensity.values.flatten()"),
    ),
    dict(
        base='energy', domain='intensity/energy', stats6=True, unit='normalised amplitude',
        quantity='Frame-wise root-mean-square energy',
        definition=(
            'RMS amplitude per analysis frame of the 16 kHz waveform, on the '
            'librosa amplitude scale (roughly -1..1 input).'),
        implementation=(
            "librosa.feature.rms(y, frame_length=features.energy.frame_length, "
            "hop_length=features.energy.hop_length)"),
    ),
    dict(
        base='jitter_local', domain='phonatory', stats6=True, unit='fraction (0-1)',
        quantity='Local jitter - cycle-to-cycle frequency perturbation',
        definition=(
            'Mean absolute difference between consecutive glottal periods, '
            'divided by the mean period. Computed on 1 s windows stepped by '
            '0.5 s; the six statistics are taken ACROSS those windows, not '
            'across periods.'),
        implementation=(
            "Praat 'To PointProcess (periodic, cc)' then 'Get jitter (local)' "
            "per window; window/hop from features.voice_quality"),
    ),
    dict(
        base='jitter_ddp', domain='phonatory', stats6=True, unit='fraction (0-1)',
        quantity='Jitter DDP - difference of differences of periods',
        definition=(
            'Mean absolute difference between consecutive differences of '
            'glottal periods, divided by the mean period. Three times the RAP '
            'by construction. Windowed exactly like jitter_local.'),
        implementation="Praat 'Get jitter (ddp)' per 1 s window",
    ),
    dict(
        base='shimmer_local', domain='phonatory', stats6=True, unit='fraction (0-1)',
        quantity='Local shimmer - cycle-to-cycle amplitude perturbation',
        definition=(
            'Mean absolute difference between the peak amplitudes of '
            'consecutive glottal periods, divided by the mean amplitude. '
            'Windowed exactly like jitter_local.'),
        implementation="Praat 'Get shimmer (local)' per 1 s window",
    ),
    dict(
        base='shimmer_apq3', domain='phonatory', stats6=True, unit='fraction (0-1)',
        quantity='Three-point amplitude perturbation quotient',
        definition=(
            'Mean absolute difference between the amplitude of a period and '
            'the average of it and its two neighbours, divided by the mean '
            'amplitude. Windowed exactly like jitter_local.'),
        implementation="Praat 'Get shimmer (apq3)' per 1 s window",
    ),
    dict(
        base='hnr', domain='phonatory', stats6=True, unit='dB',
        quantity='Harmonics-to-noise ratio',
        definition=(
            'Ratio of periodic (harmonic) to aperiodic (noise) energy per '
            'frame, in dB. Frames at or below features.voice_quality.'
            'hnr_floor_db are Praat undefined sentinels and are discarded '
            'before the statistics.'),
        implementation=(
            "parselmouth Sound.to_harmonicity_cc("
            "time_step=features.voice_quality.harmonicity_time_step, "
            "minimum_pitch=features.voice_quality.harmonicity_minimum_pitch)"),
    ),
    dict(
        base='alpha_ratio', domain='spectral/cepstral', stats2=True, unit='dB (see caveat)',
        quantity='Alpha ratio - low-band to high-band energy balance',
        definition=(
            'Ratio of summed spectral power in features.spectral_shape.'
            'alpha_low_band to that in alpha_high_band, expressed '
            'logarithmically, per STFT frame. eGeMAPS descriptor.'),
        implementation=(
            'numpy/librosa STFT; 20*log10(E_low / E_high). CAVEAT: these are '
            'POWER sums, so the eGeMAPS definition is 10*log10 - the values '
            'are twice the intended dB. See open_questions.md Q5.'),
    ),
    dict(
        base='hammarberg_index', domain='spectral/cepstral', stats2=True, unit='dB (see caveat)',
        quantity='Hammarberg index - peak energy balance across the spectrum',
        definition=(
            'Difference between the strongest spectral peak below 2 kHz and '
            'the strongest peak in 2-5 kHz, per STFT frame. eGeMAPS descriptor.'),
        implementation=(
            'numpy/librosa STFT; 20*log10(peak_low / peak_high). Same '
            '10 vs 20 caveat as alpha_ratio (Q5).'),
    ),
    dict(
        base='spectral_flux', domain='spectral/cepstral', stats2=True, unit='dimensionless',
        quantity='Spectral flux - rate of change of the spectral envelope',
        definition=(
            'Euclidean distance between the amplitude spectra of consecutive '
            'frames, after each frame is normalised to sum to one.'),
        implementation=(
            'S_norm = S / (S.sum(axis=0) + eps); '
            'sqrt(sum(diff(S_norm, axis=1)**2, axis=0))'),
    ),
    dict(
        base='spectral_slope_0_500', domain='spectral/cepstral', stats2=True, unit='dB/Hz',
        quantity='Spectral slope, 0-500 Hz band',
        definition=(
            'Least-squares regression slope of the log-amplitude spectrum '
            'against frequency within 0-500 Hz, per frame. The top SHAP '
            'feature of both pipelines.'),
        implementation=(
            'band_slope(): 20*log10(S + eps) regressed on frequency, '
            'closed-form least squares'),
    ),
    dict(
        base='spectral_slope_500_1500', domain='spectral/cepstral', stats2=True, unit='dB/Hz',
        quantity='Spectral slope, 500-1500 Hz band',
        definition=(
            'As spectral_slope_0_500, over the 500-1500 Hz band (lower bound '
            'exclusive).'),
        implementation='band_slope() on the (500, 1500] mask',
    ),
]

# Individually named features that do not follow a stats pattern.
SINGLE_FEATURES = [
    # -- temporal / fluency (WebRTC VAD, common/vad.py) ------------
    ('mean_pause_duration', 'temporal/fluency', 's', 'Mean pause duration',
     'Arithmetic mean of all detected pause durations; 0.0 if no pause was detected.'),
    ('total_speech_time', 'temporal/fluency', 's', 'Total speech time',
     'Summed duration of all speech runs. Falls back to the full recording duration '
     'if no speech run was detected.'),
    ('total_pause_time', 'temporal/fluency', 's', 'Total pause time',
     'Summed duration of all pause runs; 0.0 if none.'),
    ('pause_percentage', 'temporal/fluency', '%', 'Pause proportion',
     'total_pause_time divided by the full signal duration, times 100.'),
    ('silence_to_speech_ratio', 'temporal/fluency', 'ratio', 'Silence-to-speech ratio',
     'total_pause_time divided by total_speech_time; 0.0 if either is zero.'),
    ('pause_variability', 'temporal/fluency', 's', 'Pause duration variability',
     'Population standard deviation of pause durations; 0.0 with fewer than two pauses.'),
    ('number_of_pauses', 'temporal/fluency', 'count', 'Number of pauses',
     'Count of pause runs surviving the minimum-length filter.'),
    ('number_of_speech_segments', 'temporal/fluency', 'count', 'Number of speech runs',
     'Count of speech runs surviving the minimum-length filter. NOTE: unrelated to the '
     '5 s analysis segments of the segment pipeline.'),
    ('max_pause_duration', 'temporal/fluency', 's', 'Longest pause', 'Maximum pause duration.'),
    ('min_pause_duration', 'temporal/fluency', 's', 'Shortest pause', 'Minimum pause duration.'),
    ('pause_rate', 'temporal/fluency', 'pauses/s', 'Pause rate',
     'number_of_pauses divided by the full signal duration.'),
    ('mean_speech_segment_duration', 'temporal/fluency', 's', 'Mean speech run duration',
     'Arithmetic mean of speech run durations.'),
    ('std_speech_segment_duration', 'temporal/fluency', 's', 'Speech run variability',
     'Standard deviation of speech run durations; 0.0 with fewer than two runs.'),
    ('mean_inter_pause_interval', 'temporal/fluency', 's', 'Mean inter-pause interval',
     'Mean duration of the speech runs excluding the first and last one, i.e. of the runs '
     'that are bounded by a pause on both sides.'),
    ('std_inter_pause_interval', 'temporal/fluency', 's', 'Inter-pause interval variability',
     'Standard deviation of the same interior speech runs.'),
    ('long_pause_count', 'temporal/fluency', 'count', 'Long pauses',
     'Pauses strictly longer than vad.long_pause_threshold_s (0.5 s).'),
    ('short_pause_count', 'temporal/fluency', 'count', 'Short pauses',
     'Pauses with duration in (0, vad.long_pause_threshold_s].'),
    ('pause_duration_skewness', 'temporal/fluency', 'dimensionless', 'Pause duration skewness',
     'Skewness of the pause duration distribution; 0.0 with three or fewer pauses.'),
    # -- articulatory ---------------------------------------------------------
    ('speech_rate', 'articulatory', 'syllables/s', 'Speech rate (pauses included)',
     'Estimated syllable count divided by the full signal duration. Syllables are '
     'approximated as total voiced duration / features.articulation.seconds_per_syllable '
     '(0.15 s), floored at 1 - there is no syllable nucleus detector.'),
    ('articulation_rate', 'articulatory', 'syllables/s', 'Articulation rate (pauses excluded)',
     'Estimated syllable count divided by Praat-derived speech time. NOTE: this uses the '
     'Praat "To TextGrid (silences)" speech/silence split at -35 dB, NOT the WebRTC VAD '
     'used for the pause features - the two pause definitions coexist in one feature set.'),
    ('avg_syllable_duration', 'articulatory', 's', 'Mean syllable duration',
     'Praat speech time divided by the estimated syllable count.'),
    ('f1_std', 'articulatory', 'Hz', 'First formant variability',
     'Standard deviation of F1 over all frames where Burg formant tracking returned a '
     'defined value.'),
    ('f2_std', 'articulatory', 'Hz', 'Second formant variability',
     'Standard deviation of F2 over all frames with a defined value.'),
    ('vocal_tract_length', 'articulatory', 'cm', 'Estimated vocal tract length',
     'features.formant.speed_of_sound / (2 * (mean(F2) - mean(F1))). NaN if the formant '
     'dispersion is not positive. A coarse proxy, not an anatomical measurement.'),
]

MFCC_DOMAIN = 'spectral/cepstral'


def authoritative_feature_names(config):
    """Runs the real extractor on a synthetic vowel to read back its feature keys."""
    import soundfile as sf
    import tempfile

    sr = config['features']['target_sr']

    t = np.arange(int(3 * sr)) / sr
    voiced = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 25))
    voiced *= 0.2 / np.max(np.abs(voiced))
    y = np.concatenate([voiced, np.zeros(sr), voiced]).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, 'inventory_probe.wav')
        sf.write(wav, y, sr)
        row = extract_all(wav, config)
    return list(row.keys())


def expand_definitions():
    """Builds {feature_name: definition dict} from FAMILIES and SINGLE_FEATURES."""
    defs = {}

    for fam in FAMILIES:
        base = fam['base']
        if fam.get('stats6'):
            suffixes = STAT_SUFFIXES.items()
        elif fam.get('stats2'):
            suffixes = [(k, v) for k, v in STAT_SUFFIXES.items() if k in ('mean', 'std')]
        else:
            suffixes = []

        for suffix, (suffix_def, suffix_unit) in suffixes:
            unit = fam['unit'] if suffix_unit == 'same as base' else suffix_unit
            defs[f'{base}_{suffix}'] = dict(
                domain=fam['domain'],
                quantity=f"{fam['quantity']} - {suffix}",
                definition=f"{fam['definition']} Statistic: {suffix_def}.",
                unit=unit,
                implementation=fam['implementation'],
            )

        for e in fam.get('extra', []):
            defs[e['name']] = dict(
                domain=fam['domain'], quantity=e['quantity'],
                definition=e['definition'], unit=e['unit'],
                implementation=e['implementation'],
            )

    for name, domain, unit, quantity, definition in SINGLE_FEATURES:
        impl = ('common/vad.py pause_features(), WebRTC VAD, 20 ms frames'
                if domain == 'temporal/fluency'
                else 'common/feature_functions.py articulation_features() / formant_features()')
        defs[name] = dict(domain=domain, quantity=quantity, definition=definition,
                          unit=unit, implementation=impl)

    for i in range(1, 14):
        for stat, stat_def in (('mean', 'arithmetic mean over frames'),
                               ('std', 'standard deviation over frames')):
            defs[f'mfcc_{i}_{stat}'] = dict(
                domain=MFCC_DOMAIN,
                quantity=f'Mel-frequency cepstral coefficient {i} - {stat}',
                definition=(
                    f'Coefficient {i} of the 13-coefficient MFCC representation of the '
                    f'16 kHz signal, summarised by its {stat_def}. Coefficient 1 tracks '
                    'overall log-energy; higher coefficients describe progressively finer '
                    'spectral envelope detail.'),
                unit='dimensionless (cepstral)',
                implementation='librosa.feature.mfcc(y, sr, n_mfcc=features.mfcc.n_mfcc)',
            )
            defs[f'delta_mfcc_{i}_{stat}'] = dict(
                domain=MFCC_DOMAIN,
                quantity=f'Delta MFCC {i} - {stat}',
                definition=(
                    f'First temporal derivative of MFCC {i}, i.e. its local rate of '
                    f'change across frames, summarised by its {stat_def}.'),
                unit='dimensionless (cepstral/frame)',
                implementation='librosa.feature.delta(mfcc)',
            )

    return defs


def build_inventory(config):
    """Joins the authoritative feature names against the definition table."""
    names = authoritative_feature_names(config)
    defs = expand_definitions()

    missing = [n for n in names if n not in defs]
    extra = [n for n in defs if n not in names]
    if missing or extra:
        raise SystemExit(
            "Feature inventory is out of sync with extract_recording_level.py.\n"
            f"  Produced by the code but undocumented ({len(missing)}): {missing}\n"
            f"  Documented but not produced ({len(extra)}): {extra}")

    rows = []
    for order, name in enumerate(names, 1):
        d = defs[name]
        rows.append(dict(order=order, feature=name, domain=d['domain'],
                         quantity=d['quantity'], definition=d['definition'],
                         unit=d['unit'], implementation=d['implementation']))
    return pd.DataFrame(rows)


def write_markdown(df, path, config):
    """Renders the inventory grouped by domain."""
    fcfg = config['features']
    lines = [
        '# Acoustic Biomarker Inventory',
        '',
        f'Complete inventory of the **{len(df)} acoustic features** extracted by',
        '`src/Step2_Acoustic/Step2b_Feature_Extraction/extract_recording_level.py` (the',
        'recording-level pipeline). The segment-level extractor, which computed the same',
        'definitions for 5 s windows, was set aside after the supervisor review and is in',
        '`archive/set_aside_after_review/`.',
        '',
        'This file is generated by `Step3_Feature_QC/Step3a_Feature_Inventory/build_inventory.py`,',
        'which reads the feature names back out of the running extractor. It cannot',
        'drift from the code: if a feature is added, removed or renamed, the generator',
        'aborts instead of writing a stale document.',
        '',
        '## Extraction context',
        '',
        f'- Signal for the librosa-based features: mono, resampled to '
        f'**{fcfg["target_sr"]} Hz**',
        f'- Pitch analysis range: **{fcfg["pitch"]["floor_hz"]}-'
        f'{fcfg["pitch"]["ceiling_hz"]} Hz**',
        f'- STFT for the spectral-shape features: n_fft '
        f'**{fcfg["spectral_shape"]["n_fft"]}**, hop '
        f'**{fcfg["spectral_shape"]["hop_length"]}**',
        f'- MFCC: **{fcfg["mfcc"]["n_mfcc"]}** coefficients plus their first derivatives',
        f'- Voice quality: **{fcfg["voice_quality"]["window_s"]} s** windows, '
        f'**{fcfg["voice_quality"]["hop_s"]} s** hop',
        f'- Voice activity detection: WebRTC, aggressiveness '
        f'**{config["vad"]["aggressiveness"]}**, '
        f'**{config["vad"]["frame_ms"]} ms** frames',
        '',
        '> **Sample-rate caveat.** In the whole-recording pipeline the Praat-based',
        '> features (F0, intensity, articulation, formants, voice quality) are computed',
        '> from the file at its ORIGINAL sample rate, while the librosa-based features',
        '> (energy, MFCC, spectral shape) use the 16 kHz resampled signal. The set-aside',
        '> segment extractor used 16 kHz for both. See `open_questions.md` Q11 and Q15.',
        '',
        '## Summary by domain',
        '',
        '| Domain | Features |',
        '| --- | --- |',
    ]
    counts = df['domain'].value_counts()
    for domain in sorted(counts.index):
        lines.append(f'| {domain} | {counts[domain]} |')
    lines.append(f'| **total** | **{len(df)}** |')
    lines.append('')

    for domain in sorted(df['domain'].unique()):
        sub = df[df['domain'] == domain]
        lines += [f'## {domain} ({len(sub)} features)', '',
                  '| # | Feature | Quantity | Unit | Definition | Implementation |',
                  '| --- | --- | --- | --- | --- | --- |']
        for _, r in sub.iterrows():
            def esc(v):
                return str(v).replace('|', '\\|').replace('\n', ' ')
            lines.append(f"| {r['order']} | `{r['feature']}` | {esc(r['quantity'])} | "
                         f"{esc(r['unit'])} | {esc(r['definition'])} | "
                         f"{esc(r['implementation'])} |")
        lines.append('')

    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))


def main(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    rcfg = config['reporting']

    df = build_inventory(config)

    out_dir = resolve_path(config['paths']['docs_dir'])
    os.makedirs(out_dir, exist_ok=True)
    csv_path = out_dir / rcfg['feature_inventory_csv']
    md_path = out_dir / rcfg['feature_inventory_md']

    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path, config)

    print(f"Inventory verified against the extractor: {len(df)} features")
    print(df['domain'].value_counts().to_string())
    print(f"\nSaved -> {csv_path}")
    print(f"Saved -> {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
