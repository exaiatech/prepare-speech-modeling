"""
Shared definitions of the 135 acoustic features.

This module is the single source of truth for how every feature is computed.
Both extraction paths import from here and add only their own orchestration:

    Step2b_Feature_Extraction/extract_recording_level.py
                                      one feature vector per recording
    Step2b_Feature_Extraction/extract_segment_level.py
                                      one feature vector per 5 s segment
    Step3a_Feature_Inventory/build_inventory.py
                                      reads the feature names back out of
                                      extract_all() to generate the inventory

The functions were previously defined inside the recording-level extractor and
pulled into the other scripts with importlib.util.spec_from_file_location,
because a module named "03_extract_features" is not a valid Python identifier.
That re-executed the extractor on every import and made the shared code
impossible to import normally or test. The bodies are unchanged from that
version; only their location moved, so both pipelines produce identical
numbers.

The pause/fluency block lives in common/vad.py and is re-exported here, so one
import of this module reaches all 135 features.

FEATURE COUNT (135)
    F0 7 | intensity 6 | energy 6 | pause/fluency 18 | articulation 3 |
    formant 3 | voice quality (jitter/shimmer/HNR) 30 | MFCC + delta-MFCC 52 |
    spectral shape 10

LIBRARY NOTE (from the original notebook)
    Voice Quality (jitter/shimmer/HNR) uses Parselmouth, matching the biomarker
    table. MFCC/Delta-MFCC use librosa, also matching the table. Alpha Ratio,
    Hammarberg Index, Spectral Flux and Spectral Slope are computed directly
    from the STFT using their eGeMAPS definitions (numpy/librosa) rather than
    the actual openSMILE binary, since openSMILE requires a separate external
    install - the formulas are the same ones openSMILE uses internally.

SAMPLE RATE
    extract_all() builds parselmouth.Sound(filepath), which reads the file at
    its ORIGINAL sample rate, while librosa resamples to features.target_sr.
    extract_all_segment() builds the Sound from the already-resampled array, so
    both halves of its feature set see the same signal. The two paths therefore
    do not compute their Praat features from identical signal representations.
    See docs/open_questions.md (Q15).

This module defines no CLI: it is imported, never run.
"""

import os
from pathlib import Path

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import stats

from common.vad import pause_features


def stats6(arr):
    """mean, std, min, max, kurtosis, skewness"""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return [np.nan] * 6
    return [np.mean(arr), np.std(arr), np.min(arr), np.max(arr),
            float(stats.kurtosis(arr)), float(stats.skew(arr))]


# -- Prosody: F0, Intensity, Energy ------------------------------------------

def f0_features(pitch):
    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]
    m, s, mn, mx, ku, sk = stats6(f0)
    return {'f0_mean': m, 'f0_std': s, 'f0_min': mn, 'f0_max': mx,
            'f0_kurtosis': ku, 'f0_skewness': sk,
            'f0_std_norm': s / m if m and m > 0 else np.nan}


def intensity_features(intensity):
    vals = intensity.values.flatten()
    m, s, mn, mx, ku, sk = stats6(vals)
    return {'intensity_mean': m, 'intensity_std': s, 'intensity_min': mn,
            'intensity_max': mx, 'intensity_kurtosis': ku, 'intensity_skewness': sk}


def energy_features(y, cfg):
    rms = librosa.feature.rms(y=y, frame_length=cfg['frame_length'],
                              hop_length=cfg['hop_length'])[0]
    m, s, mn, mx, ku, sk = stats6(rms)
    return {'energy_mean': m, 'energy_std': s, 'energy_min': mn,
            'energy_max': mx, 'energy_kurtosis': ku, 'energy_skewness': sk}


# -- Articulation & Resonance -------------------------------------------------

def articulation_features(snd, pitch, intensity, cfg):
    total_dur = snd.get_total_duration()
    f0 = pitch.selected_array['frequency']
    voiced_dur = np.sum(f0 > 0) * pitch.time_step
    n_syl = max(voiced_dur / cfg['seconds_per_syllable'], 1)

    tg = call(intensity, "To TextGrid (silences)",
              cfg['silence_threshold_db'], cfg['min_silent_interval_s'],
              cfg['min_sounding_interval_s'], "silence", "speech")
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


def formant_features(snd, cfg):
    fmt = snd.to_formant_burg(time_step=cfg['time_step'],
                              max_number_of_formants=cfg['max_number_of_formants'],
                              maximum_formant=cfg['maximum_formant_hz'],
                              window_length=cfg['window_length'],
                              pre_emphasis_from=cfg['pre_emphasis_from_hz'])
    t = fmt.xs()
    f1 = np.array([call(fmt, "Get value at time", 1, ti, 'Hertz', 'Linear') for ti in t])
    f2 = np.array([call(fmt, "Get value at time", 2, ti, 'Hertz', 'Linear') for ti in t])
    f1, f2 = f1[~np.isnan(f1)], f2[~np.isnan(f2)]
    disp = np.mean(f2) - np.mean(f1) if len(f1) and len(f2) else np.nan
    vtl = cfg['speed_of_sound'] / (2 * disp) if disp and disp > 0 else np.nan
    return {'f1_std': np.std(f1) if len(f1) > 1 else np.nan,
            'f2_std': np.std(f2) if len(f2) > 1 else np.nan,
            'vocal_tract_length': vtl}


# -- Voice Quality: Jitter, Shimmer, HNR --------------------------------------

def voice_quality_features(snd, cfg, pitch_cfg):
    window, hop = cfg['window_s'], cfg['hop_s']
    total_dur = snd.get_total_duration()
    jl_vals, jd_vals, sl_vals, sa_vals = [], [], [], []

    t = 0.0
    while t + window <= total_dur:
        try:
            part = snd.extract_part(t, t + window, preserve_times=False)
            pp = call(part, "To PointProcess (periodic, cc)",
                      pitch_cfg['floor_hz'], pitch_cfg['ceiling_hz'])
            if call(pp, "Get number of points") >= cfg['min_points']:
                jl_vals.append(call(pp, "Get jitter (local)", 0, 0,
                                    cfg['period_floor'], cfg['period_ceiling'],
                                    cfg['max_period_factor']))
                jd_vals.append(call(pp, "Get jitter (ddp)", 0, 0,
                                    cfg['period_floor'], cfg['period_ceiling'],
                                    cfg['max_period_factor']))
                sl_vals.append(call([part, pp], "Get shimmer (local)", 0, 0,
                                    cfg['period_floor'], cfg['period_ceiling'],
                                    cfg['max_period_factor'], cfg['max_amplitude_factor']))
                sa_vals.append(call([part, pp], "Get shimmer (apq3)", 0, 0,
                                    cfg['period_floor'], cfg['period_ceiling'],
                                    cfg['max_period_factor'], cfg['max_amplitude_factor']))
        except Exception:
            pass
        t += hop

    out = {}
    for name, vals in [('jitter_local', jl_vals), ('jitter_ddp', jd_vals),
                       ('shimmer_local', sl_vals), ('shimmer_apq3', sa_vals)]:
        m, s, mn, mx, ku, sk = stats6(vals)
        out[f'{name}_mean'] = m;  out[f'{name}_std'] = s
        out[f'{name}_min'] = mn;  out[f'{name}_max'] = mx
        out[f'{name}_kurtosis'] = ku; out[f'{name}_skewness'] = sk

    harmonicity = snd.to_harmonicity_cc(time_step=cfg['harmonicity_time_step'],
                                        minimum_pitch=cfg['harmonicity_minimum_pitch'])
    hnr_vals = harmonicity.values.flatten()
    hnr_vals = hnr_vals[hnr_vals > cfg['hnr_floor_db']]
    m, s, mn, mx, ku, sk = stats6(hnr_vals)
    out['hnr_mean'] = m; out['hnr_std'] = s; out['hnr_min'] = mn
    out['hnr_max'] = mx; out['hnr_kurtosis'] = ku; out['hnr_skewness'] = sk
    return out


# -- Spectral: MFCC / Delta-MFCC (librosa) ------------------------------------

def mfcc_features(y, sr, cfg):
    n_mfcc = cfg['n_mfcc']
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    out = {}
    for i in range(n_mfcc):
        out[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        out[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))
    for i in range(n_mfcc):
        out[f'delta_mfcc_{i+1}_mean'] = float(np.mean(delta[i]))
        out[f'delta_mfcc_{i+1}_std'] = float(np.std(delta[i]))
    return out


# -- Spectral Shape: Alpha Ratio, Hammarberg, Flux, Slope ---------------------

def spectral_shape_features(y, sr, cfg):
    n_fft, hop_length = cfg['n_fft'], cfg['hop_length']
    eps = cfg['epsilon']

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = S ** 2

    a_lo, a_hi = cfg['alpha_low_band'], cfg['alpha_high_band']
    low = (freqs >= a_lo[0]) & (freqs < a_lo[1])
    high = (freqs >= a_hi[0]) & (freqs <= a_hi[1])
    e_low, e_high = power[low].sum(axis=0), power[high].sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha_ratio = 20 * np.log10(np.where(e_high > 0, e_low / np.where(e_high > 0, e_high, 1), np.nan))

    h_lo, h_hi = cfg['hammarberg_low_band'], cfg['hammarberg_high_band']
    h1 = (freqs < h_lo[1])
    h2 = (freqs >= h_hi[0]) & (freqs <= h_hi[1])
    peak1, peak2 = power[h1].max(axis=0), power[h2].max(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        hammarberg = 20 * np.log10(np.where(peak2 > 0, peak1 / np.where(peak2 > 0, peak2, 1), np.nan))

    S_norm = S / (S.sum(axis=0, keepdims=True) + eps)
    flux = np.sqrt(np.sum(np.diff(S_norm, axis=1) ** 2, axis=0))

    def band_slope(mask):
        x = freqs[mask]
        y_db = 20 * np.log10(S[mask] + eps)          # shape: (k_freqs, n_frames)
        x_c = x - x.mean()
        denom = np.sum(x_c ** 2)
        y_c = y_db - y_db.mean(axis=0, keepdims=True)
        return (x_c[:, None] * y_c).sum(axis=0) / denom

    named = [('alpha_ratio', alpha_ratio), ('hammarberg_index', hammarberg),
             ('spectral_flux', flux)]
    for name, band in cfg['slope_bands'].items():
        lo, hi = band
        mask = (freqs >= lo) & (freqs <= hi) if lo == 0 else (freqs > lo) & (freqs <= hi)
        named.append((name, band_slope(mask)))

    out = {}
    for name, arr in named:
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        out[f'{name}_mean'] = float(np.mean(arr)) if len(arr) else np.nan
        out[f'{name}_std'] = float(np.std(arr)) if len(arr) else np.nan
    return out


# -- Combine everything -------------------------------------------------------

def extract_all(filepath, config):
    fcfg = config['features']
    sr = fcfg['target_sr']

    y, _ = librosa.load(filepath, sr=sr, mono=True)
    snd = parselmouth.Sound(filepath)

    pitch = snd.to_pitch_ac(pitch_floor=fcfg['pitch']['floor_hz'],
                            pitch_ceiling=fcfg['pitch']['ceiling_hz'])
    intensity = snd.to_intensity(minimum_pitch=fcfg['intensity']['minimum_pitch_hz'])

    row = {}
    row.update(f0_features(pitch))
    row.update(intensity_features(intensity))
    row.update(energy_features(y, fcfg['energy']))
    row.update(pause_features(y, config['vad']))
    row.update(articulation_features(snd, pitch, intensity, fcfg['articulation']))
    row.update(formant_features(snd, fcfg['formant']))
    row.update(voice_quality_features(snd, fcfg['voice_quality'], fcfg['pitch']))
    row.update(mfcc_features(y, sr, fcfg['mfcc']))
    row.update(spectral_shape_features(y, sr, fcfg['spectral_shape']))
    return row


# -- Metadata-driven file discovery -------------------------------------------

def build_index(folder, audio_exts):
    """Walk the folder ONCE and build a lookup dict - avoids re-scanning
    the whole directory tree for every single file (this was the main
    bottleneck: os.walk was being run once per row)."""
    exact, prefix_list = {}, []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(audio_exts):
                path = os.path.join(root, f)
                exact[Path(f).stem] = path
                prefix_list.append((f, path))
    return exact, prefix_list


def find_audio(uid, exact, prefix_list):
    if uid in exact:
        return exact[uid]
    for fname, path in prefix_list:
        if fname.startswith(uid):
            return path
    return None


def extract_all_segment(y_seg, config):
    """Computes all 135 features for ONE segment waveform.

    Unlike extract_all(), the Sound is built from the already-resampled array
    rather than re-read from disk, so the Praat and librosa features share one
    signal representation. See the SAMPLE RATE note in the module docstring.
    """
    fcfg = config['features']
    sr = fcfg['target_sr']

    snd = parselmouth.Sound(y_seg, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(pitch_floor=fcfg['pitch']['floor_hz'],
                            pitch_ceiling=fcfg['pitch']['ceiling_hz'])
    intensity = snd.to_intensity(minimum_pitch=fcfg['intensity']['minimum_pitch_hz'])

    row = {}
    row.update(f0_features(pitch))
    row.update(intensity_features(intensity))
    row.update(energy_features(y_seg, fcfg['energy']))
    row.update(pause_features(y_seg, config['vad']))
    row.update(articulation_features(snd, pitch, intensity, fcfg['articulation']))
    row.update(formant_features(snd, fcfg['formant']))
    row.update(voice_quality_features(snd, fcfg['voice_quality'], fcfg['pitch']))
    row.update(mfcc_features(y_seg, sr, fcfg['mfcc']))
    row.update(spectral_shape_features(y_seg, sr, fcfg['spectral_shape']))
    return row
