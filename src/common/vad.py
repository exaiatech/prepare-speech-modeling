"""
WebRTC voice activity detection and pause/speech segmentation.

Derived from notebook cell 12 (features.ipynb), function `pause_features`.
The identical VAD logic first appeared in cell 11; the Praat-TextGrid based
predecessor (cell 10) is in archive/dead_code/03_extract_features/.

pause_features() produces the 18 temporal/fluency features of the
135-feature set and is called from common/feature_functions.py, so both the
recording-level and the segment-level extractor use it unchanged.

Step2_Acoustic/Step2a_VAD_Segmentation/vad_segment.py is a thin command-line
wrapper over this module, for inspecting the segmentation of individual files.

INPUT
    a mono waveform at vad.sample_rate (float32 in [-1, 1])
    the config.yaml `vad:` block

OUTPUT
    dict of 18 pause / speech-segment features

METHOD
    20 ms frames are classified speech/silence by WebRTC VAD (aggressiveness 2).
    Runs shorter than min_silence_frames (100 ms) or min_speech_frames (40 ms)
    are flipped, then adjacent equal labels are re-merged. Frame counts are
    converted to durations and summarised.
"""


from itertools import groupby

import numpy as np
import webrtcvad
from scipy import stats


def label_frames(y, cfg):
    """Classifies each fixed-length frame as speech (True) or silence (False)."""
    sr = cfg["sample_rate"]

    # Convert float32 -> int16 PCM (required by WebRTC VAD)
    pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)

    vad = webrtcvad.Vad(cfg["aggressiveness"])
    frame_ms = cfg["frame_ms"]
    frame_len = int(sr * frame_ms / 1000)

    labels = []
    for i in range(0, len(pcm) - frame_len, frame_len):
        try:
            labels.append(vad.is_speech(pcm[i:i + frame_len].tobytes(), sr))
        except Exception:
            labels.append(True)
    return labels


def merge_runs(labels, min_silence, min_speech):
    """Filters out very short bursts/gaps, then re-merges adjacent equal labels."""
    runs = [(k, len(list(g))) for k, g in groupby(labels)]

    merged = []
    for is_speech, count in runs:
        if not is_speech and count < min_silence:
            merged.append((True, count))
        elif is_speech and count < min_speech:
            merged.append((False, count))
        else:
            merged.append((is_speech, count))

    final = []
    for is_speech, count in merged:
        if final and final[-1][0] == is_speech:
            final[-1] = (is_speech, final[-1][1] + count)
        else:
            final.append((is_speech, count))
    return final


def segment_durations(final_runs, frame_ms):
    """Converts merged frame runs into pause and speech-segment durations (seconds)."""
    pauses, speech_segs = [], []
    for is_speech, count in final_runs:
        dur = count * frame_ms / 1000
        (speech_segs if is_speech else pauses).append(dur)
    return pauses, speech_segs


def pause_features(y, cfg):
    """
    WebRTC VAD-based pause detection - robust to background noise.

    Returns the 18 temporal/fluency features of the 135-feature set.
    """
    sr = cfg["sample_rate"]
    total_dur = len(y) / sr

    labels = label_frames(y, cfg)
    final = merge_runs(labels, cfg["min_silence_frames"], cfg["min_speech_frames"])
    pauses, speech_segs = segment_durations(final, cfg["frame_ms"])

    long_thr = cfg["long_pause_threshold_s"]

    p = np.array(pauses) if pauses else np.array([0.0])
    s = np.array(speech_segs) if speech_segs else np.array([total_dur])
    sp = s.sum()
    ipi = s[1:-1] if len(s) > 2 else np.array([0.0])
    n_p = len(pauses)

    return {
        'mean_pause_duration':          float(np.mean(p)) if n_p else 0.0,
        'total_speech_time':            float(sp),
        'total_pause_time':             float(p.sum()) if n_p else 0.0,
        'pause_percentage':             float(p.sum() / total_dur * 100) if n_p else 0.0,
        'silence_to_speech_ratio':      float(p.sum() / sp) if n_p and sp > 0 else 0.0,
        'pause_variability':            float(np.std(p)) if n_p > 1 else 0.0,
        'number_of_pauses':             n_p,
        'number_of_speech_segments':    len(speech_segs),
        'max_pause_duration':           float(np.max(p)) if n_p else 0.0,
        'min_pause_duration':           float(np.min(p)) if n_p else 0.0,
        'pause_rate':                   n_p / total_dur,
        'mean_speech_segment_duration': float(np.mean(s)),
        'std_speech_segment_duration':  float(np.std(s)) if len(s) > 1 else 0.0,
        'mean_inter_pause_interval':    float(np.mean(ipi)),
        'std_inter_pause_interval':     float(np.std(ipi)) if len(ipi) > 1 else 0.0,
        'long_pause_count':             int(np.sum(p > long_thr)),
        'short_pause_count':            int(np.sum((p > 0) & (p <= long_thr))),
        'pause_duration_skewness':      float(stats.skew(p)) if len(p) > 2 else 0.0,
    }
