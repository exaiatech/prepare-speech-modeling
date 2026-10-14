"""
Step 9d - MFA-based speech fluency biomarkers, one row per recording.

Adapted from extract_mfa_fluency_biomarkers.py, supplied by the supervisor and written for the
CAADMI recordings. The measures and the column names are his, so the two tables can be compared.
What differs here is listed under "CHANGES" below.

WHAT IT MEASURES
    1. Hesitation markers: counts of filled-pause tokens (uh, um, hmm, ...), grouped, and rates.
    2. Silent pauses: the gap between consecutive aligned words, summarised for every pause
       threshold in mfa.fluency.pause_thresholds_s (count, total, mean, median, max, 95th
       percentile, rate per minute, per 100 words, share of the recording, speech-to-pause ratio).
    3. Speech runs: stretches of words not separated by a gap of mfa.fluency.run_pause_threshold_s
       or more, and their mean length in words and seconds.
    4. Speech rate and articulation rate (words per minute).
    5. Alignment status: alignment_ok, the reason when it is 0, unk_fraction.

    It does NOT compute spectral or voice-quality features (MFCC, jitter, shimmer, formants, CPP,
    LTAS). Those stay in Step 2b.

CHANGES FROM THE SUPERVISOR'S SCRIPT
    * Paths and parameters come from config.yaml (mfa:), not from constants with his folders.
    * The recordings are the cohort of Step 9a, identified by the ID column (his column is TID).
    * The word comes from the TRANSCRIPT and its time from the TextGrid. A word that is not in
      MFA's dictionary ("uhm", "mm") is read as <unk> by common/mfa.py (its phone is a noise
      phone), so the label in the reader cannot be used to recognise a hesitation.
    * The hesitation list is English and lives in config.yaml (his is German with a few English
      entries). "uhm" is added; "ah" and "oh" are not hesitations and are not counted.
    * An alignment that fails the check in common/mfa.py (word count differs from the transcript,
      too many <unk>, missing TextGrid) gets no feature values: the row has alignment_ok = 0 and
      the reason, and every feature is empty. His script trusts every TextGrid.
    * The TextGrid is read with praatio instead of a hand-written line parser.

HOW A MEASURE IS COUNTED (unchanged from his script)
    * A pause is any positive gap between the end of one word and the start of the next. Silence
      before the first word and after the last word is not a pause.
    * Speech time is the summed duration of the aligned word intervals, hesitations included.
    * Total time is the length of the TextGrid, which is the length of the audio.
    * Speech and articulation rates are given for all words ("total_words") and for the words
      that are not hesitations ("lexical_words").

INPUTS
    <mfa.corpus_dir>/<ID>.txt, .wav        from Step 9a
    <mfa.textgrid_dir>/<ID>.TextGrid       from Step 9b
    mfa.*, mfa.fluency.*                   see config.yaml

OUTPUTS (in mfa.fluency.output_dir)
    mfa_fluency_biomarkers.csv     one row per recording
    pause_events_long.csv          one row per pause (alignment_ok recordings only)
    speech_runs_long.csv           one row per speech run
    word_intervals_long.csv        one row per aligned word

HOW TO RUN
    python src/Step9_Forced_Alignment/Step9d_Fluency_Biomarkers/extract_mfa_fluency_biomarkers.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import load_config, resolve_path
from common.mfa import attach_words, check_alignment, hesitation_lookup, read_word_intervals


def safe_div(num, den):
    if den is None or den == 0 or not np.isfinite(den):
        return float("nan")
    return float(num / den)


def percentile(values, q):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")


def build_word_table(uid, timed_words, lookup):
    """One row per word: ID, word_index, word, start, end, duration, is_hesitation, hes_type."""
    rows = []
    for i, (word, start, end) in enumerate(timed_words, start=1):
        hes = lookup.get(word.lower())
        rows.append({"ID": uid, "word_index": i, "word": word, "start": start, "end": end,
                     "duration": end - start, "is_hesitation": hes is not None, "hes_type": hes})
    return pd.DataFrame(rows, columns=["ID", "word_index", "word", "start", "end", "duration",
                                       "is_hesitation", "hes_type"])


def pause_events_from_words(words, thresholds):
    """Every positive gap between consecutive words, flagged for each threshold."""
    if len(words) < 2:
        return pd.DataFrame()
    d = words.sort_values(["start", "end"]).reset_index(drop=True)
    rows = []
    for i in range(len(d) - 1):
        gap = float(d.loc[i + 1, "start"]) - float(d.loc[i, "end"])
        if gap <= 0:
            continue
        row = {"ID": d.loc[i, "ID"], "pause_index": len(rows) + 1,
               "pause_start": float(d.loc[i, "end"]), "pause_end": float(d.loc[i + 1, "start"]),
               "pause_duration": gap, "prev_word": d.loc[i, "word"], "next_word": d.loc[i + 1, "word"],
               "prev_is_hesitation": bool(d.loc[i, "is_hesitation"]),
               "next_is_hesitation": bool(d.loc[i + 1, "is_hesitation"])}
        for t in thresholds:
            row[f"is_pause_ge_{int(t * 1000)}ms"] = bool(gap >= t)
        rows.append(row)
    return pd.DataFrame(rows)


def make_run_row(run_words, run_index, run_pause_threshold):
    start, end = float(run_words["start"].min()), float(run_words["end"].max())
    n_total = int(len(run_words))
    n_hes = int(run_words["is_hesitation"].sum())
    return {"ID": run_words["ID"].iloc[0], "run_index": run_index, "run_start": start, "run_end": end,
            "run_duration": end - start, "n_words_total": n_total, "n_lexical_words": n_total - n_hes,
            "n_hesitations": n_hes, "run_pause_threshold": run_pause_threshold,
            "run_text": " ".join(run_words["word"].tolist())}


def speech_runs_from_words(words, run_pause_threshold):
    """Splits the words into runs wherever the gap is at least run_pause_threshold."""
    if words.empty:
        return pd.DataFrame()
    d = words.sort_values(["start", "end"]).reset_index(drop=True)
    runs, current = [], [0]
    for i in range(len(d) - 1):
        if float(d.loc[i + 1, "start"]) - float(d.loc[i, "end"]) >= run_pause_threshold:
            runs.append(make_run_row(d.loc[current], len(runs) + 1, run_pause_threshold))
            current = [i + 1]
        else:
            current.append(i + 1)
    runs.append(make_run_row(d.loc[current], len(runs) + 1, run_pause_threshold))
    return pd.DataFrame(runs)


def summarize_file(words, pauses, runs, duration_sec, audio_duration_sec, thresholds, groups):
    n_total = int(len(words))
    n_hes = int(words["is_hesitation"].sum())
    n_lex = n_total - n_hes
    speech_time = float(words["duration"].sum())
    total_time = float(duration_sec) if np.isfinite(duration_sec) and duration_sec > 0 else float(audio_duration_sec)
    if not np.isfinite(total_time) or total_time <= 0:
        total_time = float("nan")
    gap_time = float(pauses["pause_duration"].sum()) if not pauses.empty else 0.0

    row = {
        "duration_sec": total_time,
        "audio_duration_sec": audio_duration_sec,
        "n_words_total": n_total,
        "n_lexical_words": n_lex,
        "n_hes_total": n_hes,
    }
    for g in groups:
        row[f"n_{g}"] = int((words["hes_type"] == g).sum())
    row.update({
        "hes_per_100_words_total": safe_div(n_hes * 100, n_total),
        "hes_per_100_lexical_words": safe_div(n_hes * 100, n_lex),
        "hes_per_min": safe_div(n_hes, total_time / 60.0),
        "speech_time_sec_word_intervals": speech_time,
        "all_interword_gap_time_sec": gap_time,
        "speech_rate_wpm_total_words": safe_div(n_total, total_time / 60.0),
        "speech_rate_wpm_lexical_words": safe_div(n_lex, total_time / 60.0),
        "articulation_rate_wpm_total_words": safe_div(n_total, speech_time / 60.0),
        "articulation_rate_wpm_lexical_words": safe_div(n_lex, speech_time / 60.0),
        "n_speech_runs": int(len(runs)),
        "mean_length_of_run_words_total": float(runs["n_words_total"].mean()) if not runs.empty else float("nan"),
        "mean_length_of_run_lexical_words": float(runs["n_lexical_words"].mean()) if not runs.empty else float("nan"),
        "mean_length_of_run_sec": float(runs["run_duration"].mean()) if not runs.empty else float("nan"),
        "median_length_of_run_sec": float(runs["run_duration"].median()) if not runs.empty else float("nan"),
    })
    for t in thresholds:
        label = f"{int(t * 1000)}ms"
        p = (pauses.loc[pauses["pause_duration"] >= t, "pause_duration"].astype(float).values
             if not pauses.empty else np.array([], dtype=float))
        total = float(np.sum(p)) if p.size else 0.0
        row.update({
            f"n_pauses_ge_{label}": int(p.size),
            f"pause_total_sec_ge_{label}": total,
            f"pause_mean_sec_ge_{label}": float(np.mean(p)) if p.size else float("nan"),
            f"pause_median_sec_ge_{label}": float(np.median(p)) if p.size else float("nan"),
            f"pause_max_sec_ge_{label}": float(np.max(p)) if p.size else float("nan"),
            f"pause_p95_sec_ge_{label}": percentile(p, 95),
            f"pause_rate_per_min_ge_{label}": safe_div(p.size, total_time / 60.0),
            f"pauses_per_100_words_ge_{label}": safe_div(p.size * 100, n_total),
            f"pause_time_ratio_ge_{label}": safe_div(total, total_time),
            f"speech_to_pause_ratio_ge_{label}": safe_div(speech_time, total),
        })
    return row


def main():
    cfg = load_config()
    mcfg = cfg["mfa"]
    fcfg = mcfg["fluency"]
    thresholds = [float(t) for t in fcfg["pause_thresholds_s"]]
    run_thr = float(fcfg["run_pause_threshold_s"])
    lookup = hesitation_lookup(fcfg["hesitations"])
    groups = list(fcfg["hesitations"])

    corpus, grids = resolve_path(mcfg["corpus_dir"]), resolve_path(mcfg["textgrid_dir"])
    out_dir = resolve_path(fcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in corpus.glob("*.txt"))
    if not ids:
        sys.exit(f"No transcripts in {corpus}. Run Step9a_Prepare_Corpus first.")

    rows, pause_rows, run_rows, word_rows = [], [], [], []
    for uid in ids:
        transcript = (corpus / f"{uid}.txt").read_text(encoding="utf-8").split()
        base = {"ID": uid, "alignment_ok": 0, "reason": "", "unk_fraction": np.nan}
        grid = grids / f"{uid}.TextGrid"
        if not grid.exists():
            rows.append({**base, "reason": "no TextGrid"})
            continue
        intervals, xmax, reason = read_word_intervals(grid, mcfg["word_tier"], mcfg["phone_tier"])
        if intervals is None:
            rows.append({**base, "reason": reason})
            continue
        ok, unk_fraction, reason = check_alignment(transcript, intervals, mcfg["max_unk_fraction"])
        base["unk_fraction"] = round(unk_fraction, 4)
        if not ok:
            rows.append({**base, "reason": reason})
            continue

        wav = corpus / f"{uid}.wav"
        audio_duration = sf.info(str(wav)).duration if wav.exists() else float("nan")
        words = build_word_table(uid, attach_words(transcript, intervals), lookup)
        pauses = pause_events_from_words(words, thresholds)
        runs = speech_runs_from_words(words, run_thr)
        feats = summarize_file(words, pauses, runs, xmax, audio_duration, thresholds, groups)
        rows.append({**base, "alignment_ok": 1, **feats})
        for lst, frame in ((pause_rows, pauses), (run_rows, runs), (word_rows, words)):
            if not frame.empty:
                lst.append(frame)

    features = pd.DataFrame(rows)
    lead = ["ID", "alignment_ok", "reason", "unk_fraction"]
    features = features[lead + [c for c in features.columns if c not in lead]]
    features.to_csv(out_dir / "mfa_fluency_biomarkers.csv", index=False)
    for name, lst in (("pause_events_long", pause_rows), ("speech_runs_long", run_rows),
                      ("word_intervals_long", word_rows)):
        (pd.concat(lst, ignore_index=True) if lst else pd.DataFrame()).to_csv(out_dir / f"{name}.csv", index=False)

    ok = int(features["alignment_ok"].sum())
    print(f"{len(features)} recordings, alignment_ok = {ok} ({100 * ok / len(features):.1f}%), "
          f"no features for {len(features) - ok}")
    if ok:
        good = features[features["alignment_ok"] == 1]
        print(f"words {int(good['n_words_total'].sum())}, hesitations {int(good['n_hes_total'].sum())}, "
              f"audio {good['duration_sec'].sum() / 60:.1f} min")
        for t in thresholds:
            print(f"pauses >= {int(t * 1000)} ms: {int(good[f'n_pauses_ge_{int(t * 1000)}ms'].sum())}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
