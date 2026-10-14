"""
Step 9c - Quality control of the forced alignment, one row per recording.

A forced aligner assumes the transcript is a faithful word-for-word record of the audio. This
is dementia-related speech, full of hesitations and restarts, so that assumption can fail, and
when it does the gaps between words, which are exactly the pause durations, come out wrong.
This step measures how far each alignment can be trusted before any pause or fluency measure is
read from it.

A recording passes (alignment_ok = 1) when
    * its TextGrid exists and can be read, and has the configured word tier;
    * the number of aligned words equals the number of transcript words;
    * no more than mfa.max_unk_fraction of the aligned words are <unk>.
The same check, from common/mfa.py, is applied by the pause-annotated BERT baseline and by the
fluency extractor, so they never disagree about which recordings are usable.

THE WORD COUNT CHECK IS WEAK FOR MFA. When MFA aligns a recording at all, it returns every word of
the transcript, so the counts nearly always agree even if the transcript does not match the audio.
The signal of a poor alignment is in the scores MFA itself writes to alignment_analysis.csv in its
output folder: the log-likelihood of the alignment, the deviation of the phone durations and the
signal-to-noise ratio. They are joined into the table below (columns mfa_*). No threshold on them
is applied here, because none has been validated on this data; read their distribution in the
report and look at the lowest-scoring recordings in Praat before choosing one. A recording MFA
could not align at all has no TextGrid and fails with the reason "no TextGrid".

INPUTS
    <mfa.corpus_dir>/<ID>.txt, .wav        from Step 9a
    <mfa.textgrid_dir>/<ID>.TextGrid       from Step 9b
    mfa.*                                  see config.yaml

OUTPUTS
    mfa.qc_csv      ID, alignment_ok, reason, n_words, n_aligned, n_unk, unk_fraction,
                    audio_duration_s, textgrid_duration_s, first_word_start_s, last_word_end_s,
                    speech_span_fraction, mfa_overall_log_likelihood, mfa_speech_log_likelihood,
                    mfa_phone_duration_deviation, mfa_snr
    mfa.qc_report   a short Markdown summary: how many pass, why the others fail, and the words
                    MFA could not find in its dictionary

    speech_span_fraction is (last word end - first word start) / audio duration. A value far from
    1 means a long stretch of the audio has no aligned word at the start or end.

HOW TO RUN
    python src/Step9_Forced_Alignment/Step9c_Alignment_QC/alignment_qc.py
"""

import collections
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import load_config, resolve_path
from common.mfa import check_alignment, read_word_intervals


def qc_one(uid, corpus, grids, tier, phone_tier, max_unk):
    """Returns (row, list of the transcript words that came back as <unk>)."""
    words = (corpus / f"{uid}.txt").read_text(encoding="utf-8").split()
    row = {"ID": uid, "alignment_ok": 0, "reason": "", "n_words": len(words), "n_aligned": pd.NA,
           "n_unk": pd.NA, "unk_fraction": pd.NA, "audio_duration_s": pd.NA, "textgrid_duration_s": pd.NA,
           "first_word_start_s": pd.NA, "last_word_end_s": pd.NA, "speech_span_fraction": pd.NA}
    wav = corpus / f"{uid}.wav"
    if wav.exists():
        row["audio_duration_s"] = round(sf.info(str(wav)).duration, 3)

    grid = grids / f"{uid}.TextGrid"
    if not grid.exists():
        row["reason"] = "no TextGrid"
        return row, []
    intervals, xmax, reason = read_word_intervals(grid, tier, phone_tier)
    if intervals is None:
        row["reason"] = reason
        return row, []

    ok, frac, reason = check_alignment(words, intervals, max_unk)
    unk_words = [w for w, (_, _, lab) in zip(words, intervals) if lab.lower() == "<unk>"]
    row.update(alignment_ok=int(ok), reason=reason, n_aligned=len(intervals), n_unk=len(unk_words),
               unk_fraction=round(frac, 4), textgrid_duration_s=round(xmax, 3))
    if intervals:
        row["first_word_start_s"] = round(intervals[0][0], 3)
        row["last_word_end_s"] = round(intervals[-1][1], 3)
        if pd.notna(row["audio_duration_s"]) and row["audio_duration_s"] > 0:
            row["speech_span_fraction"] = round((intervals[-1][1] - intervals[0][0]) / row["audio_duration_s"], 3)
    return row, unk_words


MFA_SCORES = {"overall_log_likelihood": "mfa_overall_log_likelihood",
              "speech_log_likelihood": "mfa_speech_log_likelihood",
              "phone_duration_deviation": "mfa_phone_duration_deviation",
              "snr": "mfa_snr"}


def add_mfa_scores(df, grids):
    """Joins the per-file scores MFA writes to alignment_analysis.csv. Absent file: columns stay empty."""
    for col in MFA_SCORES.values():
        df[col] = pd.NA
    path = grids / "alignment_analysis.csv"
    if not path.exists():
        print(f"  [!] {path} not found, the mfa_* score columns are empty")
        return df
    scores = pd.read_csv(path).rename(columns={"file": "ID", **MFA_SCORES})
    scores["ID"] = scores["ID"].astype(str)
    keep = ["ID"] + list(MFA_SCORES.values())
    return df.drop(columns=list(MFA_SCORES.values())).merge(scores[keep], on="ID", how="left")


def write_report(path, df, unk_counter):
    ok = int(df["alignment_ok"].sum())
    lines = ["# Alignment QC", "", f"* recordings: {len(df)}",
             f"* alignment_ok: {ok} ({100 * ok / len(df):.1f}%)", f"* failed: {len(df) - ok}", ""]
    if len(df) - ok:
        why = df.loc[df["alignment_ok"] == 0, "reason"].str.replace(r"\d+", "N", regex=True).value_counts()
        lines += ["## Reasons for failure", "", "| reason | recordings |", "| --- | --- |"]
        lines += [f"| {r} | {n} |" for r, n in why.items()] + [""]
    good = df[df["alignment_ok"] == 1]
    if len(good):
        cols = ["n_words", "unk_fraction", "audio_duration_s", "speech_span_fraction"] + list(MFA_SCORES.values())
        lines += ["## Recordings that passed", "",
                  "The mfa_* columns are MFA's own scores. Higher log-likelihood and SNR are better; a large phone duration",
                  "deviation is worse. No threshold is applied.", "", "```",
                  good[cols].astype(float).describe().round(3).to_string(), "```", ""]
    if unk_counter:
        lines += ["## Words that are not in MFA's dictionary (<unk>)", "",
                  "MFA keeps the spelling of such a word but gives it a single noise phone (spn). They are aligned, and the",
                  "fluency extractor still counts the fillers among them, but their boundaries, and so the pauses next to them,",
                  "are the least reliable. Check in particular the fillers (uhm, mm).", "",
                  "| word | count |", "| --- | --- |"]
        lines += [f"| `{w}` | {n} |" for w, n in unk_counter.most_common(30)] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    mcfg = load_config()["mfa"]
    corpus = resolve_path(mcfg["corpus_dir"])
    grids = resolve_path(mcfg["textgrid_dir"])
    ids = sorted(p.stem for p in corpus.glob("*.txt"))
    if not ids:
        sys.exit(f"No transcripts in {corpus}. Run Step9a_Prepare_Corpus first.")

    rows, unk_counter = [], collections.Counter()
    for uid in ids:
        row, unk_words = qc_one(uid, corpus, grids, mcfg["word_tier"], mcfg["phone_tier"], mcfg["max_unk_fraction"])
        rows.append(row)
        unk_counter.update(unk_words)
    df = add_mfa_scores(pd.DataFrame(rows), grids)

    out_csv, out_md = resolve_path(mcfg["qc_csv"]), resolve_path(mcfg["qc_report"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    write_report(out_md, df, unk_counter)

    ok = int(df["alignment_ok"].sum())
    print(f"{len(df)} recordings, alignment_ok = {ok} ({100 * ok / len(df):.1f}%), failed = {len(df) - ok}")
    print(f"table  -> {out_csv}")
    print(f"report -> {out_md}")
    print("Spot-check a few TextGrids in Praat before trusting any measure built on them.")


if __name__ == "__main__":
    main()
