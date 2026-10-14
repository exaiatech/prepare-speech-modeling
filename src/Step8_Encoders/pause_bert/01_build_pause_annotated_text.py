"""
Pause-annotated BERT, script 01: insert pause markers into the transcripts.

Reads the TextGrids that MFA wrote (Step9_Forced_Alignment), measures the gap between consecutive
words, and writes a transcript with a punctuation marker at every pause, following Li et al.
(arXiv:2506.11119):

    * a gap longer than 0.05 s gets a marker                      (paper)
    * short  (under 0.5 s)      ->  ","                           (paper)
    * medium (0.5 s to 2 s)     ->  "."                           (paper)
    * long   (over 2 s)         ->  "..."                         (paper)
    * other punctuation is removed and the text is lower-case    (paper, uncased BERT)

WHERE THIS DIFFERS FROM THE PAPER
    The paper took word times from Whisper and also marked "sentence pauses", the silence
    between adjacent Whisper segments. We have MFA word intervals and no segments, so there is
    only one kind of pause here: the gap between consecutive aligned words. The paper does not
    state a separate sentence-pause threshold either. Everything the paper leaves open is a
    parameter in config.yaml (stage1.pause_bert, and mfa for the alignment) marked "ours".

A FAILED ALIGNMENT IS NEVER FILLED IN
    alignment_ok is 0, and text_pause_annotated is left empty, when
      * there is no TextGrid, or it cannot be parsed, or it has no word tier;
      * the number of aligned words differs from the number of words in the transcript
        (the words cannot be matched one to one, so no pause can be placed safely);
      * more than mfa.max_unk_fraction of the aligned words are <unk>.
    This is the check of common/mfa.py, the same one Step 9c and Step 9d apply.

INPUTS
    <mfa.corpus_dir>/<ID>.txt          the transcripts written by Step 9a
    <mfa.textgrid_dir>/<ID>.TextGrid   the MFA output of Step 9b
    mfa.*, stage1.pause_bert.*         see config.yaml

OUTPUTS
    <annotated_csv>    ID, text, text_pause_annotated, n_pauses, alignment_ok, then n_short,
                       n_medium, n_long, n_words, unk_fraction, reason
    <work_dir>/alignment_report.md

HOW TO RUN
    python src/Step8_Encoders/pause_bert/01_build_pause_annotated_text.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import load_config, resolve_path
from common.mfa import check_alignment, read_word_intervals


def classify_gap(gap, pcfg):
    """Returns 'short', 'medium', 'long' or None when the gap is not long enough to mark."""
    if gap <= pcfg["pause_threshold_s"]:
        return None
    lo, hi, edge_medium = pcfg["short_below_s"], pcfg["long_above_s"], pcfg["boundary_is_medium"]
    if gap < lo or (gap == lo and not edge_medium):
        return "short"
    if gap > hi or (gap == hi and not edge_medium):
        return "long"
    return "medium"


def annotate(words, intervals, pcfg, max_unk_fraction):
    """Builds the annotated text. `words` is the transcript, `intervals` the aligned words.

    Returns (text, counts, unk_fraction) or raises ValueError with the reason the alignment
    cannot be used."""
    ok, unk_fraction, reason = check_alignment(words, intervals, max_unk_fraction)
    if not ok:
        raise ValueError(reason)

    markers = pcfg["markers"]
    counts = {"short": 0, "medium": 0, "long": 0}
    out = []

    def mark(gap):
        kind = classify_gap(gap, pcfg)
        if kind:
            counts[kind] += 1
            out.append(markers[kind])

    if pcfg["mark_before_first_word"] and intervals:
        mark(intervals[0][0])
    for i, word in enumerate(words):
        if i > 0:
            mark(max(0.0, intervals[i][0] - intervals[i - 1][1]))
        out.append(word)
    return " ".join(out), counts, unk_fraction


def main():
    cfg = load_config()
    pcfg, mcfg = cfg["stage1"]["pause_bert"], cfg["mfa"]
    corpus = resolve_path(mcfg["corpus_dir"])
    grids = resolve_path(mcfg["textgrid_dir"])
    out_csv = resolve_path(pcfg["annotated_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(corpus.glob("*.txt"))
    if not txt_files:
        sys.exit(f"No transcripts in {corpus}. Run Step9a_Prepare_Corpus first.")

    rows = []
    for txt in txt_files:
        uid = txt.stem
        words = txt.read_text(encoding="utf-8").split()
        row = {"ID": uid, "text": " ".join(words), "text_pause_annotated": "", "n_pauses": pd.NA,
               "alignment_ok": 0, "n_short": pd.NA, "n_medium": pd.NA, "n_long": pd.NA,
               "n_words": len(words), "unk_fraction": pd.NA, "reason": ""}
        grid = grids / f"{uid}.TextGrid"
        if not grid.exists():
            row["reason"] = "no TextGrid"
        else:
            intervals, _, reason = read_word_intervals(grid, mcfg["word_tier"], mcfg["phone_tier"])
            if intervals is None:
                row["reason"] = reason
            else:
                try:
                    text, counts, unk = annotate(words, intervals, pcfg, mcfg["max_unk_fraction"])
                    row.update(text_pause_annotated=text, n_pauses=sum(counts.values()), alignment_ok=1,
                               n_short=counts["short"], n_medium=counts["medium"], n_long=counts["long"],
                               unk_fraction=round(unk, 4))
                except ValueError as e:
                    row["reason"] = str(e)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    ok = int(df["alignment_ok"].sum())
    print(f"{len(df)} recordings, alignment_ok = {ok} ({100 * ok / len(df):.1f}%), failed = {len(df) - ok}")
    lines = ["# Alignment report", "", f"* recordings: {len(df)}", f"* alignment_ok: {ok} ({100 * ok / len(df):.1f}%)",
             f"* failed: {len(df) - ok}", ""]
    if len(df) - ok:
        why = df.loc[df["alignment_ok"] == 0, "reason"].str.replace(r"\d+", "N", regex=True).value_counts()
        lines += ["## Reasons for failure", "", "| reason | recordings |", "| --- | --- |"]
        lines += [f"| {r} | {n} |" for r, n in why.items()] + [""]
    good = df[df["alignment_ok"] == 1]
    if len(good):
        lines += ["## Markers per recording (alignment_ok only)", "",
                  "```", good[["n_pauses", "n_short", "n_medium", "n_long"]].astype(float).describe().round(2).to_string(), "```", ""]
    (out_csv.parent / "alignment_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"annotated text -> {out_csv}")
    print("Check alignment_ok, and spot-check a few TextGrids, before trusting any model built on this.")


if __name__ == "__main__":
    main()
