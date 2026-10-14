"""
Step 9a - Build the Montreal Forced Aligner (MFA) input corpus.

The cohort CSV carries plain transcripts and no word timestamps, so pause and fluency
measures need a forced alignment first. This script writes, for every recording of the cohort,
one 16 kHz mono wav and one .txt transcript with the same base name, which is the corpus layout
MFA reads.

It does not run MFA. Step9b_Run_MFA/run_mfa.py does.

INPUTS
    paths.cohort                       the cohort CSV (columns ID and the transcript column)
    paths.raw_audio.train / .test      the raw recordings
    mfa.*                              see config.yaml

OUTPUTS
    <corpus_dir>/<ID>.wav              16 kHz mono
    <corpus_dir>/<ID>.txt              the normalised transcript
    <input_report>                     recordings with missing audio or very short transcripts,
                                       and what the normalisation removed

TRANSCRIPT NORMALISATION (conservative: it keeps every word)
    lower-case; "&" becomes "and"; hyphens and slashes become spaces; every character other than
    a letter, a digit or an apostrophe inside a word is removed; runs of spaces collapse. Nothing
    is spelled out, merged or dropped. Words MFA cannot find in its dictionary get a noise phone
    in the alignment, which Step9c counts as <unk>.

This script reads the RAW audio. It does not read the output of Step1b_Audio_Preprocessing, which
is acoustic-feature preprocessing only (see docs/encoders/encoder_input_preparation.md).
Transcripts that are very short or empty are listed in the report, not dropped silently.

HOW TO RUN
    python src/Step9_Forced_Alignment/Step9a_Prepare_Corpus/prepare_mfa_corpus.py
    python src/Step9_Forced_Alignment/Step9a_Prepare_Corpus/prepare_mfa_corpus.py --overwrite
"""

import argparse
import collections
import re
import sys
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import load_config, resolve_path

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a"}


def normalise(text):
    """Returns (normalised transcript, Counter of removed characters)."""
    removed = collections.Counter()
    t = str(text).lower().replace("&", " and ")
    t = re.sub(r"[-/–—]", " ", t)
    out = []
    for ch in t:
        if ch.isalnum() or ch in "' ":
            out.append(ch)
        else:
            removed[ch] += 1
            out.append(" ")
    t = "".join(out)
    # an apostrophe is kept only inside a word ("mother's"), never as a stray quote
    t = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", t)
    return re.sub(r"\s+", " ", t).strip(), removed


def index_audio(dirs):
    idx = {}
    for d in dirs:
        d = Path(d)
        if d.exists():
            for p in d.rglob("*"):
                if p.suffix.lower() in AUDIO_EXT:
                    idx[p.stem] = p
    return idx


def write_report(path, n_total, written, missing_audio, short, empty, removed):
    lines = [
        "# MFA input report", "",
        f"* recordings in the cohort file: {n_total}",
        f"* corpus entries written (wav + txt): {written}",
        f"* recordings with no audio file: {len(missing_audio)}",
        f"* recordings with an empty transcript after normalisation: {len(empty)}",
        f"* recordings with a very short transcript: {len(short)}", "",
    ]
    for title, items in (("No audio file", missing_audio), ("Empty transcript", empty),
                         ("Short transcript (ID, words)", short)):
        if items:
            lines += [f"## {title}", ""] + [f"* {i}" for i in items[:100]]
            if len(items) > 100:
                lines.append(f"* ... {len(items) - 100} more")
            lines.append("")
    if removed:
        lines += ["## Characters removed by the normalisation", "", "| character | count |", "| --- | --- |"]
        lines += [f"| `{c}` | {n} |" for c, n in removed.most_common(20)]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(overwrite=False):
    cfg = load_config()
    pcfg = cfg["mfa"]
    id_col = cfg["cohort"]["id_column"]
    text_col = pcfg["text_column"]

    cohort = pd.read_csv(resolve_path(cfg["paths"]["cohort"]), usecols=[id_col, text_col])
    audio = index_audio([resolve_path(cfg["paths"]["raw_audio"][k]) for k in ("train", "test")])
    corpus = resolve_path(pcfg["corpus_dir"])
    corpus.mkdir(parents=True, exist_ok=True)
    report_path = resolve_path(pcfg["input_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)

    written, missing_audio, empty, short = 0, [], [], []
    removed = collections.Counter()
    for _, row in cohort.iterrows():
        uid = str(row[id_col])
        words, rm = normalise(row[text_col])
        removed.update(rm)
        n_words = len(words.split())
        if n_words == 0:
            empty.append(uid)
            continue
        if n_words < pcfg["min_words_report"]:
            short.append(f"{uid} ({n_words})")
        src = audio.get(uid)
        if src is None:
            missing_audio.append(uid)
            continue
        wav, txt = corpus / f"{uid}.wav", corpus / f"{uid}.txt"
        if wav.exists() and txt.exists() and not overwrite:
            written += 1
            continue
        y, _ = librosa.load(str(src), sr=pcfg["sample_rate"], mono=True)
        sf.write(str(wav), y, pcfg["sample_rate"])
        txt.write_text(words + "\n", encoding="utf-8")
        written += 1

    write_report(report_path, len(cohort), written, missing_audio, short, empty, removed)
    print(f"{written} of {len(cohort)} corpus entries in {corpus}")
    print(f"missing audio: {len(missing_audio)} | empty transcript: {len(empty)} | short transcript: {len(short)}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="rewrite entries that already exist")
    main(parser.parse_args().overwrite)
