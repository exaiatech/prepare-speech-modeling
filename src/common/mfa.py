"""
Montreal Forced Aligner (MFA) helpers shared by Step9 and the pause-annotated BERT baseline.

One alignment of the cohort feeds two consumers, and both must judge an alignment the same
way, so the reading and the checking live here:

    read_word_intervals()   the word tier of one TextGrid
    check_alignment()       is this alignment usable for this transcript?
    attach_words()          transcript words with the times of the aligned intervals
    hesitation_lookup()     token -> hesitation group, from config.yaml (mfa.fluency.hesitations)

A failed alignment is never repaired or filled in.

UNKNOWN WORDS. MFA 3.3 does not write <unk> in the word tier for a word that is not in its
dictionary. It keeps the word's own spelling there, and gives it a single "spn" (spoken noise)
phone in the phone tier. read_word_intervals() therefore reads the phone tier and returns the label
"<unk>" for a word whose phones are all "spn". Those words ("uhm", "mm", "stepstool") are aligned,
but their boundaries are only as good as a noise model, so they are counted and reported. check_alignment() says why it failed and the
caller leaves the recording out.

This module defines no CLI: it is imported, never run.
"""

from bisect import bisect_left

from praatio import textgrid

# MFA labels that are silence or noise, not words
NON_WORD_LABELS = {"", "sil", "sp", "spn", "<eps>"}


def read_word_intervals(path, tier_name, phone_tier="phones"):
    """Reads the word tier of one TextGrid.

    Returns (intervals, xmax, reason). `intervals` is a list of (start, end, label) sorted by
    time, without silences. The label is "<unk>" for a word whose phones are all "spn" (see
    UNKNOWN WORDS above); otherwise it is the word as MFA wrote it. `reason` is '' when it
    worked; otherwise intervals is None. `xmax` is the end time of the TextGrid, which is the
    length of the aligned audio. If there is no phone tier, no word is marked "<unk>".
    """
    try:
        tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=False)
    except Exception as e:
        return None, None, f"unreadable TextGrid ({type(e).__name__})"
    if tier_name not in tg.tierNames:
        return None, None, f"no tier named {tier_name!r}; found {list(tg.tierNames)}"
    entries = [(e.start, e.end, e.label.strip()) for e in tg.getTier(tier_name).entries]
    entries = sorted(e for e in entries if e[2].lower() not in NON_WORD_LABELS)

    if phone_tier in tg.tierNames:
        phones = sorted((p.start, p.end, p.label.strip().lower()) for p in tg.getTier(phone_tier).entries)
        starts = [p[0] for p in phones]
        marked = []
        for s, e, lab in entries:
            inside = []
            for k in range(bisect_left(starts, s - 1e-6), len(phones)):
                if phones[k][0] >= e - 1e-6:
                    break
                inside.append(phones[k][2])
            marked.append((s, e, "<unk>" if inside and all(x == "spn" for x in inside) else lab))
        entries = marked
    return entries, tg.maxTimestamp, ""


def check_alignment(words, intervals, max_unk_fraction):
    """Decides whether the aligned intervals can be matched to the transcript words.

    Returns (ok, unk_fraction, reason). Not ok when the number of aligned words differs from
    the number of transcript words (the words cannot be paired one to one, so no gap can be
    placed safely) or when more than `max_unk_fraction` of them are <unk>.
    """
    n = max(len(words), 1)
    unk = sum(1 for _, _, lab in intervals if lab.lower() == "<unk>")
    frac = unk / n
    if len(words) != len(intervals):
        return False, frac, f"{len(intervals)} aligned words against {len(words)} transcript words"
    if frac > max_unk_fraction:
        return False, frac, f"{unk} of {len(words)} words are <unk>"
    return True, frac, ""


def attach_words(words, intervals):
    """Pairs each transcript word with the time of its aligned interval.

    Only valid after check_alignment() returned ok. The word comes from the TRANSCRIPT, the time
    from the TextGrid. MFA writes <unk> for a word that is not in its dictionary ("uhm", "mm"),
    and a hesitation must not be lost for that reason.
    Returns a list of (word, start, end).
    """
    return [(w, s, e) for w, (s, e, _) in zip(words, intervals)]


def hesitation_lookup(groups):
    """Turns {group: [tokens]} from the config into {token: group}."""
    lookup = {}
    for group, tokens in groups.items():
        for tok in tokens:
            lookup[str(tok).strip().lower()] = group
    return lookup
