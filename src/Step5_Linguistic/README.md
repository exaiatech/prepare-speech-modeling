# Step 5: Linguistic Features

**Not written yet. This folder is a placeholder, so that the gap between
Step 4 and Step 6 is visibly deliberate rather than looking like a mistake.**

The numbering comes from the project architecture, where Step 5 is the second
half of the ASR branch. Renumbering `Step6_Modeling` down to 5
would only have to be undone once this step exists, so the gap stays.

## What belongs here

CYMO linguistic features, extracted from the transcripts that Step 4 produced
and validated.

This is the point where the ASR branch finally produces something a model can
consume. Step 4 stops at "here are the transcripts, and here is the evidence
that they are accurate enough to trust". Step 5 turns that text into feature
vectors, the way `Step2_Acoustic` turns audio into the 135 acoustic features.

## Where it sits

```
Step4_ASR                                 Step2_Acoustic
  transcripts + word error rate             135 acoustic features
        |                                         |
        v                                         |
  Step5_Linguistic                                |
    CYMO features  <-- this folder               |
        |                                         |
        '---------------> Step6_Modeling <-------'
                          multimodal fusion
                          (supervised, not in this repository)
```

## Expected inputs when it is written

* `data/interim/transcripts/*.json` from Step 4b, or the verticalized tables
  from Step 4d if the features need per-word error annotations.
* `data/metadata/recording_metadata.csv` for the labels and the cohort filter.

## Expected outputs

A feature table keyed by `uid`, matching the shape the acoustic branch already
produces, so that `Step6_Modeling` can join the two without special cases.

## Open questions before starting

1. Which CYMO feature set, and at what unit of analysis. The acoustic branch
   reports both recording level and segment level; the linguistic side has no
   obvious segment equivalent.
2. Whether features are computed on the Whisper transcripts or only on the 27
   hand-checked gold transcripts. The gold set is far too small to model on,
   but the Whisper set carries the word error rate Step 4e measured.
3. Whether disfluencies stay in. The transcription settings in Step 4b were
   deliberately tuned to preserve hesitations and stutters, so discarding them
   here would waste that decision.
