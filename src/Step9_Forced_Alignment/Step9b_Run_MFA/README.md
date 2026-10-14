# Step 9b: Run the Montreal Forced Aligner

### Purpose
Aligns every recording of the corpus to its transcript and writes one `<ID>.TextGrid` with the start and end time of each word.
`run_mfa.py` is a thin wrapper around `mfa align`: it reads the settings from the `mfa:` section of `config.yaml`, finds the MFA
executable, makes its conda environment usable and checks that every recording got a TextGrid. It adds no alignment logic.

### Install MFA (once)
MFA is installed with conda, in its own environment, not into the project's Python environment:

```bash
conda create -n aligner -c conda-forge montreal-forced-aligner -y
```

The official installation page does not state that Windows is fully supported. If the install fails there, run MFA in WSL, in
Docker or on a Linux machine, and copy the corpus and the TextGrids across. The wrapper works the same way.

Then tell the repository where it is, in `config.local.yaml` (gitignored), or put `mfa` on the PATH:

```yaml
mfa:
  executable: 'C:/Users/you/miniconda3/envs/aligner/Scripts/mfa.exe'
```

When `mfa.executable` points into a conda environment, the wrapper puts that environment's folders on `PATH` for the child process,
which is what `conda activate` would do. On Windows the executable does not start without it.

### Download the English models (once)
```bash
python src/Step9_Forced_Alignment/Step9b_Run_MFA/run_mfa.py --download-models
```
This fetches `mfa.acoustic_model` and `mfa.dictionary` (both `english_us_arpa`) and needs network access once.

### Run it
```bash
python src/Step9_Forced_Alignment/Step9b_Run_MFA/run_mfa.py
python src/Step9_Forced_Alignment/Step9b_Run_MFA/run_mfa.py --dry-run                      # print the command only
python src/Step9_Forced_Alignment/Step9b_Run_MFA/run_mfa.py --corpus DIR --output DIR      # a trial on a few files
```
The settings (`num_jobs`, `single_speaker`, `clean`, `output_format`) are in `config.yaml`. `single_speaker` is on because the data
carries no speaker information. `clean` discards the cache of a previous run, so a repeated run starts from nothing.

### The beam
MFA's default beam (10, retry 40) is narrow. On a trial of 20 cohort recordings it **could not align 8 of them** ("errors on 8"; they get no
TextGrid). With `beam: 100` and `retry_beam: 400`, the values in `config.yaml`, all 20 were aligned. The 8 that failed with the default have
alignment scores in the same range as the other 12 (Step 9c), so the cause was the beam, not the audio. A wider beam makes the search
slower and more permissive: it will also force an alignment through a transcript that does not match the audio, so the MFA scores of
Step 9c matter. Set a key to `null` to use MFA's own default.

### Why there is no `mfa validate`
With an acoustic model, `mfa validate` also trains a monophone model on the corpus. That takes as long as a small alignment and is not
needed to align. What matters is checked by Step 9c: missing TextGrids, word counts and the words MFA does not know.

### Things to know
* **Fillers and unknown words.** The English dictionary has `uh`, `um`, `hmm`, `er`, `ah` and `oh`, and does not have `uhm`, `mm`, `mhm` or
  words such as `stepstool`. MFA 3.3 does not write `<unk>` for them: it keeps the spelling in the word tier and gives the word one noise
  phone, `spn`. They are aligned (checked on 5 recordings), but their boundaries are only as good as a noise model, so the pauses next to
  them are the least reliable. Step 9c reads the phone tier, treats a word whose phones are all `spn` as `<unk>`, counts them and lists them.
* **Runtime** was not measured on the full cohort. The trial of 20 recordings with the default beam took under 2 minutes, set-up included, on 6 jobs.
  The wide beam was not timed. Time a trial on a few dozen recordings before the full run.
* **The tiers** of MFA 3.3.9 are `words` and `phones` (`mfa.word_tier`, `mfa.phone_tier`). Another version may name them differently; the
  reader then reports the tier names it found.
* **Alignment quality on disfluent speech must be checked** before any result is trusted (Step 9c, and a few TextGrids in Praat).
