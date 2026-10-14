# Step 3a: Feature Inventory

### Purpose
Generates the documentation table for all 135 biomarkers: name, domain, unit, extraction library, and the parameters used.

The inventory is **not** maintained by hand. The script runs the real extractor on a synthetic vowel (a 120 Hz harmonic complex with a silent gap) and reads the feature keys back out of `extract_all()`. If a feature is renamed, added or removed in `common/feature_functions.py` and the inventory is not updated to match, this script fails rather than emitting a stale table. That is what keeps `docs/feature_inventory.md` from drifting away from the code.

### Inputs
* **Required Files:** None. The probe signal is synthesised in memory; no dataset audio is read.
* **Parameters:** The probe is built at `features.target_sr`.

### Outputs
* **Generated Outputs:** The biomarker inventory, in two formats.
* **File Format:** `.csv` and `.md`
* **Location:** `docs/feature_inventory.csv`, `docs/feature_inventory.md`
* **Structure:** One row per feature: name, domain, description, unit, library, extraction parameters. The Markdown version is grouped by domain for reading.

### Configuration
* `paths.docs_dir`, `reporting.feature_inventory_csv`, `reporting.feature_inventory_md`.
* Domain assignment and unit semantics are declared in the script itself, since they are documentation facts rather than tunable parameters.

### Dependencies
* `librosa`, `praat-parselmouth`, `webrtcvad`, `soundfile`, `numpy`, `pandas`, `scipy`, `pyyaml`.

### Execution Instructions
```bash
python src/Step3_Feature_QC/Step3a_Feature_Inventory/build_inventory.py
```
Takes a few seconds: it runs the full extractor once, on three seconds of synthetic audio.
