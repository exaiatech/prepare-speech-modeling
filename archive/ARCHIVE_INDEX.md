# Archive Index

Everything under `archive/` is retained for reference and is not on any execution path.
`archive/legacy/` holds the folders that used to sit at the repository root. They were moved with `git mv`,
so their history is intact. Dates are the first and last commit that touched the original path.

## archive/legacy

| Original path | New path | Files | First commit | Last commit | What it was | Superseded by |
| --- | --- | --- | --- | --- | --- | --- |
| `ADReSSo Project` | `archive/legacy/ADReSSo_Project_spaced` | 11 | 2026-05-03 | 2026-05-03 | ASR pipeline scripts: audio preprocessing, Whisper transcription, alignment, verticalization, error analysis, with a data README. The folder name contains a space. | Same script file names as the ones script_migration.csv records as moved to src/Step4_ASR. Not byte-identical for every script, see the table below. |
| `ADReSSo_Project` | `archive/legacy/ADReSSo_Project` | 12 | 2026-05-17 | 2026-05-17 | The same ASR pipeline as the spaced folder, plus Alignment_App.html. | src/Step4_ASR (script_migration.csv rows for step1a_audio_preprocess.py, step2_*.py and Alignment_App.html). |
| `Prepare_Challenge` | `archive/legacy/Prepare_Challenge` | 11 | 2026-04-26 | 2026-04-26 | Metadata exploratory analysis: a summary script and a visualization script, with their CSV and PNG output. | unknown |
| `Project_Step1` | `archive/legacy/Project_Step1` | 6 | 2026-04-26 | 2026-04-26 | Audio preprocessing, audio visualization and audio evaluation scripts, each with a README. | unknown (a script named step1a_audio_preprocess.py also exists in src/Step4_ASR, see the table below) |
| `Project_Step2` | `archive/legacy/Project_Step2` | 8 | 2026-04-26 | 2026-04-26 | Speech-to-text, alignment, verticalization and error-analysis scripts, each with a README. | unknown (same script file names as src/Step4_ASR, see the table below; script_migration.csv does not record this folder) |
| `evaluation` | `archive/legacy/evaluation` | 5 | 2026-03-18 | 2026-03-27 | Three Whisper result spreadsheets (v1 raw, v2 VAD processed, v3 optimized decoding) and one gold-standard transcript. | unknown |

The same script names appear in several of these folders. The next table compares each archived script
with the file of the same name under `src/`.

## Archived scripts against the current files of the same name

Byte comparison ignores line endings.

| Archived file | Same-named file under src/ | Relation |
| --- | --- | --- |
| `archive/legacy/ADReSSo_Project/ASR evaluation/Step2b_Alignment/step2_text_alignment.py` | `src/Step4_ASR/Step4c_Alignment/step2_text_alignment.py` | differs |
| `archive/legacy/ADReSSo_Project/ASR evaluation/Step2c_Verticalization/step2_verticalization.py` | `src/Step4_ASR/Step4d_Verticalization/step2_verticalization.py` | differs |
| `archive/legacy/ADReSSo_Project/ASR evaluation/Step2d_Error_Analysis/step2_error_analysis.py` | `src/Step4_ASR/Step4e_Error_Analysis/step2_error_analysis.py` | differs |
| `archive/legacy/ADReSSo_Project/ASR transcription/step2_speech_to_text.py` | `src/Step4_ASR/Step4b_Transcription/step2_speech_to_text.py` | differs |
| `archive/legacy/ADReSSo_Project/Audio preprocessing/step1a_audio_preprocess.py` | `src/Step4_ASR/Step4a_Audio_Preprocessing/step1a_audio_preprocess.py` | differs |
| `archive/legacy/ADReSSo_Project_spaced/ASR evaluation/Step2b_Alignment/step2_text_alignment.py` | `src/Step4_ASR/Step4c_Alignment/step2_text_alignment.py` | differs |
| `archive/legacy/ADReSSo_Project_spaced/ASR evaluation/Step2c_Verticalization/step2_verticalization.py` | `src/Step4_ASR/Step4d_Verticalization/step2_verticalization.py` | differs |
| `archive/legacy/ADReSSo_Project_spaced/ASR evaluation/Step2d_Error_Analysis/step2_error_analysis.py` | `src/Step4_ASR/Step4e_Error_Analysis/step2_error_analysis.py` | differs |
| `archive/legacy/ADReSSo_Project_spaced/ASR transcription/step2_speech_to_text.py` | `src/Step4_ASR/Step4b_Transcription/step2_speech_to_text.py` | differs |
| `archive/legacy/ADReSSo_Project_spaced/Audio preprocessing/step1a_audio_preprocess.py` | `src/Step4_ASR/Step4a_Audio_Preprocessing/step1a_audio_preprocess.py` | differs |
| `archive/legacy/Prepare_Challenge/Metadata_Summary/Code/metadata_summary.py` | none | no file of this name under src/ |
| `archive/legacy/Prepare_Challenge/Metadata_Visualizations/Code/metadata_visualizations.py` | none | no file of this name under src/ |
| `archive/legacy/Project_Step1/step1a_audio_preprocess/step1a_audio_preprocess.py` | `src/Step4_ASR/Step4a_Audio_Preprocessing/step1a_audio_preprocess.py` | differs |
| `archive/legacy/Project_Step1/step1b_audio_visualize/step1b_audio_visualize.py` | none | no file of this name under src/ |
| `archive/legacy/Project_Step1/step1c_audio_evaluate/step1c_audio_evaluate.py` | none | no file of this name under src/ |
| `archive/legacy/Project_Step2/Step2a_Speech_to_Text/step2_speech_to_text.py` | `src/Step4_ASR/Step4b_Transcription/step2_speech_to_text.py` | differs |
| `archive/legacy/Project_Step2/Step2b_Alignment/step2_text_alignment.py` | `src/Step4_ASR/Step4c_Alignment/step2_text_alignment.py` | differs |
| `archive/legacy/Project_Step2/Step2c_Verticalization/step2_verticalization.py` | `src/Step4_ASR/Step4d_Verticalization/step2_verticalization.py` | differs |
| `archive/legacy/Project_Step2/Step2d_Error_Analysis/step2_error_analysis.py` | `src/Step4_ASR/Step4e_Error_Analysis/step2_error_analysis.py` | differs |

## Files that are identical between the legacy folders

These stay in place, one copy per folder, so that each folder remains complete.

* `archive/legacy/ADReSSo_Project/ASR evaluation/Step2b_Alignment/README.md.txt`, `archive/legacy/ADReSSo_Project_spaced/ASR evaluation/Step2b_Alignment/README.md.txt`, `archive/legacy/Project_Step2/Step2b_Alignment/README.md.txt`
* `archive/legacy/ADReSSo_Project/ASR evaluation/Step2b_Alignment/step2_text_alignment.py`, `archive/legacy/ADReSSo_Project_spaced/ASR evaluation/Step2b_Alignment/step2_text_alignment.py`
* `archive/legacy/ADReSSo_Project/ASR evaluation/Step2c_Verticalization/README.md.txt`, `archive/legacy/ADReSSo_Project_spaced/ASR evaluation/Step2c_Verticalization/README.md.txt`, `archive/legacy/Project_Step2/Step2c_Verticalization/README.md.txt`
* `archive/legacy/ADReSSo_Project/ASR evaluation/Step2d_Error_Analysis/README.md.txt`, `archive/legacy/ADReSSo_Project_spaced/ASR evaluation/Step2d_Error_Analysis/README.md.txt`, `archive/legacy/Project_Step2/Step2d_Error_Analysis/README.md.txt`
* `archive/legacy/ADReSSo_Project/ASR evaluation/Step2d_Error_Analysis/step2_error_analysis.py`, `archive/legacy/ADReSSo_Project_spaced/ASR evaluation/Step2d_Error_Analysis/step2_error_analysis.py`
* `archive/legacy/ADReSSo_Project/ASR transcription/README.md.txt`, `archive/legacy/ADReSSo_Project_spaced/ASR transcription/README.md.txt`, `archive/legacy/Project_Step2/Step2a_Speech_to_Text/README.md.txt`
* `archive/legacy/ADReSSo_Project/Audio preprocessing/README.md.txt`, `archive/legacy/ADReSSo_Project_spaced/Audio preprocessing/README.md.txt`, `archive/legacy/Project_Step1/step1a_audio_preprocess/README.md.txt`
* `archive/legacy/ADReSSo_Project/data/README.md.txt`, `archive/legacy/ADReSSo_Project_spaced/data/README.md.txt`

## Other archived material

| Path | What it is |
| --- | --- |
| `archive/dead_code/` | Superseded notebook cells, verbatim, with the reason for each. See its README. |
| `archive/notebook_original.ipynb` | The original whole-recording notebook. |
| `archive/features2_original.ipynb` | The original segment-level notebook. |
| `archive/set_aside_after_review/` | Scripts and tables set aside after the supervisor's review. See its README. |
