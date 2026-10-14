"""
Step 2a - Inspect the WebRTC VAD segmentation of individual recordings.

DIAGNOSTIC TOOL, NOT PART OF THE MODELING PATH. Retained, per the supervisor's
review, for checking questionable pause results. It does not need to produce a
modeling dataset.

This is a diagnostic entry point, not a pipeline stage: nothing downstream
reads its output. The VAD itself lives in common/vad.py and is applied inside
the feature extractors of Step 2b, where it produces the 18 temporal/fluency
features of the 135-feature set.

Use this script to see why a recording produced the pause statistics it did,
for example one of the recordings that report zero pauses (see
docs/open_questions.md).

INPUTS
    audio file paths given on the command line (any format librosa reads)
    config.yaml   vad.aggressiveness, vad.frame_ms, minimum run lengths

OUTPUTS
    a per-file segmentation summary printed to stdout. No files are written.

HOW TO RUN
    python src/Step2_Acoustic/Step2a_VAD_Segmentation/vad_segment.py data/raw/train_audios/xxxx.wav
    python src/Step2_Acoustic/Step2a_VAD_Segmentation/vad_segment.py --config config.yaml a.wav b.wav
"""

import argparse
import sys
from pathlib import Path

import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import DEFAULT_CONFIG, load_config
from common.vad import pause_features


def main(audio_paths, config_path=DEFAULT_CONFIG):
    """Prints the VAD segmentation summary for each given audio file."""
    config = load_config(config_path)
    cfg = config["vad"]

    if not audio_paths:
        print("No audio files given. Usage: python vad_segment.py FILE [FILE ...]")
        return

    for path in audio_paths:
        y, _ = librosa.load(path, sr=cfg["sample_rate"], mono=True)
        feats = pause_features(y, cfg)
        print(f"\n{path}")
        print("=" * len(str(path)))
        for k, v in feats.items():
            print(f"  {k:<30}{v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="*", help="audio files to segment")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()
    main(args.audio, args.config)
