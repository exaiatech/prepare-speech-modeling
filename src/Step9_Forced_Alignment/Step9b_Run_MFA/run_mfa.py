"""
Step 9b - Run the Montreal Forced Aligner on the corpus written by Step 9a.

This is a thin wrapper around `mfa align`. It adds no alignment logic: it only reads the
settings from config.yaml (mfa:), finds the MFA executable, makes the MFA environment usable
and checks that every recording got a TextGrid. MFA itself is installed separately with conda
(see README.md in this folder).

WHY A WRAPPER
    The same command is then run the same way by everyone, and on Windows the MFA executable
    only starts when the Library\bin folder of its conda environment is on PATH. The wrapper
    puts it there when mfa.executable points into a conda environment, so no activation is needed.

WHAT IT DOES NOT DO
    It does not run `mfa validate`. With an acoustic model, validate also trains a monophone
    model on the corpus, which costs as much as a small alignment and is not needed to align.
    The checks that matter (missing TextGrids, word counts, <unk>) are done by Step 9c.

INPUTS
    mfa.corpus_dir                 <ID>.wav + <ID>.txt from Step 9a
    mfa.acoustic_model, .dictionary    MFA models (download once with --download-models)

OUTPUTS
    mfa.textgrid_dir               one <ID>.TextGrid per recording

HOW TO RUN
    python src/Step9_Forced_Alignment/Step9b_Run_MFA/run_mfa.py --download-models   # once
    python src/Step9_Forced_Alignment/Step9b_Run_MFA/run_mfa.py
    python src/Step9_Forced_Alignment/Step9b_Run_MFA/run_mfa.py --dry-run            # print the command only
    python src/Step9_Forced_Alignment/Step9b_Run_MFA/run_mfa.py --corpus DIR --output DIR   # a trial on a few files
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import load_config, resolve_path


def find_mfa(configured):
    """Returns the path of the mfa executable, or exits with a clear message."""
    exe = resolve_path(configured) if configured else None
    if exe is not None and not Path(exe).exists():
        sys.exit(f"mfa.executable does not exist: {exe}")
    exe = str(exe) if exe is not None else shutil.which("mfa")
    if exe is None:
        sys.exit("MFA was not found. Install it (see README.md in this folder) and either put `mfa` on the "
                 "PATH or set mfa.executable in config.local.yaml.")
    return exe


def mfa_environment(exe):
    """Environment variables for the child process.

    When the executable sits in a conda environment, the environment's own folders go first on
    PATH. That is what `conda activate` would do, and the Windows executable needs it.
    """
    env = dict(os.environ)
    exe_dir = Path(exe).resolve().parent                    # .../envs/<name>/Scripts or .../bin
    root = exe_dir.parent
    if os.name == "nt":
        extra = [root, root / "Library" / "bin", root / "Scripts", root / "Library" / "mingw-w64" / "bin"]
    else:
        extra = [exe_dir]
    env["PATH"] = os.pathsep.join(str(p) for p in extra if p.exists()) + os.pathsep + env.get("PATH", "")
    return env


def align_command(exe, mcfg, corpus, output):
    cmd = [exe, "align", str(corpus), mcfg["dictionary"], mcfg["acoustic_model"], str(output),
           "-j", str(mcfg["num_jobs"]), "--output_format", mcfg["output_format"]]
    if mcfg["single_speaker"]:
        cmd.append("--single_speaker")
    if mcfg["clean"]:
        cmd.append("--clean")
    if mcfg.get("beam"):
        cmd += ["--beam", str(mcfg["beam"])]
    if mcfg.get("retry_beam"):
        cmd += ["--retry_beam", str(mcfg["retry_beam"])]
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download-models", action="store_true", help="download the acoustic model and dictionary, then stop")
    parser.add_argument("--dry-run", action="store_true", help="print the command and stop")
    parser.add_argument("--corpus", help="corpus directory, instead of mfa.corpus_dir")
    parser.add_argument("--output", help="output directory, instead of mfa.textgrid_dir")
    args = parser.parse_args()

    mcfg = load_config()["mfa"]
    exe = find_mfa(mcfg["executable"])
    env = mfa_environment(exe)

    if args.download_models:
        for kind, name in (("acoustic", mcfg["acoustic_model"]), ("dictionary", mcfg["dictionary"])):
            cmd = [exe, "model", "download", kind, name]
            print(" ".join(cmd))
            subprocess.run(cmd, env=env, check=True)
        return

    corpus = Path(args.corpus) if args.corpus else resolve_path(mcfg["corpus_dir"])
    output = Path(args.output) if args.output else resolve_path(mcfg["textgrid_dir"])
    stems = sorted(p.stem for p in corpus.glob("*.txt"))
    if not stems:
        sys.exit(f"No transcripts in {corpus}. Run Step9a_Prepare_Corpus first.")
    output.mkdir(parents=True, exist_ok=True)

    cmd = align_command(exe, mcfg, corpus, output)
    print(f"{len(stems)} recordings in {corpus}")
    print(" ".join(cmd))
    if args.dry_run:
        return
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(f"mfa align stopped with exit code {result.returncode}")

    done = {p.stem for p in output.glob("*.TextGrid")}
    missing = [s for s in stems if s not in done]
    print(f"\n{len(done & set(stems))} of {len(stems)} recordings have a TextGrid in {output}")
    if missing:
        print(f"{len(missing)} have none (MFA could not align them), for example: {', '.join(missing[:10])}")
    print("Next: Step9c_Alignment_QC/alignment_qc.py")


if __name__ == "__main__":
    main()
