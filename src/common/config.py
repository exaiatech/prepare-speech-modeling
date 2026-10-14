r"""Configuration loading and path resolution.

Every script in the pipeline reads its settings from the repository-level
config.yaml through this module, so that a clone of the repository runs
without anyone editing a path.

PATH RESOLUTION
    Paths in config.yaml are RELATIVE TO THE REPOSITORY ROOT (the directory
    holding config.yaml and data/, one level above src/), not to the script or
    to the current working directory. resolve_path() turns them into
    absolute paths, so `python src/Step2_Acoustic/.../extract_recording_level.py`
    behaves the same from any directory.

    An absolute path in config.yaml is passed through untouched, which is how
    someone keeps the audio on another drive.

LOCAL OVERRIDES
    If config.local.yaml exists next to config.yaml, it is merged over the
    committed configuration (recursively, key by key). It is listed in
    .gitignore. This is the supported way to point at data held elsewhere
    without dirtying the shared config:

        # config.local.yaml
        paths:
          raw_audio:
            train: 'D:\prepare\train_audios'

    Nothing else in the pipeline knows about local overrides; they are applied
    once, here, at load time.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
LOCAL_CONFIG = REPO_ROOT / "config.local.yaml"


def _deep_merge(base, override):
    """Recursively merges `override` into `base`, returning a new dict."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path=DEFAULT_CONFIG):
    """Reads config.yaml, then applies config.local.yaml if it exists."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if LOCAL_CONFIG.exists():
        with open(LOCAL_CONFIG, "r", encoding="utf-8") as f:
            local = yaml.safe_load(f) or {}
        config = _deep_merge(config, local)

    return config


def resolve_path(path_value):
    """Resolves a config path value against the repository root.

    Relative values become <repo>/<value>; absolute values are returned as
    they are. Returns a pathlib.Path.
    """
    p = Path(path_value)
    return p if p.is_absolute() else REPO_ROOT / p


def ensure_dir(path_value):
    """Resolves a config path and creates the directory if it is missing."""
    p = resolve_path(path_value)
    p.mkdir(parents=True, exist_ok=True)
    return p
