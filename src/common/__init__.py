"""Shared, importable code for the PREPARE speech biomarker pipeline.

Nothing here is runnable. The step scripts under Step*/ hold the
orchestration; this package holds what more than one of them needs:

    config             config.yaml loading and repo-relative path resolution
    vad                WebRTC VAD pause statistics
    feature_functions  the 135 acoustic feature definitions
    selection          Cohen's d statistics and correlation pruning

Every step script reaches this package through the two-line bootstrap
documented in its own README:

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common.config import load_config
"""
