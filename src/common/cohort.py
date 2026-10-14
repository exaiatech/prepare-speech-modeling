"""
Cohort membership.

The shared cohort CSV supplied by the supervisor (paths.cohort) lists the
recordings that belong to the cohort. Its ID column (cohort.id_column) is the one
authoritative source of membership. Scripts that need to know whether a recording
is in the cohort read it through this module instead of re-deriving it from
diagnosis labels or metadata filters.

This module defines no CLI: it is imported, never run.
"""

import pandas as pd

from common.config import resolve_path


def load_cohort_ids(config):
    """Returns the set of recording IDs listed in the shared cohort CSV.

    A missing or repeated ID raises, because it would make the member count
    ambiguous.
    """
    id_col = config['cohort']['id_column']
    path = resolve_path(config['paths']['cohort'])
    ids = pd.read_csv(path, usecols=[id_col])[id_col]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError(f"{path}: the {id_col} column has missing or repeated values")
    return set(ids.astype(str))
