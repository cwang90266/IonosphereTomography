"""Observation preparation package.

Public entry points are exported here so these imports work from the project root:

    from observation_preparation import prepare_ro_observations
    from observation_preparation import prepare_igs_observations
"""

from .prepare_ro_observations import (
    prepare_ro_observations,
    scan_ro_metadata,
    filter_ro_metadata,
    select_ro_occultations,
)
from .prepare_igs_observations import (
    prepare_igs_observations,
    filter_igs_by_time,
    filter_igs_by_roi,
    filter_igs_epochs_by_roi,
    collapse_igs_arc_to_central_epoch,
)
from .roi_tools import (
    DEFAULT_FIBONACCI_SPACING_DEG,
    DEFAULT_FIBONACCI_SPACING_KM,
    circular_roi_points,
    geodesic_circle_latlon,
)

__all__ = [
    "prepare_ro_observations",
    "scan_ro_metadata",
    "filter_ro_metadata",
    "select_ro_occultations",
    "prepare_igs_observations",
    "filter_igs_by_time",
    "filter_igs_by_roi",
    "filter_igs_epochs_by_roi",
    "collapse_igs_arc_to_central_epoch",
    "DEFAULT_FIBONACCI_SPACING_DEG",
    "DEFAULT_FIBONACCI_SPACING_KM",
    "circular_roi_points",
    "geodesic_circle_latlon",
]
