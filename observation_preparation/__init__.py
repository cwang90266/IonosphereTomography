"""Observation preparation package.

Public entry points are exported here so these imports work from the project root:

    from observation_preparation import prepare_ro_observations
    from observation_preparation import prepare_igs_observations
"""

from .ro_source import (
    prepare_ro_observations,
    scan_ro_metadata,
    filter_ro_metadata,
    select_ro_occultations,
)
from .igs_source import (
    prepare_igs_observations,
    filter_igs_by_time,
    filter_igs_by_roi,
    filter_igs_epochs_by_roi,
    filter_igs_epochs_by_full_los,
    collapse_igs_arc_to_central_epoch,
)
from .roi import (
    DEFAULT_FIBONACCI_SPACING_DEG,
    DEFAULT_FIBONACCI_SPACING_KM,
    circular_roi_points,
    geodesic_circle_latlon,
)
from .roi_selection import haversine_km, los_within_roi, build_roi_dict
from .schema import ObservationEntry
from .netcdf_io import write_observations, read_observations
from .diagnostics import (
    plot_geolocation,
    plot_obs_detail,
    pooled_fraction_inside_roi,
    plot_los_penetration,
    plot_tec_comparison,
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
    "filter_igs_epochs_by_full_los",
    "collapse_igs_arc_to_central_epoch",
    "DEFAULT_FIBONACCI_SPACING_DEG",
    "DEFAULT_FIBONACCI_SPACING_KM",
    "circular_roi_points",
    "geodesic_circle_latlon",
    "haversine_km",
    "los_within_roi",
    "build_roi_dict",
    "ObservationEntry",
    "write_observations",
    "read_observations",
    "plot_geolocation",
    "plot_obs_detail",
    "pooled_fraction_inside_roi",
    "plot_los_penetration",
    "plot_tec_comparison",
]
