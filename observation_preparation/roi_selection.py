"""Shared ROI-selection primitives used by both ro_source.py and igs_source.py.

``haversine_km`` deduplicated from what was previously copy-pasted verbatim
in both prepare_ro_observations.py and prepare_igs_observations.py (Plan
Section 2.4/10 step 2). ``los_within_roi`` is the full-LOS ROI containment
check (Plan Section 5.2/10 step 3).
"""
from __future__ import annotations

import numpy as np
import pyproj

_EARTH_RADIUS_KM = 6371.0

# Mirrors the transform EDPSamples.get_observation_operator uses internally
# (Plan Section 5.1) -- built once at import time since pyproj.Transformer
# construction is not free and this module has no state to hang it on.
_ECEF_TO_GEODETIC = pyproj.Transformer.from_crs(
    pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
    pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
    always_xy=True,
)


def haversine_km(lat0: float, lon0: float, lat, lon) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    p0 = np.deg2rad(float(lat0))
    p1 = np.deg2rad(lat)
    dphi = p1 - p0
    dlambda = np.deg2rad(lon - float(lon0))

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(p0) * np.cos(p1) * np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def sample_ray_geodetic(
    rec_col: np.ndarray,
    gnss_col: np.ndarray,
    num_segments: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``num_segments`` points along one ray's straight ECEF-km
    chord (receiver <-> transmitter), mirroring
    ``EDPSamples.get_observation_operator``'s own ray sampling (Plan
    Section 5.1). Shared by ``los_within_roi`` and
    ``diagnostics.plot_los_penetration`` so both use the same geometry.

    Parameters
    ----------
    rec_col, gnss_col : ndarray, shape (3,)
        One ray's receiver / transmitter ECEF position, km.

    Returns
    -------
    lats, lons, alts_km : ndarray, shape (num_segments,)
    """
    t = np.linspace(0.0, 1.0, int(num_segments))
    points = gnss_col[:, None] + (rec_col[:, None] - gnss_col[:, None]) * t
    lons, lats, alts_m = _ECEF_TO_GEODETIC.transform(
        points[0, :] * 1e3, points[1, :] * 1e3, points[2, :] * 1e3
    )
    alts_km = np.asarray(alts_m, dtype=np.float64) / 1000.0
    return np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64), alts_km


def los_within_roi(
    rec_ecef_km: np.ndarray,
    gnss_ecef_km: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    alt_limit_km: float,
    num_segments: int = 200,
    fraction_required: float = 1.0,
) -> np.ndarray:
    """Full-LOS ROI containment check (Plan Section 5.2, objective 3).

    For each ray (one receiver/transmitter ECEF-km column pair), samples
    ``num_segments`` points along the straight chord between them --
    mirroring ``EDPSamples.get_observation_operator``'s own ray sampling
    (Section 5.1) so "is this ray inside the ROI" and "does this ray
    contribute to H" stay consistent by construction -- restricts to the
    points at or below ``alt_limit_km``, and requires at least
    ``fraction_required`` of those (not necessarily all) to fall within
    ``radius_km`` of ``(center_lat, center_lon)``.

    A ray with *no* sample point at or below ``alt_limit_km`` at all (e.g.
    an occultation whose tangent altitude never gets that low) fails this
    check -- there's nothing to confirm containment of, so it can't satisfy
    a fraction-of-the-sub-altitude-limit-LOS rule, and failing closed is the
    safer default for a containment gate.

    Parameters
    ----------
    rec_ecef_km, gnss_ecef_km : ndarray, shape (3, n_rays)
        Receiver (LEO or ground station) / transmitter ECEF positions, km.
    alt_limit_km : float
        Required, user-supplied (Plan Section 8.1) -- no per-ray-batch
        auto-default, since receiver altitude varies across rays.
    fraction_required : float, default 1.0
        Confirmed first-class, tunable parameter (Plan Section 8.2), not
        just an escape hatch.

    Returns
    -------
    keep : ndarray, shape (n_rays,), bool
    """
    rec = np.asarray(rec_ecef_km, dtype=np.float64)
    gnss = np.asarray(gnss_ecef_km, dtype=np.float64)
    n_rays = rec.shape[1]
    keep = np.zeros(n_rays, dtype=bool)

    for i in range(n_rays):
        lats, lons, alts_km = sample_ray_geodetic(rec[:, i], gnss[:, i], num_segments)
        below = alts_km <= float(alt_limit_km)
        if not np.any(below):
            continue

        dist_km = haversine_km(center_lat, center_lon, lats[below], lons[below])
        keep[i] = float(np.mean(dist_km <= radius_km)) >= float(fraction_required)

    return keep


def build_roi_dict(
    center_lat: float | None,
    center_lon: float | None,
    radius_km: float | None,
    mode: str,
    *,
    alt_limit_km: float | None = None,
    fraction_required: float | None = None,
) -> dict | None:
    """The ``roi=`` dict ``netcdf_io.write_observations`` expects (Plan
    Section 6.2), or None when no ROI was requested."""
    if center_lat is None or center_lon is None or radius_km is None:
        return None
    roi = {
        "roi_center_lat": float(center_lat),
        "roi_center_lon": float(center_lon),
        "roi_radius_km": float(radius_km),
        "roi_mode": mode,
    }
    if alt_limit_km is not None:
        roi["roi_alt_limit_km"] = float(alt_limit_km)
    if fraction_required is not None:
        roi["roi_fraction_required"] = float(fraction_required)
    return roi


__all__ = [
    "haversine_km",
    "sample_ray_geodetic",
    "los_within_roi",
    "build_roi_dict",
]
