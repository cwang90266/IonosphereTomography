"""Unit tests for roi_selection.py's los_within_roi (Plan Section 5.2/10
step 3). Pure-geometry synthetic rays -- no real RO/IGS data required.
"""
from __future__ import annotations

import numpy as np
import pyproj

from observation_preparation.roi_selection import los_within_roi, build_roi_dict

_GEODETIC_TO_ECEF = pyproj.Transformer.from_crs(
    pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
    pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
    always_xy=True,
)

CENTER_LAT, CENTER_LON = 69.6, 19.2


def _ecef_km(lat, lon, alt_km):
    x, y, z = _GEODETIC_TO_ECEF.transform(lon, lat, alt_km * 1000.0)
    return np.array([x, y, z]) / 1000.0


def _antipode(lat, lon):
    return -lat, ((lon + 180.0 + 180.0) % 360.0) - 180.0


def test_radial_ray_at_center_is_fully_inside():
    """Receiver and transmitter both directly above the ROI center: every
    sampled point along the chord stays near (center_lat, center_lon)."""
    rec = _ecef_km(CENTER_LAT, CENTER_LON, 700.0).reshape(3, 1)
    gnss = _ecef_km(CENTER_LAT, CENTER_LON, 20200.0).reshape(3, 1)

    keep = los_within_roi(
        rec, gnss, CENTER_LAT, CENTER_LON, radius_km=50.0, alt_limit_km=800.0,
    )
    assert keep.shape == (1,)
    assert bool(keep[0]) is True


def test_ray_never_reaching_alt_limit_fails_closed():
    """Neither endpoint (nor anything between two high, radially-aligned
    points) ever dips below alt_limit_km -- there's nothing to confirm
    containment of, so this must be rejected, not vacuously accepted."""
    rec = _ecef_km(CENTER_LAT, CENTER_LON, 5000.0).reshape(3, 1)
    gnss = _ecef_km(CENTER_LAT, CENTER_LON, 20200.0).reshape(3, 1)

    keep = los_within_roi(
        rec, gnss, CENTER_LAT, CENTER_LON, radius_km=50.0, alt_limit_km=1000.0,
    )
    assert bool(keep[0]) is False


def test_ray_toward_antipode_is_rejected_outside_roi():
    """Receiver above the ROI center, transmitter above the antipode: the
    transmitter-side footprint is roughly 20000 km from the center, far
    outside any reasonable ROI radius."""
    anti_lat, anti_lon = _antipode(CENTER_LAT, CENTER_LON)
    rec = _ecef_km(CENTER_LAT, CENTER_LON, 700.0).reshape(3, 1)
    gnss = _ecef_km(anti_lat, anti_lon, 20200.0).reshape(3, 1)

    keep = los_within_roi(
        rec, gnss, CENTER_LAT, CENTER_LON, radius_km=2000.0, alt_limit_km=800.0,
        fraction_required=1.0,
    )
    assert bool(keep[0]) is False


def test_fraction_required_is_tunable():
    """A ray split ~evenly between the center's footprint and the
    antipode's footprint (both endpoints at the same altitude, so the
    straight chord dips well below alt_limit_km along its whole length,
    symmetric about the midpoint): strict fraction_required rejects it,
    a lenient one accepts it."""
    anti_lat, anti_lon = _antipode(CENTER_LAT, CENTER_LON)
    rec = _ecef_km(CENTER_LAT, CENTER_LON, 700.0).reshape(3, 1)
    gnss = _ecef_km(anti_lat, anti_lon, 700.0).reshape(3, 1)

    strict = los_within_roi(
        rec, gnss, CENTER_LAT, CENTER_LON, radius_km=3000.0, alt_limit_km=700.0,
        fraction_required=0.9,
    )
    lenient = los_within_roi(
        rec, gnss, CENTER_LAT, CENTER_LON, radius_km=3000.0, alt_limit_km=700.0,
        fraction_required=0.3,
    )
    assert bool(strict[0]) is False
    assert bool(lenient[0]) is True


def test_multiple_rays_vectorized_over_columns():
    """(3, n_rays) input returns one bool per ray, independent of the others."""
    anti_lat, anti_lon = _antipode(CENTER_LAT, CENTER_LON)
    rec = np.column_stack([
        _ecef_km(CENTER_LAT, CENTER_LON, 700.0),
        _ecef_km(CENTER_LAT, CENTER_LON, 700.0),
    ])
    gnss = np.column_stack([
        _ecef_km(CENTER_LAT, CENTER_LON, 20200.0),
        _ecef_km(anti_lat, anti_lon, 20200.0),
    ])

    keep = los_within_roi(rec, gnss, CENTER_LAT, CENTER_LON, radius_km=50.0, alt_limit_km=800.0)
    assert keep.shape == (2,)
    assert bool(keep[0]) is True
    assert bool(keep[1]) is False


def test_build_roi_dict_includes_full_los_fields_only_when_given():
    roi = build_roi_dict(CENTER_LAT, CENTER_LON, 2000.0, "full_los")
    assert "roi_alt_limit_km" not in roi
    assert "roi_fraction_required" not in roi

    roi2 = build_roi_dict(
        CENTER_LAT, CENTER_LON, 2000.0, "full_los",
        alt_limit_km=800.0, fraction_required=0.9,
    )
    assert roi2["roi_alt_limit_km"] == 800.0
    assert roi2["roi_fraction_required"] == 0.9

    assert build_roi_dict(None, CENTER_LON, 2000.0, "full_los") is None
