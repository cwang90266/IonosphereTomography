"""Tests for diagnostics.py (Plan Section 7/10 step 4).

Synthetic entries + a small synthetic EDPSamples instance -- no real
RO/IGS data required, mirroring EDPSamples' own analytic radial-integral
test pattern for the TEC-comparison check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import pyproj

from observation_preparation.schema import ObservationEntry
from observation_preparation.roi_selection import build_roi_dict
from observation_preparation import diagnostics

edp_samples_module = pytest.importorskip(
    "edp_samples", reason="edp_samples.py and its dependencies (pyproj, cartopy, ...) not importable"
)
EDPSamples = edp_samples_module.EDPSamples

_GEODETIC_TO_ECEF = pyproj.Transformer.from_crs(
    pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
    pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
    always_xy=True,
)

CENTER_LAT, CENTER_LON = 69.6, 19.2


def _ecef_km(lat, lon, alt_km):
    x, y, z = _GEODETIC_TO_ECEF.transform(lon, lat, alt_km * 1000.0)
    return np.array([x, y, z]) / 1000.0


def _ro_entry(n_rays=6, label="ro_a", near_center=True):
    lat, lon = (CENTER_LAT, CENTER_LON) if near_center else (-CENTER_LAT, CENTER_LON + 180.0)
    alts = np.linspace(700.0, 100.0, n_rays)
    rec = np.column_stack([_ecef_km(lat, lon, a) for a in alts])
    gnss = np.column_stack([_ecef_km(lat + 5.0, lon + 0.5, 20200.0) for _ in alts])
    return ObservationEntry(
        obs_type="RO",
        tec=np.linspace(40.0, 5.0, n_rays),
        rec_ecef_km=rec,
        gnss_ecef_km=gnss,
        tangent_alt_km=alts,
        label=label,
        date=pd.Timestamp("2025-11-18 10:15:00"),
        abel={"Ne": np.linspace(1e11, 1e12, 10), "alt_km": np.linspace(100.0, 900.0, 10)},
    )


def _igs_entry(n_rays=1, label="igs_a"):
    rec = np.column_stack([_ecef_km(CENTER_LAT, CENTER_LON, -100.0) for _ in range(n_rays)])
    gnss = np.column_stack([_ecef_km(CENTER_LAT + 5.0, CENTER_LON, 20200.0) for _ in range(n_rays)])
    return ObservationEntry(
        obs_type="IGS",
        tec=np.full(n_rays, 15.0),
        rec_ecef_km=rec,
        gnss_ecef_km=gnss,
        pierce_lat=np.full(n_rays, CENTER_LAT),
        pierce_lon=np.full(n_rays, CENTER_LON),
        elev_deg=np.full(n_rays, 45.0),
        label=label,
        date=pd.Timestamp("2025-11-18 10:30:00"),
    )


def test_plot_geolocation_writes_file(tmp_path):
    entries = [_ro_entry(), _igs_entry()]
    roi = build_roi_dict(CENTER_LAT, CENTER_LON, 2000.0, "full_los", alt_limit_km=800.0)
    path = diagnostics.plot_geolocation(entries, roi=roi, output_path=tmp_path / "geo.png")
    assert path.exists() and path.stat().st_size > 0


def test_plot_geolocation_no_roi_no_igs(tmp_path):
    path = diagnostics.plot_geolocation([_ro_entry()], output_path=tmp_path / "geo2.png")
    assert path.exists()


def test_plot_obs_detail_ro_with_abel(tmp_path):
    path = diagnostics.plot_obs_detail(_ro_entry(), output_path=tmp_path / "detail.png")
    assert path.exists()


def test_plot_obs_detail_igs_no_abel(tmp_path):
    path = diagnostics.plot_obs_detail(_igs_entry(), output_path=tmp_path / "detail_igs.png")
    assert path.exists()


def test_pooled_fraction_inside_roi_matches_geometry():
    near = _ro_entry(near_center=True)
    far = _ro_entry(near_center=False, label="ro_far")

    f_near = diagnostics.pooled_fraction_inside_roi(near, CENTER_LAT, CENTER_LON, 2000.0, alt_limit_km=800.0)
    f_far = diagnostics.pooled_fraction_inside_roi(far, CENTER_LAT, CENTER_LON, 2000.0, alt_limit_km=800.0)

    assert f_near == pytest.approx(1.0, abs=0.05)
    assert f_far == pytest.approx(0.0, abs=0.05)


def test_pooled_fraction_inside_roi_none_when_never_below_alt_limit():
    entry = _ro_entry()
    # alt_limit_km far below every sampled point's altitude -> nothing to pool.
    frac = diagnostics.pooled_fraction_inside_roi(entry, CENTER_LAT, CENTER_LON, 2000.0, alt_limit_km=-500.0)
    assert frac is None


def test_plot_los_penetration_writes_file(tmp_path):
    entries = [_ro_entry(near_center=True), _ro_entry(near_center=False, label="ro_far")]
    path = diagnostics.plot_los_penetration(
        entries, CENTER_LAT, CENTER_LON, 2000.0, alt_limit_km=800.0, output_path=tmp_path / "pen.png",
    )
    assert path.exists()


def _make_sampling_parameters(n_sample: int) -> pd.DataFrame:
    """Mirrors EDPSamples/test_edp_samples.py's own helper of the same
    name (defined there, not in edp_samples.py itself)."""
    sp = pd.DataFrame({
        "hour": [12.0] * n_sample, "f107": [120.0] * n_sample,
        "ap": [10.0] * n_sample, "ig12": [80.0] * n_sample, "rz12": [70.0] * n_sample,
    })
    sp.attrs = {}
    return sp


def test_plot_tec_comparison_matches_analytic_radial_integral(tmp_path):
    """Mirrors EDPSamples/test_edp_samples.py's own analytic check: a
    uniform-density vertical ray's predicted TEC has a closed form, so this
    validates plot_tec_comparison's H @ density computation quantitatively,
    not just that it runs."""
    n0 = 1e11
    altitude = np.linspace(0.0, 1000.0, 101)
    sp = _make_sampling_parameters(1)
    edps = np.full((len(altitude), 1, 1), n0)
    feature_edps = np.zeros((len(EDPSamples.FEATURE_LABEL), 1, 1))
    ds = EDPSamples(
        DateTime="2026-01-01", geo_type="Point", altitude=altitude, sampling_parameters=sp,
        Lon=0.0, Lat=0.0, edps=edps, feature_edps=feature_edps,
    )

    gnss = (edp_samples_module._geodetic_to_ecef(0.0, 0.0, 0.0) / 1000.0).reshape(3, 1)
    rec = (edp_samples_module._geodetic_to_ecef(0.0, 0.0, 1000.0 * 1000.0) / 1000.0).reshape(3, 1)
    # Analytic: the whole 0->1000 km radial ray lies inside the altitude
    # table, at uniform density n0, so TEC = n0 * 1,000,000 m / 1e16.
    expected_tec = n0 * 1_000_000.0 / 1e16
    entry = ObservationEntry(
        obs_type="RO", tec=np.array([expected_tec]),
        rec_ecef_km=rec, gnss_ecef_km=gnss,
        tangent_alt_km=np.array([500.0]), label="radial",
    )

    (fig_or_path, stats) = diagnostics.plot_tec_comparison(
        [entry], ds, num_segments=5000, output_path=tmp_path / "tec.png",
    )
    assert fig_or_path.exists()
    assert stats["RO"]["n"] == 1
    assert stats["RO"]["bias"] == pytest.approx(0.0, abs=0.05)
    assert stats["RO"]["rms"] == pytest.approx(0.0, abs=0.05)
