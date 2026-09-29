"""Round-trip test for schema.py + netcdf_io.py (Plan Section 10, step 1).

Uses synthetic entries -- no real RO/IGS data required -- since this
verifies the schema/netCDF layer itself, independent of the RO/IGS
scan/parse pipelines (Plan Section 10 sequencing).

Run with:
    python -m pytest observation_preparation/test_netcdf_schema.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from observation_preparation.schema import ObservationEntry
from observation_preparation import netcdf_io


def _ro_dict(n_rays=5, label="ro_event_1"):
    rng = np.random.default_rng(0)
    return {
        "tec": rng.uniform(1.0, 50.0, n_rays),
        "tangent_km": np.linspace(600.0, 100.0, n_rays),
        "LEO": rng.uniform(6800.0, 7200.0, (3, n_rays)),
        "GNSS": rng.uniform(20000.0, 26000.0, (3, n_rays)),
        "tec_type": "absolute",
        "leo_id": "LEO1",
        "prn_id": "G12",
        "label": label,
        "obs_source": "RO_podTc2",
        "date": pd.Timestamp("2025-11-18 10:15:00"),
        "lat_tecmax_tangent": 69.5,
        "lon_tecmax_tangent": 19.1,
        "occ_type": "setting",
        "snr_l1": rng.uniform(100.0, 500.0, n_rays),
        "snr_l2": rng.uniform(80.0, 400.0, n_rays),
        "abel": {
            "Ne": rng.uniform(1e10, 1e12, 20),
            "alt_km": np.linspace(100.0, 900.0, 20),
            "TEC_cal": rng.uniform(1.0, 50.0, 20),
            "TEC_forward": rng.uniform(1.0, 50.0, 20),
        },
    }


def _igs_dict(n_rays=1, label="TRO1_G05"):
    rng = np.random.default_rng(1)
    return {
        "tec": rng.uniform(5.0, 30.0, n_rays),
        "tangent_km": np.full(n_rays, 350.0),
        "LEO": rng.uniform(-2500.0, -2000.0, (3, n_rays)),
        "GNSS": rng.uniform(20000.0, 26000.0, (3, n_rays)),
        "tec_type": "absolute",
        "rx_id": "TRO",
        "prn_id": "G05",
        "label": label,
        "obs_source": "IGS_ground",
        "date": pd.Timestamp("2025-11-18 10:30:00"),
        "elev_deg": rng.uniform(20.0, 80.0, n_rays),
        "ipp_lat": rng.uniform(68.0, 71.0, n_rays),
        "ipp_lon": rng.uniform(17.0, 21.0, n_rays),
        "arc_time_sec": np.arange(n_rays, dtype=float),
        "time_utc_h": np.full(n_rays, 10.5),
        "station_lat": 69.66,
        "station_lon": 18.94,
    }


def test_from_dict_renames_legacy_keys():
    entry = ObservationEntry.from_dict(_ro_dict(), obs_type="RO")
    assert entry.rec_ecef_km.shape == (3, 5)
    assert entry.gnss_ecef_km.shape == (3, 5)
    assert entry.tangent_alt_km is not None
    assert "LEO" not in entry.extra and "GNSS" not in entry.extra
    # Unmodeled RO-specific fields preserved in extra.
    assert "lat_tecmax_tangent" in entry.extra
    assert entry.rec_id == "LEO1"


def test_from_dict_igs_rx_id_maps_to_rec_id():
    entry = ObservationEntry.from_dict(_igs_dict(), obs_type="IGS")
    assert entry.rec_id == "TRO"
    assert entry.pierce_lat is not None and entry.pierce_lon is not None
    assert "arc_time_sec" in entry.extra


def test_roundtrip_mixed_ro_and_igs(tmp_path):
    entries = [
        ObservationEntry.from_dict(_ro_dict(n_rays=6, label="ro_a"), obs_type="RO"),
        ObservationEntry.from_dict(_ro_dict(n_rays=3, label="ro_b"), obs_type="RO"),
    ]
    roi = {
        "roi_center_lat": 69.6, "roi_center_lon": 19.2, "roi_radius_km": 2000.0,
        "roi_mode": "full_los", "roi_alt_limit_km": 800.0, "roi_fraction_required": 1.0,
    }
    path = netcdf_io.write_observations(entries, tmp_path / "ro_test.nc", roi=roi)
    assert path.exists()

    back = netcdf_io.read_observations(path)
    assert len(back) == 2
    for orig, loaded in zip(entries, back):
        assert loaded.obs_type == "RO"
        assert loaded.label == orig.label
        assert loaded.n_rays == orig.n_rays
        np.testing.assert_allclose(loaded.tec, orig.tec, rtol=1e-5)
        np.testing.assert_allclose(loaded.rec_ecef_km, orig.rec_ecef_km, rtol=1e-9)
        np.testing.assert_allclose(loaded.gnss_ecef_km, orig.gnss_ecef_km, rtol=1e-9)
        np.testing.assert_allclose(loaded.tangent_alt_km, orig.tangent_alt_km, rtol=1e-5)
        np.testing.assert_allclose(loaded.snr_l1, orig.snr_l1, rtol=1e-5)
        assert loaded.date == orig.date
        assert loaded.occ_type == orig.occ_type
        assert loaded.rec_id == orig.rec_id
        assert loaded.prn_id == orig.prn_id

        # Abel profile round-trips without truncation to n_rays.
        np.testing.assert_allclose(loaded.abel["Ne"], orig.abel["Ne"], rtol=1e-5)
        assert len(loaded.abel["alt_km"]) == len(orig.abel["alt_km"]) == 20

        # extra scalar/array pass-through.
        assert loaded.extra["lat_tecmax_tangent"] == pytest.approx(orig.extra["lat_tecmax_tangent"])


def test_roundtrip_igs_no_abel_no_ro_only_fields(tmp_path):
    entries = [ObservationEntry.from_dict(_igs_dict(n_rays=4, label="igs_a"), obs_type="IGS")]
    path = netcdf_io.write_observations(entries, tmp_path / "igs_test.nc")

    back = netcdf_io.read_observations(path)
    assert len(back) == 1
    loaded = back[0]
    assert loaded.obs_type == "IGS"
    assert loaded.abel is None
    # snr_l1/snr_l2 never set for IGS -> not even written as variables, but
    # ObservationEntry still exposes them as None.
    assert loaded.snr_l1 is None
    np.testing.assert_allclose(loaded.pierce_lat, entries[0].pierce_lat, rtol=1e-5)
    np.testing.assert_allclose(loaded.extra["arc_time_sec"], entries[0].extra["arc_time_sec"])


def test_roundtrip_ragged_ray_counts_padding(tmp_path):
    """Different n_rays per obs must not leak NaN padding into the shorter
    entry's arrays after read-back (Plan Section 6.2's padded layout)."""
    entries = [
        ObservationEntry.from_dict(_ro_dict(n_rays=8, label="long"), obs_type="RO"),
        ObservationEntry.from_dict(_ro_dict(n_rays=2, label="short"), obs_type="RO"),
    ]
    path = netcdf_io.write_observations(entries, tmp_path / "ragged.nc")
    back = netcdf_io.read_observations(path)

    assert back[0].n_rays == 8
    assert back[1].n_rays == 2
    assert not np.any(np.isnan(back[1].tec))
    assert not np.any(np.isnan(back[1].rec_ecef_km))


def test_write_empty_entries_list(tmp_path):
    path = netcdf_io.write_observations([], tmp_path / "empty.nc")
    assert netcdf_io.read_observations(path) == []
