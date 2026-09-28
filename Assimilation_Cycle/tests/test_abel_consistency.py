# -*- coding: utf-8 -*-
"""
Tests for abel_consistency.py (plan Section 4.16), using the same
analytic radial-ray oracle EDPSamples/test_edp_samples.py's
TestLineOfSightTEC uses: a purely radial ray (GNSS/LEO at the same lat/lon,
differing only in altitude) has an exact analytic TEC for a constant
density N0 across [alt_lo, alt_hi]: N0 * (alt_hi - alt_lo) [m] / 1e16.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observation_preparation import ObservationEntry
from Assimilation_Cycle import abel_consistency as AC


def _sampling_parameters(n_sample: int) -> pd.DataFrame:
    sp = pd.DataFrame({
        "hour": [12.0] * n_sample, "f107": [120.0] * n_sample,
        "ap": [10.0] * n_sample, "ig12": [80.0] * n_sample, "rz12": [70.0] * n_sample,
    })
    sp.attrs = {}
    return sp


def _radial_ray(E, lat, lon, alt_top_km=1000.0):
    gnss = (E._geodetic_to_ecef(lat, lon, 0.0) / 1000.0).reshape(3, 1)
    leo = (E._geodetic_to_ecef(lat, lon, alt_top_km * 1000.0) / 1000.0).reshape(3, 1)
    return {"rec_ecef_km": leo, "gnss_ecef_km": gnss}


def _make_rectangle(E, altitude, placeholder_value=0.0, n_sample=1):
    sp = _sampling_parameters(n_sample)
    n_geo = E.EDPSamples.genRectangularArea(-10, 10, 5, 0, 10, 5)[0].shape[0]
    edps = np.full((len(altitude), n_geo, n_sample), placeholder_value)
    feature_edps = np.zeros((len(E.EDPSamples.FEATURE_LABEL), n_geo, n_sample))
    return E.EDPSamples(
        DateTime="2026-01-01", geo_type="Rectangle", altitude=altitude,
        sampling_parameters=sp, minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5,
        edps=edps, feature_edps=feature_edps,
    )


@pytest.fixture
def E():
    import edp_samples
    return edp_samples


class TestBuildUniformFieldFromAbel:
    def test_interpolates_and_replicates_across_geo(self):
        entry = ObservationEntry(
            obs_type="RO", tec=np.array([1.0]),
            gnss_ecef_km=np.zeros((3, 1)), rec_ecef_km=np.zeros((3, 1)),
            abel={"alt_km": np.array([100.0, 300.0, 500.0]), "Ne": np.array([1e10, 2e10, 3e10])},
        )
        altitude_grid = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        field = AC.build_uniform_field_from_abel(entry, altitude_grid, n_geo=4)

        assert field.shape == (5, 4, 1)
        # linear interpolation at 200/400 between the two bracketing points
        np.testing.assert_allclose(field[1, :, 0], 1.5e10)
        np.testing.assert_allclose(field[3, :, 0], 2.5e10)
        # every geo column identical (uniform field)
        for g in range(4):
            np.testing.assert_allclose(field[:, g, 0], field[:, 0, 0])

    def test_out_of_range_clamps_to_edge(self):
        entry = ObservationEntry(
            obs_type="RO", tec=np.array([1.0]),
            gnss_ecef_km=np.zeros((3, 1)), rec_ecef_km=np.zeros((3, 1)),
            abel={"alt_km": np.array([200.0, 400.0]), "Ne": np.array([5e10, 7e10])},
        )
        altitude_grid = np.array([0.0, 200.0, 400.0, 900.0])
        field = AC.build_uniform_field_from_abel(entry, altitude_grid, n_geo=1)
        assert field[0, 0, 0] == pytest.approx(5e10)   # below range -> low edge
        assert field[-1, 0, 0] == pytest.approx(7e10)  # above range -> high edge

    def test_raises_without_abel_data(self):
        entry = ObservationEntry(
            obs_type="RO", tec=np.array([1.0]),
            gnss_ecef_km=np.zeros((3, 1)), rec_ecef_km=np.zeros((3, 1)),
            abel=None,
        )
        with pytest.raises(ValueError):
            AC.build_uniform_field_from_abel(entry, np.array([100.0]), n_geo=2)

    def test_non_finite_abel_samples_are_dropped(self):
        entry = ObservationEntry(
            obs_type="RO", tec=np.array([1.0]),
            gnss_ecef_km=np.zeros((3, 1)), rec_ecef_km=np.zeros((3, 1)),
            abel={"alt_km": np.array([100.0, 300.0, np.nan, 500.0]),
                  "Ne": np.array([1e10, 2e10, np.nan, 3e10])},
        )
        field = AC.build_uniform_field_from_abel(entry, np.array([100.0, 300.0, 500.0]), n_geo=1)
        np.testing.assert_allclose(field[:, 0, 0], [1e10, 2e10, 3e10])


class TestPredictTecFromAbelAnalyticOracle:
    ALTITUDE = np.linspace(100.0, 500.0, 5)
    N0 = 1e11
    NUM_SEGMENTS = 5000
    EXPECTED_TEC = N0 * 400_000.0 / 1e16  # only [100,500] km contributes on a 0->1000 radial ray

    def test_matches_analytic_radial_integral(self, E):
        edp = _make_rectangle(E, self.ALTITUDE)
        entry = ObservationEntry(
            obs_type="RO", tec=np.array([]),
            gnss_ecef_km=np.zeros((3, 0)), rec_ecef_km=np.zeros((3, 0)),
            abel={"alt_km": self.ALTITUDE, "Ne": np.full_like(self.ALTITUDE, self.N0)},
        )
        podTc2 = _radial_ray(E, 3.0, 2.0)  # interior to a triangle, not on a grid line
        tec = AC.predict_tec_from_abel(entry, edp, podTc2, num_segments=self.NUM_SEGMENTS)
        assert tec[0] == pytest.approx(self.EXPECTED_TEC, rel=0.01)

    def test_consistency_check_end_to_end(self, E):
        edp = _make_rectangle(E, self.ALTITUDE)
        source = ObservationEntry(
            obs_type="RO", tec=np.array([]),
            gnss_ecef_km=np.zeros((3, 0)), rec_ecef_km=np.zeros((3, 0)),
            abel={"alt_km": self.ALTITUDE, "Ne": np.full_like(self.ALTITUDE, self.N0)}, label="src",
        )
        podTc2 = _radial_ray(E, 3.0, 2.0)
        # "measured" TEC deliberately set to the exact analytic value ->
        # residual should be ~0 (a perfectly Abel-consistent synthetic case).
        target = ObservationEntry(
            obs_type="IGS", tec=np.array([self.EXPECTED_TEC]),
            gnss_ecef_km=podTc2["gnss_ecef_km"], rec_ecef_km=podTc2["rec_ecef_km"], label="tgt",
        )

        result = AC.abel_consistency_check(source, [target], edp, num_segments=self.NUM_SEGMENTS)
        assert result.y_predicted[0] == pytest.approx(self.EXPECTED_TEC, rel=0.01)
        assert result.rmse < 0.01 * self.EXPECTED_TEC
        assert result.rmse_by_obs_type() == pytest.approx({"IGS": result.rmse})

    def test_consistency_check_detects_real_mismatch(self, E):
        edp = _make_rectangle(E, self.ALTITUDE)
        source = ObservationEntry(
            obs_type="RO", tec=np.array([]),
            gnss_ecef_km=np.zeros((3, 0)), rec_ecef_km=np.zeros((3, 0)),
            abel={"alt_km": self.ALTITUDE, "Ne": np.full_like(self.ALTITUDE, self.N0)},
        )
        podTc2 = _radial_ray(E, 3.0, 2.0)
        # "measured" TEC is deliberately way off from what a uniform N0
        # field would predict -> a large, easily-detected residual.
        target = ObservationEntry(
            obs_type="IGS", tec=np.array([self.EXPECTED_TEC * 5]),
            gnss_ecef_km=podTc2["gnss_ecef_km"], rec_ecef_km=podTc2["rec_ecef_km"],
        )
        result = AC.abel_consistency_check(source, [target], edp, num_segments=self.NUM_SEGMENTS)
        assert result.rmse > 0.5 * self.EXPECTED_TEC
