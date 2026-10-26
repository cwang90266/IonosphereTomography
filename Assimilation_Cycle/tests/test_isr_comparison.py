# -*- coding: utf-8 -*-
"""
Tests for isr_comparison.py (ISR_Integration_Plan.md Section 4.3).

Pure-data functions (load_isr_dataset, nearest/window matching, pooled
RMSE) are tested against small synthetic fixtures, no real IRI2020/ISR
file needed. IsrComparisonCollector is tested against a real (but
IRI2020-free) EDPSamples grid, reusing test_output.py's
``_make_rectangle_with_mesh`` helper (``edps=`` passed directly, so no
IRI2020 Fortran call happens) and a toy decode-only observation operator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import pytest

from Ensemble_Kalman_Engine import EnsembleState

from Assimilation_Cycle import isr_comparison as isrc
from Assimilation_Cycle.cycle_driver import CycleBatch

from test_output import _make_rectangle_with_mesh


def _write_synthetic_isr_file(path, n_time=20, n_alt=5, station_name="Test ISR",
                               station_lat=5.0, station_lon=0.0, start="2026-01-01T00:00:00"):
    rng = np.random.default_rng(0)
    altitude = np.linspace(100.0, 500.0, n_alt)
    time = pd.date_range(start, periods=n_time, freq="4min")
    density = rng.uniform(1e9, 1e12, size=(n_alt, n_time))

    ds = xr.Dataset(
        data_vars=dict(
            altitude=(("altitude_gate",), altitude, {"units": "km"}),
            Ne=(("altitude_gate", "time"), density, {"units": "m-3"}),
            time_utc=(("time",), np.array([t.isoformat() for t in time])),
        ),
        attrs=dict(
            station_name=station_name,
            station_latitude=station_lat,
            station_longitude=station_lon,
        ),
    )
    ds.to_netcdf(path)
    return altitude, time, density


class TestLoadIsrDataset:
    def test_reads_station_identity_from_attrs(self, tmp_path):
        path = tmp_path / "isr.nc"
        _write_synthetic_isr_file(path, station_name="Jicamarca", station_lat=-11.95, station_lon=-76.87)

        isr = isrc.load_isr_dataset(path)
        assert isr.station.name == "Jicamarca"
        assert isr.station.lat == pytest.approx(-11.95)
        assert isr.station.lon == pytest.approx(-76.87)

    def test_sorts_time_and_altitude(self, tmp_path):
        path = tmp_path / "isr.nc"
        altitude, time, density = _write_synthetic_isr_file(path, n_time=10, n_alt=4)

        isr = isrc.load_isr_dataset(path)
        assert list(isr.time) == sorted(isr.time)
        assert np.all(np.diff(isr.altitude) > 0)
        assert isr.density.shape == (4, 10)


class TestNearestAndWindow:
    def _dataset(self):
        time = pd.DatetimeIndex(pd.date_range("2026-01-01", periods=5, freq="1h"))
        altitude = np.array([100.0, 200.0, 300.0])
        density = np.arange(15, dtype=float).reshape(3, 5)
        station = isrc.IsrStation(name="X", lat=0.0, lon=0.0)
        return isrc.IsrDataset(station=station, time=time, altitude=altitude, density=density)

    def test_nearest_isr_profile_picks_closest_time(self):
        isr = self._dataset()
        target = pd.Timestamp("2026-01-01T02:40:00")  # closer to hour 3 than hour 2
        t, density = isrc.nearest_isr_profile(isr, target)
        assert t == pd.Timestamp("2026-01-01T03:00:00")
        np.testing.assert_allclose(density, isr.density[:, 3])

    def test_nearest_isr_profile_empty_dataset_returns_none(self):
        station = isrc.IsrStation(name="X", lat=0.0, lon=0.0)
        empty = isrc.IsrDataset(station=station, time=pd.DatetimeIndex([]),
                                 altitude=np.array([100.0]), density=np.zeros((1, 0)))
        assert isrc.nearest_isr_profile(empty, pd.Timestamp("2026-01-01")) is None

    def test_isr_profiles_in_window_filters_by_time(self):
        isr = self._dataset()
        window = isrc.isr_profiles_in_window(isr, "2026-01-01T01:00:00", "2026-01-01T03:00:00")
        assert window.shape == (3, 3)
        np.testing.assert_allclose(window, isr.density[:, 1:4])

    def test_batch_midpoint(self):
        batch = CycleBatch(podTc2_data={}, y_obs=np.array([1.0]), R=np.eye(1),
                            start_time="2026-01-01T10:00:00", end_time="2026-01-01T11:00:00")
        mid = isrc.batch_midpoint(batch)
        assert mid == pd.Timestamp("2026-01-01T10:30:00")

    def test_batch_midpoint_none_without_times(self):
        batch = CycleBatch(podTc2_data={}, y_obs=np.array([1.0]), R=np.eye(1))
        assert isrc.batch_midpoint(batch) is None


def _match(batch_index, isr_density, isr_time="2026-01-01T00:00:00", n_height=4, n_members=6, seed=0):
    rng = np.random.default_rng(seed)
    forecast = rng.uniform(1e10, 1e11, size=(n_height, n_members))
    analysis = forecast * 0.9
    return isrc.BatchIsrMatch(
        batch_index=batch_index, isr_time=pd.Timestamp(isr_time), isr_density=isr_density,
        forecast_profile=forecast, analysis_profile=analysis,
    )


class TestPooledRmse:
    def test_matches_hand_computed_rmse(self):
        model_altitude = np.array([100.0, 200.0, 300.0, 400.0])
        isr_altitude = np.array([200.0, 300.0])

        forecast = np.full((4, 1), 1e10)  # constant profile -> interpolation is exact at every altitude
        analysis = np.full((4, 1), 2e10)
        isr_density = np.array([1.5e10, 2.5e10])

        match = isrc.BatchIsrMatch(
            batch_index=0, isr_time=pd.Timestamp("2026-01-01"), isr_density=isr_density,
            forecast_profile=forecast, analysis_profile=analysis,
        )
        df = isrc.pooled_isr_rmse_by_altitude([match], model_altitude, isr_altitude)

        # forecast is flat 1e10 everywhere -> interpolated value at both isr gates is 1e10
        expected_forecast_rmse = np.abs(1e10 - isr_density)
        # analysis is flat 2e10 everywhere
        expected_analysis_rmse = np.abs(2e10 - isr_density)
        np.testing.assert_allclose(df["rmse_forecast_m3"].to_numpy(), expected_forecast_rmse, rtol=1e-6)
        np.testing.assert_allclose(df["rmse_analysis_m3"].to_numpy(), expected_analysis_rmse, rtol=1e-6)
        np.testing.assert_array_equal(df["n_pairs"].to_numpy(), [1, 1])

    def test_skips_batches_with_no_isr_coverage(self):
        model_altitude = np.array([100.0, 200.0, 300.0])
        isr_altitude = np.array([200.0])
        matched = _match(0, isr_density=np.array([1e10]), n_height=3)
        unmatched = _match(1, isr_density=None, n_height=3)

        df = isrc.pooled_isr_rmse_by_altitude([matched, unmatched], model_altitude, isr_altitude)
        assert df["n_pairs"].iloc[0] == 1

    def test_no_coverage_at_all_gives_nan_not_crash(self):
        model_altitude = np.array([100.0, 200.0, 300.0])
        isr_altitude = np.array([200.0])
        unmatched = _match(0, isr_density=None)

        df = isrc.pooled_isr_rmse_by_altitude([unmatched], model_altitude, isr_altitude)
        assert df["n_pairs"].iloc[0] == 0
        assert np.isnan(df["rmse_forecast_m3"].iloc[0])

    def test_cross_style_concatenates_with_style_column(self):
        model_altitude = np.array([100.0, 200.0, 300.0])
        isr_altitude = np.array([200.0])
        matches_by_style = {
            "A": [_match(0, isr_density=np.array([1e10]), n_height=3, seed=1)],
            "B": [_match(0, isr_density=np.array([1e10]), n_height=3, seed=2)],
        }
        df = isrc.pooled_isr_rmse_by_altitude_cross_style(matches_by_style, model_altitude, isr_altitude)
        assert set(df["style"]) == {"A", "B"}
        assert len(df) == 2


class TestPlotsRun:
    def _isr_dataset(self):
        time = pd.DatetimeIndex(pd.date_range("2026-01-01", periods=6, freq="1h"))
        altitude = np.array([100.0, 200.0, 300.0])
        rng = np.random.default_rng(3)
        density = rng.uniform(1e9, 1e12, size=(3, 6))
        station = isrc.IsrStation(name="Test", lat=0.0, lon=0.0)
        return isrc.IsrDataset(station=station, time=time, altitude=altitude, density=density)

    def test_nearest_point_plot_runs(self):
        isr_altitude = np.array([100.0, 200.0, 300.0])
        model_altitude = np.linspace(90.0, 400.0, 10)
        match = _match(0, isr_density=np.array([1e10, 2e10, 1.5e10]), n_height=10)

        ax = isrc.plot_isr_nearest_point_comparison(match, model_altitude, isr_altitude)
        assert ax.get_ylabel() == "Altitude (km)"
        assert ax.get_legend() is not None

    def test_nearest_point_plot_handles_missing_isr(self):
        isr_altitude = np.array([100.0, 200.0, 300.0])
        model_altitude = np.linspace(90.0, 400.0, 10)
        match = _match(0, isr_density=None, n_height=10)
        ax = isrc.plot_isr_nearest_point_comparison(match, model_altitude, isr_altitude)
        assert ax is not None

    def test_nearest_point_cross_style_plot_runs(self):
        isr_altitude = np.array([100.0, 200.0, 300.0])
        model_altitude = np.linspace(90.0, 400.0, 10)
        matches_by_style = {
            "A": _match(0, isr_density=np.array([1e10, 2e10, 1.5e10]), n_height=10, seed=1),
            "B": _match(0, isr_density=np.array([1e10, 2e10, 1.5e10]), n_height=10, seed=2),
        }
        ax = isrc.plot_isr_nearest_point_comparison_cross_style(matches_by_style, model_altitude, isr_altitude)
        assert len(ax.lines) >= 3  # 2 styles' analysis lines + the ISR line

    def test_range_plot_runs(self):
        isr = self._isr_dataset()
        model_altitude = np.linspace(90.0, 400.0, 10)
        matches = [_match(i, isr_density=None, n_height=10, seed=i) for i in range(3)]
        ax = isrc.plot_isr_range_comparison(matches, isr, "2026-01-01T00:00:00", "2026-01-01T05:00:00",
                                             model_altitude)
        assert ax.get_legend() is not None

    def test_range_plot_handles_no_coverage(self):
        isr = self._isr_dataset()
        model_altitude = np.linspace(90.0, 400.0, 10)
        matches = [_match(0, isr_density=None, n_height=10)]
        ax = isrc.plot_isr_range_comparison(matches, isr, "2030-01-01", "2030-01-02", model_altitude)
        assert ax is not None

    def test_range_cross_style_plot_runs(self):
        isr = self._isr_dataset()
        model_altitude = np.linspace(90.0, 400.0, 10)
        matches_by_style = {
            "A": [_match(i, isr_density=None, n_height=10, seed=i) for i in range(2)],
            "B": [_match(i, isr_density=None, n_height=10, seed=i + 10) for i in range(2)],
        }
        ax = isrc.plot_isr_range_comparison_cross_style(
            matches_by_style, isr, "2026-01-01T00:00:00", "2026-01-01T05:00:00", model_altitude,
        )
        assert ax is not None

    def test_pooled_rmse_plots_run(self):
        model_altitude = np.array([100.0, 200.0, 300.0])
        isr_altitude = np.array([200.0])
        matches_by_style = {
            "A": [_match(0, isr_density=np.array([1e10]), n_height=3, seed=1)],
            "B": [_match(0, isr_density=np.array([1e10]), n_height=3, seed=2)],
        }
        df_a = isrc.pooled_isr_rmse_by_altitude(matches_by_style["A"], model_altitude, isr_altitude)
        ax = isrc.plot_pooled_isr_rmse(df_a)
        assert ax is not None

        df_cross = isrc.pooled_isr_rmse_by_altitude_cross_style(matches_by_style, model_altitude, isr_altitude)
        ax_cross = isrc.plot_pooled_isr_rmse_cross_style(df_cross)
        assert len(ax_cross.lines) == 2


@dataclass
class _DecodeToyOperator:
    """Decode-only toy observation operator: ignores the ensemble's
    actual values and returns a fixed-shape random density field, just to
    exercise IsrComparisonCollector's wiring (decode -> interpolate ->
    match), not any real physics."""
    n_height: int
    n_geo: int
    seed: int = 0

    def decode(self, X_param_shape):
        n_members = X_param_shape.shape[-1]
        rng = np.random.default_rng(self.seed)
        return rng.uniform(1e10, 1e12, size=(self.n_height, self.n_geo, n_members))


class TestIsrComparisonCollector:
    def test_on_batch_collects_one_match_per_batch(self):
        altitude = np.linspace(100.0, 500.0, 5)
        edp = _make_rectangle_with_mesh(altitude)  # grid spans lon[-10,10] x lat[0,10]
        station = isrc.IsrStation(name="Test", lat=5.0, lon=0.0)  # interior point
        isr_time = pd.DatetimeIndex(pd.date_range("2026-01-01T10:00:00", periods=3, freq="30min"))
        isr_density = np.random.default_rng(7).uniform(1e9, 1e12, size=(5, 3))
        isr = isrc.IsrDataset(station=station, time=isr_time, altitude=np.linspace(100, 500, 5),
                               density=isr_density)

        n_height, n_geo = edp.edps.shape[0], edp.edps.shape[1]
        n_state, n_members = n_height * n_geo, 8
        param_shape = (n_height, n_geo)

        rng = np.random.default_rng(1)
        ensemble_forecast = EnsembleState(X=rng.normal(size=(n_state, n_members)), param_shape=param_shape)
        ensemble_analysis = EnsembleState(X=rng.normal(size=(n_state, n_members)), param_shape=param_shape)

        operator = _DecodeToyOperator(n_height=n_height, n_geo=n_geo)
        batch = CycleBatch(podTc2_data={}, y_obs=np.array([1.0]), R=np.eye(1), batch_index=0,
                            start_time="2026-01-01T10:00:00", end_time="2026-01-01T10:20:00")

        collector = isrc.IsrComparisonCollector(edp, isr)
        collector.on_batch(batch, operator, ensemble_forecast, ensemble_analysis, outcome=None)

        assert len(collector.matches) == 1
        match = collector.matches[0]
        assert match.batch_index == 0
        assert match.isr_time is not None
        assert match.forecast_profile.shape == (n_height, n_members)
        assert match.analysis_profile.shape == (n_height, n_members)
