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
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from Ensemble_Kalman_Engine import EnsembleState

from Assimilation_Cycle import isr_comparison as isrc
from Assimilation_Cycle.cycle_driver import CycleBatch
from Assimilation_Cycle.ensemble_io import save_ensemble_netcdf

from test_output import _make_rectangle_with_mesh


def _write_synthetic_isr_file(path, n_time=20, n_alt=5, station_name="Test ISR",
                               station_lat=5.0, station_lon=0.0, start="2026-01-01T00:00:00",
                               with_extrema=False, n_profiles_value=5):
    rng = np.random.default_rng(0)
    altitude = np.linspace(100.0, 500.0, n_alt)
    time = pd.date_range(start, periods=n_time, freq="4min")
    density = rng.uniform(1e9, 1e12, size=(n_alt, n_time))

    data_vars = dict(
        altitude=(("altitude_gate",), altitude, {"units": "km"}),
        Ne=(("altitude_gate", "time"), density, {"units": "m-3"}),
        time_utc=(("time",), np.array([t.isoformat() for t in time])),
    )
    if with_extrema:
        data_vars["Ne_min"] = (("altitude_gate", "time"), density * 0.8)
        data_vars["Ne_max"] = (("altitude_gate", "time"), density * 1.2)
        data_vars["profiles_used_per_gate"] = (
            ("altitude_gate", "time"), np.full((n_alt, n_time), n_profiles_value, dtype=np.int32),
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        attrs=dict(
            station_name=station_name,
            station_latitude=station_lat,
            station_longitude=station_lon,
        ),
    )
    ds.to_netcdf(path)
    return altitude, time, density


def _write_synthetic_ionosonde_file(path, n_time=10, station_name="Test ISR",
                                     station_lat=5.0, station_lon=0.0, start="2026-01-01T00:00:00",
                                     nan_fraction=0.0):
    rng = np.random.default_rng(1)
    time = pd.date_range(start, periods=n_time, freq="10min")
    nmF2 = rng.uniform(1e11, 2e11, size=n_time)
    hmF2 = rng.uniform(250.0, 350.0, size=n_time)
    if nan_fraction > 0:
        n_nan = int(n_time * nan_fraction)
        nmF2[:n_nan] = np.nan

    ds = xr.Dataset(
        data_vars=dict(
            nmF2=(("time",), nmF2),
            hmF2=(("time",), hmF2),
            time_utc=(("time",), np.array([t.isoformat() for t in time])),
        ),
        attrs=dict(
            station_name=station_name,
            station_latitude=station_lat,
            station_longitude=station_lon,
        ),
    )
    ds.to_netcdf(path)
    return time, nmF2, hmF2


def _write_synthetic_ionosonde_profile_file(path, n_time=10, n_alt=5, station_name="Test ISR",
                                             station_lat=5.0, station_lon=0.0, start="2026-01-01T00:00:00",
                                             freq="10min", seed=2):
    """Writes the full vertical-profile ionosonde schema (``altitude``/
    ``Ne``/``time_utc`` -- mirrors ``MILIonosonde_Ne_Aug2530_2025.nc``):
    ``Ne`` increases monotonically up to a per-scan random peak gate
    (that scan's own hmF2), ``nan`` above it -- the real file's
    bottomside-only convention. Returns ``(altitude, time, density,
    peak_idx)`` so a test can hand-compute the expected derived
    nmF2/hmF2."""
    rng = np.random.default_rng(seed)
    altitude = np.linspace(100.0, 500.0, n_alt)
    time = pd.date_range(start, periods=n_time, freq=freq)
    density = rng.uniform(1e9, 1e11, size=(n_alt, n_time))
    peak_idx = rng.integers(1, n_alt, size=n_time)
    for t in range(n_time):
        density[: peak_idx[t] + 1, t] = np.sort(density[: peak_idx[t] + 1, t])
        density[peak_idx[t] + 1:, t] = np.nan

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
    return altitude, time, density, peak_idx


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


class TestConfidence:
    def test_matches_formula(self):
        n = np.array([1, 2, 3, 4, 14])
        expected = np.clip(1.0 - 2.0 ** (1.0 - n), 0.0, 1.0)
        np.testing.assert_allclose(isrc._confidence(n), expected)

    def test_single_sample_gives_zero_confidence(self):
        assert isrc._confidence(1) == pytest.approx(0.0)

    def test_many_samples_approaches_one(self):
        assert isrc._confidence(30) > 0.9999

    def test_clips_below_one_sample(self):
        # n=0 would otherwise go negative (1 - 2**1 = -1)
        assert isrc._confidence(0) == 0.0


class TestLoadIsrDatasetRicherSchema:
    def test_old_schema_leaves_extrema_none(self, tmp_path):
        path = tmp_path / "isr.nc"
        _write_synthetic_isr_file(path, with_extrema=False)

        isr = isrc.load_isr_dataset(path)
        assert isr.density_min is None
        assert isr.density_max is None
        assert isr.n_profiles is None

    def test_new_schema_reads_extrema_and_sorts_them(self, tmp_path):
        path = tmp_path / "isr.nc"
        altitude, time, density = _write_synthetic_isr_file(
            path, n_time=6, n_alt=4, with_extrema=True, n_profiles_value=7,
        )
        isr = isrc.load_isr_dataset(path)

        assert isr.density_min is not None
        assert isr.density_max is not None
        assert isr.n_profiles is not None
        assert isr.density_min.shape == (4, 6)
        np.testing.assert_allclose(isr.n_profiles, 7)
        # density_min/max bracket density at every (alt, time) after sorting
        assert np.all(isr.density_min <= isr.density + 1e-6)
        assert np.all(isr.density_max >= isr.density - 1e-6)


class TestLoadIonosondeDataset:
    def test_reads_station_identity_and_sorts_time(self, tmp_path):
        path = tmp_path / "iono.nc"
        _write_synthetic_ionosonde_file(path, station_name="Test ISR", station_lat=5.0, station_lon=0.0)

        iono = isrc.load_ionosonde_dataset(path)
        assert iono.station.name == "Test ISR"
        assert list(iono.time) == sorted(iono.time)
        assert iono.nmF2.shape == iono.time.shape

    def test_nan_entries_preserved(self, tmp_path):
        path = tmp_path / "iono.nc"
        _write_synthetic_ionosonde_file(path, n_time=10, nan_fraction=0.3)

        iono = isrc.load_ionosonde_dataset(path)
        assert np.isnan(iono.nmF2).any()

    def test_old_schema_leaves_profile_fields_none(self, tmp_path):
        path = tmp_path / "iono.nc"
        _write_synthetic_ionosonde_file(path)

        iono = isrc.load_ionosonde_dataset(path)
        assert iono.profile_altitude is None
        assert iono.profile_density is None


class TestLoadIonosondeDatasetProfile:
    """The richer, full-vertical-profile schema (2026-11-25 user request;
    module docstring's "Ionosonde vertical profile overlay" section)."""

    def test_reads_profile_and_sorts_altitude(self, tmp_path):
        path = tmp_path / "iono_profile.nc"
        altitude, time, density, peak_idx = _write_synthetic_ionosonde_profile_file(
            path, n_time=6, n_alt=5,
        )
        iono = isrc.load_ionosonde_dataset(path)
        assert iono.profile_altitude is not None
        assert iono.profile_density is not None
        assert iono.profile_density.shape == (5, 6)
        assert np.all(np.diff(iono.profile_altitude) > 0)

    def test_derives_nmf2_hmf2_as_per_column_peak(self, tmp_path):
        path = tmp_path / "iono_profile.nc"
        altitude, time, density, peak_idx = _write_synthetic_ionosonde_profile_file(
            path, n_time=6, n_alt=5,
        )
        iono = isrc.load_ionosonde_dataset(path)
        expected_nmF2 = np.array([density[peak_idx[t], t] for t in range(6)])
        expected_hmF2 = np.array([altitude[peak_idx[t]] for t in range(6)])
        np.testing.assert_allclose(iono.nmF2, expected_nmF2)
        np.testing.assert_allclose(iono.hmF2, expected_hmF2)

    def test_dedicated_nmf2_hmf2_vars_take_priority_over_derivation(self, tmp_path):
        """A file carrying both schemas at once reads nmF2/hmF2 directly
        -- not re-derived from the profile -- while still populating the
        profile fields."""
        path = tmp_path / "iono_both.nc"
        altitude = np.linspace(100.0, 500.0, 3)
        time = pd.date_range("2026-01-01", periods=2, freq="1h")
        density = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]])
        ds = xr.Dataset(
            data_vars=dict(
                altitude=(("altitude_gate",), altitude),
                Ne=(("altitude_gate", "time"), density),
                nmF2=(("time",), np.array([999.0, 888.0])),
                hmF2=(("time",), np.array([111.0, 222.0])),
                time_utc=(("time",), np.array([t.isoformat() for t in time])),
            ),
            attrs=dict(station_name="X", station_latitude=0.0, station_longitude=0.0),
        )
        ds.to_netcdf(path)

        iono = isrc.load_ionosonde_dataset(path)
        np.testing.assert_allclose(iono.nmF2, [999.0, 888.0])
        np.testing.assert_allclose(iono.hmF2, [111.0, 222.0])
        assert iono.profile_density is not None

    def test_raises_when_neither_schema_present(self, tmp_path):
        path = tmp_path / "empty.nc"
        ds = xr.Dataset(
            data_vars=dict(time_utc=(("time",), np.array(["2026-01-01T00:00:00"]))),
            attrs=dict(station_name="X", station_latitude=0.0, station_longitude=0.0),
        )
        ds.to_netcdf(path)
        with pytest.raises(ValueError, match="neither nmF2/hmF2 nor"):
            isrc.load_ionosonde_dataset(path)


class TestIonosondePointsInWindow:
    def _dataset(self):
        time = pd.DatetimeIndex(pd.date_range("2026-01-01", periods=5, freq="1h"))
        nmF2 = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        hmF2 = np.array([100.0, 200.0, 300.0, np.nan, 500.0])
        station = isrc.IsrStation(name="X", lat=0.0, lon=0.0)
        return isrc.IonosondeDataset(station=station, time=time, nmF2=nmF2, hmF2=hmF2)

    def test_filters_by_time_and_drops_nonfinite(self):
        iono = self._dataset()
        nmF2, hmF2 = isrc.ionosonde_points_in_window(iono, "2026-01-01T00:00", "2026-01-01T04:00")
        # index 2 (nan nmF2) and index 3 (nan hmF2) both dropped
        np.testing.assert_allclose(nmF2, [1.0, 2.0, 5.0])
        np.testing.assert_allclose(hmF2, [100.0, 200.0, 500.0])

    def test_none_dataset_returns_empty_arrays(self):
        nmF2, hmF2 = isrc.ionosonde_points_in_window(None, "2026-01-01", "2026-01-02")
        assert len(nmF2) == 0
        assert len(hmF2) == 0

    def test_no_overlap_returns_empty(self):
        iono = self._dataset()
        nmF2, hmF2 = isrc.ionosonde_points_in_window(iono, "2030-01-01", "2030-01-02")
        assert len(nmF2) == 0


class TestDrawConfidenceBand:
    def test_draws_one_bar_per_gate_with_confidence_weighted_alpha(self):
        _, ax = plt.subplots()
        altitude = np.array([100.0, 200.0, 300.0])
        lo = np.array([1.0, 1.0, 1.0])
        hi = np.array([2.0, 2.0, 2.0])
        n = np.array([1, 4, 14])
        isrc._draw_isr_confidence_band(ax, altitude, lo, hi, n)
        assert len(ax.patches) == 3
        alphas = [p.get_facecolor()[3] for p in ax.patches]
        # confidence (hence alpha) strictly increases with n
        assert alphas[0] < alphas[1] < alphas[2]
        assert alphas[0] == pytest.approx(0.0)  # n=1 -> zero confidence

    def test_flat_alpha_when_n_not_given(self):
        _, ax = plt.subplots()
        altitude = np.array([100.0, 200.0])
        lo = np.array([1.0, 1.0])
        hi = np.array([2.0, 2.0])
        isrc._draw_isr_confidence_band(ax, altitude, lo, hi, n=None, max_alpha=0.15)
        alphas = [p.get_facecolor()[3] for p in ax.patches]
        assert alphas == pytest.approx([0.15, 0.15])

    def test_noop_when_bounds_missing(self):
        _, ax = plt.subplots()
        isrc._draw_isr_confidence_band(ax, np.array([100.0]), None, None)
        assert len(ax.patches) == 0


class TestIonosondeProfileWindowStats:
    def _dataset_with_profile(self):
        time = pd.DatetimeIndex(pd.date_range("2026-01-01T00:00", periods=4, freq="15min"))
        altitude = np.array([100.0, 200.0, 300.0])
        density = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, np.nan, 40.0],
            [np.nan, np.nan, np.nan, np.nan],
        ])
        station = isrc.IsrStation(name="X", lat=0.0, lon=0.0)
        return isrc.IonosondeDataset(
            station=station, time=time, nmF2=np.array([10.0, 20.0, 3.0, 40.0]),
            hmF2=np.array([200.0, 200.0, 100.0, 200.0]),
            profile_altitude=altitude, profile_density=density,
        )

    def test_pools_median_and_range_within_window(self):
        iono = self._dataset_with_profile()
        # window covers columns 0,1,2 (00:00, 00:15, 00:30); 00:45 excluded
        stats = isrc.ionosonde_profile_window_stats(iono, "2026-01-01T00:00", "2026-01-01T00:30")
        np.testing.assert_allclose(stats.altitude, [100.0, 200.0, 300.0])
        np.testing.assert_allclose(stats.median, [2.0, 15.0, np.nan], equal_nan=True)
        np.testing.assert_allclose(stats.lo, [1.0, 10.0, np.nan], equal_nan=True)
        np.testing.assert_allclose(stats.hi, [3.0, 20.0, np.nan], equal_nan=True)
        np.testing.assert_array_equal(stats.n, [3, 2, 0])

    def test_none_dataset_returns_none(self):
        assert isrc.ionosonde_profile_window_stats(None, "2026-01-01", "2026-01-02") is None

    def test_dataset_without_profile_returns_none(self):
        iono = isrc.IonosondeDataset(
            station=isrc.IsrStation(name="X", lat=0.0, lon=0.0),
            time=pd.DatetimeIndex(pd.date_range("2026-01-01", periods=2, freq="1h")),
            nmF2=np.array([1.0, 2.0]), hmF2=np.array([200.0, 210.0]),
        )
        assert isrc.ionosonde_profile_window_stats(iono, "2026-01-01", "2026-01-02") is None

    def test_no_scan_in_window_returns_none(self):
        iono = self._dataset_with_profile()
        assert isrc.ionosonde_profile_window_stats(iono, "2030-01-01", "2030-01-02") is None

    def test_percentile_band_narrower_than_minmax(self):
        rng = np.random.default_rng(0)
        n_time = 20
        time = pd.DatetimeIndex(pd.date_range("2026-01-01", periods=n_time, freq="1min"))
        altitude = np.array([100.0])
        density = rng.uniform(0, 100, size=(1, n_time))
        iono = isrc.IonosondeDataset(
            station=isrc.IsrStation(name="X", lat=0.0, lon=0.0), time=time,
            nmF2=np.zeros(n_time), hmF2=np.zeros(n_time),
            profile_altitude=altitude, profile_density=density,
        )
        stats_minmax = isrc.ionosonde_profile_window_stats(iono, time[0], time[-1], percentile=0.0)
        stats_pct = isrc.ionosonde_profile_window_stats(iono, time[0], time[-1], percentile=10.0)
        assert (stats_pct.hi[0] - stats_pct.lo[0]) < (stats_minmax.hi[0] - stats_minmax.lo[0])


class TestDrawIonosondeProfileOverlay:
    def test_draws_band_and_median_line(self):
        _, ax = plt.subplots()
        profile = isrc.IonosondeProfileWindow(
            altitude=np.array([100.0, 200.0, 300.0]), median=np.array([1.0, 2.0, 3.0]),
            lo=np.array([0.5, 1.5, 2.5]), hi=np.array([1.5, 2.5, 3.5]), n=np.array([3, 3, 3]),
        )
        isrc._draw_ionosonde_profile_overlay(ax, profile)
        assert len(ax.patches) == 3   # range band, one bar per gate
        assert len(ax.lines) == 1     # median curve
        assert ax.lines[0].get_linestyle() == "--"

    def test_noop_when_profile_none(self):
        _, ax = plt.subplots()
        isrc._draw_ionosonde_profile_overlay(ax, None)
        assert len(ax.patches) == 0
        assert len(ax.lines) == 0

    def test_skips_median_line_when_all_nan(self):
        _, ax = plt.subplots()
        profile = isrc.IonosondeProfileWindow(
            altitude=np.array([100.0]), median=np.array([np.nan]),
            lo=np.array([np.nan]), hi=np.array([np.nan]), n=np.array([0]),
        )
        isrc._draw_ionosonde_profile_overlay(ax, profile)
        assert len(ax.patches) == 0
        assert len(ax.lines) == 0


class TestPlotAWithConfidenceBandAndIonosonde:
    def test_nearest_point_plot_draws_band_and_red_dots(self):
        isr_altitude = np.array([100.0, 200.0, 300.0])
        model_altitude = np.linspace(90.0, 400.0, 10)
        match = isrc.BatchIsrMatch(
            batch_index=0, isr_time=pd.Timestamp("2026-01-01"),
            isr_density=np.array([1e10, 2e10, 1.5e10]),
            forecast_profile=np.random.default_rng(0).uniform(1e10, 1e11, size=(10, 5)),
            analysis_profile=np.random.default_rng(1).uniform(1e10, 1e11, size=(10, 5)),
            isr_density_min=np.array([0.8e10, 1.8e10, 1.2e10]),
            isr_density_max=np.array([1.2e10, 2.2e10, 1.8e10]),
            isr_n_profiles=np.array([1, 4, 14]),
            ionosonde_nmF2=np.array([1.6e10]),
            ionosonde_hmF2=np.array([290.0]),
        )
        ax = isrc.plot_isr_nearest_point_comparison(match, model_altitude, isr_altitude)
        assert len(ax.patches) == 3          # confidence band, one bar per gate
        # 2 fill_betweenx (forecast/analysis std) + 1 ionosonde scatter
        assert len(ax.collections) == 3

    def test_without_new_fields_draws_no_band_or_dots(self):
        isr_altitude = np.array([100.0, 200.0, 300.0])
        model_altitude = np.linspace(90.0, 400.0, 10)
        match = _match(0, isr_density=np.array([1e10, 2e10, 1.5e10]), n_height=10)
        ax = isrc.plot_isr_nearest_point_comparison(match, model_altitude, isr_altitude)
        assert len(ax.patches) == 0
        # 2 fill_betweenx (forecast/analysis std) remain; no confidence band, no ionosonde scatter
        assert len(ax.collections) == 2

    def test_draws_ionosonde_profile_overlay_when_present(self):
        isr_altitude = np.array([100.0, 200.0, 300.0])
        model_altitude = np.linspace(90.0, 400.0, 10)
        ionosonde_profile = isrc.IonosondeProfileWindow(
            altitude=np.array([100.0, 200.0]), median=np.array([1e10, 2e10]),
            lo=np.array([0.8e10, 1.8e10]), hi=np.array([1.2e10, 2.2e10]), n=np.array([3, 3]),
        )
        match = _match(0, isr_density=None, n_height=10)
        match.ionosonde_profile = ionosonde_profile
        ax = isrc.plot_isr_nearest_point_comparison(match, model_altitude, isr_altitude)
        assert len(ax.patches) == 2   # ionosonde range band, one bar per gate (no ISR confidence band here)
        assert any(line.get_linestyle() == "--" for line in ax.lines)


def _write_synthetic_run(tmp_path, label="synthrun", styles=("raw",), n_sample=6):
    """Builds a tiny, real (IRI2020-free) two-batch run tree with exactly
    the files ``package_run.py`` saves (``{label}_edp_samples.nc``, each
    style's ``..._parameterized.nc`` + ``..._ensembles/ensemble_batch_*.nc``)
    -- enough for :func:`isrc._build_batch_isr_matches_from_saved`/
    :func:`isrc.replot_isr_comparison_tree` to reconstruct from, with no
    EnKF/pipeline involved in building the fixture itself."""
    from Parameterization import Parameterized_EDPSamples

    altitude = np.linspace(100.0, 500.0, 5)
    edp = _make_rectangle_with_mesh(altitude, n_sample=n_sample, placeholder_value=1e11)
    rng = np.random.default_rng(0)
    edp["EDPs"] = (edp["EDPs"].dims, edp.edps + rng.normal(0, 1e9, size=edp.edps.shape))
    edp.saveNetCDF(tmp_path / f"{label}_edp_samples.nc")

    for style in styles:
        pes = Parameterized_EDPSamples(edp, style=style, hyper_params={})
        pes.saveNetCDF(tmp_path / f"{label}_{style}_parameterized.nc")

        forecast = EnsembleState.from_parameterized_edp_samples(pes)
        analysis0 = EnsembleState(X=forecast.X * 0.9, param_shape=forecast.param_shape)
        analysis1 = EnsembleState(X=forecast.X * 0.8, param_shape=forecast.param_shape)

        ensembles_dir = tmp_path / f"{label}_{style}_ensembles"
        save_ensemble_netcdf(
            analysis0, ensembles_dir / "ensemble_batch_0000.nc",
            style=style, hyper_params={}, cycle_start_time="2026-01-01T00:00:00",
            cycle_end_time="2026-01-01T02:00:00", batch_index=0,
            batch_start_time="2026-01-01T00:00:00", batch_end_time="2026-01-01T01:00:00",
        )
        save_ensemble_netcdf(
            analysis1, ensembles_dir / "ensemble_batch_0001.nc",
            style=style, hyper_params={}, cycle_start_time="2026-01-01T00:00:00",
            cycle_end_time="2026-01-01T02:00:00", batch_index=1,
            batch_start_time="2026-01-01T01:00:00", batch_end_time="2026-01-01T02:00:00",
        )

    return altitude, edp


class TestBuildBatchIsrMatchesFromSaved:
    def test_batch0_forecast_is_pes_reconstruction_batch1_forecast_is_batch0_analysis(self, tmp_path):
        altitude, edp = _write_synthetic_run(tmp_path, label="synthrun", styles=("raw",))
        isr = isrc.IsrDataset(
            station=isrc.IsrStation(name="X", lat=5.0, lon=0.0),
            time=pd.DatetimeIndex(pd.date_range("2026-01-01T00:30:00", periods=3, freq="30min")),
            altitude=altitude.copy(), density=np.full((5, 3), 1e11),
        )

        prefix = tmp_path / "synthrun_raw"
        recon = isrc._build_batch_isr_matches_from_saved(prefix, isr, isr.station.lat, isr.station.lon)

        assert len(recon.matches) == 2
        assert recon.cycle_start == pd.Timestamp("2026-01-01T00:00:00")
        assert recon.cycle_end == pd.Timestamp("2026-01-01T02:00:00")

        m0, m1 = recon.matches
        # batch 1's forecast is exactly batch 0's saved analysis (no
        # inter-batch inflation -- plan Section 8 decision 3)
        np.testing.assert_allclose(m1.forecast_profile, m0.analysis_profile)
        # batch 0's forecast differs from batch 1's (the two saved
        # analyses were scaled differently -- 0.9x vs 0.8x -- in the fixture)
        assert not np.allclose(m0.forecast_profile, m1.forecast_profile)


class TestReplotIsrComparisonTree:
    def test_writes_per_style_and_cross_style_plots(self, tmp_path):
        altitude, edp = _write_synthetic_run(
            tmp_path, label="synthrun", styles=("raw", "density_10ex"),
        )
        isr_path = tmp_path / "isr.nc"
        _write_synthetic_isr_file(
            isr_path, n_time=3, n_alt=5, station_lat=5.0, station_lon=0.0,
            start="2026-01-01T00:30:00", with_extrema=True,
        )
        # overwrite altitude to match the run's model grid exactly (the
        # synthetic ISR writer above draws its own random grid)
        with xr.open_dataset(isr_path) as ds:
            ds = ds.load()
        ds = ds.assign(altitude=(("altitude_gate",), altitude.copy()))
        ds.to_netcdf(isr_path, mode="w")

        iono_path = tmp_path / "iono.nc"
        _write_synthetic_ionosonde_file(
            iono_path, n_time=4, station_lat=5.0, station_lon=0.0, start="2026-01-01T00:15:00",
        )

        written = isrc.replot_isr_comparison_tree(tmp_path, isr_path, ionosonde_file_path=iono_path)

        written_set = {p.relative_to(tmp_path) for p in written}
        for style in ("raw", "density_10ex"):
            for batch in (0, 1):
                assert Path(f"synthrun_isr_comparison/{style}/nearest_point/batch{batch:04d}.png") in written_set
            assert Path(f"synthrun_isr_comparison/{style}/range_comparison.png") in written_set
            assert Path(f"synthrun_isr_comparison/{style}/pooled_rmse.png") in written_set
            assert (tmp_path / f"synthrun_isr_comparison/{style}/pooled_rmse.csv").exists()

        for batch in (0, 1):
            assert Path(f"synthrun_isr_comparison_cross_style/nearest_point/batch{batch:04d}.png") in written_set
        assert Path("synthrun_isr_comparison_cross_style/range_comparison.png") in written_set


class TestReplotIsrComparisonTreeWithIonosondeProfile:
    """End-to-end: the post-processing path
    (:func:`isrc.replot_isr_comparison_tree`, the same function
    ``Postprocessing/regenerate_all_local_mil_isr_comparison_plots.py``
    calls) against the richer, full-vertical-profile ionosonde schema --
    confirms the whole wiring (``_build_batch_isr_matches_from_saved`` ->
    ``ionosonde_profile_window_stats`` -> the plotting functions) works
    for an already-completed run with no pipeline rerun, not just the
    individual pieces tested above."""

    def test_writes_plots_with_profile_schema_ionosonde_file(self, tmp_path):
        altitude, edp = _write_synthetic_run(tmp_path, label="synthrun2", styles=("raw",))
        isr_path = tmp_path / "isr2.nc"
        _write_synthetic_isr_file(
            isr_path, n_time=3, n_alt=5, station_lat=5.0, station_lon=0.0,
            start="2026-01-01T00:30:00",
        )
        with xr.open_dataset(isr_path) as ds:
            ds = ds.load()
        ds = ds.assign(altitude=(("altitude_gate",), altitude.copy()))
        ds.to_netcdf(isr_path, mode="w")

        iono_path = tmp_path / "iono_profile.nc"
        _write_synthetic_ionosonde_profile_file(
            iono_path, n_time=6, n_alt=5, station_lat=5.0, station_lon=0.0,
            start="2026-01-01T00:00:00", freq="20min",
        )

        written = isrc.replot_isr_comparison_tree(tmp_path, isr_path, ionosonde_file_path=iono_path)

        written_set = {p.relative_to(tmp_path) for p in written}
        for batch in (0, 1):
            assert Path(f"synthrun2_isr_comparison/raw/nearest_point/batch{batch:04d}.png") in written_set
        assert Path("synthrun2_isr_comparison/raw/range_comparison.png") in written_set
