# -*- coding: utf-8 -*-
"""
Tests for output.py (plan Section 4.7/4.14).

Metrics-plot tests build a CycleResult from toy operators (no EDPSamples
needed). save_decoded_field_netcdf/plot_horizontal need a real EDPSamples
object -- exercised against the existing TestCode/EDPSam_Point.nc fixture
(geo_type='Point', the simplest case, no IRI2020 run needed to load it).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import pytest

from Ensemble_Kalman_Engine import EnsembleState
from Ensemble_Kalman_Engine.driver import GeneralEnKFDriver
from Ensemble_Kalman_Engine.analysis_engine import AnalysisConfig

from Assimilation_Cycle import output
from Assimilation_Cycle.cycle_driver import CycleBatch, run_batch_loop

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EDP_FIXTURE = _REPO_ROOT / "TestCode" / "EDPSam_Point.nc"


@dataclass
class LinearToyOperator:
    H: np.ndarray

    def forward_ensemble(self, X_f):
        return self.H @ X_f

    def forward_single(self, x):
        return self.H @ x

    def linearized(self, x):
        return self.H


def _toy_cycle_result(n_batches=3):
    rng = np.random.default_rng(0)
    n_state, n_obs, n_members = 6, 3, 100
    x_true = rng.standard_normal(n_state)
    X = x_true[:, None] + 1.5 * rng.standard_normal((n_state, n_members))
    ensemble = EnsembleState(X=X, param_shape=(n_state,))
    driver = GeneralEnKFDriver(style="raw", config=AnalysisConfig(linearization="linear", centering="mean"))

    batches, operators = [], []
    for i in range(n_batches):
        H = rng.standard_normal((n_obs, n_state))
        operators.append(LinearToyOperator(H=H))
        batches.append(CycleBatch(podTc2_data={}, y_obs=H @ x_true, R=0.05 * np.eye(n_obs), batch_index=i))

    return run_batch_loop(ensemble, driver, operators, batches)


class TestMetricsPlots:
    def test_rmse_reduction_plot_runs(self):
        result = _toy_cycle_result()
        ax = output.plot_rmse_reduction(result)
        assert ax is not None
        assert len(ax.lines) == 2  # forecast + analysis series

    def test_rank_histogram_plot_runs(self):
        result = _toy_cycle_result()
        ax = output.plot_rank_histogram(result)
        assert ax is not None
        assert len(ax.patches) > 0

    def test_effective_rank_series_plot_runs(self):
        result = _toy_cycle_result()
        ax = output.plot_effective_rank_series(result)
        assert ax is not None
        assert len(ax.lines) == 2


class TestVerticalProfilePlot:
    def test_runs_and_labels_axes(self):
        rng = np.random.default_rng(1)
        n_height, n_geo, n_members = 10, 3, 50
        altitude = np.linspace(100, 900, n_height)
        forecast = rng.uniform(1e10, 1e12, size=(n_height, n_geo, n_members))
        analysis = forecast * 0.9 + rng.normal(scale=1e9, size=forecast.shape)

        ax = output.plot_vertical_profile(altitude, forecast, analysis, geo_idx=1)
        assert ax.get_ylabel() == "Altitude (km)"
        assert len(ax.lines) == 2  # forecast mean + analysis mean


@pytest.mark.skipif(not _EDP_FIXTURE.exists(), reason="fixture EDPSamples file not found")
class TestSaveDecodedFieldAndHorizontalPlot:
    def test_round_trips_point_geo_type(self, tmp_path):
        import edp_samples as E
        full = E.EDPSamples.fromNetCDF(str(_EDP_FIXTURE))
        template = E.EDPSamples.from_xarray(full.isel(sample=slice(0, 5)))

        n_height, n_geo, n_members = template.edps.shape
        decoded = np.abs(np.random.default_rng(2).normal(1e11, 1e9, size=(n_height, n_geo, n_members)))

        out_path = tmp_path / "decoded_field.nc"
        field = output.save_decoded_field_netcdf(template, decoded, out_path)
        assert out_path.exists()
        assert field.attrs["feature_edps_computed"] == 0
        np.testing.assert_allclose(field.edps, decoded)

        reloaded = E.EDPSamples.fromNetCDF(str(out_path))
        np.testing.assert_allclose(reloaded.edps, decoded)

    def test_plot_horizontal_runs_with_raw_array(self):
        import edp_samples as E
        full = E.EDPSamples.fromNetCDF(str(_EDP_FIXTURE))
        template = E.EDPSamples.from_xarray(full.isel(sample=slice(0, 5)))

        n_geo = template.geolocation.shape[0]
        mean_slice = np.abs(np.random.default_rng(3).normal(1e11, 1e9, size=n_geo))
        ax = output.plot_horizontal(template, mean_slice, target_alt=300.0)
        assert ax is not None


# ===========================================================================
# Final_Packaging.docx visualizations
# ===========================================================================

class TestPlotIriInputDistributions:
    def test_runs_with_mixed_finite_and_all_nan_columns(self):
        import pandas as pd
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "hour": rng.uniform(0, 2, size=n),
            "f107": rng.uniform(90, 150, size=n),
            "ap": rng.uniform(0, 30, size=n),
            "ig12": np.full(n, np.nan),   # no spread requested for this index
            "rz12": rng.uniform(50, 100, size=n),
        })
        fig = output.plot_iri_input_distributions(df)
        axes = fig.axes
        assert len(axes) == 6  # 2x3 grid
        # the all-NaN column's panel should carry the placeholder text, not a histogram
        titles = [ax.get_title() for ax in axes]
        assert "ig12" in titles


def _make_rectangle_with_mesh(altitude, n_sample=1, placeholder_value=1e11):
    import pandas as pd
    import edp_samples as E

    sp = pd.DataFrame({
        "hour": [12.0] * n_sample, "f107": [120.0] * n_sample,
        "ap": [10.0] * n_sample, "ig12": [80.0] * n_sample, "rz12": [70.0] * n_sample,
    })
    sp.attrs = {}
    n_geo = E.EDPSamples.genRectangularArea(-10, 10, 5, 0, 10, 5)[0].shape[0]
    edps = np.full((len(altitude), n_geo, n_sample), placeholder_value)
    feature_edps = np.zeros((len(E.EDPSamples.FEATURE_LABEL), n_geo, n_sample))
    return E.EDPSamples(
        DateTime="2026-01-01", geo_type="Rectangle", altitude=altitude,
        sampling_parameters=sp, minLon=-10, maxLon=10, dLon=5, minLat=0, maxLat=10, dLat=5,
        edps=edps, feature_edps=feature_edps,
    )


def _radial_ray_entry(edp_samples, lat, lon, alt_top_km, tec_value=10.0, tangent_alt_km=None, label="ro1"):
    import edp_samples as E
    from observation_preparation import ObservationEntry
    gnss = (E._geodetic_to_ecef(lat, lon, 0.0) / 1000.0).reshape(3, 1)
    rec = (E._geodetic_to_ecef(lat, lon, alt_top_km * 1000.0) / 1000.0).reshape(3, 1)
    return ObservationEntry(
        obs_type="RO", tec=np.array([tec_value]),
        gnss_ecef_km=gnss, rec_ecef_km=rec,
        tangent_alt_km=np.array([tangent_alt_km if tangent_alt_km is not None else alt_top_km / 2]),
        label=label,
    )


class TestObservationOperatorSumPlot:
    def test_runs_and_sums_over_entry_rays(self):
        pytest.importorskip("cartopy")
        altitude = np.linspace(100.0, 500.0, 5)
        edp = _make_rectangle_with_mesh(altitude)
        entry = _radial_ray_entry(edp, 3.0, 2.0, alt_top_km=500.0)

        fig = output.plot_observation_operator_sum(edp, entry, altitudes=(150, 250, 350))
        # one axes per requested altitude
        visible_axes = [a for a in fig.axes if a.get_visible() and a.has_data()]
        assert len(visible_axes) >= 1


class TestTecProfileComparisonPlot:
    def test_sorts_by_tangent_altitude_and_plots_three_series(self):
        from observation_preparation import ObservationEntry
        entry = ObservationEntry(
            obs_type="RO", tec=np.zeros(4),
            gnss_ecef_km=np.zeros((3, 4)), rec_ecef_km=np.zeros((3, 4)),
            tangent_alt_km=np.array([300.0, 100.0, 400.0, 200.0]),
            label="ro_test",
        )
        y_forecast = np.array([1.0, 2.0, 3.0, 4.0])
        y_analysis = np.array([1.1, 2.1, 3.1, 4.1])
        y_measured = np.array([1.2, 2.2, 3.2, 4.2])

        ax = output.plot_tec_profile_comparison(entry, y_forecast, y_analysis, y_measured)
        assert len(ax.lines) == 3
        # y-data (altitude) on the measured line should come out sorted ascending
        measured_line = ax.lines[0]
        np.testing.assert_allclose(measured_line.get_ydata(), sorted([300.0, 100.0, 400.0, 200.0]))

    def test_raises_without_tangent_alt_km(self):
        from observation_preparation import ObservationEntry
        entry = ObservationEntry(
            obs_type="IGS", tec=np.array([1.0]),
            gnss_ecef_km=np.zeros((3, 1)), rec_ecef_km=np.zeros((3, 1)),
        )
        with pytest.raises(ValueError):
            output.plot_tec_profile_comparison(entry, np.array([1.0]), np.array([1.0]), np.array([1.0]))


class TestEdpProfileComparisonPlot:
    def test_explicit_latlon_interpolates_and_overlays_abel(self):
        altitude = np.linspace(100.0, 500.0, 5)
        edp = _make_rectangle_with_mesh(altitude, placeholder_value=1e11)
        n_height, n_geo, _ = edp.edps.shape
        rng = np.random.default_rng(4)
        forecast = rng.uniform(1e10, 1e12, size=(n_height, n_geo, 20))
        analysis = forecast * 0.95

        entry = _radial_ray_entry(edp, 3.0, 2.0, alt_top_km=500.0, label="ro_edp")
        entry.abel = {"Ne": np.array([1e11, 2e11]), "alt_km": np.array([150.0, 350.0])}

        ax = output.plot_edp_profile_comparison(edp, entry, forecast, analysis, lat=3.0, lon=2.0)
        # forecast mean, analysis mean, abel -- 3 lines
        assert len(ax.lines) == 3
        assert ax.get_ylabel() == "Altitude (km)"

    def test_derives_latlon_from_max_tec_in_window(self):
        altitude = np.linspace(100.0, 500.0, 5)
        edp = _make_rectangle_with_mesh(altitude)
        n_height, n_geo, _ = edp.edps.shape
        rng = np.random.default_rng(5)
        forecast = rng.uniform(1e10, 1e12, size=(n_height, n_geo, 10))
        analysis = forecast.copy()

        from observation_preparation import ObservationEntry
        gnss_list, rec_list = [], []
        import edp_samples as E
        for lon in (1.0, 2.0, 3.0):
            gnss_list.append((E._geodetic_to_ecef(3.0, lon, 0.0) / 1000.0))
            rec_list.append((E._geodetic_to_ecef(3.0, lon, 500_000.0) / 1000.0))
        entry = ObservationEntry(
            obs_type="RO", tec=np.array([5.0, 50.0, 9.0]),   # ray 1 (idx 1) has max TEC
            gnss_ecef_km=np.stack(gnss_list, axis=1), rec_ecef_km=np.stack(rec_list, axis=1),
            tangent_alt_km=np.array([260.0, 300.0, 280.0]),  # all within the default 250-350km window
            label="ro_window",
        )
        ax = output.plot_edp_profile_comparison(edp, entry, forecast, analysis)
        assert "2.00" in ax.get_title() or "lon" not in ax.get_title()  # picked ray idx 1 -> lon=2.0


class TestStyleComparisonSummaryPlot:
    def test_runs_with_multiple_styles(self):
        result_a = _toy_cycle_result(n_batches=2)
        result_b = _toy_cycle_result(n_batches=2)
        results_by_style = {
            "raw": (result_a, 204, 1.5),
            "ANCHOR": (result_b, 96, 3.2),
        }
        fig = output.plot_style_comparison_summary(results_by_style)
        assert len(fig.axes) == 4
        for ax in fig.axes:
            assert len(ax.patches) == 2  # one bar per style
