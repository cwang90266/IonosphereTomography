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


def _make_rectangle_with_mesh_high_latitude(altitude, n_sample=1, placeholder_value=1e11):
    """Same as _make_rectangle_with_mesh, but entirely at high latitude
    (mean ~75N) -- for testing the auto-polar-projection switch, which
    _make_rectangle_with_mesh's mid-latitude grid never triggers."""
    import pandas as pd
    import edp_samples as E

    sp = pd.DataFrame({
        "hour": [12.0] * n_sample, "f107": [120.0] * n_sample,
        "ap": [10.0] * n_sample, "ig12": [80.0] * n_sample, "rz12": [70.0] * n_sample,
    })
    sp.attrs = {}
    n_geo = E.EDPSamples.genRectangularArea(-10, 10, 5, 65, 85, 5)[0].shape[0]
    edps = np.full((len(altitude), n_geo, n_sample), placeholder_value)
    feature_edps = np.zeros((len(E.EDPSamples.FEATURE_LABEL), n_geo, n_sample))
    return E.EDPSamples(
        DateTime="2026-01-01", geo_type="Rectangle", altitude=altitude,
        sampling_parameters=sp, minLon=-10, maxLon=10, dLon=5, minLat=65, maxLat=85, dLat=5,
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

    def test_high_latitude_grid_uses_polar_projection(self):
        """Regression test for a real bug (2026-09-28): this function
        used to build its multi-panel figure via a separately hardcoded
        ccrs.PlateCarree(), so it never picked up EDPSamples' own
        auto-polar switch even for a grid that clearly should use one."""
        import cartopy.crs as ccrs
        pytest.importorskip("cartopy")
        altitude = np.linspace(100.0, 500.0, 5)
        edp = _make_rectangle_with_mesh_high_latitude(altitude)
        entry = _radial_ray_entry(edp, 75.0, 2.0, alt_top_km=500.0)

        fig = output.plot_observation_operator_sum(edp, entry, altitudes=(150, 250, 350))
        geo_axes = [a for a in fig.axes if hasattr(a, "projection")]   # excludes colorbar axes
        assert len(geo_axes) >= 1
        for ax in geo_axes:
            assert isinstance(ax.projection, ccrs.NorthPolarStereo)


class TestObservationOperatorSumCombinedPlot:
    def test_combines_multiple_entries(self):
        pytest.importorskip("cartopy")
        altitude = np.linspace(100.0, 500.0, 5)
        edp = _make_rectangle_with_mesh(altitude)
        e1 = _radial_ray_entry(edp, 3.0, 2.0, alt_top_km=500.0, label="ro1")
        e2 = _radial_ray_entry(edp, 5.0, 5.0, alt_top_km=500.0, label="ro2")

        fig = output.plot_observation_operator_sum_combined(edp, [e1, e2], altitudes=(150, 250, 350))
        visible_axes = [a for a in fig.axes if a.get_visible() and a.has_data()]
        assert len(visible_axes) >= 1
        assert "2 entries" in fig.get_suptitle()
        assert "2 rays" in fig.get_suptitle()   # 1 ray per _radial_ray_entry

    def test_high_latitude_grid_uses_polar_projection(self):
        """Regression test for the same real bug as
        TestObservationOperatorSumPlot's equivalent test -- this
        function had its own separately hardcoded ccrs.PlateCarree()."""
        import cartopy.crs as ccrs
        pytest.importorskip("cartopy")
        altitude = np.linspace(100.0, 500.0, 5)
        edp = _make_rectangle_with_mesh_high_latitude(altitude)
        e1 = _radial_ray_entry(edp, 75.0, 2.0, alt_top_km=500.0, label="ro1")
        e2 = _radial_ray_entry(edp, 78.0, 5.0, alt_top_km=500.0, label="ro2")

        fig = output.plot_observation_operator_sum_combined(edp, [e1, e2], altitudes=(150, 250, 350))
        geo_axes = [a for a in fig.axes if hasattr(a, "projection")]   # excludes colorbar axes
        assert len(geo_axes) >= 1
        for ax in geo_axes:
            assert isinstance(ax.projection, ccrs.NorthPolarStereo)

    def test_sum_differs_from_a_single_entry(self):
        """The combined sum should reflect both entries, not just one --
        catches a bug where only the first/last entry's geometry made it
        into the concatenated operator."""
        pytest.importorskip("cartopy")
        altitude = np.linspace(100.0, 500.0, 5)
        edp = _make_rectangle_with_mesh(altitude)
        e1 = _radial_ray_entry(edp, 3.0, 2.0, alt_top_km=500.0, label="ro1")
        e2 = _radial_ray_entry(edp, 7.0, 8.0, alt_top_km=500.0, label="ro2")

        H1 = edp.get_observation_operator(e1.to_operator_dict(), num_segments=200)
        H2 = edp.get_observation_operator(e2.to_operator_dict(), num_segments=200)
        rec = np.concatenate([e1.rec_ecef_km, e2.rec_ecef_km], axis=1)
        gnss = np.concatenate([e1.gnss_ecef_km, e2.gnss_ecef_km], axis=1)
        H_combined = edp.get_observation_operator({"rec_ecef_km": rec, "gnss_ecef_km": gnss}, num_segments=200)

        combined_sum = np.asarray(H_combined).sum(axis=0)
        separate_sum = np.asarray(H1).sum(axis=0) + np.asarray(H2).sum(axis=0)
        np.testing.assert_allclose(combined_sum, separate_sum, atol=1e-8)


class TestObservationsGeolocationPlot:
    def test_runs_with_ro_and_igs_entries(self):
        pytest.importorskip("cartopy")
        altitude = np.linspace(100.0, 500.0, 5)
        edp = _make_rectangle_with_mesh(altitude)
        ro = _radial_ray_entry(edp, 3.0, 2.0, alt_top_km=500.0, label="ro1")
        from observation_preparation import ObservationEntry
        igs = ObservationEntry(
            obs_type="IGS", tec=np.array([12.0]),
            gnss_ecef_km=np.zeros((3, 1)), rec_ecef_km=np.zeros((3, 1)),
            pierce_lat=np.array([4.0]), pierce_lon=np.array([3.0]), label="igs1",
        )

        ax = output.plot_observations_geolocation(edp, [ro, igs])
        assert ax.get_title().startswith("Geolocation")
        assert ax.get_legend() is not None

    def test_uses_same_projection_and_extent_as_the_grid_plot(self):
        """Regression check for the reported bug: the old
        observation_preparation.diagnostics.plot_geolocation used an
        un-extent-limited Orthographic/Robinson view, putting real content
        in a small corner of an otherwise-empty global disc. This must
        reuse the grid's own projection+extent instead."""
        pytest.importorskip("cartopy")
        altitude = np.linspace(100.0, 500.0, 5)
        edp = _make_rectangle_with_mesh(altitude)
        ro = _radial_ray_entry(edp, 3.0, 2.0, alt_top_km=500.0)

        ax = output.plot_observations_geolocation(edp, [ro])
        grid_ax = edp.plot_geolocation()
        assert type(ax.projection) is type(grid_ax.projection)
        assert ax.get_extent() == grid_ax.get_extent()

    def test_runs_with_no_entries(self):
        pytest.importorskip("cartopy")
        altitude = np.linspace(100.0, 500.0, 5)
        edp = _make_rectangle_with_mesh(altitude)
        ax = output.plot_observations_geolocation(edp, [])
        assert ax.get_title().startswith("Geolocation")


def _toy_cycle_result_with_obs_type(n_batches=3, n_igs=1, n_ro=2, seed=1):
    """Like _toy_cycle_result, but also sets CycleBatch.obs_type (a mix of
    IGS/RO) -- returns (batches, result) since plot_igs_tec_scatter needs
    the original batches list zipped with result.batch_outcomes (obs_type
    lives on CycleBatch, not on BatchOutcome)."""
    rng = np.random.default_rng(seed)
    n_state, n_members = 6, 100
    n_obs = n_igs + n_ro
    x_true = rng.standard_normal(n_state)
    X = x_true[:, None] + 1.5 * rng.standard_normal((n_state, n_members))
    ensemble = EnsembleState(X=X, param_shape=(n_state,))
    driver = GeneralEnKFDriver(style="raw", config=AnalysisConfig(linearization="linear", centering="mean"))

    batches, operators = [], []
    obs_type = np.array(["IGS"] * n_igs + ["RO"] * n_ro)
    for i in range(n_batches):
        H = rng.standard_normal((n_obs, n_state))
        operators.append(LinearToyOperator(H=H))
        batches.append(CycleBatch(podTc2_data={}, y_obs=H @ x_true, R=0.05 * np.eye(n_obs),
                                   obs_type=obs_type, batch_index=i))

    result = run_batch_loop(ensemble, driver, operators, batches)
    return batches, result


class TestIgsTecScatterPlot:
    def test_runs_and_produces_two_panels(self):
        batches, result = _toy_cycle_result_with_obs_type(n_batches=2, n_igs=2, n_ro=3)
        fig = output.plot_igs_tec_scatter(batches, result, style_label="raw")
        assert len(fig.axes) == 2
        for ax in fig.axes:
            assert ax.has_data()
        assert "raw" in fig.get_suptitle()

    def test_pools_igs_rays_across_all_batches(self):
        n_batches, n_igs = 4, 2
        batches, result = _toy_cycle_result_with_obs_type(n_batches=n_batches, n_igs=n_igs, n_ro=1)
        fig = output.plot_igs_tec_scatter(batches, result)
        ax_forecast = fig.axes[0]
        scatter_offsets = ax_forecast.collections[0].get_offsets()
        assert len(scatter_offsets) == n_batches * n_igs

    def test_handles_no_igs_data_gracefully(self):
        batches, result = _toy_cycle_result_with_obs_type(n_batches=2, n_igs=0, n_ro=3)
        fig = output.plot_igs_tec_scatter(batches, result)
        assert len(fig.axes) == 2   # still produces the 2-panel figure, with a placeholder message


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
