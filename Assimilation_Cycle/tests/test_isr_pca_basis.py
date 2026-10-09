# -*- coding: utf-8 -*-
"""
Tests for isr_pca_basis.py (ISR_Integration_Plan.md Section 4.1).

All synthetic -- no real IRI2020-extended ISR file needed (that real file
is exercised separately, in a one-off real-data smoke script, not in this
unit suite).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from Assimilation_Cycle import isr_pca_basis as ipb


def _write_synthetic_extended_isr_file(path, n_alt=20, n_profile=50, n_bad=0, seed=0):
    """A synthetic 'extended ISR' file: two shared modes (a smooth
    altitude-dependent shape plus a scaled perturbation) so get_PCA has
    real, known structure to recover, not pure noise."""
    rng = np.random.default_rng(seed)
    altitude = np.linspace(100.0, 800.0, n_alt)
    mode0 = np.exp(-((altitude - 300.0) / 150.0) ** 2) * 2.0 + 9.0   # log10(Ne)-like shape
    mode1 = (altitude - 450.0) / 400.0

    coeff0 = rng.normal(0.0, 1.0, size=n_profile)
    coeff1 = rng.normal(0.0, 0.3, size=n_profile)
    log_density = mode0[:, None] + coeff0[None, :] * 0.5 + mode1[:, None] * coeff1[None, :]
    density = (10.0 ** log_density).astype(np.float32)

    if n_bad > 0:
        density[:, :n_bad] = np.nan   # profiles that failed extension entirely

    ds = xr.Dataset(
        data_vars=dict(
            altitude=(("altitude_gate",), altitude, {"units": "km"}),
            Ne=(("altitude_gate", "time"), density, {"units": "m-3"}),
        ),
    )
    ds.to_netcdf(path)
    return altitude, density


class TestResampleIsrProfilesToGrid:
    def test_matches_hand_computed_log_linear_interpolation(self):
        isr_altitude = np.array([100.0, 200.0, 300.0])
        isr_density = np.array([[1e10], [1e11], [1e12]])  # one profile, exact powers of 10
        altitude_grid = np.array([150.0, 250.0])

        log_density = ipb.resample_isr_profiles_to_grid(isr_altitude, isr_density, altitude_grid)
        # log10 is exactly linear in altitude here (10, 11, 12 at 100/200/300),
        # so log-linear interpolation at the midpoints is exact.
        np.testing.assert_allclose(log_density[:, 0], [10.5, 11.5], atol=1e-10)

    def test_out_of_range_grid_point_is_nan(self):
        isr_altitude = np.array([100.0, 200.0])
        isr_density = np.array([[1e10], [1e11]])
        altitude_grid = np.array([50.0, 150.0, 900.0])

        log_density = ipb.resample_isr_profiles_to_grid(isr_altitude, isr_density, altitude_grid)
        assert np.isnan(log_density[0, 0])    # below range
        assert np.isfinite(log_density[1, 0])  # inside range
        assert np.isnan(log_density[2, 0])    # above range


class TestBuildIsrPcaBasis:
    def test_builds_basis_with_expected_shapes(self, tmp_path):
        path = tmp_path / "extended.nc"
        _write_synthetic_extended_isr_file(path, n_alt=20, n_profile=80)
        altitude_grid = np.linspace(120.0, 750.0, 15)

        basis = ipb.build_isr_pca_basis(path, altitude_grid, retaining_threshold=0.99)
        assert basis.PCA.shape[0] == len(altitude_grid)
        assert basis.PCA.shape[1] == basis.diagnostics.n_retained
        assert basis.PCA_mean.shape == (len(altitude_grid),)
        assert basis.diagnostics.n_profiles_used == 80
        assert basis.diagnostics.n_profiles_total == 80
        # Two real modes were injected -- a handful of components should
        # already explain the great majority of variance.
        assert basis.diagnostics.n_retained <= 5

    def test_drops_profiles_below_min_valid_fraction(self, tmp_path):
        path = tmp_path / "extended.nc"
        _write_synthetic_extended_isr_file(path, n_alt=20, n_profile=50, n_bad=10)
        altitude_grid = np.linspace(120.0, 750.0, 15)

        basis = ipb.build_isr_pca_basis(path, altitude_grid, retaining_threshold=0.99, min_valid_fraction=0.5)
        assert basis.diagnostics.n_profiles_total == 50
        assert basis.diagnostics.n_profiles_used == 40

    def test_raises_if_no_profile_survives(self, tmp_path):
        path = tmp_path / "extended.nc"
        _write_synthetic_extended_isr_file(path, n_alt=20, n_profile=5, n_bad=5)
        altitude_grid = np.linspace(120.0, 750.0, 15)

        with pytest.raises(ValueError):
            ipb.build_isr_pca_basis(path, altitude_grid, retaining_threshold=0.99)

    def test_raises_if_altitude_grid_outside_isr_range(self, tmp_path):
        path = tmp_path / "extended.nc"
        _write_synthetic_extended_isr_file(path, n_alt=10, n_profile=20)
        altitude_grid = np.linspace(2000.0, 3000.0, 5)  # entirely outside

        with pytest.raises(ValueError):
            ipb.build_isr_pca_basis(path, altitude_grid, retaining_threshold=0.99)


class TestSaveLoadRoundTrip:
    def test_round_trips(self, tmp_path):
        src = tmp_path / "extended.nc"
        _write_synthetic_extended_isr_file(src, n_alt=20, n_profile=60)
        altitude_grid = np.linspace(120.0, 750.0, 12)
        basis = ipb.build_isr_pca_basis(src, altitude_grid, retaining_threshold=0.999)

        out_path = tmp_path / "basis.nc"
        ipb.save_isr_pca_basis(basis, out_path)
        reloaded = ipb.load_isr_pca_basis(out_path)

        np.testing.assert_allclose(reloaded.altitude, basis.altitude)
        np.testing.assert_allclose(reloaded.PCA, basis.PCA)
        np.testing.assert_allclose(reloaded.PCA_mean, basis.PCA_mean)
        assert reloaded.diagnostics.n_retained == basis.diagnostics.n_retained
        assert reloaded.diagnostics.n_profiles_used == basis.diagnostics.n_profiles_used


class TestHyperParamsFromBasis:
    def test_returns_pca_and_mean(self, tmp_path):
        path = tmp_path / "extended.nc"
        _write_synthetic_extended_isr_file(path, n_alt=10, n_profile=20)
        altitude_grid = np.linspace(150.0, 700.0, 8)
        basis = ipb.build_isr_pca_basis(path, altitude_grid, retaining_threshold=0.99)

        hp = ipb.hyper_params_from_basis(basis)
        assert set(hp) == {"PCA", "PCA_mean"}
        np.testing.assert_array_equal(hp["PCA"], basis.PCA)


@dataclass
class _FakeCfg:
    isr_pca_basis_path: str | None = None


class TestResolveHyperParamsForStyle:
    def test_noop_for_other_styles(self):
        cfg = _FakeCfg(isr_pca_basis_path="/doesnt/matter.nc")
        hp = {"retaining_threshold": 0.99}
        assert ipb.resolve_hyper_params_for_style(cfg, "PCA_1D_10ex", hp) is hp

    def test_noop_when_pca_already_supplied(self):
        cfg = _FakeCfg(isr_pca_basis_path="/doesnt/exist.nc")
        pca = np.zeros((3, 2))
        hp = {"PCA": pca}
        result = ipb.resolve_hyper_params_for_style(cfg, "PCA_1D_10ex_ISR", hp)
        assert result["PCA"] is pca   # not reloaded from the (nonexistent) path

    def test_raises_style_without_path_or_pca_leaves_empty(self):
        cfg = _FakeCfg(isr_pca_basis_path=None)
        result = ipb.resolve_hyper_params_for_style(cfg, "PCA_1D_10ex_ISR", None)
        assert result == {}   # caller (Parameterized_EDPSamples) is what actually raises

    def test_loads_and_merges_from_path(self, tmp_path):
        src = tmp_path / "extended.nc"
        _write_synthetic_extended_isr_file(src, n_alt=10, n_profile=20)
        altitude_grid = np.linspace(150.0, 700.0, 8)
        basis = ipb.build_isr_pca_basis(src, altitude_grid, retaining_threshold=0.99)
        basis_path = tmp_path / "basis.nc"
        ipb.save_isr_pca_basis(basis, basis_path)

        cfg = _FakeCfg(isr_pca_basis_path=str(basis_path))
        hp = ipb.resolve_hyper_params_for_style(cfg, "PCA_1D_10ex_ISR", {"some_other_key": 1})
        assert "PCA" in hp and "PCA_mean" in hp
        assert hp["some_other_key"] == 1
        np.testing.assert_allclose(hp["PCA"], basis.PCA)
