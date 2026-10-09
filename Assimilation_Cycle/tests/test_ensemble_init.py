# -*- coding: utf-8 -*-
"""
Tests for ensemble_init.py (plan Section 4.4/4.9).

The precomputed-file-mode tests need no IRI2020 executable (they load an
existing EDPSamples/Parameterized_EDPSamples netCDF instead). The "fresh"
path tests run real IRI2020 over a deliberately tiny grid/ensemble --
skipped when IRI2020_PATH (or the compiled driver it should point at)
isn't available in this environment, same convention as
Ensemble_Kalman_Engine/tests/test_end_to_end_real_data.py's real-data skip.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from Assimilation_Cycle.cycle_config import CycleConfig
from Assimilation_Cycle import ensemble_init

_REPO_ROOT = Path(__file__).resolve().parents[2]
_IRI2020_DIR = _REPO_ROOT / "iri2020_new" / "src" / "iri2020"
_IRI2020_EXE = _IRI2020_DIR / "iri2020_namelist_driver"
_EDP_FIXTURE = _REPO_ROOT / "TestCode" / "EDPSam_Point.nc"

_has_iri2020 = _IRI2020_EXE.exists() and os.access(_IRI2020_EXE, os.X_OK)


@pytest.fixture(autouse=True)
def _iri2020_env(monkeypatch):
    if _has_iri2020:
        monkeypatch.setenv("IRI2020_PATH", str(_IRI2020_DIR))


def _make_cfg(**overrides) -> CycleConfig:
    defaults = dict(
        start_time="2024-06-15T00:00:00", end_time="2024-06-15T01:00:00",
        center_lat=69.6, center_lon=18.9, radius_km=5.0,
        altitude_grid=np.arange(100.0, 401.0, 100.0),
        horizontal_resolution_deg=5.0,
        n_ensemble=3,
        style="raw",
        iri_spread_kwargs={
            "hour_sample_range": 0, "f107_sample_range": 2, "ap_sample_range": 1,
            "ig_sample_range": 1, "rz_sample_range": 1,
        },
    )
    defaults.update(overrides)
    return CycleConfig(**defaults)


@pytest.mark.skipif(not _has_iri2020, reason="no compiled IRI2020 driver in this environment")
class TestFreshBuildRealIri2020:
    def test_edp_samples_shape_matches_grid_and_ensemble(self):
        cfg = _make_cfg()
        edp = ensemble_init.load_or_build_edp_samples(cfg)
        n_height, n_geo, n_sample = edp.edps.shape
        assert n_height == len(cfg.altitude_grid)
        assert n_sample == cfg.n_ensemble
        assert n_geo >= 1

    def test_saves_and_reloads_edp_samples(self, tmp_path):
        out = tmp_path / "edp_samples.nc"
        cfg = _make_cfg(edp_samples_output_path=out)
        edp = ensemble_init.load_or_build_edp_samples(cfg)
        assert out.exists()

        cfg_reload = _make_cfg(edp_samples_path=out)
        reloaded = ensemble_init.load_or_build_edp_samples(cfg_reload)
        np.testing.assert_allclose(reloaded.edps, edp.edps)

    def test_full_build_end_to_end(self):
        cfg = _make_cfg(style="density_10ex")
        ensemble, edp_samples, parameterization = ensemble_init.build(cfg)
        n_height, n_geo, n_sample = edp_samples.edps.shape
        assert ensemble.n_members == cfg.n_ensemble == n_sample
        assert ensemble.param_shape == (n_height, n_geo)
        assert ensemble.X.shape == (n_height * n_geo, n_sample)

    def test_zero_spread_on_every_index_no_longer_crashes(self):
        """Regression check for the pd.isna vs. np.isnan fix in
        write_IRI2020_namelist -- an all-None driving-index column
        (no spread requested anywhere) must not raise."""
        cfg = _make_cfg(iri_spread_kwargs={})
        edp = ensemble_init.load_or_build_edp_samples(cfg)
        assert edp.edps.shape[-1] == cfg.n_ensemble


class TestDiagonalBoostWiring:
    def test_disabled_by_default_returns_identical_edps(self, monkeypatch):
        monkeypatch.delenv("IRI2020_PATH", raising=False)
        cfg = _make_cfg(edp_samples_path=_EDP_FIXTURE)
        edp = ensemble_init.load_or_build_edp_samples(cfg)

        import edp_samples as E
        original = E.EDPSamples.fromNetCDF(str(_EDP_FIXTURE))
        np.testing.assert_array_equal(edp.edps, original.edps)

    def test_enabled_changes_the_loaded_ensemble(self, monkeypatch):
        monkeypatch.delenv("IRI2020_PATH", raising=False)
        cfg = _make_cfg(
            edp_samples_path=_EDP_FIXTURE,
            diagonal_boost_amplitude=0.5, diagonal_boost_rng_seed=1,
        )
        boosted = ensemble_init.load_or_build_edp_samples(cfg)

        import edp_samples as E
        original = E.EDPSamples.fromNetCDF(str(_EDP_FIXTURE))
        assert not np.allclose(boosted.edps, original.edps)
        assert boosted.edps.shape == original.edps.shape

    def test_seeded_boost_is_reproducible_through_the_config(self, monkeypatch):
        monkeypatch.delenv("IRI2020_PATH", raising=False)
        cfg = _make_cfg(
            edp_samples_path=_EDP_FIXTURE,
            diagonal_boost_amplitude=0.4, diagonal_boost_rng_seed=7,
        )
        r1 = ensemble_init.load_or_build_edp_samples(cfg)
        r2 = ensemble_init.load_or_build_edp_samples(cfg)
        np.testing.assert_array_equal(r1.edps, r2.edps)

    def test_boost_applied_before_save(self, tmp_path, monkeypatch):
        monkeypatch.delenv("IRI2020_PATH", raising=False)
        out = tmp_path / "boosted_edp_samples.nc"
        cfg = _make_cfg(
            edp_samples_path=_EDP_FIXTURE,
            diagonal_boost_amplitude=0.5, diagonal_boost_rng_seed=2,
            edp_samples_output_path=out,
        )
        boosted = ensemble_init.load_or_build_edp_samples(cfg)
        assert out.exists()

        import edp_samples as E
        reloaded = E.EDPSamples.fromNetCDF(str(out))
        np.testing.assert_allclose(reloaded.edps, boosted.edps)


def _synthetic_isr_basis_for_fixture(tmp_path, altitude_grid):
    """A small synthetic ISR PCA basis, on the exact altitude grid
    ``_EDP_FIXTURE`` uses, for ``TestDiagonalBoostWithIsrPcaStyle`` below
    (no real ISR/IRI2020-extension file needed -- see
    test_isr_pca_basis.py for the module's own dedicated unit tests)."""
    import xarray as xr
    from Assimilation_Cycle import isr_pca_basis as ipb

    rng = np.random.default_rng(0)
    n_profile = 40
    mode = np.exp(-((altitude_grid - 300.0) / 150.0) ** 2) * 2.0 + 9.0
    log_density = mode[:, None] + rng.normal(0.0, 0.3, size=(1, n_profile))
    density = (10.0 ** log_density).astype(np.float32)

    extended_path = tmp_path / "synthetic_extended_isr.nc"
    xr.Dataset(
        data_vars=dict(
            altitude=(("altitude_gate",), altitude_grid),
            Ne=(("altitude_gate", "time"), density),
        ),
    ).to_netcdf(extended_path)

    basis = ipb.build_isr_pca_basis(extended_path, altitude_grid, retaining_threshold=0.99)
    basis_path = tmp_path / "isr_pca_basis.nc"
    ipb.save_isr_pca_basis(basis, basis_path)
    return basis_path


class TestDiagonalBoostWithIsrPcaStyle:
    """Plan Section 4.1.4: diagonal_boost.py operates on the raw density
    ensemble strictly before parameterization, so it should need no
    changes to support 'PCA_1D_10ex_ISR' -- this is the wiring check
    confirming that combination (log-space boosting, required for every
    log10-based style per diagonal_boost.py's own module docstring)
    actually runs end to end, never exercised before this style existed."""

    def test_boost_log_space_plus_isr_pca_style_builds_successfully(self, tmp_path, monkeypatch):
        monkeypatch.delenv("IRI2020_PATH", raising=False)
        import edp_samples as E
        original = E.EDPSamples.fromNetCDF(str(_EDP_FIXTURE))
        altitude_grid = np.asarray(original.altitude, dtype=float)
        basis_path = _synthetic_isr_basis_for_fixture(tmp_path, altitude_grid)

        cfg = _make_cfg(
            edp_samples_path=_EDP_FIXTURE,
            altitude_grid=altitude_grid,
            style="PCA_1D_10ex_ISR",
            isr_pca_basis_path=basis_path,
            diagonal_boost_amplitude=0.3, diagonal_boost_log_space=True, diagonal_boost_rng_seed=3,
        )
        ensemble, edp_samples, parameterization = ensemble_init.build(cfg)

        assert ensemble.n_members == edp_samples.edps.shape[-1]
        # Boosting changed the raw ensemble (same assertion style as
        # TestDiagonalBoostWiring above) ...
        assert not np.allclose(np.asarray(edp_samples.edps), np.asarray(original.edps))
        # ... and the ISR-basis style still successfully encoded it (a
        # density/PCA altitude-grid mismatch would have raised here, the
        # real bug this exact combination surfaced during real-data
        # smoke-testing -- see CycleConfig.isr_pca_basis_path's docstring).
        assert parameterization.hyper_params["PCA"].shape[0] == len(altitude_grid)


class TestPrecomputedFileMode:
    def test_loads_edp_samples_without_running_iri2020(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("IRI2020 should not run when edp_samples_path is set")
        monkeypatch.delenv("IRI2020_PATH", raising=False)

        cfg = _make_cfg(edp_samples_path=_EDP_FIXTURE)
        edp = ensemble_init.load_or_build_edp_samples(cfg)
        assert edp.edps.shape[0] > 0

    def test_build_from_precomputed_edp_samples_skips_iri_selection(self, monkeypatch):
        monkeypatch.delenv("IRI2020_PATH", raising=False)

        def _boom(*a, **k):
            raise AssertionError("sampling_parameters_for_cycle should not be called")
        monkeypatch.setattr(ensemble_init, "sampling_parameters_for_cycle", _boom)

        cfg = _make_cfg(edp_samples_path=_EDP_FIXTURE, style="raw")
        ensemble, edp_samples, parameterization = ensemble_init.build(cfg)
        assert ensemble.n_members == edp_samples.edps.shape[-1]

    def test_build_from_precomputed_parameterized_edp_samples_skips_edp_build(self, monkeypatch, tmp_path):
        monkeypatch.delenv("IRI2020_PATH", raising=False)

        def _boom(*a, **k):
            raise AssertionError("load_or_build_edp_samples should not be called")
        monkeypatch.setattr(ensemble_init, "load_or_build_edp_samples", _boom)

        # Build the fixture once (real EDPSamples.fromNetCDF + a cheap style)
        import edp_samples as E
        from Parameterization import Parameterized_EDPSamples
        edp = E.EDPSamples.fromNetCDF(str(_EDP_FIXTURE))
        # keep this fast: subsample members
        edp_small = E.EDPSamples.from_xarray(edp.isel(sample=slice(0, 20)))
        pes = Parameterized_EDPSamples(edp_small, style="raw")
        pes_path = tmp_path / "pes.nc"
        pes.saveNetCDF(str(pes_path))

        cfg = _make_cfg(parameterized_edp_samples_path=pes_path, style="raw")
        ensemble, edp_samples, parameterization = ensemble_init.build(cfg)
        assert ensemble.n_members == 20
