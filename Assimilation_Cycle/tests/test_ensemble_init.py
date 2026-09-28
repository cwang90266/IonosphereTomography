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
