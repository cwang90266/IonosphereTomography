# -*- coding: utf-8 -*-
"""
Tests for style_sweep.py (plan Section 4.12): real IRI2020 (tiny grid,
shared across styles), observation_stream.assemble monkeypatched to a
synthetic vertical-ray batch (no real RO/IGS data needed -- same
stand-in-geometry rationale as test_run_cycle.py).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from Assimilation_Cycle.cycle_config import CycleConfig
from Assimilation_Cycle import style_sweep, observation_stream
from Assimilation_Cycle.cycle_driver import CycleBatch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_IRI2020_DIR = _REPO_ROOT / "iri2020_new" / "src" / "iri2020"
_IRI2020_EXE = _IRI2020_DIR / "iri2020_namelist_driver"
_has_iri2020 = _IRI2020_EXE.exists() and os.access(_IRI2020_EXE, os.X_OK)


@pytest.fixture(autouse=True)
def _iri2020_env(monkeypatch):
    if _has_iri2020:
        monkeypatch.setenv("IRI2020_PATH", str(_IRI2020_DIR))


def _make_cfg(**overrides) -> CycleConfig:
    defaults = dict(
        start_time="2024-06-15T00:00:00", end_time="2024-06-15T01:00:00",
        center_lat=69.6, center_lon=18.9, radius_km=5.0,
        altitude_grid=np.arange(100.0, 601.0, 100.0),
        horizontal_resolution_deg=5.0,
        n_ensemble=30,
        batch_size=5,
        obs_sigma=1.0,
        iri_spread_kwargs={
            "hour_sample_range": 0, "f107_sample_range": 3, "ap_sample_range": 2,
            "ig_sample_range": 1, "rz_sample_range": 1,
        },
    )
    defaults.update(overrides)
    return CycleConfig(**defaults)


def _fake_assemble(edp_samples_holder):
    """Returns a monkeypatch-ready assemble() that builds vertical-ray
    batches once EDPSamples is known -- style_sweep builds EDPSamples
    itself, so this reads it back from a mutable holder set by the caller
    after the real edp_samples build happens inside run_style_sweep."""
    def _assemble(cfg):
        import edp_samples as E
        edp = edp_samples_holder["edp"]
        geolocation = edp.geolocation
        n_geo = geolocation.shape[0]
        alt_top_km = float(edp.altitude.max())
        rng = np.random.default_rng(0)
        batches = []
        for b in range(2):
            gnss_list, rec_list = [], []
            for _ in range(3):
                geo_idx = rng.integers(0, n_geo)
                lon, lat = geolocation[geo_idx]
                gnss_list.append(E._geodetic_to_ecef(lat, lon, 50_000.0) / 1000.0)
                rec_list.append(E._geodetic_to_ecef(lat, lon, alt_top_km * 1000.0) / 1000.0)
            podTc2_data = {
                "rec_ecef_km": np.stack(rec_list, axis=1),
                "gnss_ecef_km": np.stack(gnss_list, axis=1),
            }
            batches.append(CycleBatch(podTc2_data=podTc2_data, y_obs=np.full(3, 10.0),
                                       R=1.0 * np.eye(3), batch_index=b))
        return batches
    return _assemble


@pytest.mark.skipif(not _has_iri2020, reason="no compiled IRI2020 driver in this environment")
class TestStyleSweep:
    def test_shares_edp_samples_across_styles(self, monkeypatch):
        cfg = _make_cfg()
        build_calls = []
        from Assimilation_Cycle import ensemble_init
        real_build = ensemble_init.load_or_build_edp_samples

        holder = {}

        def _tracked_build(c):
            edp = real_build(c)
            build_calls.append(edp)
            holder["edp"] = edp
            return edp

        monkeypatch.setattr(style_sweep.ensemble_init, "load_or_build_edp_samples", _tracked_build)

        # observation_stream.assemble needs edp_samples's own grid for
        # ray geometry, but is called (once) before ensemble_init inside
        # run_style_sweep -- patch it to a closure reading the same holder,
        # filled in by the *first* real build call triggered from within.
        def _lazy_assemble(c):
            # trigger a real build once, up front, purely to get grid info
            # for synthetic ray geometry (style_sweep will reuse the same
            # cached EDPSamples for its own real build right after).
            if "edp" not in holder:
                holder["edp"] = real_build(c)
            return _fake_assemble(holder)(c)

        monkeypatch.setattr(style_sweep.observation_stream, "assemble", _lazy_assemble)

        results = style_sweep.run_style_sweep(cfg, ["raw", "density_10ex"])

        assert set(results) == {"raw", "density_10ex"}
        # EDPSamples should be built exactly once (shared), not once per style
        assert len(build_calls) == 1

        for style, result in results.items():
            assert len(result.batch_outcomes) == 2
            for outcome in result.batch_outcomes:
                # A wiring check, not a solver-convergence proof: y_obs
                # here is an arbitrary placeholder constant, not
                # necessarily achievable by 'density_10ex''s iterated
                # Gauss-Newton loop within its iteration cap -- just
                # confirm it ran and produced well-formed diagnostics.
                assert outcome.diagnostics.n_iterations >= 1
                assert np.isfinite(outcome.rmse_reduction.rmse_analysis)

    def test_hyper_params_by_style_forwarded(self, monkeypatch):
        cfg = _make_cfg(n_ensemble=20)
        holder = {}

        from Assimilation_Cycle import ensemble_init
        real_build = ensemble_init.load_or_build_edp_samples

        def _lazy_assemble(c):
            if "edp" not in holder:
                holder["edp"] = real_build(c)
            return _fake_assemble(holder)(c)

        monkeypatch.setattr(style_sweep.observation_stream, "assemble", _lazy_assemble)

        results = style_sweep.run_style_sweep(
            cfg, ["PCA_1D"], hyper_params_by_style={"PCA_1D": {"retaining_threshold": 0.9}},
        )
        assert "PCA_1D" in results
        assert len(results["PCA_1D"].batch_outcomes) == 2
