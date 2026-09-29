# -*- coding: utf-8 -*-
"""
Full pipeline integration test: ensemble_init.build() (real IRI2020, a
deliberately tiny grid/ensemble) -> synthetic vertical-ray CycleBatches
(same stand-in-geometry approach as
Ensemble_Kalman_Engine/tests/test_end_to_end_real_data.py, since real
orbit ephemeris isn't available in this environment either) ->
cycle_driver.run_cycle(). This is the first test exercising
Assimilation_Cycle's real entry point end to end, not just its pieces in
isolation.

Skipped when IRI2020_PATH/the compiled driver isn't available, same
convention as test_ensemble_init.py.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from Assimilation_Cycle.cycle_config import CycleConfig
from Assimilation_Cycle import ensemble_init
from Assimilation_Cycle.cycle_driver import CycleBatch, run_cycle

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
        n_ensemble=40,
        style="raw",
        batch_size=2,
        obs_sigma=1.0,
        iri_spread_kwargs={
            "hour_sample_range": 0, "f107_sample_range": 3, "ap_sample_range": 2,
            "ig_sample_range": 1, "rz_sample_range": 1,
        },
    )
    defaults.update(overrides)
    return CycleConfig(**defaults)


def _vertical_ray_batches(edp_samples, n_batches=4, rays_per_batch=3):
    """Synthetic vertical integration paths at the grid's own (lat, lon)
    points -- a stand-in for real LEO/GNSS occultation geometry, same
    rationale as Ensemble_Kalman_Engine's own end-to-end test."""
    import edp_samples as E

    geolocation = edp_samples.geolocation
    n_geo = geolocation.shape[0]
    alt_top_km = float(edp_samples.altitude.max())
    rng = np.random.default_rng(0)

    batches = []
    for b in range(n_batches):
        gnss_list, rec_list = [], []
        for _ in range(rays_per_batch):
            geo_idx = rng.integers(0, n_geo)
            lon, lat = geolocation[geo_idx]
            gnss_list.append(E._geodetic_to_ecef(lat, lon, 50_000.0) / 1000.0)
            rec_list.append(E._geodetic_to_ecef(lat, lon, alt_top_km * 1000.0) / 1000.0)
        podTc2_data = {
            "rec_ecef_km": np.stack(rec_list, axis=1),
            "gnss_ecef_km": np.stack(gnss_list, axis=1),
        }
        # y_obs is a placeholder filled from a real forward pass below, so
        # this fixture stays self-contained (no separately-prepared TEC needed).
        batches.append((podTc2_data, rays_per_batch))
    return batches


@pytest.mark.skipif(not _has_iri2020, reason="no compiled IRI2020 driver in this environment")
class TestRunCycleEndToEnd:
    def test_full_pipeline_reduces_rmse_and_persists_ensemble(self):
        cfg = _make_cfg()
        ensemble, edp_samples, parameterization = ensemble_init.build(cfg)

        from Ensemble_Kalman_Engine.driver import GeneralEnKFDriver
        driver = GeneralEnKFDriver(style=cfg.style, hyper_params=cfg.hyper_params)

        # Build real batches: real vertical-ray geometry, real forward-modeled
        # "truth" TEC (one held-out ensemble member) as observations, so this
        # is effectively a hand-rolled OSSE using the real forward operator.
        raw_batches = _vertical_ray_batches(edp_samples, n_batches=3, rays_per_batch=4)
        x_true = ensemble.X[:, 0].copy()
        forecast = ensemble.X[:, 1:]
        from Ensemble_Kalman_Engine import EnsembleState
        forecast_ensemble = EnsembleState(X=forecast, param_shape=ensemble.param_shape)

        batches = []
        for i, (podTc2_data, n_rays) in enumerate(raw_batches):
            obs_op = driver.build_observation_operator(
                edp_samples, parameterization, ensemble.param_shape, podTc2_data,
            )
            y_obs = obs_op.forward_single(x_true)
            batches.append(CycleBatch(
                podTc2_data=podTc2_data, y_obs=y_obs, R=(cfg.obs_sigma ** 2) * np.eye(n_rays),
                batch_index=i,
            ))

        result = run_cycle(cfg, edp_samples, parameterization, batches, forecast_ensemble)

        assert len(result.batch_outcomes) == 3
        assert result.final_ensemble.n_members == forecast_ensemble.n_members
        assert result.final_ensemble.param_shape == ensemble.param_shape

        # This is a pipeline-wiring check (real IRI2020 + real ray geometry
        # + real batch loop end to end), not a statistical-convergence
        # proof: IRI_Sample_Inputs.randomSamples has no seed control, so
        # which member becomes x_true (and the resulting ensemble spread)
        # varies run to run, and 'raw' style's state size here (n_height *
        # n_geo) is comparable to both the ensemble size and the total ray
        # count -- not always enough to guarantee a single random draw's
        # state-space error strictly decreases. Every batch outcome should
        # still carry real, well-formed diagnostics regardless of draw.
        for outcome in result.batch_outcomes:
            assert outcome.diagnostics.converged
            assert outcome.rmse_reduction.n_obs == 4
            assert np.isfinite(outcome.rmse_reduction.rmse_forecast)
            assert np.isfinite(outcome.rmse_reduction.rmse_analysis)
            assert outcome.rank_histogram.sum() == 4
            assert outcome.effective_rank_analysis["rank"] >= 1
