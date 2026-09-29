# -*- coding: utf-8 -*-
"""
Tests for cycle_driver.py's run_batch_loop (plan Section 4.6/4.10/4.11),
using toy observation operators so these stay independent of
EDPSamples/real ray geometry -- same pattern as
Ensemble_Kalman_Engine/tests/test_analysis_engine.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from Ensemble_Kalman_Engine import EnsembleState
from Ensemble_Kalman_Engine.driver import GeneralEnKFDriver
from Ensemble_Kalman_Engine.analysis_engine import AnalysisConfig

from Assimilation_Cycle.cycle_config import CycleConfig
from Assimilation_Cycle.cycle_driver import CycleBatch, run_batch_loop
from Assimilation_Cycle.ensemble_io import load_ensemble_netcdf


@dataclass
class LinearToyOperator:
    """y = H @ x -- same toy operator shape as
    Ensemble_Kalman_Engine/tests/test_analysis_engine.py."""
    H: np.ndarray

    def forward_ensemble(self, X_f: np.ndarray) -> np.ndarray:
        return self.H @ X_f

    def forward_single(self, x: np.ndarray) -> np.ndarray:
        return self.H @ x

    def linearized(self, x: np.ndarray) -> np.ndarray:
        return self.H

    def decode(self, param_vec: np.ndarray) -> np.ndarray:
        return param_vec


def _make_ensemble(rng, n_state, n_members, mean, spread=1.0):
    X = mean[:, None] + spread * rng.standard_normal((n_state, n_members))
    return EnsembleState(X=X, param_shape=(n_state,))


def _make_cfg(**overrides) -> CycleConfig:
    defaults = dict(
        start_time="2025-11-18T10:00:00", end_time="2025-11-18T11:00:00",
        center_lat=69.6, center_lon=18.9, radius_km=500.0,
        style="raw",
    )
    defaults.update(overrides)
    return CycleConfig(**defaults)


def _linear_driver(config=None) -> GeneralEnKFDriver:
    config = config or AnalysisConfig(linearization="linear", centering="mean")
    return GeneralEnKFDriver(style="raw", config=config)


class TestBasicLoop:
    def test_single_batch_matches_direct_analyze_call(self):
        rng = np.random.default_rng(0)
        n_state, n_obs, n_members = 6, 3, 300
        H = rng.standard_normal((n_obs, n_state))
        R = 0.05 * np.eye(n_obs)
        op = LinearToyOperator(H=H)
        x_true = rng.standard_normal(n_state)
        ensemble = _make_ensemble(rng, n_state, n_members, mean=x_true + 0.5)
        y_obs = H @ x_true

        driver = _linear_driver()
        batch = CycleBatch(podTc2_data={}, y_obs=y_obs, R=R, batch_index=0)
        result = run_batch_loop(ensemble, driver, [op], [batch])

        assert len(result.batch_outcomes) == 1
        outcome = result.batch_outcomes[0]
        assert outcome.diagnostics.converged
        # analysis mean should be closer to truth than the forecast mean was
        forecast_err = np.linalg.norm(ensemble.ensemble_mean() - x_true)
        analysis_err = np.linalg.norm(result.final_ensemble.ensemble_mean() - x_true)
        assert analysis_err < forecast_err
        assert outcome.rmse_reduction.reduction > 0

    def test_multi_batch_persistence_carries_ensemble_forward(self):
        """Across several batches of the same truth, RMSE (obs vs. analysis
        mean) should trend down as more observations accumulate -- checks
        that the posterior really is carried forward as the next prior."""
        rng = np.random.default_rng(1)
        n_state, n_obs, n_members = 10, 4, 400
        x_true = rng.standard_normal(n_state)
        ensemble = _make_ensemble(rng, n_state, n_members, mean=x_true + 2.0, spread=2.0)

        driver = _linear_driver()
        batches, operators = [], []
        for i in range(5):
            H = rng.standard_normal((n_obs, n_state))
            R = 0.02 * np.eye(n_obs)
            operators.append(LinearToyOperator(H=H))
            batches.append(CycleBatch(podTc2_data={}, y_obs=H @ x_true, R=R, batch_index=i))

        result = run_batch_loop(ensemble, driver, operators, batches)
        assert len(result.batch_outcomes) == 5

        final_err = np.linalg.norm(result.final_ensemble.ensemble_mean() - x_true)
        initial_err = np.linalg.norm(ensemble.ensemble_mean() - x_true)
        assert final_err < initial_err

    def test_batch_order_mismatch_raises(self):
        rng = np.random.default_rng(0)
        ensemble = _make_ensemble(rng, 4, 20, mean=np.zeros(4))
        driver = _linear_driver()
        op = LinearToyOperator(H=rng.standard_normal((2, 4)))
        batch = CycleBatch(podTc2_data={}, y_obs=np.zeros(2), R=np.eye(2))
        with pytest.raises(ValueError):
            run_batch_loop(ensemble, driver, [op, op], [batch])

    def test_batch_outcome_carries_forecast_and_analysis_predicted_tec(self):
        rng = np.random.default_rng(2)
        n_state, n_obs, n_members = 6, 3, 200
        H = rng.standard_normal((n_obs, n_state))
        op = LinearToyOperator(H=H)
        x_true = rng.standard_normal(n_state)
        ensemble = _make_ensemble(rng, n_state, n_members, mean=x_true + 0.5)
        batch = CycleBatch(podTc2_data={}, y_obs=H @ x_true, R=0.05 * np.eye(n_obs), batch_index=0)

        driver = _linear_driver()
        result = run_batch_loop(ensemble, driver, [op], [batch])
        outcome = result.batch_outcomes[0]

        assert outcome.y_forecast.shape == (n_obs,)
        assert outcome.y_analysis.shape == (n_obs,)
        np.testing.assert_allclose(outcome.y_forecast, H @ ensemble.ensemble_mean())
        # analysis predicted TEC should be closer to the real observation than forecast was
        assert (np.linalg.norm(outcome.y_analysis - batch.y_obs)
                < np.linalg.norm(outcome.y_forecast - batch.y_obs))


class TestOnBatchCallback:
    def test_invoked_once_per_batch_with_forecast_and_analysis_ensembles(self):
        rng = np.random.default_rng(3)
        n_state, n_obs, n_members = 5, 2, 50
        x_true = rng.standard_normal(n_state)
        ensemble = _make_ensemble(rng, n_state, n_members, mean=x_true)
        driver = _linear_driver()

        batches, operators = [], []
        for i in range(3):
            H = rng.standard_normal((n_obs, n_state))
            operators.append(LinearToyOperator(H=H))
            batches.append(CycleBatch(podTc2_data={}, y_obs=H @ x_true, R=0.1 * np.eye(n_obs), batch_index=i))

        calls = []

        def _on_batch(batch, obs_operator, ensemble_forecast, ensemble_analysis, outcome):
            calls.append((batch.batch_index, ensemble_forecast.n_members, ensemble_analysis.n_members,
                          outcome.batch_index))

        result = run_batch_loop(ensemble, driver, operators, batches, on_batch=_on_batch)

        assert len(calls) == 3
        assert [c[0] for c in calls] == [0, 1, 2]
        assert all(c[1] == n_members and c[2] == n_members for c in calls)
        assert [c[3] for c in calls] == [0, 1, 2]

    def test_default_none_does_not_require_a_callback(self):
        rng = np.random.default_rng(4)
        ensemble = _make_ensemble(rng, 4, 20, mean=np.zeros(4))
        driver = _linear_driver()
        op = LinearToyOperator(H=rng.standard_normal((2, 4)))
        batch = CycleBatch(podTc2_data={}, y_obs=np.zeros(2), R=np.eye(2))
        result = run_batch_loop(ensemble, driver, [op], [batch])  # no on_batch -- must not raise
        assert len(result.batch_outcomes) == 1


class TestIntermediatePersistence:
    def test_saves_one_file_per_batch(self, tmp_path):
        rng = np.random.default_rng(2)
        n_state, n_obs, n_members = 5, 2, 100
        x_true = rng.standard_normal(n_state)
        ensemble = _make_ensemble(rng, n_state, n_members, mean=x_true)
        driver = _linear_driver()

        batches, operators = [], []
        for i in range(3):
            H = rng.standard_normal((n_obs, n_state))
            operators.append(LinearToyOperator(H=H))
            batches.append(CycleBatch(podTc2_data={}, y_obs=H @ x_true, R=0.1 * np.eye(n_obs), batch_index=i))

        cfg = _make_cfg(save_intermediate_ensembles=True, ensemble_output_dir=tmp_path)
        result = run_batch_loop(ensemble, driver, operators, batches, cfg=cfg)

        for i in range(3):
            path = tmp_path / f"ensemble_batch_{i:04d}.nc"
            assert path.exists()
            loaded, meta = load_ensemble_netcdf(path)
            assert loaded.n_members == n_members
            assert meta["batch_index"] == i
            assert meta["style"] == "raw"

        # results still line up with the in-memory outcomes
        assert len(result.batch_outcomes) == 3

    def test_no_files_written_when_flag_off(self, tmp_path):
        rng = np.random.default_rng(3)
        ensemble = _make_ensemble(rng, 4, 50, mean=np.zeros(4))
        driver = _linear_driver()
        op = LinearToyOperator(H=rng.standard_normal((2, 4)))
        batch = CycleBatch(podTc2_data={}, y_obs=np.zeros(2), R=np.eye(2))
        cfg = _make_cfg(save_intermediate_ensembles=False, ensemble_output_dir=tmp_path)

        run_batch_loop(ensemble, driver, [op], [batch], cfg=cfg)
        assert list(tmp_path.iterdir()) == []


class TestInterBatchInflation:
    def test_none_leaves_analysis_spread_unchanged(self):
        rng = np.random.default_rng(10)
        n_state, n_obs, n_members = 6, 3, 300
        H = rng.standard_normal((n_obs, n_state))
        x_true = rng.standard_normal(n_state)
        ensemble = _make_ensemble(rng, n_state, n_members, mean=x_true + 0.5, spread=1.0)
        batch = CycleBatch(podTc2_data={}, y_obs=H @ x_true, R=0.05 * np.eye(n_obs), batch_index=0)
        op = LinearToyOperator(H=H)

        # Fixed driver.config.rng, shared across both calls -- the
        # stochastic perturbed-obs EnKF update draws its own noise from
        # *this* rng (not run_batch_loop's own `rng` parameter, which only
        # covers OSSE noise/rank-histogram tie-breaking), so an unseeded
        # comparison would conflate that sampling randomness with the
        # effect being tested here.
        driver1 = _linear_driver(AnalysisConfig(linearization="linear", centering="mean", rng=np.random.default_rng(99)))
        cfg = _make_cfg(inter_batch_inflation_factor=None)
        result = run_batch_loop(ensemble, driver1, [op], [batch], cfg=cfg)
        std_no_inflation = result.final_ensemble.X.std(axis=1)

        # Same setup, factor=1.0 (a no-op inflation) should match exactly
        driver2 = _linear_driver(AnalysisConfig(linearization="linear", centering="mean", rng=np.random.default_rng(99)))
        cfg_noop = _make_cfg(inter_batch_inflation_factor=1.0)
        result_noop = run_batch_loop(ensemble, driver2, [op], [batch], cfg=cfg_noop)
        std_noop = result_noop.final_ensemble.X.std(axis=1)

        np.testing.assert_allclose(std_no_inflation, std_noop, rtol=1e-8)

    def test_factor_greater_than_one_inflates_analysis_spread(self):
        rng = np.random.default_rng(11)
        n_state, n_obs, n_members = 6, 3, 300
        H = rng.standard_normal((n_obs, n_state))
        x_true = rng.standard_normal(n_state)
        ensemble = _make_ensemble(rng, n_state, n_members, mean=x_true + 0.5, spread=1.0)
        batch = CycleBatch(podTc2_data={}, y_obs=H @ x_true, R=0.05 * np.eye(n_obs), batch_index=0)
        op = LinearToyOperator(H=H)

        driver1 = _linear_driver(AnalysisConfig(linearization="linear", centering="mean", rng=np.random.default_rng(7)))
        cfg_base = _make_cfg(inter_batch_inflation_factor=None)
        result_base = run_batch_loop(ensemble, driver1, [op], [batch], cfg=cfg_base)
        std_base = result_base.final_ensemble.X.std(axis=1)

        factor = 1.5
        driver2 = _linear_driver(AnalysisConfig(linearization="linear", centering="mean", rng=np.random.default_rng(7)))
        cfg_inflated = _make_cfg(inter_batch_inflation_factor=factor)
        result_inflated = run_batch_loop(ensemble, driver2, [op], [batch], cfg=cfg_inflated)
        std_inflated = result_inflated.final_ensemble.X.std(axis=1)

        np.testing.assert_allclose(std_inflated, factor * std_base, rtol=1e-6)
        # means should be unaffected -- inflation only scales spread around the mean
        np.testing.assert_allclose(
            result_inflated.final_ensemble.ensemble_mean(),
            result_base.final_ensemble.ensemble_mean(), rtol=1e-6,
        )

    def test_inflation_compounds_across_batches(self):
        rng = np.random.default_rng(12)
        n_state, n_obs, n_members = 5, 2, 200
        x_true = rng.standard_normal(n_state)
        ensemble = _make_ensemble(rng, n_state, n_members, mean=x_true, spread=1.0)
        driver = _linear_driver()

        batches, operators = [], []
        for i in range(3):
            H = rng.standard_normal((n_obs, n_state))
            operators.append(LinearToyOperator(H=H))
            batches.append(CycleBatch(podTc2_data={}, y_obs=H @ x_true, R=0.1 * np.eye(n_obs), batch_index=i))

        cfg = _make_cfg(inter_batch_inflation_factor=1.1)
        result = run_batch_loop(ensemble, driver, operators, batches, cfg=cfg)
        # with repeated re-inflation every batch, spread should not have
        # collapsed to near-zero the way unbounded persistence tends to
        # after several informative batches
        assert result.final_ensemble.X.std(axis=1).mean() > 1e-3


class TestOsseMode:
    def test_holds_out_one_member_and_recovers_truth(self):
        rng = np.random.default_rng(4)
        n_state, n_obs, n_members = 8, 4, 500
        # Ensemble members are all draws around a common mean; x_true will
        # be one of these draws (held out), so it's a plausible member of
        # the remaining forecast's distribution.
        common_mean = rng.standard_normal(n_state)
        ensemble = _make_ensemble(rng, n_state, n_members, mean=common_mean, spread=1.5)

        driver = _linear_driver()
        batches, operators = [], []
        for i in range(4):
            H = rng.standard_normal((n_obs, n_state))
            operators.append(LinearToyOperator(H=H))
            # y_obs here is a placeholder -- OSSE mode overwrites it.
            batches.append(CycleBatch(podTc2_data={}, y_obs=np.zeros(n_obs), R=0.01 * np.eye(n_obs), batch_index=i))

        cfg = _make_cfg(osse_mode=True, osse_rng_seed=123)
        result = run_batch_loop(ensemble, driver, operators, batches, cfg=cfg)

        assert result.osse_x_true is not None
        np.testing.assert_allclose(result.osse_x_true, ensemble.X[:, 0])

        # the forecast ensemble that actually got assimilated should have
        # n_members - 1 columns (one held out as truth)
        assert result.final_ensemble.n_members == n_members - 1

        # after several batches of OSSE observations, analysis mean should
        # be close to the held-out truth
        final_err = np.linalg.norm(result.final_ensemble.ensemble_mean() - result.osse_x_true)
        assert final_err < 0.5

    def test_osse_needs_at_least_two_members(self):
        rng = np.random.default_rng(5)
        ensemble = EnsembleState(X=rng.standard_normal((4, 1)), param_shape=(4,))
        driver = _linear_driver()
        op = LinearToyOperator(H=rng.standard_normal((2, 4)))
        batch = CycleBatch(podTc2_data={}, y_obs=np.zeros(2), R=np.eye(2))
        cfg = _make_cfg(osse_mode=True)
        with pytest.raises(ValueError):
            run_batch_loop(ensemble, driver, [op], [batch], cfg=cfg)

    def test_osse_reproducible_with_fixed_seed(self):
        rng = np.random.default_rng(6)
        n_state, n_obs, n_members = 5, 3, 50
        mean = rng.standard_normal(n_state)
        H = rng.standard_normal((n_obs, n_state))
        R = 0.05 * np.eye(n_obs)
        driver = _linear_driver()

        def _run():
            ens = _make_ensemble(np.random.default_rng(6), n_state, n_members, mean=mean)
            op = LinearToyOperator(H=H)
            batch = CycleBatch(podTc2_data={}, y_obs=np.zeros(n_obs), R=R)
            cfg = _make_cfg(osse_mode=True, osse_rng_seed=99)
            return run_batch_loop(ens, driver, [op], [batch], cfg=cfg)

        result_a = _run()
        result_b = _run()
        np.testing.assert_allclose(
            result_a.batch_outcomes[0].y_obs_used, result_b.batch_outcomes[0].y_obs_used,
        )
