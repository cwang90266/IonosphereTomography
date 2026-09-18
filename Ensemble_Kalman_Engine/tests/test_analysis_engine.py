# -*- coding: utf-8 -*-
"""
Tests for the analysis engine (plan Section 6): the linear/iterated x
mean/nearest_analog design, exercised on small toy observation operators
so these checks stay independent of any ionosphere-specific code (plan
Section 5.7).

Two toy operators stand in for a "parameterization + H_grid" observation
operator (duck-typed: ``forward_ensemble``, ``forward_single``,
``linearized``):

- ``LinearToyOperator``  -- ``y = H @ x``, genuinely linear.
- ``CubicToyOperator``   -- ``y = H @ (x + 0.3 x^3)``, elementwise
  nonlinear with an analytic Jacobian, standing in for a
  ``density_10ex``-shaped style.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from Ensemble_Kalman_Engine.analysis_engine import AnalysisEngine, AnalysisConfig
from Ensemble_Kalman_Engine.ensemble_state import EnsembleState
from Ensemble_Kalman_Engine.osse import generate_osse_observation, recovery_error


@dataclass
class LinearToyOperator:
    H: np.ndarray

    def forward_ensemble(self, X_f: np.ndarray) -> np.ndarray:
        return self.H @ X_f

    def forward_single(self, x: np.ndarray) -> np.ndarray:
        return self.H @ x

    def linearized(self, x: np.ndarray) -> np.ndarray:
        return self.H


@dataclass
class CubicToyOperator:
    """y = H @ (x + 0.3 x^3): elementwise nonlinear, analytic Jacobian."""

    H: np.ndarray

    @staticmethod
    def _phi(x: np.ndarray) -> np.ndarray:
        return x + 0.3 * x ** 3

    @staticmethod
    def _dphi(x: np.ndarray) -> np.ndarray:
        return 1.0 + 0.9 * x ** 2

    def forward_ensemble(self, X_f: np.ndarray) -> np.ndarray:
        return self.H @ self._phi(X_f)

    def forward_single(self, x: np.ndarray) -> np.ndarray:
        return self.H @ self._phi(x)

    def linearized(self, x: np.ndarray) -> np.ndarray:
        return self.H * self._dphi(x)[np.newaxis, :]


def _make_ensemble(rng, n_state, n_members, mean, spread):
    X = mean[:, None] + spread * rng.standard_normal((n_state, n_members))
    return EnsembleState(X=X, param_shape=(n_state,))


def test_linear_mode_mean_centering_matches_kalman_core():
    """AnalysisEngine's 'linear'+'mean' path should agree with the
    directly-computed ensemble Kalman update (kalman_core, already
    checked against the exact closed-form identity in
    test_kalman_core.py) -- this only checks the engine wires it up
    correctly, not the math itself again."""
    from Ensemble_Kalman_Engine.kalman_core import ensemble_anomalies, kalman_gain

    rng = np.random.default_rng(0)
    n_state, n_obs, n_members = 6, 3, 500
    H = rng.standard_normal((n_obs, n_state))
    R = 0.05 * np.eye(n_obs)
    op = LinearToyOperator(H=H)

    ensemble = _make_ensemble(rng, n_state, n_members, mean=rng.standard_normal(n_state), spread=1.0)
    y_obs = rng.standard_normal(n_obs)

    config = AnalysisConfig(linearization="linear", centering="mean", rng=np.random.default_rng(1))
    result, diag = AnalysisEngine().analyze(ensemble, op, y_obs, R, config)

    X_f = ensemble.X
    Y_f = op.forward_ensemble(X_f)
    x_mean, S = ensemble_anomalies(X_f)
    _, Y_anom = ensemble_anomalies(Y_f)
    K, _ = kalman_gain(S, Y_anom, R)
    x_expected = x_mean + K @ (y_obs - Y_f.mean(axis=1))

    # Compare against the diagnostic's noise-free point estimate, not
    # result.ensemble_mean(): the perturbed-obs ensemble's own average is
    # expected to carry O(1/sqrt(n_members)) sampling noise from the
    # observation perturbations (see AnalysisDiagnostics.x_mean_a).
    np.testing.assert_allclose(diag.x_mean_a, x_expected, atol=1e-8, rtol=1e-6)
    assert diag.converged
    assert diag.n_iterations == 1


def test_iterated_recovers_truth_better_than_linear_for_strong_nonlinearity():
    """Plan Section 6.1's failure mode, demonstrated directly: with a wide
    ensemble and a genuinely nonlinear operator, the O(||x-xbar||^2)
    remainder the one-shot 'linear' gain ignores should make 'iterated'
    (Remedy A) noticeably better at recovering a known OSSE truth."""
    rng = np.random.default_rng(42)
    n_state, n_obs, n_members = 5, 4, 300
    H = rng.standard_normal((n_obs, n_state)) * 0.7
    R = 0.02 * np.eye(n_obs)

    op = CubicToyOperator(H=H)
    x_true = rng.standard_normal(n_state) * 2.5   # deliberately far from 0
    ensemble = _make_ensemble(rng, n_state, n_members, mean=np.zeros(n_state), spread=2.0)

    osse = generate_osse_observation(x_true, op, R, rng=np.random.default_rng(7))

    linear_cfg = AnalysisConfig(linearization="linear", centering="mean", rng=np.random.default_rng(1))
    iterated_cfg = AnalysisConfig(linearization="iterated", centering="mean", rng=np.random.default_rng(1))

    _, diag_linear = AnalysisEngine().analyze(ensemble, op, osse.y_obs, R, linear_cfg)
    _, diag_iterated = AnalysisEngine().analyze(ensemble, op, osse.y_obs, R, iterated_cfg)

    err_linear = recovery_error(x_true, diag_linear.x_mean_a)["rmse"]
    err_iterated = recovery_error(x_true, diag_iterated.x_mean_a)["rmse"]

    assert err_iterated < err_linear
    assert diag_iterated.n_iterations >= 1


def test_nearest_analog_centering_beats_mean_for_skewed_ensemble():
    """Construct an ensemble where the mean is a poor representative (most
    members far from truth, a small cluster near it) -- nearest_analog
    centering should out-perform mean centering at recovering the truth,
    even with the 'linear' linearization strategy (isolates the centering
    effect from Remedy A)."""
    rng = np.random.default_rng(3)
    n_state, n_obs = 4, 3
    H = rng.standard_normal((n_obs, n_state))
    R = 0.01 * np.eye(n_obs)
    op = LinearToyOperator(H=H)

    x_true = np.array([3.0, -2.0, 1.5, 0.5])

    n_far, n_near = 180, 20
    far_members = rng.standard_normal((n_state, n_far)) * 3.0 - 5.0
    near_members = x_true[:, None] + 0.05 * rng.standard_normal((n_state, n_near))
    X = np.concatenate([far_members, near_members], axis=1)
    ensemble = EnsembleState(X=X, param_shape=(n_state,))

    osse = generate_osse_observation(x_true, op, R, rng=np.random.default_rng(9))

    mean_cfg = AnalysisConfig(linearization="linear", centering="mean", rng=np.random.default_rng(1))
    analog_cfg = AnalysisConfig(linearization="linear", centering="nearest_analog", rng=np.random.default_rng(1))

    _, diag_mean = AnalysisEngine().analyze(ensemble, op, osse.y_obs, R, mean_cfg)
    _, diag_analog = AnalysisEngine().analyze(ensemble, op, osse.y_obs, R, analog_cfg)

    err_mean = recovery_error(x_true, diag_mean.x_mean_a)["rmse"]
    err_analog = recovery_error(x_true, diag_analog.x_mean_a)["rmse"]

    assert err_analog < err_mean
    assert diag_analog.center_source_history[0] == "ensemble_member"


def test_escape_hatch_mean_centering_never_touches_ensemble_members():
    rng = np.random.default_rng(5)
    n_state, n_obs, n_members = 5, 3, 200
    H = rng.standard_normal((n_obs, n_state))
    R = 0.05 * np.eye(n_obs)
    op = CubicToyOperator(H=H)

    ensemble = _make_ensemble(rng, n_state, n_members, mean=np.zeros(n_state), spread=1.5)
    y_obs = rng.standard_normal(n_obs)

    config = AnalysisConfig(linearization="iterated", centering="mean", rng=np.random.default_rng(1))
    _, diag = AnalysisEngine().analyze(ensemble, op, y_obs, R, config)

    assert "ensemble_member" not in diag.center_source_history


def test_output_ensemble_shape_and_covariance_reduced_by_observation():
    from Ensemble_Kalman_Engine.kalman_core import ensemble_anomalies

    rng = np.random.default_rng(6)
    n_state, n_obs, n_members = 6, 4, 400
    H = rng.standard_normal((n_obs, n_state))
    R = 0.05 * np.eye(n_obs)
    op = LinearToyOperator(H=H)

    ensemble = _make_ensemble(rng, n_state, n_members, mean=rng.standard_normal(n_state), spread=1.0)
    y_obs = rng.standard_normal(n_obs)

    config = AnalysisConfig(linearization="linear", centering="mean", rng=np.random.default_rng(1))
    result, _ = AnalysisEngine().analyze(ensemble, op, y_obs, R, config)

    assert result.X.shape == ensemble.X.shape

    _, S_f = ensemble_anomalies(ensemble.X)
    _, S_a = ensemble_anomalies(result.X)
    # Assimilating an informative observation should shrink total spread.
    assert np.trace(S_a @ S_a.T) < np.trace(S_f @ S_f.T)
