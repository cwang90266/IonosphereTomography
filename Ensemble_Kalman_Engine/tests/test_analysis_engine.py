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


def test_nearest_analog_skips_members_with_nonfinite_forecast():
    """Regression: np.argmin returns the index of a NaN, so a diverged
    member (NaN predicted obs) used to be chosen as the analog."""
    from Ensemble_Kalman_Engine.analysis_engine import _best_ensemble_analog

    y_obs = np.zeros(3)
    Y_f = np.array([[5.0, np.nan, 0.1, np.inf],
                    [5.0, 0.0, 0.1, 0.0],
                    [5.0, 0.0, 0.1, 0.0]])
    i_star, dist = _best_ensemble_analog(y_obs, Y_f)
    assert i_star == 2
    assert np.isfinite(dist)


@dataclass
class OverflowAfterFirstLinearization(CubicToyOperator):
    """Cubic toy whose re-linearization overflows after the first call,
    standing in for a parameterization singularity hit mid-iteration."""

    n_calls: int = 0

    def linearized(self, x: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        W = super().linearized(x)
        return W if self.n_calls == 1 else W * 1e200


def test_iterated_stops_on_overflowing_linearization_instead_of_raising():
    rng = np.random.default_rng(7)
    n_state, n_obs, n_members = 5, 4, 60
    op = OverflowAfterFirstLinearization(H=rng.standard_normal((n_obs, n_state)))
    R = 0.05 * np.eye(n_obs)
    ensemble = _make_ensemble(rng, n_state, n_members, mean=np.zeros(n_state), spread=1.0)
    y_obs = rng.standard_normal(n_obs)

    config = AnalysisConfig(linearization="iterated", centering="mean",
                            convergence_tol=0.0, residual_rel_tol=-1.0,
                            rng=np.random.default_rng(1))
    result, diag = AnalysisEngine().analyze(ensemble, op, y_obs, R, config)

    assert diag.stopped_nonfinite_linearization
    assert np.all(np.isfinite(result.X))
    assert np.all(np.isfinite(diag.K_mean))


class TestOnIterationHook:
    """``Impact_Vector_Plan.md`` Section 6/11 step 2: a new, optional
    per-iteration hook on ``analyze()`` -- the third deliberate exception
    to this project's "no logic added to the survivor modules" rule, so
    it needs its own explicit zero-regression proof (Section 9 test 4)
    plus confirmation it reports exactly what the plan's decomposition
    needs (Section 9 tests 2-3, folded in here since they're really
    about what this hook reports, not about ``impact_vectors.py`` yet)."""

    def test_fires_once_for_linear_branch_with_correct_values(self):
        rng = np.random.default_rng(10)
        n_state, n_obs, n_members = 5, 3, 100
        H = rng.standard_normal((n_obs, n_state))
        R = 0.05 * np.eye(n_obs)
        op = LinearToyOperator(H=H)
        ensemble = _make_ensemble(rng, n_state, n_members, mean=rng.standard_normal(n_state), spread=1.0)
        y_obs = rng.standard_normal(n_obs)
        config = AnalysisConfig(linearization="linear", centering="mean", rng=np.random.default_rng(1))

        calls = []
        result, diag = AnalysisEngine().analyze(
            ensemble, op, y_obs, R, config,
            on_iteration=lambda k, x_before, K, r: calls.append((k, x_before.copy(), K.copy(), r.copy())),
        )

        assert len(calls) == 1
        k, x_before, K, r = calls[0]
        assert k == 0
        np.testing.assert_allclose(x_before, ensemble.X.mean(axis=1), atol=1e-10)
        # K @ r is exactly this (only) step's contribution: x_before + K@r
        # must reproduce the engine's own reported analysis mean.
        np.testing.assert_allclose(x_before + K @ r, diag.x_mean_a, atol=1e-8)

    def test_fires_once_per_accepted_iteration_for_iterated_branch(self):
        rng = np.random.default_rng(11)
        n_state, n_obs, n_members = 5, 4, 200
        H = rng.standard_normal((n_obs, n_state)) * 0.5
        R = 0.02 * np.eye(n_obs)
        op = CubicToyOperator(H=H)
        ensemble = _make_ensemble(rng, n_state, n_members, mean=np.zeros(n_state), spread=1.5)
        y_obs = rng.standard_normal(n_obs)
        config = AnalysisConfig(linearization="iterated", centering="mean", rng=np.random.default_rng(1))

        calls = []
        _, diag = AnalysisEngine().analyze(
            ensemble, op, y_obs, R, config,
            on_iteration=lambda k, x_before, K, r: calls.append(k),
        )

        assert len(calls) == diag.n_iterations
        assert calls == list(range(diag.n_iterations))

    def test_exactness_identity_reconstructs_final_state_via_decay_weights(self):
        """``Impact_Vector_Plan.md`` Finding 3: the final accepted state's
        deviation from the anchor is a *decay-weighted* sum of every
        iteration's own ``K@r`` proposal (``w_k = alpha_k * prod_{m>k}
        (1-alpha_m)``), not a flat sum -- confirmed directly against the
        engine's own accepted trajectory (``centering="mean"``, so there
        are no ``nearest_analog`` resets to break the identity)."""
        rng = np.random.default_rng(12)
        n_state, n_obs, n_members = 5, 4, 200
        H = rng.standard_normal((n_obs, n_state)) * 0.5
        R = 0.02 * np.eye(n_obs)
        op = CubicToyOperator(H=H)
        x_true = rng.standard_normal(n_state) * 2.0
        ensemble = _make_ensemble(rng, n_state, n_members, mean=np.zeros(n_state), spread=2.0)
        osse = generate_osse_observation(x_true, op, R, rng=np.random.default_rng(13))
        config = AnalysisConfig(linearization="iterated", centering="mean", rng=np.random.default_rng(1))

        calls = []
        _, diag = AnalysisEngine().analyze(
            ensemble, op, osse.y_obs, R, config,
            on_iteration=lambda k, x_before, K, r: calls.append((x_before.copy(), K.copy(), r.copy())),
        )

        x_p = calls[0][0]
        alphas = diag.alpha_history
        J = len(calls)
        assert len(alphas) == J

        D = np.zeros(n_state)
        for k in range(J):
            decay = 1.0
            for m in range(k + 1, J):
                decay *= (1.0 - alphas[m])
            w_k = alphas[k] * decay
            _, K, r = calls[k]
            D += w_k * (K @ r)

        np.testing.assert_allclose(x_p + D, diag.x_mean_a, atol=1e-6, rtol=1e-6)

    def test_damped_iteration_contribution_matches_weighted_formula_not_flat_sum(self):
        """Specifically exercises a case with real backtracking damping
        (alpha<1 on at least one iteration) -- Section 9 test 3, the part
        a flat (unweighted) sum across iterations would get wrong."""
        rng = np.random.default_rng(15)
        n_state, n_obs, n_members = 4, 3, 150
        H = rng.standard_normal((n_obs, n_state)) * 1.5
        R = 0.01 * np.eye(n_obs)
        op = CubicToyOperator(H=H)
        x_true = rng.standard_normal(n_state) * 4.0   # far enough to force damping
        ensemble = _make_ensemble(rng, n_state, n_members, mean=np.zeros(n_state), spread=3.0)
        osse = generate_osse_observation(x_true, op, R, rng=np.random.default_rng(16))
        config = AnalysisConfig(linearization="iterated", centering="mean", rng=np.random.default_rng(1))

        calls = []
        _, diag = AnalysisEngine().analyze(
            ensemble, op, osse.y_obs, R, config,
            on_iteration=lambda k, x_before, K, r: calls.append((x_before.copy(), K.copy(), r.copy())),
        )
        assert any(a < 1.0 for a in diag.alpha_history), (
            "test setup didn't actually force damping -- adjust x_true/spread"
        )

        x_p = calls[0][0]
        alphas = diag.alpha_history
        J = len(calls)

        D_weighted = np.zeros(n_state)
        D_flat = np.zeros(n_state)
        for k in range(J):
            decay = 1.0
            for m in range(k + 1, J):
                decay *= (1.0 - alphas[m])
            w_k = alphas[k] * decay
            _, K, r = calls[k]
            D_weighted += w_k * (K @ r)
            D_flat += K @ r

        np.testing.assert_allclose(x_p + D_weighted, diag.x_mean_a, atol=1e-6, rtol=1e-6)
        assert not np.allclose(x_p + D_flat, diag.x_mean_a, atol=1e-3), (
            "flat sum should NOT match once real damping occurred -- "
            "if it does, the test setup isn't exercising damping meaningfully"
        )

    def test_omitting_hook_changes_nothing(self):
        """Zero-regression (Section 9 test 4): the new ``on_iteration``
        parameter, defaulted to ``None``, must not alter ``analyze()``'s
        existing behavior at all -- passing it explicitly as ``None``
        must reproduce byte-identical results to omitting it."""
        rng = np.random.default_rng(14)
        n_state, n_obs, n_members = 5, 4, 150
        H = rng.standard_normal((n_obs, n_state))
        R = 0.03 * np.eye(n_obs)
        op = CubicToyOperator(H=H)
        ensemble = _make_ensemble(rng, n_state, n_members, mean=np.zeros(n_state), spread=1.5)
        y_obs = rng.standard_normal(n_obs)

        config_a = AnalysisConfig(linearization="iterated", centering="nearest_analog", rng=np.random.default_rng(1))
        config_b = AnalysisConfig(linearization="iterated", centering="nearest_analog", rng=np.random.default_rng(1))

        result_a, diag_a = AnalysisEngine().analyze(ensemble, op, y_obs, R, config_a)
        result_b, diag_b = AnalysisEngine().analyze(ensemble, op, y_obs, R, config_b, on_iteration=None)

        np.testing.assert_array_equal(result_a.X, result_b.X)
        assert diag_a.n_iterations == diag_b.n_iterations
        assert diag_a.alpha_history == diag_b.alpha_history
        assert diag_a.center_source_history == diag_b.center_source_history


def test_iterated_raises_clearly_when_initial_linearization_overflows():
    rng = np.random.default_rng(8)
    n_state, n_obs, n_members = 5, 4, 60
    op = OverflowAfterFirstLinearization(H=rng.standard_normal((n_obs, n_state)), n_calls=1)
    ensemble = _make_ensemble(rng, n_state, n_members, mean=np.zeros(n_state), spread=1.0)

    with pytest.raises(FloatingPointError):
        AnalysisEngine().analyze(ensemble, op, rng.standard_normal(n_obs), 0.05 * np.eye(n_obs),
                                 AnalysisConfig(linearization="iterated", centering="mean"))
