# -*- coding: utf-8 -*-
"""
Closed-form checks for kalman_core.py (plan Section 5.7's first bullet):
verify the ensemble-based machinery reproduces the exact linear-Gaussian
Kalman identities on a small toy problem, independent of any
ionosphere-specific code.
"""
from __future__ import annotations

import numpy as np
import pytest

from Ensemble_Kalman_Engine.kalman_core import (
    ensemble_anomalies,
    kalman_gain,
    apply_inflation,
    stochastic_update,
    ensrf_update,
)


@pytest.fixture
def toy_problem():
    rng = np.random.default_rng(0)
    n_state, n_obs = 6, 3
    H = rng.standard_normal((n_obs, n_state))
    R = 0.1 * np.eye(n_obs)
    Q_true = rng.standard_normal((n_state, n_state))
    Q_true = Q_true @ Q_true.T + np.eye(n_state)   # SPD
    return rng, n_state, n_obs, H, R, Q_true


def _exact_gain(Q, H, R):
    return Q @ H.T @ np.linalg.inv(H @ Q @ H.T + R)


def _exact_Qa(Q, H, R):
    K = _exact_gain(Q, H, R)
    return Q - K @ H @ Q


def test_ensemble_anomalies_reproduce_sample_covariance(toy_problem):
    rng, n_state, n_obs, H, R, Q_true = toy_problem
    n_members = 5000
    X = rng.multivariate_normal(np.zeros(n_state), Q_true, size=n_members).T
    x_mean, S = ensemble_anomalies(X)
    Q_sample = S @ S.T
    np.testing.assert_allclose(x_mean, X.mean(axis=1))
    # Large-n sample covariance should be close to the true covariance.
    assert np.max(np.abs(Q_sample - Q_true)) < 0.15 * np.max(np.abs(Q_true))


def test_kalman_gain_matches_exact_formula(toy_problem):
    rng, n_state, n_obs, H, R, Q_true = toy_problem
    n_members = 20000
    X = rng.multivariate_normal(np.zeros(n_state), Q_true, size=n_members).T
    _, S = ensemble_anomalies(X)
    Y = H @ S
    K_ens, _ = kalman_gain(S, Y, R)
    K_exact = _exact_gain(Q_true, H, R)
    np.testing.assert_allclose(K_ens, K_exact, atol=0.05, rtol=0.1)


def test_stochastic_update_analysis_covariance_matches_exact_Qa(toy_problem):
    """
    Plan Section 1's closed-form identity:
        Q_a = Q - Q H^T (H Q H^T + R)^-1 H Q
            = S S^T - S Y^T (Y Y^T + R)^-1 Y S^T,   Y = H S
    Check the *sample* covariance of the stochastic-update output ensemble
    converges to the exact Q_a (in expectation -- perturbed-obs EnKF is
    unbiased in covariance, not exact member-by-member).
    """
    rng, n_state, n_obs, H, R, Q_true = toy_problem
    n_members = 20000
    x_p_mean = rng.standard_normal(n_state)
    X_f = x_p_mean[:, None] + rng.multivariate_normal(
        np.zeros(n_state), Q_true, size=n_members
    ).T
    Y_f = H @ X_f
    y_obs = H @ x_p_mean + rng.multivariate_normal(np.zeros(n_obs), R)

    result = stochastic_update(X_f, Y_f, y_obs, R, rng=np.random.default_rng(1))
    _, S_a = ensemble_anomalies(result.X_a)
    Q_a_sample = S_a @ S_a.T
    Q_a_exact = _exact_Qa(Q_true, H, R)

    assert np.max(np.abs(Q_a_sample - Q_a_exact)) < 0.2 * np.max(np.abs(Q_a_exact))


def test_ensrf_update_matches_exact_Qa_precisely(toy_problem):
    """The deterministic EnSRF branch should match Q_a far more tightly
    than the stochastic branch, since it has no random perturbation
    sampling error -- only ensemble-covariance (finite-n) error."""
    rng, n_state, n_obs, H, R, Q_true = toy_problem
    n_members = 2000
    x_p_mean = rng.standard_normal(n_state)
    X_f = x_p_mean[:, None] + rng.multivariate_normal(
        np.zeros(n_state), Q_true, size=n_members
    ).T
    Y_f = H @ X_f
    y_obs = H @ x_p_mean + rng.multivariate_normal(np.zeros(n_obs), R)

    result = ensrf_update(X_f, Y_f, y_obs, R)
    _, S_a = ensemble_anomalies(result.X_a)
    Q_a_sample = S_a @ S_a.T
    Q_a_exact = _exact_Qa(Q_true, H, R)

    assert np.max(np.abs(Q_a_sample - Q_a_exact)) < 0.05 * np.max(np.abs(Q_a_exact))

    # Mean update should match the standard Kalman formula *for the same
    # ensemble-estimated gain* exactly (isolates "is the update formula
    # implemented correctly" from "how well does a 2000-member sample
    # covariance approximate Q_true", which is a separate, already-tested
    # property -- comparing against the exact-Q_true gain would just
    # re-measure finite-sample noise here).
    x_mean_f = X_f.mean(axis=1)
    _, S_f = ensemble_anomalies(X_f)
    _, Y_f_anom = ensemble_anomalies(Y_f)
    K_ens, _ = kalman_gain(S_f, Y_f_anom, R)
    x_a_expected = x_mean_f + K_ens @ (y_obs - H @ x_mean_f)
    np.testing.assert_allclose(result.x_mean_a, x_a_expected, atol=1e-8, rtol=1e-6)


def test_qa_does_not_depend_on_observed_value(toy_problem):
    """Plan Section 1: Q_a is independent of the realized observation --
    only the mean update should move when y_obs changes."""
    rng, n_state, n_obs, H, R, Q_true = toy_problem
    n_members = 2000
    X_f = rng.multivariate_normal(np.zeros(n_state), Q_true, size=n_members).T
    Y_f = H @ X_f

    result_a = ensrf_update(X_f, Y_f, np.zeros(n_obs), R)
    result_b = ensrf_update(X_f, Y_f, rng.standard_normal(n_obs) * 5, R)

    _, S_a = ensemble_anomalies(result_a.X_a)
    _, S_b = ensemble_anomalies(result_b.X_a)
    np.testing.assert_allclose(S_a @ S_a.T, S_b @ S_b.T, atol=1e-8)
    assert not np.allclose(result_a.x_mean_a, result_b.x_mean_a)


def test_localization_zeroes_out_masked_cross_covariance(toy_problem):
    rng, n_state, n_obs, H, R, Q_true = toy_problem
    n_members = 500
    X_f = rng.multivariate_normal(np.zeros(n_state), Q_true, size=n_members).T
    Y_f = H @ X_f
    _, S = ensemble_anomalies(X_f)
    _, Y = ensemble_anomalies(Y_f)

    localization = np.ones((n_obs, n_state))
    localization[:, -1] = 0.0   # mask out the last state component entirely

    K, _ = kalman_gain(S, Y, R, localization=localization)
    assert np.allclose(K[-1, :], 0.0)


def test_apply_inflation_scalar_and_selective():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4, 1000))
    x_mean = X.mean(axis=1, keepdims=True)

    X_inflated = apply_inflation(X, 2.0)
    np.testing.assert_allclose(X_inflated - x_mean, 2.0 * (X - x_mean))

    mask = np.array([1.0, 1.0, 3.0, 1.0])
    X_selective = apply_inflation(X, mask)
    np.testing.assert_allclose(X_selective[2] - x_mean[2, 0], 3.0 * (X[2] - x_mean[2, 0]))
    for k in (0, 1, 3):
        np.testing.assert_allclose(X_selective[k], X[k])

    assert np.allclose(apply_inflation(X, None), X)
    assert np.allclose(apply_inflation(X, 1.0), X)
