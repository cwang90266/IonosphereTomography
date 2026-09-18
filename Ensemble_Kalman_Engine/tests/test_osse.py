# -*- coding: utf-8 -*-
"""
Tests for OSSE support (plan Section 5.8): the synthetic-observation
generator, and the "strongest available test" end-to-end loop from plan
Section 5.7 -- generate an observation from a known truth, run the
engine's normal analyze() entry point on it, and check the analysis
measurably recovers the truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Ensemble_Kalman_Engine.osse import generate_osse_observation, recovery_error
from Ensemble_Kalman_Engine.analysis_engine import AnalysisEngine, AnalysisConfig
from Ensemble_Kalman_Engine.ensemble_state import EnsembleState


@dataclass
class LinearOp:
    H: np.ndarray

    def forward_ensemble(self, X_f):
        return self.H @ X_f

    def forward_single(self, x):
        return self.H @ x

    def linearized(self, x):
        return self.H


def test_generate_osse_observation_is_forward_plus_noise():
    rng = np.random.default_rng(0)
    n_state, n_obs = 5, 3
    H = rng.standard_normal((n_obs, n_state))
    op = LinearOp(H=H)
    R = 0.01 * np.eye(n_obs)
    x_true = rng.standard_normal(n_state)

    osse = generate_osse_observation(x_true, op, R, rng=np.random.default_rng(1))

    np.testing.assert_allclose(osse.y_true_noiseless, H @ x_true)
    assert not np.allclose(osse.y_obs, osse.y_true_noiseless)
    # noise magnitude should be roughly consistent with R (loose check)
    noise = osse.y_obs - osse.y_true_noiseless
    assert np.all(np.abs(noise) < 10 * np.sqrt(np.diag(R)))


def test_generate_osse_observation_uses_same_operator_no_special_case():
    """The whole point of the split (plan 5.8): OSSE just calls the
    operator's normal forward_single -- no engine-side special case."""
    rng = np.random.default_rng(2)
    H = rng.standard_normal((3, 4))
    op = LinearOp(H=H)
    R = 0.01 * np.eye(3)
    x_true = rng.standard_normal(4)

    osse = generate_osse_observation(x_true, op, R, rng=np.random.default_rng(0))
    assert np.allclose(osse.y_true_noiseless, op.forward_single(x_true))


def test_recovery_error_zero_for_perfect_match():
    x_true = np.array([1.0, 2.0, 3.0])
    err = recovery_error(x_true, x_true.copy())
    assert err["rmse"] == 0.0
    assert err["max_abs_error"] == 0.0
    assert err["relative_l2_error"] == 0.0


def test_osse_recovery_end_to_end_moves_estimate_towards_truth():
    """The core OSSE validation loop (plan Section 5.7): assimilating a
    synthetic observation generated from a known truth should move the
    analysis mean substantially closer to that truth than the prior
    ensemble mean was."""
    rng = np.random.default_rng(10)
    n_state, n_obs, n_members = 6, 4, 400
    H = rng.standard_normal((n_obs, n_state))
    R = 0.02 * np.eye(n_obs)
    op = LinearOp(H=H)

    prior_mean = np.zeros(n_state)
    x_true = prior_mean + rng.standard_normal(n_state) * 1.5
    X_f = prior_mean[:, None] + 1.0 * rng.standard_normal((n_state, n_members))
    ensemble = EnsembleState(X=X_f, param_shape=(n_state,))

    osse = generate_osse_observation(x_true, op, R, rng=np.random.default_rng(11))

    prior_err = recovery_error(x_true, prior_mean)["rmse"]

    config = AnalysisConfig(linearization="linear", centering="mean", rng=np.random.default_rng(3))
    _, diag = AnalysisEngine().analyze(ensemble, op, osse.y_obs, R, config)
    posterior_err = recovery_error(x_true, diag.x_mean_a)["rmse"]

    assert posterior_err < 0.5 * prior_err
