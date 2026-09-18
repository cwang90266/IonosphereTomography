# -*- coding: utf-8 -*-
"""Tests for the analysis-state / analysis-covariance perturbation hook
(plan Section 5.3): dither, selective inflation, bounds clamp, and their
composition -- all confirmed to be usable together, not either/or."""
from __future__ import annotations

import numpy as np

from Ensemble_Kalman_Engine.perturbation import (
    GaussianDither,
    SelectiveInflation,
    BoundsClamp,
    CustomPerturbation,
    PerturbationChain,
)


def test_default_chain_is_identity():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 50))
    chain = PerturbationChain([])
    np.testing.assert_array_equal(chain(X), X)


def test_gaussian_dither_changes_values_with_expected_scale():
    rng = np.random.default_rng(1)
    X = np.zeros((3, 20000))
    dither = GaussianDither(std=0.5)
    X_out = dither(X, rng=rng)
    assert not np.allclose(X_out, X)
    np.testing.assert_allclose(X_out.std(axis=1), 0.5, atol=0.02)
    np.testing.assert_allclose(X_out.mean(axis=1), 0.0, atol=0.02)


def test_gaussian_dither_per_component_std():
    rng = np.random.default_rng(2)
    X = np.zeros((2, 20000))
    dither = GaussianDither(std=np.array([0.1, 1.0]))
    X_out = dither(X, rng=rng)
    assert X_out[1].std() > 5 * X_out[0].std()


def test_selective_inflation_only_inflates_chosen_components():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((4, 1000))
    factor = np.array([1.0, 1.0, 4.0, 1.0])
    inflate = SelectiveInflation(factor=factor)
    X_out = inflate(X)

    x_mean = X.mean(axis=1)
    for k in (0, 1, 3):
        np.testing.assert_allclose(X_out[k], X[k])
    np.testing.assert_allclose(X_out[2] - x_mean[2], 4.0 * (X[2] - x_mean[2]))


def test_bounds_clamp_matches_np_clip_and_is_deterministic():
    X = np.array([[-5.0, 0.5, 10.0], [2.0, -2.0, 3.0]])
    lo = np.array([0.0, -1.0])
    hi = np.array([1.0, 1.0])
    clamp = BoundsClamp(lo=lo, hi=hi)
    X_out_1 = clamp(X)
    X_out_2 = clamp(X)
    np.testing.assert_array_equal(X_out_1, X_out_2)
    expected = np.array([[0.0, 0.5, 1.0], [1.0, -1.0, 1.0]])
    np.testing.assert_array_equal(X_out_1, expected)


def test_custom_perturbation_wraps_arbitrary_callable():
    X = np.ones((2, 3))
    custom = CustomPerturbation(fn=lambda arr: arr * 2.0)
    np.testing.assert_array_equal(custom(X), 2.0 * X)


def test_chain_composes_stages_in_order():
    rng = np.random.default_rng(4)
    X = np.ones((3, 5000))
    lo = np.array([-0.5, -0.5, -0.5])
    hi = np.array([0.5, 0.5, 0.5])

    chain = PerturbationChain([
        SelectiveInflation(factor=np.array([1.0, 1.0, 100.0])),  # blow up row 2
        BoundsClamp(lo=lo, hi=hi),                                # then clamp everything
    ])
    X_out = chain(X, rng=rng)
    assert np.all(X_out >= lo[:, None] - 1e-9)
    assert np.all(X_out <= hi[:, None] + 1e-9)
