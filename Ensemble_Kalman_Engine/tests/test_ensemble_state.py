# -*- coding: utf-8 -*-
"""
Tests for the general ensemble/state container (plan Section 5.2):
flatten/unflatten round trips for every confirmed per-style shape --
``(n_height, n_geo)`` for raw/density_10ex, ``(8, n_geo)`` for ANCHOR,
``(nPCA, n_geo)`` for PCA_1D*, ``(nPCA,)`` for PCA_3D*.
"""
from __future__ import annotations

import numpy as np
import pytest

from Ensemble_Kalman_Engine.ensemble_state import (
    EnsembleState,
    flatten_param_vec,
    unflatten_to_param_shape,
)


@pytest.mark.parametrize(
    "param_shape",
    [
        (5, 3),      # raw / density_10ex: (n_height, n_geo)
        (8, 3),      # ANCHOR: (8, n_geo)
        (4, 3),      # PCA_1D*: (nPCA, n_geo)
        (6,),        # PCA_3D*: (nPCA,) -- no n_geo factor
    ],
)
def test_flatten_unflatten_round_trip(param_shape):
    rng = np.random.default_rng(0)
    n_members = 7
    param_vec = rng.standard_normal(param_shape + (n_members,))

    X = flatten_param_vec(param_vec, param_shape)
    assert X.shape == (int(np.prod(param_shape)), n_members)

    back = unflatten_to_param_shape(X, param_shape)
    np.testing.assert_array_equal(back, param_vec)


def test_ensemble_state_from_param_vec_and_accessors():
    param_shape = (8, 4)
    n_members = 10
    rng = np.random.default_rng(1)
    param_vec = rng.standard_normal(param_shape + (n_members,))

    state = EnsembleState.from_param_vec(param_vec, param_shape)
    assert state.n_state == 32
    assert state.n_members == n_members

    np.testing.assert_allclose(state.ensemble_mean(), param_vec.reshape(32, n_members).mean(axis=1))

    reshaped_all = state.to_param_shape()
    np.testing.assert_array_equal(reshaped_all, param_vec)

    reshaped_one = state.to_param_shape(column=3)
    np.testing.assert_array_equal(reshaped_one, param_vec[..., 3])


def test_ensemble_state_rejects_mismatched_shape():
    X = np.zeros((10, 5))
    with pytest.raises(ValueError):
        EnsembleState(X=X, param_shape=(3, 4))   # 12 != 10


def test_pca3d_shape_has_no_geo_factor():
    """Confirmed shape mapping: PCA_3D compresses across every horizontal
    grid point jointly, so its param_shape carries no n_geo axis at all."""
    param_shape = (12,)   # (nPCA,)
    n_members = 5
    param_vec = np.arange(12 * n_members, dtype=float).reshape(*param_shape, n_members)
    state = EnsembleState.from_param_vec(param_vec, param_shape)
    assert state.X.shape == (12, n_members)
