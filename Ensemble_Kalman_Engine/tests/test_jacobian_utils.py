# -*- coding: utf-8 -*-
"""
Tests for compose_observation_jacobian's three shape-dispatch cases (see
jacobian_utils.py's module docstring): diagonal (raw/density_10ex),
block-diagonal-by-geo-column (ANCHOR, PCA_1D*), and the dense fallback
(PCA_3D*). Each is checked against a manually-built dense reference
matrix -- the thing the efficient paths are shortcuts for.
"""
from __future__ import annotations

import numpy as np
import pytest

from Ensemble_Kalman_Engine.jacobian_utils import compose_observation_jacobian


def test_diagonal_case_matches_dense_diag_reference():
    rng = np.random.default_rng(0)
    n_height, n_geo = 4, 3
    param_shape = (n_height, n_geo)
    n_obs = 5

    H_grid = rng.standard_normal((n_obs, n_height * n_geo))
    jac_array = rng.uniform(0.5, 2.0, size=(n_height, n_geo))

    W_hat = compose_observation_jacobian(H_grid, jac_array, param_shape, n_height, n_geo, jac_kind="diagonal")

    J_dense = np.diag(jac_array.reshape(-1))
    W_hat_expected = H_grid @ J_dense
    np.testing.assert_allclose(W_hat, W_hat_expected)


def test_block_diagonal_case_matches_dense_reference():
    rng = np.random.default_rng(1)
    n_height, n_geo, n_param_per_col = 4, 3, 2
    param_shape = (n_param_per_col, n_geo)
    n_obs = 5

    H_grid = rng.standard_normal((n_obs, n_height * n_geo))
    jac_array = rng.standard_normal((n_height, n_param_per_col, n_geo))

    W_hat = compose_observation_jacobian(H_grid, jac_array, param_shape, n_height, n_geo, jac_kind="per_point")

    n_state = n_param_per_col * n_geo
    n_physical = n_height * n_geo
    J_dense = np.zeros((n_physical, n_state))
    for h in range(n_height):
        for g in range(n_geo):
            phys_idx = h * n_geo + g
            for k in range(n_param_per_col):
                state_idx = k * n_geo + g
                J_dense[phys_idx, state_idx] = jac_array[h, k, g]
    W_hat_expected = H_grid @ J_dense
    np.testing.assert_allclose(W_hat, W_hat_expected, atol=1e-10)


def test_dense_fallback_case_pca3d_style():
    rng = np.random.default_rng(2)
    n_height, n_geo, n_pca = 4, 3, 5
    param_shape = (n_pca,)
    n_obs = 6

    H_grid = rng.standard_normal((n_obs, n_height * n_geo))
    # PCA_3D-style Jacobian shape (n_height, n_geo, nPCA) for a single
    # parameter vector (trailing sample axis already squeezed).
    jac_array = rng.standard_normal((n_height, n_geo, n_pca))

    W_hat = compose_observation_jacobian(H_grid, jac_array, param_shape, n_height, n_geo, jac_kind="full")

    J_dense = jac_array.reshape(n_height * n_geo, n_pca)
    W_hat_expected = H_grid @ J_dense
    np.testing.assert_allclose(W_hat, W_hat_expected)


def test_unrecognized_shape_raises():
    H_grid = np.zeros((3, 12))
    jac_array = np.zeros((7, 7))   # doesn't match any recognized pattern
    with pytest.raises(ValueError):
        compose_observation_jacobian(H_grid, jac_array, (7, 7), n_height=4, n_geo=3)
