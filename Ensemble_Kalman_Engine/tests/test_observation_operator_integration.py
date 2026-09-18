# -*- coding: utf-8 -*-
"""
Integration tests for GenericObservationOperator + jacobian_utils against
the *real* Parameterization.EDP_Parameterization styles (plan Section 5.4)
-- not the hand-rolled toy operators used in test_analysis_engine.py.

Uses a small synthetic H_grid (not a real EDPSamples/IRI-backed one) so
these stay fast and don't require IRI data files -- see the plan doc's
Section 5.7 "per-style sanity cycle" bullet for the full, real-data
version of this check, which is out of scope here.

The key check is a finite-difference validation of ``linearized()``
against ``forward_single()`` for each style: this exercises the real
per-style Jacobian shapes returned by ``get_Jacobian`` (including
ANCHOR's nontrivial, physically nonlinear one) through
``jacobian_utils.compose_observation_jacobian``'s shape-dispatch logic
end-to-end, which the unit tests in test_jacobian_utils.py only check
against hand-built synthetic arrays.
"""
from __future__ import annotations

import numpy as np
import pytest

Parameterization = pytest.importorskip(
    "Parameterization", reason="Parameterization.py and its dependencies not importable in this environment"
)

from Ensemble_Kalman_Engine.observation_operator import GenericObservationOperator


def _finite_difference_check(obs_op, x, rng, eps=1e-5, rtol=1e-3):
    y0 = obs_op.forward_single(x)
    W = obs_op.linearized(x)
    dx = rng.standard_normal(x.shape)
    dx = dx / np.linalg.norm(dx)
    y1 = obs_op.forward_single(x + eps * dx)
    fd = (y1 - y0) / eps
    analytic = W @ dx
    rel_err = np.linalg.norm(fd - analytic) / max(np.linalg.norm(analytic), 1e-12)
    assert rel_err < rtol, f"finite-difference mismatch: {rel_err}"
    return y0, W


@pytest.fixture
def synthetic_grid():
    rng = np.random.default_rng(0)
    n_height, n_geo = 15, 3
    altitude = np.linspace(90.0, 500.0, n_height)
    n_obs = 5
    H_grid = rng.standard_normal((n_obs, n_height * n_geo)) * 0.1
    return rng, n_height, n_geo, altitude, H_grid


def test_raw_style_jacobian_is_identity_composed_with_H(synthetic_grid):
    rng, n_height, n_geo, altitude, H_grid = synthetic_grid
    param_shape = (n_height, n_geo)
    param = Parameterization.EDP_Parameterization(style="raw")
    obs_op = GenericObservationOperator(
        parameterization=param, H_grid=H_grid, param_shape=param_shape,
        n_height=n_height, n_geo=n_geo, altitude=None,
    )
    x = rng.uniform(9.0, 12.0, size=param_shape).reshape(-1)
    y0, W = _finite_difference_check(obs_op, x, rng)
    np.testing.assert_allclose(y0, H_grid @ x)
    np.testing.assert_allclose(W, H_grid)


def test_density_10ex_style_jacobian_matches_finite_difference(synthetic_grid):
    rng, n_height, n_geo, altitude, H_grid = synthetic_grid
    param_shape = (n_height, n_geo)
    param = Parameterization.EDP_Parameterization(style="density_10ex")
    obs_op = GenericObservationOperator(
        parameterization=param, H_grid=H_grid, param_shape=param_shape,
        n_height=n_height, n_geo=n_geo, altitude=None,
    )
    x = rng.uniform(9.0, 12.0, size=param_shape).reshape(-1)
    _finite_difference_check(obs_op, x, rng)


def test_anchor_style_jacobian_matches_finite_difference(synthetic_grid):
    """The physically nonlinear, block-diagonal-by-geo-column case --
    what Remedy A (Section 6.2) actually relies on for ANCHOR."""
    rng, n_height, n_geo, altitude, H_grid = synthetic_grid
    param_shape = (8, n_geo)
    param = Parameterization.EDP_Parameterization(style="ANCHOR")
    obs_op = GenericObservationOperator(
        parameterization=param, H_grid=H_grid, param_shape=param_shape,
        n_height=n_height, n_geo=n_geo, altitude=altitude,
    )
    base = np.array([12.0, 300.0, 50.0, 0.5, 100.0, 2.0, 11.0, 110.0])
    param_vec = np.tile(base[:, None], (1, n_geo)) + rng.normal(scale=0.01, size=(8, n_geo))
    x = param_vec.reshape(-1)
    _finite_difference_check(obs_op, x, rng, rtol=1e-3)


def test_anchor_jacobian_is_block_diagonal_by_geo_column(synthetic_grid):
    """Physical sanity check backing jacobian_utils' block-diagonal
    shortcut: perturbing one geo column's ANCHOR parameters must not move
    the simulated density in any *other* geo column."""
    rng, n_height, n_geo, altitude, H_grid = synthetic_grid
    param_shape = (8, n_geo)
    param = Parameterization.EDP_Parameterization(style="ANCHOR")

    base = np.array([12.0, 300.0, 50.0, 0.5, 100.0, 2.0, 11.0, 110.0])
    param_vec = np.tile(base[:, None], (1, n_geo))

    density_before = param.get_density(param_vec, alt=altitude)
    perturbed = param_vec.copy()
    perturbed[:, 0] *= 1.01   # perturb only geo column 0
    density_after = param.get_density(perturbed, alt=altitude)

    np.testing.assert_allclose(density_after[:, 1:], density_before[:, 1:])
    assert not np.allclose(density_after[:, 0], density_before[:, 0])


def test_pca3d_style_single_vector_forward_has_no_stray_trailing_axis(synthetic_grid):
    """Regression test: PCA2EDP_3D (unlike every other style's to_density)
    keeps a trailing size-1 'sample' axis even for a single, no-member-axis
    input. Caught via a real-data plot script where forward_single silently
    returned shape (n_obs, 1) instead of (n_obs,) for PCA_3D_10ex, breaking
    the analysis engine's broadcast. decode()/forward_single() must squeeze
    that singleton so every style gives the same (n_obs,) contract."""
    rng, n_height, n_geo, altitude, H_grid = synthetic_grid
    n_pca = 4
    param_shape = (n_pca,)

    PCA = rng.standard_normal((n_height, n_geo, n_pca))
    PCA_mean = 11.0 + 0.1 * rng.standard_normal((n_height, n_geo))
    param = Parameterization.EDP_Parameterization(
        style="PCA_3D_10ex", hyper_params={"PCA": PCA, "PCA_mean": PCA_mean}
    )
    obs_op = GenericObservationOperator(
        parameterization=param, H_grid=H_grid, param_shape=param_shape,
        n_height=n_height, n_geo=n_geo, altitude=None,
    )

    x = rng.standard_normal(n_pca)
    density_single = obs_op.decode(x.reshape(param_shape))
    assert density_single.shape == (n_height, n_geo)

    y_single = obs_op.forward_single(x)
    assert y_single.shape == (H_grid.shape[0],)

    _finite_difference_check(obs_op, x, rng, rtol=1e-3)

    # Ensemble path should be unaffected (already had the right shape).
    X_f = rng.standard_normal((n_pca, 6))
    Y_f = obs_op.forward_ensemble(X_f)
    assert Y_f.shape == (H_grid.shape[0], 6)
