#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compose ``EDP_Parameterization.get_Jacobian`` output with the physical,
linear ``H_grid`` operator to build the linearized observation operator
``W_hat = H_grid @ grad_E(P)`` used by the ``iterated`` analysis mode
(plan Section 6.2).

``get_Jacobian`` returns one of several shapes depending on style (see
``Parameterization.py``'s ``ParameterizationSpec.jacobian`` for each
style). Rather than trust the style's own ``kind`` label, this module
dispatches on the *shape* of the returned array relative to the known
physical grid ``(n_height, n_geo)`` and the state's ``param_shape`` --
more robust, and it naturally unifies styles whose Jacobian happens to
share a layout (e.g. ANCHOR's ``per_point`` and PCA_1D's
``block_diagonal`` are both literally an ``(n_height, k, n_geo)``,
block-diagonal-by-geo-column tensor, and are handled by the same code
path below).

Three shape patterns are recognized, for a Jacobian evaluated at a
*single* parameter vector (no ensemble axis):

1. **Diagonal** -- ``param_shape == (n_height, n_geo)`` and the Jacobian
   has that same shape (e.g. ``density_10ex``, ``raw``): an elementwise
   map, composed with ``H_grid`` by column-scaling (no dense matrix
   formed).
2. **Block-diagonal by geo column** -- ``param_shape == (k, n_geo)`` and
   the Jacobian has shape ``(n_height, k, n_geo)`` (e.g. ``ANCHOR``,
   ``PCA_1D*``): each geo column's physical profile depends only on that
   column's own parameters. Composed via a per-column einsum, still
   without ever forming the full ``(n_physical, n_state)`` dense matrix.
3. **Dense fallback** -- anything else (e.g. ``PCA_3D*``, which compresses
   across geo columns jointly): the Jacobian's leading two axes are
   ``(n_height, n_geo)``; the rest is reshaped and matched to ``n_state``.
   Formed densely -- correct for any style, and cheap in practice because
   every confirmed dense-fallback style (PCA) has a small reduced state.
"""

from __future__ import annotations

import numpy as np


def compose_observation_jacobian(
    H_grid: np.ndarray,
    jac_array: np.ndarray,
    param_shape: tuple[int, ...],
    n_height: int,
    n_geo: int,
    jac_kind: str | None = None,
) -> np.ndarray:
    """
    Build ``W_hat = H_grid @ grad_E(P)`` for a single parameter vector.

    Parameters
    ----------
    H_grid : ndarray, shape (n_obs, n_height * n_geo)
        Physical, style-independent ray-integration operator (from
        ``EDPSamples.get_observation_operator``).
    jac_array : ndarray
        The array half of ``EDP_Parameterization.get_Jacobian(p, alt)``'s
        ``(kind, array)`` return, evaluated at a *single* parameter
        vector of shape ``param_shape`` (no ensemble/member axis).
    param_shape : tuple[int, ...]
        The state's natural (pre-flatten) shape, e.g. ``(8, n_geo)`` for
        ANCHOR, ``(n_height, n_geo)`` for raw/density_10ex, ``(nPCA,)``
        for PCA_3D.
    n_height, n_geo : int
        Physical grid dimensions (``H_grid``'s column count is
        ``n_height * n_geo``).
    jac_kind : str, optional
        The ``kind`` string from ``get_Jacobian``, kept only for error
        messages / diagnostics -- dispatch itself is shape-based (see
        module docstring for why).

    Returns
    -------
    ndarray, shape (n_obs, n_state), where ``n_state = prod(param_shape)``.
    """
    n_obs = H_grid.shape[0]
    n_state = int(np.prod(param_shape))
    jac_array = np.asarray(jac_array, dtype=float)

    # -- Case 1: diagonal (elementwise) map -------------------------------
    if param_shape == (n_height, n_geo) and jac_array.shape == (n_height, n_geo):
        jac_flat = jac_array.reshape(n_height * n_geo)
        return H_grid * jac_flat[np.newaxis, :]

    # -- Case 2: block-diagonal by geo column ------------------------------
    if (
        len(param_shape) == 2
        and param_shape[1] == n_geo
        and jac_array.shape == (n_height, param_shape[0], n_geo)
    ):
        n_param_per_col = param_shape[0]
        H_reshaped = H_grid.reshape(n_obs, n_height, n_geo)
        W_reshaped = np.einsum("ohg,hkg->okg", H_reshaped, jac_array)
        return W_reshaped.reshape(n_obs, n_param_per_col * n_geo)

    # -- Case 3: dense fallback ---------------------------------------------
    if jac_array.shape[:2] == (n_height, n_geo):
        jac_dense = jac_array.reshape(n_height * n_geo, -1)
        if jac_dense.shape[1] != n_state:
            raise ValueError(
                f"compose_observation_jacobian: dense Jacobian has "
                f"{jac_dense.shape[1]} trailing components after the "
                f"(n_height, n_geo) leading axes, expected n_state="
                f"{n_state} for param_shape={param_shape} (kind="
                f"{jac_kind!r}, raw shape={jac_array.shape})."
            )
        return H_grid @ jac_dense

    raise ValueError(
        f"compose_observation_jacobian: unrecognized Jacobian shape "
        f"{jac_array.shape} for param_shape={param_shape}, physical grid "
        f"=({n_height}, {n_geo}), kind={jac_kind!r}."
    )
