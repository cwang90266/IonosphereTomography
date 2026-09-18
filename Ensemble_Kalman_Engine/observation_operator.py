#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic, style-independent observation operator (plan Section 5.4).

Confirmed: the ANCHOR analytic fast-path integrator
(``Ionosphere_Tomography_Inverter/observation_operator.py``) is retired
entirely. Every style, ANCHOR included, goes through the same two-step
route:

    1. decode:  physical density = EDP_Parameterization.get_density(param_vec)
    2. predict: TEC = H_grid @ flatten(physical density)

``H_grid`` (the ray line-integral matrix) comes from
``EDPSamples.get_observation_operator`` -- imported, not replicated, per
the plan's Section 5.0 code-reuse policy: it is shared, style-independent
infrastructure, not part of the old ANCHOR-only analysis engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .jacobian_utils import compose_observation_jacobian


@dataclass
class GenericObservationOperator:
    """
    Wraps a parameterization + the physical ``H_grid`` into one operator
    usable by the analysis engine regardless of style.

    Parameters
    ----------
    parameterization : Parameterization.EDP_Parameterization
        Any style ('raw', 'density_10ex', 'ANCHOR', 'PCA_1D', 'PCA_3D',
        ...) -- this class never branches on which one.
    H_grid : ndarray, shape (n_obs, n_height * n_geo)
        From ``EDPSamples.get_observation_operator(...)``; TEC units
        already applied (see that method's docstring).
    param_shape : tuple[int, ...]
        The state's natural (pre-flatten) shape for this style -- e.g.
        ``(8, n_geo)`` for ANCHOR, ``(n_height, n_geo)`` for
        raw/density_10ex, ``(nPCA, n_geo)`` for PCA_1D*, ``(nPCA,)`` for
        PCA_3D* (plan Section 5.2).
    n_height, n_geo : int
        Physical grid dimensions.
    altitude : ndarray, shape (n_height,), optional
        Passed through to ``get_density``/``get_Jacobian``; ignored by
        styles where ``needs_altitude`` is False (e.g. everything except
        ANCHOR).
    """

    parameterization: object
    H_grid: np.ndarray
    param_shape: tuple[int, ...]
    n_height: int
    n_geo: int
    altitude: np.ndarray | None = None

    @property
    def n_state(self) -> int:
        return int(np.prod(self.param_shape))

    @property
    def n_obs(self) -> int:
        return self.H_grid.shape[0]

    def _alt_arg(self) -> np.ndarray | None:
        needs_alt = getattr(self.parameterization, "needs_altitude", False)
        return self.altitude if needs_alt else None

    def _flatten_physical(self, density: np.ndarray) -> np.ndarray:
        trailing = density.shape[2:]
        return density.reshape(self.n_height * self.n_geo, *trailing)

    def decode(self, param_vec: np.ndarray) -> np.ndarray:
        """
        ``param_vec`` shape ``param_shape`` (single member) or
        ``param_shape + (n_members,)`` (ensemble) -> physical density of
        shape ``(n_height, n_geo)`` or ``(n_height, n_geo, n_members)``.

        Some styles' ``to_density`` (e.g. ``PCA_3D``/``PCA_3D_10ex``, via
        ``PCA2EDP_3D``) keep a trailing "sample" axis even for a single,
        no-member-axis input, unlike every other style -- caught by a
        real-data regression test (a single-vector ``forward_single`` call
        silently returned shape ``(n_obs, 1)`` instead of ``(n_obs,)`` for
        ``PCA_3D_10ex``, corrupting the analysis-engine broadcast). Squeeze
        that trailing singleton here so every style gives this method one
        consistent contract.
        """
        density = np.asarray(
            self.parameterization.get_density(param_vec, alt=self._alt_arg())
        )
        is_single = param_vec.shape == self.param_shape
        expected_ndim = 2 if is_single else 3
        if density.ndim == expected_ndim + 1 and density.shape[-1] == 1:
            density = density[..., 0]
        elif density.ndim != expected_ndim:
            raise ValueError(
                f"GenericObservationOperator.decode: unexpected density "
                f"shape {density.shape} for param_vec.shape={param_vec.shape} "
                f"(param_shape={self.param_shape}); expected ndim "
                f"{expected_ndim} (or {expected_ndim + 1} with a trailing "
                "singleton)."
            )
        return density

    def forward(self, param_vec: np.ndarray) -> np.ndarray:
        """
        ``param_vec`` shape ``param_shape`` or ``param_shape + (n_members,)``
        -> simulated TEC, shape ``(n_obs,)`` or ``(n_obs, n_members)``.
        """
        density_flat = self._flatten_physical(self.decode(param_vec))
        return self.H_grid @ density_flat

    def forward_ensemble(self, X_f: np.ndarray) -> np.ndarray:
        """
        ``X_f``: flat filter-space ensemble, shape ``(n_state, n_members)``
        -> ``Y_f``, shape ``(n_obs, n_members)``.
        """
        n_members = X_f.shape[1]
        param_vec = X_f.reshape(*self.param_shape, n_members)
        return self.forward(param_vec)

    def forward_single(self, x: np.ndarray) -> np.ndarray:
        """
        ``x``: flat filter-space single state vector, shape ``(n_state,)``
        -> simulated TEC, shape ``(n_obs,)``.
        """
        param_vec = x.reshape(*self.param_shape)
        return self.forward(param_vec)

    def linearized(self, x: np.ndarray) -> np.ndarray:
        """
        Re-linearize at a single point (plan Section 6.2): ``x`` shape
        ``(n_state,)`` -> ``W_hat = H_grid @ grad_E(x)``, shape
        ``(n_obs, n_state)``.
        """
        param_vec = x.reshape(*self.param_shape)
        jac_kind, jac_array = self.parameterization.get_Jacobian(
            param_vec, alt=self._alt_arg()
        )
        return compose_observation_jacobian(
            self.H_grid, jac_array, self.param_shape,
            self.n_height, self.n_geo, jac_kind=jac_kind,
        )

    @classmethod
    def from_edp_samples(
        cls,
        edp_samples,
        parameterization,
        param_shape: tuple[int, ...],
        podTc2_data: dict,
        num_segments: int = 1000,
    ) -> "GenericObservationOperator":
        """
        Build the operator from a real ``EDPSamples`` instance: fetches
        ``H_grid`` via ``EDPSamples.get_observation_operator`` (the same
        integrator used for every style, per this module's docstring).
        """
        H_grid = edp_samples.get_observation_operator(
            podTc2_data, num_segments=num_segments
        )
        altitude = np.asarray(edp_samples.altitude)
        n_height = len(altitude)
        n_geo = edp_samples.geolocation.shape[0]
        return cls(
            parameterization=parameterization,
            H_grid=H_grid,
            param_shape=param_shape,
            n_height=n_height,
            n_geo=n_geo,
            altitude=altitude,
        )
