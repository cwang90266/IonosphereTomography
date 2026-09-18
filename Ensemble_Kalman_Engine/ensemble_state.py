#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General, style-agnostic ensemble/state container (plan Section 5.2).

Replaces the ANCHOR-only ``IonosphericState``: wraps a filter-space
ensemble as a plain ``(n_state, n_members)`` array regardless of which
per-style shape ``n_state`` was flattened from --
``(n_height, n_geo)`` for ``raw``/``density_10ex``, ``(8, n_geo)`` for
``ANCHOR``, ``(nPCA, n_geo)`` for ``PCA_1D``/``PCA_1D_10ex``, or
``(nPCA,)`` for ``PCA_3D``/``PCA_3D_10ex`` (no ``n_geo`` factor, since
``PCA_3D`` compresses across all horizontal grid points jointly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def flatten_param_vec(param_vec: np.ndarray, param_shape: tuple[int, ...]) -> np.ndarray:
    """``param_shape (+ n_members)`` -> ``(n_state, n_members)``."""
    n_state = int(np.prod(param_shape))
    if param_vec.shape == param_shape:
        return param_vec.reshape(n_state, 1)
    if param_vec.shape[: len(param_shape)] != param_shape:
        raise ValueError(
            f"flatten_param_vec: expected leading shape {param_shape}, got "
            f"{param_vec.shape}"
        )
    n_members = param_vec.shape[len(param_shape):]
    return param_vec.reshape(n_state, *n_members) if n_members else param_vec.reshape(n_state)


def unflatten_to_param_shape(X: np.ndarray, param_shape: tuple[int, ...]) -> np.ndarray:
    """``(n_state,)`` or ``(n_state, n_members)`` -> ``param_shape (+ n_members)``."""
    if X.ndim == 1:
        return X.reshape(*param_shape)
    n_members = X.shape[1]
    return X.reshape(*param_shape, n_members)


@dataclass
class EnsembleState:
    """
    Filter-space ensemble container: a plain ``(n_state, n_members)``
    array plus the per-style shape needed to reconstruct a physical
    (decodable) parameter vector from any single column or the whole
    ensemble (``GenericObservationOperator`` and ``AnalysisEngine`` both
    consume this shape via ``to_param_shape``).
    """

    X: np.ndarray
    param_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        n_state = int(np.prod(self.param_shape))
        if self.X.shape[0] != n_state:
            raise ValueError(
                f"EnsembleState: X has {self.X.shape[0]} state components, "
                f"expected {n_state} from param_shape={self.param_shape}"
            )

    @property
    def n_state(self) -> int:
        return self.X.shape[0]

    @property
    def n_members(self) -> int:
        return self.X.shape[1]

    def ensemble_mean(self) -> np.ndarray:
        return self.X.mean(axis=1)

    def to_param_shape(self, column: int | None = None) -> np.ndarray:
        """
        Reshape back to the style's natural layout, for handing to
        ``EDP_Parameterization.get_density``/``get_Jacobian``.

        ``column=None`` reshapes the whole ensemble to
        ``param_shape + (n_members,)``; an integer reshapes just that
        one member to ``param_shape``.
        """
        if column is None:
            return unflatten_to_param_shape(self.X, self.param_shape)
        return unflatten_to_param_shape(self.X[:, column], self.param_shape)

    @classmethod
    def from_parameterized_edp_samples(cls, parameterized_edp_samples) -> "EnsembleState":
        """
        Build the initial ensemble from a real
        ``Parameterization.Parameterized_EDPSamples`` instance -- so the
        forecast ensemble is always real IRI-sampled data run through the
        chosen style's ``get_parameter`` encode, never a synthetic
        Gaussian draw (plan Section 5.2).
        """
        param_vec = np.asarray(parameterized_edp_samples.EDPSamples["param_vec"].values)
        param_shape = param_vec.shape[:-1]
        n_members = param_vec.shape[-1]
        X = param_vec.reshape(-1, n_members)
        return cls(X=X, param_shape=param_shape)

    @classmethod
    def from_param_vec(cls, param_vec: np.ndarray, param_shape: tuple[int, ...]) -> "EnsembleState":
        """Build directly from an already-encoded ``param_shape + (n_members,)`` array."""
        return cls(X=flatten_param_vec(param_vec, param_shape), param_shape=param_shape)
