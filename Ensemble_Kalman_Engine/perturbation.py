#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis-state / analysis-covariance perturbation hook (plan Section 5.3).

The post-update bounds clamp used by the old ANCHOR-only pipeline is not a
separate mechanism here -- it is one (deterministic) special case of a more
general, optional, additive perturbation applied after the analysis, to
the state and/or to the analysis covariance:

- ``GaussianDither``   -- additive state perturbation (stochastic).
- ``SelectiveInflation`` -- multiplicative covariance perturbation applied
  to only a chosen portion of the state vector's components.
- ``BoundsClamp``      -- additive state perturbation
  ``clip(x, lo, hi) - x`` (deterministic, zero almost everywhere) -- the
  special case that replaces ``clamp_to_physical_bounds`` from the old
  pipeline.

All three share one interface, ``perturb(X_a) -> X_a'``, and compose via
``PerturbationChain`` -- confirmed design: additive dither and selective
inflation are used *together*, not either/or.  The default
(``PerturbationChain([])`` / ``None``) is the identity: confirmed not
needed in most cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from .kalman_core import apply_inflation


class Perturbation(Protocol):
    """Interface every perturbation hook satisfies: ``X_a -> X_a'``."""

    def __call__(self, X_a: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        ...


@dataclass
class GaussianDither:
    """
    Additive zero-mean Gaussian noise on each analysis member, guarding
    against ensemble collapse under repeated analysis cycles.

    Parameters
    ----------
    std : float or ndarray, shape (n_state,)
        Per-component standard deviation of the dither. A scalar applies
        the same std to every component.
    """

    std: float | np.ndarray

    def __call__(self, X_a: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        std = np.asarray(self.std, dtype=float)
        n_state, n_members = X_a.shape
        noise = rng.standard_normal((n_state, n_members))
        if std.ndim == 0:
            return X_a + std * noise
        return X_a + std[:, np.newaxis] * noise


@dataclass
class SelectiveInflation:
    """
    Multiplicative covariance inflation applied to only a chosen portion
    of the state vector (plan Section 5.3): a per-component weight/mask
    over the inflation factor, rather than one global scalar.

    Parameters
    ----------
    factor : float or ndarray, shape (n_state,)
        Inflation factor per component (1.0 = no change). Pass a scalar
        for uniform inflation, or an array with 1.0 on components that
        should be left untouched to inflate only a subset.
    """

    factor: float | np.ndarray

    def __call__(self, X_a: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return apply_inflation(X_a, self.factor)


@dataclass
class BoundsClamp:
    """
    Deterministic additive perturbation ``clip(x, lo, hi) - x``: the
    special case that replaces the old pipeline's
    ``clamp_to_physical_bounds``.

    Parameters
    ----------
    lo, hi : ndarray, shape (n_state,)
        Per-component bounds. Use ``-np.inf``/``np.inf`` on components
        that should not be clamped.
    """

    lo: np.ndarray
    hi: np.ndarray

    def __call__(self, X_a: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return np.clip(X_a, self.lo[:, np.newaxis], self.hi[:, np.newaxis])


@dataclass
class CustomPerturbation:
    """Wrap an arbitrary ``X_a -> X_a'`` callable as a perturbation hook."""

    fn: Callable[[np.ndarray], np.ndarray]

    def __call__(self, X_a: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return self.fn(X_a)


@dataclass
class PerturbationChain:
    """
    Compose several perturbation hooks, applied in order.

    Confirmed default: an empty chain (identity) -- most analyses need no
    perturbation hook at all. Build one explicitly, e.g.::

        PerturbationChain([
            SelectiveInflation(factor=mask),
            GaussianDither(std=0.01),
        ])
    """

    stages: list[Perturbation] = field(default_factory=list)

    def __call__(self, X_a: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        for stage in self.stages:
            X_a = stage(X_a, rng=rng)
        return X_a
