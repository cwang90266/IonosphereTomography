#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Kalman-filter linear algebra for the general-purpose ensemble engine.

Replicates the math already proven in
``Ionosphere_Tomography_Inverter/enkf_update.py`` (per the plan doc's
Section 5.0 code-reuse policy: rewritten from scratch here, not imported,
so this module carries no dependency on the ANCHOR-specific pipeline).

All functions operate on plain ``(n_state, n_members)`` / ``(n_obs, ...)``
arrays and know nothing about electron density, parameterizations, or
grids -- see ``General_EnKF_Implementation_Plan.md`` Section 1 for the
derivation this implements:

    Q  ~ S S^T,                         S: (n_state, n_members)
    K  = S Y^T (Y Y^T + R)^-1,          Y = H S: (n_obs, n_members)
    S_a[:, i] = S[:, i] - K (Y[:, i] + eps_i),   eps_i ~ (0, R)

Localization (Gaspari-Cohn or otherwise) is taken as a precomputed
``(n_obs, n_state)`` weight matrix supplied by the caller -- unlike the
old file, this module has no built-in assumption about how many
parameters sit at each grid point, so it never needs to reshape/tile a
per-grid-point localization matrix itself (Section 5.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as la


# ── Ensemble anomalies ──────────────────────────────────────────────────────

def ensemble_anomalies(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split an ensemble into its mean and scaled anomaly matrix.

    Parameters
    ----------
    X : ndarray, shape (n_state, n_members)

    Returns
    -------
    x_mean : ndarray, shape (n_state,)
    S : ndarray, shape (n_state, n_members)
        Columns are ``(X[:, i] - x_mean) / sqrt(n_members - 1)``, so that
        ``S @ S.T`` is the sample covariance of ``X`` (plan Section 1).
    """
    n_members = X.shape[1]
    if n_members < 2:
        raise ValueError(f"ensemble_anomalies: need >= 2 members, got {n_members}")
    x_mean = X.mean(axis=1)
    S = (X - x_mean[:, np.newaxis]) / np.sqrt(n_members - 1)
    return x_mean, S


# ── Kalman gain ──────────────────────────────────────────────────────────────

def kalman_gain(
    S: np.ndarray,
    Y: np.ndarray,
    R: np.ndarray,
    localization: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Ensemble Kalman gain, never forming the ``n_state x n_state`` covariance.

    Implements ``K = S Y^T (Y Y^T + R)^-1`` (plan Section 1), i.e.
    ``K = Q H^T (H Q H^T + R)^-1`` with ``Q ~ S S^T`` and ``Y = H S``.

    This is the single primitive shared by every analysis mode (plan
    Section 6.2): the one-shot ``linear`` update calls it with
    ``Y = H_grid @ S`` (the full nonlinear forecast anomalies); the
    ``iterated`` Gauss-Newton loop calls it once per iteration with
    ``Y = W_hat_i @ S`` for the re-linearized observation operator, holding
    ``S`` fixed throughout (the ensemble is not re-drawn between passes).

    Parameters
    ----------
    S : ndarray, shape (n_state, n_members)
        Scaled state anomalies (from ``ensemble_anomalies``, held fixed
        across a Gauss-Newton iteration).
    Y : ndarray, shape (n_obs, n_members)
        ``H @ S`` (linear case) or ``W_hat_i @ S`` (iterated case) --
        scaled observation-space anomalies consistent with ``S``.
    R : ndarray, shape (n_obs, n_obs)
        Observation error covariance.
    localization : ndarray, shape (n_obs, n_state), optional
        Elementwise (Schur-product) weights in [0, 1] applied to the
        state/observation cross-covariance before the gain solve.  Caller
        supplies whatever shape is appropriate for the state layout in
        use -- this module makes no assumption about grid structure.

    Returns
    -------
    K : ndarray, shape (n_state, n_obs)
    diagnostics : dict with ``P_yy`` and (if used) the localized ``P_xy``.
    """
    n_obs = Y.shape[0]
    P_yy = Y @ Y.T                      # (n_obs, n_obs)
    P_xy = S @ Y.T                      # (n_state, n_obs)

    if localization is not None:
        P_xy = P_xy * localization.T    # (n_state, n_obs), elementwise

    D = P_yy + R
    try:
        K_T = la.solve(D, P_xy.T, assume_a="pos")   # (n_obs, n_state)
    except la.LinAlgError:
        K_T = la.pinv(D) @ P_xy.T
    K = K_T.T

    return K, {"P_yy": P_yy, "P_xy": P_xy}


# ── Selective multiplicative inflation ──────────────────────────────────────

def apply_inflation(
    X: np.ndarray,
    inflation: float | np.ndarray | None,
) -> np.ndarray:
    """
    Multiplicative covariance inflation, optionally applied to only a
    portion of the state vector's components (plan Section 5.3).

    Parameters
    ----------
    X : ndarray, shape (n_state, n_members)
    inflation : None, float, or ndarray of shape (n_state,)
        ``None``/``1.0`` leaves ``X`` unchanged. A scalar inflates every
        component uniformly (the old, global-scalar behaviour). A
        per-component array inflates each state component by its own
        factor -- pass 1.0 for components that should be left alone, so
        inflation can be confined to a chosen subset (e.g. one ANCHOR
        parameter across all grid points, or one PCA mode).

    Returns
    -------
    ndarray, same shape as ``X``, inflated about the ensemble mean.
    """
    if inflation is None:
        return X
    factor = np.asarray(inflation, dtype=float)
    if factor.ndim == 0 and float(factor) == 1.0:
        return X
    x_mean = X.mean(axis=1, keepdims=True)
    if factor.ndim == 0:
        return x_mean + (X - x_mean) * factor
    return x_mean + (X - x_mean) * factor[:, np.newaxis]


# ── Stochastic (perturbed-observation) analysis update ─────────────────────

@dataclass
class AnalysisResult:
    """Container for one analysis-engine call's output and diagnostics."""

    X_a: np.ndarray
    x_mean_a: np.ndarray
    K: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def stochastic_update(
    X_f: np.ndarray,
    Y_f: np.ndarray,
    y_obs: np.ndarray,
    R: np.ndarray,
    localization: np.ndarray | None = None,
    inflation: float | np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> AnalysisResult:
    """
    One-shot stochastic (perturbed-observation) EnKF analysis update.

    This is the direct implementation of plan Section 1's final formula:
    the ``i``-th analysis member is

        X_a[:, i] = X_f[:, i] + K (y_obs + eps_i - Y_f[:, i]),  eps_i ~ (0, R)

    so that ``Q_a ~ S_a S_a^T`` for the resulting ensemble, without ever
    forming ``Q_a`` explicitly.

    Parameters
    ----------
    X_f : ndarray, shape (n_state, n_members)
        Forecast ensemble in filter space.
    Y_f : ndarray, shape (n_obs, n_members)
        Forecast ensemble mapped through the (possibly nonlinear)
        observation operator -- ``Y_f[:, i] = H(X_f[:, i])``.
    y_obs : ndarray, shape (n_obs,)
    R : ndarray, shape (n_obs, n_obs)
    localization, inflation : see ``kalman_gain`` / ``apply_inflation``.
    rng : numpy.random.Generator, optional
        Source of the observation perturbations; defaults to
        ``np.random.default_rng()``.

    Returns
    -------
    AnalysisResult
    """
    if rng is None:
        rng = np.random.default_rng()

    n_members = X_f.shape[1]
    n_obs = Y_f.shape[0]

    X_f = apply_inflation(X_f, inflation)

    x_mean, S = ensemble_anomalies(X_f)
    _, Y = ensemble_anomalies(Y_f)

    K, diag = kalman_gain(S, Y, R, localization=localization)

    try:
        L_chol = la.cholesky(R, lower=True)
    except la.LinAlgError:
        L_chol = la.cholesky(R + 1e-8 * np.eye(n_obs), lower=True)
    eps = L_chol @ rng.standard_normal((n_obs, n_members))

    innovation = (y_obs[:, np.newaxis] + eps) - Y_f
    X_a = X_f + K @ innovation

    x_mean_a = X_a.mean(axis=1)
    diag.update({
        "innovation_mean": innovation.mean(axis=1),
        "innovation_std": innovation.std(axis=1),
    })
    return AnalysisResult(X_a=X_a, x_mean_a=x_mean_a, K=K, diagnostics=diag)


# ── Deterministic (ensemble square-root) analysis update ───────────────────

def ensrf_update(
    X_f: np.ndarray,
    Y_f: np.ndarray,
    y_obs: np.ndarray,
    R: np.ndarray,
    localization: np.ndarray | None = None,
    inflation: float | np.ndarray | None = None,
) -> AnalysisResult:
    """
    One-shot deterministic EnSRF analysis update (no random perturbations).

    Mean update ``x_a = x_f + K(y_obs - y_f_mean)``; anomalies transformed
    by the symmetric square root ``W`` of ``I - Htilde^T D^-1 Htilde``, so
    that ``Q_a`` matches the exact identity in plan Section 1 to machine
    precision (rather than in expectation, as with ``stochastic_update``).

    Parameters and return value: as ``stochastic_update``.
    """
    n_members = X_f.shape[1]
    n_obs = Y_f.shape[0]

    X_f = apply_inflation(X_f, inflation)

    x_mean, S = ensemble_anomalies(X_f)
    y_mean, Y = ensemble_anomalies(Y_f)

    K, diag = kalman_gain(S, Y, R, localization=localization)

    x_mean_a = x_mean + K @ (y_obs - y_mean)

    D = Y @ Y.T + R
    try:
        D_inv_Y = la.solve(D, Y, assume_a="pos")
    except la.LinAlgError:
        D_inv_Y = la.pinv(D) @ Y
    A = np.eye(n_members) - Y.T @ D_inv_Y
    evals, evecs = np.linalg.eigh(A)
    evals = np.maximum(evals, 0.0)
    W = (evecs * np.sqrt(evals)) @ evecs.T

    S_a = S @ W
    X_a = x_mean_a[:, np.newaxis] + S_a * np.sqrt(n_members - 1)

    diag.update({"W_eigenvalues": evals})
    return AnalysisResult(X_a=X_a, x_mean_a=x_mean_a, K=K, diagnostics=diag)
