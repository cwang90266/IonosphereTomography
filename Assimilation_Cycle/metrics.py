#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regular DA metrics (plan Section 4.13): RMSE reduction, rank histograms,
and effective ensemble rank. None of this existed anywhere in the
codebase before this module -- ``AnalysisDiagnostics``
(``Ensemble_Kalman_Engine.analysis_engine``) tracks convergence/iteration
internals, not forecast/analysis skill against observations.

Each function here is a pure computation over already-available arrays
(``y_obs``, forward-modeled ensembles, filter-space ensembles) -- no
``EDPSamples``/``Parameterization``/xarray dependency, so these are cheap
to unit test and can run on any batch regardless of parameterization
style.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rmse(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(residual) ** 2)))


@dataclass
class RmseReduction:
    """Observation-minus-forecast vs. observation-minus-analysis RMSE, for
    one batch. ``obs_type`` is per-observation (e.g. ``"RO"``/``"IGS"``),
    same length as ``y_obs``; omit for an overall-only result."""

    rmse_forecast: float
    rmse_analysis: float
    n_obs: int
    by_obs_type: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def reduction(self) -> float:
        """Positive means assimilation helped (analysis RMSE < forecast RMSE)."""
        return self.rmse_forecast - self.rmse_analysis

    @property
    def fractional_reduction(self) -> float:
        if self.rmse_forecast <= 0:
            return 0.0
        return self.reduction / self.rmse_forecast


def compute_rmse_reduction(
    y_obs: np.ndarray,
    y_forecast_mean: np.ndarray,
    y_analysis_mean: np.ndarray,
    obs_type: np.ndarray | None = None,
) -> RmseReduction:
    """``y_forecast_mean``/``y_analysis_mean``: ``H @ ensemble_mean`` for
    the forecast/analysis ensembles, same shape as ``y_obs``."""
    y_obs = np.asarray(y_obs)
    y_forecast_mean = np.asarray(y_forecast_mean)
    y_analysis_mean = np.asarray(y_analysis_mean)

    by_type: dict[str, dict[str, float]] = {}
    if obs_type is not None:
        obs_type = np.asarray(obs_type)
        for t in np.unique(obs_type):
            mask = obs_type == t
            by_type[str(t)] = {
                "rmse_forecast": rmse(y_obs[mask] - y_forecast_mean[mask]),
                "rmse_analysis": rmse(y_obs[mask] - y_analysis_mean[mask]),
                "n_obs": int(mask.sum()),
            }

    return RmseReduction(
        rmse_forecast=rmse(y_obs - y_forecast_mean),
        rmse_analysis=rmse(y_obs - y_analysis_mean),
        n_obs=int(y_obs.shape[0]),
        by_obs_type=by_type,
    )


def rank_histogram_counts(
    y_obs: np.ndarray,
    Y_ensemble: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Verification rank of each observation within its forward-modeled
    ensemble spread (a Talagrand-diagram-style calibration check): for
    each observation ``i``, where ``y_obs[i]`` falls among the sorted
    ``Y_ensemble[i, :]`` values. Ties are broken by adding a vanishingly
    small random jitter (the standard rank-histogram convention), so an
    exact tie doesn't silently bias toward the lower rank.

    Parameters
    ----------
    y_obs : ndarray, shape (n_obs,)
    Y_ensemble : ndarray, shape (n_obs, n_members)
        Forward-modeled ensemble (e.g. the forecast ensemble, the
        conventional choice for a calibration check -- passing the
        analysis ensemble instead is valid too, just a different question).
    rng : numpy.random.Generator, optional
        For tie-breaking jitter.

    Returns
    -------
    counts : ndarray, shape (n_members + 1,)
        Histogram of ranks in ``[0, n_members]`` (rank ``k`` means ``k`` of
        the ``n_members`` ensemble values were below ``y_obs[i]``). A
        near-uniform histogram over many observations/batches indicates a
        well-calibrated ensemble; U-shaped means under-dispersive (too
        many observations fall outside the spread); peaked means
        over-dispersive.
    """
    if rng is None:
        rng = np.random.default_rng()
    y_obs = np.asarray(y_obs)
    Y_ensemble = np.asarray(Y_ensemble)
    n_obs, n_members = Y_ensemble.shape

    jitter = rng.normal(scale=1e-9 * (1.0 + np.abs(Y_ensemble)), size=Y_ensemble.shape)
    ranks = np.sum((Y_ensemble + jitter) < y_obs[:, np.newaxis], axis=1)

    counts = np.bincount(ranks, minlength=n_members + 1)
    return counts[: n_members + 1]


def effective_rank(X: np.ndarray, tol: float | None = None) -> dict:
    """
    Numerical rank and singular-value spectrum of an ensemble's anomaly
    matrix ``X - mean(X, axis=1)`` (plan Section 4.13's "effective
    ensemble rank" -- monitors rank collapse/degeneracy over batches,
    ties to the N-vs-n rank-deficiency discussion in
    ``General_EnKF_Implementation_Plan.md`` Section 6.6).

    Parameters
    ----------
    X : ndarray, shape (n_state, n_members)
    tol : float, optional
        Singular values below ``tol`` don't count toward the numerical
        rank; defaults to numpy's own ``matrix_rank`` convention (scaled
        by the largest singular value and matrix dimensions).

    Returns
    -------
    dict with keys ``rank`` (int), ``singular_values`` (ndarray, length
    ``min(n_state, n_members) - 1`` since the mean is removed first, so
    the anomaly matrix has at most ``n_members - 1`` independent
    directions), ``n_members``.
    """
    X = np.asarray(X)
    n_members = X.shape[1]
    anomalies = X - X.mean(axis=1, keepdims=True)
    singular_values = np.linalg.svd(anomalies, compute_uv=False)
    rank = int(np.linalg.matrix_rank(anomalies, tol=tol))
    return {
        "rank": rank,
        "singular_values": singular_values,
        "n_members": n_members,
    }
