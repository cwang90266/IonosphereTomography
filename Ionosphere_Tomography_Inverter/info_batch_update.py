#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
info_batch_update.py
=====================
Plain information-form (normal-equations) batch update for ionospheric
tomography — the streaming version of:

    Λ  = P_0^{-1}
    N  = P_0^{-1} x̄_0
    for each observation batch (H_i, y_i, R_i):
        Λ += H_i^T R_i^{-1} H_i
        N += H_i^T R_i^{-1} y_i
    x_post = Λ^{-1} N
    P_post = Λ^{-1}

This never assembles the full stacked observation matrix (n_obs × n_state)
or the innovation covariance (n_obs × n_obs) — only the fixed-size
(n_state × n_state) information matrix Λ and (n_state,) information vector N
are held in memory, updated via ordinary BLAS matrix products (H^T H, a
symmetric rank-k update) rather than a Householder QR re-triangularisation.

This is mathematically equivalent to :class:`SRIFBatchUpdate
<Ionosphere_Tomography_Inverter.srif_batch_update.SRIFBatchUpdate>` (which
maintains a square-root factor of Λ instead of Λ itself) but is
computationally cheaper per observation batch — no QR decomposition is
performed until the single final :meth:`solve` call — at the cost of
squaring the condition number of H relative to the square-root form
(standard information-filter tradeoff; acceptable here since observation
noise is modest and the state is regularised).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

__all__ = ["InfoBatchUpdate"]


class InfoBatchUpdate:
    """
    Streaming information-form (normal-equations) batch update.

    Maintains the information matrix ``Λ`` (n × n) and information vector
    ``N`` (n,) such that, at any point:

        x_estimate = Λ^{-1} N
        P_estimate = Λ^{-1}

    Parameters
    ----------
    x_prior : ndarray, shape (n,)
    P_prior : ndarray, shape (n, n)
        Prior covariance.  Must be symmetric positive-definite.
    obs_sigma : float
        Observation noise standard deviation.  Assumed isotropic:
        R = obs_sigma² I.
    reg_eps : float
        Relative regularisation strength applied to P_prior if it is
        near-singular.  Default 1e-6.
    """

    def __init__(
        self,
        x_prior:   np.ndarray,
        P_prior:   np.ndarray,
        obs_sigma: float = 5.0,
        reg_eps:   float = 1e-6,
    ) -> None:
        x_prior = np.asarray(x_prior, dtype=float)
        P_prior = np.asarray(P_prior, dtype=float)

        n = x_prior.shape[0]
        if P_prior.shape != (n, n):
            raise ValueError(
                f"P_prior must be ({n},{n}), got {P_prior.shape}")

        self._n         = n
        self._sigma_obs = float(obs_sigma)

        d     = np.maximum(np.diag(P_prior), 1.0)
        P_reg = P_prior + reg_eps * np.diag(d)

        L         = _chol_regularised(P_reg, eps=reg_eps)
        L_inv     = la.solve_triangular(L, np.eye(n), lower=True, check_finite=False)
        self._Lam = L_inv.T @ L_inv                     # Λ = P_reg^{-1}, symmetric
        self._N   = self._Lam @ x_prior

        self._n_obs_total:  int   = 0
        self._n_arcs:       int   = 0
        self._sum_sq_resid: float = 0.0

    # ── Alternate constructor: Kronecker-structured prior ─────────────────────

    @classmethod
    def from_kron_prior(
        cls,
        x_prior:   np.ndarray,
        sigma_abs: np.ndarray,
        C_v:       np.ndarray,
        C_s:       np.ndarray,
        obs_sigma: float = 5.0,
        reg_eps:   float = 1e-6,
    ) -> "InfoBatchUpdate":
        """
        Fast constructor for priors of the separable form::

            P_f = diag(sigma_abs) @ kron(C_v, C_s) @ diag(sigma_abs)

        Builds the information matrix directly::

            Λ = diag(1/sigma_abs) @ kron(C_v^{-1}, C_s^{-1}) @ diag(1/sigma_abs)

        factorising/inverting only the small ``C_v`` (n_alt × n_alt) and
        ``C_s`` (n_grid × n_grid) blocks — no O(n_state³) operation on the
        full state dimension is performed.  The only O(n_state²) cost is
        materialising the ``kron(C_v^{-1}, C_s^{-1})`` product itself (same
        size as ``Λ``).
        """
        x_prior   = np.asarray(x_prior, dtype=float)
        sigma_abs = np.asarray(sigma_abs, dtype=float)
        C_v       = np.asarray(C_v, dtype=float)
        C_s       = np.asarray(C_s, dtype=float)

        n_v = C_v.shape[0]
        n_s = C_s.shape[0]
        n   = n_v * n_s
        if x_prior.shape[0] != n or sigma_abs.shape[0] != n:
            raise ValueError(
                f"x_prior/sigma_abs must have length n_alt*n_grid = {n}; "
                f"got {x_prior.shape[0]} / {sigma_abs.shape[0]}")

        C_v_inv = _inv_regularised(C_v, eps=reg_eps)
        C_s_inv = _inv_regularised(C_s, eps=reg_eps)

        Lam  = np.kron(C_v_inv, C_s_inv)
        inv_sigma = 1.0 / sigma_abs
        Lam *= inv_sigma[None, :]
        Lam *= inv_sigma[:, None]

        self = cls.__new__(cls)
        self._n         = n
        self._sigma_obs = float(obs_sigma)
        self._Lam       = np.ascontiguousarray(Lam)
        self._N         = self._Lam @ x_prior

        self._n_obs_total:  int   = 0
        self._n_arcs:       int   = 0
        self._sum_sq_resid: float = 0.0
        return self

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def n_state(self) -> int:
        """Dimension of the state vector."""
        return self._n

    @property
    def n_obs(self) -> int:
        """Total number of observation rows assimilated so far."""
        return self._n_obs_total

    @property
    def n_arcs(self) -> int:
        """Number of ``update()`` calls that contributed ≥1 finite row."""
        return self._n_arcs

    @property
    def weighted_rss(self) -> float:
        """Accumulated weighted residual sum-of-squares Σ_i (y_i - H_i x̂)² / σ²."""
        return self._sum_sq_resid

    # ── Core update ───────────────────────────────────────────────────────────

    def update(self, H_rows: np.ndarray, y_rows: np.ndarray) -> int:
        """
        Assimilate a batch of m observations::

            Λ += H^T H / σ²
            N += H^T y / σ²

        Parameters
        ----------
        H_rows : ndarray, shape (m, n)
        y_rows : ndarray, shape (m,)

        Returns
        -------
        int — number of observation rows successfully assimilated.
        """
        H_rows = np.asarray(H_rows, dtype=float)
        y_rows = np.asarray(y_rows, dtype=float)
        m_tot  = H_rows.shape[0]

        if m_tot == 0:
            return 0
        if H_rows.shape[1] != self._n:
            raise ValueError(
                f"H_rows has {H_rows.shape[1]} columns; expected {self._n}")
        if y_rows.shape[0] != m_tot:
            raise ValueError(
                f"H_rows has {m_tot} rows but y_rows has {y_rows.shape[0]}")

        fin_mask = np.isfinite(y_rows) & np.all(np.isfinite(H_rows), axis=1)
        n_fin    = int(fin_mask.sum())
        if n_fin == 0:
            return 0

        H_fin = H_rows[fin_mask]
        y_fin = y_rows[fin_mask]
        inv_var = 1.0 / (self._sigma_obs ** 2)

        # Symmetric rank-k update via BLAS GEMM (H^T H); avoids Householder QR.
        self._Lam += inv_var * (H_fin.T @ H_fin)
        self._N   += inv_var * (H_fin.T @ y_fin)

        # Diagnostic only (not used in the solve): raw weighted y^T y of this
        # batch's residual input. Unlike SRIF's exact QR residual norm, this
        # does not net out the evolving state estimate — a monitoring proxy.
        self._sum_sq_resid += float(inv_var * (y_fin @ y_fin))

        self._n_obs_total += n_fin
        self._n_arcs       += 1
        return n_fin

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve(
        self, return_full_cov: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve the posterior state estimate and posterior variance.

        Parameters
        ----------
        return_full_cov : bool
            If True, also return the full dense posterior covariance
            ``P_post = Λ^{-1}`` (n × n).

        Returns
        -------
        x_post : ndarray, shape (n,)
        diag_P_post : ndarray, shape (n,)
        P_post : ndarray, shape (n, n) — only when ``return_full_cov=True``.
        """
        n   = self._n
        Lam = self._Lam
        N   = self._N

        c_and_lower = la.cho_factor(Lam, lower=True, check_finite=False)
        x_post = la.cho_solve(c_and_lower, N, check_finite=False)

        P_post = la.cho_solve(c_and_lower, np.eye(n), check_finite=False)
        diag_P_post = np.diag(P_post)

        if return_full_cov:
            return x_post, diag_P_post, P_post
        return x_post, diag_P_post

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def innovation_stats(self) -> dict:
        """Return a dict of summary statistics about the current information state."""
        d = np.diag(self._Lam)
        return {
            "n_obs":                     self._n_obs_total,
            "n_arcs":                    self._n_arcs,
            "weighted_rss":              self._sum_sq_resid,
            "Lam_diag_min":              float(np.min(d)),
            "Lam_diag_max":              float(np.max(d)),
            "condition_number_estimate": float(np.max(d) / max(np.min(d), 1e-30)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chol_regularised(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Attempt Cholesky of M; regularise the diagonal on failure and retry."""
    try:
        return la.cholesky(M, lower=True, check_finite=False)
    except la.LinAlgError:
        d   = np.maximum(np.diag(M), 1.0)
        reg = M + eps * np.diag(d)
        return la.cholesky(reg, lower=True, check_finite=False)


def _inv_regularised(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Invert a small SPD matrix via Cholesky, regularising on failure."""
    n     = M.shape[0]
    L     = _chol_regularised(M, eps=eps)
    L_inv = la.solve_triangular(L, np.eye(n), lower=True, check_finite=False)
    return L_inv.T @ L_inv
