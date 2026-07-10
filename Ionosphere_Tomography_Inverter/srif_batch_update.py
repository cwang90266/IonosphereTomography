#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
srif_batch_update.py
====================
Square Root Information Filter (SRIF) batch update for ionospheric tomography.

Background
----------
The standard Kalman / information-form batch update requires either:

  • The full observation matrix  H  (n_obs × n_state) — infeasible when n_obs
    exceeds ~10 000 and n_state is a few thousand (H alone can exceed 1 GB).
  • The innovation covariance  S = H P H^T + R  (n_obs × n_obs) — even worse.

The SRIF reformulation avoids both by working entirely with upper-triangular
square-root factors of the information matrix (n_state × n_state), processing
observations in streaming batches so that the full H is never assembled.

Mathematical summary
--------------------
State prior:   x_f ∈ R^n,  P_f = L_f L_f^T   (lower Cholesky)

Information square root (upper triangular R̄_f, required invariant
R̄_f^T R̄_f = P_f^{-1} — note this is *not* the same as (L_f^{-1})^T, which
instead satisfies (L_f^{-1})^T (L_f^{-1}) ... = P_f^{-1} under the opposite
ordering; the two coincide only when P_f is diagonal):

    L_f^{-1} = Q R   (QR decomposition)  →  R̄_f := R,
    since  L_f^{-T} L_f^{-1} = R^T Q^T Q R = R^T R = P_f^{-1}

For the separable (Kronecker) ionospheric prior
``P_f = diag(sigma_abs) @ kron(C_v, C_s) @ diag(sigma_abs)`` this reduces to
a closed form that avoids ever factorising the full n×n P_f — see
``SRIFBatchUpdate.from_kron_prior``.

Information vector:
    z_f = R̄_f x_f   →   R̄_f^T z_f = P_f^{-1} x_f  ≡ N_f (prior normal eqs)

Observation model:  y = H x + ε,  ε ~ N(0, σ² I),  W^{1/2} = I/σ

SRIF batch update (one chunk of m observations at a time):

    Augmented system (n + m rows, n + 1 columns)::

        A = ┌ R̄    │ z   ┐   ← n rows  (current information state)
            └ H/σ  │ y/σ ┘   ← m rows  (scaled observations)

    Householder QR:  A = Q [ R̄_new │ z_new ]  +  [0 │ ε_chunk]
                              ──────────────────
                              n × (n+1) upper block

    After processing all batches:  (R̄_a, z_a) encodes the posterior.

Posterior retrieval:
    x_a      = R̄_a^{-1} z_a              (back-substitution)
    diag(P_a)= Σ_j  (R̄_a^{-1}[i,j])^2  (row-norms of the triangular inverse)

The triangular inverse is computed column-by-column via solve_triangular so
the full n×n P_a is never materialised.

Usage
-----
::

    srif = SRIFBatchUpdate(x_prior, P_prior, obs_sigma=5.0)
    for H_chunk, y_chunk in arc_generator:
        srif.update(H_chunk, y_chunk)
    x_post, diag_P_post = srif.solve()

Reference
---------
Bierman, G. J. (1977). *Factorization Methods for Discrete Sequential
Estimation*. Academic Press.  Chapter VI.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

__all__ = ["SRIFBatchUpdate"]


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _chol_regularised(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Attempt Cholesky of M; if it fails, add eps * diag(max(diag(M), 1.0))
    and retry.  Raises LinAlgError if the regularised matrix still fails.
    """
    try:
        return la.cholesky(M, lower=True, check_finite=False)
    except la.LinAlgError:
        d   = np.maximum(np.diag(M), 1.0)
        reg = M + eps * np.diag(d)
        return la.cholesky(reg, lower=True, check_finite=False)


def _upper_chol_of_inverse(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Return upper-triangular  U  such that  U^T U = inv(M),  without ever
    forming ``inv(M)`` explicitly.

    ``M = L L^T`` (lower Cholesky).  ``inv(M) = L^{-T} L^{-1}``.  Taking the
    QR decomposition of the lower-triangular ``L^{-1} = Q R`` gives
    ``L^{-T} L^{-1} = R^T Q^T Q R = R^T R``, so ``U = R``.

    Note: this is algebraically distinct from simply transposing
    ``L^{-1}`` — ``(L^{-1})^T`` satisfies ``(L^{-1})(L^{-1})^T = inv(M)``
    (the *other* matrix ordering), not ``U^T U = inv(M)``, and the two only
    coincide when ``M`` is diagonal.
    """
    n       = M.shape[0]
    L       = _chol_regularised(M, eps=eps)
    L_inv   = la.solve_triangular(L, np.eye(n), lower=True, check_finite=False)
    _, U    = np.linalg.qr(L_inv, mode='reduced')
    return np.ascontiguousarray(U)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class SRIFBatchUpdate:
    """
    Square Root Information Filter — streaming batch update.

    The filter maintains an upper-triangular information square-root  R̄
    (shape n × n) and a paired information-state vector  z  (shape n,) such
    that, at any point in time:

        x_estimate  =  R̄^{-1} z
        P_estimate  = (R̄^T R̄)^{-1}

    Observations are ingested in arbitrary-sized batches via :meth:`update`.
    No large intermediate matrices (full H, full P, full innovation covariance)
    are ever formed.

    Parameters
    ----------
    x_prior : ndarray, shape (n,)
        Background / prior state vector  x_f.
    P_prior : ndarray, shape (n, n)
        Prior covariance  P_f.  Must be symmetric positive-definite.
        A small diagonal regularisation is applied automatically if Cholesky
        fails on the first attempt.
    obs_sigma : float
        Observation noise standard deviation (same units as y and H·x).
        Assumed isotropic: R = obs_sigma² I.
    chunk_size : int
        Maximum number of observation rows processed in a single QR call.
        Smaller values use less working RAM; larger values are faster.
        Default 512.
    reg_eps : float
        Relative regularisation strength applied to P_prior if it is
        near-singular.  Default 1e-6.
    """

    def __init__(
        self,
        x_prior:   np.ndarray,
        P_prior:   np.ndarray,
        obs_sigma: float = 5.0,
        chunk_size: int  = 512,
        reg_eps:   float = 1e-6,
    ) -> None:
        x_prior = np.asarray(x_prior, dtype=float)
        P_prior = np.asarray(P_prior, dtype=float)

        n = x_prior.shape[0]
        if P_prior.shape != (n, n):
            raise ValueError(
                f"P_prior must be ({n},{n}), got {P_prior.shape}")

        self._n          = n
        self._sigma_obs  = float(obs_sigma)
        self._chunk_size = int(chunk_size)

        # ── Build prior information square root ───────────────────────────────
        # Regularise: add eps × max(diag(P), 1) to each diagonal entry so that
        # state elements clamped to the Ne floor (σ ≈ 0) don't cause singularity.
        d       = np.maximum(np.diag(P_prior), 1.0)
        P_reg   = P_prior + reg_eps * np.diag(d)

        self._R: np.ndarray = _upper_chol_of_inverse(P_reg, eps=reg_eps)  # (n, n)

        # Information vector:  z_f = R̄_f x_f
        # Satisfies: R̄_f^T z_f = P_f^{-1} x_f  (the prior normal equations)
        self._z: np.ndarray = self._R @ x_prior                  # (n,)

        # Accumulated diagnostics
        self._n_obs_total:  int   = 0
        self._n_arcs:       int   = 0   # arcs that contributed ≥1 finite row
        self._sum_sq_resid: float = 0.0

    # ── Alternate constructor: Kronecker-structured prior ─────────────────────

    @classmethod
    def from_kron_prior(
        cls,
        x_prior:    np.ndarray,
        sigma_abs:  np.ndarray,
        C_v:        np.ndarray,
        C_s:        np.ndarray,
        obs_sigma:  float = 5.0,
        chunk_size: int   = 512,
        reg_eps:    float = 1e-6,
    ) -> "SRIFBatchUpdate":
        """
        Fast constructor for priors of the separable form::

            P_f = diag(sigma_abs) @ kron(C_v, C_s) @ diag(sigma_abs)

        (vertical correlation ``C_v`` (n_alt × n_alt) ⊗ horizontal correlation
        ``C_s`` (n_grid × n_grid)), as used for the ionospheric grid prior.

        Building the full ``P_f`` and Cholesky-factorising/inverting it is
        O(n_state³) with n_state = n_alt·n_grid — infeasible for n_state in
        the tens of thousands (this is what previously caused multi-GB
        RAM/swap thrashing during initialisation). This constructor instead
        factorises only the small ``C_v``/``C_s`` blocks (O(n_alt³ + n_grid³))
        and assembles the information square root in closed form::

            U_v, U_s  upper triangular, U_v^T U_v = C_v^{-1}, U_s^T U_s = C_s^{-1}
            R̄_f = kron(U_v, U_s) @ diag(1 / sigma_abs)

        ``kron(U_v, U_s)`` is upper triangular (Kronecker product of two
        upper-triangular factors), and one can verify
        ``R̄_f^T R̄_f = diag(1/sigma_abs) @ kron(C_v^{-1}, C_s^{-1}) @ diag(1/sigma_abs)
        = P_f^{-1}`` — the same invariant required by the default constructor.

        The only O(n_state²) cost is materialising the ``kron(U_v, U_s)``
        product itself (same size as ``R̄_f``); no O(n_state³) operation is
        performed on the full state dimension.

        Parameters
        ----------
        x_prior : ndarray, shape (n_state,)
        sigma_abs : ndarray, shape (n_state,)
            Absolute (non-normalised) marginal std. dev. of each state element.
        C_v : ndarray, shape (n_alt, n_alt)
            Vertical correlation matrix.
        C_s : ndarray, shape (n_grid, n_grid)
            Horizontal correlation matrix.
        obs_sigma, chunk_size, reg_eps : see ``__init__``.
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

        U_v = _upper_chol_of_inverse(C_v, eps=reg_eps)
        U_s = _upper_chol_of_inverse(C_s, eps=reg_eps)

        R = np.kron(U_v, U_s)
        R *= (1.0 / sigma_abs)[None, :]

        self = cls.__new__(cls)
        self._n          = n
        self._sigma_obs  = float(obs_sigma)
        self._chunk_size = int(chunk_size)
        self._R: np.ndarray = np.ascontiguousarray(R)
        self._z: np.ndarray = self._R @ x_prior

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
        """
        Number of arcs (``update()`` calls) that contributed at least one
        finite observation row to the SRIF.  Calls where every row was NaN
        or where H was all-zero are not counted.
        """
        return self._n_arcs

    @property
    def weighted_rss(self) -> float:
        """
        Accumulated weighted residual sum-of-squares from all QR updates.
        Equals  Σ_k ε_k²  where ε_k = R[n, n] from the k-th Householder step.
        """
        return self._sum_sq_resid

    # ── Core update ───────────────────────────────────────────────────────────

    def update(self, H_rows: np.ndarray, y_rows: np.ndarray) -> int:
        """
        Assimilate a batch of  m  observations into the SRIF.

        Internally the batch is split into chunks of at most
        ``chunk_size`` rows to bound working memory.

        Parameters
        ----------
        H_rows : ndarray, shape (m, n)
            Observation operator rows for this batch.
        y_rows : ndarray, shape (m,)
            Observed values (same units as ``obs_sigma``).

        Returns
        -------
        int
            Number of observation rows successfully assimilated.
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

        # Process in chunks to bound working RAM.
        # Each QR call handles (n + chunk_size) × (n + 1) ≈ 40 MB for n=2254,
        # chunk_size=512 — far smaller than the full H matrix.
        assimilated = 0
        cs = self._chunk_size
        for start in range(0, m_tot, cs):
            end       = min(start + cs, m_tot)
            H_chunk   = H_rows[start:end]     # (m_c, n)
            y_chunk   = y_rows[start:end]      # (m_c,)
            fin_mask  = np.isfinite(y_chunk) & np.all(np.isfinite(H_chunk), axis=1)
            if fin_mask.sum() == 0:
                continue
            assimilated += self._householder_update(
                H_chunk[fin_mask], y_chunk[fin_mask])

        self._n_obs_total += assimilated
        if assimilated > 0:
            self._n_arcs += 1
        return assimilated

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve(
        self, return_full_cov: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve the posterior state estimate and posterior variance.

        Uses back-substitution only — no matrix inversion is performed for
        the state estimate itself.

        Parameters
        ----------
        return_full_cov : bool
            If True, also return the full dense posterior covariance
            ``P_post = R̄_a^{-1} R̄_a^{-T}`` (n × n).  The triangular inverse
            is already computed as an intermediate for the diagonal, so this
            costs no extra RAM beyond the one (n × n) buffer — comparable to
            holding a single dense state-covariance matrix, as callers that
            propagate a full P across assimilation cycles already do.

        Returns
        -------
        x_post : ndarray, shape (n,)
            Posterior state vector  x_a = R̄_a^{-1} z_a.
        diag_P_post : ndarray, shape (n,)
            Diagonal of the posterior covariance  P_a = (R̄_a^T R̄_a)^{-1}.
            Computed as the row-wise squared-norm of  R̄_a^{-1}.
        P_post : ndarray, shape (n, n) — only when ``return_full_cov=True``.
            Full posterior covariance matrix.
        """
        n = self._n
        R = self._R
        z = self._z

        # ── Posterior state via back-substitution ─────────────────────────────
        x_post = la.solve_triangular(R, z, lower=False, check_finite=False)

        # ── R̄_a^{-1}, needed for the diagonal (and optionally the full) P_a ──
        # R̄_a^{-1} is upper triangular (inverse of upper triangular is upper tri).
        # We solve all n right-hand sides at once via a single triangular solve
        # with the identity matrix.  Result is (n × n).  RAM: n² × 8 bytes = 40 MB
        # for n = 2254 — acceptable.  For very large n (> 10 000) consider the
        # column-by-column approach to avoid large temporaries.
        try:
            R_inv = la.solve_triangular(R, np.eye(n),
                                        lower=False, check_finite=False)
            if return_full_cov:
                P_post      = R_inv @ R_inv.T
                diag_P_post = np.diag(P_post)
            else:
                diag_P_post = np.sum(R_inv ** 2, axis=1)
        except la.LinAlgError:
            # Fallback: build R_inv column-by-column via back-substitution of
            # each basis vector (avoids the single large np.eye(n) solve).
            R_inv = np.empty((n, n))
            e     = np.zeros(n)
            for i in range(n - 1, -1, -1):
                e[i]       = 1.0
                R_inv[:, i] = la.solve_triangular(R, e, lower=False, check_finite=False)
                e[i]       = 0.0
            if return_full_cov:
                P_post      = R_inv @ R_inv.T
                diag_P_post = np.diag(P_post)
            else:
                diag_P_post = np.sum(R_inv ** 2, axis=1)

        if return_full_cov:
            return x_post, diag_P_post, P_post
        return x_post, diag_P_post

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def innovation_stats(self, x_prior: np.ndarray | None = None) -> dict:
        """
        Return a dict of summary statistics about the current information state.

        Parameters
        ----------
        x_prior : ndarray or None
            If provided, computes prior innovation (z − R̄ x_prior) norms.

        Returns
        -------
        dict with keys: n_obs, weighted_rss, R_diag_min, R_diag_max,
        condition_number_estimate, [prior_innov_norm if x_prior given].
        """
        d = np.diag(self._R)
        info = {
            "n_obs":                 self._n_obs_total,
            "n_arcs":                self._n_arcs,
            "weighted_rss":          self._sum_sq_resid,
            "R_diag_min":            float(np.min(np.abs(d))),
            "R_diag_max":            float(np.max(np.abs(d))),
            "condition_number_estimate": float(
                np.max(np.abs(d)) / max(np.min(np.abs(d)), 1e-30)
            ),
        }
        if x_prior is not None:
            innov = self._z - self._R @ np.asarray(x_prior, dtype=float)
            info["prior_innov_norm"] = float(np.linalg.norm(innov))
        return info

    # ── Internal Householder step ─────────────────────────────────────────────

    def _householder_update(
        self, H_c: np.ndarray, y_c: np.ndarray
    ) -> int:
        """
        Single Householder QR step for a chunk of  m_c  observations.

        Builds the  (n + m_c) × (n + 1)  augmented matrix::

            A = ┌ R̄  │ z  ┐
                └ H/σ │ y/σ┘

        applies scipy's Householder QR, and overwrites  self._R  and  self._z
        with the first  n  rows of the result.

        The residual contribution  |R_aug[n, n]|²  is accumulated into
        ``self._sum_sq_resid``.

        Parameters
        ----------
        H_c : ndarray, shape (m_c, n) — already filtered to finite rows.
        y_c : ndarray, shape (m_c,)

        Returns
        -------
        int — number of rows processed (m_c).
        """
        n   = self._n
        σ   = self._sigma_obs
        m_c = H_c.shape[0]

        # Scale observations by 1/σ (white-noise normalisation)
        Hs  = H_c / σ            # (m_c, n)
        ys  = y_c / σ            # (m_c,)

        # Augmented matrix: (n + m_c) × (n + 1)
        # Top block: [R̄ | z]   (already upper triangular — QR exploits this)
        # Bottom block: [Hs | ys]
        top    = np.column_stack([self._R, self._z])          # (n,   n+1)
        bottom = np.column_stack([Hs,      ys])               # (m_c, n+1)
        A_aug  = np.vstack([top, bottom])                     # (n+m_c, n+1)

        # Householder QR:  A_aug = Q @ R_aug
        # mode='economic': Q is (n+m_c)×(n+1), R_aug is (n+1)×(n+1).
        # Only R_aug is needed; Q is discarded.
        _, R_aug = la.qr(A_aug, mode='economic', check_finite=False)

        # Extract updated information square root and vector
        self._R = np.ascontiguousarray(R_aug[:n, :n])   # (n, n)
        self._z = R_aug[:n, n]                           # (n,)

        # Accumulate squared residual from this chunk
        # R_aug[n, n] is the (n+1, n+1) entry — the aggregated residual norm
        self._sum_sq_resid += float(R_aug[n, n]) ** 2

        return m_c
