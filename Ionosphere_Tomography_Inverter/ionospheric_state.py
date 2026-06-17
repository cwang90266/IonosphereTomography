#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IonosphericState — parametric state vector and ensemble management for the
ANCHOR-inspired parametric EnKF.

State vector at each horizontal grid point (8 parameters):

  x = [log10(NmF2),  hmF2,  H0,  gamma,  B0,  B1,  log10(NmE),  hmE]^T

Density parameters (indices 0 and 6) are stored in log10 space to guarantee
positive electron densities after any linear perturbation.

Units
-----
  NmF2, NmE  : m^-3  (stored as log10)
  hmF2, hmE  : km
  H0, gamma  : km
  B0         : km
  B1         : dimensionless
"""

from __future__ import annotations

import numpy as np

# ── Index constants ──────────────────────────────────────────────────────────
I_LOG_NMF2 = 0
I_HMF2     = 1
I_H0       = 2
I_GAMMA    = 3
I_B0       = 4
I_B1       = 5
I_LOG_NME  = 6
I_HME      = 7
N_STATE    = 8

# Indices of the two log-space parameters
LOG_INDICES   = np.array([I_LOG_NMF2, I_LOG_NME], dtype=int)
LINEAR_INDICES = np.setdiff1d(np.arange(N_STATE), LOG_INDICES)

PARAM_NAMES = [
    "log10(NmF2)", "hmF2", "H0", "gamma", "B0", "B1", "log10(NmE)", "hmE"
]


class IonosphericState:
    """
    Manages the parametric state vector ensemble for a 2-D horizontal grid.

    Parameters
    ----------
    n_grid_points : int
        Number of horizontal grid points (lat/lon nodes).
    n_members : int
        Default ensemble size; can be overridden in generate_ensemble.

    Attributes
    ----------
    ensemble : ndarray, shape (N_STATE, n_grid_points, n_members)
        The forecast ensemble.  Density parameters in log10 space.
    """

    def __init__(self, n_grid_points: int, n_members: int = 50) -> None:
        self.n_grid_points = n_grid_points
        self.n_members     = n_members
        self.ensemble: np.ndarray | None = None

    # ── Ensemble generation ───────────────────────────────────────────────────

    def generate_ensemble(
        self,
        mean_state: np.ndarray,
        covariance_matrix: np.ndarray,
        n_members: int | None = None,
    ) -> np.ndarray:
        """
        Draw the forecast ensemble from the empirical background distribution.

        Parameters
        ----------
        mean_state : ndarray, shape (N_STATE, n_grid_points)
            Background mean state.  Density entries must already be in log10.
        covariance_matrix : ndarray, shape (N_STATE, N_STATE)
            Background error covariance (same for every grid point; extend to
            a block-diagonal form externally if spatial correlations matter).
        n_members : int, optional
            Override the instance default ensemble size.

        Returns
        -------
        X_f : ndarray, shape (N_STATE, n_grid_points, n_members)
            Forecast ensemble matrix.  Density params remain in log10 space.
        """
        n_members = n_members or self.n_members

        if mean_state.shape != (N_STATE, self.n_grid_points):
            raise ValueError(
                f"mean_state must be ({N_STATE}, {self.n_grid_points}), "
                f"got {mean_state.shape}"
            )
        if covariance_matrix.shape != (N_STATE, N_STATE):
            raise ValueError(
                f"covariance_matrix must be ({N_STATE}, {N_STATE}), "
                f"got {covariance_matrix.shape}"
            )

        # Draw perturbations: shape (n_grid_points, n_members, N_STATE)
        # multivariate_normal broadcasts naturally over the leading batch dim.
        perturbations = np.random.multivariate_normal(
            mean=np.zeros(N_STATE),
            cov=covariance_matrix,
            size=(self.n_grid_points, n_members),
        )  # (n_grid_points, n_members, N_STATE)

        # Transpose to (N_STATE, n_grid_points, n_members) and add mean
        X_f = (
            mean_state[:, :, np.newaxis]           # (N_STATE, n_grid_points, 1)
            + perturbations.transpose(2, 0, 1)     # (N_STATE, n_grid_points, n_members)
        )

        self.ensemble = X_f
        self.n_members = n_members
        return X_f

    def generate_ensemble_spatial(
        self,
        mean_state: np.ndarray,
        param_covariance: np.ndarray,
        spatial_corr: np.ndarray,
        n_members: int | None = None,
    ) -> np.ndarray:
        """
        Draw the forecast ensemble with spatial correlations between grid points.

        Uses the Kronecker structure of the joint background covariance:

            C_joint = param_covariance ⊗ spatial_corr

        meaning that the correlation between parameter p1 at grid point g1 and
        parameter p2 at grid point g2 is:

            C_joint[(p1,g1), (p2,g2)] = param_covariance[p1, p2] × spatial_corr[g1, g2]

        Samples are drawn efficiently as:

            pert[:, :, m] = L_param @ Z[:, :, m] @ L_spatial.T

        where L_param = chol(param_covariance), L_spatial = chol(spatial_corr),
        and Z[:, :, m] ~ N(0, I) of shape (N_STATE, n_grid_points).

        Parameters
        ----------
        mean_state : ndarray, shape (N_STATE, n_grid_points)
            Background mean state (density params in log10 space).
        param_covariance : ndarray, shape (N_STATE, N_STATE)
            Per-parameter background error covariance.
        spatial_corr : ndarray, shape (n_grid_points, n_grid_points)
            Spatial correlation matrix (symmetric, PD).  Diagonal entries = 1.
            Off-diagonal entries encode how correlated adjacent grid points are.
        n_members : int, optional
            Override the instance default ensemble size.

        Returns
        -------
        X_f : ndarray, shape (N_STATE, n_grid_points, n_members)
        """
        n_members = n_members or self.n_members

        if mean_state.shape != (N_STATE, self.n_grid_points):
            raise ValueError(
                f"mean_state must be ({N_STATE}, {self.n_grid_points}), "
                f"got {mean_state.shape}"
            )
        if param_covariance.shape != (N_STATE, N_STATE):
            raise ValueError(
                f"param_covariance must be ({N_STATE}, {N_STATE})."
            )
        if spatial_corr.shape != (self.n_grid_points, self.n_grid_points):
            raise ValueError(
                f"spatial_corr must be ({self.n_grid_points}, {self.n_grid_points})."
            )

        # Cholesky factors
        try:
            L_param = np.linalg.cholesky(param_covariance)       # (N_STATE, N_STATE)
            L_space = np.linalg.cholesky(spatial_corr)           # (n_grid, n_grid)
        except np.linalg.LinAlgError:
            # Fallback: add small nugget if matrix is near-singular
            eps = 1e-8
            L_param = np.linalg.cholesky(
                param_covariance + eps * np.eye(N_STATE))
            L_space = np.linalg.cholesky(
                spatial_corr + eps * np.eye(self.n_grid_points))

        # White noise: (N_STATE, n_grid_points, n_members)
        Z = np.random.randn(N_STATE, self.n_grid_points, n_members)

        # Apply parameter covariance: tmp[p, g, m] = sum_k L_param[p,k] Z[k,g,m]
        tmp = np.einsum('pk,kgm->pgm', L_param, Z)

        # Apply spatial covariance: pert[p, g, m] = sum_l tmp[p,l,m] L_space[g,l]
        # (= tmp @ L_space.T along the grid dimension)
        pert = np.einsum('plm,gl->pgm', tmp, L_space)

        X_f = mean_state[:, :, np.newaxis] + pert
        self.ensemble = X_f
        self.n_members = n_members
        return X_f

    # ── Log ↔ linear conversions ──────────────────────────────────────────────

    def to_linear_densities(
        self, ensemble: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Return a copy of the ensemble with log10 density params converted to
        linear (m^-3) space.  All other parameters are unchanged.

        Parameters
        ----------
        ensemble : ndarray, shape (N_STATE, n_grid_points, n_members), optional
            Defaults to ``self.ensemble`` if omitted.

        Returns
        -------
        ndarray, same shape — density rows in m^-3, rest unchanged.
        """
        if ensemble is None:
            ensemble = self.ensemble
        if ensemble is None:
            raise RuntimeError("No ensemble available; call generate_ensemble first.")

        out = ensemble.copy()
        out[LOG_INDICES] = 10.0 ** ensemble[LOG_INDICES]
        return out

    def to_log_densities(self, ensemble_linear: np.ndarray) -> np.ndarray:
        """
        Inverse of to_linear_densities — clamps densities to a minimum of
        1 m^-3 before taking log10 to avoid -inf values.
        """
        out = ensemble_linear.copy()
        out[LOG_INDICES] = np.log10(np.maximum(ensemble_linear[LOG_INDICES], 1.0))
        return out

    # ── Convenience extractors ────────────────────────────────────────────────

    def get_params(
        self,
        ensemble: np.ndarray | None = None,
        grid_point: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Extract named parameters from the ensemble.

        Returns a dict mapping parameter name → ndarray.  Shape is
        ``(n_members,)`` when grid_point is given, else
        ``(n_grid_points, n_members)``.
        """
        if ensemble is None:
            ensemble = self.ensemble
        lin = self.to_linear_densities(ensemble)

        if grid_point is not None:
            lin = lin[:, grid_point, :]   # (N_STATE, n_members)
            return {name: lin[i] for i, name in enumerate(PARAM_NAMES)}

        return {name: lin[i] for i, name in enumerate(PARAM_NAMES)}

    # ── Physical bounds ───────────────────────────────────────────────────────

    # Valid ranges for each parameter (min, max).  Derived from IRI climatology
    # and the profile formulation constraints.
    PARAM_BOUNDS = np.array([
        [9.5,  13.5],   # log10(NmF2)  m^-3
        [150., 500.],   # hmF2         km
        [10.,  200.],   # H0           km  (topside scale height parameter)
        [0.01, 1.5],    # gamma        (dimensionless topside shape)
        [20.,  500.],   # B0           km  (bottomside thickness)
        [0.3,  3.0],    # B1           (dimensionless bottomside shape, must be > 0)
        [8.0,  12.0],   # log10(NmE)   m^-3
        [90.,  130.],   # hmE          km
    ])

    def clamp_to_physical_bounds(self) -> None:
        """
        Clamp every ensemble member to physically valid parameter ranges in
        place.  Also enforces the ordering constraint hmF2 > hmE + 20 km so
        the intermediate connection region is always well-defined.

        Call this immediately after any EnKF update step.
        """
        if self.ensemble is None:
            return
        for i, (lo, hi) in enumerate(self.PARAM_BOUNDS):
            self.ensemble[i] = np.clip(self.ensemble[i], lo, hi)
        # Structural constraint: F2 peak must sit at least 20 km above E peak
        min_hmF2 = self.ensemble[I_HME] + 20.0
        self.ensemble[I_HMF2] = np.maximum(self.ensemble[I_HMF2], min_hmF2)

    # ── Ensemble statistics ───────────────────────────────────────────────────

    def ensemble_mean(self, ensemble: np.ndarray | None = None) -> np.ndarray:
        """Mean over the member dimension → (N_STATE, n_grid_points)."""
        e = ensemble if ensemble is not None else self.ensemble
        return e.mean(axis=2)

    def ensemble_perturbations(
        self, ensemble: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Anomaly matrix X' = X - mean(X).

        Returns ndarray, shape (N_STATE, n_grid_points, n_members).
        """
        e = ensemble if ensemble is not None else self.ensemble
        return e - self.ensemble_mean(e)[:, :, np.newaxis]

    def sample_covariance(
        self, ensemble: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Flatten spatial and parameter dimensions and compute the full sample
        covariance.  Useful for diagnostics on small grids; for operational
        use, compute localised covariances in the EnKF update step.

        Returns ndarray, shape (N_STATE * n_grid_points, N_STATE * n_grid_points).
        """
        e = ensemble if ensemble is not None else self.ensemble
        N, G, M = e.shape
        A = e.reshape(N * G, M)           # flatten state dims
        A_prime = A - A.mean(axis=1, keepdims=True)
        return (A_prime @ A_prime.T) / (M - 1)
