#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EnKF update step for the parametric ionospheric tomography system.

Implements a Deterministic Ensemble Square-Root Filter (EnSRF) analysis:

    mean:       x_a  = x_f + K (y_obs − y_f_mean)
    anomalies:  X'_a = X'_f @ W,   WW' = I_N − H̃' D⁻¹ H̃

where H̃ = Y'_f / √(N-1),  D = P_yy + R,  and W is the symmetric matrix
square root of (I_N − H̃' D⁻¹ H̃).  Optional Gaspari-Cohn localization is
applied to the forecast-state cross-covariance before the Kalman gain is
computed (mean update only; the W transform operates in ensemble space).

References
----------
Katzfuss et al. (2016), The American Statistician — EnSRF derivation.
Gaspari & Cohn (1999), QJRMS — compact-support localisation function.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from .ionospheric_state import IonosphericState, N_STATE


# ── Gaspari-Cohn localisation ─────────────────────────────────────────────────

def gaspari_cohn(r: np.ndarray) -> np.ndarray:
    """
    Gaspari-Cohn (1999) compact-support localisation weight.

    Parameters
    ----------
    r : ndarray
        Normalised distance  d / localization_radius.  Exactly zero for r >= 2.

    Returns
    -------
    ndarray, same shape as r, values in [0, 1].
    """
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)

    m1 = r <= 1.0
    m2 = (r > 1.0) & (r < 2.0)

    r1 = r[m1]
    out[m1] = (
        1.0
        - (5.0 / 3.0) * r1 ** 2
        + (5.0 / 8.0) * r1 ** 3
        + (1.0 / 2.0) * r1 ** 4
        - (1.0 / 4.0) * r1 ** 5
    )

    r2 = r[m2]
    out[m2] = (
        4.0
        - 5.0 * r2
        + (5.0 / 3.0) * r2 ** 2
        + (5.0 / 8.0) * r2 ** 3
        - (1.0 / 2.0) * r2 ** 4
        + (1.0 / 12.0) * r2 ** 5
        - 2.0 / (3.0 * r2)
    )

    return out


def _haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Great-circle distance (km) between a single point and an array of points.
    """
    R_E = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2.0 * R_E * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_localisation_matrix(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    tangent_lats: np.ndarray,
    tangent_lons: np.ndarray,
    loc_radius_km: float,
) -> np.ndarray:
    """
    Compute the Gaspari-Cohn localisation weights between every observation
    and every grid point.

    Parameters
    ----------
    grid_lats, grid_lons : ndarray, shape (n_grid,)
        Horizontal positions of the state grid points.
    tangent_lats, tangent_lons : ndarray, shape (n_obs,)
        Tangent-point (or pierce-point) positions of the GNSS rays.
    loc_radius_km : float
        Half-support radius.  The weight is exactly zero beyond
        ``2 * loc_radius_km``.

    Returns
    -------
    L : ndarray, shape (n_obs, n_grid)
        Localisation weights in [0, 1].
    """
    n_obs  = len(tangent_lats)
    n_grid = len(grid_lats)

    # Pairwise distances: (n_obs, n_grid)
    dist_km = _haversine_km(
        tangent_lats[:, np.newaxis],   # (n_obs, 1)
        tangent_lons[:, np.newaxis],
        grid_lats[np.newaxis, :],      # (1, n_grid)
        grid_lons[np.newaxis, :],
    )

    r_norm = dist_km / loc_radius_km   # normalised distance
    return gaspari_cohn(r_norm)        # (n_obs, n_grid)


def build_ray_localisation_matrix(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    ray_trajectories: list,
    loc_radius_km: float,
) -> np.ndarray:
    """
    Gaspari-Cohn localisation weights based on the **minimum distance from
    each grid point to any sample point along the ray path**.

    This is more physically meaningful than using the tangent point alone:
    a ray that grazes a grid point at 300 km altitude should influence it even
    if the tangent point is 400 km away horizontally.

    Parameters
    ----------
    grid_lats, grid_lons : ndarray, shape (n_grid,)
    ray_trajectories : list of ndarray, each shape (n_pts, 3) — [lat, lon, alt_km]
        One trajectory per observation (the representative ray for that arc).
    loc_radius_km : float
        Gaspari-Cohn half-support radius (km).

    Returns
    -------
    L : ndarray, shape (n_obs, n_grid)  — weights in [0, 1].
    """
    n_obs  = len(ray_trajectories)
    n_grid = len(grid_lats)
    L = np.zeros((n_obs, n_grid), dtype=float)

    for i, ray in enumerate(ray_trajectories):
        ray_lats = ray[:, 0][:, np.newaxis]   # (n_pts, 1)
        ray_lons = ray[:, 1][:, np.newaxis]

        # Distances from every ray point to every grid point: (n_pts, n_grid)
        dist_km = _haversine_km(
            ray_lats, ray_lons,
            grid_lats[np.newaxis, :],
            grid_lons[np.newaxis, :],
        )

        # Use the minimum distance (closest approach of the ray to each grid point)
        min_dist = dist_km.min(axis=0)         # (n_grid,)
        r_norm   = min_dist / loc_radius_km
        L[i]     = gaspari_cohn(r_norm)

    return L


# ── EnKF update ───────────────────────────────────────────────────────────────
def enkf_update(
    X_f: np.ndarray,
    Y_f: np.ndarray,
    y_obs: np.ndarray,
    R: np.ndarray,
    localisation_weights: np.ndarray | None = None,
    inflation: float = 1.0,
    max_update_step: float | None = None, # Left in signature for compatibility, but ignored
    deterministic: bool = False,          # Defaulted to False for ES-MDA
) -> tuple[np.ndarray, dict]:
    
    n_state, n_members = X_f.shape
    n_obs = Y_f.shape[0]

    # ── 1. Drop members whose forward-model output is non-finite ─────────────
    # Non-finite TEC predictions (from e.g. negative gamma or overflow in the
    # Chapman profile) corrupt P_yy and cause la.solve to crash.  Exclude those
    # members from covariance estimation; they still receive the update computed
    # from the valid subset, which is the least-bad option without re-running
    # the forward model.
    valid = np.all(np.isfinite(Y_f), axis=0)   # (N,) boolean
    n_valid = int(valid.sum())
    if n_valid < n_members:
        print(f"    [enkf_update] WARNING: {n_members - n_valid}/{n_members} members "
              f"have non-finite TEC predictions — excluded from covariance estimation.")
    if n_valid < 2:
        raise ValueError(
            f"enkf_update: only {n_valid} finite member(s) — cannot form a covariance. "
            "Check the forward model for NaN/Inf (likely negative gamma or overflow)."
        )

    # ── 2. Ensemble anomalies (with multiplicative inflation) ─────────────────
    # Means and anomalies are computed from the valid subset only.
    X_mean = X_f[:, valid].mean(axis=1, keepdims=True)    # (n_state, 1)
    Y_mean = Y_f[:, valid].mean(axis=1, keepdims=True)    # (n_obs, 1)

    # Anomalies for ALL members (so every member gets updated), but the
    # covariance factors below use only valid members.
    X_prime = (X_f - X_mean) * inflation       # (n_state, N)
    Y_prime = Y_f - Y_mean                     # (n_obs, N)

    # Safely reconstruct the inflated prior state
    X_inflated = X_mean + X_prime

    # ── 3. Ensemble square-root factors (valid members only) ─────────────────
    sq      = np.sqrt(1.0 / (n_valid - 1))
    L_tilde = X_prime[:, valid] * sq   # (n_state, n_valid)
    H_tilde = Y_prime[:, valid] * sq   # (n_obs,   n_valid)

    # ── 4. Forecast covariances ───────────────────────────────────────────────
    P_yy     = H_tilde @ H_tilde.T    # (n_obs, n_obs)
    P_xy_raw = L_tilde @ H_tilde.T    # (n_state, n_obs)

    # ── 5. Localisation ───────────────────────────────────────────────────────
    if localisation_weights is not None:
        n_grid   = localisation_weights.shape[1]
        n_params = n_state // n_grid
        T_xy = np.tile(localisation_weights.T, (n_params, 1))   # (n_state, n_obs)
        P_xy = P_xy_raw * T_xy
    else:
        P_xy = P_xy_raw

    # ── 6. Innovation covariance and localised Kalman gain ────────────────────
    import scipy.linalg as la
    D = P_yy + R    # (n_obs, n_obs)
    try:
        K_T = la.solve(D, P_xy.T, assume_a="pos")   # (n_obs, n_state)
    except la.LinAlgError:
        K_T = la.pinv(D) @ P_xy.T
    K = K_T.T       # (n_state, n_obs)

    innov_mean = y_obs - Y_mean[:, 0]   # (n_obs,)

    if deterministic:
        # ── 6a. Mean update ───────────────────────────────────────────────────
        x_mean_a = X_mean[:, 0] + K @ innov_mean   # (n_state,)

        # ── 6b. Anomaly update — EnSRF transformation matrix W ────────────────
        try:
            D_inv_H = la.solve(D, H_tilde, assume_a="pos")   # (n_obs, N)
        except la.LinAlgError:
            D_inv_H = la.pinv(D) @ H_tilde

        A     = np.eye(n_valid) - H_tilde.T @ D_inv_H      # (n_valid, n_valid)
        evals, evecs = np.linalg.eigh(A)
        evals = np.maximum(evals, 0.0)                     # clip numerical noise
        W     = (evecs * np.sqrt(evals)) @ evecs.T         # symmetric √A: (n_valid, n_valid)

        X_prime_a_valid = X_prime[:, valid] @ W             # (n_state, n_valid)
        # Broadcast back to all N members: valid members get the EnSRF anomaly,
        # invalid members collapse to the posterior mean.
        X_a = np.full((n_state, n_members), x_mean_a[:, np.newaxis])
        X_a[:, valid] = x_mean_a[:, np.newaxis] + X_prime_a_valid

        diag_innov_mean = innov_mean
        diag_innov_std  = np.zeros(n_obs)
        diag_W_evals    = evals                            

    else:
        # ── 6c. Stochastic (perturbed-observation) EnKF ───────────────────────
        try:
            L_chol = la.cholesky(R, lower=True)
        except la.LinAlgError:
            L_chol = la.cholesky(R + 1e-8 * np.eye(n_obs), lower=True)

        # Draw synthetic noise and perturb observations
        v_t        = L_chol @ np.random.randn(n_obs, n_members)
        Y_obs_ens  = y_obs[:, np.newaxis] + v_t
        innovation = Y_obs_ens - Y_f
        
        # Calculate raw update
        delta_X = K @ innovation

        # ── 7. Smart Mean-Only Step Limiter (Trust Region) ───────────────
        if max_update_step is not None:
            # Separate the update into Mean and Anomalies
            delta_mean = delta_X.mean(axis=1, keepdims=True)
            delta_anom = delta_X - delta_mean
            
            # --- FIX: Handle both scalar and array inputs ---
            if isinstance(max_update_step, (float, int)):
                # Fallback to previous logic if a scalar is passed
                prior_std = np.maximum(X_prime.std(axis=1, keepdims=True), 1e-8)
                step_limit = max_update_step * prior_std
            else:
                # Use the array passed in directly as the limit
                # Ensure it is (n_state, 1)
                step_limit = max_update_step[:, np.newaxis]
            
            # Clip ONLY the mean shift
            delta_mean_clipped = np.clip(delta_mean, -step_limit, step_limit)
            
            # Recombine safely
            delta_X = delta_mean_clipped + delta_anom

        # Apply the update to the INFLATED prior state
        X_a = X_inflated + delta_X

        diag_innov_mean = innovation.mean(axis=1)
        diag_innov_std  = innovation.std(axis=1)
        diag_W_evals    = None

    # Step 7 (clipping) has been completely removed to prevent covariance destruction.
    
    diagnostics = {
        "innovation_mean":   diag_innov_mean,
        "innovation_std":    diag_innov_std,
        "kalman_gain":       K,
        "P_yy":              P_yy,
        "W_eigenvalues":     diag_W_evals,
    }

    return X_a, diagnostics
# ── Ensemble reshaping helpers ────────────────────────────────────────────────

def flatten_ensemble(ensemble: np.ndarray) -> np.ndarray:
    """
    (N_STATE, n_grid, n_members) → (N_STATE * n_grid, n_members).

    The flattening order is C-contiguous (state parameter varies slowest,
    grid point varies fastest within each parameter block).
    """
    N, G, M = ensemble.shape
    return ensemble.reshape(N * G, M)


def unflatten_ensemble(
    flat: np.ndarray,
    n_params: int,
    n_grid: int,
) -> np.ndarray:
    """
    (N_STATE * n_grid, n_members) → (N_STATE, n_grid, n_members).
    """
    M = flat.shape[1]
    return flat.reshape(n_params, n_grid, M)


# ── Convenience wrapper ───────────────────────────────────────────────────────

class ParametricEnKF:
    """
    Thin wrapper that manages the full assimilation cycle for the parametric
    ionospheric state.

    Parameters
    ----------
    state : IonosphericState
        The state object (holds the ensemble and grid metadata).
    grid_lats, grid_lons : ndarray, shape (n_grid,)
        Geographic positions of the horizontal grid points.
    loc_radius_km : float
        Gaspari-Cohn half-support radius (km).  Observations influence only
        grid points within ``2 * loc_radius_km`` of their tangent point.
        Set to ``np.inf`` to disable spatial localisation.
    inflation : float
        Multiplicative inflation applied to ensemble anomalies before the
        update.  Compensates for sampling error and model error
        underestimation.
    """

    def __init__(
        self,
        state: IonosphericState,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        loc_radius_km: float = 500.0,
        inflation: float = 1.0,
    ) -> None:
        self.state         = state
        self.grid_lats     = np.asarray(grid_lats)
        self.grid_lons     = np.asarray(grid_lons)
        self.loc_radius_km = loc_radius_km
        self.inflation     = inflation

    def assimilate(
        self,
        Y_f: np.ndarray,
        y_obs: np.ndarray,
        R: np.ndarray,
        tangent_lats: np.ndarray | None = None,
        tangent_lons: np.ndarray | None = None,
        localisation_matrix: np.ndarray | None = None,
        max_update_step: float | None = 0.5,
        deterministic: bool = False,
        apply_bounds: bool =True,
    ) -> tuple[np.ndarray, dict]:
        """
        Run one EnKF analysis cycle and update ``self.state.ensemble`` in place.

        Parameters
        ----------
        Y_f : ndarray, shape (n_obs, n_members)
            Simulated observation ensemble from ObservationOperator.
        y_obs : ndarray, shape (n_obs,)
            Observed sTEC values (TECU).
        R : ndarray, shape (n_obs, n_obs)
            Observation error covariance.
        tangent_lats, tangent_lons : ndarray, shape (n_obs,), optional
            Tangent-point coordinates used to build a Gaspari-Cohn localisation
            matrix automatically.  Ignored when ``localisation_matrix`` is given.
        localisation_matrix : ndarray, shape (n_obs, n_grid), optional
            Pre-computed localisation weights (e.g. from
            ``build_ray_localisation_matrix``).  When supplied, takes precedence
            over the tangent-point-based calculation.
        max_update_step : float or None
            Passed directly to ``enkf_update`` for log-space safety clipping.

        Returns
        -------
        analysis_mean : ndarray, shape (N_STATE, n_grid)
            Posterior ensemble mean in the same (mixed log/linear) space as
            the state vector.
        diagnostics : dict
            From ``enkf_update``.
        """
        ensemble = self.state.ensemble
        if ensemble is None:
            raise RuntimeError("Call generate_ensemble before assimilate.")

        n_grid = self.state.n_grid_points

        # Localisation weights — pre-built matrix takes priority
        if localisation_matrix is not None:
            L = localisation_matrix
        elif n_grid > 1 and tangent_lats is not None and np.isfinite(self.loc_radius_km):
            L = build_localisation_matrix(
                self.grid_lats, self.grid_lons,
                tangent_lats, tangent_lons,
                self.loc_radius_km,
            )
        else:
            L = None

        # Flatten, update, unflatten
        X_flat = flatten_ensemble(ensemble)                 # (N_STATE*n_grid, M)
        X_a_flat, diag = enkf_update(
            X_flat, Y_f, y_obs, R,
            localisation_weights=L,
            inflation=self.inflation,
            max_update_step=max_update_step,
            deterministic=deterministic,
        )
        self.state.ensemble = unflatten_ensemble(X_a_flat, N_STATE, n_grid)

        # Clamp every member to physically valid parameter ranges so non-linear
        # profile evaluation never encounters degenerate inputs (e.g. B1 ≤ 0,
        # hmF2 < hmE, negative altitudes).
        if apply_bounds:
            self.state.clamp_to_physical_bounds()

        return self.state.ensemble_mean(), diag
