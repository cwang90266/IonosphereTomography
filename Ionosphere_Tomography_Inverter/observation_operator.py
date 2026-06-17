#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ObservationOperator — parametric forward model H(x) for sTEC observations.

Maps a parametric IonosphericState ensemble (8 params per grid point) into
simulated slant TEC (sTEC) observations by:

  1. Evaluating Ne(h) at every altitude sample along each GNSS ray path using
     the IRI-based Epstein/bottomside formulations.
  2. Integrating Ne along the ray with scipy.integrate.trapezoid.

All broadcasting is over the ensemble-member dimension — no Python loops over
members.

Profile formulations (4 regions, IRI-2016 style)
-------------------------------------------------
Topside  (h >= hmF2):

    H_top = H0 * (1 + r*γ*dh / (r*H0 + γ*dh)),   r = 100
    z     = dh / H_top
    Ne    = 4*NmF2 * exp(z) / (1 + exp(z))^2

Pure F2 bottomside (h_ST <= h < hmF2):

    x_bs  = (hmF2 - h) / B0
    Ne    = NmF2 * exp(-x_bs^B1) / cosh(x_bs)

Intermediate connection region (hmE <= h < h_ST):

    h_ST is the altitude where the pure-F2 bottomside equals NmE:
        solve  NmF2 * exp(-x^B1) / cosh(x) = NmE  for x  (bisection)
        h_ST  = hmF2 - x * B0

    HZ = (h_ST + hmE) / 2
    T  = (HZ - h_ST)^2 / (h_ST - hmE)

    For h >= HZ:  h_eff = h        (pure bottomside directly)
    For h <  HZ:  h_eff = 2·HZ − h  (reflection → h_eff(hmE) = h_ST → Ne(hmE) = NmE)

    Ne(h) = NmF2 * exp(-x_eff^B1) / cosh(x_eff),  x_eff = (hmF2 - h_eff) / B0

E-layer (h < hmE):

    ze    = (h - hmE) / H_E,   H_E = 20 km
    Ne_E  = 4*NmE * exp(ze) / (1 + exp(ze))^2

Units
-----
  Altitudes  : km
  NmF2, NmE  : m^-3  (passed in linear space — call to_linear_densities first)
  sTEC output: TECU  (10^16 m^-2);  multiply by 1e-16 after integration in m^-3·m
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from .ionospheric_state import (
    IonosphericState,
    I_LOG_NMF2, I_HMF2, I_H0, I_GAMMA, I_B0, I_B1, I_LOG_NME, I_HME,
)

# Topside shape parameter (IRI default)
_R_TOPSIDE: float = 100.0
# E-layer half-thickness (km)
_H_E_KM: float = 20.0
# 1 TECU = 1e16 m^-2
_TECU: float = 1.0e16


def _pure_bottomside(hmF2, B0, B1, NmF2, h):
    """
    Pure F2 bottomside formula (vectorised, no clipping of h relative to hmF2).

    Ne = NmF2 * exp(-x^B1) / cosh(x),   x = (hmF2 - h) / B0

    Parameters broadcast freely — caller is responsible for masking to the
    appropriate altitude region.

    Parameters
    ----------
    hmF2, B0, B1, NmF2 : broadcast-compatible arrays
    h : array, same broadcast shape as the others

    Returns
    -------
    Ne : same shape as h (after broadcasting)
    """
    x = np.maximum((hmF2 - h) / (B0 + 1e-9), 0.0)
    # Clip x away from zero before the power to avoid 0**B1 = inf when B1 <= 0
    x_pow = np.where(x > 0, x, 1e-30)
    return NmF2 * np.exp(-x_pow ** B1) / np.cosh(np.clip(x, 0.0, 700.0))


def _find_hst_bisection(NmF2, hmF2, B0, B1, NmE, n_iter: int = 60):
    """
    Find h_ST (km) per ensemble member via bisection in x-space.

    Solves:  NmF2 * exp(-x^B1) / cosh(x) = NmE   for x > 0
    then:    h_ST = hmF2 - x * B0

    Parameters
    ----------
    NmF2, hmF2, B0, B1, NmE : ndarray, shape (n_members,)
        State parameters in linear density space.
    n_iter : int
        Bisection iterations (60 gives machine-precision convergence for
        x ∈ [0, 100]).

    Returns
    -------
    h_ST : ndarray, shape (n_members,)
        Transition altitude in km.  Clipped to [hmE, hmF2] by callers.
    """
    # We bisect in x = (hmF2 - h) / B0
    # f(x) = NmF2 * exp(-x^B1) / cosh(x) - NmE
    # f(0) > 0 (equals NmF2 > NmE), f(large) → 0 − NmE < 0

    x_lo = np.zeros_like(NmF2)
    x_hi = np.full_like(NmF2, 100.0)   # x=100 → h well below E-layer

    for _ in range(n_iter):
        x_mid = 0.5 * (x_lo + x_hi)
        f_mid = NmF2 * np.exp(-(x_mid ** B1)) / np.cosh(np.clip(x_mid, 0, 700)) - NmE
        x_lo = np.where(f_mid > 0.0, x_mid, x_lo)
        x_hi = np.where(f_mid > 0.0, x_hi,  x_mid)

    x_st = 0.5 * (x_lo + x_hi)
    return hmF2 - x_st * B0


def _ne_profile_ensemble(
    alts_km: np.ndarray,
    params_lin: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Ne(h) for every altitude and every ensemble member at ONE grid
    point — full IRI 4-region formulation.

    Parameters
    ----------
    alts_km : ndarray, shape (n_alt,)
        Sample altitudes in km.
    params_lin : ndarray, shape (N_STATE, n_members)
        State parameters in LINEAR density space for this grid point.

    Returns
    -------
    Ne : ndarray, shape (n_alt, n_members)
        Electron density in m^-3.
    """
    NmF2  = params_lin[I_LOG_NMF2]   # (n_members,) — already linear
    hmF2  = params_lin[I_HMF2]
    H0    = params_lin[I_H0]
    gamma = params_lin[I_GAMMA]
    B0    = params_lin[I_B0]
    B1    = params_lin[I_B1]
    NmE   = params_lin[I_LOG_NME]
    hmE   = params_lin[I_HME]

    h = alts_km[:, np.newaxis]          # (n_alt, 1)  — broadcast over members

    # ── Region 1: Topside  (h >= hmF2) ───────────────────────────────────────
    dh = h - hmF2                       # (n_alt, n_members)
    r  = _R_TOPSIDE
    H_top = H0 * (1.0 + (r * gamma * dh) / (r * H0 + gamma * dh + 1e-9))
    z_top = dh / (H_top + 1e-9)
    exp_z = np.exp(np.clip(z_top, -80, 80))
    Ne_top = 4.0 * NmF2 * exp_z / (1.0 + exp_z) ** 2

    # ── Transition altitude h_ST (one value per member) ───────────────────────
    # Bisection is over members only; broadcast to (1, n_members) for altitude masks.
    h_ST = _find_hst_bisection(NmF2, hmF2, B0, B1, NmE)  # (n_members,)
    # Safety: h_ST must lie in [hmE, hmF2]
    h_ST = np.clip(h_ST, hmE, hmF2)    # (n_members,)

    # Midpoint of the valley (used for the Region-3 mirror mapping)
    HZ = 0.5 * (h_ST + hmE)             # (n_members,)

    # ── Region 2: Pure F2 bottomside  (h_ST <= h < hmF2) ─────────────────────
    Ne_pure_bot = _pure_bottomside(hmF2, B0, B1, NmF2, h)   # (n_alt, n_members)

    # ── Region 3: Intermediate connection  (hmE <= h < h_ST) ─────────────────
    # Mirror h across HZ so that h_eff(h_ST) = h_ST and h_eff(hmE) = h_ST.
    # Both endpoints then evaluate to Ne = NmE via the pure-bottomside formula,
    # making the profile continuous with the E-layer at hmE and with the pure
    # F2 bottomside at h_ST.
    #
    # For h >= HZ : h_eff = h         (direct — connects smoothly to Region 2)
    # For h <  HZ : h_eff = 2·HZ − h  (reflection — ensures h_eff(hmE) = h_ST)
    h_eff_inter = np.where(h >= HZ, h, 2.0 * HZ - h)             # (n_alt, n_members)
    Ne_inter    = _pure_bottomside(hmF2, B0, B1, NmF2, h_eff_inter)

    # ── Region 4: E-layer (h < hmE) — symmetric Epstein ─────────────────────
    ze    = (h - hmE) / _H_E_KM
    exp_e = np.exp(np.clip(ze, -80, 80))
    Ne_E  = 4.0 * NmE * exp_e / (1.0 + exp_e) ** 2

    # ── Composite profile ─────────────────────────────────────────────────────
    Ne = np.where(
        h >= hmF2,
        Ne_top,
        np.where(
            h >= h_ST,
            Ne_pure_bot,
            np.where(
                h >= hmE,
                Ne_inter,
                Ne_E,
            ),
        ),
    )

    return np.maximum(Ne, 0.0)   # (n_alt, n_members)


class ObservationOperator:
    """
    Forward observation operator H: x → y_sTEC.

    Parameters
    ----------
    state : IonosphericState
        The state object managing the ensemble.
    alt_grid_km : ndarray, shape (n_alt,)
        Altitude grid used for profile integration (km).
    """

    def __init__(
        self,
        state: IonosphericState,
        alt_grid_km: np.ndarray,
    ) -> None:
        self.state       = state
        self.alt_grid_km = np.asarray(alt_grid_km, dtype=float)

    # ── Public interface ──────────────────────────────────────────────────────

    def compute_stec_ensemble(
        self,
        ray_trajectories: list[np.ndarray],
        ensemble: np.ndarray | None = None,
        grid_point_indices: np.ndarray | None = None,
        grid_point_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Forward-model sTEC for every ray and every ensemble member.

        Parameters
        ----------
        ray_trajectories : list of ndarray, each shape (n_pts, 3)
            Each element is one ray described by [lat (deg), lon (deg), alt (km)]
            sample points along the path.
        ensemble : ndarray, shape (N_STATE, n_grid_points, n_members), optional
            Defaults to ``self.state.ensemble``.
        grid_point_indices : ndarray of int, shape (n_rays,), optional
            Nearest-neighbour assignment: index into n_grid_points for each ray.
            Used only when grid_point_weights is None.  For a single grid point
            (n_grid_points == 1) this is inferred automatically.
        grid_point_weights : ndarray, shape (n_rays, n_grid_points), optional
            IDW (or any non-negative) weights over grid points for each ray.
            The Ne profile for each ray is a weighted average of the profiles
            from all grid points with non-zero weight.  Rows need not sum to 1;
            they are normalised internally.  When supplied, grid_point_indices
            is ignored.

        Returns
        -------
        Y_f : ndarray, shape (n_rays, n_members)
            Simulated sTEC in TECU for each ray and ensemble member.
        """
        if ensemble is None:
            ensemble = self.state.ensemble
        if ensemble is None:
            raise RuntimeError("No ensemble available.")

        # Convert log10 densities → linear for profile evaluation
        params_lin = self.state.to_linear_densities(ensemble)
        # params_lin : (N_STATE, n_grid_points, n_members)

        n_grid = self.state.n_grid_points
        n_rays = len(ray_trajectories)
        n_members = ensemble.shape[2]

        Y_f = np.empty((n_rays, n_members), dtype=float)

        if grid_point_weights is not None:
            # IDW blending path — smooth across cell boundaries
            for i, ray in enumerate(ray_trajectories):
                Y_f[i] = self._integrate_ray_idw(
                    ray, params_lin, grid_point_weights[i]
                )
        else:
            if grid_point_indices is None:
                if n_grid == 1:
                    grid_point_indices = np.zeros(n_rays, dtype=int)
                else:
                    raise ValueError(
                        "Provide grid_point_indices or grid_point_weights "
                        "for multi-column domains."
                    )
            for i, ray in enumerate(ray_trajectories):
                gp = grid_point_indices[i]
                Y_f[i] = self._integrate_ray(
                    ray, params_lin[:, gp, :]   # (N_STATE, n_members)
                )

        return Y_f

    # ── Ray integration ───────────────────────────────────────────────────────

    def _integrate_ray(
        self,
        ray: np.ndarray,
        params_lin: np.ndarray,
    ) -> np.ndarray:
        """
        Numerically integrate Ne along one ray for all ensemble members.

        Parameters
        ----------
        ray : ndarray, shape (n_pts, 3)   [lat, lon, alt_km]
        params_lin : ndarray, shape (N_STATE, n_members)   linear density space

        Returns
        -------
        stec : ndarray, shape (n_members,)   sTEC in TECU
        """
        alts_km = ray[:, 2]                 # (n_pts,)
        path_len_km = self._arc_length_km(ray)  # (n_pts,)

        # Evaluate Ne at every sample altitude along the ray
        # Ne_ray : (n_pts, n_members)
        Ne_ray = _ne_profile_ensemble(alts_km, params_lin)

        # Integrate along the arc-length coordinate (km → m conversion)
        # trapezoid(y, x) with x in km, y in m^-3 → result in m^-3 · km
        stec_m3_km = trapezoid(Ne_ray, path_len_km, axis=0)  # (n_members,)

        # Convert m^-3·km → m^-3·m → TECU
        stec_tecu = stec_m3_km * 1.0e3 / _TECU               # (n_members,)
        return stec_tecu

    def _integrate_ray_idw(
        self,
        ray: np.ndarray,
        params_lin: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        IDW-blended integration: Ne = weighted average of profiles from each
        grid point with non-zero weight, then integrate along the ray.

        Blending the Ne *profiles* (not the parameters) is correct because the
        profile formula is nonlinear — blending parameters would give wrong
        results when adjacent grid points have different hmF2 etc.

        Parameters
        ----------
        ray        : (n_pts, 3)
        params_lin : (N_STATE, n_grid_points, n_members)
        weights    : (n_grid_points,)  non-negative IDW weights (normalised here)

        Returns
        -------
        stec : (n_members,)
        """
        alts_km     = ray[:, 2]
        path_len_km = self._arc_length_km(ray)

        w = np.asarray(weights, dtype=float)
        active = np.where(w > 0.0)[0]
        w_active = w[active] / w[active].sum()  # normalise

        # Accumulate weighted Ne across active grid points
        Ne_blend = np.zeros((len(alts_km), params_lin.shape[2]), dtype=float)
        for gp, wg in zip(active, w_active):
            Ne_blend += wg * _ne_profile_ensemble(alts_km, params_lin[:, gp, :])

        stec_m3_km = trapezoid(Ne_blend, path_len_km, axis=0)
        return stec_m3_km * 1.0e3 / _TECU

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _arc_length_km(ray: np.ndarray) -> np.ndarray:
        """
        Cumulative arc-length along the ray in km (spherical Earth, R=6371 km).

        Parameters
        ----------
        ray : ndarray, shape (n_pts, 3)   [lat_deg, lon_deg, alt_km]

        Returns
        -------
        s : ndarray, shape (n_pts,)  starts at 0.
        """
        R_E = 6371.0                            # km
        lat = np.radians(ray[:, 0])
        lon = np.radians(ray[:, 1])
        alt = ray[:, 2]
        r   = R_E + alt                         # geocentric radius, km

        # ECEF positions (no need for a full ellipsoid here)
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)

        pts   = np.stack([x, y, z], axis=1)     # (n_pts, 3)
        diffs = np.diff(pts, axis=0)             # (n_pts-1, 3)
        segs  = np.linalg.norm(diffs, axis=1)   # (n_pts-1,)
        return np.concatenate([[0.0], np.cumsum(segs)])

    # ── Linearised H matrix (for diagnostics / EKF fallback) ─────────────────

    def linearised_H(
        self,
        ray_trajectories: list[np.ndarray],
        mean_state: np.ndarray,
        eps_rel: float = 1e-4,
    ) -> np.ndarray:
        """
        Finite-difference Jacobian of H evaluated at mean_state.

        Useful for sanity-checking the ensemble spread or as a fall-back EKF
        linearisation.

        Parameters
        ----------
        mean_state : ndarray, shape (N_STATE, n_grid_points)
        eps_rel : float
            Relative perturbation size.

        Returns
        -------
        H_lin : ndarray, shape (n_rays, N_STATE * n_grid_points)
        """
        from .ionospheric_state import N_STATE

        n_rays    = len(ray_trajectories)
        n_state   = N_STATE * self.state.n_grid_points

        # Wrap mean into a trivial single-member ensemble
        ens_mean = mean_state[:, :, np.newaxis]   # (N_STATE, n_grid_points, 1)

        tmp_state = IonosphericState(self.state.n_grid_points, n_members=1)
        tmp_state.ensemble = ens_mean
        tmp_op = ObservationOperator(tmp_state, self.alt_grid_km)

        y0 = tmp_op.compute_stec_ensemble(ray_trajectories)[:, 0]   # (n_rays,)

        H_lin = np.zeros((n_rays, n_state), dtype=float)

        flat = mean_state.ravel()                 # (N_STATE * n_grid_points,)
        for j in range(n_state):
            flat_p = flat.copy()
            dv = max(abs(flat[j]) * eps_rel, 1e-8)
            flat_p[j] += dv
            pert = flat_p.reshape(mean_state.shape)

            tmp_state.ensemble = pert[:, :, np.newaxis]
            y_p = tmp_op.compute_stec_ensemble(ray_trajectories)[:, 0]
            H_lin[:, j] = (y_p - y0) / dv

        return H_lin
