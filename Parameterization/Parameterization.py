#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameterizations of electron density profiles.

Defines a family of mappings between a vertical electron-density profile
(EDP) and a compact parameter vector, plus their Jacobians, for use in
(extended) Kalman filter state estimation:

- ``'density_10ex'``  : param = log10(density), pointwise (no compression).
- ``'ANCHOR'``        : an 8-parameter analytic Epstein/Chapman-style model
                        of the whole profile (NmF2, hmF2, H0, gamma, B0, B1,
                        NmE, hmE), fit by nonlinear least squares.
- ``'PCA_1D'``/``'PCA_1D_10ex'`` : per-geolocation empirical-orthogonal-
                        function (PCA) compression of the altitude profile.
- ``'PCA_3D'``/``'PCA_3D_10ex'`` : joint (altitude, geolocation) PCA
                        compression -- one basis shared across the mesh.

Each style is bundled into a ``ParameterizationSpec`` (three callables plus
a couple of flags) registered in ``EDP_Parameterization._build_spec``.
Adding or fixing a style means editing that one function, rather than
keeping several parallel ``match`` blocks in sync by hand.

``EDP_Parameterization`` and ``Parameterized_EDPSamples`` also carry
diagnostic methods (``reconstruction_error``, ``error_summary``, and a
family of ``plot_*`` methods) that answer "how much error does this
parameterization introduce for a given EDP sample" -- PCA truncation error
for the ``PCA_*`` styles, ANCHOR fit error for ``'ANCHOR'``.

Created on Wed Sep 9 08:18:22 2026
@author: cwang
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import xarray as xr
from scipy.optimize import minimize
from tqdm import tqdm

from edp_samples import EDPSamples

__all__ = [
    "density_10ex", "Jacobian_density_10ex", "log10_param",
    "ANCHOR_PARAM_LABELS", "ANCHOR_DEFAULT_BOUNDS",
    "anchor_region_boundaries",
    "PCADiagnostics", "get_PCA",
    "PCA2EDP_1D", "EDP2PCA_1D", "PCA2EDP_1D_map",
    "PCA2EDP_3D", "EDP2PCA_3D", "PCA2EDP_3D_map",
    "Parameterization_Style", "EDP_Parameterization", "Parameterized_EDPSamples",
]


# ===========================================================================
# 'density_10ex': pointwise log10(density) <-> density, no compression.
# ===========================================================================
#
# These are pure functions: they never mutate their input arrays (values
# below the floor are clamped in a freshly built array via np.where).

def density_10ex(log10_density: np.ndarray, minlog10Density: float = 4.0) -> np.ndarray:
    """log10(density) -> density, flooring the input at ``minlog10Density``."""
    log10_density = np.asarray(log10_density, dtype=float)
    log10_density = np.where(log10_density < minlog10Density, minlog10Density, log10_density)
    return 10.0 ** log10_density


def Jacobian_density_10ex(log10_density: np.ndarray, minlog10Density: float = 4.0) -> np.ndarray:
    """
    d(density)/d(log10_density); diagonal, so only the diagonal is returned.

    Zero below the floor: ``density_10ex`` is constant (clamped) there, so
    its true local derivative is zero, not ``ln(10) * floor``.
    """
    log10_density = np.asarray(log10_density, dtype=float)
    density = density_10ex(log10_density, minlog10Density=minlog10Density)
    jacobian = np.log(10.0) * density
    return np.where(log10_density < minlog10Density, 0.0, jacobian)


def log10_param(density: np.ndarray, minlog10Density: float = 4.0) -> np.ndarray:
    """density -> log10(density), flooring density at ``10**minlog10Density`` first."""
    density = np.asarray(density, dtype=float)
    floor = 10.0 ** minlog10Density
    density = np.where(density < floor, floor, density)
    return np.log10(density)


# ===========================================================================
# 'ANCHOR': 8-parameter analytic profile model + its Jacobian + the fit
# that recovers the 8 parameters from an arbitrary EDP.
# ===========================================================================

ANCHOR_PARAM_LABELS = (
    "log10_NmF2", "hmF2", "H0", "gamma", "B0", "B1", "log10_NmE", "hmE",
)
ANCHOR_N_STATE = len(ANCHOR_PARAM_LABELS)

ANCHOR_DEFAULT_BOUNDS: dict[str, Any] = {
    "log10_NmF2": [9.0, 13.0],
    "hmF2": [100.0, 600.0],
    "H0": [10.0, 300.0],
    "gamma": [0.05, 2.0],
    "B0": [20.0, 300.0],
    "B1": [0.5, 4.0],
    "log10_NmE": [7.0, 12.0],
    "hmE": [80.0, 180.0],
}

# Indices of the 8 state components within a param_vec's leading axis.
_I_LOG_NMF2, _I_HMF2, _I_H0, _I_GAMMA, _I_B0, _I_B1, _I_LOG_NME, _I_HME = range(8)

# Fixed model constants (kept identical to the original derivation).
_R_TOPSIDE: float = 100.0   # Topside shape parameter (IRI default)
_H_E_KM: float = 15.0       # E-layer scale height (km)


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
        x in [0, 100]).

    Returns
    -------
    h_ST : ndarray, shape (n_members,)
        Transition altitude in km. Clipped to [hmE, hmF2] by callers.
    """
    x_lo = np.zeros_like(NmF2)
    x_hi = np.full_like(NmF2, 100.0)   # x=100 -> h well below E-layer

    for _ in range(n_iter):
        x_mid = 0.5 * (x_lo + x_hi)
        f_mid = NmF2 * np.exp(-(x_mid ** B1)) / np.cosh(np.clip(x_mid, 0, 700)) - NmE
        x_lo = np.where(f_mid > 0.0, x_mid, x_lo)
        x_hi = np.where(f_mid > 0.0, x_hi, x_mid)

    x_st = 0.5 * (x_lo + x_hi)
    return hmF2 - x_st * B0


def anchor_region_boundaries(param_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ``(hmE, h_ST, hmF2)`` region boundaries for an ANCHOR parameter vector.

    Parameters
    ----------
    param_vec : (8, ...) array in the ANCHOR log/linear convention.

    Returns
    -------
    hmE, h_ST, hmF2 : arrays shaped like ``param_vec.shape[1:]`` (a scalar
        float triple if ``param_vec`` is 1-D).  Useful for diagnostics that
        want to mark the piecewise model's region transitions (e.g.
        ``Parameterized_EDPSamples.plot_reconstruction_profile``) without
        re-deriving them from scratch.
    """
    p = np.asarray(param_vec, dtype=float)
    if p.shape[0] != ANCHOR_N_STATE:
        raise ValueError(f"Leading dimension of param_vec is {p.shape[0]}, expected {ANCHOR_N_STATE}.")
    tail_shape = p.shape[1:]
    flat = p.reshape(ANCHOR_N_STATE, -1)

    NmF2 = 10.0 ** flat[_I_LOG_NMF2]
    hmF2 = flat[_I_HMF2]
    B0 = flat[_I_B0]
    B1 = flat[_I_B1]
    NmE = 10.0 ** flat[_I_LOG_NME]
    hmE = flat[_I_HME]

    h_ST = np.clip(_find_hst_bisection(NmF2, hmF2, B0, B1, NmE), hmE, hmF2)

    if tail_shape:
        return hmE.reshape(tail_shape), h_ST.reshape(tail_shape), hmF2.reshape(tail_shape)
    return float(hmE[0]), float(h_ST[0]), float(hmF2[0])


def _ne_profile_derivatives(
    alts_km: np.ndarray,
    params_lin: np.ndarray,
    partial: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute Ne and, if requested, analytical d Ne(h)/d P_k for each altitude
    and each state parameter.

    Treats h_ST (the F2/E-layer transition altitude from bisection) as a
    fixed boundary when computing region derivatives -- the standard
    piecewise-smooth approximation: boundaries have measure zero and don't
    affect integral derivatives.

    Parameters are in LINEAR density space (NmF2, NmE in m^-3).  The
    returned derivatives are w.r.t. the stored log10 parameters at indices
    I_LOG_NMF2 and I_LOG_NME, with the ln(10)*Nm chain-rule factor applied.

    Parameters
    ----------
    alts_km    : (n_alt,)  altitude sample points in km
    params_lin : (N_STATE, ...)  parameters in linear density space
    partial    : if True (default), also compute and return the Jacobian
        dNe_dP; if False, compute density only (cheaper).

    Returns
    -------
    Ne_ensemble : (n_alt, ...)  electron density profile (m^-3)
    dNe_dP_ensemble : (n_alt, N_STATE, ...)  dNe/dP_k at every altitude,
        only returned when ``partial`` is True.
    """
    def _bs_quantities(h_eff_arr):
        """Return (Ne_bs, log_df, x, x_pow, log_x) for given effective altitudes."""
        x = np.maximum((hmF2 - h_eff_arr) / (B0 + 1e-9), 0.0)
        x_pow = np.where(x > 1e-30, x, 1e-30)
        cosh_x = np.cosh(np.clip(x, 0.0, 700.0))
        tanh_x = np.tanh(np.clip(x, 0.0, 700.0))
        Ne_bs = NmF2 * np.exp(-(x_pow ** B1)) / cosh_x
        log_df = np.where(x > 1e-30, -B1 * x_pow ** (B1 - 1.0) - tanh_x, -tanh_x)
        log_x = np.log(x_pow)
        return Ne_bs, log_df, x, x_pow, log_x

    h = np.asarray(alts_km, dtype=float)
    n_alt = len(h)
    params_lin = np.asarray(params_lin, dtype=float)
    params_shape = list(params_lin.shape)
    if params_shape[0] != ANCHOR_N_STATE:
        raise ValueError(f"Leading dimension of param_vec is {params_shape[0]}, "
                          f"expected {ANCHOR_N_STATE}.")

    params_lin = params_lin.reshape(ANCHOR_N_STATE, -1)
    nPts = params_lin.shape[1]

    Ne_ensemble = np.zeros((n_alt, nPts), dtype=float)
    if partial:
        dNe_dP_ensemble = np.zeros((n_alt, ANCHOR_N_STATE, nPts), dtype=float)

    for iPts in range(nPts):
        # NmF2/NmE are stored as log10(density) (see ANCHOR_PARAM_LABELS,
        # _fit_iri_params) -- exponentiate to the linear density these
        # formulas (and their ln(10)-chain-rule Jacobian below) assume.
        NmF2 = 10.0 ** float(params_lin[_I_LOG_NMF2, iPts])
        hmF2 = float(params_lin[_I_HMF2, iPts])
        H0 = float(params_lin[_I_H0, iPts])
        gamma = float(params_lin[_I_GAMMA, iPts])
        B0 = float(params_lin[_I_B0, iPts])
        B1 = float(params_lin[_I_B1, iPts])
        NmE = 10.0 ** float(params_lin[_I_LOG_NME, iPts])
        hmE = float(params_lin[_I_HME, iPts])

        # -- Region boundaries -----------------------------------------------
        h_ST = float(np.clip(
            _find_hst_bisection(
                np.array([NmF2]), np.array([hmF2]),
                np.array([B0]), np.array([B1]), np.array([NmE]),
            )[0],
            hmE, hmF2,
        ))

        mask_top = h >= hmF2
        mask_bot = (h >= h_ST) & ~mask_top
        mask_int = (h >= hmE) & (h < h_ST)
        mask_e = h < hmE

        # -- Build composite Ne profile ---------------------------------------
        r = _R_TOPSIDE
        # Topside
        dh_all = h - hmF2
        D_all = r * H0 + gamma * dh_all + 1e-9
        H_all = H0 * (1.0 + r * gamma * dh_all / D_all)
        z_all = dh_all / (H_all + 1e-9)
        exp_z = np.exp(np.clip(z_all, -80, 80))
        Ne_top_v = 4.0 * NmF2 * exp_z / (1.0 + exp_z) ** 2

        # Bottomside: direct branch (also used for pure bottomside region) and
        # mirrored branch (h_eff = h_ST + hmE - h), blended in Region 3 below.
        h_eff_all = hmE + h_ST - h
        Ne_bs_bot, _, _, _, _ = _bs_quantities(h)
        Ne_bs_mir, _, _, _, _ = _bs_quantities(h_eff_all)

        # Region 3 (intermediate connection) smoothstep blend:
        #   t = clip((h-hmE)/(h_ST-hmE), 0, 1),  w = 3t^2 - 2t^3
        #   Ne_bs_int = w*Ne_bs_bot + (1-w)*Ne_bs_mir
        t_all = np.clip((h - hmE) / (h_ST - hmE + 1e-9), 0.0, 1.0)
        w_all = 3.0 * t_all ** 2 - 2.0 * t_all ** 3
        Ne_bs_int = w_all * Ne_bs_bot + (1.0 - w_all) * Ne_bs_mir

        # E-layer -- bottomside alpha-Chapman
        ze_all = np.clip((h - hmE) / _H_E_KM, -80, 80)
        exp_neg_ze = np.exp(-ze_all)
        Ne_E_v = NmE * np.exp(0.5 * (1.0 - ze_all - exp_neg_ze))

        Ne = np.where(
            mask_top, Ne_top_v,
            np.where(mask_bot, Ne_bs_bot,
            np.where(mask_int, Ne_bs_int, Ne_E_v)),
        )
        Ne = np.maximum(Ne, 0.0)
        Ne_ensemble[:, iPts] = Ne
        if not partial:
            continue

        dNe_dP = np.zeros((ANCHOR_N_STATE, n_alt), dtype=float)
        # -- Topside derivatives (h >= hmF2) -----------------------------------
        if mask_top.any():
            dh_t = dh_all[mask_top]
            D_t = D_all[mask_top]
            N_num = r * H0 + gamma * (1.0 + r) * dh_t
            H_t = H_all[mask_top]
            sig_t = exp_z[mask_top] / (1.0 + exp_z[mask_top]) ** 2
            dsig_t = sig_t * (1.0 - exp_z[mask_top]) / (1.0 + exp_z[mask_top])

            Ne_t = Ne_top_v[mask_top]

            dNe_dP[_I_LOG_NMF2, mask_top] = Ne_t * np.log(10.0)

            dHt_ddh = r ** 2 * gamma * H0 ** 2 / D_t ** 2
            dz_hmF2 = (-H_t + dh_t * dHt_ddh) / (H_t + 1e-9) ** 2
            dNe_dP[_I_HMF2, mask_top] = 4.0 * NmF2 * dsig_t * dz_hmF2

            dHt_H0 = N_num / D_t - r ** 2 * H0 * gamma * dh_t / D_t ** 2
            dz_H0 = -dh_t * dHt_H0 / (H_t + 1e-9) ** 2
            dNe_dP[_I_H0, mask_top] = 4.0 * NmF2 * dsig_t * dz_H0

            dHt_gam = r ** 2 * H0 ** 2 * dh_t / D_t ** 2
            dz_gam = -dh_t * dHt_gam / (H_t + 1e-9) ** 2
            dNe_dP[_I_GAMMA, mask_top] = 4.0 * NmF2 * dsig_t * dz_gam
            # B0, B1, log10(NmE), hmE: 0 in topside -> already zero

        # -- Pure bottomside derivatives (h_ST <= h < hmF2) --------------------
        if mask_bot.any():
            h_b = h[mask_bot]
            Ne_b, ldf_b, x_b, xp_b, lx_b = _bs_quantities(h_b)

            dNe_dP[_I_LOG_NMF2, mask_bot] = Ne_b * np.log(10.0)
            dNe_dP[_I_HMF2, mask_bot] = Ne_b * ldf_b / (B0 + 1e-9)
            dNe_dP[_I_B0, mask_bot] = Ne_b * ldf_b * (-x_b / (B0 + 1e-9))
            dNe_dP[_I_B1, mask_bot] = Ne_b * (-xp_b ** B1 * lx_b)

        # -- Intermediate connection derivatives (hmE <= h < h_ST) -------------
        #
        # Ne_inter = w*Ne_A + (1-w)*Ne_B,   w = 3t^2 - 2t^3,  t = (h-hmE)/(h_ST-hmE)
        #   Ne_A = direct branch  (h_eff = h)        -- same form as pure bottomside
        #   Ne_B = mirrored branch (h_eff = h_ST + hmE - h)
        #
        # dNe_inter/dtheta = (dw/dtheta)*(Ne_A-Ne_B) + w*dNe_A/dtheta + (1-w)*dNe_B/dtheta
        #
        # h_ST = hmF2 - x_ST*B0 where x_ST satisfies the bisection equation
        #   g(x_ST) = NmF2*exp(-x_ST^B1)/cosh(x_ST) - NmE = 0
        #
        # Implicit function theorem gives dx_ST/dtheta = -(dg/dtheta)/(dg/dx_ST):
        #   dg/dx_ST       = NmE * ldf(x_ST)          (ldf < 0)
        #   dg/dhmF2       = 0  -> dh_ST/dhmF2 = 1
        #   dg/dB0         = 0  -> dh_ST/dB0   = -x_ST
        #   dg/dB1         = NmE*(-x_ST^B1*ln x_ST)
        #                  -> dx_ST/dB1 = x_ST^B1*ln(x_ST) / ldf(x_ST)
        #                  -> dh_ST/dB1 = -B0*dx_ST/dB1
        #   dg/dNmF2       = NmE/NmF2
        #                  -> dx_ST/dNmF2 = -1/(NmF2*ldf(x_ST))
        #                  -> dh_ST/dNmF2 = B0/(NmF2*ldf(x_ST))  (negative)
        #   dg/dNmE        = -1
        #                  -> dx_ST/dNmE  = 1/(NmE*ldf(x_ST))
        #                  -> dh_ST/dNmE  = -B0/(NmE*ldf(x_ST))  (positive)
        #   dg/dhmE        = 0  -> dh_ST/dhmE = 0
        #
        # For Ne_B: x_eff = (hmF2 - h_eff)/B0 = (hmF2 - h_ST - hmE + h)/B0
        #   dx_eff/dtheta = (dhmF2/dtheta - dh_ST/dtheta - dhmE/dtheta) / B0 - x_eff * dB0/dtheta / B0
        #
        # For w: t = (h-hmE)/(h_ST-hmE) = N/D.  For theta affecting h_ST only
        # (NmF2, B0, B1, NmE): dt/dtheta = -t*dh_ST/dtheta / D.  For theta = hmE
        # (dh_ST/dhmE=0 but hmE enters both N and D directly): dt/dhmE = (t-1)/D.
        if mask_int.any():
            h_i = h[mask_int]
            h_eff_i = h_ST + hmE - h_i

            x_ST_v = float((hmF2 - h_ST) / (B0 + 1e-9))
            xST_pow = max(x_ST_v, 1e-30)
            tanh_xST = float(np.tanh(min(x_ST_v, 700.0)))
            ldf_xST = float((-B1 * xST_pow ** (B1 - 1.0) - tanh_xST)
                            if x_ST_v > 1e-30 else -tanh_xST)
            safe_ldf_xST = ldf_xST if abs(ldf_xST) > 1e-30 else -1e-30
            log_xST = float(np.log(max(xST_pow, 1e-30)))
            hst_b1_coeff = xST_pow ** B1 * log_xST / safe_ldf_xST

            Ne_A, ldf_A, x_A, xp_A, lx_A = _bs_quantities(h_i)
            dNeA = np.zeros((ANCHOR_N_STATE, len(h_i)))
            dNeA[_I_LOG_NMF2] = Ne_A * np.log(10.0)
            dNeA[_I_HMF2] = Ne_A * ldf_A / (B0 + 1e-9)
            dNeA[_I_B0] = Ne_A * ldf_A * (-x_A / (B0 + 1e-9))
            dNeA[_I_B1] = Ne_A * (-xp_A ** B1 * lx_A)

            Ne_B, ldf_B, x_B, xp_B, lx_B = _bs_quantities(h_eff_i)
            dNeB = np.zeros((ANCHOR_N_STATE, len(h_i)))
            dNeB[_I_LOG_NMF2] = Ne_B * np.log(10.0) * (1.0 - ldf_B / safe_ldf_xST)
            dNeB[_I_B0] = Ne_B * ldf_B * (x_ST_v - x_B) / (B0 + 1e-9)
            dNeB[_I_B1] = Ne_B * (ldf_B * hst_b1_coeff - xp_B ** B1 * lx_B)
            dNeB[_I_LOG_NME] = Ne_B * np.log(10.0) * ldf_B / safe_ldf_xST
            dNeB[_I_HME] = Ne_B * ldf_B * (-1.0 / (B0 + 1e-9))

            D = h_ST - hmE + 1e-9
            t = np.clip((h_i - hmE) / D, 0.0, 1.0)
            w = 3.0 * t ** 2 - 2.0 * t ** 3
            dw_dt = 6.0 * t * (1.0 - t)

            dhST = np.zeros(ANCHOR_N_STATE)
            dhST[_I_HMF2] = 1.0
            dhST[_I_B0] = -x_ST_v
            dhST[_I_B1] = -B0 * hst_b1_coeff
            dhST[_I_LOG_NMF2] = B0 * np.log(10.0) / safe_ldf_xST
            dhST[_I_LOG_NME] = -B0 * np.log(10.0) / safe_ldf_xST

            dt_dtheta = np.zeros((ANCHOR_N_STATE, len(h_i)))
            for k in (_I_LOG_NMF2, _I_HMF2, _I_B0, _I_B1, _I_LOG_NME):
                dt_dtheta[k] = -t * dhST[k] / D
            dt_dtheta[_I_HME] = (t - 1.0) / D
            dw_dtheta = dw_dt * dt_dtheta

            idx_int = np.where(mask_int)[0]
            dNe_dP[:, idx_int] = (
                dw_dtheta * (Ne_A - Ne_B)
                + w * dNeA
                + (1.0 - w) * dNeB
            )

        # -- E-layer derivatives (h < hmE) -- bottomside alpha-Chapman ---------
        if mask_e.any():
            exp_neg_ze_e = exp_neg_ze[mask_e]
            Ne_Ev = Ne_E_v[mask_e]

            dNe_dP[_I_LOG_NME, mask_e] = Ne_Ev * np.log(10.0)
            dNe_dP[_I_HME, mask_e] += Ne_Ev * 0.5 * (1.0 - exp_neg_ze_e) / _H_E_KM

        dNe_dP_ensemble[:, :, iPts] = dNe_dP.T

    params_shape[0] = n_alt
    Ne_ensemble = Ne_ensemble.reshape(params_shape)
    if not partial:
        return Ne_ensemble

    dNe_dP_ensemble = dNe_dP_ensemble.reshape([n_alt, ANCHOR_N_STATE] + params_shape[1:])
    return Ne_ensemble, dNe_dP_ensemble


def extract_robust_f2_peak(profile: np.ndarray, alt_grid: np.ndarray,
                            min_alt: float = 150.0, max_alt: float = 650.0,
                            _half_window: int = 4, _weight_sigma_km: float = 14.0,
                            _min_side_pts: int = 2):
    """
    Robustly find NmF2 and hmF2 from a sampled Ne(h) profile by fitting a
    *two-sided* (asymmetric) parabola in log(Ne).

    A 3-point parabola in LINEAR Ne (the naive approach) is biased HIGH by
    roughly +6 to +16 km on the geometric altitude grid: the F2 peak is
    asymmetric -- the bottomside is steeper than the topside -- and the grid
    spacing grows with altitude, so a symmetric-parabola vertex is pulled
    toward the flatter topside. Worse, that bias grows with bottomside
    breadth (B0), so a broader (more physically faithful) bottomside
    spuriously raises the detected hmF2 even when the true peak height is
    unchanged.

    The fix fits Ne(h) near the peak as two half-parabolas in log(Ne)
    sharing a common vertex, with independent curvature above and below (so
    asymmetry no longer biases the vertex), and Gaussian proximity weights
    so distant, non-parabolic samples do not distort the local fit.
    """
    search_mask = (alt_grid >= min_alt) & (alt_grid <= max_alt) & ~np.isnan(profile)
    if not np.any(search_mask):
        return np.nan, np.nan

    search_alts = alt_grid[search_mask]
    search_prof = profile[search_mask]

    local_max_idx = int(np.argmax(search_prof))
    discrete_hmF2 = search_alts[local_max_idx]
    discrete_NmF2 = search_prof[local_max_idx]

    lo = max(0, local_max_idx - _half_window)
    hi = min(len(search_alts) - 1, local_max_idx + _half_window)
    if hi - lo < 4:
        return discrete_NmF2, discrete_hmF2

    h = search_alts[lo:hi + 1]
    y = np.log(np.maximum(search_prof[lo:hi + 1], 1.0))

    half_step = (search_alts[min(local_max_idx + 1, len(search_alts) - 1)]
                 - search_alts[max(local_max_idx - 1, 0)]) / 2.0
    best_sse, best_hmF2, best_NmF2 = np.inf, discrete_hmF2, discrete_NmF2
    for h0 in np.linspace(discrete_hmF2 - half_step, discrete_hmF2 + half_step, 81):
        d = h - h0
        below = d < 0.0
        if np.sum(below) < _min_side_pts or np.sum(~below) < _min_side_pts:
            continue
        w = np.exp(-0.5 * (d / _weight_sigma_km) ** 2)
        A = np.column_stack([np.ones_like(d),
                              -np.where(below, d ** 2, 0.0),
                              -np.where(~below, d ** 2, 0.0)])
        try:
            coeffs, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        except np.linalg.LinAlgError:
            continue
        y0, k_lo, k_hi = coeffs
        if k_lo <= 0.0 or k_hi <= 0.0:
            continue
        sse = float(np.sum(((A @ coeffs - y) * w) ** 2))
        if sse < best_sse:
            best_sse, best_hmF2, best_NmF2 = sse, float(h0), float(np.exp(y0))

    return best_NmF2, best_hmF2


def _h0_seed_from_profile(ne_topside, alts_topside, nm_f2, hm_f2) -> float:
    """
    Estimate the initial topside scale height H0 from the IRI ne profile
    without any external call.

    Uses the Epstein half-power point: 4*e^z/(1+e^z)^2 = 0.5  ->  z ~ 1.317,
    so H0_seed ~ Delta_h_half / 1.317, where Delta_h_half is the altitude
    above hmF2 where ne first drops to NmF2/2.

    Falls back to 60 km if no valid half-power point is found.
    """
    _EPSTEIN_Z_HALF = 1.3169578

    ne_arr = np.asarray(ne_topside)
    alt_arr = np.asarray(alts_topside)

    valid = np.isfinite(ne_arr) & (ne_arr > 0)
    if valid.sum() < 2:
        return 60.0

    ne_v = ne_arr[valid]
    alt_v = alt_arr[valid]
    half_target = nm_f2 / 2.0

    below = np.where(ne_v < half_target)[0]
    if len(below) == 0:
        H0_seed = float(alt_v[-1] - hm_f2) / _EPSTEIN_Z_HALF
    elif below[0] == 0:
        H0_seed = 60.0
    else:
        i = below[0]
        frac = (half_target - ne_v[i - 1]) / (ne_v[i] - ne_v[i - 1] + 1e-9)
        h_half = alt_v[i - 1] + frac * (alt_v[i] - alt_v[i - 1])
        H0_seed = (h_half - hm_f2) / _EPSTEIN_Z_HALF

    return float(np.clip(H0_seed, 10.0, 200.0))


def _fit_topside_H0_gamma(ne_topside, alts_topside, nm_f2, hm_f2, H0_seed) -> tuple[float, float]:
    """
    Jointly fit H0 and gamma from the valid (finite, positive) IRI ne topside
    profile by minimising log10-RMSE.

    Returns (H0, gamma) with H0 in [10, 300] km and gamma in [0.01, 2.0].
    """
    r = 100.0

    ne_arr = np.asarray(ne_topside)
    alts_arr = np.asarray(alts_topside)

    valid = np.isfinite(ne_arr) & (ne_arr > 0)
    if valid.sum() < 3:
        return H0_seed, 0.5

    ne_ref = np.log10(ne_arr[valid])
    dh = alts_arr[valid] - hm_f2

    def _cost(x):
        H0_t = np.exp(x[0])
        g = np.exp(x[1])
        H_eff = H0_t * (1.0 + r * g * dh / (r * H0_t + g * dh + 1e-9))
        z = np.clip(dh / (H_eff + 1e-9), -80, 80)
        ne_m = np.maximum(4.0 * nm_f2 * np.exp(z) / (1.0 + np.exp(z)) ** 2, 1.0)
        return float(np.mean((np.log10(ne_m) - ne_ref) ** 2))

    x0 = np.array([np.log(H0_seed), np.log(0.5)])
    bds = [(np.log(10.0), np.log(300.0)), (np.log(0.01), np.log(2.0))]
    res = minimize(_cost, x0, method="L-BFGS-B", bounds=bds,
                    options={"maxiter": 200, "ftol": 1e-6})
    H0_fit = float(np.clip(np.exp(res.x[0]), 10.0, 300.0))
    gamma_fit = float(np.clip(np.exp(res.x[1]), 0.01, 2.0))
    return H0_fit, gamma_fit


def _fit_bottomside_B0_B1(ne_bottom, alts_bottom, nm_f2, hm_f2, B0_seed, B1_seed) -> tuple[float, float]:
    """
    Jointly fit B0 and B1 from the valid (finite, positive) IRI ne
    bottomside profile by minimising log10-RMSE against the model's own
    pure-bottomside formula, ``Ne(h) = NmF2 * exp(-x**B1) / cosh(x)``.

    Returns (B0, B1) with B0 in [20, 300] km and B1 in [0.5, 4.0].
    """
    ne_arr = np.asarray(ne_bottom)
    alts_arr = np.asarray(alts_bottom)

    valid = np.isfinite(ne_arr) & (ne_arr > 0)
    if valid.sum() < 4:
        return B0_seed, B1_seed

    ne_ref = np.log10(ne_arr[valid])
    hh = alts_arr[valid]

    def _cost(x):
        B0_t = np.exp(x[0])
        B1_t = np.exp(x[1])
        xx = np.maximum((hm_f2 - hh) / (B0_t + 1e-9), 0.0)
        xp = np.where(xx > 0, xx, 1e-30)
        ne_m = np.maximum(
            nm_f2 * np.exp(-xp ** B1_t) / np.cosh(np.clip(xx, 0.0, 700.0)), 1.0
        )
        return float(np.mean((np.log10(ne_m) - ne_ref) ** 2))

    x0 = np.array([np.log(np.clip(B0_seed, 20.0, 300.0)),
                    np.log(np.clip(B1_seed, 0.5, 4.0))])
    bds = [(np.log(20.0), np.log(300.0)), (np.log(0.5), np.log(4.0))]
    try:
        res = minimize(_cost, x0, method="L-BFGS-B", bounds=bds,
                        options={"maxiter": 200, "ftol": 1e-7})
        if res.fun <= _cost(x0):
            x0 = res.x
    except Exception:
        pass
    B0_fit = float(np.clip(np.exp(x0[0]), 20.0, 300.0))
    B1_fit = float(np.clip(np.exp(x0[1]), 0.5, 4.0))
    return B0_fit, B1_fit


def _fit_iri_params(ne_profile: np.ndarray, alt_grid: np.ndarray,
                     param_bound: dict) -> np.ndarray:
    """
    Fit the 8 ANCHOR state vector components to a single Ne(h) profile using
    a combination of direct extraction and numerical optimisation.

    Parameters
    ----------
    ne_profile : (n_alt,)  electron density in m^-3 (linear).
    alt_grid   : (n_alt,)  altitude grid in km.
    param_bound : dict mapping each name in ANCHOR_PARAM_LABELS to [lo, hi].

    Returns
    -------
    params_log : (8,)  in the mixed log/linear state-vector convention:
        [log10(NmF2), hmF2, H0, gamma, B0, B1, log10(NmE), hmE]
    """
    ne = np.maximum(np.asarray(ne_profile, dtype=float), 1.0)
    alts = np.asarray(alt_grid, dtype=float)

    # -- F2 peak ---------------------------------------------------------------
    nm_f2, hm_f2 = extract_robust_f2_peak(ne, alts)
    if np.isnan(nm_f2) or np.isnan(hm_f2):
        nm_f2 = float(np.nanmax(ne))
        hm_f2 = float(alts[np.nanargmax(ne)])

    # -- E-layer peak (95-140 km) ------------------------------------------------
    e_mask = (alts >= 95.0) & (alts <= 140.0)
    nm_e, hm_e = None, None
    if e_mask.sum() >= 3:
        ne_e = ne[e_mask]
        alt_e = alts[e_mask]
        k = int(np.nanargmax(ne_e))
        interior = 0 < k < len(ne_e) - 1
        if interior and ne_e[k] < 0.5 * nm_f2:
            nm_e = float(ne_e[k])
            hm_e = float(alt_e[k])
    if nm_e is None:
        nm_e = float(np.clip(nm_f2 * 0.05, 1e9, 0.3 * nm_f2))
        hm_e = 110.0

    # -- Topside: H0, gamma -- analytic seed + joint log-space fit -------------
    top_mask = alts > hm_f2
    H0_seed = _h0_seed_from_profile(ne[top_mask], alts[top_mask], nm_f2, hm_f2)
    H0, gamma = H0_seed, 0.5
    if top_mask.sum() >= 5:
        try:
            H0, gamma = _fit_topside_H0_gamma(
                ne[top_mask], alts[top_mask], nm_f2, hm_f2, H0_seed
            )
        except Exception:
            pass

    # -- Bottomside: B0, B1 -- half-width seed + joint log-space fit -----------
    bot_mask = (alts < hm_f2) & (alts > 100.0)
    if bot_mask.sum() >= 3:
        ne_bot = ne[bot_mask]
        alt_bot = alts[bot_mask]
        target = nm_f2 / np.e
        below = alt_bot[ne_bot >= target]
        B0 = float(hm_f2 - below[0]) if len(below) > 0 else 80.0
        B0 = np.clip(B0, 20.0, 250.0)
        B1 = 1.5
    else:
        B0 = 80.0
        B1 = 1.5
    fit_mask = (alts > 150.0) & (alts < hm_f2)
    if fit_mask.sum() >= 4:
        try:
            B0, B1 = _fit_bottomside_B0_B1(
                ne[fit_mask], alts[fit_mask], nm_f2, hm_f2, B0, B1
            )
        except Exception:
            pass

    x0 = np.array([np.log10(nm_f2), hm_f2, H0, gamma, B0, B1,
                    np.log10(nm_e), hm_e])

    _lo = np.array([param_bound[name][0] for name in ANCHOR_PARAM_LABELS])
    _hi = np.array([param_bound[name][1] for name in ANCHOR_PARAM_LABELS])

    def _residual(x):
        p = np.minimum(np.maximum(x, _lo), _hi)
        params_lin = np.array([
            10.0 ** p[0], p[1], p[2], p[3], p[4], p[5], 10.0 ** p[6], p[7]
        ])[:, np.newaxis]
        ne_model = _ne_profile_derivatives(alts, params_lin, partial=False)[:, 0]
        return float(np.nanmean((np.log10(np.maximum(ne_model, 1.0))
                                  - np.log10(ne)) ** 2))

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                _residual, x0,
                method="Nelder-Mead",
                options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-4},
            )
        if res.fun < _residual(x0):
            x0 = np.minimum(np.maximum(res.x, _lo), _hi)
    except Exception:
        pass

    return np.clip(x0, _lo, _hi)


def _fit_iri_params_ensemble(ne_profiles: np.ndarray, alt_grid: np.ndarray,
                              param_bound: dict, show_progress: bool = True) -> np.ndarray:
    """
    Vectorising wrapper around ``_fit_iri_params``.

    ``_fit_iri_params`` is a per-profile nonlinear fit (it calls
    ``scipy.optimize.minimize`` internally) and cannot be vectorised across
    an ensemble the way the forward model can. This loops over every
    trailing "column" of ``ne_profiles`` independently.

    Parameters
    ----------
    ne_profiles : (n_alt, ...) array -- one or more EDPs sharing ``alt_grid``.

    Returns
    -------
    (8, ...) array of fitted ANCHOR parameters, one 8-vector per input profile.
    """
    ne_profiles = np.asarray(ne_profiles, dtype=float)
    n_alt = ne_profiles.shape[0]
    tail_shape = ne_profiles.shape[1:]
    flat = ne_profiles.reshape(n_alt, -1)
    n_profiles = flat.shape[1]

    out = np.empty((ANCHOR_N_STATE, n_profiles), dtype=float)
    for i in tqdm(range(n_profiles), desc="Fitting ANCHOR parameters",
                  disable=not show_progress or n_profiles <= 1):
        out[:, i] = _fit_iri_params(flat[:, i], alt_grid, param_bound)

    return out.reshape((ANCHOR_N_STATE,) + tail_shape)


# ===========================================================================
# PCA-based parameterizations.
# ===========================================================================

@dataclass
class PCADiagnostics:
    """
    Full PCA spectrum plus the truncation choice actually applied.

    Kept alongside the (truncated) PCA basis so ``EDP_Parameterization.
    plot_pca_spectrum``/``Parameterized_EDPSamples.error_summary`` can show
    *why* the truncation error looks the way it does, not just its effect.
    """
    singular_values: np.ndarray             # descending, full spectrum
    explained_variance_ratio: np.ndarray    # per-component fraction of total variance
    cumulative_variance_ratio: np.ndarray   # cumulative sum, descending order
    n_retained: int
    retaining_threshold: float


def get_PCA(edps: np.ndarray, retaining_threshold: float) -> tuple[np.ndarray, np.ndarray, PCADiagnostics]:
    """
    Principal components ("empirical orthogonal functions") of an ensemble.

    Parameters
    ----------
    edps : (n_state, n_sample) ensemble matrix; the mean is removed
        internally (NaNs are treated as a zero perturbation so they don't
        inflate variance).
    retaining_threshold : float in (0, 1]
        Minimum fraction of total ensemble variance the retained components
        must jointly explain. e.g. 0.99 retains the smallest number of
        leading (highest-variance) components whose cumulative explained-
        variance fraction is >= 0.99.

    Returns
    -------
    PCA : (n_state, n_retained) retained eigenvectors, ordered by
        decreasing variance explained -- the *dominant* modes are kept.
    mean : (n_state,) the ensemble mean that was subtracted before computing
        PCA. **Required** for correct reconstruction: PCA2EDP_1D/EDP2PCA_1D
        (and the 3D variants) must add/subtract this same mean, since PCA
        itself only spans the ensemble's zero-mean variance directions --
        without it, even full-rank reconstruction silently drops the mean
        profile.
    diagnostics : PCADiagnostics
        The full (untruncated) spectrum plus bookkeeping about the
        truncation actually applied.
    """
    edps = np.asarray(edps, dtype=float)
    n_state, n_sample = edps.shape
    edps_mean = np.nanmean(edps, axis=1, keepdims=True)
    edps_c = np.where(np.isnan(edps), 0.0, edps - edps_mean)

    if n_state <= n_sample:
        cov = edps_c @ edps_c.T
        eigval, eigvec = np.linalg.eigh(cov)
    else:
        # "Snapshot method": eigendecompose the (n_sample, n_sample) Gram
        # matrix instead, then project its eigenvectors back into state
        # space -- much cheaper when n_state >> n_sample.
        gram = edps_c.T @ edps_c
        eigval, eigvec = np.linalg.eigh(gram)

    order = np.argsort(eigval)[::-1]           # eigh returns ascending order
    eigval = np.clip(eigval[order], 0.0, None)
    eigvec = eigvec[:, order]

    # Mean-centering removes one degree of freedom, so a Gram matrix built
    # from n_sample centered snapshots has rank at most n_sample-1: its
    # smallest eigenvalue is numerically ~0. Recovering that direction's
    # "mode" via edps_c @ eigvec then dividing by its (near-zero) norm would
    # amplify floating-point noise into a spurious basis vector, so drop any
    # eigenvalue below a relative-machine-precision floor before doing that.
    eig_floor = eigval.max() * n_state * np.finfo(eigval.dtype).eps if eigval.size else 0.0
    n_numerical_rank = int(np.count_nonzero(eigval > eig_floor)) or 1
    eigval = eigval[:n_numerical_rank]
    eigvec = eigvec[:, :n_numerical_rank]

    if n_state > n_sample:
        modes = edps_c @ eigvec
        norms = np.linalg.norm(modes, axis=0)
        norms[norms == 0] = 1.0
        eigvec = modes / norms

    singular_values = np.sqrt(eigval)
    total_var = eigval.sum()
    explained = eigval / total_var if total_var > 0 else np.zeros_like(eigval)
    cumulative = np.cumsum(explained)

    n_retained = int(np.searchsorted(cumulative, retaining_threshold) + 1)
    n_retained = int(np.clip(n_retained, 1, eigvec.shape[1]))

    PCA = eigvec[:, :n_retained]
    diagnostics = PCADiagnostics(
        singular_values=singular_values,
        explained_variance_ratio=explained,
        cumulative_variance_ratio=cumulative,
        n_retained=n_retained,
        retaining_threshold=retaining_threshold,
    )
    return PCA, edps_mean.ravel(), diagnostics


def PCA2EDP_1D(PCA_state: np.ndarray, PCA: np.ndarray, mean: np.ndarray | None = None,
               linear: bool = True, minlog10Density: float = 4.0) -> np.ndarray:
    """
    Coefficient space -> EDP space.

    PCA has dimension (nAlt, nPCA); PCA_state has dimension (nPCA, ...);
    the returned density has dimension (nAlt, ...). ``mean`` (nAlt,), if
    given, is added back after projecting -- it must be the same mean
    ``get_PCA`` subtracted when fitting ``PCA``, or reconstruction will be
    missing the ensemble's mean profile entirely.
    """
    PCA_state = np.asarray(PCA_state, dtype=float)
    PCA = np.asarray(PCA, dtype=float)
    PCA_state_shape = PCA_state.shape
    PCA_shape = PCA.shape
    if PCA_state_shape[0] != PCA_shape[1]:
        raise ValueError("PCA_state and PCA dimensions are inconsistent.")

    flat_state = PCA_state.reshape(PCA_state_shape[0], -1)
    density = PCA @ flat_state
    if mean is not None:
        density = density + np.asarray(mean, dtype=float).reshape(-1, 1)
    density = density.reshape([PCA_shape[0]] + list(PCA_state_shape[1:]))
    if not linear:
        density = 10.0 ** density
    return density


def EDP2PCA_1D(density: np.ndarray, PCA: np.ndarray, mean: np.ndarray | None = None,
               linear: bool = True, minlog10Density: float = 4.0) -> np.ndarray:
    """
    Project density onto the PCA coefficient space (inverse of PCA2EDP_1D).

    ``mean`` (nAlt,), if given, is subtracted before projecting -- must
    match the mean ``get_PCA`` used when fitting ``PCA``.
    """
    density = np.asarray(density, dtype=float)
    PCA = np.asarray(PCA, dtype=float)
    density_shape = density.shape
    PCA_shape = PCA.shape
    if density_shape[0] != PCA_shape[0]:
        raise ValueError("density and PCA dimensions are inconsistent.")

    flat_density = density.reshape(density_shape[0], -1)
    if not linear:
        floor = 10.0 ** minlog10Density
        flat_density = np.log10(np.where(flat_density < floor, floor, flat_density))
    if mean is not None:
        flat_density = flat_density - np.asarray(mean, dtype=float).reshape(-1, 1)

    PCA_state = PCA.T @ flat_density
    PCA_state = PCA_state.reshape([PCA_shape[1]] + list(density_shape[1:]))
    return PCA_state


def PCA2EDP_1D_map(PCA_state: np.ndarray, PCA: np.ndarray, mean: np.ndarray | None = None,
                    linear: bool = True, minlog10Density: float = 4.0) -> np.ndarray:
    """
    Jacobian of ``PCA2EDP_1D`` w.r.t. ``PCA_state``.

    Shape (nAlt, nPCA, ...), matching ``PCA_state.shape[1:]`` in the
    trailing axes. For ``linear=True`` the Jacobian is exactly ``PCA``,
    shared identically across every point/sample (broadcast, not
    materialized as a dense block per point) -- adding ``mean`` back in
    ``PCA2EDP_1D`` doesn't change this, since d(mean)/d(PCA_state) = 0. For
    ``linear=False`` (density = 10**(PCA @ PCA_state + mean)) it
    additionally scales by ``ln(10) * density`` at each point, since the
    map is then nonlinear.
    """
    PCA_state = np.asarray(PCA_state, dtype=float)
    PCA = np.asarray(PCA, dtype=float)
    PCA_state_shape = PCA_state.shape
    PCA_shape = PCA.shape
    if PCA_state_shape[0] != PCA_shape[1]:
        raise ValueError("PCA_state and PCA dimensions are inconsistent.")

    tail_shape = PCA_state_shape[1:]
    idx = (slice(None), slice(None)) + (np.newaxis,) * len(tail_shape)
    jac = np.broadcast_to(PCA[idx], PCA_shape + tail_shape).copy()

    if not linear:
        density = PCA2EDP_1D(PCA_state, PCA, mean=mean, linear=False, minlog10Density=minlog10Density)
        jac = np.log(10.0) * jac * density[:, np.newaxis, ...]
    return jac


def PCA2EDP_3D(PCA_state: np.ndarray, PCA: np.ndarray, mean: np.ndarray | None = None,
               linear: bool = True, minlog10Density: float = 4.0) -> np.ndarray:
    """
    Coefficient space -> EDP space, joint (altitude, geolocation) basis.

    PCA has dimension (nAlt, nGeo, nPCA); PCA_state has dimension
    (nPCA, nSample); the returned density has dimension (nAlt, nGeo, nSample).
    ``mean`` (nAlt, nGeo) or (nAlt*nGeo,), if given, is added back after
    projecting -- must match the mean ``get_PCA`` used when fitting ``PCA``.
    """
    PCA_state = np.asarray(PCA_state, dtype=float)
    PCA = np.asarray(PCA, dtype=float)
    PCA_state_shape = PCA_state.shape
    PCA_shape = PCA.shape
    if PCA_state_shape[0] != PCA_shape[2]:
        raise ValueError("PCA_state and PCA dimensions are inconsistent.")

    PCA_flat = PCA.reshape(PCA_shape[0] * PCA_shape[1], PCA_shape[2])
    flat_state = PCA_state.reshape(PCA_state_shape[0], -1)
    density = PCA_flat @ flat_state
    if mean is not None:
        density = density + np.asarray(mean, dtype=float).reshape(-1, 1)
    density = density.reshape([PCA_shape[0], PCA_shape[1], flat_state.shape[1]])
    if not linear:
        density = 10.0 ** density
    return density


def EDP2PCA_3D(density: np.ndarray, PCA: np.ndarray, mean: np.ndarray | None = None,
               linear: bool = True, minlog10Density: float = 4.0) -> np.ndarray:
    """
    Project density onto the joint PCA coefficient space (inverse of PCA2EDP_3D).

    ``mean`` (nAlt, nGeo) or (nAlt*nGeo,), if given, is subtracted before
    projecting -- must match the mean ``get_PCA`` used when fitting ``PCA``.
    """
    density = np.asarray(density, dtype=float)
    PCA = np.asarray(PCA, dtype=float)
    density_shape = density.shape
    PCA_shape = PCA.shape
    if density_shape[0] != PCA_shape[0] or density_shape[1] != PCA_shape[1]:
        raise ValueError("density and PCA dimensions are inconsistent.")

    flat_density = density.reshape(density_shape[0] * density_shape[1], -1)
    if not linear:
        floor = 10.0 ** minlog10Density
        flat_density = np.log10(np.where(flat_density < floor, floor, flat_density))
    if mean is not None:
        flat_density = flat_density - np.asarray(mean, dtype=float).reshape(-1, 1)

    PCA_flat = PCA.reshape(PCA_shape[0] * PCA_shape[1], PCA_shape[2])
    PCA_state = PCA_flat.T @ flat_density
    PCA_state = PCA_state.reshape([PCA_shape[2], flat_density.shape[1]])
    return PCA_state


def PCA2EDP_3D_map(PCA_state: np.ndarray, PCA: np.ndarray, mean: np.ndarray | None = None,
                    linear: bool = True, minlog10Density: float = 4.0) -> np.ndarray:
    """
    Jacobian of ``PCA2EDP_3D`` w.r.t. ``PCA_state``.

    Shape (nAlt, nGeo, nPCA, nSample). Same broadcasting approach as
    ``PCA2EDP_1D_map``: for ``linear=True`` the Jacobian is ``PCA`` itself,
    tiled over the sample axis; for ``linear=False`` it additionally scales
    by ``ln(10) * density`` per sample.
    """
    PCA_state = np.asarray(PCA_state, dtype=float)
    PCA = np.asarray(PCA, dtype=float)
    PCA_state_shape = PCA_state.shape
    PCA_shape = PCA.shape
    if PCA_state_shape[0] != PCA_shape[2]:
        raise ValueError("PCA_state and PCA dimensions are inconsistent.")

    nSample = PCA_state_shape[1] if PCA_state.ndim > 1 else 1
    jac = np.broadcast_to(PCA[..., np.newaxis], PCA_shape + (nSample,)).copy()

    if not linear:
        density = PCA2EDP_3D(PCA_state, PCA, mean=mean, linear=False, minlog10Density=minlog10Density)
        jac = np.log(10.0) * jac * density[:, :, np.newaxis, :]
    return jac


# ===========================================================================
# Registry: bundle each style's three operations into one place.
# ===========================================================================

Parameterization_Style = Literal[
    'raw', 'density_10ex', 'ANCHOR', 'PCA_1D', 'PCA_3D', 'PCA_1D_10ex', 'PCA_3D_10ex',
]


@dataclass
class ParameterizationSpec:
    """
    Bundles the three operations a parameterization style must provide, so
    that ``get_density``/``get_parameter``/``get_Jacobian`` can never drift
    out of sync with each other the way three separate ``match`` blocks can.
    """
    to_density: Callable[[np.ndarray, np.ndarray | None], np.ndarray]
    to_parameter: Callable[[np.ndarray, np.ndarray | None], np.ndarray]
    jacobian: Callable[[np.ndarray, np.ndarray | None], tuple[str, np.ndarray]]
    needs_altitude: bool = False
    is_lossy: bool = False   # True for styles where get_density(get_parameter(x)) != x


class EDP_Parameterization:
    """
    A parameterization style bundled with its hyperparameters.

    Provides ``get_density``/``get_parameter``/``get_Jacobian`` for the
    configured ``style``, plus ``reconstruction_error``/``plot_pca_spectrum``
    diagnostics that quantify the approximation error the parameterization
    itself introduces (PCA truncation error, ANCHOR fit error).
    """

    def __init__(self, style: Parameterization_Style = 'density_10ex',
                 hyper_params: dict | None = None):
        self.style = style
        self.hyper_params = self._init_hyper_params(style, hyper_params)
        self._spec = self._build_spec(style, self.hyper_params)

    @property
    def needs_altitude(self) -> bool:
        return self._spec.needs_altitude

    @property
    def is_lossy(self) -> bool:
        return self._spec.is_lossy

    @staticmethod
    def _init_hyper_params(style: str, hyper_params: dict | None) -> dict:
        # For styles with defaults, merge rather than replace: an empty {}
        # (not just None) should still pick up defaults, and a caller
        # supplying only some keys (e.g. one overridden ANCHOR bound) should
        # get the rest filled in rather than hitting a KeyError later.
        if style == 'raw':
            return dict(hyper_params) if hyper_params else {}
        if style == 'ANCHOR':
            merged = dict(ANCHOR_DEFAULT_BOUNDS)
            if hyper_params:
                merged.update(hyper_params)
            return merged
        if style == 'density_10ex':
            merged = {'minlog10Density': 4.0}
            if hyper_params:
                merged.update(hyper_params)
            return merged
        if style in ('PCA_1D', 'PCA_1D_10ex'):
            if hyper_params is None or 'PCA' not in hyper_params:
                raise ValueError(f"Parameterization {style} requires 'PCA' in hyper_params.")
            if np.asarray(hyper_params['PCA']).ndim != 2:
                raise ValueError(f"Parameterization {style} requires PCA to be 2-D (nAlt, nPCA).")
            return dict(hyper_params)
        if style in ('PCA_3D', 'PCA_3D_10ex'):
            if hyper_params is None or 'PCA' not in hyper_params:
                raise ValueError(f"Parameterization {style} requires 'PCA' in hyper_params.")
            if np.asarray(hyper_params['PCA']).ndim != 3:
                raise ValueError(f"Parameterization {style} requires PCA to be 3-D (nAlt, nGeo, nPCA).")
            return dict(hyper_params)
        raise ValueError(f"Parameterization style '{style}' is not defined.")

    @staticmethod
    def _build_spec(style: str, hyper_params: dict) -> ParameterizationSpec:
        if style == 'raw':
            return ParameterizationSpec(
                to_density=lambda p, alt: p,
                to_parameter=lambda d, alt: d,
                jacobian=lambda p, alt: ('diagonal', np.ones_like(p)),
            )
        if style == 'density_10ex':
            mld = hyper_params['minlog10Density']
            return ParameterizationSpec(
                to_density=lambda p, alt: density_10ex(p, minlog10Density=mld),
                to_parameter=lambda d, alt: log10_param(d, minlog10Density=mld),
                jacobian=lambda p, alt: ('diagonal', Jacobian_density_10ex(p, minlog10Density=mld)),
            )
        if style == 'ANCHOR':
            def _jacobian(p, alt):
                _, dNe_dP = _ne_profile_derivatives(alt, p, partial=True)
                return ('per_point', dNe_dP)
            return ParameterizationSpec(
                to_density=lambda p, alt: _ne_profile_derivatives(alt, p, partial=False),
                to_parameter=lambda d, alt: _fit_iri_params_ensemble(d, alt, hyper_params),
                jacobian=_jacobian,
                needs_altitude=True,
            )
        if style == 'PCA_1D':
            PCA = hyper_params['PCA']
            mean = hyper_params.get('PCA_mean')
            return ParameterizationSpec(
                to_density=lambda p, alt: PCA2EDP_1D(p, PCA, mean=mean, linear=True),
                to_parameter=lambda d, alt: EDP2PCA_1D(d, PCA, mean=mean, linear=True),
                jacobian=lambda p, alt: ('block_diagonal', PCA2EDP_1D_map(p, PCA, mean=mean, linear=True)),
                is_lossy=True,
            )
        if style == 'PCA_1D_10ex':
            PCA = hyper_params['PCA']
            mean = hyper_params.get('PCA_mean')
            return ParameterizationSpec(
                to_density=lambda p, alt: PCA2EDP_1D(p, PCA, mean=mean, linear=False),
                to_parameter=lambda d, alt: EDP2PCA_1D(d, PCA, mean=mean, linear=False),
                jacobian=lambda p, alt: ('full', PCA2EDP_1D_map(p, PCA, mean=mean, linear=False)),
                is_lossy=True,
            )
        if style == 'PCA_3D':
            PCA = hyper_params['PCA']
            mean = hyper_params.get('PCA_mean')
            return ParameterizationSpec(
                to_density=lambda p, alt: PCA2EDP_3D(p, PCA, mean=mean, linear=True),
                to_parameter=lambda d, alt: EDP2PCA_3D(d, PCA, mean=mean, linear=True),
                jacobian=lambda p, alt: ('full', PCA2EDP_3D_map(p, PCA, mean=mean, linear=True)),
                is_lossy=True,
            )
        if style == 'PCA_3D_10ex':
            PCA = hyper_params['PCA']
            mean = hyper_params.get('PCA_mean')
            return ParameterizationSpec(
                to_density=lambda p, alt: PCA2EDP_3D(p, PCA, mean=mean, linear=False),
                to_parameter=lambda d, alt: EDP2PCA_3D(d, PCA, mean=mean, linear=False),
                jacobian=lambda p, alt: ('full', PCA2EDP_3D_map(p, PCA, mean=mean, linear=False)),
                is_lossy=True,
            )
        raise ValueError(f"Parameterization style '{style}' is not defined.")

    def get_density(self, param_vec: np.ndarray, alt: np.ndarray | None = None) -> np.ndarray:
        a = alt if self._spec.needs_altitude else None
        return self._spec.to_density(np.asarray(param_vec, dtype=float), a)

    def get_parameter(self, density: np.ndarray, alt: np.ndarray | None = None) -> np.ndarray:
        a = alt if self._spec.needs_altitude else None
        return self._spec.to_parameter(np.asarray(density, dtype=float), a)

    def get_Jacobian(self, param_vec: np.ndarray, alt: np.ndarray | None = None) -> tuple[str, np.ndarray]:
        a = alt if self._spec.needs_altitude else None
        return self._spec.jacobian(np.asarray(param_vec, dtype=float), a)

    # -- Diagnostics: how much error does this parameterization introduce? ----

    def reconstruction_error(self, density: np.ndarray, alt: np.ndarray | None = None,
                              in_log10: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode ``density`` then decode it back, and report what's left over.

        This one computation *is* the diagnostic for every style:
        ``'density_10ex'``'s residual should be ~0 off the clamp floor (a
        sanity check on the round trip itself); ``'ANCHOR'``'s residual is
        the analytic model's fit error against the true profile; the
        ``PCA_*`` styles' residual is the truncation error from discarding
        modes beyond ``retaining_threshold``.

        Returns
        -------
        residual : density - density_hat (or log10-space difference if
            ``in_log10``), same shape as ``density``.
        density_hat : the reconstructed density.
        """
        density = np.asarray(density, dtype=float)
        param_vec = self.get_parameter(density, alt=alt)
        density_hat = np.asarray(self.get_density(param_vec, alt=alt), dtype=float)
        if in_log10:
            mld = self.hyper_params.get('minlog10Density', 4.0)
            floor = 10.0 ** mld
            residual = (np.log10(np.maximum(density, floor))
                        - np.log10(np.maximum(density_hat, floor)))
        else:
            residual = density - density_hat
        return residual, density_hat

    def plot_pca_spectrum(self, ax=None):
        """
        Scree plot for ``PCA_*`` styles: singular-value spectrum plus
        cumulative variance explained, with the ``retaining_threshold``
        cutoff marked. Requires PCA diagnostics to be attached to
        ``self.hyper_params['pca_diagnostics']`` -- normally arranged by
        ``Parameterized_EDPSamples`` when it fits a new PCA basis.
        """
        import matplotlib.pyplot as plt

        diag: PCADiagnostics | None = self.hyper_params.get('pca_diagnostics')
        if diag is None:
            raise ValueError(
                "No PCA diagnostics available for this EDP_Parameterization. "
                "Construct it via Parameterized_EDPSamples (which fits a new "
                "PCA basis and attaches diagnostics), or supply "
                "hyper_params['pca_diagnostics'] explicitly."
            )

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        else:
            axes = ax
            fig = axes[0].figure

        ax0, ax1 = axes
        n = len(diag.singular_values)
        comp = np.arange(1, n + 1)
        # Numerically-zero singular values (rank-deficient ensembles) can't
        # be log-scaled; drop them from the semilog view rather than warn.
        sv_plot = np.where(diag.singular_values > 0, diag.singular_values, np.nan)

        ax0.semilogy(comp, sv_plot, marker='o', ms=3)
        ax0.axvline(diag.n_retained + 0.5, color='C3', linestyle='--',
                    label=f'Retained: {diag.n_retained}')
        ax0.set_xlabel('Component')
        ax0.set_ylabel('Singular value')
        ax0.set_title('PCA spectrum')
        ax0.legend(loc='best')
        ax0.grid(True, alpha=0.4, linestyle=':')

        ax1.plot(comp, diag.cumulative_variance_ratio, marker='o', ms=3)
        ax1.axhline(diag.retaining_threshold, color='gray', linestyle=':',
                    label=f'Threshold: {diag.retaining_threshold:.3f}')
        ax1.axvline(diag.n_retained + 0.5, color='C3', linestyle='--',
                    label=f'Retained: {diag.n_retained}')
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Cumulative variance explained')
        ax1.set_ylim(0, 1.02)
        ax1.set_title('Truncation choice')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.4, linestyle=':')

        fig.tight_layout()
        return axes


# ===========================================================================
# Parameterized_EDPSamples: wires a parameterization to an EDPSamples dataset.
# ===========================================================================

class Parameterized_EDPSamples:
    """
    An ``EDPSamples`` dataset augmented with a parameterization's
    ``param_vec`` (and, for PCA styles, the fitted ``PCA`` basis).

    Does not mutate the ``EDPSamples`` passed in (works on a shallow copy),
    so the caller's original object is unaffected.
    """
    __slots__ = ('EDPSamples', 'style', 'Parameterization')

    _PARAM_DIMS = {
        'raw': lambda E: (E.DIM_HEIGHT, E.DIM_GEO, E.DIM_SAMPLE),
        'density_10ex': lambda E: (E.DIM_HEIGHT, E.DIM_GEO, E.DIM_SAMPLE),
        'ANCHOR': lambda E: ('anchor_param', E.DIM_GEO, E.DIM_SAMPLE),
        'PCA_1D': lambda E: ('nPCA', E.DIM_GEO, E.DIM_SAMPLE),
        'PCA_1D_10ex': lambda E: ('nPCA', E.DIM_GEO, E.DIM_SAMPLE),
        'PCA_3D': lambda E: ('nPCA', E.DIM_SAMPLE),
        'PCA_3D_10ex': lambda E: ('nPCA', E.DIM_SAMPLE),
    }

    def __init__(self, EDP: EDPSamples, style: Parameterization_Style = 'density_10ex',
                 hyper_params: dict | None = None):
        hyper_params = dict(hyper_params) if hyper_params is not None else {}
        is_pca = style in ('PCA_1D', 'PCA_1D_10ex', 'PCA_3D', 'PCA_3D_10ex')
        if is_pca and 'retaining_threshold' not in hyper_params and 'PCA' not in hyper_params:
            raise ValueError(
                f"Parameterization style '{style}' requires either "
                "'retaining_threshold' (to fit a new PCA basis from this "
                "dataset) or a pre-fit 'PCA' matrix in hyper_params."
            )

        edp = EDP.copy(deep=False)
        edps = edp.edps
        altitude = edp.altitude
        n_height, n_geo, n_sample = edps.shape

        new_vars: dict = {}
        if is_pca:
            is_3d = style.startswith('PCA_3D')
            is_log = style.endswith('_10ex')
            if 'PCA' in hyper_params:
                PCA = hyper_params['PCA']
                mean = hyper_params.get('PCA_mean')
            else:
                if is_3d:
                    flat = edps.reshape(n_height * n_geo, n_sample)
                else:
                    flat = edps.reshape(n_height, -1)
                if is_log:
                    mld = hyper_params.get('minlog10Density', 4.0)
                    floor = 10.0 ** mld
                    flat = np.log10(np.where(flat < floor, floor, flat))
                PCA_flat, mean_flat, pca_diag = get_PCA(flat, hyper_params['retaining_threshold'])
                PCA = PCA_flat.reshape(n_height, n_geo, -1) if is_3d else PCA_flat
                mean = mean_flat.reshape(n_height, n_geo) if is_3d else mean_flat
                hyper_params['pca_diagnostics'] = pca_diag
            hyper_params['PCA'] = PCA
            hyper_params['PCA_mean'] = mean
            pca_dims = ((EDPSamples.DIM_HEIGHT, EDPSamples.DIM_GEO, 'nPCA') if is_3d
                        else (EDPSamples.DIM_HEIGHT, 'nPCA'))
            mean_dims = ((EDPSamples.DIM_HEIGHT, EDPSamples.DIM_GEO) if is_3d
                         else (EDPSamples.DIM_HEIGHT,))
            new_vars['PCA'] = (pca_dims, PCA)
            if mean is not None:
                new_vars['PCA_mean'] = (mean_dims, mean)

        parameterization = EDP_Parameterization(style=style, hyper_params=hyper_params)
        alt_arg = altitude if parameterization.needs_altitude else None
        param_vec = parameterization.get_parameter(edps, alt=alt_arg)

        param_dims = self._PARAM_DIMS[style](EDPSamples)
        new_vars['param_vec'] = (
            param_dims, param_vec,
            {"long_name": "Parameter values",
             "description": f"{style} parameterization of the electron density profiles"},
        )

        edp = edp.assign(new_vars)
        if style == 'ANCHOR':
            edp = edp.assign_coords({'anchor_param': list(ANCHOR_PARAM_LABELS)})

        edp.attrs['Parameterization_style'] = style
        for key, value in hyper_params.items():
            if key not in ('PCA', 'PCA_mean', 'pca_diagnostics'):
                edp.attrs[f'Parameterization_{key}'] = value

        self.EDPSamples = edp
        self.style = style
        self.Parameterization = parameterization

    @property
    def _alt_arg(self) -> np.ndarray | None:
        return self.EDPSamples.altitude if self.Parameterization.needs_altitude else None

    @classmethod
    def fromNetCDF(cls, path: str, **kwargs: Any) -> "Parameterized_EDPSamples":
        """
        Load a ``Parameterized_EDPSamples`` from NetCDF (e.g. written by
        :meth:`saveNetCDF`).

        Note: only the truncated PCA basis is persisted, not the full
        singular-value spectrum, so ``plot_pca_spectrum``/``error_summary``'s
        component-count diagnostics are unavailable after a round trip
        unless the PCA basis is refit.
        """
        with xr.open_dataset(path, **kwargs) as ds:
            ds.load()
            return Parameterized_EDPSamples.from_xarray(ds)

    @classmethod
    def from_xarray(cls, ds) -> "Parameterized_EDPSamples":
        if 'Parameterization_style' not in ds.attrs:
            raise ValueError(
                "The dataset does not contain a saved Parameterized_EDPSamples "
                "('Parameterization_style' attribute missing)."
            )
        style = ds.attrs['Parameterization_style']

        EDPSam = EDPSamples.from_xarray(ds)

        hyper_params: dict = {}
        prefix = 'Parameterization_'
        for key, value in ds.attrs.items():
            if key.startswith(prefix) and key != 'Parameterization_style':
                hyper_params[key[len(prefix):]] = value

        if 'PCA' in ds.data_vars:
            hyper_params['PCA'] = ds.data_vars['PCA'].to_numpy()
        if 'PCA_mean' in ds.data_vars:
            hyper_params['PCA_mean'] = ds.data_vars['PCA_mean'].to_numpy()

        return Parameterized_EDPSamples(EDPSam, style=style, hyper_params=hyper_params)

    def saveNetCDF(self, path: str, **kwargs: Any) -> None:
        """Write the wrapped ``EDPSamples`` (including param_vec/PCA) to NetCDF."""
        self.EDPSamples.saveNetCDF(path, **kwargs)

    # -- Diagnostics: how much error does this parameterization introduce? ----

    def reconstruction_error(self, in_log10: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruction residual/estimate for every (height, geo, sample)."""
        return self.Parameterization.reconstruction_error(
            self.EDPSamples.edps, alt=self._alt_arg, in_log10=in_log10)

    def error_summary(self, in_log10: bool = True) -> dict:
        """
        Numeric summary of the parameterization's approximation error:
        overall RMSE, RMSE vs. altitude, worst-case location, and (style-
        specific) region-broken-out RMSE for ANCHOR or retained-component
        count for the PCA styles.
        """
        residual, _ = self.reconstruction_error(in_log10=in_log10)
        altitude = self.EDPSamples.altitude

        rmse_overall = float(np.sqrt(np.nanmean(residual ** 2)))
        rmse_by_altitude = np.sqrt(np.nanmean(residual ** 2, axis=(1, 2)))
        abs_res = np.abs(residual)
        flat_idx = int(np.nanargmax(abs_res))
        max_idx = np.unravel_index(flat_idx, residual.shape)

        summary: dict = {
            'style': self.style,
            'in_log10': in_log10,
            'rmse_overall': rmse_overall,
            'rmse_by_altitude': rmse_by_altitude,
            'altitude': altitude,
            'max_abs_error': float(abs_res[max_idx]),
            'max_abs_error_index': {
                'altitude_idx': int(max_idx[0]),
                'geo_idx': int(max_idx[1]),
                'sample_idx': int(max_idx[2]),
            },
        }

        if self.style == 'ANCHOR':
            summary['rmse_by_region'] = self._anchor_rmse_by_region(residual)

        if self.style in ('PCA_1D', 'PCA_1D_10ex', 'PCA_3D', 'PCA_3D_10ex'):
            diag: PCADiagnostics | None = self.Parameterization.hyper_params.get('pca_diagnostics')
            if diag is not None:
                summary['n_retained_components'] = diag.n_retained
                summary['cumulative_variance_explained'] = float(
                    diag.cumulative_variance_ratio[diag.n_retained - 1])

        return summary

    def _anchor_rmse_by_region(self, residual: np.ndarray) -> dict:
        altitude = self.EDPSamples.altitude
        param_vec = self.EDPSamples['param_vec'].to_numpy()
        hmE, h_ST, hmF2 = anchor_region_boundaries(param_vec)

        h = altitude[:, None, None]
        mask_top = h >= hmF2[None, :, :]
        mask_bot = (h >= h_ST[None, :, :]) & ~mask_top
        mask_int = (h >= hmE[None, :, :]) & (h < h_ST[None, :, :])
        mask_e = h < hmE[None, :, :]

        out = {}
        for name, mask in (('topside', mask_top), ('bottomside', mask_bot),
                            ('intermediate', mask_int), ('E_layer', mask_e)):
            vals = residual[mask]
            out[name] = float(np.sqrt(np.nanmean(vals ** 2))) if vals.size else float('nan')
        return out

    def plot_pca_spectrum(self, ax=None):
        """See ``EDP_Parameterization.plot_pca_spectrum``."""
        return self.Parameterization.plot_pca_spectrum(ax=ax)

    def plot_reconstruction_profile(self, sample_idx: int = 0, geo_idx: int = 0, ax=None):
        """
        Original vs. reconstructed density for one (sample, geo) profile,
        plus its residual. For ``'ANCHOR'``, the region boundaries (hmE,
        h_ST, hmF2) are marked so you can see where the fit struggles.
        """
        import matplotlib.pyplot as plt

        altitude = self.EDPSamples.altitude
        edps = self.EDPSamples.edps

        if self.style in ('PCA_3D', 'PCA_3D_10ex'):
            # PCA_3D's basis mixes altitude and geolocation jointly, so a
            # single profile can't be encoded/decoded on its own -- run the
            # whole spatial field for this sample, then slice one geo point.
            density_block = edps[:, :, sample_idx:sample_idx + 1]
            residual_block, density_hat_block = self.Parameterization.reconstruction_error(density_block)
            density = density_block[:, geo_idx, 0]
            density_hat = density_hat_block[:, geo_idx, 0]
            residual = residual_block[:, geo_idx, 0]
        else:
            density = edps[:, geo_idx, sample_idx]
            residual, density_hat = self.Parameterization.reconstruction_error(
                density, alt=self._alt_arg)

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
        else:
            axes = ax
            fig = axes[0].figure
        ax0, ax1 = axes

        ax0.plot(density, altitude, label='Original EDP', color='black', lw=1.5)
        ax0.plot(density_hat, altitude, label=f'{self.style} reconstruction',
                 color='C1', lw=1.5, linestyle='--')
        ax0.set_xlabel('Electron Density (m$^{-3}$)')
        ax0.set_ylabel('Altitude (km)')
        ax0.set_title(f'Sample {sample_idx}, Geo {geo_idx}')
        ax0.legend(loc='best')
        ax0.grid(True, alpha=0.4, linestyle=':')

        ax1.plot(residual, altitude, color='C3')
        ax1.axvline(0, color='black', lw=0.8)
        ax1.set_xlabel('Residual (Original - Reconstructed)')
        ax1.set_title('Reconstruction error')
        ax1.grid(True, alpha=0.4, linestyle=':')

        if self.style == 'ANCHOR':
            param_vec = self.EDPSamples['param_vec'].to_numpy()[:, geo_idx, sample_idx]
            hmE, h_ST, hmF2 = anchor_region_boundaries(param_vec)
            for a in (ax0, ax1):
                for h_val, label in ((hmE, 'hmE'), (h_ST, 'h_ST'), (hmF2, 'hmF2')):
                    a.axhline(h_val, color='gray', lw=0.8, linestyle=':')
            ax0.text(0.02, 0.02, 'gray lines: hmE / h_ST / hmF2',
                     transform=ax0.transAxes, fontsize=8, color='gray')

        fig.tight_layout()
        return axes

    def plot_reconstruction_error_statistics(self, ax=None):
        """
        Percentile envelope of reconstruction error vs. altitude across the
        whole ensemble -- the error analogue of ``EDPSamples.
        plot_edp_statistics``. Left panel: absolute error; right panel:
        error as a percentage of the local |EDP| (more comparable across
        altitude than the absolute value, since EDP spans many orders of
        magnitude).
        """
        import matplotlib.pyplot as plt

        altitude = self.EDPSamples.altitude
        edps = self.EDPSamples.edps
        residual, _ = self.reconstruction_error(in_log10=False)

        safe_edps = np.where(np.abs(edps) > 0, np.abs(edps), np.nan)
        rel_residual = residual / safe_edps * 100.0

        pct_levels = [1, 5, 16, 50, 84, 95, 99]
        p01, p05, p16, med, p84, p95, p99 = np.nanpercentile(residual, pct_levels, axis=(1, 2))
        rp01, rp05, rp16, rmed, rp84, rp95, rp99 = np.nanpercentile(rel_residual, pct_levels, axis=(1, 2))

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
        else:
            axes = ax
            fig = axes[0].figure
        ax0, ax1 = axes

        ax0.axvline(0, color='black', lw=1)
        ax0.fill_betweenx(altitude, p01, p99, color='lightcoral', alpha=0.3, label='1st-99th %ile')
        ax0.fill_betweenx(altitude, p05, p95, color='indianred', alpha=0.5, label='5th-95th %ile')
        ax0.fill_betweenx(altitude, p16, p84, color='darkred', alpha=0.7, label='16th-84th %ile')
        ax0.plot(med, altitude, color='black', lw=1.5, label='Median')
        ax0.set_xlabel('Reconstruction error (m$^{-3}$)')
        ax0.set_ylabel('Altitude (km)')
        ax0.set_title(f'{self.style}: absolute error')
        ax0.legend(loc='best')
        ax0.grid(True, alpha=0.4, linestyle=':')

        ax1.axvline(0, color='black', lw=1)
        ax1.fill_betweenx(altitude, rp01, rp99, color='lightblue', alpha=0.3, label='1st-99th %ile')
        ax1.fill_betweenx(altitude, rp05, rp95, color='dodgerblue', alpha=0.5, label='5th-95th %ile')
        ax1.fill_betweenx(altitude, rp16, rp84, color='blue', alpha=0.7, label='16th-84th %ile')
        ax1.plot(rmed, altitude, color='black', lw=1.5, label='Median')
        ax1.set_xlabel('Reconstruction error (% of |EDP|)')
        ax1.set_title(f'{self.style}: relative error')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.4, linestyle=':')

        fig.tight_layout()
        return axes
