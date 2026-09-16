#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameterizations of electron density profile
This module defines a class of objects that are functions used for parameterize
a vertical electrondensity profile.  
Created on Wed Sep  9 08:18:22 2026

@author: cwang
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import warnings
from scipy.optimize import minimize
from edp_samples import EDPSamples
from pathlib import Path
from typing import Literal, Any

def density_10ex(log10_density:np.array,minlog10Density:float=4)->np.array:
    idx = np.where(log10_density<minlog10Density)
    log10_density[idx] = minlog10Density
    density = 10**log10_density
    return density

def Jacobian_density_10ex(log10_density:np.array,minlog10Density:float=4)->np.array:
    idx = np.where(log10_density<minlog10Density)
    log10_density[idx] = minlog10Density
    density = 10**log10_density
    Jacobian = np.log(10)*density
    # Jacobian is a diagonal matrix so only the diagonal is returned.
    return Jacobian

def log10_param(density:np.array,minlog10Density:float=4)->np.array:
    idx = np.where(density<10**minlog10Density)
    density[idx] = 10**minlog10Density
    log10_density = np.log10(density)
    return log10_density

def _ne_profile_derivatives(
    alts_km: np.ndarray,
    params_lin: np.ndarray,
    partial: bool =True,
) -> tuple:
    """
    Compute Ne and analytical ∂Ne(h)/∂P_k for each altitude and each state parameter.

    Treats h_ST (the F2/E-layer transition altitude from bisection) as a
    fixed boundary when computing region derivatives — the standard
    piecewise-smooth approximation: boundaries have measure zero and don't
    affect integral derivatives.

    Parameters are in LINEAR density space (NmF2, NmE in m⁻³).  The
    returned derivatives are w.r.t. the stored log10 parameters at indices
    I_LOG_NMF2 and I_LOG_NME, with the ln(10)·Nm chain-rule factor applied.

    Parameters
    ----------
    alts_km    : (n_alt,)  altitude sample points in km
    params_lin : (N_STATE,)  parameters in linear density space

    Returns
    -------
    Ne      : (n_alt,)          electron density profile (m⁻³)
    dNe_dP  : (N_STATE, n_alt)  ∂Ne/∂P_k at every altitude
    """
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
    
    # ── Helper: bottomside Ne and log-derivative quantities at h_eff ──────────
    def _bs_quantities(h_eff_arr):
        """Return (Ne_bs, log_df, x, x_pow, log_x) for given effective altitudes."""
        x = np.maximum((hmF2 - h_eff_arr) / (B0 + 1e-9), 0.0)
        x_pow  = np.where(x > 1e-30, x, 1e-30)
        cosh_x = np.cosh(np.clip(x, 0.0, 700.0))
        tanh_x = np.tanh(np.clip(x, 0.0, 700.0))
        Ne_bs  = NmF2 * np.exp(-(x_pow ** B1)) / cosh_x
        # d/dx[log f(x)], f(x) = exp(-x^B1)/cosh(x)
        log_df = np.where(x > 1e-30, -B1 * x_pow ** (B1 - 1.0) - tanh_x, -tanh_x)
        log_x  = np.log(x_pow)
        return Ne_bs, log_df, x, x_pow, log_x

    # Topside shape parameter (IRI default)
    _R_TOPSIDE: float = 100.0
    # E-layer scale height (km)
    _H_E_KM: float = 15.0
    # 1 TECU = 1e16 m^-2
    # _TECU: float = 1.0e16
    
    I_LOG_NMF2 = 0
    I_HMF2     = 1
    I_H0       = 2
    I_GAMMA    = 3
    I_B0       = 4
    I_B1       = 5
    I_LOG_NME  = 6
    I_HME      = 7
    N_STATE    = 8
    
    h     = alts_km                          # (n_alt,)
    n_alt = len(h)
    ndim = params_lin.ndim
    params_shape = params_lin.shape
    if not params_shape[0] == N_STATE:
        raise ValueError(f"Dimension of Parameter is {ndim[0]} not equal to 8.")
        
    params_lin = params_lin.reshape(N_STATE,-1)
    ns,nPts = params_lin.shape
    Ne_ensemble = np.zeros((n_alt,nPts),dtype=float)
    if partial:
        dNe_dP_ensemble = np.zeros((n_alt, N_STATE, nPts), dtype=float)
    for iPts in range(nPts):
        if nPts == 1:
            NmF2  = float(params_lin[I_LOG_NMF2])
            hmF2  = float(params_lin[I_HMF2])
            H0    = float(params_lin[I_H0])
            gamma = float(params_lin[I_GAMMA])
            B0    = float(params_lin[I_B0])
            B1    = float(params_lin[I_B1])
            NmE   = float(params_lin[I_LOG_NME])
            hmE   = float(params_lin[I_HME])
        else:
            NmF2  = float(params_lin[I_LOG_NMF2,iPts])
            hmF2  = float(params_lin[I_HMF2,iPts])
            H0    = float(params_lin[I_H0,iPts])
            gamma = float(params_lin[I_GAMMA,iPts])
            B0    = float(params_lin[I_B0,iPts])
            B1    = float(params_lin[I_B1,iPts])
            NmE   = float(params_lin[I_LOG_NME,iPts])
            hmE   = float(params_lin[I_HME,iPts])

    
        # ── Region boundaries ─────────────────────────────────────────────────────
        h_ST = float(np.clip(
            _find_hst_bisection(
                np.array([NmF2]), np.array([hmF2]),
                np.array([B0]),   np.array([B1]),   np.array([NmE]),
            )[0],
            hmE, hmF2,
        ))
    
        mask_top = h >= hmF2
        mask_bot = (h >= h_ST) & ~mask_top
        mask_int = (h >= hmE)  & (h < h_ST)
        mask_e   =  h <  hmE
    
    
        # ── Build composite Ne profile ────────────────────────────────────────────
        r = _R_TOPSIDE
        # Topside
        dh_all  = h - hmF2
        D_all   = r * H0 + gamma * dh_all + 1e-9
        H_all   = H0 * (1.0 + r * gamma * dh_all / D_all)
        z_all   = dh_all / (H_all + 1e-9)
        exp_z   = np.exp(np.clip(z_all, -80, 80))
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
    
        # E-layer — bottomside alpha-Chapman
        ze_all     = np.clip((h - hmE) / _H_E_KM, -80, 80)
        exp_neg_ze = np.exp(-ze_all)
        Ne_E_v     = NmE * np.exp(0.5 * (1.0 - ze_all - exp_neg_ze))
    
        Ne = np.where(
            mask_top, Ne_top_v,
            np.where(mask_bot, Ne_bs_bot,
            np.where(mask_int, Ne_bs_int, Ne_E_v)),
        )
        Ne = np.maximum(Ne, 0.0)
        if  not partial:
            Ne_ensemble[:,iPts] = Ne
            continue

        dNe_dP = np.zeros((N_STATE, n_alt), dtype=float)
        # ── Topside derivatives (h >= hmF2) ───────────────────────────────────────
        if mask_top.any():
            dh_t  = dh_all[mask_top]
            D_t   = D_all[mask_top]
            N_num = r * H0 + gamma * (1.0 + r) * dh_t  # numerator factor of H_top/H0
            H_t   = H_all[mask_top]
            sig_t = exp_z[mask_top] / (1.0 + exp_z[mask_top]) ** 2  # = Ne_top / (4*NmF2)
            dsig_t = sig_t * (1.0 - exp_z[mask_top]) / (1.0 + exp_z[mask_top])
    
            Ne_t = Ne_top_v[mask_top]
    
            dNe_dP[I_LOG_NMF2, mask_top] = Ne_t * np.log(10.0)
    
            # ∂/∂hmF2: 4·NmF2·dsig·dz/dhmF2
            # dz/dhmF2 = (−H_t + dh_t·r²γH0²/D_t²) / H_t²
            dHt_ddh  = r ** 2 * gamma * H0 ** 2 / D_t ** 2
            dz_hmF2  = (-H_t + dh_t * dHt_ddh) / (H_t + 1e-9) ** 2
            dNe_dP[I_HMF2, mask_top] = 4.0 * NmF2 * dsig_t * dz_hmF2
    
            # ∂/∂H0: 4·NmF2·dsig·dz/dH0
            # ∂H_t/∂H0 = N_num/D_t − r²·H0·γ·dh_t/D_t²
            dHt_H0 = N_num / D_t - r ** 2 * H0 * gamma * dh_t / D_t ** 2
            dz_H0  = -dh_t * dHt_H0 / (H_t + 1e-9) ** 2
            dNe_dP[I_H0, mask_top] = 4.0 * NmF2 * dsig_t * dz_H0
    
            # ∂/∂gamma: 4·NmF2·dsig·dz/dgamma
            # ∂H_t/∂γ = r²·H0²·dh_t/D_t²
            dHt_gam = r ** 2 * H0 ** 2 * dh_t / D_t ** 2
            dz_gam  = -dh_t * dHt_gam / (H_t + 1e-9) ** 2
            dNe_dP[I_GAMMA, mask_top] = 4.0 * NmF2 * dsig_t * dz_gam
    
            # B0, B1, log10(NmE), hmE: 0 in topside → already zero
    
        # ── Pure bottomside derivatives (h_ST ≤ h < hmF2) ────────────────────────
        if mask_bot.any():
            h_b = h[mask_bot]
            Ne_b, ldf_b, x_b, xp_b, lx_b = _bs_quantities(h_b)
    
            dNe_dP[I_LOG_NMF2, mask_bot] = Ne_b * np.log(10.0)
            # ∂x/∂hmF2 = 1/B0
            dNe_dP[I_HMF2,     mask_bot] = Ne_b * ldf_b / (B0 + 1e-9)
            # ∂x/∂B0 = −x/B0
            dNe_dP[I_B0,       mask_bot] = Ne_b * ldf_b * (-x_b / (B0 + 1e-9))
            # ∂/∂B1: Ne·(−x_pow^B1·ln x_pow)
            dNe_dP[I_B1,       mask_bot] = Ne_b * (-xp_b ** B1 * lx_b)
    
        # ── Intermediate connection derivatives (hmE ≤ h < h_ST) ─────────────────
        #
        # Ne_inter = w·Ne_A + (1−w)·Ne_B,   w = 3t² − 2t³,  t = (h−hmE)/(h_ST−hmE)
        #   Ne_A = direct branch  (h_eff = h)        — same form as pure bottomside
        #   Ne_B = mirrored branch (h_eff = h_ST + hmE − h)
        #
        # ∂Ne_inter/∂θ = (∂w/∂θ)·(Ne_A−Ne_B) + w·∂Ne_A/∂θ + (1−w)·∂Ne_B/∂θ
        #
        # h_ST = hmF2 − x_ST·B0 where x_ST satisfies the bisection equation
        #   g(x_ST) = NmF2·exp(−x_ST^B1)/cosh(x_ST) − NmE = 0
        #
        # Implicit function theorem gives ∂x_ST/∂θ = −(∂g/∂θ)/(∂g/∂x_ST):
        #   ∂g/∂x_ST       = NmE · ldf(x_ST)          (ldf < 0)
        #   ∂g/∂hmF2       = 0  → ∂h_ST/∂hmF2 = 1
        #   ∂g/∂B0         = 0  → ∂h_ST/∂B0   = −x_ST
        #   ∂g/∂B1         = NmE·(−x_ST^B1·ln x_ST)
        #                  → ∂x_ST/∂B1 = x_ST^B1·ln(x_ST) / ldf(x_ST)
        #                  → ∂h_ST/∂B1 = −B0·∂x_ST/∂B1
        #   ∂g/∂NmF2       = NmE/NmF2
        #                  → ∂x_ST/∂NmF2 = −1/(NmF2·ldf(x_ST))
        #                  → ∂h_ST/∂NmF2 = B0/(NmF2·ldf(x_ST))  (negative)
        #   ∂g/∂NmE        = −1
        #                  → ∂x_ST/∂NmE  = 1/(NmE·ldf(x_ST))
        #                  → ∂h_ST/∂NmE  = −B0/(NmE·ldf(x_ST))  (positive)
        #   ∂g/∂hmE        = 0  → ∂h_ST/∂hmE = 0
        #
        # For Ne_B: x_eff = (hmF2 − h_eff)/B0 = (hmF2 − h_ST − hmE + h)/B0
        #   ∂x_eff/∂θ = (∂hmF2/∂θ − ∂h_ST/∂θ − ∂hmE/∂θ) / B0 − x_eff · ∂B0/∂θ / B0
        #
        # For w: t = (h−hmE)/(h_ST−hmE) = N/D.  For θ affecting h_ST only
        # (NmF2, B0, B1, NmE): ∂t/∂θ = −t·∂h_ST/∂θ / D.  For θ = hmE (∂h_ST/∂hmE=0
        # but hmE enters both N and D directly): ∂t/∂hmE = (t−1)/D.
        if mask_int.any():
            h_i     = h[mask_int]
            h_eff_i = h_ST + hmE - h_i    # mirrored altitude (= 2*HZ - h_i)
    
            # x_ST and its ldf — scalar quantities for this grid point
            x_ST_v   = float((hmF2 - h_ST) / (B0 + 1e-9))
            xST_pow  = max(x_ST_v, 1e-30)
            tanh_xST = float(np.tanh(min(x_ST_v, 700.0)))
            ldf_xST  = float((-B1 * xST_pow ** (B1 - 1.0) - tanh_xST)
                             if x_ST_v > 1e-30 else -tanh_xST)
            # Guard against numerical zero in ldf_xST (should be negative)
            safe_ldf_xST = ldf_xST if abs(ldf_xST) > 1e-30 else -1e-30
            log_xST  = float(np.log(max(xST_pow, 1e-30)))
            hst_b1_coeff = xST_pow ** B1 * log_xST / safe_ldf_xST   # ∂x_ST/∂B1
    
            # ── Branch A: direct (h_eff = h) — same form as pure bottomside ──────
            Ne_A, ldf_A, x_A, xp_A, lx_A = _bs_quantities(h_i)
            dNeA = np.zeros((N_STATE, len(h_i)))
            dNeA[I_LOG_NMF2] = Ne_A * np.log(10.0)
            dNeA[I_HMF2]     = Ne_A * ldf_A / (B0 + 1e-9)
            dNeA[I_B0]       = Ne_A * ldf_A * (-x_A / (B0 + 1e-9))
            dNeA[I_B1]       = Ne_A * (-xp_A ** B1 * lx_A)
            # I_LOG_NME, I_HME → 0 (Ne_A doesn't track h_ST or hmE)
    
            # ── Branch B: mirrored (h_eff = h_ST + hmE − h) — tracks h_ST(θ), hmE ─
            Ne_B, ldf_B, x_B, xp_B, lx_B = _bs_quantities(h_eff_i)
            dNeB = np.zeros((N_STATE, len(h_i)))
            # ∂x_eff/∂hmF2 = (1 − ∂h_ST/∂hmF2)/B0 = (1−1)/B0 = 0 → dNeB[I_HMF2] = 0
            dNeB[I_LOG_NMF2] = Ne_B * np.log(10.0) * (1.0 - ldf_B / safe_ldf_xST)
            dNeB[I_B0]       = Ne_B * ldf_B * (x_ST_v - x_B) / (B0 + 1e-9)
            dNeB[I_B1]       = Ne_B * (ldf_B * hst_b1_coeff - xp_B ** B1 * lx_B)
            dNeB[I_LOG_NME]  = Ne_B * np.log(10.0) * ldf_B / safe_ldf_xST
            dNeB[I_HME]      = Ne_B * ldf_B * (-1.0 / (B0 + 1e-9))
    
            # ── Smoothstep blend weight w(t), t = (h−hmE)/(h_ST−hmE) ─────────────
            D = h_ST - hmE + 1e-9
            t = np.clip((h_i - hmE) / D, 0.0, 1.0)
            w = 3.0 * t ** 2 - 2.0 * t ** 3
            dw_dt = 6.0 * t * (1.0 - t)
    
            dhST = np.zeros(N_STATE)
            dhST[I_HMF2]     = 1.0
            dhST[I_B0]       = -x_ST_v
            dhST[I_B1]       = -B0 * hst_b1_coeff
            dhST[I_LOG_NMF2] =  B0 * np.log(10.0) / safe_ldf_xST
            dhST[I_LOG_NME]  = -B0 * np.log(10.0) / safe_ldf_xST
            # dhST[I_HME] = dhST[I_H0] = dhST[I_GAMMA] = 0 (already zero)
    
            dt_dtheta = np.zeros((N_STATE, len(h_i)))
            for k in (I_LOG_NMF2, I_HMF2, I_B0, I_B1, I_LOG_NME):
                dt_dtheta[k] = -t * dhST[k] / D
            dt_dtheta[I_HME] = (t - 1.0) / D
            dw_dtheta = dw_dt * dt_dtheta
    
            # ── Combine: Ne_inter = w·Ne_A + (1−w)·Ne_B ──────────────────────────
            idx_int = np.where(mask_int)[0]
            dNe_dP[:, idx_int] = (
                dw_dtheta * (Ne_A - Ne_B)
                + w * dNeA
                + (1.0 - w) * dNeB
            )
    
        # ── E-layer derivatives (h < hmE) — bottomside alpha-Chapman ─────────────
        # Ne_E = NmE * exp(0.5*(1 - ze - exp(-ze))),  ze = (h - hmE)/H_E
        # d Ne/d(log10 NmE) = Ne * ln(10)                        (NmE enters linearly)
        # d Ne/d(hmE)        = Ne * 0.5*(1 - exp(-ze)) / H_E      (chain rule via ze)
        if mask_e.any():
            exp_neg_ze_e = exp_neg_ze[mask_e]
            Ne_Ev        = Ne_E_v[mask_e]
    
            dNe_dP[I_LOG_NME, mask_e]  = Ne_Ev * np.log(10.0)
            dNe_dP[I_HME,     mask_e] += Ne_Ev * 0.5 * (1.0 - exp_neg_ze_e) / _H_E_KM
        dNe_dP_ensemble[:,:,iPts] = dNe_dP.T

    params_shape[0] = n_alt
    Ne_ensemble = Ne_ensemble.reshape(params_shape)
    if partial:
        return Ne_ensemble

    dNe_dP_ensemble = dNe_dP_ensemble.reshape([n_alt, N_STATE]+list(params_shape)[1:])    
    return Ne_ensemble, dNe_dP_ensemble

def extract_robust_f2_peak(profile: np.ndarray, alt_grid: np.ndarray,
                           min_alt: float = 150.0, max_alt: float = 650.0,
                           _half_window: int = 4, _weight_sigma_km: float = 14.0,
                           _min_side_pts: int = 2):
    """
    Robustly find NmF2 and hmF2 from a sampled Ne(h) profile by fitting a
    *two-sided* (asymmetric) parabola in log(Ne).

    A 3-point parabola in LINEAR Ne (the previous approach) is biased HIGH by
    ~+6 to +16 km on the geometric altitude grid: the F2 peak is asymmetric —
    the bottomside is steeper than the topside — and the grid spacing grows
    with altitude, so a symmetric-parabola vertex is pulled toward the flatter
    topside. Worse, that bias grows with bottomside breadth (B0), so a broader
    (more physically faithful) bottomside spuriously raises the detected hmF2
    even when the true peak height is unchanged.

    The fix fits Ne(h) near the peak as two half-parabolas in log(Ne) sharing a
    common vertex, with independent curvature above and below (so asymmetry no
    longer biases the vertex), and Gaussian proximity weights so distant,
    non-parabolic samples do not distort the local fit. Validated on analytic
    profiles from observation_operator._ne_profile_ensemble (known true hmF2):
    mean bias ~0 km with no residual B0 dependence, versus +6 km and strong B0
    correlation for the old linear fit.
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
        return discrete_NmF2, discrete_hmF2   # too few points to fit — fall back

    h = search_alts[lo:hi + 1]
    y = np.log(np.maximum(search_prof[lo:hi + 1], 1.0))

    # Search the vertex over one grid step either side of the discrete argmax;
    # for each trial vertex the two-sided model is linear in (y0, k_lo, k_hi).
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
        if k_lo <= 0.0 or k_hi <= 0.0:        # must curve down on both sides
            continue
        sse = float(np.sum(((A @ coeffs - y) * w) ** 2))
        if sse < best_sse:
            best_sse, best_hmF2, best_NmF2 = sse, float(h0), float(np.exp(y0))

    return best_NmF2, best_hmF2

def _h0_seed_from_profile(
    ne_topside: np.ndarray,
    alts_topside: np.ndarray,
    nm_f2: float,
    hm_f2: float,
) -> float:
    """
    Estimate the initial topside scale height H0 from the IRI ne profile
    without any external call.

    Uses the Epstein half-power point: 4·e^z/(1+e^z)² = 0.5  →  z ≈ 1.317,
    so H0_seed ≈ Δh_half / 1.317, where Δh_half is the altitude above hmF2
    where ne first drops to NmF2/2.

    Falls back to 60 km if no valid half-power point is found.
    """
    _EPSTEIN_Z_HALF = 1.3169578  # solve 4e^z/(1+e^z)^2 = 0.5

    ne_arr  = np.asarray(ne_topside)
    alt_arr = np.asarray(alts_topside)

    valid = np.isfinite(ne_arr) & (ne_arr > 0)
    if valid.sum() < 2:
        return 60.0

    ne_v   = ne_arr[valid]
    alt_v  = alt_arr[valid]
    half_target = nm_f2 / 2.0

    # Find the first valid point where ne drops below half-maximum
    below = np.where(ne_v < half_target)[0]
    if len(below) == 0:
        # Profile never drops to half-power in the available range — use last point
        H0_seed = float(alt_v[-1] - hm_f2) / _EPSTEIN_Z_HALF
    elif below[0] == 0:
        H0_seed = 60.0
    else:
        # Linear interpolation between the point just above and just below
        i = below[0]
        frac = (half_target - ne_v[i-1]) / (ne_v[i] - ne_v[i-1] + 1e-9)
        h_half = alt_v[i-1] + frac * (alt_v[i] - alt_v[i-1])
        H0_seed = (h_half - hm_f2) / _EPSTEIN_Z_HALF

    return float(np.clip(H0_seed, 10.0, 200.0))

def _fit_topside_H0_gamma(
    ne_topside: np.ndarray,
    alts_topside: np.ndarray,
    nm_f2: float,
    hm_f2: float,
    H0_seed: float,
) -> tuple[float, float]:
    """
    Jointly fit H0 and gamma from the valid (finite, positive) IRI ne topside
    profile by minimising log₁₀-RMSE.

    The variable-scale-height Epstein topside (from _ne_profile_ensemble):
        H_eff(h) = H0 * (1 + r*gamma*(h-hmF2) / (r*H0 + gamma*(h-hmF2)))
        z(h)     = (h - hmF2) / H_eff(h)
        ne(h)    = 4 * NmF2 * exp(z) / (1 + exp(z))^2

    H0_seed is used as the starting point; PyIRI B_top is a good choice.

    Returns (H0, gamma) with H0 in [10, 300] km and gamma in [0.01, 2.0].
    """
    from scipy.optimize import minimize
    r = 100.0   # _R_TOPSIDE

    ne_arr   = np.asarray(ne_topside)
    alts_arr = np.asarray(alts_topside)

    # Only fit against valid, positive IRI points
    valid = np.isfinite(ne_arr) & (ne_arr > 0)
    if valid.sum() < 3:
        return H0_seed, 0.5

    ne_ref = np.log10(ne_arr[valid])
    dh     = alts_arr[valid] - hm_f2

    def _cost(x):
        H0_t = np.exp(x[0])
        g    = np.exp(x[1])
        H_eff = H0_t * (1.0 + r * g * dh / (r * H0_t + g * dh + 1e-9))
        z     = np.clip(dh / (H_eff + 1e-9), -80, 80)
        ne_m  = np.maximum(4.0 * nm_f2 * np.exp(z) / (1.0 + np.exp(z))**2, 1.0)
        return float(np.mean((np.log10(ne_m) - ne_ref)**2))

    x0  = np.array([np.log(H0_seed), np.log(0.5)])
    bds = [(np.log(10.0), np.log(300.0)), (np.log(0.01), np.log(2.0))]
    res = minimize(_cost, x0, method="L-BFGS-B", bounds=bds,
                   options={"maxiter": 200, "ftol": 1e-6})
    H0_fit    = float(np.clip(np.exp(res.x[0]), 10.0, 300.0))
    gamma_fit = float(np.clip(np.exp(res.x[1]), 0.01, 2.0))
    return H0_fit, gamma_fit

def _fit_bottomside_B0_B1(
    ne_bottom: np.ndarray,
    alts_bottom: np.ndarray,
    nm_f2: float,
    hm_f2: float,
    B0_seed: float,
    B1_seed: float,
) -> tuple[float, float]:
    """
    Jointly fit B0 and B1 from the valid (finite, positive) IRI ne bottomside
    profile by minimising log₁₀-RMSE against OUR pure-bottomside formula.

    IRI's own B0/B1 parameterise IRI's bottomside; copying them into the
    Epstein-style bottomside used by _ne_profile_ensemble

        Ne(h) = NmF2 * exp(-x**B1) / cosh(x),   x = (hmF2 - h) / B0

    is systematically wrong (audit: bottomside log10-RMSE ~0.50 with the IRI
    values vs ~0.09 when fitted). We therefore treat the IRI B0/B1 only as a
    starting point and refit to reproduce the actual IRI profile shape in the
    model's own basis — the bottomside analogue of _fit_topside_H0_gamma.

    B0_seed/B1_seed are the IRI feature values (already clipped by the caller).

    Returns (B0, B1) with B0 in [20, 300] km and B1 in [0.5, 4.0].
    """
    from scipy.optimize import minimize

    ne_arr   = np.asarray(ne_bottom)
    alts_arr = np.asarray(alts_bottom)

    valid = np.isfinite(ne_arr) & (ne_arr > 0)
    if valid.sum() < 4:
        return B0_seed, B1_seed

    ne_ref = np.log10(ne_arr[valid])
    hh     = alts_arr[valid]

    def _cost(x):
        B0_t = np.exp(x[0])
        B1_t = np.exp(x[1])
        xx   = np.maximum((hm_f2 - hh) / (B0_t + 1e-9), 0.0)
        xp   = np.where(xx > 0, xx, 1e-30)
        ne_m = np.maximum(
            nm_f2 * np.exp(-xp ** B1_t) / np.cosh(np.clip(xx, 0.0, 700.0)), 1.0
        )
        return float(np.mean((np.log10(ne_m) - ne_ref) ** 2))

    x0  = np.array([np.log(np.clip(B0_seed, 20.0, 300.0)),
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

def _fit_iri_params(
    ne_profile: np.ndarray,
    alt_grid: np.ndarray,
    param_bound: dict
) -> np.ndarray:
    """
    Fit the 8 IRI parametric state vector components to a Ne(h) profile using
    a combination of direct extraction and numerical optimisation.

    Parameters
    ----------
    ne_profile : (n_alt,)  electron density in m⁻³ (linear).
    alt_grid   : (n_alt,)  altitude grid in km.

    Returns
    -------
    params_log : (N_STATE,)  in the mixed log/linear state-vector convention:
        [log10(NmF2), hmF2, H0, gamma, B0, B1, log10(NmE), hmE]
    """
    ne   = np.maximum(ne_profile, 1.0)
    alts = alt_grid

    # ── F2 peak ───────────────────────────────────────────────────────────────
    nm_f2, hm_f2 = extract_robust_f2_peak(ne, alts)
    if np.isnan(nm_f2) or np.isnan(hm_f2):
        nm_f2 = float(np.nanmax(ne))
        hm_f2 = float(alts[np.nanargmax(ne)])

    # ── E-layer peak (95–140 km) ──────────────────────────────────────────────
    # Only accept a *genuine* E-peak: an interior local maximum whose amplitude
    # is a physically sensible fraction of NmF2. When the E-region is F1-filled
    # or rising monotonically into the F-layer, np.nanmax lands on the window
    # edge and grabs an F-region density (~NmF2); using that as NmE wrecks the
    # E/valley reconstruction (audit: E-region log10-RMSE blew up to ~3.6).
    e_mask = (alts >= 95.0) & (alts <= 140.0)
    nm_e, hm_e = None, None
    if e_mask.sum() >= 3:
        ne_e  = ne[e_mask]
        alt_e = alts[e_mask]
        k     = int(np.nanargmax(ne_e))
        interior = 0 < k < len(ne_e) - 1
        if interior and ne_e[k] < 0.5 * nm_f2:
            nm_e = float(ne_e[k])
            hm_e = float(alt_e[k])
    if nm_e is None:
        nm_e = float(np.clip(nm_f2 * 0.05, 1e9, 0.3 * nm_f2))
        hm_e = 110.0

    # ── Topside: H0, gamma — analytic seed + joint log-space fit ─────────────
    # (same region-wise approach as _state_from_iri_direct; the previous single
    #  global Nelder-Mead over all 8 params was less accurate on every region.)
    top_mask = alts > hm_f2
    H0_seed  = _h0_seed_from_profile(ne[top_mask], alts[top_mask], nm_f2, hm_f2)
    H0, gamma = H0_seed, 0.5
    if top_mask.sum() >= 5:
        try:
            H0, gamma = _fit_topside_H0_gamma(
                ne[top_mask], alts[top_mask], nm_f2, hm_f2, H0_seed
            )
        except Exception:
            pass

    # ── Bottomside: B0, B1 — half-width seed + joint log-space fit ───────────
    bot_mask = (alts < hm_f2) & (alts > 100.0)
    if bot_mask.sum() >= 3:
        ne_bot  = ne[bot_mask]
        alt_bot = alts[bot_mask]
        target  = nm_f2 / np.e
        below   = alt_bot[ne_bot >= target]
        B0      = float(hm_f2 - below[0]) if len(below) > 0 else 80.0
        B0      = np.clip(B0, 20.0, 250.0)
        B1      = 1.5    # Chapman-like default
    else:
        B0 = 80.0
        B1 = 1.5
    # Refit over the F-region bottomside ONLY (150 km -> hmF2); the 100-150 km
    # E/valley band is F1-filled in IRI and unrepresentable by the 4-region
    # model, so including it distorts the fit (see _state_from_iri_direct).
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

    # ── Optional global polish, seeded from the region fits ──────────────────
    # The region-wise fits already give an excellent seed; a light bounded
    # Nelder-Mead over the whole profile can mop up residual valley/E coupling.
    # Accepted ONLY if it strictly lowers the whole-profile log-RMSE, so it can
    # never regress the region fits.
    _lo = np.array([param_bound['log10_NmF2'][0],
                    param_bound['hmF2'][0],
                    param_bound['H0'][0],   
                    param_bound['gamma'][0],
                    param_bound['B0'][0],
                    param_bound['B1'][0],
                    param_bound['log10_NmE'][0],
                    param_bound['hmE'][0]])
    _hi = np.array([param_bound['log10_NmF2'][1],
                    param_bound['hmF2'][1],
                    param_bound['H0'][1],   
                    param_bound['gamma'][1],
                    param_bound['B0'][1],
                    param_bound['B1'][1],
                    param_bound['log10_NmE'][1],
                    param_bound['hmE'][1]])
    def _residual(x):
        p = np.minimum(np.maximum(x, _lo), _hi)
        params_lin = np.array([
            10.0 ** p[0], p[1], p[2], p[3], p[4], p[5], 10.0 ** p[6], p[7]
        ])[:, np.newaxis]  # (8, 1)
        ne_model = _ne_profile_derivatives(alts, params_lin)[:, 0]
        return float(np.nanmean((np.log10(np.maximum(ne_model, 1.0))
                                 - np.log10(ne)) ** 2))

    try:
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

    # Clamp to physically plausible bounds
    x0[0] = np.clip(x0[0], param_bound['log10_NmF2'][0], param_bound['log10_NmF2'][1])    # log10(NmF2)
    x0[1] = np.clip(x0[1], param_bound['hmF2'][0], param_bound['hmF2'][1]) # hmF2
    x0[2] = np.clip(x0[2], param_bound['H0'][0], param_bound['H0'][1])  # H0
    x0[3] = np.clip(x0[3], param_bound['gamma'][0], param_bound['gamma'][1])    # gamma
    x0[4] = np.clip(x0[4], param_bound['B0'][0], param_bound['B0'][1])  # B0
    x0[5] = np.clip(x0[5], param_bound['B1'][0], param_bound['B1'][1])     # B1
    x0[6] = np.clip(x0[6], param_bound['log10_NmE'][0], param_bound['log10_NmE'][1])    # log10(NmE)
    x0[7] = np.clip(x0[7], param_bound['hmE'][0], param_bound['hmE'][1])  # hmE

    return x0   # (N_STATE,) in log-space convention

def PCA2EDP_1D(PCA_state:np.ndarray,PCA:np.ndarray,
               linear:bool=True, minlog10Density:float=4)->np.ndarray:
    #
    # From coefficient space to EDP space.
    # 1 PCA matrix has the dimension nAlt, nPCA
    # PCA_State has the dimension nPCA, nPts, nSample
    # EDP has the dimension nAlt, nPts, nSample
    #
    PCA_state_shape = PCA_state.shape
    PCA_shape =PCA.shape
    assert PCA_state_shape[0] == PCA_shape[1], "The dimesnions of PCA_State and PCA are inconsistent"
    PCA_state = PCA_state.reshape(PCA_state_shape[0],-1)
    density = PCA @ PCA_state
    density_shape = [PCA_shape[0]]+list(PCA_state_shape)[1:]
    density = density.reshape(density_shape)
    if not linear:
        density = 10 ** density
    return density

def EDP2PCA_1D(density:np.ndarray,PCA:np.ndarray,
               linear:bool=True, minlog10Density:float=4)->np.ndarray:
    #
    # Project density to PCA coefficients
    # 1 PCA matrix has the dimension nAlt, nPCA
    # EDP has the dimension nAlt, nPts, nSample
    # PCA_State has the dimension nPCA, nPts, nSample
    #
    density_shape = density.shape
    PCA_shape =PCA.shape
    assert density_shape[0] == PCA_shape[0], "The dimesnions of density and PCA are inconsistent"
    density = density.reshape(density_shape[0],-1)
    if not linear:
        density[np.where(density<10**minlog10Density)]=10*minlog10Density
        density = np.log10(density)
        
    PCA_state = PCA.T @ density
    PCA_state_shape = [PCA_shape[1]]+list(density_shape)[1:]
    PCA_state = PCA_state.reshape(PCA_state_shape)
    return PCA_state

def PCA2EDP_1D_map(PCA_state:np.ndarray,PCA:np.ndarray,
                   linear:bool=True, minlog10Density:float=4)->np.ndarray:    
    #
    # From coefficient space to EDP space.
    # 1 PCA matrix has the dimension nAlt, nPCA
    # PCA_State has the dimension nPCA, nPts, nSample
    # PCA2EDP_map has the dimension nAlt, nPCA, nPts, nSample
    #
    PCA_state_shape = PCA_state.shape
    PCA_shape =PCA.shape
    assert PCA_state_shape[0] == PCA_shape[1], "The dimesnions of PCA_State and PCA are inconsistent"
    A = np.ones(PCA_state_shape[1:])
    PCA2EDP_map = np.kron(A,PCA)
    if not linear:
        PCA2EDP_map = np.log(10)*PCA2EDP_map
        PCA_state = PCA_state.reshape(PCA_state_shape[0],-1)
        density = PCA @ PCA_state
        density_shape = [PCA_shape[0],1]+list(PCA_state_shape)[1:]
        density = density.reshape(density_shape)
        density = 10 ** density
        for idx in range(PCA_shape[1]):
            PCA2EDP_map[:,idx,:,:] = PCA2EDP_map[:,idx,:,:] * density
    return PCA2EDP_map

def PCA2EDP_3D(PCA_state:np.ndarray,PCA:np.ndarray,
               linear:bool=True, minlog10Density:float=4)->np.ndarray:
    #
    # From coefficient space to EDP space.
    # 3D PCA matrix has the dimension nAlt, nPts, nPCA
    # PCA_State has the dimension nPCA, nSample
    # EDP has the dimension nAlt, nPts, nSample
    #
    PCA_state_shape = PCA_state.shape
    PCA_shape =PCA.shape
    assert PCA_state_shape[0] == PCA_shape[2], "The dimesnions of PCA_State and PCA are inconsistent"
    PCA = PCA.reshape([PCA_shape[0]*PCA_shape[1],PCA_shape[2]])
    PCA_state = PCA_state.reshape(PCA_state_shape[0],-1)
    density = PCA @ PCA_state
    density_shape = [PCA_shape[0],PCA_shape[1],PCA_state_shape[1]]
    density = density.reshape(density_shape)
    if not linear:
        density = 10** density
    return density

def EDP2PCA_3D(density:np.ndarray,PCA:np.ndarray,
               linear:bool=True, minlog10Density:float=4)->np.ndarray:
    #
    # Project density to PCA coefficients
    # 3D PCA matrix has the dimension nAlt, nPts, nPCA
    # EDP has the dimension nAlt, nPts, nSample
    # PCA_State has the dimension nPCA, nSample
    #
    density_shape = density.shape
    PCA_shape =PCA.shape
    assert density_shape[0] == PCA_shape[0], "The dimesnions of density and PCA are inconsistent"
    assert density_shape[1] == PCA_shape[1], "The dimesnions of density and PCA are inconsistent"
    density = density.reshape([density_shape[0]*density_shape[1],density_shape[2]])
    if not linear:
        density[np.where(density<10**minlog10Density)]=10*minlog10Density
        density = np.log10(density)
        
    PCA = PCA.reshape([PCA_shape[0]*PCA_shape[1],PCA_shape[2]])
    PCA_state = PCA.T @ density
    PCA_state_shape = [PCA_shape[2],density_shape[2]]
    PCA_state = PCA_state.reshape(PCA_state_shape)
    return PCA_state

def PCA2EDP_3D_map(PCA_state:np.ndarray,PCA:np.ndarray,
                   linear:bool=True, minlog10Density:float=4)->np.ndarray:    
    #
    # From coefficient space to EDP space.
    # 3D PCA matrix has the dimension nAlt, nPts, nPCA
    # PCA_state has the dimension nPCA, nSample
    # PCA2EDP_map has the dimension nAlt, nPts, nPCA, nSample
    #
    PCA_state_shape = PCA_state.shape
    PCA_shape = PCA.shape
    assert PCA_state_shape[0] == PCA_shape[2], "The dimesnions of PCA_State and PCA are inconsistent"
    A = np.ones(PCA_state_shape[1:])
    PCA2EDP_map = np.kron(A,PCA)
    if not linear:
        PCA = PCA.shape([PCA_shape[0]*PCA_shape[1],PCA_shape[2]])
        density = PCA @ PCA_state
        density_shape = [PCA_shape[0],PCA_shape[1],1,PCA_state_shape[1]]
        density = density.reshape(density_shape)
        density = 10 ** density
        PCA2EDP_map = np.log(10)* PCA2EDP_map
        for idx in range(PCA_shape[2]):
            PCA2EDP_map[:,:,idx,:] = PCA2EDP_map[:,:,idx,:]*density
            
    return PCA2EDP_map

def get_PCA(edps:np.ndarray, retaining_threshold:float)-> np.ndarray:
    nSize, n_sample = edps.shape()
    # Mean-centre; treat NaNs as zero perturbation so they don't inflate variance
    edps_mean = np.nanmean(edps, axis=1, keepdims=True)
    edps_c    = np.where(np.isnan(edps), 0.0, edps - edps_mean).astype(np.float32)
    if nSize <= n_sample:
        edps_sq = edps_c @ edps_c.T
        eigenval, eigenvect = np.linalg.eigh(edps_sq)
        singularval = np.sqrt(eigenval)
        idx = np.where(singularval/singularval[-1]<=retaining_threshold)
        PCA = eigenvect[:,idx]
    else:
        edps_sq =  edps_c.T @ edps_c
        eigenval, eigenvect = np.linalg.eigh(edps_sq)
        singularval = np.sqrt(eigenval)
        idx = np.where(singularval/singularval[-1]<=retaining_threshold)
        sv = edps_c @ eigenvect
        PCA = sv[:,idx]
        
    return PCA
        
        
    
Parameterization_Style = Literal['density_10ex','ANCHOR','PCA_1D','PCA_3D','PCA_1D_10ex','PCA_3D_10ex']    

class EDP_Parameterization:
    def __init__(cls,style: Parameterization_Style = 'density_10ex',
                 hyper_params:dict=None):
        cls.style = style
        match style:
            case 'ANCHOR':
                if hyper_params is None:
                    cls.hyper_params ={'log10_NmF2':[9.0, 13.0],
                                'hmF2':[100.0, 600.0],
                                'H0':[10.0, 300.0],
                                'gamma':[0.05, 2.0],
                                'B0':[20.0, 300.0],
                                'B1':[0.5, 4.0],
                                'log10_NmE':[7.0, 12.0],
                                'hmE':[80.0, 180.0],
                                'N_STATE':8}
                else:
                    cls.bhyper_params = hyper_params
                    
            case 'density_10ex':
                if hyper_params is None:
                    cls.hyper_params ={'minlog10Density':4}
                else:
                    cls.hyper_params = hyper_params
            case 'PCA_1D' | 'PCA_1D_10ex':
                if hyper_params is None:
                    raise ValueError(f"Parameterization {cls.style} requires PCA in hyper_params.")
                if not 'PCA' in hyper_params.keys:    
                    raise ValueError(f"Parameterization {cls.style} requires PCA in hyper_params.")
                if not hyper_params['PCA'].ndim == 2:
                    raise ValueError(f"Parameterization {cls.style} requires PCA to be 2-dimensional (nAlt,nPCA).")
                cls.hyper_params = hyper_params
            case 'PCA_3D'|'PCA_3D_10ex':
                if hyper_params is None:
                    raise ValueError(f"Parameterization {cls.style} requires PCA in hyper_params.")
                if not 'PCA' in hyper_params.keys:    
                    raise ValueError(f"Parameterization {cls.style} requires PCA in hyper_params.")
                if not hyper_params['PCA'].ndim == 3:
                    raise ValueError(f"Parameterization {cls.style} requires PCA to be 3-dimensional (nAlt,nGrid,nPCA).")
    
    
    def get_density(self, param_vec: np.array, alt:np.array=None)-> np.array:
        match self.style:
            case 'density_10ex':
                density = density_10ex(param_vec,
                                       minlog10Density=self.hyper_params['minlog10Density'])
            case 'ANCHOR':
                density = _ne_profile_derivatives(alt,param_vec,partial=False)   
            case 'PCA_1D':
                density = PCA2EDP_1D(param_vec,self.hyper_params['PCA'],
                                     linear=True, minlog10Density=4)
            case 'PCA_1D_10ex':
                density = PCA2EDP_1D(param_vec,self.hyper_params['PCA'],
                                     linear=False, minlog10Density=4)
            case 'PCA_3D':
                density = PCA2EDP_3D(param_vec,self.hyper_params['PCA'],
                                     linear=True, minlog10Density=4)
            case 'PCA_3D_10ex':
                density = PCA2EDP_3D(param_vec,self.hyper_params['PCA'],
                                     linear=False, minlog10Density=4)                
            case _:
                raise ValueError(f"Parameterization, {self.style} not defined.")
        return density

    def get_parameter(self, density:np.array, alt:np.array=None)-> np.array:
        match self.style:
            case 'density_10ex':
                param_vec = log10_param(density,
                                       minlog10Density=self.hyper_params['minlog10Density'])
            case 'ANCHOR':
                param_vec = _fit_iri_params(density,alt,param_bounds=self.hyper_params)
            case 'PCA_1D':
                param_vec = EDP2PCA_1D(density,self.hyper_params['PCA'],
                                     linear=True, minlog10Density=4)
            case 'PCA_1D_10ex':
                param_vec = EDP2PCA_1D(density,self.hyper_params['PCA'],
                                     linear=False, minlog10Density=4)
            case 'PCA_3D':
                param_vec = EDP2PCA_3D(density,self.hyper_params['PCA'],
                                     linear=True, minlog10Density=4)
            case 'PCA_3D_10ex':
                param_vec = EDP2PCA_3D(density,self.hyper_params['PCA'],
                                     linear=False, minlog10Density=4)                
            case _:
                raise ValueError(f"Parameterization, {self.style} not defined.")
        return param_vec
    def get_Jacobian(self, param_vec: np.array, alt:np.array=None)-> tuple(str,np.array):
        match self.style:
            case 'density_10ex':
                Jacb_matrix_style = 'diagonal'
                Jacobian = Jacobian_density_10ex(param_vec,
                                       minlog10Density=self.hyper_params['minlog10Density'])
            case 'ANCHOR':
                Jacb_matrix_style = 'block_diagonal'
                density, Jacobian = _ne_profile_derivatives(alt,param_vec)
            case 'PCA_1D':
                Jacb_matrix_style = 'block_diagonal'
                Jacobian = PCA2EDP_1D_map(param_vec,self.hyper_params['PCA'],
                                     linear=True, minlog10Density=4)
            case 'PCA_1D_10ex':
                Jacb_matrix_style = 'block_diagonal'
                Jacobian = PCA2EDP_1D_map(param_vec,self.hyper_params['PCA'],
                                     linear=False, minlog10Density=4)
            case 'PCA_3D':
                Jacb_matrix_style = 'full'
                Jacobian = PCA2EDP_3D_map(param_vec,self.hyper_params['PCA'],
                                     linear=True, minlog10Density=4)
            case 'PCA_3D_10ex':
                Jacb_matrix_style = 'full'
                Jacobian = PCA2EDP_3D_map(param_vec,self.hyper_params['PCA'],
                                     linear=False, minlog10Density=4)                
            case _:
                raise ValueError(f"Parameterization, {self.style} not defined.")
        return Jacb_matrix_style, Jacobian

class Parameterized_EDPSamples:
    __slots__ = ('EDPSamples', 'style', 'Parameterization')
    
    def __init__(cls, EDP:EDPSamples,style: Parameterization_Style = 'density_10ex',
                 hyper_params:dict=None):
        cls.EDPSamples = EDP
        cls.style = style
        
        if style in ['PCA_1D','PCA_1D_10ex','PCA_3D','PCA_3D_10ex']:
            assert "retaining_threshold" in hyper_params.keys(), "For using PCA parameterization, retaining threshold must be provided."

        match cls.style:
            case 'density_10ex':
                cls.Parameterization = EDP_Parameterization(style=style,hyper_params=hyper_params)
                param_vec = cls.Parameterization.get_parameter(cls.EDPSamples.edps,hyper_params=hyper_params)
                cls.EDPSamples.data_vars["param_vec"]=(
                                (EDPSamples.DIM_HEIGHT, EDPSamples.DIM_GEO, EDPSamples.DIM_SAMPLE),
                                param_vec,
                                {"long_name": "Paramteter Values",
                                 "description": "electron density profiles samples"}
                                )

            case 'ANCHOR':
                cls.Parameterization = EDP_Parameterization(style=style,hyper_params=hyper_params)
                param_vec = cls.Parameterization.get_parameter(cls.EDPSamples.edps,hyper_params=hyper_params)
                cls.EDPSamples.data_vars["param_vec"]=(
                                (EDPSamples.DIM_HEIGHT, EDPSamples.DIM_GEO, EDPSamples.DIM_SAMPLE),
                                param_vec,
                                {"long_name": "Paramteter Values",
                                 "description": "electron density profiles samples"}
                                )
            case 'PCA_1D':
                edps = cls.EDPSamples.edps
                edps_shape = edps.shape
                edps = edps.reshape(edps_shape[0],-1)
                PCA = get_PCA(edps, hyper_params["retaining_threshold"])
                cls.EDPSamples.data_vars["PCA"]=(
                                (EDPSamples.DIM_HEIGHT, "nPCA"),
                                PCA,
                                {"long_name": "PCA",
                                 "description": "Principle components"}
                                )
                hyper_params['PCA'] = PCA
                cls.Parameterization = EDP_Parameterization(style=style,hyper_params=hyper_params)
                param_vec = cls.Parameterization.get_parameter(cls.EDPSamples.edps,hyper_params=hyper_params)
                cls.EDPSamples.data_vars["param_vec"]=(
                                (EDPSamples.DIM_HEIGHT, EDPSamples.DIM_GEO, EDPSamples.DIM_SAMPLE),
                                param_vec,
                                {"long_name": "Paramteter Values",
                                 "description": "electron density profiles samples"}
                                )
                
            case 'PCA_1D_10ex':
                if 'minlog10Density' in hyper_params.keys():
                    minlog10Density = hyper_params['minlog10Density']
                else:
                    minlog10Density = 4
                    cls.hyper_params['minlog10Density']=4
                    
                edps = cls.EDPSamples.edps
                edps[edps.where(edps<10**minlog10Density)] = 10**minlog10Density
                edps = np.log10(edps)
                edps_shape = edps.shape
                edps = edps.reshape(edps_shape[0],-1)
                PCA = get_PCA(edps, hyper_params["retaining_threshold"])
                cls.EDPSamples.data_vars["PCA"]=(
                                (EDPSamples.DIM_HEIGHT, EDPSamples.DIM_GEO, "nPCA"),
                                PCA,
                                {"long_name": "PCA",
                                 "description": "Principle components"}
                                )
                hyper_params['PCA'] = PCA
                cls.Parameterization = EDP_Parameterization(style=style,hyper_params=hyper_params)
                param_vec = cls.Parameterization.get_parameter(cls.EDPSamples.edps,hyper_params=hyper_params)
                cls.EDPSamples.data_vars["param_vec"]=(
                                (EDPSamples.DIM_HEIGHT, EDPSamples.DIM_GEO, EDPSamples.DIM_SAMPLE),
                                param_vec,
                                {"long_name": "Paramteter Values",
                                 "description": "electron density profiles samples"}
                                )
                
            case 'PCA_3D':
                edps = cls.EDPSamples.edps
                edps_shape = edps.shape
                edps = edps.reshape(edps_shape[0]*edps_shape[1],-1)
                PCA = get_PCA(edps, hyper_params["retaining_threshold"])
                nD, nPCA = PCA.shape()
                PCA = PCA.reshape([edps_shape[0],edps_shape[1],nPCA])
                cls.EDPSamples.data_vars["PCA"]=(
                                (EDPSamples.DIM_HEIGHT, EDPSamples.DIM_GEO, "nPCA"),
                                PCA,
                                {"long_name": "PCA",
                                 "description": "Principle components"}
                                )
                hyper_params['PCA'] = PCA
                cls.Parameterization = EDP_Parameterization(style=style,hyper_params=hyper_params)
                param_vec = cls.Parameterization.get_parameter(cls.EDPSamples.edps,hyper_params=hyper_params)
                cls.EDPSamples.data_vars["param_vec"]=(
                                (EDPSamples.DIM_HEIGHT, EDPSamples.DIM_GEO, EDPSamples.DIM_SAMPLE),
                                param_vec,
                                {"long_name": "Paramteter Values",
                                 "description": "electron density profiles samples"}
                                )
            case 'PCA_3D_10ex':
                if 'minlog10Density' in hyper_params.keys():
                    minlog10Density = hyper_params['minlog10Density']
                else:
                    minlog10Density = 4
                    cls.hyper_params['minlog10Density']=4
                edps = cls.EDPSamples.edps
                edps[edps.where(edps<10**minlog10Density)] = 10**minlog10Density
                edps = np.log10(edps)
                edps_shape = edps.shape
                edps = edps.reshape(edps_shape[0]*edps_shape[1],-1)
                PCA = get_PCA(edps, hyper_params["retaining_threshold"])
                nD, nPCA = PCA.shape()
                PCA = PCA.reshape([edps_shape[0],edps_shape[1],nPCA])
                cls.EDPSamples.data_vars["PCA"]=(
                                (EDPSamples.DIM_HEIGHT, EDPSamples.DIM_GEO, "nPCA"),
                                PCA,
                                {"long_name": "PCA",
                                 "description": "Principle components"}
                                )
                hyper_params['PCA'] = PCA
                cls.Parameterization = EDP_Parameterization(style=style,hyper_params=hyper_params)
                param_vec = cls.Parameterization.get_parameter(cls.EDPSamples.edps,hyper_params=hyper_params)
                cls.EDPSamples.data_vars["param_vec"]=(
                                (EDPSamples.DIM_HEIGHT, EDPSamples.DIM_GEO, EDPSamples.DIM_SAMPLE),
                                param_vec,
                                {"long_name": "Paramteter Values",
                                 "description": "electron density profiles samples"}
                                )
                    
    
        cls.EDPSamples.attrs['Parameterization_style']=style
        for key in hyper_params.keys():
            if not key == "PCA":
                cls.EDPSamples.attrs['Parameterization_'+key] =hyper_params[key]

    @classmethod    
    def fromNetCDF(cls, path: str | Path, **kwargs: Any) -> Parameterized_EDPSamples:
        """
        Load an ``Parameterized_EDPSample`` from NetCDF (e.g. written by :meth:`saveNetCDF`).

        The file is read fully into memory, then closed.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the ``.nc`` file.
        **kwargs
            Forwarded to :func:`xarray.open_dataset` (e.g. ``engine``, ``group``,
            ``decode_times``, ``mask_and_scale``).

        Returns
        -------
        Parameterized_EDPSample
        """
        with xr.open_dataset(path, **kwargs) as ds:
            ds.load()
            EDPSam = Parameterized_EDPSamples.from_xarray(ds)
            
            return EDPSam

    @classmethod
    def from_xarray(cls, ds) -> Parameterized_EDPSamples:
        EDPSam = EDPSamples.from_xarray(ds)
        EDPSam.EDPSamples.data_vars = ds.data_vars
        if 'Parameterization_style' in ds.attrs.keys():
            style = ds.attrs['Parameterization_style']
        else: 
            raise ValueError("The nc4 file does not contain a saved Parameterized_EDPSamples.")

        hyper_params={'style':ds.attrs['Parameterization_style']}
        for key,value in ds.attrs.keys():
            if 'Parameterization_' in key:
                nkey = key.replace("Parameterization_","")
                hyper_params[nkey] = value
                
        if "PCA" in ds.data_vars.keys():
            hyper_params["PCA"] = ds.data_vars["PCA"].to_numpy()

        Params_EDPS = Parameterized_EDPSamples(EDPSam,style = style,
                     hyper_params=hyper_params)
        return Params_EDPS    

    @classmethod
    def saveNetCDF(self, path: str | Path, **kwargs: Any) -> None:
        self.EDPSamples.saveNetCDF(path, **kwargs)