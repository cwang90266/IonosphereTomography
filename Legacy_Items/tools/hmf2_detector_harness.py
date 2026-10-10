#!/usr/bin/env python3
"""
Validation harness for the hmF2 peak detector (extract_robust_f2_peak).

Generates analytic Ne(h) profiles from _ne_profile_ensemble where the TRUE
hmF2 is exactly the state param, samples them on the project's geometric
ALT_GRID, and measures detected-vs-true hmF2 bias for the current detector
and for candidate replacements.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Ionosphere_Tomography_Inverter"))

from Ionosphere_Tomography_Inverter.observation_operator import _ne_profile_ensemble
from Ionosphere_Tomography_Inverter.ionospheric_state import (
    I_LOG_NMF2, I_HMF2, I_H0, I_GAMMA, I_B0, I_B1, I_LOG_NME, I_HME, N_STATE)

ALT_GRID = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)

# ---------------------------------------------------------------------------
# Current detector (verbatim copy from demo.py extract_robust_f2_peak)
# ---------------------------------------------------------------------------
def extract_robust_f2_peak(profile, alt_grid, min_alt=150.0, max_alt=650.0):
    search_mask = (alt_grid >= min_alt) & (alt_grid <= max_alt) & ~np.isnan(profile)
    if not np.any(search_mask):
        return np.nan, np.nan
    search_alts = alt_grid[search_mask]
    search_prof = profile[search_mask]
    local_max_idx = np.argmax(search_prof)
    discrete_hmF2 = search_alts[local_max_idx]
    discrete_NmF2 = search_prof[local_max_idx]
    if 0 < local_max_idx < len(search_prof) - 1:
        h1, h2, h3 = search_alts[local_max_idx-1: local_max_idx+2]
        n1, n2, n3 = search_prof[local_max_idx-1: local_max_idx+2]
        try:
            coeffs = np.polyfit([h1, h2, h3], [n1, n2, n3], deg=2)
            a, b, c = coeffs
            if a < 0:
                refined_hmF2 = -b / (2.0 * a)
                refined_NmF2 = (a * refined_hmF2**2) + (b * refined_hmF2) + c
                if abs(refined_hmF2 - discrete_hmF2) <= (h3 - h1):
                    return refined_NmF2, refined_hmF2
        except np.linalg.LinAlgError:
            pass
    return discrete_NmF2, discrete_hmF2


# ---------------------------------------------------------------------------
# Candidate A: parabola in log(Ne) on a locally uniform-resampled window
# ---------------------------------------------------------------------------
def peak_logNe_uniform(profile, alt_grid, min_alt=150.0, max_alt=650.0,
                       half_window=3, n_fine=8):
    search_mask = (alt_grid >= min_alt) & (alt_grid <= max_alt) & ~np.isnan(profile)
    if not np.any(search_mask):
        return np.nan, np.nan
    sa = alt_grid[search_mask]
    sp = profile[search_mask]
    idx = int(np.argmax(sp))
    h0, N0 = sa[idx], sp[idx]
    lo = max(0, idx - half_window)
    hi = min(len(sa) - 1, idx + half_window)
    if hi - lo < 2:
        return N0, h0
    hw = sa[lo:hi+1]
    Nw = sp[lo:hi+1]
    # Resample onto uniform h with log-Ne interpolation
    Nw_pos = np.maximum(Nw, 1.0)
    hf = np.linspace(hw[0], hw[-1], (len(hw) - 1) * n_fine + 1)
    logNf = np.interp(hf, hw, np.log(Nw_pos))
    j = int(np.argmax(logNf))
    if 0 < j < len(hf) - 1:
        x = hf[j-1:j+2]
        y = logNf[j-1:j+2]
        a, b, c = np.polyfit(x, y, 2)
        if a < 0:
            hpk = -b / (2.0 * a)
            if hw[0] <= hpk <= hw[-1]:
                Npk = np.exp(a * hpk**2 + b * hpk + c)
                return Npk, hpk
    return np.exp(logNf[j]), hf[j]


# ---------------------------------------------------------------------------
# Candidate B: 3-point parabola in log(Ne) directly on native (unequal) grid
# ---------------------------------------------------------------------------
def peak_logNe_native(profile, alt_grid, min_alt=150.0, max_alt=650.0):
    search_mask = (alt_grid >= min_alt) & (alt_grid <= max_alt) & ~np.isnan(profile)
    if not np.any(search_mask):
        return np.nan, np.nan
    sa = alt_grid[search_mask]
    sp = profile[search_mask]
    idx = int(np.argmax(sp))
    h0, N0 = sa[idx], sp[idx]
    if 0 < idx < len(sp) - 1:
        h = sa[idx-1:idx+2]
        n = np.maximum(sp[idx-1:idx+2], 1.0)
        a, b, c = np.polyfit(h, np.log(n), 2)
        if a < 0:
            hpk = -b / (2.0 * a)
            if abs(hpk - h0) <= (h[2] - h[0]):
                return np.exp(a*hpk**2 + b*hpk + c), hpk
    return N0, h0


# ---------------------------------------------------------------------------
# True hmF2 from analytic profile: it's exactly the state param (I_HMF2).
# ---------------------------------------------------------------------------
def make_profile(nmf2_log, hmf2, h0, gamma, b0, b1, nme_log, hme, alt=ALT_GRID):
    p = np.zeros((N_STATE, 1))
    p[I_LOG_NMF2, 0] = 10.0 ** nmf2_log
    p[I_HMF2, 0] = hmf2
    p[I_H0, 0] = h0
    p[I_GAMMA, 0] = gamma
    p[I_B0, 0] = b0
    p[I_B1, 0] = b1
    p[I_LOG_NME, 0] = 10.0 ** nme_log
    p[I_HME, 0] = hme
    return _ne_profile_ensemble(alt, p)[:, 0]


def run_ensemble(detector, rng, n=400):
    errs = []
    b0s = []
    for _ in range(n):
        hmf2 = rng.uniform(280.0, 420.0)
        b0 = rng.uniform(25.0, 120.0)
        b1 = rng.uniform(1.0, 3.0)
        prof = make_profile(
            nmf2_log=rng.uniform(11.3, 12.4),
            hmf2=hmf2, h0=rng.uniform(30.0, 80.0),
            gamma=rng.uniform(0.3, 1.2), b0=b0, b1=b1,
            nme_log=rng.uniform(10.5, 11.3), hme=rng.uniform(100.0, 130.0))
        _, det = detector(prof, ALT_GRID)
        errs.append(det - hmf2)
        b0s.append(b0)
    errs = np.array(errs)
    b0s = np.array(b0s)
    # correlation of error with B0 (bottomside breadth)
    corr = np.corrcoef(errs, b0s)[0, 1]
    return errs, corr


from scipy.interpolate import PchipInterpolator, CubicSpline


# Candidate C: log-Ne, wide symmetric window, single parabola vertex on native h
def peak_logNe_wide(profile, alt_grid, min_alt=150.0, max_alt=650.0, hw=3):
    m = (alt_grid >= min_alt) & (alt_grid <= max_alt) & ~np.isnan(profile)
    if not np.any(m):
        return np.nan, np.nan
    sa, sp = alt_grid[m], profile[m]
    i = int(np.argmax(sp))
    lo, hi = max(0, i - hw), min(len(sa) - 1, i + hw)
    if hi - lo < 2:
        return sp[i], sa[i]
    a, b, c = np.polyfit(sa[lo:hi+1], np.log(np.maximum(sp[lo:hi+1], 1.0)), 2)
    if a < 0:
        hpk = -b / (2.0 * a)
        if sa[lo] <= hpk <= sa[hi]:
            return np.exp(a*hpk**2 + b*hpk + c), hpk
    return sp[i], sa[i]


# Candidate D: pchip on log-Ne, fine argmax over wide window
def peak_pchip(profile, alt_grid, min_alt=150.0, max_alt=650.0, hw=4):
    m = (alt_grid >= min_alt) & (alt_grid <= max_alt) & ~np.isnan(profile)
    if not np.any(m):
        return np.nan, np.nan
    sa, sp = alt_grid[m], profile[m]
    i = int(np.argmax(sp))
    lo, hi = max(0, i - hw), min(len(sa) - 1, i + hw)
    if hi - lo < 3:
        return sp[i], sa[i]
    hw_a, Nw = sa[lo:hi+1], np.maximum(sp[lo:hi+1], 1.0)
    spl = PchipInterpolator(hw_a, np.log(Nw))
    hf = np.linspace(hw_a[0], hw_a[-1], 2000)
    yf = spl(hf)
    j = int(np.argmax(yf))
    return np.exp(yf[j]), hf[j]


# Candidate E: cubic spline on log-Ne, analytic stationary point near peak
def peak_cubic(profile, alt_grid, min_alt=150.0, max_alt=650.0, hw=4):
    m = (alt_grid >= min_alt) & (alt_grid <= max_alt) & ~np.isnan(profile)
    if not np.any(m):
        return np.nan, np.nan
    sa, sp = alt_grid[m], profile[m]
    i = int(np.argmax(sp))
    lo, hi = max(0, i - hw), min(len(sa) - 1, i + hw)
    if hi - lo < 3:
        return sp[i], sa[i]
    hw_a, Nw = sa[lo:hi+1], np.maximum(sp[lo:hi+1], 1.0)
    spl = CubicSpline(hw_a, np.log(Nw))
    hf = np.linspace(hw_a[0], hw_a[-1], 4000)
    yf = spl(hf)
    j = int(np.argmax(yf))
    return np.exp(yf[j]), hf[j]


# Candidate F: quartic in log-Ne over window, local max via derivative root
def peak_quartic(profile, alt_grid, min_alt=150.0, max_alt=650.0, hw=3):
    m = (alt_grid >= min_alt) & (alt_grid <= max_alt) & ~np.isnan(profile)
    if not np.any(m):
        return np.nan, np.nan
    sa, sp = alt_grid[m], profile[m]
    i = int(np.argmax(sp))
    lo, hi = max(0, i - hw), min(len(sa) - 1, i + hw)
    if hi - lo < 4:
        return sp[i], sa[i]
    hloc = sa[lo:hi+1]
    yloc = np.log(np.maximum(sp[lo:hi+1], 1.0))
    hc = hloc.mean()
    coeffs = np.polyfit(hloc - hc, yloc, 4)
    dcoef = np.polyder(coeffs)
    roots = np.roots(dcoef)
    roots = roots[np.abs(roots.imag) < 1e-6].real + hc
    roots = roots[(roots >= sa[i] - 25) & (roots <= sa[i] + 25)]
    if len(roots):
        vals = np.polyval(coeffs, roots - hc)
        hpk = roots[int(np.argmax(vals))]
        return np.exp(np.polyval(coeffs, hpk - hc)), hpk
    return sp[i], sa[i]


# Candidate G: cubic spline over the FULL F-region search window
def peak_cubic_full(profile, alt_grid, min_alt=150.0, max_alt=650.0):
    m = (alt_grid >= min_alt) & (alt_grid <= max_alt) & ~np.isnan(profile)
    if not np.any(m):
        return np.nan, np.nan
    sa, sp = alt_grid[m], profile[m]
    if len(sa) < 4:
        return sp[int(np.argmax(sp))], sa[int(np.argmax(sp))]
    spl = CubicSpline(sa, np.log(np.maximum(sp, 1.0)))
    hf = np.linspace(sa[0], sa[-1], 6000)
    yf = spl(hf)
    j = int(np.argmax(yf))
    return np.exp(yf[j]), hf[j]


# FINAL: two-sided (asymmetric) parabola in log-Ne with Gaussian proximity
# weights — the chosen production detector. Unbiased, B0-confound removed.
def peak_final(profile, alt_grid, min_alt=150.0, max_alt=650.0,
               hw=4, sig=14.0, minside=2):
    m = (alt_grid >= min_alt) & (alt_grid <= max_alt) & ~np.isnan(profile)
    if not np.any(m):
        return np.nan, np.nan
    sa, sp = alt_grid[m], profile[m]
    i = int(np.argmax(sp))
    lo, hi = max(0, i - hw), min(len(sa) - 1, i + hw)
    if hi - lo < 4:
        return sp[i], sa[i]
    h = sa[lo:hi+1]
    y = np.log(np.maximum(sp[lo:hi+1], 1.0))
    dh = (sa[min(i+1, len(sa)-1)] - sa[max(i-1, 0)]) / 2.0
    best = (np.inf, sa[i], sp[i])
    for h0 in np.linspace(sa[i] - dh, sa[i] + dh, 81):
        d = h - h0
        below = d < 0
        if np.sum(below) < minside or np.sum(~below) < minside:
            continue
        w = np.exp(-0.5 * (d / sig) ** 2)
        A = np.column_stack([np.ones_like(d),
                             -np.where(below, d**2, 0.0),
                             -np.where(~below, d**2, 0.0)])
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        y0, k_lo, k_hi = coef
        if k_lo <= 0 or k_hi <= 0:
            continue
        sse = float(np.sum(((A @ coef - y) * w) ** 2))
        if sse < best[0]:
            best = (sse, h0, np.exp(y0))
    return best[2], best[1]


# Candidate H: two-sided (asymmetric) parabola in log-Ne.
#   y(h) = y0 - k_lo*(h-h0)^2   (h<h0)
#   y(h) = y0 - k_hi*(h-h0)^2   (h>=h0)
# For each trial vertex h0 the model is linear in (y0, k_lo, k_hi); grid-search
# h0 over [argmax-Δ, argmax+Δ] and keep the least-squares-best vertex.
def peak_twosided(profile, alt_grid, min_alt=150.0, max_alt=650.0, hw=3):
    m = (alt_grid >= min_alt) & (alt_grid <= max_alt) & ~np.isnan(profile)
    if not np.any(m):
        return np.nan, np.nan
    sa, sp = alt_grid[m], profile[m]
    i = int(np.argmax(sp))
    lo, hi = max(0, i - hw), min(len(sa) - 1, i + hw)
    if hi - lo < 4:
        return sp[i], sa[i]
    h = sa[lo:hi+1]
    y = np.log(np.maximum(sp[lo:hi+1], 1.0))
    dh_grid = (sa[min(i+1, len(sa)-1)] - sa[max(i-1, 0)]) / 2.0
    h0_cands = np.linspace(sa[i] - dh_grid, sa[i] + dh_grid, 61)
    best = (np.inf, sa[i], sp[i])
    for h0 in h0_cands:
        d = h - h0
        below = d < 0
        # columns: [1, -(d^2 if below), -(d^2 if above)]
        col_lo = np.where(below, d**2, 0.0)
        col_hi = np.where(~below, d**2, 0.0)
        A = np.column_stack([np.ones_like(d), -col_lo, -col_hi])
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        y0, k_lo, k_hi = coef
        if k_lo <= 0 or k_hi <= 0:      # must curve down on both sides
            continue
        pred = A @ coef
        sse = float(np.sum((pred - y) ** 2))
        if sse < best[0]:
            best = (sse, h0, np.exp(y0))
    return best[2], best[1]


if __name__ == "__main__":
    print(f"ALT_GRID: {len(ALT_GRID)} pts, spacing near 350km "
          f"= {np.diff(ALT_GRID)[np.searchsorted(ALT_GRID,350)]:.1f} km, "
          f"near 400km = {np.diff(ALT_GRID)[np.searchsorted(ALT_GRID,400)]:.1f} km")
    dets = {
        "CURRENT (linear parabola, native)": extract_robust_f2_peak,
        "A: logNe, uniform-resampled window": peak_logNe_uniform,
        "B: logNe, native 3-point": peak_logNe_native,
        "C: logNe, wide sym window parabola": peak_logNe_wide,
        "D: pchip logNe fine argmax (hw=4)": peak_pchip,
        "E: cubic-spline logNe fine argmax": peak_cubic,
        "F: quartic logNe deriv-root": peak_quartic,
        "G: cubic-spline logNe full window": peak_cubic_full,
        "H: two-sided asym parabola logNe": peak_twosided,
        "FINAL: two-sided + proximity weights": peak_final,
    }
    for name, fn in dets.items():
        rng = np.random.default_rng(1234)
        errs, corr = run_ensemble(fn, rng)
        print(f"\n{name}")
        print(f"  mean err   = {errs.mean():+7.2f} km")
        print(f"  median err = {np.median(errs):+7.2f} km")
        print(f"  std        = {errs.std():7.2f} km")
        print(f"  max |err|  = {np.abs(errs).max():7.2f} km")
        print(f"  frac |err|<2km = {np.mean(np.abs(errs) < 2.0):.2%}")
        print(f"  corr(err, B0)  = {corr:+.3f}")
