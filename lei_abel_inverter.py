"""
Lei-Abel Ionospheric Inversion
Ported from AuroraRetreivalnew_BC.py (Chunming Wang / PlanetiQ).

Entry point: run_abel_inversion(podTc2_data) -> dict
"""

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

from TEC_model.podTc_file_processing import rayTangent, parse_podTc2_nc_file


# ─────────────────────────────────────────────────────────────────────────────
# Internal math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coef(p):
    """
    Abel inversion coefficients for impact-parameter vector p (Lei et al. 2007).
    p must be in metres.
    """
    m = len(p) - 1
    epsilon = (p - p[0]) / p[0]
    c_i = np.zeros(m + 1)

    def _sqrt(x):
        return np.sqrt(x) if x > 0 else 0.0

    if epsilon[1] > 0:
        s1 = _sqrt(epsilon[1] * (2 + epsilon[1]))
        denom = 1 + epsilon[1] + s1
        c_i[0] = (1 / epsilon[1]) * ((1 + epsilon[1]) * s1 - np.log(denom)) if denom > 0 else 0.0

    for k in range(1, m):
        e1, e0, em1 = epsilon[k + 1], epsilon[k], epsilon[k - 1]
        s1, s0, sm1 = _sqrt(e1 * (2 + e1)), _sqrt(e0 * (2 + e0)), _sqrt(em1 * (2 + em1))
        d1, d2 = e1 - e0, e0 - em1
        ln1 = 1 + e1 + s1
        ln0a = 1 + e0 + s0
        ln0b = 1 + e0 + s0
        lnm1 = 1 + em1 + sm1
        t1 = (1 / d1) * ((1 + e1) * (s1 - s0) - np.log(ln1 / ln0a)) if d1 != 0 and ln1 > 0 and ln0a > 0 else 0.0
        t2 = (1 / d2) * ((1 + em1) * (s0 - sm1) - np.log(ln0b / lnm1)) if d2 != 0 and ln0b > 0 and lnm1 > 0 else 0.0
        c_i[k] = t1 - t2

    e_l, e_l2 = epsilon[-1], epsilon[-2]
    s_l, s_l2 = _sqrt(e_l * (2 + e_l)), _sqrt(e_l2 * (2 + e_l2))
    d_l = e_l - e_l2
    ln_l, ln_l2 = 1 + e_l + s_l, 1 + e_l2 + s_l2
    if d_l != 0 and ln_l > 0 and ln_l2 > 0:
        c_i[-1] = -(1 / d_l) * (
            (1 - e_l + 2 * e_l2) * s_l - (1 + e_l2) * s_l2 - np.log(ln_l / ln_l2)
        )

    return c_i, m, epsilon


def _lei_abel_invert(p_km, TEC_tecu):
    """
    Discrete Abel inversion (Lei et al. 2007, updated top-altitude handling).

    Parameters
    ----------
    p_km : ndarray (m,)   — tangent-point impact radii in km, descending order
    TEC_tecu : ndarray (m,) — calibrated TEC in TECU

    Returns
    -------
    N_e : ndarray (m,)  — electron density in e-/m³
    """
    m = len(p_km)
    N_e = np.zeros(m)
    TEC = TEC_tecu * 1e16      # TECU → e-/m²
    p = p_km * 1e3             # km → m

    # Fit top boundary condition
    a, n, num_top = 10.0, 20e3, 3
    while a > 0 and n < 250e3:
        idx = np.where((p[0] - p) <= n)
        p_sl, T_sl = p[idx], TEC[idx]
        if len(idx[0]) == 0:
            p_sl, T_sl = p, TEC
        X = p_sl[:, np.newaxis]
        A = np.hstack([X, np.ones_like(X)])
        coeffs, *_ = np.linalg.lstsq(A, T_sl ** 2, rcond=None)
        a, b = coeffs
        if a < 0:
            w = np.linspace(0.6, 0.3, len(p_sl))
            N_top = w * np.sqrt(np.abs(a) / (8 * p_sl)) + (1 - w) * np.sqrt(np.abs(b) / (8 * p_sl ** 2))
            num_top = min(15, len(N_top))
            for i in range(num_top):
                N_e[-(i + 1)] = N_top[-(i + 1)]
            break
        n += 5e3
    else:
        N_e[-3:] = np.nan

    # Main downward inversion loop
    i = m - num_top - 1
    while i >= 0:
        SUM = 0.0
        P = p[i:m]
        c_i, _, _ = _coef(P)
        M = m - i
        for k in range(1, M):
            N = N_e[i + k]
            SUM += c_i[-M + k] * (0.0 if np.isnan(N) else N)
        if np.isnan(SUM):
            SUM = 0.0
        denom = c_i[0] if c_i[0] != 0 else 1e-10
        Ne = (1 / denom) * (TEC[i] / p[i] - SUM)
        N_e[i] = Ne if not np.isnan(Ne) else (N_e[i + 1] if i < m - 1 else 0.0)
        i -= 1

    # Smooth transition region
    ts, te = m - num_top - 3, m - num_top + 2
    if ts >= 0 and te < m:
        for i in range(ts, min(te, m)):
            if 0 < i < m - 1:
                N_e[i] = 0.25 * N_e[i - 1] + 0.5 * N_e[i] + 0.25 * N_e[i + 1]

    return N_e


def _lei_abel_invert_m_layers(p_km, TEC_tecu, M=30):
    """
    Multi-layer Abel inversion resampled to M uniform layers.

    Returns
    -------
    N_e : ndarray (M,)
    p_out : ndarray (M,)  — impact radii in km
    TEC_out : ndarray (M,) — TEC in TECU
    """
    p_uni = np.linspace(p_km[0], p_km[-1], M)
    TEC_uni = np.interp(p_uni, p_km, TEC_tecu)

    N_e = np.zeros(M)
    TEC = TEC_uni * 1e12       # TECU → (CGS-like internal units)
    p = p_uni * 1e5            # km → cm

    a, n = 10.0, 10e5
    while a > 0 and n < 200e5:
        if p[-1] - p[0] >= n:
            idx = np.where((p[-1] - p) <= n)
            p_sl, T_sl = (p[idx], TEC[idx]) if len(idx[0]) > 0 else (p, TEC)
        else:
            p_sl, T_sl = p, TEC
        X = p_sl[:, np.newaxis]
        A = np.hstack([X, np.ones_like(X)])
        coeffs, *_ = np.linalg.lstsq(A, T_sl ** 2, rcond=None)
        a, b = coeffs
        if a < 0:
            N_e[-1] = 0.5 * np.sqrt(np.abs(a) / (8 * p_sl[-1])) + 0.5 * np.sqrt(np.abs(b) / (8 * p_sl[-1] ** 2))
        n += 5e5

    i = M - 2
    while i >= 0:
        SUM = 0.0
        P = p[i:M]
        c_i, _, _ = _coef(P)
        M_loc = M - i
        for k in range(1, M_loc):
            N = N_e[i + k]
            SUM += c_i[-M_loc + k] * (0.0 if np.isnan(N) else N)
        if np.isnan(SUM):
            SUM = 0.0
        denom = c_i[0] if c_i[0] != 0 else 1e-10
        Ne = (1 / denom) * (TEC[i] / p[i] - SUM)
        N_e[i] = Ne if not np.isnan(Ne) else N_e[i + 1]
        i -= 1

    return N_e, p * 1e-5, TEC * 1e-12   # back to km / TECU


def _ne_calc_gradient(p_km, TEC_tecu):
    """
    Electron density via numerical Abel integral of the TEC gradient.

    Returns
    -------
    N_e : ndarray (m,) — electron density in e-/m³
    """
    TEC = TEC_tecu * 1e12
    p = p_km * 1e5             # km → cm
    dTEC = np.gradient(TEC) / np.gradient(p)
    N_e = np.full(len(p), np.nan)
    for i in range(len(p) - 2):
        R = p[i]
        idx = np.where(p > R)[0]
        r, dT = p[idx], dTEC[idx]
        valid = ~np.isnan(dT) & ~np.isnan(r)
        if valid.sum() < 2:
            continue
        try:
            A = -(1 / np.pi) * integrate.cumulative_trapezoid(
                dT[valid] / np.sqrt(r[valid] ** 2 - R ** 2), r[valid]
            )
            N_e[i] = A[-1] if len(A) > 0 else np.nan
        except Exception:
            pass
    return N_e


def _tec_cal_schreiner(TEC_in, LEO_km, p_km, tangent_point_km):
    """
    Remove topside TEC contribution using the Schreiner calibration method.

    Parameters
    ----------
    TEC_in : ndarray (n,)   — raw TEC in TECU
    LEO_km : ndarray (3, n) — LEO positions in km
    p_km   : ndarray (n,)   — impact radii in km
    tangent_point_km : ndarray (3, n)

    Returns
    -------
    TEC_cal : ndarray (n,) — calibrated TEC in TECU
    r_LEO   : ndarray (n,) — LEO radii in km
    """
    TEC = TEC_in.copy()
    r_LEO = np.linalg.norm(LEO_km, axis=0)
    r_tan = np.linalg.norm(tangent_point_km, axis=0)

    valid_idx = np.where((np.abs(r_LEO - p_km) <= 1) | (r_LEO <= p_km))[0]
    if len(valid_idx) < 3:
        return TEC, r_LEO

    sort_idx = np.argsort(r_tan[valid_idx])
    r_s = r_tan[valid_idx][sort_idx]
    T_s = TEC[valid_idx][sort_idx]

    unique = np.concatenate(([True], np.diff(r_s) > 1e-6))
    r_s, T_s = r_s[unique], T_s[unique]

    try:
        f_top = interp1d(r_s, T_s, kind='linear', fill_value='extrapolate', bounds_error=False)
    except Exception:
        return TEC, r_LEO

    needs_cal = r_LEO > p_km
    if np.any(needs_cal):
        rt_clip = np.clip(r_tan[needs_cal], r_s.min(), r_s.max())
        TEC_cal = TEC.copy()
        TEC_cal[needs_cal] = np.maximum(TEC[needs_cal] - f_top(rt_clip), 0.0)
        return TEC_cal, r_LEO

    return TEC, r_LEO


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator (mirrors lei_Abel from AuroraRetreivalnew_BC.py)
# ─────────────────────────────────────────────────────────────────────────────

def _lei_abel(TEC_in, LEO_km, GNSS_km, time):
    """
    Full Lei-Abel pipeline: calibrate TEC, clip to valid arc, invert.

    Returns
    -------
    N_e      : ndarray — electron density from discrete inversion (e-/m³)
    N_e_grad : ndarray — electron density from gradient method (e-/m³)
    p        : ndarray — impact radii for N_e / N_e_grad (km)
    TEC_cal  : ndarray — calibrated TEC used (TECU)
    time_sel : ndarray — time array clipped to valid arc
    N_e_m    : ndarray — multi-layer inversion (e-/m³, M=30 layers)
    p_m      : ndarray — impact radii for N_e_m (km)
    """
    tangent_point, p1, _ = rayTangent(LEO_km, GNSS_km, units='km')
    TEC_cal, r_LEO = _tec_cal_schreiner(TEC_in, LEO_km, p1, tangent_point)

    valid = np.where((p1 < r_LEO) & (p1 < np.max(p1)) & (p1 > 0))[0]
    if len(valid) == 0:
        empty = np.full_like(TEC_in, np.nan)
        return empty, empty, p1, TEC_cal, time, np.full(30, np.nan), np.full(30, np.nan)

    p        = p1[valid].copy()
    TEC_sel  = TEC_cal[valid].copy()
    time_sel = time[valid].copy()

    # Trim to monotonically increasing arc
    IDX = len(p) - 1
    _, _, alt_sel = rayTangent(LEO_km[:, valid], GNSS_km[:, valid], units='km')
    for i in range(1, len(alt_sel)):
        if alt_sel[i] < alt_sel[i - 1]:
            IDX = i - 1
            break

    p        = p[:IDX]
    TEC_sel  = TEC_sel[:IDX]
    time_sel = time_sel[:IDX]
    alt_sel  = alt_sel[:IDX]

    # Hermite smooth to zero at the top boundary
    try:
        IDX_sm = np.argmin(np.abs(alt_sel - (alt_sel.max() - 50.0)))
        if IDX_sm < len(TEC_sel) - 2:
            anchor = TEC_sel[IDX_sm]
            dTEC = (TEC_sel[IDX_sm + 1] - anchor) / (alt_sel[IDX_sm + 1] - alt_sel[IDX_sm])
            h_span = alt_sel[-1] - alt_sel[IDX_sm]
            t = (alt_sel[IDX_sm:] - alt_sel[IDX_sm]) / h_span
            h00 = 2 * t ** 3 - 3 * t ** 2 + 1
            h10 = t ** 3 - 2 * t ** 2 + t
            TEC_sel[IDX_sm:] = np.maximum(h00 * anchor + h10 * (dTEC * h_span), 0.0)
            TEC_sel[-1] = 0.0
    except Exception as e:
        print(f"  [Abel] Top smoothing skipped: {e}")

    if len(p) < 4:
        empty = np.full_like(p, np.nan)
        return empty, empty, p, TEC_sel, time_sel, np.full(30, np.nan), np.full(30, np.nan)

    N_e      = _lei_abel_invert(p, TEC_sel)
    N_e_grad = _ne_calc_gradient(p, TEC_sel)
    N_e_m, p_m, _ = _lei_abel_invert_m_layers(p, TEC_sel, M=30)

    return N_e, N_e_grad, p, TEC_sel, time_sel, N_e_m, p_m


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_abel_inversion(podTc2_data):
    """
    Run the Lei-Abel inversion on a parsed podTc2 data dictionary.

    Parameters
    ----------
    podTc2_data : dict
        Output of ``parse_podTc2_nc_file``, or a file path string.
        Must contain keys: 'TEC_podTc2', 'LEO', 'GNSS', 'time'.

    Returns
    -------
    dict with keys:
        'alt_km'      — altitude above WGS84 ellipsoid (km) for Ne / Ne_grad
        'Ne'          — electron density, discrete Lei-Abel method (e-/m³)
        'Ne_grad'     — electron density, gradient method (e-/m³)
        'alt_km_m'    — altitude grid for multi-layer inversion (km)
        'Ne_m'        — electron density, multi-layer inversion (e-/m³)
        'TEC_cal'     — calibrated TEC profile used (TECU)
    or None if the data is invalid.
    """
    if isinstance(podTc2_data, str):
        podTc2_data = parse_podTc2_nc_file(podTc2_data)
    if podTc2_data is None:
        return None

    try:
        TEC_in  = podTc2_data['TEC_podTc2']
        LEO     = podTc2_data['LEO']
        GNSS    = podTc2_data['GNSS']
        time    = podTc2_data['time']
    except KeyError as e:
        print(f"  [Abel] Missing key in podTc2_data: {e}")
        return None

    print("  -> Running Lei-Abel inversion...")
    N_e, N_e_grad, p, TEC_cal, _, N_e_m, p_m = _lei_abel(TEC_in, LEO, GNSS, time)

    # p is Earth-centre distance in km; subtract mean sphere radius for altitude
    EARTH_RADIUS_KM = 6371.0
    alt_km   = p   - EARTH_RADIUS_KM
    alt_km_m = p_m - EARTH_RADIUS_KM

    return {
        'alt_km':   alt_km,
        'Ne':       N_e,
        'Ne_grad':  N_e_grad,
        'alt_km_m': alt_km_m,
        'Ne_m':     N_e_m,
        'TEC_cal':  TEC_cal,
    }
