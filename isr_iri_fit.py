#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
isr_iri_fit.py — Fit a parameterised IRI profile to Millstone Hill ISR data.

Loads ISR electron density sweeps from a Madrigal netCDF file, fits a
piecewise IRI-style profile (via calculate_iri_electron_density) to each
sweep, and saves a two-panel comparison figure per sweep.

Run from the project root:
    python isr_iri_fit.py

Or call the functions directly:
    from isr_iri_fit import fit_iri_to_isr_profile, plot_iri_isr_fit
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EDPSamples" / "Locate in mesh" / "outputs"))
sys.path.insert(0, str(ROOT / "iri2020_new" / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo import extract_robust_f2_peak


# ─────────────────────────────────────────────────────────────────────────────
# ISR site constants (Millstone Hill)
# ─────────────────────────────────────────────────────────────────────────────

ISR_LAT = 42.62   # °N
ISR_LON = 288.51  # °E


# ─────────────────────────────────────────────────────────────────────────────
# §1  Load ISR profiles from a Madrigal netCDF file
# ─────────────────────────────────────────────────────────────────────────────

def load_isr_profiles(isr_files: list[str]) -> list[dict]:
    """
    Load ISR electron density profiles from Millstone Hill Madrigal netCDF
    files.  Returns a list of dicts, one per time sweep, each containing:

        'hour_utc' : float    — UTC hour of the sweep
        'unix_sec' : float    — Unix timestamp of the sweep
        'alt_km'   : ndarray  — altitude grid (km)
        'ne'       : ndarray  — electron density (m⁻³)
        'nm_f2'    : float    — peak Ne (m⁻³)
        'hm_f2'    : float    — peak altitude (km)

    Sweeps with fewer than 10 valid range gates are discarded.
    """
    import netCDF4

    profiles = []
    for fpath in isr_files:
        if not os.path.exists(fpath):
            print(f"  [ISR] File not found, skipping: {fpath}")
            continue
        try:
            ds    = netCDF4.Dataset(fpath, "r")
            alt   = np.array(ds.variables["gdalt"][:])
            nel   = np.ma.filled(np.array(ds.variables["nel"][:]), np.nan)
            times = np.array(ds.variables["timestamps"][:])
            ds.close()

            if nel.ndim == 2 and nel.shape[1] != alt.shape[0]:
                nel = nel.T

            n_time = nel.shape[0]
            for i in range(n_time):
                row   = nel[i, :]
                valid = ~np.isnan(row)
                if valid.sum() < 10:
                    continue
                alt_v    = alt[valid]
                ne_v     = 10.0 ** row[valid]
                t_sec    = float(times[i]) if times.ndim == 1 else float(np.nanmean(times[i, :]))
                hour_utc = (t_sec % 86400) / 3600.0
                nm, hm   = extract_robust_f2_peak(ne_v, alt_v)
                profiles.append({
                    "hour_utc": hour_utc,
                    "unix_sec": t_sec,
                    "alt_km":   alt_v,
                    "ne":       ne_v,
                    "nm_f2":    nm,
                    "hm_f2":    hm,
                })
        except Exception as exc:
            print(f"  [ISR] Error reading {fpath}: {exc}")

    print(f"  Loaded {len(profiles)} ISR sweeps from {len(isr_files)} file(s).")
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# §2  Fit an IRI-style parameterised profile to a single ISR sweep
# ─────────────────────────────────────────────────────────────────────────────

def fit_iri_to_isr_profile(
    isr_profile: dict,
    obs_datetime,
    lat: float = ISR_LAT,
    lon: float = ISR_LON,
    *,
    optimize_e_layer: bool = True,
    e_layer_alt_threshold: float = 140.0,
    verbose: bool = False,
) -> tuple[dict, np.ndarray]:
    """
    Fit a parameterised IRI-style electron density profile to a single ISR sweep.

    Strategy
    --------
    1.  NmF2 and hmF2 are read directly from the ISR data — they are the most
        reliable quantities and are not free parameters.
    2.  IRI is called at (obs_datetime, lat, lon) to obtain physically motivated
        starting guesses for all shape parameters (B0, B1, NmF1, hmF1, NmE,
        hmE, VNER, HEF, C1, NmD, hmD).  Layer peaks are rescaled so the IRI
        NmF2 matches the ISR NmF2 while preserving relative amplitudes.
    3.  The topside (H0, gamma) is pre-fitted with curve_fit against the
        above-peak ISR data using the same Epstein layer used in section1().
    4.  B0, B1, H0, gamma are then refined jointly via Nelder-Mead followed by
        L-BFGS-B to minimise log-space weighted RMSE against the full profile.
    5.  If optimize_e_layer is True and ISR data exist below
        e_layer_alt_threshold km, NmE and hmE are also free parameters.

    Parameters
    ----------
    isr_profile : dict
        A sweep dict from load_isr_profiles.
    obs_datetime : datetime-like
        UTC datetime of the sweep (used to query IRI for shape priors).
    lat, lon : float
        Geodetic latitude (°N) and east longitude (°E) of the ISR site.
    optimize_e_layer : bool
        Optimise NmE and hmE when E-layer data are present.
    e_layer_alt_threshold : float
        Altitude (km) below which data are treated as E-layer information.
    verbose : bool
        Print optimisation summary after fitting.

    Returns
    -------
    iri_params : dict
        Best-fit parameter dict accepted by calculate_iri_electron_density.
    ne_fit : ndarray, shape (n_alt,)
        Reconstructed electron density on the ISR altitude grid.
    """
    from scipy.optimize import minimize, curve_fit
    from iri2020 import IRI
    from IRI_ARR_Samples.iri_arr_samples import calculate_iri_electron_density

    alt_isr = np.asarray(isr_profile["alt_km"], dtype=float)
    ne_isr  = np.asarray(isr_profile["ne"],     dtype=float)

    # ── 1. IRI shape-parameter priors ─────────────────────────────────────────
    alt_min  = max(60.0,   float(alt_isr.min()))
    alt_max  = min(1000.0, float(alt_isr.max()))
    alt_step = max(1.0, round((alt_max - alt_min) / 200.0))
    iri_alts = [alt_min, alt_max, alt_step]   # IRI expects [start, stop, step]

    if not isinstance(obs_datetime, datetime):
        obs_datetime = pd.Timestamp(obs_datetime).to_pydatetime()

    try:
        iono = IRI(obs_datetime, iri_alts, lat, lon)
    except Exception as exc:
        raise RuntimeError(f"IRI call failed for {obs_datetime}: {exc}") from exc

    def _get(name):
        try:
            v = float(iono[name].values.flat[0])
            return v if v > 0 else None
        except Exception:
            return None

    # Use the ISR's rough peak estimate only as an initial guess for NmF2/hmF2.
    # Both are free parameters in the optimisation below.
    isr_nm_f2 = float(isr_profile.get("nm_f2", np.nan))
    isr_hm_f2 = float(isr_profile.get("hm_f2", np.nan))
    iri_NmF2  = _get("NmF2") or 1e11
    iri_hmF2  = _get("hmF2") or 300.0

    # Initial NmF2/hmF2 guess: prefer ISR rough estimate when valid
    init_NmF2 = isr_nm_f2 if (np.isfinite(isr_nm_f2) and isr_nm_f2 > 0) else iri_NmF2
    init_hmF2 = isr_hm_f2 if (np.isfinite(isr_hm_f2) and isr_hm_f2 > 0) else iri_hmF2

    # Scale subordinate IRI layer peaks relative to IRI's own NmF2
    scale = init_NmF2 / iri_NmF2

    def _scaled(name):
        v = _get(name)
        return v * scale if v is not None else None

    init_params = {
        "NMF2":  init_NmF2,
        "HMF2":  init_hmF2,
        "NMF1":  _scaled("NmF1"),
        "HMF1":  _get("hmF1"),
        "NME":   _scaled("NmE"),
        "HME":   _get("hmE"),
        "NMD":   _scaled("NmD"),
        "HMD":   _get("hmD"),
        "B0":    _get("B0"),
        "B1":    _get("B1"),
        "VNER":  _scaled("VNER"),
        "HEF":   _get("HEF"),
        "C1":    _get("C1"),
        "H0":    50.0,
        "gamma": 0.15,
    }

    # ── 2. Exclude the topmost range gate (noisy) from all fitting ────────────
    fit_mask = np.ones(len(alt_isr), dtype=bool)
    fit_mask[np.argmax(alt_isr)] = False
    valid = (ne_isr > 0) & fit_mask
    if valid.sum() < 5:
        raise ValueError("Not enough valid data points in ISR profile to fit.")

    alt_fit    = alt_isr[valid]
    log_ne_obs = np.log(ne_isr[valid])

    # ── 3. Pre-fit topside with curve_fit to seed H0 / gamma ─────────────────
    # Use the initial hmF2 guess to separate topside from bottomside.
    mask_top = (alt_fit >= init_hmF2)
    if mask_top[:-1].sum() >= 3:   # exclude top gate already removed above
        h_top  = alt_fit[mask_top]
        ne_top = ne_isr[valid][mask_top]

        def _epstein_top(h, NmF2_t, hmF2_t, H0, gamma):
            r   = 100.0
            dh  = h - hmF2_t
            H_t = H0 * (1.0 + (r * gamma * dh) / (r * H0 + gamma * dh + 1e-9))
            z   = np.clip(dh / H_t, -100, 100)
            return 4.0 * NmF2_t * np.exp(z) / (1.0 + np.exp(z)) ** 2

        try:
            popt, _ = curve_fit(
                _epstein_top, h_top, ne_top,
                p0=[init_NmF2, init_hmF2, 70.0, 0.15],
                bounds=(
                    [init_NmF2 * 0.1, max(150.0, init_hmF2 - 150.0), 20.0,  1e-3],
                    [init_NmF2 * 10., min(600.0, init_hmF2 + 150.0), 350.0, 1.5],
                ),
                maxfev=8000,
            )
            init_params["NMF2"] = float(popt[0])
            init_params["HMF2"] = float(popt[1])
            init_params["H0"]   = float(popt[2])
            init_params["gamma"]= float(popt[3])
        except RuntimeError:
            pass

    # ── 4. Full joint optimisation ────────────────────────────────────────────
    # Always free: NmF2, hmF2, B0, B1, H0, gamma
    #              VNER, HEF          (E-valley floor and top)
    #              NmF1, hmF1, C1     (F1 layer and ledge)
    # Conditionally free: NmE, hmE   (only when E-layer data present)
    has_e_data = (alt_isr < e_layer_alt_threshold).sum() >= 5
    fit_e      = optimize_e_layer and has_e_data and init_params["NME"] is not None

    # ── Physical bounds ────────────────────────────────────────────────────────
    NE_ABS_LO = 1e8    # sub-D-region noise floor (m⁻³)
    NE_ABS_HI = 1e13   # implausible upper ceiling (m⁻³)
    GAP       = 10.0   # minimum km separation enforced between every layer pair

    # Collect the best available initial guesses for all layer heights so that
    # each layer's bounds can be anchored relative to its neighbours.
    init_VNER = init_params["VNER"] or (NE_ABS_LO * 10)
    init_HEF  = init_params["HEF"]  or 150.0
    init_NmF1 = init_params["NMF1"] or (init_NmF2 * 0.3)
    init_hmF1 = init_params["HMF1"] or 200.0
    init_C1   = init_params["C1"]   or 0.3
    init_hmE  = (init_params["HME"] or 110.0) if fit_e else None

    # ── Layer altitude bounds (each layer bounded above by the one above it) ───
    # hmE  :  95 – (HEF_init – GAP)
    hmE_lo = 95.0
    hmE_hi = min(130.0, init_HEF - GAP)          # must stay below HEF

    # HEF  :  (hmE_init + GAP) – (hmF1_init – GAP)
    HEF_lo = max(100.0, (init_hmE or hmE_lo) + GAP)
    HEF_hi = min(160.0, init_hmF1 - GAP)          # must stay below hmF1

    # hmF1 :  (HEF_init + GAP) – (hmF2_init – GAP)
    hmF1_lo = max(HEF_lo + GAP, init_HEF + GAP)
    hmF1_hi = min(300.0,        init_hmF2 - GAP)  # must stay below hmF2

    # hmF2 :  (hmF1_init + GAP) – 600 km
    hmF2_lo = max(150.0,         init_hmF1 + GAP)
    hmF2_hi = min(600.0,         init_hmF2 + 150.0)

    # ── Density bounds ─────────────────────────────────────────────────────────
    NmF2_lo = max(NE_ABS_LO, init_NmF2 * 0.1)
    NmF2_hi = min(NE_ABS_HI, init_NmF2 * 10.0)

    NmF1_lo = NE_ABS_LO
    NmF1_hi = init_NmF2 * 0.9    # F1 always weaker than F2

    VNER_lo = NE_ABS_LO
    VNER_hi = init_NmF2 * 0.1    # valley minimum well below F2

    NmE_lo  = NE_ABS_LO
    NmE_hi  = init_NmF2 * 0.3    # E-layer always weaker than F2

    # ── Shape / coefficient bounds ─────────────────────────────────────────────
    B0_lo,    B0_hi    = 30.0,  400.0   # IRI bottomside thickness (km)
    B1_lo,    B1_hi    = 0.5,   3.0     # IRI bottomside shape
    H0_lo,    H0_hi    = 20.0,  350.0   # NeQuick topside scale height (km)
    gamma_lo, gamma_hi = 1e-3,  1.5     # NeQuick asymmetry (>1.5 collapses)
    C1_lo,    C1_hi    = 0.0,   1.0     # IRI F1 ledge coefficient

    # ── Slot layout ───────────────────────────────────────────────────────────
    # [NmF2, hmF2, B0, B1, H0, gamma, VNER, HEF, NmF1, hmF1, C1]
    # + optionally [NmE, hmE]
    x0 = [
        np.clip(init_params["NMF2"],          NmF2_lo,  NmF2_hi),
        np.clip(init_params["HMF2"],          hmF2_lo,  hmF2_hi),
        np.clip(init_params["B0"]  or 100.0,  B0_lo,    B0_hi),
        np.clip(init_params["B1"]  or 1.5,    B1_lo,    B1_hi),
        np.clip(init_params["H0"],             H0_lo,    H0_hi),
        np.clip(init_params["gamma"],          gamma_lo, gamma_hi),
        np.clip(init_VNER,                     VNER_lo,  VNER_hi),
        np.clip(init_HEF,                      HEF_lo,   HEF_hi),
        np.clip(init_NmF1,                     NmF1_lo,  NmF1_hi),
        np.clip(init_hmF1,                     hmF1_lo,  hmF1_hi),
        np.clip(init_C1,                       C1_lo,    C1_hi),
    ]
    bounds = [
        (NmF2_lo,  NmF2_hi),
        (hmF2_lo,  hmF2_hi),
        (B0_lo,    B0_hi),
        (B1_lo,    B1_hi),
        (H0_lo,    H0_hi),
        (gamma_lo, gamma_hi),
        (VNER_lo,  VNER_hi),
        (HEF_lo,   HEF_hi),
        (NmF1_lo,  NmF1_hi),
        (hmF1_lo,  hmF1_hi),
        (C1_lo,    C1_hi),
    ]

    if fit_e:
        x0    += [np.clip(init_params["NME"],         NmE_lo, NmE_hi),
                  np.clip(init_params["HME"] or 110.0, hmE_lo, hmE_hi)]
        bounds += [(NmE_lo, NmE_hi), (hmE_lo, hmE_hi)]

    def _clip(x):
        """Clamp every element to its bound — keeps all evaluations physical."""
        return [np.clip(x[i], bounds[i][0], bounds[i][1]) for i in range(len(x))]

    # ── Altitude-dependent weights ────────────────────────────────────────────
    # Up-weight the E-valley / intermediate transition band (hmE to ~hmF1) so
    # the new free parameters for that region are actually pulled by the data.
    hef_ref  = init_HEF
    hmf1_ref = init_hmF1
    weights  = np.ones_like(alt_fit)
    weights[alt_fit < 100.0]                                     = 0.5  # D-region, sparse
    weights[alt_fit > 600.0]                                     = 0.5  # far topside, sparse
    valley_band = (alt_fit >= 100.0) & (alt_fit <= hef_ref)
    inter_band  = (alt_fit > hef_ref) & (alt_fit <= hmf1_ref)
    weights[valley_band] = 2.0   # E-valley floor — previously unconstrained
    weights[inter_band]  = 1.5   # intermediate region

    def _objective(x):
        xc = _clip(x)
        params = dict(init_params)
        params["NMF2"]  = xc[0]
        params["HMF2"]  = xc[1]
        params["B0"]    = xc[2]
        params["B1"]    = xc[3]
        params["H0"]    = xc[4]
        params["gamma"] = xc[5]
        params["VNER"]  = xc[6]
        params["HEF"]   = xc[7]
        params["NMF1"]  = xc[8]
        params["HMF1"]  = xc[9]
        params["C1"]    = xc[10]
        if fit_e:
            params["NME"] = xc[11]
            params["HME"] = xc[12]
        # Enforce strict layer ordering with a minimum GAP separation (km):
        #   hmE + GAP <= HEF + GAP <= hmF1 + GAP <= hmF2
        if params["HMF1"] + GAP > params["HMF2"]:
            return 1e10
        if params["HEF"] + GAP > params["HMF1"]:
            return 1e10
        if fit_e and params["HME"] + GAP > params["HEF"]:
            return 1e10
        try:
            ne_calc = calculate_iri_electron_density(alt_fit, params)
        except Exception:
            return 1e10
        pos = ne_calc > 0
        if pos.sum() < 3:
            return 1e10
        log_calc = np.where(pos, np.log(np.where(pos, ne_calc, 1.0)), 0.0)
        diff = (log_calc - log_ne_obs) * weights
        return float(np.sqrt(np.mean(diff[pos] ** 2)))

    # Nelder-Mead: broad search; _clip inside _objective keeps evaluations physical
    res1 = minimize(
        _objective, x0, method="Nelder-Mead",
        options={"maxiter": 12000, "xatol": 1e-3, "fatol": 1e-4, "disp": False},
    )
    # L-BFGS-B: gradient polish within hard bounds
    res2 = minimize(
        _objective, _clip(res1.x), method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 4000, "ftol": 1e-10},
    )
    best_x = _clip(res2.x if res2.fun < res1.fun else res1.x)

    # ── 5. Assemble final parameter dict ──────────────────────────────────────
    best_params = dict(init_params)
    best_params["NMF2"]  = float(best_x[0])
    best_params["HMF2"]  = float(best_x[1])
    best_params["B0"]    = float(best_x[2])
    best_params["B1"]    = float(best_x[3])
    best_params["H0"]    = float(best_x[4])
    best_params["gamma"] = float(best_x[5])
    best_params["VNER"]  = float(best_x[6])
    best_params["HEF"]   = float(best_x[7])
    best_params["NMF1"]  = float(best_x[8])
    best_params["HMF1"]  = float(best_x[9])
    best_params["C1"]    = float(best_x[10])
    if fit_e:
        best_params["NME"] = float(best_x[11])
        best_params["HME"] = float(best_x[12])

    ne_fit = calculate_iri_electron_density(alt_isr, best_params)
    ne_iri = calculate_iri_electron_density(alt_isr, init_params)

    if verbose:
        log_rmse = _objective(best_x)
        print(
            f"  [fit_iri_to_isr] log-RMSE = {log_rmse:.4f}\n"
            f"    NmF2={best_params['NMF2']:.2e}  hmF2={best_params['HMF2']:.1f} km\n"
            f"    NmF1={best_params['NMF1']:.2e}  hmF1={best_params['HMF1']:.1f} km  C1={best_params['C1']:.3f}\n"
            f"    VNER={best_params['VNER']:.2e}  HEF={best_params['HEF']:.1f} km\n"
            f"    B0={best_params['B0']:.1f} km  B1={best_params['B1']:.3f}  "
            f"H0={best_params['H0']:.1f} km  γ={best_params['gamma']:.4f}"
        )
        if fit_e:
            print(
                f"    NmE={best_params['NME']:.2e} m⁻³  hmE={best_params['HME']:.1f} km"
            )

    return best_params, ne_fit, init_params, ne_iri


# ─────────────────────────────────────────────────────────────────────────────
# §3  Comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_iri_isr_fit(
    isr_profile: dict,
    best_params: dict,
    ne_fit: np.ndarray,
    init_params: dict = None,
    ne_iri: np.ndarray = None,
    *,
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Two-panel figure: EDP overlay (log-x) and relative error (%) vs altitude.

    Parameters
    ----------
    isr_profile : dict
        Original ISR sweep dict (needs 'alt_km', 'ne', 'hour_utc').
    best_params : dict
        Best-fit parameter dict returned by fit_iri_to_isr_profile.
    ne_fit : ndarray
        Best-fit electron density returned by fit_iri_to_isr_profile.
    init_params : dict, optional
        Initial IRI parameter dict (third return value of fit_iri_to_isr_profile).
        When provided, the IRI prior profile and its control points are also plotted.
    ne_iri : ndarray, optional
        IRI prior electron density (fourth return value of fit_iri_to_isr_profile).
    save_path : str, optional
        If given, save the figure to this path.
    show : bool
        Call plt.show() after plotting.
    """
    alt_isr = isr_profile["alt_km"]
    ne_isr  = isr_profile["ne"]
    hour    = isr_profile.get("hour_utc", np.nan)

    valid = ne_isr > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(
            valid,
            (ne_fit - ne_isr) / ne_isr * 100.0,
            np.nan,
        )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 9), sharey=True,
        gridspec_kw={"width_ratios": [3, 1]},
    )
    time_str = (
        f"{int(hour):02d}:{int((hour % 1) * 60):02d} UTC"
        if not np.isnan(hour) else "unknown"
    )
    fig.suptitle(f"IRI Best-Fit vs. MH ISR  ({time_str})", fontsize=13)

    # ── Main profiles ──────────────────────────────────────────────────────────
    ax1.plot(ne_isr, alt_isr, color="steelblue", lw=2.5, label="MH ISR measured")
    if ne_iri is not None:
        ax1.plot(ne_iri, alt_isr, color="gray", lw=1.8, ls=":", label="IRI prior")
    ax1.plot(ne_fit, alt_isr, color="crimson", lw=2.0, ls="--", label="IRI best-fit")

    # ── Control-point markers ──────────────────────────────────────────────────
    # Each entry: (Ne key, h key, marker, colour)
    cp_spec = [
        ("NMF2", "HMF2", "*", "crimson"),
        ("NMF1", "HMF1", "^", "darkorange"),
        ("NME",  "HME",  "s", "seagreen"),
    ]
    for ne_key, h_key, mkr, col in cp_spec:
        # Best-fit marker (filled)
        nm = best_params.get(ne_key)
        hm = best_params.get(h_key)
        if nm and hm:
            ax1.plot(nm, hm, marker=mkr, ms=12, color=col,
                     mec="black", mew=0.8, zorder=7)
            ax1.axhline(hm, color=col, ls=":", lw=0.9, alpha=0.6)
            ax2.axhline(hm, color=col, ls=":", lw=0.9, alpha=0.6)
        # IRI prior marker (hollow), if available
        if init_params is not None:
            nm0 = init_params.get(ne_key)
            hm0 = init_params.get(h_key)
            if nm0 and hm0:
                ax1.plot(nm0, hm0, marker=mkr, ms=12, color="gray",
                         mec="gray", mew=1.2, mfc="none", zorder=6)

    # ax1.set_xscale("log")
    ax1.set_xlabel("Electron Density (m⁻³)", fontsize=11)
    ax1.set_ylabel("Altitude (km)", fontsize=11)
    ax1.set_title("EDP Comparison", fontsize=11)
    ax1.grid(True, which="major", ls="-", alpha=0.5)
    ax1.grid(True, which="minor", ls=":", alpha=0.3)
    ax1.legend(fontsize=9, loc="upper right")

    # ── Parameter table: IRI prior | Best-fit, side by side ───────────────────
    def _fmt(v, density=False):
        if v is None:
            return "—"
        return f"{v:.2e}" if density else f"{v:.2f}"

    # Rows: (display name, params key, is_density)
    row_spec = [
        ("NmF2", "NMF2", True),
        ("hmF2", "HMF2", False),
        ("NmF1", "NMF1", True),
        ("hmF1", "HMF1", False),
        ("NmE",  "NME",  True),
        ("hmE",  "HME",  False),
        ("VNER", "VNER", True),
        ("HEF",  "HEF",  False),
        ("B0",   "B0",   False),
        ("B1",   "B1",   False),
        ("H0",   "H0",   False),
        ("γ",    "gamma",False),
        ("C1",   "C1",   False),
    ]

    show_prior = init_params is not None
    if show_prior:
        hdr = f"{'Param':<6}  {'IRI Prior':>10}  {'Best-Fit':>10}  {'Unit'}\n" + "─" * 42 + "\n"
    else:
        hdr = f"{'Param':<6}  {'Best-Fit':>10}  {'Unit'}\n" + "─" * 30 + "\n"

    units = {"NMF2": "m⁻³", "HMF2": "km", "NMF1": "m⁻³", "HMF1": "km",
             "NME": "m⁻³", "HME": "km", "VNER": "m⁻³", "HEF": "km",
             "B0": "km", "B1": "", "H0": "km", "gamma": "", "C1": ""}

    table_text = hdr
    for name, key, is_dens in row_spec:
        fit_val  = _fmt(best_params.get(key), is_dens)
        unit     = units.get(key, "")
        if show_prior:
            pri_val = _fmt(init_params.get(key), is_dens)
            table_text += f"{name:<6}  {pri_val:>10}  {fit_val:>10}  {unit}\n"
        else:
            table_text += f"{name:<6}  {fit_val:>10}  {unit}\n"

    ax1.text(
        0.02, 0.02, table_text,
        transform=ax1.transAxes,
        fontsize=7.5,
        verticalalignment="bottom",
        horizontalalignment="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.85),
    )

    # ── Error panel ───────────────────────────────────────────────────────────
    ax2.plot(rel_err, alt_isr, color="purple", lw=1.8)
    ax2.axvline(0, color="black", ls="--", lw=1.2)
    ax2.set_xlabel("Error (%)", fontsize=11)
    ax2.set_title("Relative Error", fontsize=11)
    ax2.grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Main — run over all sweeps in the configured ISR file
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── User settings ──────────────────────────────────────────────────────────
    ISR_FILES = [
        "./DataFiles/EDPS/mlh250603m.002.nc",
    ]
    # Date of the ISR file (used to build the datetime for each sweep)
    ISR_DATE = datetime(2025, 6, 3)

    SAVE_DIR    = "./Figures/ISR_IRI_Fit/"
    VERBOSE     = True
    OPT_E_LAYER = True     # also optimise NmE / hmE when E-layer data present
    # ──────────────────────────────────────────────────────────────────────────

    profiles = load_isr_profiles(ISR_FILES)
    if not profiles:
        print("No ISR profiles loaded.  Check ISR_FILES paths.")
        return

    print(f"\nFitting IRI to {len(profiles)} ISR sweep(s) …\n")

    for i, prof in enumerate(profiles):
        hour    = prof["hour_utc"]
        obs_dt  = ISR_DATE + timedelta(hours=hour)
        tag     = f"{ISR_DATE.strftime('%Y%m%d')}_{int(hour):02d}h{int((hour % 1) * 60):02d}m"

        print(f"[{i+1}/{len(profiles)}] {tag} UTC  NmF2={prof['nm_f2']:.2e}  hmF2={prof['hm_f2']:.0f} km")

        try:
            params, ne_fit, init_params, ne_iri = fit_iri_to_isr_profile(
                prof, obs_dt,
                optimize_e_layer=OPT_E_LAYER,
                verbose=VERBOSE,
            )
            plot_iri_isr_fit(
                prof, params, ne_fit, init_params, ne_iri,
                save_path=os.path.join(SAVE_DIR, f"iri_fit_{tag}.png"),
                show=False,
            )
        except Exception as exc:
            print(f"  [warn] Sweep {tag} failed: {exc}")


if __name__ == "__main__":
    main()
