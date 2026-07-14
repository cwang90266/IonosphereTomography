#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
test_param_iono.py — Validation and smoothness tests for the parameterized
ionosphere forward model in enkf_update.py / observation_operator.py.

Workflow
--------
1. Scan the day directory for podTc2 occultation files once, bin qualifying
   arcs into WINDOW_MINUTES-wide windows (keyed on TEC-max epoch), then apply
   the ISR ROI gate + round-robin constellation selection per window. Windows
   with fewer than MIN_ARCS_PER_WINDOW arcs are dropped.
2. For every retained window (main() loops over them), rebuild the geometry-
   dependent Fibonacci grids from that window's arcs, then run the full
   time-dependent pipeline:
     • 1-deg + 5-deg Fibonacci grids (anchored on arc tangent tracks +
       MST/SLERP waypoints for connectivity).
     • IRI truth ensemble (9 deterministic +1σ members) on the 1-deg grid.
     • Stochastic IRI ensemble (N_MEMBERS members) on the 5-deg grid.
     • Parametric forward model (ObservationOperator) with IDW interpolation
       (12 nearest neighbours) for every arc on both grids.
     • Gridded Ne linear KF + iterative EKF_Param on the parametric state.
3. Every per-window figure is suffixed with the window's "_HHMM" tag so
   successive windows never overwrite each other. Per-window results (RMSE,
   filter outputs, arc/ray metadata) are collected into a dict keyed by
   window_key for downstream cross-window comparison plotting (separate).

Filenames (all suffixed "_{YYYY}_{DOY}_{HHMM}"):
     • param_iono_test_…             — 2×2 TEC panels + globe map
     • ensemble_histograms_…         — 8×1 per-parameter histograms
     • param_sensitivity[_abs]_…     — 8×2 sensitivity sweeps
     • kf_covariance_…, kf_tec_edp_…, edp_spatial_error_kf_…
     • ekf_param_tec_edp_…, edp_spatial_error_ekf_param_…
     • kf_vs_ekf_param_comparison_…
"""

from __future__ import annotations

import sys
import os
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "iri2020_new" / "src"))

import warnings
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, SymLogNorm

# NumPy >=2.0 removed np.trapz (renamed to np.trapezoid); NumPy <2.0 doesn't
# have np.trapezoid yet. Resolve once at import time so call sites work
# under either version.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj
import netCDF4
from scipy.spatial import cKDTree
from tqdm import tqdm

from TEC_model.podTc_file_processing import parse_podTc2_nc_file
from Ionosphere_Tomography_Inverter.ionospheric_state import (
    IonosphericState, N_STATE, PARAM_NAMES, LOG_INDICES,
    I_LOG_NMF2, I_HMF2, I_H0, I_GAMMA, I_B0, I_B1, I_LOG_NME, I_HME,
)
from Ionosphere_Tomography_Inverter.observation_operator import ObservationOperator
from Ionosphere_Tomography_Inverter.enkf_update import (
    _haversine_km, ParametricEnKF, build_ray_localisation_matrix,
)
from demo_compare_kf_enkf import (
    _build_gnss_to_leo_ray, _tangent_latlon_single, _solar_sampling_df,
    _get_iri_edp_and_features_batch, _state_from_iri_direct,
    _default_background_covariance, _TRANSFORMER,
    _parametric_to_edp, _parametric_to_edp_ensemble,
    _precompute_ray_phi, _plot_arc_innovation_diagnostic,
    _idw_weights as _idw_weights_enkf,
    _OPT_METHOD_STYLES, _OPT_BOUNDS_PER_PARAM,
    _optimize_grid_point, _run_parametric_optimization,
    _draw_param_boxes,
)
from demo_group import CONSTELLATION_CONFIG, _CONST_FALLBACK_CMAP
from demo_isr_initial_conditions import INSTRUMENTS
from demo import extract_robust_f2_peak


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-domain metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def ne_to_mhz(ne_m3: np.ndarray) -> np.ndarray:
    """Electron density [m⁻³] → plasma frequency [MHz]."""
    return 8.978e-6 * np.sqrt(np.maximum(np.asarray(ne_m3, dtype=float), 0.0))


def extract_e_layer_peak(ne_arr: np.ndarray, alt_arr: np.ndarray,
                          e_alt_min: float = 90.0,
                          e_alt_max: float = 150.0) -> tuple[float, float]:
    """
    Return (NmE [m⁻³], hmE [km]) — the peak electron density in the E-layer
    altitude band [e_alt_min, e_alt_max] km. Returns (nan, nan) if the band
    contains no valid points.
    """
    ne_arr  = np.asarray(ne_arr, dtype=float)
    alt_arr = np.asarray(alt_arr, dtype=float)
    mask = (alt_arr >= e_alt_min) & (alt_arr <= e_alt_max) & np.isfinite(ne_arr)
    if mask.sum() == 0:
        return np.nan, np.nan
    band_ne  = ne_arr[mask]
    band_alt = alt_arr[mask]
    idx = int(np.argmax(band_ne))
    return float(band_ne[idx]), float(band_alt[idx])


def compute_retrieval_freq_metrics(
    truth_ne: np.ndarray,          # (n_alt,) truth profile at one grid point
    prior_ne: np.ndarray,          # (n_alt,)
    post_ne:  np.ndarray,          # (n_alt,)
    alt_grid: np.ndarray,          # (n_alt,) km
    mhz_thresholds: tuple = (0.5, 0.2, 0.1),
) -> dict:
    """
    Compute frequency-domain retrieval metrics comparing prior/posterior Ne
    profiles against truth at a single grid point.

    Returns a dict with:
        truth_foF2, prior_foF2, post_foF2          [MHz]
        truth_foE,  prior_foE,  post_foE           [MHz]
        prior_foF2_err, post_foF2_err              [MHz, signed: est - truth]
        prior_foE_err,  post_foE_err               [MHz]
        prior_fp_rmse,  post_fp_rmse               [MHz, profile RMSE in plasma freq]
        within_{X}mhz_foF2_prior/post              [bool, per threshold]
        within_{X}mhz_foE_prior/post               [bool]
        within_{X}mhz_profile_prior/post           [bool]
    """
    truth_ne = np.asarray(truth_ne, dtype=float)
    prior_ne = np.asarray(prior_ne, dtype=float)
    post_ne  = np.asarray(post_ne,  dtype=float)
    alt_grid = np.asarray(alt_grid, dtype=float)

    # F2 peak
    truth_nmf2, _ = extract_robust_f2_peak(truth_ne, alt_grid)
    prior_nmf2, _ = extract_robust_f2_peak(prior_ne, alt_grid)
    post_nmf2,  _ = extract_robust_f2_peak(post_ne,  alt_grid)

    truth_foF2 = ne_to_mhz(truth_nmf2)
    prior_foF2 = ne_to_mhz(prior_nmf2)
    post_foF2  = ne_to_mhz(post_nmf2)

    # E layer peak
    truth_nme, _ = extract_e_layer_peak(truth_ne, alt_grid)
    prior_nme, _ = extract_e_layer_peak(prior_ne, alt_grid)
    post_nme,  _ = extract_e_layer_peak(post_ne,  alt_grid)

    truth_foE = ne_to_mhz(truth_nme)
    prior_foE = ne_to_mhz(prior_nme)
    post_foE  = ne_to_mhz(post_nme)

    # Full-profile plasma-frequency RMSE
    truth_fp = ne_to_mhz(truth_ne)
    prior_fp = ne_to_mhz(prior_ne)
    post_fp  = ne_to_mhz(post_ne)
    valid = np.isfinite(truth_fp) & np.isfinite(prior_fp) & np.isfinite(post_fp)
    prior_fp_rmse = float(np.sqrt(np.mean((prior_fp[valid] - truth_fp[valid])**2))) if valid.any() else np.nan
    post_fp_rmse  = float(np.sqrt(np.mean((post_fp[valid]  - truth_fp[valid])**2))) if valid.any() else np.nan

    out = dict(
        truth_foF2=float(truth_foF2), prior_foF2=float(prior_foF2), post_foF2=float(post_foF2),
        truth_foE=float(truth_foE),   prior_foE=float(prior_foE),   post_foE=float(post_foE),
        prior_foF2_err=float(prior_foF2 - truth_foF2),
        post_foF2_err =float(post_foF2  - truth_foF2),
        prior_foE_err =float(prior_foE  - truth_foE),
        post_foE_err  =float(post_foE   - truth_foE),
        prior_fp_rmse=prior_fp_rmse,
        post_fp_rmse =post_fp_rmse,
    )
    for thr in mhz_thresholds:
        thr_str = str(thr).replace(".", "")
        out[f"within_{thr_str}mhz_foF2_prior"] = bool(abs(out["prior_foF2_err"]) <= thr)
        out[f"within_{thr_str}mhz_foF2_post"]  = bool(abs(out["post_foF2_err"])  <= thr)
        out[f"within_{thr_str}mhz_foE_prior"]  = bool(abs(out["prior_foE_err"])  <= thr)
        out[f"within_{thr_str}mhz_foE_post"]   = bool(abs(out["post_foE_err"])   <= thr)
        out[f"within_{thr_str}mhz_profile_prior"] = bool(prior_fp_rmse <= thr)
        out[f"within_{thr_str}mhz_profile_post"]  = bool(post_fp_rmse  <= thr)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# §0  Configuration
# ─────────────────────────────────────────────────────────────────────────────

# ┌─────────────────────────────────────────────────────────────────────────┐
# │                        FILTER CONTROL PANEL                             │
# │  Set each flag to True / False to enable or disable that method.        │
# │  All methods share the same truth ionosphere, synthetic TEC, and prior. │
# └─────────────────────────────────────────────────────────────────────────┘
RUN_KF        = True   # Gridded Ne linear Kalman Filter (Ne-space, single-step)
RUN_EKF_PARAM = True   # Iterative Extended KF on parametric IRI state

DOY           = 265
YYYY          = 2025
BASE_PATH     = (f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/"
                 f"{YYYY}.{DOY}/")
SAVE_DIR      = "./Figures/test_param_iono/"
IRI_CACHE_DIR = "./Data/IRI_param_cache/"

# ── §0b  Multi-date sweep configuration ─────────────────────────────────────
# Each entry is (YYYY, DOY, description) defining which day directories to test.
# Cover: winter solstice, spring equinox, summer solstice, autumn equinox +
# the nominal default day → captures seasonal variation.
SWEEP_DATES = [
    # (year, doy, description)
    # (2025,  1,  ""),
    (2026, 43,  "late_winter_spring"),
    (2025, 172, "summer_solstice"),
    (2025, 265, "autumn_equinox"),     
    (2025, 352, "winter_solstice"),
]

# N_OCC sweep range
SWEEP_N_OCC_VALUES = list(range(10, 101, 10))   # [10, 20, 30, …, 100]

# Where to save sweep results
SWEEP_RESULTS_CSV = "./Data/occ_sweep_results.csv"
SWEEP_SAVE_DIR    = "./Figures/occ_sweep/"

# Root directory holding per-day podTc2 subfolders ("{YYYY}.{DOY}/").
SWEEP_PODTC2_ROOT = "/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2"

ALT_MIN_TEC   = 100.0   # km — integration band for measured TEC (lower bound)
ALT_MAX_TEC   = 400.0   # km — integration band for measured TEC (upper bound)

N_OCC_MAX     = 30     # maximum number of occultations to process per window

ISR_SITES      = ("ESR", "TRO")
ISR_ROI_MAX_KM = 2500.0   # RO peak-tangent-point → ISR site gate (great-circle km)

# ── Simulated IGS ground-station observations ────────────────────────────────
# Extends arc_truth_list with synthetic ground-to-GNSS sTEC observations at
# fixed Nordic IGS station coordinates.  Satellite positions are taken from a
# real broadcast ephemeris (BRDC RINEX file in Data/RINEX_Cache), so each
# (station, SV) arc mirrors the geometry an actual IGS receiver would see.
# The truth state is forward-modelled through those rays the same way as RO
# arcs, so the KF and EKF see a longer arc_truth_list without any change to
# their update logic.  Set USE_SIMULATED_IGS = False to disable.
USE_SIMULATED_IGS       = True
IGS_SIM_STATIONS        = ["TRO1", "WUTH", "NYA1", "KIR0", "SOD3", "ALRT","SCOR", "HOFN", 'REYK']  # 4-char prefixes into IGSNetwork.json
IGS_SIM_STATIONS_JSON   = "./Data/IGS_Stations/IGSNetwork.json"
IGS_SIM_ELEV_CUTOFF_DEG = 20.0   # visibility cutoff — SVs below this elevation are skipped
IGS_SIM_N_EPOCHS        = 8      # target sample epochs per (station, SV) arc,
                                 # spread across the window centred on the arc's time
IGS_SIM_N_RAY_PTS       = 500    # ECEF samples per ray before altitude filtering
                                 # (large because the LOS from receiver to GNSS SV is
                                 # ~20 000+ km long but only ~1000 km sits in the ionosphere)
IGS_SIM_IPP_ALT_KM      = 300.0  # nominal ionospheric pierce altitude for tp_lat/tp_lon
IGS_SIM_CONID           = "I"    # constellation tag (falls back to Greys cmap)
IGS_SIM_CONSTELLATIONS  = ("G", "R", "E", "C")  # which SV prefixes to include
IGS_SIM_BRDC_DIR        = "./Data/RINEX_Cache"

# Observation-mode sweep — the filter suite is run once per mode with per-mode
# figures suffixed "_{mode}".  igs_only rebuilds the Fibonacci grid from IGS
# pierce points so its ROI is IGS-centred rather than RO-centred.
FILTER_MODES = ("ro_only", "ro_igs", "igs_only")

# ── Multi-window sweep across the day ────────────────────────────────────────
# The pipeline is executed once per discrete time window across the target day.
# WINDOW_MINUTES controls the binning width (30 matches demo_group.py). Given
# the ISR ROI gate typically leaves only ~30 arcs/day, most 30-min windows are
# sparse; MIN_ARCS_PER_WINDOW filters out windows too thin for meaningful
# assimilation. Coarsen WINDOW_MINUTES (e.g. 60 or 180) to process fewer,
# denser windows.
WINDOW_MINUTES        = 60   # per-window bin width (minutes)
MIN_ARCS_PER_WINDOW   = 16    # skip windows with fewer ROI-passing arcs

# Altitude grid for Ne integration (log-spaced 60–800 km)
ALT_GRID = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)

GRID_MARGIN_DEG = 15.0   # degrees of margin added around arc tangent tracks
CORR_LENGTH_KM  = 500.0  # exponential spatial correlation length (km)

N_MEMBERS  = 200   # stochastic model ensemble size
MAX_EPOCHS = 200   # maximum ray epochs per arc (decimated if more)

# Per-parameter 1-σ perturbation for the truth ensemble members
# log10(NmF2), hmF2, H0, gamma, B0, B1, log10(NmE), hmE
_TRUTH_SIGMA = np.array([1.7, 38.0, 33.0, 0.2, 23.0, 0.30, -1.4, 12.0])


# Gaussian kernel interpolation settings (applied identically to both 1-deg and 5-deg grids)
_IDW_NEAREST    = 12   # node cap in _idw_weights; also used for min_points in Fibonacci grids
_IDW_SIGMA_SCALE = 1.5  # sigma = SIGMA_SCALE × distance-to-3rd-nearest (local density estimate)

# Parameter sensitivity sweep
_N_SWEEP = 61      # number of sweep points (odd → symmetric about baseline)
_N_SIGMA = 5.0     # sweep range in units of _TRUTH_SIGMA

# Constellation panel layout (matches demo_group.CONSTELLATION_CONFIG)
_CONST_POS = {"G": (0, 0), "R": (1, 0), "E": (0, 1), "C": (1, 1)}

# ── §7  EnKF retrieval experiment ─────────────────────────────────────────────
# Truth ionosphere is IRI evaluated +TRUTH_HOUR_OFFSET hours after the arc
# representative time, with F10.7 increased by TRUTH_F107_DELTA solar-flux units.
TRUTH_HOUR_OFFSET    = 2       # hours added to mean time for truth ionosphere
# A +10 F10.7 increment is a realistic active-to-moderate solar-activity step
# that gives ~15–30 TECU innovations — large enough to test the filter but
# small enough to stay within the stochastic EnKF's linear regime.
TRUTH_F107_DELTA     = 15   # solar flux unit increment for truth conditions

ENKF_N_MEMBERS       = 500     # EnKF ensemble size (model prior)
ENKF_LOC_RADIUS_KM   = 200.0   # Gaspari-Cohn half-support radius (km)
# sigma_obs must be large enough that R_mda = n_mda * sigma^2 keeps the
# Kalman gain moderate (members don't saturate physical bounds after step 1).
ENKF_SIGMA_OBS       = 15.0     # observation noise std-dev (TECU)
ENKF_INFLATION       = 1.0     # multiplicative prior-ensemble inflation
ENKF_N_MDA           = 4       # ES-MDA iterations (1 = standard single-step EnKF)
# Keep update rays << n_members to avoid rank-deficiency in D = P_yy + R.
# ~10 levels per arc gives ~80 total for 8 arcs, well within the 200-member rank.
ENKF_MAX_UPDATE_RAYS = 50      # maximum rays per arc used in the EnKF update
ENKF_MAX_UPDATE_STEP = 1.0     # per-element log-space update clip
ENKF_APPLY_BOUNDS    = True    # clamp ensemble members to physical bounds after each update
                               # set False to diagnose whether clamping distorts the posterior

# Altitudes (km) for the 5×2 EDP spatial-error orthographic plots
ERROR_ALTITUDES      = [100, 200, 300, 400, 500]

# ── §9b  EKF_Param — iterative EKF on the parametric IRI state ───────────────
EKF_PARAM_ALPHA       = 0.8    # step-size scale (1.0 = full Kalman step; <1 is more conservative)
EKF_PARAM_TOL         = 5e-4   # convergence: stop when ||ΔP||/||P|| < tol
EKF_PARAM_MAX_ITER    = 20     # maximum EKF iterations
EKF_PARAM_SIGMA_OBS   = ENKF_SIGMA_OBS        # observation noise std-dev (TECU)
EKF_PARAM_UPDATE_RAYS = ENKF_MAX_UPDATE_RAYS   # representative rays per arc
EKF_PARAM_APPLY_BOUNDS = True  # clamp parameters to physical bounds after each step
EKF_PARAM_EPS_JAC     = 1e-3   # relative step for finite-difference Jacobian
EKF_PARAM_N_WORKERS   = 10      # thread-pool size: ray-loop parallelism + vectorised Jacobian
EKF_PARAM_JAC_ANALYTICAL = True  # True → analytical Jacobian; False → finite-difference

# ── §15  scipy non-linear optimisation ────────────────────────────────────────
SCIPY_OPT_N_WORKERS   = 4      # thread-pool size for grid-point parallelism
SCIPY_OPT_MAXITER_UNC = 300    # iteration cap for BFGS / Nelder-Mead / Newton-CG
SCIPY_OPT_MAXITER_CON = 600    # iteration cap for SLSQP / trust-constr
SCIPY_OPT_SIGMA_OBS   = ENKF_SIGMA_OBS        # observation noise std-dev (TECU)
SCIPY_OPT_UPDATE_RAYS = ENKF_MAX_UPDATE_RAYS   # representative rays per arc


# ─────────────────────────────────────────────────────────────────────────────
# §1  File scanning and selection
# ─────────────────────────────────────────────────────────────────────────────

def _near_isr_site_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    sites: tuple[str, ...] = ISR_SITES,
    max_km: float = ISR_ROI_MAX_KM,
) -> np.ndarray:
    """True where (lat, lon) is within max_km of at least one ISR site."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    within = np.zeros(lat.shape, dtype=bool)
    for site in sites:
        inst = INSTRUMENTS[site]
        dist_km = _haversine_km(inst["lat"], inst["lon"], lat, lon)
        within |= (dist_km <= max_km)
    return within


def _round_robin_by_constellation(
    candidates: list[dict],
    max_files: int,
) -> list[dict]:
    """
    Distribute *candidates* across GNSS constellations (G/R/E/C) in
    round-robin order, capping at *max_files*.  Order preserved within each
    per-constellation queue.
    """
    const_pool: dict[str, list] = defaultdict(list)
    for r in candidates:
        const_pool[r["conid"]].append(r)

    queues = [list(const_pool[c])
              for c in sorted(const_pool, key=lambda c: -len(const_pool[c]))]
    selected: list = []
    while len(selected) < max_files and any(queues):
        for q in queues:
            if q and len(selected) < max_files:
                selected.append(q.pop(0))
        queues = [q for q in queues if q]
    return selected


def scan_and_select_files_per_window(
    base_path: str,
    alt_min: float        = ALT_MIN_TEC,
    alt_max: float        = ALT_MAX_TEC,
    alt_min_tangent: float = 90.0,
    max_files: int        = N_OCC_MAX,
    file_suffix: str      = ".0001_nc",
    time_window_min: int  = WINDOW_MINUTES,
    min_arcs_per_window: int = MIN_ARCS_PER_WINDOW,
) -> list[dict]:
    """
    Scan *base_path* for podTc2 netCDF files, filter to those whose tangent
    altitude range overlaps [alt_min, alt_max] km AND whose full arc probes
    below *alt_min_tangent* km, then bin into *time_window_min*-minute windows
    keyed on the TEC-max epoch datetime.  For each window: apply the ISR ROI
    gate and select up to *max_files* records via round-robin over GNSS
    constellations.  Windows with fewer than *min_arcs_per_window* ROI-passing
    arcs are dropped.

    Parameters
    ----------
    alt_min_tangent : float
        Require the arc's minimum tangent altitude to be below this value (km).
        Ensures the ray path probes deep enough into the lower ionosphere /
        upper mesosphere.  Default 90 km.
    time_window_min : int
        Bin width in minutes.  30 matches demo_group.py's convention.
    min_arcs_per_window : int
        Minimum number of ROI-passing arcs required for a window to be
        retained.  KF/EKF assimilation needs several arcs for meaningful
        geometric coverage; sparser windows are silently dropped.

    Returns
    -------
    windows : list of dicts, one per retained window, sorted chronologically.
        Each dict has keys:
            window_key : "YYYY-MM-DD_HHMM"  (window start, floored to the bin)
            hhmm       : "HHMM"             (for filename suffixing)
            time_dt    : pd.Timestamp       (window start; also the IRI
                                             evaluation time for this window)
            records    : list[dict]         (selected arc records)
    """
    records = []
    for fname in sorted(os.listdir(base_path)):
        if not fname.endswith(file_suffix):
            continue
        fpath = os.path.join(base_path, fname)
        try:
            ds = netCDF4.Dataset(fpath, "r")
            # Global attributes
            slta_tec = float(ds.getncattr("slta_tecmax_tangent"))
            lat_tec  = float(ds.getncattr("lat_tecmax_tangent"))
            lon_tec  = float(ds.getncattr("lon_tecmax_tangent"))
            yr       = int(ds.getncattr("year"))
            mo       = int(ds.getncattr("month"))
            dy       = int(ds.getncattr("day"))
            hh       = int(ds.getncattr("hour"))
            mm       = int(ds.getncattr("minute"))
            ss       = int(ds.getncattr("second"))
            prn_id   = str(ds.getncattr("prn_id"))
            leo_id   = str(ds.getncattr("leo_id"))
            conid    = str(ds.getncattr("conid"))

            # Time and TEC variables — used to find the TEC-max epoch datetime
            time_arr = np.asarray(ds.variables["time"][:],  dtype=float)
            tec_arr  = np.asarray(ds.variables["TEC"][:],   dtype=float)

            # Position arrays (km) — used to compute minimum tangent altitude
            x_leo = np.asarray(ds.variables["x_LEO"][:], dtype=float)
            y_leo = np.asarray(ds.variables["y_LEO"][:], dtype=float)
            z_leo = np.asarray(ds.variables["z_LEO"][:], dtype=float)
            x_gps = np.asarray(ds.variables["x_GPS"][:], dtype=float)
            y_gps = np.asarray(ds.variables["y_GPS"][:], dtype=float)
            z_gps = np.asarray(ds.variables["z_GPS"][:], dtype=float)
            ds.close()
        except Exception as exc:
            warnings.warn(f"Could not read {fpath}: {exc}")
            continue

        # slta_tecmax_tangent may be in metres — convert if needed
        slta_km = slta_tec / 1000.0 if slta_tec > 1000.0 else slta_tec

        # Constellation letter
        conid = conid.upper()
        if conid not in "GREC":
            conid = prn_id[0].upper() if prn_id[0].upper() in "GREC" else "G"

        # Filter 1: TEC-max tangent altitude within the integration band
        if not (alt_min <= slta_km <= alt_max):
            continue

        # Filter 2: arc must probe below alt_min_tangent km.
        # Compute per-epoch tangent altitudes with a spherical approximation
        # (R_E = 6371 km) — accurate to <1 km for this screening purpose.
        leo  = np.array([x_leo, y_leo, z_leo])   # (3, n)
        gnss = np.array([x_gps, y_gps, z_gps])   # (3, n)
        v    = leo - gnss
        denom = np.einsum("ij,ij->j", v, v)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        t_s   = np.clip(-np.einsum("ij,ij->j", gnss, v) / denom, 0.0, 1.0)
        tang_pt     = gnss + v * t_s[np.newaxis, :]
        tang_alt_km = np.linalg.norm(tang_pt, axis=0) - 6371.0
        min_tang_km = float(np.min(tang_alt_km))

        if min_tang_km > alt_min_tangent:
            continue

        # File start datetime (from year/month/day/hour/minute/second attributes)
        start_dt = pd.Timestamp(yr, mo, dy, hh, mm, ss)

        # TEC-max epoch datetime.
        # time_arr uses GPS seconds (epoch 1980-01-06), not Unix time, so we
        # derive the TEC-max time as an elapsed offset from the file start
        # rather than converting the absolute GPS timestamp.
        tec_max_idx    = int(np.argmax(tec_arr))
        tec_max_offset = float(time_arr[tec_max_idx] - time_arr[0])
        tec_max_dt     = start_dt + pd.Timedelta(seconds=tec_max_offset)

        records.append(dict(
            path        = fpath,
            leo_id      = leo_id,
            prn_id      = prn_id,
            conid       = conid,
            lat         = lat_tec,
            lon         = lon_tec,
            slta        = slta_km,
            min_tang_km = min_tang_km,
            date        = start_dt,      # file start time
            tec_max_dt  = tec_max_dt,    # datetime of TEC maximum within arc
        ))

    if not records:
        raise RuntimeError(f"No qualifying podTc2 files found in {base_path}")

    # Bin records by TEC-max epoch into time_window_min windows.  Keying on
    # tec_max_dt (rather than file start) colocates arcs whose ionospheric
    # peak falls in the same bin.
    def _window_slot_start(t: pd.Timestamp) -> pd.Timestamp:
        total_min = t.hour * 60 + t.minute
        floored   = (total_min // time_window_min) * time_window_min
        h, m      = divmod(floored, 60)
        return pd.Timestamp(t.year, t.month, t.day, int(h), int(m))

    groups: dict[pd.Timestamp, list] = defaultdict(list)
    for r in records:
        groups[_window_slot_start(r["tec_max_dt"])].append(r)

    print(f"  Qualifying arcs: {len(records)} across {len(groups)} "
          f"{time_window_min}-min bins (before ROI gate)")

    windows: list[dict] = []
    kept_arcs_total   = 0
    dropped_sparse    = 0
    dropped_no_roi    = 0

    for window_dt in sorted(groups):
        cands = groups[window_dt]

        # ROI gate: keep only occultations within ISR_ROI_MAX_KM of ESR/TRO
        lats = np.array([r["lat"] for r in cands])
        lons = np.array([r["lon"] for r in cands])
        roi_mask = _near_isr_site_mask(lats, lons)
        cands = [r for r, ok in zip(cands, roi_mask) if ok]

        if not cands:
            dropped_no_roi += 1
            continue
        if len(cands) < min_arcs_per_window:
            dropped_sparse += 1
            continue

        selected = _round_robin_by_constellation(cands, max_files)
        hhmm     = f"{window_dt.hour:02d}{window_dt.minute:02d}"
        wkey     = f"{window_dt.strftime('%Y-%m-%d')}_{hhmm}"

        # Constellation breakdown for reporting
        breakdown: dict[str, int] = defaultdict(int)
        for r in selected:
            breakdown[r["conid"]] += 1
        const_str = "  ".join(f"{c}:{n}" for c, n in sorted(breakdown.items()))
        print(f"    Window {wkey}: {len(selected)} arcs  [{const_str}]")

        windows.append(dict(
            window_key = wkey,
            hhmm       = hhmm,
            time_dt    = window_dt,
            records    = selected,
        ))
        kept_arcs_total += len(selected)

    print(f"  Retained {len(windows)} windows "
          f"({kept_arcs_total} total arcs); "
          f"dropped {dropped_no_roi} (no ROI arcs) + "
          f"{dropped_sparse} (< {min_arcs_per_window} arcs).")

    return windows


def _isr_site_grid_index(grid_lats: np.ndarray, grid_lons: np.ndarray,
                          site: str) -> int:
    """Index of the grid node nearest ISR *site* (great-circle distance)."""
    inst = INSTRUMENTS[site]
    d = _haversine_km(float(inst["lat"]), float(inst["lon"]),
                      np.asarray(grid_lats, dtype=float),
                      np.asarray(grid_lons, dtype=float))
    return int(np.argmin(d))


def run_occ_count_sweep(
    window: dict,
    model_state: "IonosphericState",
    grid_lats: np.ndarray,              # 5-deg model grid lats
    grid_lons: np.ndarray,              # 5-deg model grid lons
    alt_grid: np.ndarray,
    all_ro_arcs: list[dict],            # full RO arc_truth_list (forward-modelled)
    igs_arcs: list[dict],               # forward-modelled IGS arc_truth_list (may be empty)
    truth_ne_1deg: np.ndarray,          # (n_alt, n_truth) truth Ne on the 1-deg grid
    grid_lats_truth: np.ndarray,        # 1-deg truth grid lats
    grid_lons_truth: np.ndarray,        # 1-deg truth grid lons
    truth_time=None,
    n_occ_values: list[int] = SWEEP_N_OCC_VALUES,
    modes: tuple = FILTER_MODES,
    save_dir: str = SWEEP_SAVE_DIR,
) -> list[dict]:
    """
    Sweep the number of assimilated RO occultations and record frequency-domain
    retrieval accuracy at the ISR sites.

    For every N_OCC in *n_occ_values* the full RO arc list is sub-sampled to
    N_OCC arcs via round-robin constellation balancing, then EKF_Param is run in
    each observation *mode*:

        ro_only  : sub-sampled RO arcs only.
        ro_igs   : sub-sampled RO arcs + all simulated IGS arcs.
        igs_only : simulated IGS arcs only (N_OCC does not change the input, so
                   the EKF is run once and its metrics replicated per N_OCC row).

    The prior/posterior Ne columns are read out at the 5-deg model grid node
    nearest each ISR site; the truth column is read from *truth_ne_1deg* at the
    nearest 1-deg node.  compute_retrieval_freq_metrics turns the three columns
    into foF2/foE/plasma-frequency-RMSE metrics.

    Returns a list of per-(N_OCC, mode, site) row dicts for CSV accumulation.
    """
    window_key = window["window_key"]
    time_dt    = window["time_dt"]
    doy        = int(pd.Timestamp(time_dt).dayofyear)
    hour       = int(pd.Timestamp(time_dt).hour)

    # Nearest 1-deg truth column per ISR site (truth is N_OCC-invariant).
    truth_col_by_site: dict[str, np.ndarray] = {}
    for site in ISR_SITES:
        t_idx = _isr_site_grid_index(grid_lats_truth, grid_lons_truth, site)
        truth_col_by_site[site] = np.asarray(truth_ne_1deg[:, t_idx], dtype=float)

    # Nearest 5-deg model column index per ISR site (grid is N_OCC-invariant).
    model_idx_by_site: dict[str, int] = {
        site: _isr_site_grid_index(grid_lats, grid_lons, site)
        for site in ISR_SITES
    }

    ekf_kwargs = dict(
        sigma_obs           = EKF_PARAM_SIGMA_OBS,
        max_update_rays     = EKF_PARAM_UPDATE_RAYS,
        alpha               = EKF_PARAM_ALPHA,
        tol                 = EKF_PARAM_TOL,
        max_iter            = EKF_PARAM_MAX_ITER,
        apply_bounds        = EKF_PARAM_APPLY_BOUNDS,
        eps_jac             = EKF_PARAM_EPS_JAC,
        n_workers           = EKF_PARAM_N_WORKERS,
        jacobian_analytical = EKF_PARAM_JAC_ANALYTICAL,
    )

    def _rows_from_ekf(ekf: dict, mode: str, n_occ: int, n_arcs_used: int) -> list[dict]:
        prior_edp = ekf["prior_edp"]        # (n_alt, n_grid5)
        post_edp  = ekf["posterior_edp"]    # (n_alt, n_grid5)
        rows_out: list[dict] = []
        for site in ISR_SITES:
            g5       = model_idx_by_site[site]
            prior_ne = np.asarray(prior_edp[:, g5], dtype=float)
            post_ne  = np.asarray(post_edp[:, g5],  dtype=float)
            truth_ne = truth_col_by_site[site]
            fm = compute_retrieval_freq_metrics(truth_ne, prior_ne, post_ne, alt_grid)
            rows_out.append(dict(
                window_key    = window_key,
                time_dt       = pd.Timestamp(time_dt).isoformat(),
                truth_time    = (pd.Timestamp(truth_time).isoformat()
                                 if truth_time is not None else None),
                doy           = doy,
                hour          = hour,
                mode          = mode,
                n_occ         = n_occ,
                n_arcs_used   = n_arcs_used,
                site          = site,
                converged     = bool(ekf.get("converged", False)),
                n_iterations  = int(ekf.get("n_iterations", 0)),
                prior_rmse_tecu = float(ekf.get("prior_rmse", np.nan)),
                post_rmse_tecu  = float(ekf.get("post_rmse",  np.nan)),
                **fm,
            ))
        return rows_out

    rows: list[dict] = []

    # igs_only is N_OCC-invariant → run its EKF once and replicate rows.
    igs_only_ekf: dict | None = None
    if "igs_only" in modes and igs_arcs:
        print(f"\n  [sweep {window_key}] igs_only EKF ({len(igs_arcs)} IGS arcs) …")
        try:
            igs_only_ekf = EKF_Param(
                igs_arcs, model_state, grid_lats, grid_lons, alt_grid, **ekf_kwargs,
            )
        except Exception as exc:
            print(f"    [warn] igs_only EKF failed: {type(exc).__name__}: {exc}")
            igs_only_ekf = None

    for n_occ in n_occ_values:
        sub_ro = _round_robin_by_constellation(all_ro_arcs, n_occ)
        n_ro   = len(sub_ro)
        print(f"\n  [sweep {window_key}] N_OCC={n_occ}  "
              f"(RO arcs available={len(all_ro_arcs)}, used={n_ro})")

        for mode in modes:
            if mode == "ro_only":
                arcs = sub_ro
            elif mode == "ro_igs":
                arcs = sub_ro + list(igs_arcs)
            elif mode == "igs_only":
                if igs_only_ekf is not None:
                    rows.extend(_rows_from_ekf(
                        igs_only_ekf, "igs_only", n_occ, len(igs_arcs)))
                continue
            else:
                continue

            if not arcs:
                print(f"    [skip] mode={mode}: no arcs.")
                continue

            n_arcs_used = len(arcs)
            try:
                ekf = EKF_Param(arcs, model_state, grid_lats, grid_lons,
                                alt_grid, **ekf_kwargs)
            except Exception as exc:
                print(f"    [warn] mode={mode} N_OCC={n_occ} EKF failed: "
                      f"{type(exc).__name__}: {exc}")
                continue
            rows.extend(_rows_from_ekf(ekf, mode, n_occ, n_arcs_used))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# §2  Fibonacci sphere grid construction
# ─────────────────────────────────────────────────────────────────────────────

def _fibonacci_sphere_latlons(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate *n_points* approximately uniformly distributed lat/lon positions
    using the golden-ratio Fibonacci sphere method.

    Avoids polar over-sampling and longitude wrap-around artefacts that arise
    from rectangular lat/lon grids.
    """
    phi  = (1.0 + np.sqrt(5.0)) / 2.0
    i    = np.arange(n_points, dtype=float)
    lats = np.degrees(np.arcsin(1.0 - 2.0 * (i + 0.5) / n_points))
    lons = (np.degrees(2.0 * np.pi * i / phi) + 180.0) % 360.0 - 180.0
    return lats, lons


def _mst_edges(lats: np.ndarray, lons: np.ndarray) -> list[tuple[int, int]]:
    """
    Compute a Minimum Spanning Tree over a set of geographic points using
    Prim's algorithm with haversine distances.

    Returns
    -------
    edges : list of (i, j) index pairs forming the MST.
    """
    n = len(lats)
    if n <= 1:
        return []

    # Full pairwise haversine distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        dist[i] = _haversine_km(lats[i], lons[i], lats, lons)

    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    edges: list[tuple[int, int]] = []

    for _ in range(n - 1):
        in_idx  = np.where(in_tree)[0]
        out_idx = np.where(~in_tree)[0]
        d_sub   = dist[np.ix_(in_idx, out_idx)]
        flat    = int(np.argmin(d_sub))
        r_best, c_best = np.unravel_index(flat, d_sub.shape)
        i = int(in_idx[r_best])
        j = int(out_idx[c_best])
        in_tree[j] = True
        edges.append((i, j))

    return edges


def _great_circle_waypoints(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    step_deg: float,
) -> list[tuple[float, float]]:
    """
    Return intermediate (lat, lon) waypoints along the great circle from
    (lat1, lon1) to (lat2, lon2) at approximately *step_deg* intervals.

    Endpoints are excluded.  Uses SLERP interpolation in 3-D Cartesian space.
    """
    def _v(lat_d: float, lon_d: float) -> np.ndarray:
        lr, lo = np.radians(lat_d), np.radians(lon_d)
        return np.array([np.cos(lr) * np.cos(lo),
                         np.cos(lr) * np.sin(lo),
                         np.sin(lr)])

    v1, v2   = _v(lat1, lon1), _v(lat2, lon2)
    cos_th   = float(np.clip(v1 @ v2, -1.0, 1.0))
    theta_deg = np.degrees(np.arccos(cos_th))

    if theta_deg < step_deg:
        return []

    n_seg  = int(np.ceil(theta_deg / step_deg))
    sin_th = np.sin(np.radians(theta_deg))
    waypoints = []
    for k in range(1, n_seg):
        t = k / n_seg
        p = (np.sin((1 - t) * np.radians(theta_deg)) * v1
             + np.sin(t * np.radians(theta_deg)) * v2) / sin_th
        waypoints.append((
            float(np.degrees(np.arcsin(np.clip(p[2], -1.0, 1.0)))),
            float(np.degrees(np.arctan2(p[1], p[0]))),
        ))
    return waypoints


def _make_fibonacci_grid(
    tp_lats: np.ndarray,
    tp_lons: np.ndarray,
    spacing_deg: float,
    margin_deg: float  = GRID_MARGIN_DEG,
    min_points: int    = _IDW_NEAREST + 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a Fibonacci sphere sub-grid that covers the geometry of all
    occultation arc tangent points.

    Anchor set = tangent points  +  MST great-circle waypoints (at
    spacing_deg/2 intervals) to guarantee connectivity between arcs that
    may be geographically separated.

    Parameters
    ----------
    tp_lats, tp_lons : anchor positions (all decimated tangent-point locations
                       across all arcs and epochs).
    spacing_deg      : target inter-node spacing (degrees).
    margin_deg       : angular margin around each anchor.
    min_points       : minimum number of nodes to keep (expanded if needed).

    Returns
    -------
    lats, lons : (n_grid,) selected Fibonacci node positions.
    """
    tp_lats = np.asarray(tp_lats)
    tp_lons = np.asarray(tp_lons)

    # Total nodes on the full sphere for this spacing
    n_total = max(int(np.round(4.0 * np.pi / np.radians(spacing_deg) ** 2)), 12)
    fib_lats, fib_lons = _fibonacci_sphere_latlons(n_total)

    # Build anchor set: tangent points + MST SLERP waypoints
    anchor_lats = tp_lats.tolist()
    anchor_lons = tp_lons.tolist()
    step = spacing_deg / 2.0
    for i, j in _mst_edges(tp_lats, tp_lons):
        for wp_lat, wp_lon in _great_circle_waypoints(
                tp_lats[i], tp_lons[i], tp_lats[j], tp_lons[j], step):
            anchor_lats.append(wp_lat)
            anchor_lons.append(wp_lon)
    anchor_lats = np.array(anchor_lats)
    anchor_lons = np.array(anchor_lons)

    # Convert to unit 3-D Cartesian for chord-distance search
    def _xyz(lat_d: np.ndarray, lon_d: np.ndarray) -> np.ndarray:
        lr = np.radians(lat_d)
        ln = np.radians(lon_d)
        return np.column_stack([np.cos(lr) * np.cos(ln),
                                 np.cos(lr) * np.sin(ln),
                                 np.sin(lr)])

    fib_xyz    = _xyz(fib_lats, fib_lons)
    anchor_xyz = _xyz(anchor_lats, anchor_lons)
    tree       = cKDTree(fib_xyz)

    r: float          = float(margin_deg)
    indices: set[int] = set()
    while True:
        chord = 2.0 * np.sin(np.radians(r) / 2.0)
        for xyz in anchor_xyz:
            indices.update(tree.query_ball_point(xyz, r=chord))
        if len(indices) >= min_points or r >= 180.0:
            break
        r *= 1.5

    if r > margin_deg:
        print(f"    [info] Fibonacci grid margin expanded "
              f"{margin_deg:.1f}° → {r:.1f}° ({len(indices)} points)")

    idx_arr = np.array(sorted(indices))
    return fib_lats[idx_arr], fib_lons[idx_arr]


# ─────────────────────────────────────────────────────────────────────────────
# §3  IRI state grid construction
# ─────────────────────────────────────────────────────────────────────────────

def _exp_spatial_corr(
    lats: np.ndarray,
    lons: np.ndarray,
    corr_length_km: float = CORR_LENGTH_KM,
) -> np.ndarray:
    """
    Build an exponential spatial correlation matrix for *n* grid points.

    C[i,j] = exp(-d_km[i,j] / corr_length_km)

    Normalised to unit diagonal (correlation matrix, not covariance).
    A small nugget (1e-6 × I) is added for numerical conditioning.
    """
    n = len(lats)
    C = np.empty((n, n))
    for i in range(n):
        d    = _haversine_km(lats[i], lons[i], lats, lons)
        C[i] = np.exp(-d / corr_length_km)
    C = 0.5 * (C + C.T)
    C += 1e-6 * np.eye(n)
    std = np.sqrt(np.diag(C))
    C /= np.outer(std, std)
    return C


def build_iri_state_grid(
    time_dt: pd.Timestamp,
    lats: np.ndarray,
    lons: np.ndarray,
    alt_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Call IRI2020 for every grid point and assemble the 8-parameter state matrix.

    Parameters
    ----------
    time_dt  : observation time.
    lats, lons : (n_grid,) grid-point coordinates.
    alt_grid : (n_alt,) altitude levels (km).

    Returns
    -------
    mean_state  : (N_STATE, n_grid) IRI background mean in log/km/dim-less units.
    ne_profiles : (n_alt, n_grid)   IRI electron density profiles (m⁻³).
    """
    n_grid     = len(lats)
    sampling_df = _solar_sampling_df(time_dt)
    ne_profiles, feature_vecs = _get_iri_edp_and_features_batch(
        time_dt, lats, lons, alt_grid, sampling_df=sampling_df,
    )
    mean_state = np.empty((N_STATE, n_grid))
    for g in range(n_grid):
        mean_state[:, g] = _state_from_iri_direct(
            ne_profiles[:, g], feature_vecs[:, g], alt_grid,
        )
    return mean_state, ne_profiles


def _iri_cache_path(
    time_dt: pd.Timestamp,
    spacing_deg: float,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    cache_dir: str = IRI_CACHE_DIR,
) -> str:
    """Return the .npz cache file path for a given IRI grid run."""
    fname = (
        f"iri_{time_dt.year}_{time_dt.dayofyear:03d}_"
        f"{time_dt.hour:02d}{time_dt.minute:02d}_"
        f"{spacing_deg:.1f}deg_"
        f"lat{lat_min:.1f}_{lat_max:.1f}_"
        f"lon{lon_min:.1f}_{lon_max:.1f}.npz"
    )
    return os.path.join(cache_dir, fname)


def build_iri_state_grid_cached(
    time_dt: pd.Timestamp,
    lats: np.ndarray,
    lons: np.ndarray,
    alt_grid: np.ndarray,
    spacing_deg: float,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    cache_dir: str = IRI_CACHE_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper around build_iri_state_grid with .npz caching.

    Cache key includes time, spacing, and actual bounding box of the Fibonacci
    grid so a changed grid layout invalidates the cache automatically.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _iri_cache_path(
        time_dt, spacing_deg, lat_min, lat_max, lon_min, lon_max, cache_dir,
    )

    if os.path.isfile(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=False)
            # Validate alt_grid match
            if (data["alt_grid"].shape == alt_grid.shape
                    and np.allclose(data["alt_grid"], alt_grid, rtol=1e-5)):
                print(f"  [cache hit]  {os.path.basename(cache_path)}")
                return data["mean_state"], data["ne_profiles"]
            else:
                print(f"  [cache miss] alt_grid mismatch — rebuilding.")
        except Exception as exc:
            print(f"  [cache miss] Could not load {cache_path}: {exc}")

    print(f"  Building IRI grid  ({len(lats)} pts, {spacing_deg:.1f}°)…")
    mean_state, ne_profiles = build_iri_state_grid(time_dt, lats, lons, alt_grid)
    np.savez(
        cache_path,
        mean_state  = mean_state,
        ne_profiles = ne_profiles,
        lats        = lats,
        lons        = lons,
        alt_grid    = alt_grid,
    )
    print(f"  [cache write] {os.path.basename(cache_path)}")
    return mean_state, ne_profiles


def build_truth_state(
    mean_state: np.ndarray,
    truth_sigma: np.ndarray = _TRUTH_SIGMA,
) -> IonosphericState:
    """
    Build 9 deterministic ensemble members representing the "truth":

      member 0        : IRI baseline (no perturbation)
      member k+1      : baseline with parameter k shifted by +truth_sigma[k]

    Parameters
    ----------
    mean_state   : (N_STATE, n_grid)  IRI background mean.
    truth_sigma  : (N_STATE,)         per-parameter 1-σ offsets.

    Returns
    -------
    IonosphericState with ensemble clamped to physical bounds.
    """
    n_grid    = mean_state.shape[1]
    n_members = N_STATE + 1   # member 0 + one per parameter

    ensemble = np.broadcast_to(
        mean_state[:, :, np.newaxis],
        (N_STATE, n_grid, n_members),
    ).copy()

    for k in range(N_STATE):
        ensemble[k, :, k + 1] += truth_sigma[k]

    state = IonosphericState(n_grid_points=n_grid, n_members=n_members)
    state.ensemble = ensemble
    state.clamp_to_physical_bounds()
    return state


def build_model_ensemble(
    mean_state: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    n_members: int  = N_MEMBERS,
    corr_length_km: float = CORR_LENGTH_KM,
) -> IonosphericState:
    """
    Draw a stochastic ensemble from the IRI background distribution using the
    Kronecker-structured spatial covariance (param_cov ⊗ spatial_corr).

    Parameters
    ----------
    mean_state     : (N_STATE, n_grid)  IRI background mean.
    lats, lons     : (n_grid,) grid coordinates for the spatial correlation.
    n_members      : ensemble size.
    corr_length_km : exponential spatial correlation length.

    Returns
    -------
    IonosphericState with ensemble clamped to physical bounds.
    """
    n_grid      = mean_state.shape[1]
    param_cov   = _default_background_covariance()
    spatial_corr = _exp_spatial_corr(lats, lons, corr_length_km)

    state = IonosphericState(n_grid_points=n_grid, n_members=n_members)
    try:
        state.generate_ensemble_spatial(mean_state, param_cov, spatial_corr,
                                         n_members=n_members)
    except np.linalg.LinAlgError:
        warnings.warn("generate_ensemble_spatial failed; falling back to "
                      "independent (non-spatial) sampling.")
        state.generate_ensemble(mean_state, param_cov, n_members=n_members)

    # Jensen bias correction for log-space parameters.
    #
    # Drawing ε ~ N(0, σ²) and storing θ = μ + ε means:
    #   E[10^θ] = 10^μ · exp(½ σ² ln²10)  >  10^μ
    # The ensemble-mean Ne is systematically above the IRI baseline Ne, which
    # biases innovations negative and drives the filter in the wrong direction.
    # Subtracting σ²·ln(10)/2 from every log-space member shifts the ensemble
    # mean in log-space so that E[10^θ] = 10^μ exactly.
    # log_var = np.diag(param_cov)[LOG_INDICES]          # variance per log param
    # bias    = log_var * np.log(10) / 2                 # log₁₀ units
    # state.ensemble[LOG_INDICES] -= bias[:, np.newaxis, np.newaxis]

    # Pin member 0 to the unperturbed IRI baseline so callers can use
    # ensemble[:, :, 0] as a deterministic prior mean.
    state.ensemble[:, :, 0] = mean_state
    state.clamp_to_physical_bounds()
    return state


# ─────────────────────────────────────────────────────────────────────────────
# §4  Forward model
# ─────────────────────────────────────────────────────────────────────────────

def _arc_tangent_tracks(
    parsed_list: list[dict],
    max_epochs: int = MAX_EPOCHS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the full set of decimated tangent-point (lat, lon) locations across
    all arcs and epochs.

    Used to anchor the Fibonacci grids so that both ends of each arc (including
    high-altitude tangent points) are included in the coverage region.

    This function is lightweight — it calls _tangent_latlon_single per epoch
    but does NOT build ray objects.

    Parameters
    ----------
    parsed_list : list of arc dicts from parse_podTc2_nc_file.
    max_epochs  : maximum epochs to retain per arc (same decimation as
                  _build_arc_rays to keep the grid consistent).

    Returns
    -------
    all_lats, all_lons : flat arrays of tangent-point coordinates.
    """
    all_lats: list[float] = []
    all_lons: list[float] = []

    for arc in parsed_list:
        leo  = arc["LEO"]    # (3, n_epochs) ECEF km
        gnss = arc["GNSS"]   # (3, n_epochs) ECEF km
        tang_alts = np.asarray(arc["tangent_alt_km"])
        n_eps = leo.shape[1]

        # Decimate in the same way as _build_arc_rays
        if n_eps > max_epochs:
            stride  = int(np.ceil(n_eps / max_epochs))
            idx     = np.arange(0, n_eps, stride)
            deepest = int(np.argmin(tang_alts))
            if deepest not in idx:
                idx = np.sort(np.append(idx, deepest))
        else:
            idx = np.arange(n_eps)

        for i in idx:
            lat, lon = _tangent_latlon_single(gnss[:, i], leo[:, i])
            all_lats.append(float(lat))
            all_lons.append(float(lon))

    return np.array(all_lats), np.array(all_lons)


def _idw_weights(
    tp_lat: float,
    tp_lon: float,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
) -> np.ndarray:
    """
    Smooth Gaussian interpolation weights from (tp_lat, tp_lon) to grid nodes.

    Sigma is set adaptively to _IDW_SIGMA_SCALE × the distance to the 3rd-
    nearest node, which scales correctly to both the 1-deg and 5-deg Fibonacci
    grids without a fixed bandwidth.  Active nodes are capped at _IDW_NEAREST
    by proximity so cost in _integrate_ray_idw stays bounded.

    The Gaussian kernel replaces the former hard-cutoff IDW (power=2) scheme.
    With power-law weights, swapping the k-th and (k+1)-th nearest nodes caused
    a discrete jump of ~0.5 % per transition in the blended Ne and hence in the
    integrated TEC — enough to produce visible oscillations across the arc.
    The Gaussian decays much faster: by the time the k-th node is reached its
    weight is negligible relative to the nearest nodes, so any swap at the
    cap boundary has a sub-0.01 % effect on the blended TEC.
    """
    d_km = _haversine_km(tp_lat, tp_lon, grid_lats, grid_lons)
    d_km = np.maximum(d_km, 0.01)
    # Adaptive sigma from the 3rd-nearest distance (index 2 after sort).
    sigma_km = float(np.sort(d_km)[min(2, len(d_km) - 1)]) * _IDW_SIGMA_SCALE
    w = np.exp(-(d_km / sigma_km) ** 2)
    # Cap to _IDW_NEAREST nodes by proximity (Gaussian is monotone in d,
    # so the cap picks the same nodes that would dominate the weight sum).
    if len(w) > _IDW_NEAREST:
        cap_idx = np.argpartition(d_km, _IDW_NEAREST)[:_IDW_NEAREST]
        mask = np.zeros(len(w), dtype=bool)
        mask[cap_idx] = True
        w = np.where(mask, w, 0.0)
    if w.sum() == 0.0:
        w[np.argmin(d_km)] = 1.0
    w /= w.sum()
    return w


def _build_arc_rays(
    arc: dict,
    max_epochs: int = MAX_EPOCHS,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the ray trajectory objects and per-epoch metadata for one arc.

    Parameters
    ----------
    arc        : parsed arc dict from parse_podTc2_nc_file.
    max_epochs : maximum epochs to retain (decimated if more).

    Returns
    -------
    rays      : list of (n_iono, 3) ray arrays [lat, lon, alt_km].
    tp_lats   : (n_ep,) tangent-point latitudes.
    tp_lons   : (n_ep,) tangent-point longitudes.
    tang_alts : (n_ep,) tangent-point altitudes (km).
    tec_meas  : (n_ep,) measured podTc2 sTEC (TECU) — raw, not band-limited.
    """
    leo       = arc["LEO"]    # (3, n_epochs) ECEF km
    gnss      = arc["GNSS"]   # (3, n_epochs) ECEF km
    tang_alts_full = np.asarray(arc["tangent_alt_km"])
    tec_full  = np.asarray(arc.get("TEC_podTc2", np.full(leo.shape[1], np.nan)))
    n_eps     = leo.shape[1]

    # Decimation
    if n_eps > max_epochs:
        stride  = int(np.ceil(n_eps / max_epochs))
        idx     = np.arange(0, n_eps, stride)
        deepest = int(np.argmin(tang_alts_full))
        if deepest not in idx:
            idx = np.sort(np.append(idx, deepest))
    else:
        idx = np.arange(n_eps)

    rays: list     = []
    tp_lats: list  = []
    tp_lons: list  = []
    tang_list: list = []
    tec_list: list  = []

    for i in idx:
        ray = _build_gnss_to_leo_ray(gnss[:, i], leo[:, i])
        rays.append(ray)
        lat, lon = _tangent_latlon_single(gnss[:, i], leo[:, i])
        tp_lats.append(float(lat))
        tp_lons.append(float(lon))
        tang_list.append(float(tang_alts_full[i]))
        tec_list.append(float(tec_full[i]))

    return (rays,
            np.array(tp_lats),
            np.array(tp_lons),
            np.array(tang_list),
            np.array(tec_list))


def forward_model_arc(
    rays: list,
    tp_lats: np.ndarray,
    tp_lons: np.ndarray,
    obs_op: ObservationOperator,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    use_idw: bool = True,
) -> np.ndarray:
    """
    Compute the forward-modelled sTEC ensemble for every epoch of one arc.

    Parameters
    ----------
    rays         : list of ray arrays (length n_ep).
    tp_lats/lons : tangent-point coordinates (n_ep,).
    obs_op       : ObservationOperator bound to the appropriate state.
    grid_lats/lons : (n_grid,) grid coordinates.
    use_idw      : if True use IDW weights; if False use nearest-neighbour.

    Returns
    -------
    Y_f : (n_ep, n_members) simulated sTEC ensemble (TECU).
    """
    n_ep = len(rays)

    if use_idw:
        # Build weight matrix (n_ep, n_grid) — one IDW row per tangent point
        W = np.stack([
            _idw_weights(tp_lats[k], tp_lons[k], grid_lats, grid_lons)
            for k in range(n_ep)
        ])  # (n_ep, n_grid)
        Y_f = obs_op.compute_stec_ensemble(rays, grid_point_weights=W)
    else:
        tree = cKDTree(np.column_stack([grid_lats, grid_lons]))
        _, gpi = tree.query(np.column_stack([tp_lats, tp_lons]))
        gpi = gpi.ravel().astype(int)
        Y_f = obs_op.compute_stec_ensemble(rays, grid_point_indices=gpi)

    return Y_f


def run_forward_models(
    parsed_list: list[dict],
    truth_state: IonosphericState,
    model_state: IonosphericState,
    grid_lats_1deg: np.ndarray,
    grid_lons_1deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """
    Run the parametric forward model for every arc on both the truth (1-deg)
    and model (5-deg) grids.

    Returns
    -------
    truth_arcs : list of dicts per arc with keys:
        tangent_km, tec_all (n_ep, 9), tec_measured (n_ep,),
        tp_lats, tp_lons, conid, prn_id, leo_id
    model_arcs : list of dicts per arc with keys:
        tangent_km, tec_mean (n_ep,), tec_std (n_ep,),
        tp_lats, tp_lons, conid, prn_id, leo_id
    """
    obs_truth = ObservationOperator(truth_state, alt_grid)
    obs_model = ObservationOperator(model_state, alt_grid)

    truth_arcs: list[dict] = []
    model_arcs: list[dict] = []

    for arc_idx, arc in enumerate(parsed_list):
        prn_id = str(arc.get("prn_id", arc.get("prn", "?")))
        leo_id = str(arc.get("leo_id", "?"))
        conid  = str(arc.get("conid", prn_id[0].upper()
                              if prn_id[0].upper() in "GREC" else "?")).upper()

        print(f"  Arc {arc_idx+1}/{len(parsed_list)}: "
              f"{leo_id} {conid}{prn_id} …", end=" ", flush=True)

        rays, tp_lats, tp_lons, tang_km, tec_meas = _build_arc_rays(arc)

        # Truth forward model on 1-deg grid (IDW)
        Y_truth = forward_model_arc(
            rays, tp_lats, tp_lons, obs_truth,
            grid_lats_1deg, grid_lons_1deg, use_idw=True,
        )  # (n_ep, 9)

        # Model forward model on 5-deg grid (IDW)
        Y_model = forward_model_arc(
            rays, tp_lats, tp_lons, obs_model,
            grid_lats_5deg, grid_lons_5deg, use_idw=True,
        )  # (n_ep, N_MEMBERS)

        print(f"done  ({len(rays)} epochs)")

        truth_arcs.append(dict(
            tangent_km   = tang_km,
            tec_all      = Y_truth,           # (n_ep, 9)
            tec_measured = tec_meas,          # (n_ep,)  measured podTc2
            tp_lats      = tp_lats,
            tp_lons      = tp_lons,
            conid        = conid,
            prn_id       = prn_id,
            leo_id       = leo_id,
        ))
        model_arcs.append(dict(
            tangent_km = tang_km,
            tec_mean   = Y_model.mean(axis=1),
            tec_std    = Y_model.std(axis=1),
            tp_lats    = tp_lats,
            tp_lons    = tp_lons,
            conid      = conid,
            prn_id     = prn_id,
            leo_id     = leo_id,
        ))

    return truth_arcs, model_arcs


# ─────────────────────────────────────────────────────────────────────────────
# §4b  Simulated IGS ground-station geometry + forward model
# ─────────────────────────────────────────────────────────────────────────────
#
# Ground-based ionospheric assimilation uses fixed receivers on the Earth's
# surface with line-of-sight rays to GNSS satellites overhead.  Here we simulate
# that geometry with:
#
#   • Real station coordinates loaded from Data/IGS_Stations/IGSNetwork.json
#     (matched by 4-char prefix, filtered to those inside ISR_ROI_MAX_KM of the
#     ISR sites).
#   • Real GNSS satellite positions propagated from the day's broadcast
#     navigation file (BRDC RINEX in Data/RINEX_Cache) via TEC_model's
#     BroadcastEphemeris — one arc per (station, SV) pair, with rays taken at
#     several epochs across the RO window so each arc sweeps out the SV's
#     actual sky track for that observation window.
#   • The same truth IonosphericState + ObservationOperator + IDW weighting as
#     RO arcs, so the resulting arc dicts drop straight into arc_truth_list.
#
# _build_gnss_to_leo_ray filters ray samples to the ionospheric band exactly
# as it does for RO — the ground-to-SV LOS is ~20 000+ km long but only the
# ~1000 km inside the ionosphere contributes to sTEC.

_WGS84_A_KM = 6378.137
_WGS84_E2   = 6.694379990141317e-3


def _geodetic_to_ecef_km(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    """(lat_deg, lon_deg, h_m) → ECEF (x, y, z) in km."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N   = _WGS84_A_KM / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
    h_km = h_m * 1e-3
    x = (N + h_km) * np.cos(lat) * np.cos(lon)
    y = (N + h_km) * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - _WGS84_E2) + h_km) * np.sin(lat)
    return np.array([x, y, z], dtype=float)


def _load_igs_sim_stations(
    json_path: str,
    codes: list[str],
    roi_max_km: float = ISR_ROI_MAX_KM,
) -> list[dict]:
    """
    Load requested IGS station coordinates from IGSNetwork.json and gate them
    against the ISR ROI.

    Matches by case-insensitive 4-char prefix (e.g. "TRO1" → "TRO100NOR").  For
    each matched station, prefers the JSON's X/Y/Z (metres) if present,
    otherwise derives ECEF from Latitude/Longitude/Height via WGS84.

    Returns
    -------
    List of dicts (preserving the input code order), keys:
        code, lat, lon, height_m, ecef_km.
    """
    try:
        with open(json_path, "r") as fh:
            network = json.load(fh)
    except FileNotFoundError:
        print(f"  [IGS-sim] Station JSON not found: {json_path}  → no stations.")
        return []

    codes_upper = [c.upper() for c in codes]
    resolved: dict[str, dict] = {}
    for entry_key, entry in network.items():
        prefix = entry_key[:4].upper()
        if prefix not in codes_upper or prefix in resolved:
            continue
        try:
            lat = float(entry["Latitude"])
            lon = float(entry["Longitude"])
            h_m = float(entry["Height"])
        except (KeyError, TypeError, ValueError):
            continue
        # ROI gate — station itself must be within ISR ROI
        if not _near_isr_site_mask(
            np.array([lat]), np.array([lon]), max_km=roi_max_km,
        )[0]:
            continue
        try:
            ecef_km = np.array([
                float(entry["X"]) * 1e-3,
                float(entry["Y"]) * 1e-3,
                float(entry["Z"]) * 1e-3,
            ], dtype=float)
        except (KeyError, TypeError, ValueError):
            ecef_km = _geodetic_to_ecef_km(lat, lon, h_m)
        resolved[prefix] = dict(
            code=prefix, lat=lat, lon=lon, height_m=h_m, ecef_km=ecef_km,
        )

    ordered = [resolved[c] for c in codes_upper if c in resolved]
    dropped = [c for c in codes_upper if c not in resolved]
    if dropped:
        print(f"  [IGS-sim] Skipped stations (not in JSON or outside ROI): {dropped}")
    if ordered:
        print(f"  [IGS-sim] Resolved {len(ordered)} station(s): "
              f"{[s['code'] for s in ordered]}")
    return ordered


def _elev_from_station(
    sv_ecef_km: np.ndarray,
    rx_ecef_km: np.ndarray,
    lat_deg: float,
    lon_deg: float,
) -> float:
    """
    Elevation angle (degrees) of an SV seen from the station at
    (lat_deg, lon_deg).  Uses the local ENU basis expressed in ECEF —
    elevation = 90° means the SV is overhead, 0° means at the horizon.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    E = np.array([-np.sin(lon),                 np.cos(lon),                0.0        ])
    N = np.array([-np.sin(lat) * np.cos(lon),  -np.sin(lat) * np.sin(lon),  np.cos(lat)])
    U = np.array([ np.cos(lat) * np.cos(lon),   np.cos(lat) * np.sin(lon),  np.sin(lat)])
    d = sv_ecef_km - rx_ecef_km
    e_v = float(np.dot(d, E))
    n_v = float(np.dot(d, N))
    u_v = float(np.dot(d, U))
    horiz = np.hypot(e_v, n_v)
    return float(np.degrees(np.arctan2(u_v, horiz)))


def _resolve_brdc_path(
    time_dt: pd.Timestamp,
    cache_dir: str = IGS_SIM_BRDC_DIR,
) -> Path | None:
    """
    Return the path of the multi-constellation BRDC navigation file that covers
    time_dt's UTC day, or None if none is cached.

    Recognises both BRDC00IGS_R_{YYYY}{DOY}0000_01D_MN.rnx and the .gz variant,
    and falls back to BRDM00IGS if BRDC is missing.  The path is passed as-is
    to BroadcastEphemeris; georinex transparently decompresses .gz.
    """
    yyyy = time_dt.year
    doy  = int(time_dt.strftime("%j"))
    cache = Path(cache_dir)
    for prefix in ("BRDC00IGS", "BRDM00IGS"):
        for ext in (".rnx", ".rnx.gz"):
            p = cache / f"{prefix}_R_{yyyy}{doy:03d}0000_01D_MN{ext}"
            if p.exists():
                return p
    return None


def _load_broadcast_ephemeris(time_dt: pd.Timestamp):
    """
    Load the BroadcastEphemeris object for *time_dt*'s UTC day.  Returns None
    (with a printed message) if the RINEX nav file isn't cached — the caller
    should fall back to RO-only in that case.
    """
    from TEC_model.igs_tec_pipeline import BroadcastEphemeris
    nav_path = _resolve_brdc_path(time_dt)
    if nav_path is None:
        print(f"  [IGS-sim] No BRDC nav file cached for "
              f"{time_dt.strftime('%Y-DOY%j')} under {IGS_SIM_BRDC_DIR}  "
              f"→ IGS geometry unavailable.")
        return None
    print(f"  [IGS-sim] Broadcast ephemeris: {nav_path.name}")
    try:
        return BroadcastEphemeris(nav_path)
    except Exception as exc:
        print(f"  [IGS-sim] BroadcastEphemeris load failed ({exc!r}) "
              f"— IGS geometry unavailable.")
        return None


def _ipp_latlon_km(
    sv_ecef_km: np.ndarray,
    rx_ecef_km: np.ndarray,
    h_ipp_km: float = IGS_SIM_IPP_ALT_KM,
) -> tuple[float, float]:
    """
    Ionospheric pierce point (lat, lon) at altitude h_ipp_km along one LOS.
    Used as the horizontal anchor (tp_lat, tp_lon) for IDW weighting.
    """
    t_vals = np.linspace(0.0, 1.0, 200)
    pts    = sv_ecef_km[:, None] + t_vals * (rx_ecef_km[:, None] - sv_ecef_km[:, None])
    lons, lats, alts_m = _TRANSFORMER.transform(
        pts[0] * 1e3, pts[1] * 1e3, pts[2] * 1e3,
    )
    alts_km = alts_m / 1e3
    idx     = int(np.argmin(np.abs(alts_km - h_ipp_km)))
    return float(lats[idx]), float(lons[idx])


def _build_igs_sim_arcs(
    stations: list[dict],
    time_dt: pd.Timestamp,
    ephem,
    window_minutes: float = WINDOW_MINUTES,
    n_epochs: int         = IGS_SIM_N_EPOCHS,
    elev_cutoff_deg: float = IGS_SIM_ELEV_CUTOFF_DEG,
    constellations: tuple[str, ...] = IGS_SIM_CONSTELLATIONS,
    n_ray_pts: int  = IGS_SIM_N_RAY_PTS,
) -> list[dict]:
    """
    Build geometry-only arcs for the simulated IGS ground-station observations
    using the day's real broadcast ephemeris.

    For each (station, SV) pair the SV's ECEF position is propagated from the
    BRDC nav file at *n_epochs* sample times spread across the window
    (time_dt ± window_minutes/2).  Rays with SV elevation below
    *elev_cutoff_deg* are dropped; a (station, SV) pair with fewer than two
    surviving rays is skipped entirely.  Ray endpoints (rx, sv) run through
    _build_gnss_to_leo_ray for the same ionospheric-band filtering used by
    RO arcs.

    Parameters
    ----------
    ephem : TEC_model.igs_tec_pipeline.BroadcastEphemeris
        Loaded ephemeris object — see _load_broadcast_ephemeris.
    time_dt : pd.Timestamp
        Window centre (used to compute GPS SOW / UTC SOD for each epoch).
    window_minutes : float
        Full duration of the sampling window in minutes.  Rays are taken at
        *n_epochs* evenly spaced instants across [-window/2, +window/2].

    Returns
    -------
    List of dicts (no tec_truth yet — filled in by generate_simulated_igs_tec):
        rays     : list of (n_iono, 3) ray arrays
        tp_lats  : (n_rays,) IPP latitudes (deg)
        tp_lons  : (n_rays,) IPP longitudes (deg)
        tang_km  : (n_rays,) elevation (deg) — kept in the "tang_km" slot as
                    the arc's per-ray ordering axis, replacing the RO tangent
                    altitude that has no analogue for ground-based rays.
        conid    : IGS_SIM_CONID
        prn_id   : "{station}_{SV}"  (e.g. "TRO1_G05")
        leo_id   : station code
    """
    from TEC_model.igs_tec_pipeline import _utc_to_gps_sow

    if ephem is None or not stations:
        return []

    # Sample epochs across the window
    half_sec = 0.5 * window_minutes * 60.0
    offsets  = np.linspace(-half_sec, +half_sec, max(int(n_epochs), 2))
    epoch_times = [time_dt + pd.Timedelta(seconds=float(o)) for o in offsets]

    # Restrict to the constellations we want; skip unknown SVs
    all_svs = [sv for sv in ephem._cache.keys() if sv[:1] in constellations]
    all_svs.sort()

    arcs: list[dict] = []
    n_pairs_tried  = 0
    n_pairs_kept   = 0
    for sta in stations:
        rx_km = sta["ecef_km"]
        lat0  = sta["lat"]
        lon0  = sta["lon"]
        for sv in all_svs:
            n_pairs_tried += 1
            rays: list = []
            tp_lats: list[float] = []
            tp_lons: list[float] = []
            elevs: list[float]   = []
            for t_ep in epoch_times:
                # GLONASS ephemeris uses UTC seconds-of-day; others use GPS SOW
                if sv.startswith("R"):
                    t_arg = float(t_ep.hour * 3600 + t_ep.minute * 60 + t_ep.second)
                else:
                    t_arg = _utc_to_gps_sow(t_ep)
                sv_ecef = ephem.sv_position_km(sv, t_arg)
                if sv_ecef is None:
                    continue
                el = _elev_from_station(sv_ecef, rx_km, lat0, lon0)
                if el < elev_cutoff_deg:
                    continue
                ray = _build_gnss_to_leo_ray(sv_ecef, rx_km, n_pts=n_ray_pts)
                ipp_lat, ipp_lon = _ipp_latlon_km(sv_ecef, rx_km)
                rays.append(ray)
                tp_lats.append(ipp_lat)
                tp_lons.append(ipp_lon)
                elevs.append(el)
            if len(rays) < 2:
                continue
            n_pairs_kept += 1
            arcs.append(dict(
                rays    = rays,
                tp_lats = np.array(tp_lats),
                tp_lons = np.array(tp_lons),
                tang_km = np.array(elevs),
                conid   = IGS_SIM_CONID,
                prn_id  = f"{sta['code']}_{sv}",
                leo_id  = sta["code"],
            ))
    print(f"  [IGS-sim] {n_pairs_kept}/{n_pairs_tried} (station, SV) pairs "
          f"visible above {elev_cutoff_deg:.0f}° across "
          f"{window_minutes:.0f}-min window.")
    return arcs


def generate_simulated_igs_tec(
    sim_arcs: list[dict],
    truth_state: IonosphericState,
    grid_lats_truth: np.ndarray,
    grid_lons_truth: np.ndarray,
    alt_grid: np.ndarray,
) -> list[dict]:
    """
    Forward-model simulated ground-to-GNSS rays through the truth state to
    produce synthetic sTEC observations, mirroring generate_truth_tec.

    Parameters
    ----------
    sim_arcs             : geometry arcs from _build_igs_sim_arcs.
    truth_state          : single-member IonosphericState on the 1-deg truth grid.
    grid_lats/lons_truth : (n_truth,) truth grid coordinates.
    alt_grid             : altitude integration grid (km).

    Returns
    -------
    Arc dicts with the same field set as generate_truth_tec:
        rays, tec_truth, tp_lats, tp_lons, tang_km, conid, prn_id, leo_id.
    """
    obs_truth = ObservationOperator(truth_state, alt_grid)
    out_arcs: list[dict] = []

    for arc_idx, sim in enumerate(sim_arcs):
        rays    = sim["rays"]
        tp_lats = sim["tp_lats"]
        tp_lons = sim["tp_lons"]
        n_ep    = len(rays)

        W = np.stack([
            _idw_weights(tp_lats[k], tp_lons[k],
                         grid_lats_truth, grid_lons_truth)
            for k in range(n_ep)
        ])  # (n_ep, n_truth)

        Y_truth   = obs_truth.compute_stec_ensemble(rays, grid_point_weights=W)
        tec_truth = Y_truth[:, 0]

        print(f"  IGS-sim arc {arc_idx+1}/{len(sim_arcs)}: "
              f"{sim['leo_id']} {sim['prn_id']}  "
              f"{n_ep} rays  "
              f"TEC {tec_truth.min():.1f}–{tec_truth.max():.1f} TECU")

        out_arcs.append(dict(
            rays      = rays,
            tec_truth = tec_truth,
            tp_lats   = tp_lats,
            tp_lons   = tp_lons,
            tang_km   = sim["tang_km"],
            conid     = sim["conid"],
            prn_id    = sim["prn_id"],
            leo_id    = sim["leo_id"],
        ))

    return out_arcs


# ─────────────────────────────────────────────────────────────────────────────
# §5  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_conid(arc: dict) -> str:
    """Return G/R/E/C for an arc.
    IGS-sim arcs store conid='I'; the real GNSS letter is the first character
    of the SV portion of prn_id (e.g. 'TRO1_G05' → 'G').
    """
    conid = str(arc.get("conid", "?")).upper()
    if conid not in _CONST_POS:
        sv = str(arc.get("prn_id", "")).split("_")[-1]
        c = sv[:1].upper()
        conid = c if c in _CONST_POS else "G"
    return conid


def _arc_label(arc: dict) -> str:
    """Short legend label for an arc.
    For IGS-sim arcs (conid not in G/R/E/C) the prn_id already encodes
    station+SV (e.g. 'TRO1_G05'), so use it directly.
    For RO arcs use the conventional 'LEO CPRN' format.
    """
    raw_conid = str(arc.get("conid", "?")).upper()
    prn_id = str(arc.get("prn_id", "?"))
    if raw_conid not in _CONST_POS:
        return prn_id
    return f"{arc.get('leo_id', '?')} {raw_conid}{prn_id}"


def _capped_legend(
    ax: plt.Axes,
    max_items: int = 8,
    handles: list | None = None,
    labels: list[str] | None = None,
    **kwargs,
) -> None:
    """Call ax.legend() but truncate to max_items entries to prevent overflow."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    elif labels is None:
        labels = [h.get_label() for h in handles]
    handles = list(handles)
    labels  = list(labels)
    if not handles:
        return
    if len(handles) > max_items:
        n_extra = len(handles) - (max_items - 1)
        handles = handles[:max_items - 1] + [Line2D([0], [0], color="none")]
        labels  = labels[:max_items - 1]  + [f"… +{n_extra} more"]
    ax.legend(handles=handles, labels=labels, **kwargs)


def _occ_colors(parsed_list: list[dict]) -> list:
    """
    Assign a unique colour to each occultation, deepening shade within each
    GNSS constellation using that constellation's colourmap.
    """
    # Count per constellation (using resolved G/R/E/C regardless of raw conid)
    const_counts: dict[str, int] = defaultdict(int)
    for arc in parsed_list:
        const_counts[_resolve_conid(arc)] += 1

    const_idx: dict[str, int] = defaultdict(int)
    colors = []
    for arc in parsed_list:
        conid  = _resolve_conid(arc)
        cfg    = CONSTELLATION_CONFIG.get(conid, {})
        cmap   = mpl.colormaps.get_cmap(cfg.get("cmap", _CONST_FALLBACK_CMAP))
        n      = max(const_counts[conid], 1)
        # Shade range 0.4–0.85 to avoid very light / very dark extremes
        shade  = 0.4 + 0.45 * const_idx[conid] / n
        colors.append(cmap(shade))
        const_idx[conid] += 1

    return colors


def plot_results(
    truth_arcs: list[dict],
    model_arcs: list[dict],
    grid_lats_1deg: np.ndarray,
    grid_lons_1deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    time_dt: pd.Timestamp,
    save_path: str,
) -> None:
    """
    Produce the main diagnostic figure:

    Layout — GridSpec(2, 3, width_ratios=[1, 1, 1.6])
    ──────────────────────────────────────────────────
    [0,0] GPS TEC     [0,1] Galileo TEC   [0:2, 2] Globe map
    [1,0] GLONASS TEC [1,1] BeiDou TEC

    TEC panels
    ----------
    • Black solid       : measured podTc2 sTEC
    • Thick coloured solid : 1-deg IRI baseline (truth member 0)
    • 8 thin tab10 lines   : truth members 1–8 (one per parameter perturbation)
    • Coloured dashed      : 5-deg model ensemble mean
    • Coloured shading     : 5-deg model ±1σ

    Shared parameter legend below the 4 TEC panels (tab10 palette, PARAM_NAMES).

    Globe panel
    -----------
    • Small grey dots    : 1-deg Fibonacci grid nodes
    • Steelblue squares  : 5-deg Fibonacci grid nodes
    • Coloured dots      : arc track tangent points (colour = constellation)
    • Large circle       : TEC-max tangent point per arc
    """
    occ_colors = _occ_colors(
        [{"conid": tr["conid"], "prn_id": tr["prn_id"]} for tr in truth_arcs]
    )
    tab10 = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(18, 9), facecolor="#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    gs  = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.6],
                   hspace=0.35, wspace=0.28,
                   left=0.06, right=0.97, top=0.92, bottom=0.14)

    # Panel axes for 4 constellations
    tec_axes: dict[str, plt.Axes] = {}
    for const, (row, col) in _CONST_POS.items():
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#2b2b2b")
        cfg = CONSTELLATION_CONFIG.get(const, {"name": const,
                                                "title_color": "white"})
        ax.set_title(cfg["name"], color=cfg.get("title_color", "white"),
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("sTEC (TECU)",      color="lightgray", fontsize=7)
        ax.set_ylabel("Tang. alt. (km)",  color="lightgray", fontsize=7)
        ax.tick_params(colors="lightgray", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        tec_axes[const] = ax

    # Orthographic globe centerd on the mean tangent-point location of all arcs
    all_tp_lats = np.concatenate([tr["tp_lats"] for tr in truth_arcs])
    all_tp_lons = np.concatenate([tr["tp_lons"] for tr in truth_arcs])
    cen_lat = float(np.mean(all_tp_lats))
    cen_lon = float(np.mean(all_tp_lons))

    ax_globe = fig.add_subplot(gs[:, 2],
                                projection=ccrs.Orthographic(
                                    central_longitude=cen_lon,
                                    central_latitude=cen_lat,
                                ))
    ax_globe.set_facecolor("#2b2b2b")
    ax_globe.add_feature(cfeature.COASTLINE, linewidth=0.4,
                          edgecolor="#aaaaaa")
    ax_globe.add_feature(cfeature.BORDERS,   linewidth=0.2,
                          edgecolor="#888888")
    ax_globe.gridlines(draw_labels=False, linewidth=0.3,
                       color="gray", alpha=0.5)

    # Plot 1-deg grid nodes
    ax_globe.scatter(
        grid_lons_1deg, grid_lats_1deg,
        s=1, color="lightgray", alpha=0.3, transform=ccrs.PlateCarree(),
        zorder=2, label="1° grid",
    )
    # Plot 5-deg grid nodes
    ax_globe.scatter(
        grid_lons_5deg, grid_lats_5deg,
        s=8, color="steelblue", alpha=0.6, marker="s",
        transform=ccrs.PlateCarree(), zorder=3, label="5° grid",
    )

    globe_legend_handles: list = [
        Line2D([0], [0], color="lightgray", marker="o", ms=3,
               linestyle="none", label="1° grid"),
        Line2D([0], [0], color="steelblue", marker="s", ms=4,
               linestyle="none", label="5° grid"),
    ]

    for arc_i, (tr, mo, col) in enumerate(
            zip(truth_arcs, model_arcs, occ_colors)):
        const  = _resolve_conid(tr)
        prn_id = str(tr["prn_id"])
        leo_id = str(tr["leo_id"])
        tang   = tr["tangent_km"]
        label  = _arc_label(tr)

        ax = tec_axes.get(const, tec_axes["G"])

        # Measured TEC (black solid)  — sTEC on x, altitude on y
        tec_meas = tr["tec_measured"]
        if np.any(np.isfinite(tec_meas)):
            ax.plot(tec_meas, tang, color="black", linewidth=1.4,
                    zorder=6, label="Measured" if arc_i == 0 else None)

        # Truth member 0: IRI baseline (thick coloured solid)
        ax.plot(tr["tec_all"][:, 0], tang, color=col,
                linewidth=2.0, zorder=5, label=label)

        # Truth members 1–8: parameter perturbations (thin tab10 lines)
        for k in range(1, N_STATE + 1):
            ax.plot(tr["tec_all"][:, k], tang,
                    color=tab10(k - 1), linewidth=0.7,
                    alpha=0.8, zorder=4)

        # Model ensemble mean ± 1σ (coloured dashed + shading)
        ax.plot(mo["tec_mean"], tang, color=col, linewidth=1.2,
                linestyle="--", zorder=5)
        ax.fill_betweenx(
            tang,
            mo["tec_mean"] - mo["tec_std"],
            mo["tec_mean"] + mo["tec_std"],
            color=col, alpha=0.18, zorder=3,
        )

        # Globe: arc track
        ax_globe.scatter(
            tr["tp_lons"], tr["tp_lats"],
            s=4, color=col, alpha=0.6, transform=ccrs.PlateCarree(), zorder=4,
        )
        # Globe: TEC-max marker
        tec_max_idx = int(np.argmin(tang))
        ax_globe.scatter(
            tr["tp_lons"][tec_max_idx], tr["tp_lats"][tec_max_idx],
            s=60, color=col, edgecolors="white", linewidth=0.6,
            marker="o", transform=ccrs.PlateCarree(), zorder=5,
        )
        globe_legend_handles.append(
            Line2D([0], [0], color=col, marker="o", ms=5,
                   linestyle="none", label=label)
        )

    # Add legend on TEC axes
    for const, ax in tec_axes.items():
        if ax.lines:
            _capped_legend(ax, fontsize=5, facecolor="#2b2b2b",
                           labelcolor="lightgray", loc="best",
                           framealpha=0.7)

    # Shared parameter legend below TEC panels
    param_handles = [
        Line2D([0], [0], color=tab10(k), linewidth=1.5,
               label=PARAM_NAMES[k])
        for k in range(N_STATE)
    ]
    param_handles.insert(0, Line2D([0], [0], color="black", linewidth=1.5,
                                    label="Measured"))
    fig.legend(
        handles=param_handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.38, 0.01),
        fontsize=7,
        facecolor="#2b2b2b",
        labelcolor="lightgray",
        framealpha=0.85,
        title="Truth perturbation",
        title_fontsize=7,
    )

    # Globe legend
    _capped_legend(
        ax_globe,
        handles=globe_legend_handles,
        fontsize=6, facecolor="#2b2b2b",
        labelcolor="lightgray", loc="lower left",
        framealpha=0.7, markerscale=1.3,
    )

    title_str = (f"Parametric IonosphericState forward model — "
                 f"{time_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    fig.suptitle(title_str, color="white", fontsize=11, y=0.97)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §5.1  Ensemble parameter histograms
# ─────────────────────────────────────────────────────────────────────────────

def plot_ensemble_histograms(
    model_state: IonosphericState,
    mean_state: np.ndarray,
    time_dt: pd.Timestamp,
    save_path: str,
    n_bins: int = 40,
) -> None:
    """
    8×1 figure showing the ensemble distribution of each IRI parameter across
    all grid points and ensemble members.

    For each parameter:
      • Histogram of ensemble[k].ravel()  (n_grid × n_members samples)
      • Red dashed : IRI mean (mean_state[k].mean())
      • Gray dotted : ±1σ sample std lines

    Parameters
    ----------
    model_state : IonosphericState containing the drawn ensemble.
    mean_state  : (N_STATE, n_grid) IRI background mean.
    time_dt     : observation time (used in title).
    save_path   : output file path.
    n_bins      : histogram bin count.
    """
    ens = model_state.ensemble   # (N_STATE, n_grid, n_members)

    fig, axes = plt.subplots(N_STATE, 1, figsize=(7, 14),
                              facecolor="#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    fig.suptitle(
        f"Ensemble parameter distributions — "
        f"{time_dt.strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"({model_state.n_members} members × "
        f"{model_state.n_grid_points} grid points)",
        color="white", fontsize=10, y=0.99,
    )

    for k, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        samples = ens[k].ravel()
        mu      = float(mean_state[k].mean())
        sigma   = float(samples.std())

        ax.set_facecolor("#2b2b2b")
        ax.hist(samples, bins=n_bins, color="steelblue",
                alpha=0.75, density=True)
        ax.axvline(mu, color="red",  linestyle="--", linewidth=1.4,
                   label=f"IRI mean = {mu:.3g}")
        ax.axvline(mu - sigma, color="lightgray", linestyle=":",
                   linewidth=1.0)
        ax.axvline(mu + sigma, color="lightgray", linestyle=":",
                   linewidth=1.0, label=f"σ = {sigma:.3g}")
        ax.set_ylabel(name, color="lightgray", fontsize=7, rotation=0,
                      labelpad=70, va="center")
        ax.tick_params(colors="lightgray", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        ax.legend(fontsize=6, facecolor="#2b2b2b", labelcolor="lightgray",
                  loc="upper right", framealpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §5.5  Parameter sensitivity sweep
# ─────────────────────────────────────────────────────────────────────────────

def _sweep_one_param(
    arc_rays: list,
    mean_state_1gp: np.ndarray,
    param_idx: int,
    alt_grid: np.ndarray,
    n_sweep: int   = _N_SWEEP,
    n_sigma: float = _N_SIGMA,
) -> dict:
    """
    Sweep parameter *param_idx* over ±n_sigma·σ around its IRI baseline value
    and compute the forward-modelled sTEC for each sweep member.

    The state has a single grid point so no IDW interpolation is needed.

    Parameters
    ----------
    arc_rays       : list of ray arrays for one arc.
    mean_state_1gp : (N_STATE,)  IRI baseline at the representative grid point.
    param_idx      : index into PARAM_NAMES / state vector to vary.
    alt_grid       : altitude integration grid.
    n_sweep        : number of sweep values (odd → symmetric about baseline).
    n_sigma        : sweep half-range in units of _TRUTH_SIGMA[param_idx].

    Returns
    -------
    dict with keys:
        param_values   : (n_sweep,) actual swept values after clamping.
        tec_matrix     : (n_ep, n_sweep) forward-modelled sTEC.
        baseline_idx   : index of the nearest sweep member to the IRI baseline.
        baseline_val   : IRI baseline value for this parameter.
        truth_idx      : index of member at baseline + _TRUTH_SIGMA[param_idx].
        truth_val      : baseline + _TRUTH_SIGMA[param_idx].
    """
    baseline_val = float(mean_state_1gp[param_idx])
    sigma        = float(_TRUTH_SIGMA[param_idx])
    truth_val    = baseline_val + sigma

    param_values = np.linspace(
        baseline_val - n_sigma * sigma,
        baseline_val + n_sigma * sigma,
        n_sweep,
    )

    # Build single-gp ensemble: shape (N_STATE, 1, n_sweep)
    ens = np.tile(mean_state_1gp[:, np.newaxis, np.newaxis],
                  (1, 1, n_sweep)).copy()
    ens[param_idx, 0, :] = param_values

    state = IonosphericState(n_grid_points=1, n_members=n_sweep)
    state.ensemble = ens
    state.clamp_to_physical_bounds()
    param_values = state.ensemble[param_idx, 0, :].copy()

    obs_op    = ObservationOperator(state, alt_grid)
    gp_indices = np.zeros(len(arc_rays), dtype=int)
    Y_f = obs_op.compute_stec_ensemble(
        arc_rays, grid_point_indices=gp_indices,
    )  # (n_ep, n_sweep)

    baseline_idx = int(np.argmin(np.abs(param_values - baseline_val)))
    truth_idx    = int(np.argmin(np.abs(param_values - truth_val)))

    return dict(
        param_values  = param_values,
        tec_matrix    = Y_f,
        baseline_idx  = baseline_idx,
        baseline_val  = baseline_val,
        truth_idx     = truth_idx,
        truth_val     = truth_val,
    )


def plot_parameter_influence(
    arc_rays: list,
    arc_tang_km: np.ndarray,
    mean_state_1gp: np.ndarray,
    alt_grid: np.ndarray,
    time_dt: pd.Timestamp,
    arc_label: str,
    save_path: str,
    n_sweep: int    = _N_SWEEP,
    n_sigma: float  = _N_SIGMA,
    abs_error: bool = False,
    log_err_scale: bool = False,
    tec_truth_perturbed: np.ndarray | None = None,
) -> None:
    """
    8×2 parameter sensitivity figure.

    For each of the 8 IRI parameters:
      Left  : pcolormesh(param_values, tang_km, sTEC) — colourmap "cividis"
               white dashed line = IRI baseline
               cyan solid line   = +1σ perturbation marker
      Right : pcolormesh(param_values, tang_km, sTEC − truth_col)
               signed → "coolwarm"  (abs_error=False)
               absolute → "magma"   (abs_error=True)

    Parameters
    ----------
    arc_rays       : ray list from _build_arc_rays for one arc.
    arc_tang_km    : (n_ep,) tangent altitudes (km).
    mean_state_1gp : (N_STATE,)  IRI state at the representative 1-deg grid point.
    alt_grid       : altitude integration grid.
    time_dt        : observation time (for plot title).
    arc_label      : string label, e.g. "GN05 G15".
    save_path      : output file path.
    n_sweep, n_sigma, abs_error : sweep settings.
    log_err_scale  : if True, use logarithmic colour normalization on the error
                     panels.  For abs_error=True uses LogNorm; for abs_error=False
                     uses SymLogNorm (handles both signs).
    tec_truth_perturbed : (n_ep, N_STATE) TEC from the 1×1-deg IDW forward model
                     with each parameter individually perturbed by +1σ.
                     Column k is the truth reference for parameter panel k —
                     i.e. truth_arcs[arc_idx]["tec_all"][:, 1:].  Must be in the
                     same epoch order as arc_tang_km.
    """
    if log_err_scale:
        save_path += "_log"
    # Sort epochs by ascending tangent altitude for pcolormesh
    sort_idx  = np.argsort(arc_tang_km)
    tang_plot = arc_tang_km[sort_idx]
    rays_sort = [arc_rays[i] for i in sort_idx]

    # Sort the perturbed truth array into ascending-altitude order.
    # Result shape: (n_ep, N_STATE) — column k is the truth for parameter panel k.
    tec_truth_sorted = (
        tec_truth_perturbed[sort_idx, :]
        if tec_truth_perturbed is not None else None
    )

    fig, axes = plt.subplots(
        N_STATE, 2, figsize=(12, 22),
        facecolor="#1e1e1e",
    )
    fig.patch.set_facecolor("#1e1e1e")

    truth_src  = "1°-grid perturbed TEC" if tec_truth_perturbed is not None else "+1σ perturbation"
    title_type = "Absolute error" if abs_error else "Signed error"
    scale_tag  = "  [log scale]" if log_err_scale else ""
    fig.suptitle(
        f"Parameter sensitivity — {arc_label} — "
        f"{time_dt.strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"(left: sTEC TECU;  right: {title_type} vs {truth_src}{scale_tag})",
        color="white", fontsize=10, y=0.995,
    )

    # ── Pre-pass: compute all sweeps and errors so we can set a shared scale ──
    sweep_results = []
    for k in range(N_STATE):
        result = _sweep_one_param(
            rays_sort, mean_state_1gp, k, alt_grid, n_sweep, n_sigma,
        )
        tec   = result["tec_matrix"]
        t_idx = result["truth_idx"]
        if tec_truth_sorted is not None:
            truth_col = tec_truth_sorted[:, k:k+1]
        else:
            truth_col = tec[:, t_idx:t_idx+1]
        err = np.abs(tec - truth_col) if abs_error else (tec - truth_col)
        sweep_results.append((result, err))

    # Global colour limits across all 8 error panels
    all_errors   = np.concatenate([e.ravel() for _, e in sweep_results])
    global_vmax  = float(np.nanpercentile(np.abs(all_errors), 98)) or 1.0

    if abs_error:
        cmap_r = "magma"
        if log_err_scale:
            pos_vals        = all_errors[all_errors > 0]
            global_vmin_log = float(np.nanpercentile(pos_vals, 2)) if len(pos_vals) else 1e-4
            global_vmin_log = max(global_vmin_log, 1e-4)
            global_norm_r   = LogNorm(vmin=global_vmin_log, vmax=global_vmax)
        else:
            global_norm_r = mpl.colors.Normalize(vmin=0, vmax=global_vmax)
    else:
        cmap_r = "coolwarm"
        if log_err_scale:
            linthresh     = max(global_vmax * 0.01, 1e-4)
            global_norm_r = SymLogNorm(linthresh=linthresh,
                                        vmin=-global_vmax, vmax=global_vmax)
        else:
            global_norm_r = mpl.colors.Normalize(vmin=-global_vmax, vmax=global_vmax)

    # ── Plotting loop ─────────────────────────────────────────────────────────
    for k, (result, err) in enumerate(sweep_results):
        pv    = result["param_values"]
        tec   = result["tec_matrix"]
        b_val = result["baseline_val"]
        t_val = result["truth_val"]

        ax_left  = axes[k, 0]
        ax_right = axes[k, 1]

        for ax in (ax_left, ax_right):
            ax.set_facecolor("#2b2b2b")
            ax.tick_params(colors="lightgray", labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor("#555")

        # Bin edges for pcolormesh
        pv_edges   = np.concatenate([[2*pv[0]-pv[1]],
                                      0.5*(pv[:-1]+pv[1:]),
                                      [2*pv[-1]-pv[-2]]])
        tang_edges = np.concatenate([[2*tang_plot[0]-tang_plot[1]],
                                      0.5*(tang_plot[:-1]+tang_plot[1:]),
                                      [2*tang_plot[-1]-tang_plot[-2]]])

        # ── Left panel: raw sTEC ─────────────────────────────────────────────
        pcm_l  = ax_left.pcolormesh(pv_edges, tang_edges, tec,
                                     cmap="cividis", shading="flat")
        cbar_l = fig.colorbar(pcm_l, ax=ax_left, fraction=0.04, pad=0.02)
        cbar_l.set_label("sTEC (TECU)", color="lightgray")
        cbar_l.ax.yaxis.label.set_color("lightgray")
        cbar_l.ax.tick_params(colors="lightgray")

        ax_left.axvline(b_val, color="white", linestyle="--", linewidth=1.0)
        ax_left.axvline(t_val, color="cyan",  linestyle="-",  linewidth=1.0)
        ax_left.set_ylabel("Tang. alt. (km)", color="lightgray", fontsize=7)
        ax_left.set_title(PARAM_NAMES[k], color="lightgray", fontsize=8)

        # ── Right panel: error (shared norm, individual colorbar) ────────────
        pcm_r  = ax_right.pcolormesh(pv_edges, tang_edges, err,
                                      cmap=cmap_r, norm=global_norm_r, shading="flat")
        cbar_r = fig.colorbar(pcm_r, ax=ax_right, fraction=0.04, pad=0.02)
        cbar_r.set_label(title_type + " (TECU)", color="lightgray")
        cbar_r.ax.yaxis.label.set_color("lightgray")
        cbar_r.ax.tick_params(colors="lightgray")

        ax_right.axvline(b_val, color="white", linestyle="--", linewidth=1.0)
        ax_right.axvline(t_val, color="cyan",  linestyle="-",  linewidth=1.0)
        ax_right.set_title(PARAM_NAMES[k], color="lightgray", fontsize=8)

        if k == N_STATE - 1:
            ax_left.set_xlabel("Parameter value", color="lightgray", fontsize=7)
            ax_right.set_xlabel("Parameter value", color="lightgray", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §5b  Per-parameter error summary (companion to group-summary)
# ─────────────────────────────────────────────────────────────────────────────

def _ne_to_params_easy(
    ne_profiles: np.ndarray,
    alt_grid: np.ndarray,
) -> np.ndarray:
    """
    Extract the 8-parameter state from Ne(h) profiles without IRI feature
    scalars.  Used to evaluate the gridded-Ne KF posterior in parameter space.

    Parameters 0,1,6,7 (NmF2, hmF2, NmE, hmE) are read from profile peaks.
    Parameters 2,3 (H0, gamma) are fitted from the topside.
    Parameters 4,5 (B0, B1) are set to NaN — they cannot be reliably
    recovered from a Ne profile alone.

    Returns (N_STATE, n_geo) in log10/km/dimensionless units.
    """
    from demo_compare_kf_enkf import _fit_topside_H0_gamma, _h0_seed_from_profile

    n_alt, n_geo = ne_profiles.shape
    out = np.full((N_STATE, n_geo), np.nan)

    f2_mask = alt_grid > 100.0
    e_lo, e_hi = 80.0, 160.0
    e_mask = (alt_grid >= e_lo) & (alt_grid <= e_hi)

    for g in range(n_geo):
        ne = ne_profiles[:, g]

        # NmF2 / hmF2 — peak above 100 km
        if f2_mask.any():
            f2_ne  = ne[f2_mask]
            f2_alt = alt_grid[f2_mask]
            i_peak = int(np.argmax(f2_ne))
            nm_f2  = float(f2_ne[i_peak])
            hm_f2  = float(f2_alt[i_peak])
        else:
            nm_f2, hm_f2 = 1e11, 300.0

        # NmE / hmE — peak in E-layer window
        if e_mask.any():
            e_ne  = ne[e_mask]
            e_alt = alt_grid[e_mask]
            i_e   = int(np.argmax(e_ne))
            nm_e  = float(e_ne[i_e])
            hm_e  = float(e_alt[i_e])
        else:
            nm_e, hm_e = 1e9, 110.0

        out[I_LOG_NMF2, g] = np.log10(max(nm_f2, 1.0))
        out[I_HMF2,     g] = hm_f2
        out[I_LOG_NME,  g] = np.log10(max(nm_e, 1.0))
        out[I_HME,      g] = hm_e

        # H0, gamma — topside L-BFGS-B fit (same as _state_from_iri_direct)
        top_mask = alt_grid > hm_f2
        if top_mask.sum() >= 5:
            try:
                H0_seed = _h0_seed_from_profile(
                    ne[top_mask], alt_grid[top_mask], nm_f2, hm_f2,
                )
                H0, gamma = _fit_topside_H0_gamma(
                    ne[top_mask], alt_grid[top_mask], nm_f2, hm_f2, H0_seed,
                )
                out[I_H0,    g] = float(H0)
                out[I_GAMMA, g] = float(gamma)
            except Exception:
                pass

    return out


def plot_param_error_summary(
    filter_results_by_mode: dict[str, dict],
    save_path: str,
) -> None:
    """
    Per-parameter absolute error figure — companion to the group summary plot.

    For each of the 8 IRI state parameters, shows the mean absolute error
    (averaged over all 5-deg grid points) for:
      • Prior (IRI baseline vs truth)            — grey
      • Gridded Ne KF posterior vs truth         — steelblue
        (NmF2, hmF2, NmE, hmE, H0, gamma fitted from posterior Ne;
         B0, B1 not plotted — KF does not update parameters directly)
      • EKF_Param posterior vs truth             — seagreen
        (all 8 parameters available directly)

    Grouped by the 3 observation modes on the x-axis of each sub-panel.
    Layout: 4 rows × 2 columns, one panel per parameter.
    """
    modes        = [m for m in FILTER_MODES if filter_results_by_mode.get(m)]
    mode_labels  = {"ro_only": "RO only", "ro_igs": "RO+IGS", "igs_only": "IGS only"}
    x_labels     = [mode_labels.get(m, m) for m in modes]
    n_modes      = len(modes)
    if n_modes == 0:
        return

    # ── collect per-(mode, filter) mean absolute parameter errors ──────────────
    # shape: (n_modes,) per category
    prior_err  = np.full((N_STATE, n_modes), np.nan)
    kf_err     = np.full((N_STATE, n_modes), np.nan)
    ekf_err    = np.full((N_STATE, n_modes), np.nan)

    for mi, mode in enumerate(modes):
        fr = filter_results_by_mode[mode]
        truth = fr.get("truth_mean_5deg")
        if truth is None:
            continue
        truth = np.asarray(truth)           # (N_STATE, n_geo)

        kf  = fr.get("kf_result")
        ekf = fr.get("ekf_param")

        # Prior — prefer EKF's explicit parameter record; fall back to
        # extracting from the KF prior Ne profiles if EKF did not run.
        prior_state = None
        if ekf is not None:
            prior_state = ekf.get("prior_mean_state")
        if prior_state is None and kf is not None:
            prior_ne = kf.get("prior_ne_5deg")
            if prior_ne is not None:
                prior_state = _ne_to_params_easy(np.asarray(prior_ne), ALT_GRID)
        if prior_state is not None:
            prior_state = np.asarray(prior_state)
            prior_err[:, mi] = np.nanmean(np.abs(prior_state - truth), axis=1)

        # Gridded Ne KF posterior — fit parameters from posterior Ne profiles
        if kf is not None:
            post_ne = kf.get("posterior_ne_5deg")
            if post_ne is not None:
                kf_params = _ne_to_params_easy(np.asarray(post_ne), ALT_GRID)
                kf_err[:, mi] = np.nanmean(np.abs(kf_params - truth), axis=1)

        # EKF_Param posterior — parameters available directly
        if ekf is not None:
            post_p = ekf.get("posterior_mean_5deg")
            if post_p is not None:
                post_p = np.asarray(post_p)
                ekf_err[:, mi] = np.nanmean(np.abs(post_p - truth), axis=1)

    # ── figure ─────────────────────────────────────────────────────────────────
    param_units = [
        "log₁₀(m⁻³)", "km", "km", "–", "km", "–", "log₁₀(m⁻³)", "km",
    ]
    n_cols, n_rows = 2, 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(10, 10),
        facecolor="#1e1e1e",
    )
    fig.patch.set_facecolor("#1e1e1e")

    x   = np.arange(n_modes, dtype=float)
    bw  = 0.24
    off = np.array([-bw, 0.0, bw])   # prior, KF, EKF

    for param_i in range(N_STATE):
        row = param_i // n_cols
        col = param_i %  n_cols
        ax  = axes[row, col]
        ax.set_facecolor("#2b2b2b")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        ax.tick_params(colors="lightgray", labelsize=7)

        pv = prior_err[param_i]
        kv = kf_err[param_i]
        ev = ekf_err[param_i]

        # B0 (index 4) and B1 (index 5) — KF cannot recover these
        kf_na = param_i in (I_B0, I_B1)

        bars_prior = ax.bar(x + off[0], pv, bw, color="#888888",
                            alpha=0.85, label="Prior")
        bars_kf    = ax.bar(x + off[1], kv if not kf_na else np.zeros(n_modes),
                            bw, color="steelblue", alpha=0.85,
                            label="KF post" if not kf_na else "_")
        bars_ekf   = ax.bar(x + off[2], ev, bw, color="seagreen",
                            alpha=0.85, label="EKF post")

        if kf_na:
            # shade KF bars to indicate N/A
            for b in bars_kf:
                b.set_hatch("///")
                b.set_edgecolor("#aaaaaa")
                b.set_alpha(0.35)

        ax.set_title(
            f"{PARAM_NAMES[param_i]}  ({param_units[param_i]})",
            color="white", fontsize=8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7, color="lightgray")
        ax.set_ylabel("Mean |error|", color="lightgray", fontsize=7)
        ax.grid(axis="y", lw=0.3, alpha=0.4, color="gray")

        if param_i == 0:
            legend = ax.legend(
                fontsize=7, facecolor="#2b2b2b", labelcolor="lightgray",
                loc="upper right", framealpha=0.8,
            )

    fig.suptitle(
        "Per-parameter absolute error vs truth  "
        "(mean over 5° grid)   B0/B1 hatched = not recovered by Ne-space KF",
        color="white", fontsize=9, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §5c  Group-summary metrics plot (simulated-truth adapter)
# ─────────────────────────────────────────────────────────────────────────────
#
# plotIonosphereTomography._plot_group_summary_metrics was built for the real
# ISR-DA comparison pipeline: it expects a nested filter_results[obs_mode]
# [filter_type] dict, an eds_occ-like object with a .geolocation attribute
# per config, and an ISR-style "window_edps" truth list.
#
# The wrapper below repackages test_param_iono's per-mode retrieval output
# into that schema.  The only functional difference from the real-ISR flow is
# where the "truth" comes from: instead of ISR EDPs pulled from
# demo_esr_isr.load_edps(), we sample the *synthetic* truth ionosphere
# (truth_ne_1deg) at the nearest 1-deg grid vertex to each ISR site.  The
# resulting profile is tagged kindat="simulated" so the plot's legend labels
# it "Simulated truth EDP" (see _ISR_KINDAT_STYLE).

def plot_group_summary_metrics_simulated(
    filter_results_by_mode: dict[str, dict],
    save_dir: str,
    hhmm: str,
) -> str | None:
    """
    Adapter to plotIonosphereTomography._plot_group_summary_metrics for the
    test_param_iono synthetic-truth pipeline.

    Parameters
    ----------
    filter_results_by_mode
        `{mode: filter_result}` — the `filter_results` dict returned by
        `_process_time_window`, where each value has `kf_result`, `ekf_param`,
        `truth_ne_1deg`, and the mode-specific `grid_lats_1deg` / `grid_lons_1deg`
        / `grid_lats_5deg` / `grid_lons_5deg`.
    save_dir
        Directory to save the summary figure into.
    hhmm
        The window's HHMM tag; used in the figure filename.

    Returns
    -------
    str | None
        Path to the saved figure, or None if nothing plotable was produced.
    """
    from types import SimpleNamespace
    from plotIonosphereTomography import _plot_group_summary_metrics

    # Reference mode for building the ISR-site truth spaghetti — every mode
    # samples the same underlying truth ionosphere, but only the RO-anchored
    # grids are guaranteed to have a vertex near the ISR sites, so we prefer
    # ro_igs → ro_only → igs_only.
    ref_mode = next(
        (m for m in ("ro_igs", "ro_only", "igs_only")
         if filter_results_by_mode.get(m) is not None),
        None,
    )
    if ref_mode is None:
        print("  [skip] group summary metrics plot: no filter results")
        return None
    ref = filter_results_by_mode[ref_mode]
    ref_grid_lats_1 = np.asarray(ref["grid_lats_1deg"])
    ref_grid_lons_1 = np.asarray(ref["grid_lons_1deg"])
    ref_truth_1deg  = np.asarray(ref["truth_ne_1deg"])

    # ── Build window_edps: one simulated "truth" profile per ISR site ────────
    window_edps: list[dict] = []
    for site in ISR_SITES:
        inst = INSTRUMENTS[site]
        lat  = float(inst["lat"])
        lon  = float(inst["lon"])
        d    = _haversine_km(lat, lon, ref_grid_lats_1, ref_grid_lons_1)
        idx1 = int(np.argmin(d))
        window_edps.append(dict(
            alt_km = np.asarray(ALT_GRID, dtype=float),
            ne_m3  = np.asarray(ref_truth_1deg[:, idx1], dtype=float),
            lat    = lat,
            lon    = lon,
            kindat = "simulated",
        ))

    # ── Build filter_results[obs_mode][filter_type] from per-mode dicts ──────
    filter_results_nested: dict[str, dict[str, dict]] = {}
    for mode in FILTER_MODES:
        fr = filter_results_by_mode.get(mode)
        mode_dict: dict[str, dict] = {}
        if fr is not None:
            glats5 = np.asarray(fr["grid_lats_5deg"])
            glons5 = np.asarray(fr["grid_lons_5deg"])
            # _group_edp_rmse_vs_isr expects geolocation as (n_geo, 2) with
            # columns [lon, lat]; the mesh_pts array is then built as
            # np.column_stack([geoloc[:,1], geoloc[:,0]]) = (lat, lon).
            geoloc = np.column_stack([glons5, glats5])
            eds_occ = SimpleNamespace(geolocation=geoloc)

            kf = fr.get("kf_result")
            if kf is not None:
                mode_dict["gridded_kf"] = dict(
                    prior_edp_3d   = np.asarray(kf["prior_edp"]),
                    post_edp_3d    = np.asarray(kf["posterior_edp"]),
                    alt_grid       = np.asarray(ALT_GRID),
                    eds_occ        = eds_occ,
                    prior_tec_rmse = float(kf["prior_rmse"]),
                    post_tec_rmse  = float(kf["post_rmse"]),
                )
            ekf = fr.get("ekf_param")
            if ekf is not None:
                entry: dict = dict(
                    prior_edp_3d   = np.asarray(ekf["prior_edp"]),
                    post_edp_3d    = np.asarray(ekf["posterior_edp"]),
                    alt_grid       = np.asarray(ALT_GRID),
                    eds_occ        = eds_occ,
                    prior_tec_rmse = float(ekf["prior_rmse"]),
                    post_tec_rmse  = float(ekf["post_rmse"]),
                )
                pstate = ekf.get("posterior_mean_5deg")
                if pstate is not None:
                    entry["posterior_mean_state"] = np.asarray(pstate)
                mode_dict["parametric_ekf"] = entry
        filter_results_nested[mode] = mode_dict

    group_key = f"{YYYY}_{DOY:03d}_{hhmm}"
    summary_path = _plot_group_summary_metrics(
        group_key      = group_key,
        filter_results = filter_results_nested,
        window_edps    = window_edps,
        save_dir       = Path(save_dir),
    )

    # ── Parameter-error companion figure ──────────────────────────────────────
    plot_param_error_summary(
        filter_results_by_mode = filter_results_by_mode,
        save_path = os.path.join(
            save_dir, f"group_param_error_{group_key}.png"
        ),
    )

    return summary_path


# ─────────────────────────────────────────────────────────────────────────────
# §6  main()
# ─────────────────────────────────────────────────────────────────────────────

def _process_time_window(
    window: dict,
    save_dir: str = SAVE_DIR,
) -> dict:
    """
    Run Steps 2–14 of the pipeline for a single time window.

    Grid geometry (Fibonacci sub-grid over the arc tangent tracks) depends on
    the window's specific arc set, so grids are rebuilt per window even though
    the *procedure* itself is time-invariant.  Every downstream state (IRI
    truth/baseline, ensemble draws, forward models, KF/EKF assimilation) is
    time-dependent and rebuilt from scratch here.

    All figures produced for this window are namespaced with the window's
    "_HHMM" suffix so successive windows do not overwrite each other.
    """
    window_key = window["window_key"]
    hhmm       = window["hhmm"]
    time_dt    = window["time_dt"]
    records    = window["records"]
    save_suffix = f"_{hhmm}"

    print(f"  Parsing {len(records)} files …")
    parsed_list: list[dict] = []
    for rec in records:
        try:
            data = parse_podTc2_nc_file(rec["path"])
            data["conid"]  = rec["conid"]
            data["prn_id"] = rec["prn_id"]
            data["leo_id"] = rec["leo_id"]
            parsed_list.append(data)
        except Exception as exc:
            warnings.warn(f"Could not parse {rec['path']}: {exc}")

    if not parsed_list:
        print("  All files failed to parse.  Skipping window.")
        return {"window_key": window_key, "hhmm": hhmm, "time_dt": time_dt,
                "error": "no arcs parsed"}

    print(f"  Window time_dt (bin start): {time_dt}")

    # ── Step 2: build Fibonacci grids ────────────────────────────────────────
    print("\nStep 2: Building Fibonacci grids …")
    tp_lats_all, tp_lons_all = _arc_tangent_tracks(parsed_list)
    print(f"  Arc tangent-point anchors: {len(tp_lats_all)}")

    grid_lats_1deg, grid_lons_1deg = _make_fibonacci_grid(
        tp_lats_all, tp_lons_all, spacing_deg=1.0,
    )
    print(f"  1-deg grid: {len(grid_lats_1deg)} nodes")

    grid_lats_5deg, grid_lons_5deg = _make_fibonacci_grid(
        tp_lats_all, tp_lons_all, spacing_deg=5.0,
    )
    print(f"  5-deg grid: {len(grid_lats_5deg)} nodes")

    # ── Step 2a: 1-deg IRI grid → truth state ────────────────────────────────
    print("\nStep 2a: Building 1-deg IRI state (truth) …")
    lat_min_1 = float(grid_lats_1deg.min())
    lat_max_1 = float(grid_lats_1deg.max())
    lon_min_1 = float(grid_lons_1deg.min())
    lon_max_1 = float(grid_lons_1deg.max())

    mean_1deg, _ = build_iri_state_grid_cached(
        time_dt, grid_lats_1deg, grid_lons_1deg, ALT_GRID,
        spacing_deg=1.0,
        lat_min=lat_min_1, lat_max=lat_max_1,
        lon_min=lon_min_1, lon_max=lon_max_1,
    )
    truth_state = build_truth_state(mean_1deg)
    print(f"  Truth ensemble: {truth_state.n_members} members "
          f"× {truth_state.n_grid_points} grid points")

    # ── Step 2b: 5-deg IRI grid → model ensemble ─────────────────────────────
    print("\nStep 2b: Building 5-deg IRI state (model ensemble) …")
    lat_min_5 = float(grid_lats_5deg.min())
    lat_max_5 = float(grid_lats_5deg.max())
    lon_min_5 = float(grid_lons_5deg.min())
    lon_max_5 = float(grid_lons_5deg.max())

    mean_5deg, _ = build_iri_state_grid_cached(
        time_dt, grid_lats_5deg, grid_lons_5deg, ALT_GRID,
        spacing_deg=5.0,
        lat_min=lat_min_5, lat_max=lat_max_5,
        lon_min=lon_min_5, lon_max=lon_max_5,
    )
    model_state = build_model_ensemble(
        mean_5deg, grid_lats_5deg, grid_lons_5deg,
        n_members=N_MEMBERS, corr_length_km=CORR_LENGTH_KM,
    )
    print(f"  Model ensemble: {model_state.n_members} members "
          f"× {model_state.n_grid_points} grid points")

    ens_var = model_state.ensemble.var(axis=2)
    print("  Per-parameter ensemble std (mean over grid):")
    for k, name in enumerate(PARAM_NAMES):
        print(f"    {name:20s}  σ = {float(np.sqrt(ens_var[k].mean())):.4g}")

    plot_ensemble_histograms(
        model_state, mean_5deg, time_dt,
        save_path=os.path.join(
            save_dir, f"ensemble_histograms_{YYYY}_{DOY:03d}{save_suffix}.png"
        ),
    )

    # ── Steps 3–4: forward models ────────────────────────────────────────────
    print("\nStep 3: Running forward models …")
    truth_arcs, model_arcs = run_forward_models(
        parsed_list,
        truth_state, model_state,
        grid_lats_1deg, grid_lons_1deg,
        grid_lats_5deg, grid_lons_5deg,
        ALT_GRID,
    )

    print("\nArc summary (IRI baseline vs 5-deg model mean):")
    for tr, mo in zip(truth_arcs, model_arcs):
        diff = tr["tec_all"][:, 0] - mo["tec_mean"]
        print(f"  {tr['leo_id']} {tr['conid']}{tr['prn_id']:>3s}  "
              f"mean_diff={diff.mean():+.3f} TECU  "
              f"rmse={float(np.sqrt((diff**2).mean())):.3f} TECU")

    # ── Step 5: main results plot ────────────────────────────────────────────
    print("\nStep 5: Plotting results …")
    save_results = os.path.join(
        save_dir, f"param_iono_test_{YYYY}_{DOY:03d}{save_suffix}.png"
    )
    plot_results(
        truth_arcs, model_arcs,
        grid_lats_1deg, grid_lons_1deg,
        grid_lats_5deg, grid_lons_5deg,
        time_dt, save_results,
    )

    # ── Step 6: parameter sensitivity sweep ──────────────────────────────────
    print("\nStep 6: Parameter sensitivity sweep …")
    tree_1deg = cKDTree(np.column_stack([grid_lats_1deg, grid_lons_1deg]))
    ref_lat = records[0]["lat"]
    ref_lon = records[0]["lon"]
    _, _gp_idx = tree_1deg.query([[ref_lat, ref_lon]])
    gp_idx = int(_gp_idx[0])
    mean_1gp = mean_1deg[:, gp_idx]

    rays_0, tp_lats_0, tp_lons_0, tang_km_0, _ = _build_arc_rays(parsed_list[0])
    tec_truth_arc0 = truth_arcs[0]["tec_all"][:, 1:]

    arc0 = parsed_list[0]
    arc_label_0 = (f"{str(arc0.get('leo_id', '?'))} "
                   f"{str(arc0.get('conid', '?'))}"
                   f"{str(arc0.get('prn_id', '?'))}")

    save_sens = os.path.join(
        save_dir, f"param_sensitivity_{YYYY}_{DOY:03d}{save_suffix}"
    )
    save_abs = os.path.join(
        save_dir, f"param_sensitivity_abs_{YYYY}_{DOY:03d}{save_suffix}"
    )

    plot_parameter_influence(
        rays_0, tang_km_0, mean_1gp, ALT_GRID,
        time_dt, arc_label_0, save_sens,
        n_sweep=_N_SWEEP, n_sigma=_N_SIGMA, abs_error=False, log_err_scale=True,
        tec_truth_perturbed=tec_truth_arc0,
    )
    plot_parameter_influence(
        rays_0, tang_km_0, mean_1gp, ALT_GRID,
        time_dt, arc_label_0, save_abs,
        n_sweep=_N_SWEEP, n_sigma=_N_SIGMA, abs_error=True, log_err_scale=True,
        tec_truth_perturbed=tec_truth_arc0,
    )

    # ── Step 6b: build simulated IGS geometry once (shared by ro_igs / igs_only) ─
    igs_sim_geometry: list[dict] = []
    if USE_SIMULATED_IGS:
        print(f"\nStep 6b: Building simulated IGS geometry from broadcast ephemeris "
              f"(stations {IGS_SIM_STATIONS}) …")
        igs_stations = _load_igs_sim_stations(
            IGS_SIM_STATIONS_JSON, IGS_SIM_STATIONS,
            roi_max_km=ISR_ROI_MAX_KM,
        )
        if igs_stations:
            ephem = _load_broadcast_ephemeris(time_dt)
            if ephem is not None:
                igs_sim_geometry = _build_igs_sim_arcs(
                    igs_stations, time_dt, ephem,
                    window_minutes=WINDOW_MINUTES,
                )
        else:
            print("  [IGS-sim] No stations resolved within ROI — no IGS arcs.")

    # ── Step 6c: build an IGS-centred grid + prior for the igs_only mode ─────
    # The RO-derived grid is anchored on tangent tracks; when the filter has
    # no RO observations at all we want the ROI to follow the IGS pierce
    # points instead, so we build a separate Fibonacci grid from them.
    igs_grid_lats_1deg = igs_grid_lons_1deg = None
    igs_grid_lats_5deg = igs_grid_lons_5deg = None
    igs_mean_5deg      = None
    if igs_sim_geometry:
        print("\nStep 6c: Building IGS-only Fibonacci grids from IPP points …")
        ipp_lats_all = np.concatenate([a["tp_lats"] for a in igs_sim_geometry])
        ipp_lons_all = np.concatenate([a["tp_lons"] for a in igs_sim_geometry])
        igs_grid_lats_1deg, igs_grid_lons_1deg = _make_fibonacci_grid(
            ipp_lats_all, ipp_lons_all, spacing_deg=1.0,
        )
        igs_grid_lats_5deg, igs_grid_lons_5deg = _make_fibonacci_grid(
            ipp_lats_all, ipp_lons_all, spacing_deg=5.0,
        )
        print(f"  IGS 1-deg grid: {len(igs_grid_lats_1deg)} nodes  "
              f"({float(igs_grid_lats_1deg.min()):.1f}° "
              f"→ {float(igs_grid_lats_1deg.max()):.1f}° lat, "
              f"{float(igs_grid_lons_1deg.min()):.1f}° "
              f"→ {float(igs_grid_lons_1deg.max()):.1f}° lon)")
        print(f"  IGS 5-deg grid: {len(igs_grid_lats_5deg)} nodes")

        igs_lat_min_5 = float(igs_grid_lats_5deg.min())
        igs_lat_max_5 = float(igs_grid_lats_5deg.max())
        igs_lon_min_5 = float(igs_grid_lons_5deg.min())
        igs_lon_max_5 = float(igs_grid_lons_5deg.max())
        igs_mean_5deg, _ = build_iri_state_grid_cached(
            time_dt, igs_grid_lats_5deg, igs_grid_lons_5deg, ALT_GRID,
            spacing_deg=5.0,
            lat_min=igs_lat_min_5, lat_max=igs_lat_max_5,
            lon_min=igs_lon_min_5, lon_max=igs_lon_max_5,
        )

    # ── Steps 7–14: filter retrieval experiment, once per observation mode ───
    filter_results: dict[str, dict] = {}
    for mode in FILTER_MODES:
        if mode == "ro_igs" and not igs_sim_geometry:
            print(f"\n[skip] mode=ro_igs — no IGS geometry available; using "
                  f"ro_only result instead.")
            continue
        if mode == "igs_only" and (not igs_sim_geometry or igs_mean_5deg is None):
            print(f"\n[skip] mode=igs_only — no IGS geometry / grid available.")
            continue

        if mode == "igs_only":
            mode_grid_1 = (igs_grid_lats_1deg, igs_grid_lons_1deg)
            mode_grid_5 = (igs_grid_lats_5deg, igs_grid_lons_5deg)
            mode_mean_5 = igs_mean_5deg
        else:
            mode_grid_1 = (grid_lats_1deg, grid_lons_1deg)
            mode_grid_5 = (grid_lats_5deg, grid_lons_5deg)
            mode_mean_5 = mean_5deg

        print("\n" + "-" * 66)
        print(f"  Observation mode: {mode}")
        print("-" * 66)
        filter_results[mode] = _run_enkf_retrieval_experiment(
            parsed_list      = parsed_list,
            meta_list        = records,
            mean_5deg        = mode_mean_5,
            grid_lats_1deg   = mode_grid_1[0],
            grid_lons_1deg   = mode_grid_1[1],
            grid_lats_5deg   = mode_grid_5[0],
            grid_lons_5deg   = mode_grid_5[1],
            time_dt          = time_dt,
            save_dir         = save_dir,
            save_suffix      = f"{save_suffix}_{mode}",
            mode             = mode,
            igs_sim_geometry = igs_sim_geometry if mode in ("ro_igs", "igs_only") else None,
        )

    # Backward-compatible primary result for cross-window aggregation: prefer
    # ro_igs (the fullest observation set) → ro_only → igs_only.
    primary_result = (
        filter_results.get("ro_igs")
        or filter_results.get("ro_only")
        or filter_results.get("igs_only")
    )

    # ── Step 14b: cross-mode group summary metrics figure ────────────────────
    # Mirrors plotIonosphereTomography._plot_group_summary_metrics from the
    # ISR-DA pipeline, but with the synthetic truth EDP substituted for real
    # ISR profiles.  Skipped if none of the three modes produced a result.
    if filter_results:
        print("\nStep 14b: Plotting cross-mode group summary metrics …")
        try:
            plot_group_summary_metrics_simulated(
                filter_results_by_mode = filter_results,
                save_dir               = save_dir,
                hhmm                   = hhmm,
            )
        except Exception as exc:
            print(f"  [warn] group summary metrics plot failed: "
                  f"{type(exc).__name__}: {exc}")

    return {
        "window_key":     window_key,
        "hhmm":           hhmm,
        "time_dt":        time_dt,
        "n_arcs":         len(parsed_list),
        "records":        records,
        "parsed_list":    parsed_list,
        "truth_arcs":     truth_arcs,
        "model_arcs":     model_arcs,
        "grid_1deg":      (grid_lats_1deg, grid_lons_1deg),
        "grid_5deg":      (grid_lats_5deg, grid_lons_5deg),
        "filter_result":  primary_result,
        "filter_results": filter_results,
    }


def main() -> None:
    os.makedirs(SAVE_DIR,      exist_ok=True)
    os.makedirs(IRI_CACHE_DIR, exist_ok=True)

    # ── Step 1: full-day file scan → per-window record lists ─────────────────
    print("=" * 60)
    print(f"Step 1: Scanning podTc2 files in {BASE_PATH}")
    print(f"        window size = {WINDOW_MINUTES} min, "
          f"min arcs/window = {MIN_ARCS_PER_WINDOW}")
    windows = scan_and_select_files_per_window(BASE_PATH)

    if not windows:
        print("No windows retained after ROI + sparsity filtering.  Exiting.")
        return

    print("\n" + "=" * 60)
    print(f"Processing {len(windows)} window(s) on {YYYY}.{DOY:03d}")
    print("=" * 60)

    all_results: dict[str, dict] = {}
    for i, w in enumerate(windows):
        header = f" Window {i+1}/{len(windows)}: {w['window_key']} " \
                 f"({len(w['records'])} arcs) "
        print("\n" + "#" * 70)
        print("#" + header.center(68) + "#")
        print("#" * 70)
        try:
            all_results[w["window_key"]] = _process_time_window(w, SAVE_DIR)
        except Exception as exc:
            print(f"  ✗ Window {w['window_key']} FAILED: {type(exc).__name__}: {exc}")
            all_results[w["window_key"]] = {
                "window_key": w["window_key"],
                "hhmm":       w["hhmm"],
                "time_dt":    w["time_dt"],
                "error":      f"{type(exc).__name__}: {exc}",
            }

    # ── Cross-window summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f" Cross-window summary — {YYYY}.{DOY:03d}")
    print("=" * 60)
    print(f"  {'Window':<20}  {'mode':<9}  {'n_arcs':>6}  "
          f"{'prior':>8}  {'KF':>8}  {'EKF_P':>8}")

    def _fmt(x):
        return f"{x:>8.3f}" if isinstance(x, (int, float)) else " " * 8

    for wkey, r in all_results.items():
        if "error" in r:
            print(f"  {wkey:<20}  ERROR: {r['error']}")
            continue
        per_mode = r.get("filter_results") or {}
        if not per_mode:
            # Legacy single-result path (shouldn't trigger now, but keep safe).
            fr = r.get("filter_result") or {}
            prior = fr.get("prior_rmse")
            kf    = (fr.get("kf_result") or {}).get("post_rmse")
            ekf   = (fr.get("ekf_param") or {}).get("post_rmse")
            print(f"  {wkey:<20}  {'(single)':<9}  "
                  f"{r.get('n_arcs', 0):>6d}  "
                  f"{_fmt(prior)}  {_fmt(kf)}  {_fmt(ekf)}")
            continue
        for mode in FILTER_MODES:
            fr = per_mode.get(mode)
            if fr is None:
                print(f"  {wkey:<20}  {mode:<9}  {'skipped':>6}")
                continue
            prior = fr.get("prior_rmse")
            kf    = (fr.get("kf_result") or {}).get("post_rmse")
            ekf   = (fr.get("ekf_param") or {}).get("post_rmse")
            n_arc = len(fr.get("arc_truth_list") or [])
            print(f"  {wkey:<20}  {mode:<9}  {n_arc:>6d}  "
                  f"{_fmt(prior)}  {_fmt(kf)}  {_fmt(ekf)}")

    n_err = sum(1 for r in all_results.values() if "error" in r)

    # ── Cross-window EDP-RMSE plots (per-site & regional) ────────────────────
    n_ok = len(all_results) - n_err
    if n_ok > 0:
        print("\n" + "=" * 60)
        print(f" Cross-window EDP-RMSE plots — {YYYY}.{DOY:03d}")
        print("=" * 60)
        plot_edp_site_rmse_across_windows(
            all_results, ALT_GRID,
            save_path=os.path.join(
                SAVE_DIR, f"edp_site_rmse_across_windows_{YYYY}_{DOY:03d}.png"
            ),
        )
        plot_edp_regional_rmse_across_windows(
            all_results, ALT_GRID,
            save_path=os.path.join(
                SAVE_DIR, f"edp_regional_rmse_across_windows_{YYYY}_{DOY:03d}.png"
            ),
        )

    print(f"\n✓ test_param_iono.py completed.")
    print(f"  Windows attempted: {len(all_results)}  "
          f"(errors: {n_err})   Figures in: {SAVE_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# §7  Truth ionosphere: 1×1 grid, +1 h, adjusted F10.7
# ─────────────────────────────────────────────────────────────────────────────

def _truth_solar_conditions(
    base_time: pd.Timestamp,
    hour_offset: float = TRUTH_HOUR_OFFSET,
    f107_delta: float  = TRUTH_F107_DELTA,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """
    Return (truth_time, sampling_df) for the truth ionosphere.

    truth_time    = base_time + hour_offset hours
    sampling_df   = solar-index DataFrame with F10.7 incremented by f107_delta
    """
    truth_time   = base_time + pd.Timedelta(hours=hour_offset)
    df           = _solar_sampling_df(truth_time).copy()
    df["f107"]   = df["f107"] + f107_delta
    return truth_time, df


def build_truth_iri_grid(
    truth_time: pd.Timestamp,
    truth_sampling_df: pd.DataFrame,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    alt_grid: np.ndarray,
    label: str = "",
) -> tuple[IonosphericState, np.ndarray, np.ndarray]:
    """
    Evaluate IRI at every grid point using the modified solar conditions and
    assemble the 8-parameter state.

    Returns
    -------
    truth_state  : single-member IonosphericState (the deterministic truth).
    ne_profiles  : (n_alt, n_grid) electron density profiles (m⁻³) reconstructed
                   from the fitted 8-parameter state (not the raw IRI lookup).
    mean_state   : (N_STATE, n_grid) parameter state in log/km/dim-less units.
    """
    n_grid = len(grid_lats)
    tag    = f" [{label}]" if label else ""
    print(f"  Building IRI truth grid{tag}  ({n_grid} pts, "
          f"{truth_time.strftime('%H:%M')} UTC, "
          f"F10.7+{TRUTH_F107_DELTA:.0f}) …")

    ne_profiles_iri, feature_vecs = _get_iri_edp_and_features_batch(
        truth_time, grid_lats, grid_lons, alt_grid, truth_sampling_df,
    )
    mean_state = np.empty((N_STATE, n_grid))
    for g in range(n_grid):
        mean_state[:, g] = _state_from_iri_direct(
            ne_profiles_iri[:, g], feature_vecs[:, g], alt_grid,
        )

    truth_state          = IonosphericState(n_grid_points=n_grid, n_members=1)
    truth_state.ensemble = mean_state[:, :, np.newaxis].copy()
    truth_state.clamp_to_physical_bounds()

    # Reconstruct Ne profiles from the fitted 8-parameter state so that
    # truth_ne_{1,5}deg used in EDP plots is consistent with what the
    # ObservationOperator integrates (the analytic 8-param formula), not the
    # raw IRI lookup which the forward model never directly uses.
    ne_profiles = _parametric_to_edp(truth_state, truth_state.ensemble, alt_grid)

    return truth_state, ne_profiles, mean_state


# ─────────────────────────────────────────────────────────────────────────────
# §8  Generate truth TEC measurements
# ─────────────────────────────────────────────────────────────────────────────

def generate_truth_tec(
    parsed_list: list[dict],
    truth_state: IonosphericState,
    grid_lats_truth: np.ndarray,
    grid_lons_truth: np.ndarray,
    alt_grid: np.ndarray,
) -> list[dict]:
    """
    Evaluate the truth IonosphericState along every arc ray to produce
    synthetic sTEC observations.

    Parameters
    ----------
    parsed_list      : list of parsed podTc2 arc dicts.
    truth_state      : single-member IonosphericState on the 1-deg truth grid.
    grid_lats/lons_truth : (n_truth,) 1-deg Fibonacci grid coordinates.
    alt_grid         : altitude integration grid (km).

    Returns
    -------
    List of dicts — one per arc — with keys:
        rays, tec_truth, tp_lats, tp_lons, tang_km, conid, prn_id, leo_id
    """
    obs_truth = ObservationOperator(truth_state, alt_grid)
    arc_list: list[dict] = []

    for arc_idx, arc in enumerate(parsed_list):
        prn_id = str(arc.get("prn_id", arc.get("prn", "?")))
        leo_id = str(arc.get("leo_id", "?"))
        conid  = str(arc.get("conid",
                    prn_id[0].upper() if prn_id[0].upper() in "GREC" else "?")).upper()

        rays, tp_lats, tp_lons, tang_km, _ = _build_arc_rays(arc)
        n_ep = len(rays)

        # IDW weights from each epoch's tangent point to the 1-deg truth grid
        W = np.stack([
            _idw_weights(tp_lats[k], tp_lons[k],
                         grid_lats_truth, grid_lons_truth)
            for k in range(n_ep)
        ])  # (n_ep, n_truth)

        Y_truth = obs_truth.compute_stec_ensemble(
            rays, grid_point_weights=W,
        )  # (n_ep, 1)
        tec_truth = Y_truth[:, 0]   # (n_ep,)

        print(f"  Arc {arc_idx+1}/{len(parsed_list)}: "
              f"{leo_id} {conid}{prn_id}  "
              f"{n_ep} epochs  "
              f"TEC {tec_truth.min():.1f}–{tec_truth.max():.1f} TECU")

        arc_list.append(dict(
            rays      = rays,
            tec_truth = tec_truth,
            tp_lats   = tp_lats,
            tp_lons   = tp_lons,
            tang_km   = tang_km,
            conid     = conid,
            prn_id    = prn_id,
            leo_id    = leo_id,
        ))

    return arc_list
def _ekf_param_jacobian(
    mean_state_2d: np.ndarray,
    ray_list: list,
    idw_weights: np.ndarray,
    alt_grid: np.ndarray,
    n_grid: int,
    eps_rel: float = 1e-4,
    n_workers: int = 1,
) -> tuple:
    """
    Vectorised finite-difference Jacobian of the sTEC forward model.

    All N_STATE*n_grid perturbations are batched into a single forward-model
    call rather than computed serially.  The ensemble has n_state+1 members:
    member 0 is the unperturbed baseline; member j+1 has state element j
    increased by dv[j].  A single call to compute_stec_ensemble then returns
    all predicted sTEC values in one vectorised sweep, eliminating the
    O(N_STATE * n_grid) Python loop that previously dominated each EKF iteration.

    Thread-level parallelism over rays is forwarded to compute_stec_ensemble
    via n_workers when > 1.

    Parameters
    ----------
    mean_state_2d : (N_STATE, n_grid)   current EKF iterate.
    ray_list      : list of (n_pts, 3) arrays  [lat, lon, alt_km] per ray.
    idw_weights   : (n_rays, n_grid)   normalised IDW weights per ray.
    alt_grid      : (n_alt,)
    n_grid        : int
    eps_rel       : float   relative perturbation size.
    n_workers     : int     threads for the ray loop inside compute_stec_ensemble.

    Returns
    -------
    J  : ndarray, shape (n_rays, N_STATE * n_grid)
    y0 : ndarray, shape (n_rays,)   baseline sTEC predictions.
    """
    N_S     = mean_state_2d.shape[0]        # == N_STATE
    n_state = N_S * n_grid

    flat = mean_state_2d.ravel()            # (n_state,)  C-order
    dv   = np.maximum(np.abs(flat) * eps_rel, 1e-8)   # (n_state,)

    # Build the perturbed ensemble in one allocation.
    # Shape: (N_STATE, n_grid, n_state + 1)
    # Member 0 = baseline; member j+1 has element j of the flat state += dv[j].
    ens = np.tile(mean_state_2d[:, :, np.newaxis], (1, 1, n_state + 1))

    # A C-contiguous (N_STATE, n_grid, n_state+1) array reshapes to
    # (N_STATE*n_grid, n_state+1) = (n_state, n_state+1) as a *view*.
    # ens_flat[j, k] == ens[j // n_grid, j % n_grid, k]
    ens_flat = ens.reshape(n_state, n_state + 1)
    ens_flat[np.arange(n_state), np.arange(1, n_state + 1)] += dv

    tmp_state = IonosphericState(n_grid, n_members=n_state + 1)
    tmp_state.ensemble = ens
    tmp_op = ObservationOperator(tmp_state, alt_grid)

    Y = tmp_op.compute_stec_ensemble(
        ray_list,
        grid_point_weights=idw_weights,
        n_workers=n_workers,
    )                                       # (n_rays, n_state + 1)

    y0 = Y[:, 0]                            # (n_rays,)
    J  = (Y[:, 1:] - y0[:, np.newaxis]) / dv[np.newaxis, :]   # (n_rays, n_state)

    return J, y0


def _ne_profile_derivatives(
    alts_km: np.ndarray,
    params_lin: np.ndarray,
) -> tuple:
    """
    Analytical ∂Ne(h)/∂P_k for each altitude and each state parameter.

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
    from Ionosphere_Tomography_Inverter.observation_operator import (
        _R_TOPSIDE, _H_E_KM, _find_hst_bisection,
    )
    from Ionosphere_Tomography_Inverter.ionospheric_state import (
        I_LOG_NMF2, I_HMF2, I_H0, I_GAMMA, I_B0, I_B1, I_LOG_NME, I_HME,
    )

    NmF2  = float(params_lin[I_LOG_NMF2])
    hmF2  = float(params_lin[I_HMF2])
    H0    = float(params_lin[I_H0])
    gamma = float(params_lin[I_GAMMA])
    B0    = float(params_lin[I_B0])
    B1    = float(params_lin[I_B1])
    NmE   = float(params_lin[I_LOG_NME])
    hmE   = float(params_lin[I_HME])

    h     = alts_km                          # (n_alt,)
    n_alt = len(h)
    dNe_dP = np.zeros((N_STATE, n_alt), dtype=float)

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

    return Ne, dNe_dP


def _ekf_param_analytical_jacobian(
    mean_state_2d: np.ndarray,
    ray_list: list,
    idw_weights: np.ndarray,
    alt_grid: np.ndarray,
    n_grid: int,
    n_workers: int = 1,
) -> tuple:
    """
    Analytical Jacobian of the sTEC forward model.

    For each ray i and state element (k, g):

        J[i, k*n_grid + g] = w_{i,g} * ∫ ∂Ne_k(h; params_g) d(path) / TECU

    where ∂Ne_k is the analytical derivative from _ne_profile_derivatives
    and the path integral uses the trapezoidal rule over the ray's altitude
    samples.

    Parameters
    ----------
    mean_state_2d : (N_STATE, n_grid)  current EKF iterate (log10/km/dimless)
    ray_list      : list of (n_pts, 3) arrays  [lat, lon, alt_km] per ray
    idw_weights   : (n_rays, n_grid)   normalised IDW weights per ray
    alt_grid      : (n_alt,)           unused (altitudes come from the rays)
    n_grid        : int
    n_workers     : int  threads for the outer ray loop

    Returns
    -------
    J  : (n_rays, N_STATE * n_grid)  analytical Jacobian
    y0 : (n_rays,)                   baseline sTEC predictions
    """
    from Ionosphere_Tomography_Inverter.ionospheric_state import LOG_INDICES
    from Ionosphere_Tomography_Inverter.observation_operator import (
        ObservationOperator as _ObsOp, _TECU,
    )

    n_rays  = len(ray_list)
    n_state = N_STATE * n_grid

    # Convert log10 densities → linear for all grid points
    params_lin = mean_state_2d.copy()              # (N_STATE, n_grid)
    params_lin[LOG_INDICES] = 10.0 ** params_lin[LOG_INDICES]

    J  = np.zeros((n_rays, n_state), dtype=float)
    y0 = np.zeros(n_rays,           dtype=float)

    def _process_ray(i):
        ray     = ray_list[i]
        alts_km = ray[:, 2]
        path_km = _ObsOp._arc_length_km(ray)      # (n_pts,) cumulative arc-length

        w      = idw_weights[i]
        active = np.where(w > 0.0)[0]
        w_norm = w[active] / (w[active].sum() + 1e-30)

        y0_i   = 0.0
        dJ_row = np.zeros(n_state, dtype=float)

        for gp, wg in zip(active, w_norm):
            Ne_gp, dNe_gp = _ne_profile_derivatives(alts_km, params_lin[:, gp])
            scale = wg * 1.0e3 / _TECU               # km → m, then → TECU
            y0_i += scale * float(_trapz(Ne_gp, path_km))
            for k in range(N_STATE):
                dJ_row[k * n_grid + gp] += scale * float(_trapz(dNe_gp[k], path_km))

        return i, y0_i, dJ_row

    if n_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_process_ray, range(n_rays)))
        for i, y0_i, dJ_row in results:
            y0[i]    = y0_i
            J[i, :]  = dJ_row
    else:
        for i in range(n_rays):
            _, y0_i, dJ_row = _process_ray(i)
            y0[i]   = y0_i
            J[i, :] = dJ_row

    return J, y0


def EKF_Param(
    arc_truth_list: list,
    model_state: "IonosphericState",
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    alt_grid: np.ndarray,
    sigma_obs: float     = ENKF_SIGMA_OBS,
    max_update_rays: int = ENKF_MAX_UPDATE_RAYS,
    alpha: float         = 0.5,
    tol: float           = 1e-4,
    max_iter: int        = 20,
    apply_bounds: bool   = True,
    eps_jac: float       = 1e-4,
    n_workers: int       = 1,
    jacobian_analytical: bool = False,
) -> dict:
    """
    Iterative Extended Kalman Filter on the parametric IRI state vector.

    State vector P has shape (N_STATE * n_grid,) — the 8 IRI parameters at
    every horizontal grid point, stored in C order (parameter index varies
    slowest, grid-point index varies fastest within each parameter block).
    Density parameters (log10 NmF2, log10 NmE) remain in log10 space
    throughout, matching the IonosphericState convention.

    Prior covariance Q is factored through the ensemble anomaly matrix so that
    the full (n_state × n_state) matrix is never formed explicitly:

        Q ≈ X_c X_c^T / (M - 1),   X_c = ensemble anomalies, shape (n_state, M)

    EKF iteration (i = 0, 1, …):

        J_i    = ∂H(P_i)/∂P          (Jacobian, (n_obs, n_state), finite diff.)
        Ŵ_i    = J_i                  (effective linearised observation matrix)
        G_i    = α (X_c X_c^T / nm1) Ŵ_i^T (Ŵ_i X_c X_c^T Ŵ_i^T / nm1 + R)^{-1}
               = α X_c A_i^T / nm1 * solve(A_i A_i^T / nm1 + R)
               where A_i = Ŵ_i X_c = J_i X_c   (n_obs, M)
        ΔP_i   = G_i (y_obs − H(P_i))
        P_{i+1} = P_i + ΔP_i

    Stops when  ||ΔP||₂ / ||P_i||₂ < tol  or after max_iter steps.

    Parameters
    ----------
    arc_truth_list  : list of arc dicts (output of generate_truth_tec).
    model_state     : IonosphericState  prior ensemble; member 0 = IRI baseline.
    grid_lats/lons  : (n_grid,)  horizontal grid coordinates.
    alt_grid        : (n_alt,)   altitude integration grid (km).
    sigma_obs       : observation noise standard deviation (TECU).
    max_update_rays : maximum representative rays per arc used in the update.
    alpha           : EKF step-size scale ∈ (0, 1].  alpha < 1 limits the step
                      analogously to gradient-descent damping.
    tol             : relative convergence tolerance on the state update norm.
    max_iter        : maximum EKF iterations.
    apply_bounds        : clamp P to IonosphericState.PARAM_BOUNDS after each step.
    eps_jac             : relative perturbation size for finite-difference Jacobian.
    jacobian_analytical : if True, use the analytical Jacobian from
                          _ekf_param_analytical_jacobian instead of the
                          finite-difference version.  The analytical Jacobian
                          treats h_ST as a fixed boundary (standard
                          piecewise-smooth approximation) and is exact for all
                          other parameters.  Faster than finite differences
                          for large state vectors; eps_jac is ignored.

    Returns
    -------
    dict with the same keys as run_gridded_ne_kf for
    compatibility with downstream diagnostic plots, plus EKF-specific keys:
        converged, n_iterations, residual_history, update_norm_history.
    """
    from Ionosphere_Tomography_Inverter.enkf_update import flatten_ensemble

    n_geo     = model_state.n_grid_points
    n_members = model_state.n_members
    n_occ     = len(arc_truth_list)
    _idw_k    = min(4, n_geo)

    # ── 1. Decimate update rays (same stride as EnKF / KF) ───────────────────
    rep_rays:          list = []
    rep_tp_lats_list:  list = []
    rep_tp_lons_list:  list = []
    rep_tec_obs_list:  list = []
    arc_update_counts: list = []
    ray_counts:        list = []
    arc_all_tec:       list = []
    per_arc_tp_lats:   list = []
    per_arc_tp_lons:   list = []

    for arc in arc_truth_list:
        rays    = arc["rays"]
        tec     = arc["tec_truth"]
        tp_lats = arc["tp_lats"]
        tp_lons = arc["tp_lons"]
        n_s     = len(rays)

        per_arc_tp_lats.append(tp_lats)
        per_arc_tp_lons.append(tp_lons)
        ray_counts.append(n_s)
        arc_all_tec.append(tec)

        if n_s > max_update_rays:
            stride = int(np.ceil(n_s / max_update_rays))
            chosen = list(range(0, n_s, stride))
            if n_s - 1 not in chosen:
                chosen.append(n_s - 1)
        else:
            chosen = list(range(n_s))

        for idx in chosen:
            rep_rays.append(rays[idx])
            rep_tp_lats_list.append(float(tp_lats[idx]))
            rep_tp_lons_list.append(float(tp_lons[idx]))
            rep_tec_obs_list.append(float(tec[idx]))
        arc_update_counts.append(len(chosen))

    y_obs_arc = np.array(rep_tec_obs_list)   # (n_obs,)
    y_obs_all = np.concatenate(arc_all_tec)
    n_obs     = len(rep_rays)

    all_tp_lats_flat = np.concatenate([a.tolist() for a in per_arc_tp_lats])
    all_tp_lons_flat = np.concatenate([a.tolist() for a in per_arc_tp_lons])
    all_sample_rays  = [r for arc in arc_truth_list for r in arc["rays"]]

    print(f"  [EKF] {n_occ} arcs  |  "
          f"{len(all_sample_rays)} profile rays  |  "
          f"{n_obs} update rays  |  n_state = {N_STATE * n_geo}")

    # ── 2. IDW weights ────────────────────────────────────────────────────────
    rep_W = _idw_weights_enkf(
        np.array(rep_tp_lats_list), np.array(rep_tp_lons_list),
        grid_lats, grid_lons, k=_idw_k,
    )
    all_sample_W = _idw_weights_enkf(
        all_tp_lats_flat, all_tp_lons_flat,
        grid_lats, grid_lons, k=_idw_k,
    )

    # ── 3. Prior ensemble anomaly X_c and initial state P_0 ──────────────────
    # Use member 0 (unperturbed IRI baseline) as the starting iterate P_0.
    # X_c captures the prior uncertainty for the covariance factor.
    X_f = flatten_ensemble(model_state.ensemble)         # (n_state, M)
    mu  = X_f.mean(axis=1)                               # (n_state,) ensemble mean
    X_c = X_f - mu[:, np.newaxis]                        # (n_state, M) anomalies
    nm1 = max(n_members - 1, 1)

    P_i = X_f[:, 0].copy()                               # (n_state,) P_0 = member 0

    # ── 4. Prior diagnostics (all sample rays) ────────────────────────────────
    prior_state_snap = IonosphericState(n_geo, n_members=1)
    prior_state_snap.ensemble = P_i.reshape(N_STATE, n_geo)[:, :, np.newaxis].copy()
    prior_op  = ObservationOperator(prior_state_snap, alt_grid)

    y_prior_all  = prior_op.compute_stec_ensemble(
        all_sample_rays, grid_point_weights=all_sample_W, n_workers=n_workers,
    )[:, 0]
    y_prior_arc  = prior_op.compute_stec_ensemble(
        rep_rays, grid_point_weights=rep_W, n_workers=n_workers,
    )[:, 0]

    prior_inno = y_obs_arc - y_prior_arc
    print(f"  [EKF] Prior innovations  "
          f"mean={prior_inno.mean():.2f}  std={prior_inno.std():.2f}  "
          f"max_abs={np.abs(prior_inno).max():.2f} TECU")

    # ── 5. Observation noise covariance R ─────────────────────────────────────
    R_mat = (sigma_obs ** 2) * np.eye(n_obs)

    # ── 6. EKF iteration ──────────────────────────────────────────────────────
    residual_history:    list = []
    update_norm_history: list = []
    converged = False
    rel_norm  = np.inf

    jac_mode = "analytical" if jacobian_analytical else "finite-diff"
    _iter_bar = tqdm(
        range(max_iter), desc=f"  [EKF] iterating [{jac_mode}]", unit="it", leave=False,
    )
    for it in _iter_bar:
        P_2d = P_i.reshape(N_STATE, n_geo)

        # Jacobian Ŵ_i = ∂H(P_i)/∂P  and baseline predictions y_hat
        if jacobian_analytical:
            J_i, y_hat = _ekf_param_analytical_jacobian(
                P_2d, rep_rays, rep_W, alt_grid, n_geo,
                n_workers=n_workers,
            )
        else:
            J_i, y_hat = _ekf_param_jacobian(
                P_2d, rep_rays, rep_W, alt_grid, n_geo,
                eps_rel=eps_jac, n_workers=n_workers,
            )

        # Innovation at current iterate
        innov = y_obs_arc - y_hat                        # (n_obs,)
        resid = float(np.sqrt(np.mean(innov ** 2)))
        residual_history.append(resid)

        # Low-rank covariance products — avoid forming Q explicitly:
        #   A_i = J_i X_c                  (n_obs, M)
        #   P_yy ≈ A_i A_i^T / nm1         (n_obs, n_obs)
        #   P_xy ≈ X_c A_i^T   / nm1       (n_state, n_obs)
        A_i  = J_i @ X_c                                 # (n_obs, M)
        P_yy = (A_i @ A_i.T) / nm1                       # (n_obs, n_obs)
        P_xy = (X_c @ A_i.T) / nm1                       # (n_state, n_obs)

        S = P_yy + R_mat                                  # (n_obs, n_obs)
        try:
            import scipy.linalg as _la
            K_T = _la.solve(S, P_xy.T, assume_a="pos")   # (n_obs, n_state)
        except Exception:
            K_T = np.linalg.lstsq(S, P_xy.T, rcond=None)[0]
        G = alpha * K_T.T                                 # (n_state, n_obs)

        dP = G @ innov                                    # (n_state,)

        # Convergence check before applying the step
        rel_norm = float(np.linalg.norm(dP) / (np.linalg.norm(P_i) + 1e-30))
        update_norm_history.append(rel_norm)
        _iter_bar.set_postfix(RMSE=f"{resid:.3f}TECU", rel_dP=f"{rel_norm:.2e}")

        P_i = P_i + dP

        # Clamp to physical bounds after each step
        if apply_bounds:
            P_2d_new = P_i.reshape(N_STATE, n_geo)
            for k_p, (lo, hi) in enumerate(IonosphericState.PARAM_BOUNDS):
                P_2d_new[k_p] = np.clip(P_2d_new[k_p], lo, hi)
            # Structural constraint: hmF2 > hmE + 20 km
            from Ionosphere_Tomography_Inverter.ionospheric_state import I_HMF2, I_HME
            P_2d_new[I_HMF2] = np.maximum(P_2d_new[I_HMF2], P_2d_new[I_HME] + 20.0)
            P_i = P_2d_new.ravel()

        if rel_norm < tol:
            converged = True
            _iter_bar.write(f"  [EKF] Converged at iteration {it + 1}  "
                             f"(||ΔP||/||P|| = {rel_norm:.2e} < tol={tol:.1e})")
            break
    else:
        _iter_bar.write(f"  [EKF] Reached max_iter={max_iter} without convergence  "
                         f"(final ||ΔP||/||P|| = {rel_norm:.2e})")
    _iter_bar.close()

    n_iterations = len(residual_history)

    # ── 6b. Analytical prior/posterior covariance (dense, parametric space) ──
    # Reuses the FINAL iteration's linearization (J_i via P_xy/S, and X_c/nm1)
    # already computed inside the loop above — no extra Jacobian evaluation.
    # State ordering is (N_STATE, n_geo) C-order (param-major, geo-minor), the
    # same convention as P_i.reshape(N_STATE, n_geo) — this is NOT the gridded
    # KF's (n_alt, n_geo) Ne-space ordering, since EKF_Param's state is the 8
    # Chapman/IRI parameters per grid point, not Ne(alt) itself.
    n_state = N_STATE * n_geo
    prior_P = (X_c @ X_c.T) / nm1                        # (n_state, n_state)
    try:
        import scipy.linalg as _la
        K_gain = _la.solve(S, P_xy.T, assume_a="pos").T  # (n_state, n_obs), unscaled (no alpha damping)
    except Exception:
        K_gain = np.linalg.lstsq(S, P_xy.T, rcond=None)[0].T
    post_P = prior_P - K_gain @ P_xy.T                   # (I - KH) P_prior
    post_P = 0.5 * (post_P + post_P.T)                   # enforce symmetry

    # ── 7. Posterior predictions (all sample rays) ────────────────────────────
    P_post_2d   = P_i.reshape(N_STATE, n_geo)
    post_state  = IonosphericState(n_geo, n_members=1)
    post_state.ensemble = P_post_2d[:, :, np.newaxis].copy()
    post_op = ObservationOperator(post_state, alt_grid)

    y_post_all = post_op.compute_stec_ensemble(
        all_sample_rays, grid_point_weights=all_sample_W, n_workers=n_workers,
    )[:, 0]
    y_post_arc = post_op.compute_stec_ensemble(
        rep_rays, grid_point_weights=rep_W, n_workers=n_workers,
    )[:, 0]

    post_inno = y_obs_arc - y_post_arc
    print(f"  [EKF] Posterior innovations  "
          f"mean={post_inno.mean():.2f}  std={post_inno.std():.2f}  "
          f"max_abs={np.abs(post_inno).max():.2f} TECU")

    prior_rmse = float(np.sqrt(np.nanmean((y_obs_all - y_prior_all) ** 2)))
    post_rmse  = float(np.sqrt(np.nanmean((y_obs_all - y_post_all) ** 2)))
    print(f"  [EKF] Prior RMSE {prior_rmse:.3f} TECU  →  "
          f"Post RMSE {post_rmse:.3f} TECU")

    # ── 8. Convert parametric states → Ne profile grids ──────────────────────
    prior_state_for_edp = IonosphericState(n_geo, n_members=1)
    prior_state_for_edp.ensemble = X_f[:, 0].reshape(N_STATE, n_geo)[:, :, np.newaxis].copy()
    prior_edp = _parametric_to_edp(prior_state_for_edp, prior_state_for_edp.ensemble, alt_grid)
    post_edp  = _parametric_to_edp(post_state, post_state.ensemble, alt_grid)

    # ── 9. TEC slices and per-arc statistics ─────────────────────────────────
    tec_slices:      list = []
    all_prior_resid  = y_obs_all - y_prior_all
    all_post_resid   = y_obs_all - y_post_all

    arc_prior_mean_l, arc_post_mean_l = [], []
    arc_prior_rmse_l, arc_post_rmse_l = [], []
    arc_lats_l, arc_lons_l, arc_lbl_l = [], [], []

    soff = 0
    for i, arc in enumerate(arc_truth_list):
        n_s = ray_counts[i]
        sl  = slice(soff, soff + n_s)
        tec_slices.append(dict(
            tec_truth = arc["tec_truth"],
            prior_tec = y_prior_all[sl].copy(),
            post_tec  = y_post_all[sl].copy(),
            tang_km   = arc["tang_km"],
        ))
        rp = all_prior_resid[sl]
        ra = all_post_resid[sl]
        arc_prior_mean_l.append(float(np.nanmean(rp)))
        arc_post_mean_l.append(float(np.nanmean(ra)))
        arc_prior_rmse_l.append(float(np.sqrt(np.nanmean(rp ** 2))))
        arc_post_rmse_l.append(float(np.sqrt(np.nanmean(ra ** 2))))
        arc_lats_l.append(float(arc["tp_lats"].mean()))
        arc_lons_l.append(float(arc["tp_lons"].mean()))
        arc_lbl_l.append(f"{arc['conid']}{arc['prn_id']}")
        soff += n_s

    return dict(
        prior_ne_5deg        = prior_edp,
        posterior_ne_5deg    = post_edp,
        prior_edp            = prior_edp,
        posterior_edp        = post_edp,
        prior_mean_state     = X_f[:, 0].reshape(N_STATE, n_geo).copy(),  # (N_STATE, n_geo)
        posterior_mean_5deg  = P_post_2d.copy(),          # (N_STATE, n_geo)
        posterior_mean_state = P_post_2d.copy(),
        prior_P              = prior_P,                  # (n_state, n_state), param-major/geo-minor
        post_P               = post_P,
        tec_slices           = tec_slices,
        prior_rmse           = prior_rmse,
        post_rmse            = post_rmse,
        all_prior_resid      = all_prior_resid,
        all_post_resid       = all_post_resid,
        arc_prior_mean       = np.array(arc_prior_mean_l),
        arc_post_mean        = np.array(arc_post_mean_l),
        arc_prior_rmse       = np.array(arc_prior_rmse_l),
        arc_post_rmse        = np.array(arc_post_rmse_l),
        arc_lats             = np.array(arc_lats_l),
        arc_lons             = np.array(arc_lons_l),
        arc_labels           = arc_lbl_l,
        grid_lats            = grid_lats,
        grid_lons            = grid_lons,
        mda_arc_means_list   = None,
        mda_flat_list        = None,
        converged            = converged,
        n_iterations         = n_iterations,
        residual_history     = residual_history,
        update_norm_history  = update_norm_history,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §10  EnKF TEC + globe + EDP spaghetti plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot_tec_edp_figure(
    arc_truth_list: list[dict],
    tec_slices: list[dict],
    prior_edp: np.ndarray,
    post_edp: np.ndarray,
    truth_ne_1deg: np.ndarray,
    grid_lats_1deg: np.ndarray,
    grid_lons_1deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
    suptitle: str,
    save_path: str,
    prior_mean_state: np.ndarray | None = None,
    post_mean_state:  np.ndarray | None = None,
) -> None:
    """
    Shared TEC + globe + EDP spaghetti figure used by both EnKF and KF wrappers.

    prior_mean_state, post_mean_state : optional (N_STATE, n_geo) parametric
        Chapman/Epstein state vectors (only meaningful for the parametric
        EnKF retrieval); when supplied, a colour-coded parameter box is drawn
        for the model-grid centre vertex.

    Layout — GridSpec(2, 4, width_ratios=[1,1,1.5,1.2])
    ──────────────────────────────────────────────────────
    [0,0] GPS        [0,1] Galileo     [0:2,2] Globe   [0:2,3] EDP spaghetti
    [1,0] GLONASS    [1,1] BeiDou                       (full-height column)
    """
    occ_colors = _occ_colors(
        [{"conid": a["conid"], "prn_id": a["prn_id"]} for a in arc_truth_list]
    )

    fig = plt.figure(figsize=(26, 9), facecolor="#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    gs  = GridSpec(2, 4, figure=fig,
                   width_ratios=[1, 1, 1.5, 1.2],
                   wspace=0.40, hspace=0.45,
                   left=0.05, right=0.97, top=0.92, bottom=0.10)

    # ── TEC panels ───────────────────────────────────────────────────────────
    tec_axes: dict[str, plt.Axes] = {}
    first_ax = None
    for const, (row, col) in _CONST_POS.items():
        ax = fig.add_subplot(gs[row, col],
                             sharey=first_ax if first_ax is not None else None)
        ax.set_facecolor("#2b2b2b")
        cfg = CONSTELLATION_CONFIG.get(const, {"name": const, "title_color": "white"})
        ax.set_title(cfg["name"], color=cfg.get("title_color", "white"),
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("sTEC (TECU)", color="lightgray", fontsize=7)
        ax.set_ylabel("Tang. alt. (km)", color="lightgray", fontsize=7)
        ax.tick_params(colors="lightgray", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        tec_axes[const] = ax
        if first_ax is None:
            first_ax = ax

    # ── Globe ─────────────────────────────────────────────────────────────────
    all_tp_lats = np.concatenate([a["tp_lats"] for a in arc_truth_list])
    all_tp_lons = np.concatenate([a["tp_lons"] for a in arc_truth_list])
    cen_lat = float(np.mean(all_tp_lats))
    cen_lon = float(np.mean(all_tp_lons))
    ax_globe = fig.add_subplot(
        gs[0:2, 2],
        projection=ccrs.Orthographic(
            central_longitude=cen_lon, central_latitude=cen_lat,
        ),
    )
    ax_globe.set_facecolor("#2b2b2b")
    ax_globe.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#aaaaaa")
    ax_globe.add_feature(cfeature.BORDERS,   linewidth=0.2, edgecolor="#888888")
    ax_globe.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)
    ax_globe.scatter(grid_lons_1deg, grid_lats_1deg,
                     s=1, color="lightgray", alpha=0.3,
                     transform=ccrs.PlateCarree(), zorder=2)
    ax_globe.scatter(grid_lons_5deg, grid_lats_5deg,
                     s=10, color="steelblue", alpha=0.7, marker="s",
                     transform=ccrs.PlateCarree(), zorder=3)

    # ── EDP spaghetti — full right column ────────────────────────────────────
    ax_edp = fig.add_subplot(gs[0:2, 3])
    ax_edp.set_facecolor("#2b2b2b")
    ax_edp.set_title("EDP — prior vs posterior", color="white", fontsize=8)
    ax_edp.set_xlabel("Ne (m⁻³)", color="lightgray", fontsize=7)
    ax_edp.set_ylabel("Altitude (km)", color="lightgray", fontsize=7)
    ax_edp.tick_params(colors="lightgray", labelsize=6)
    for sp in ax_edp.spines.values():
        sp.set_edgecolor("#555")

    # center of the 5-deg model grid (nearest node to the mean lat/lon)
    cen_5deg_dist = _haversine_km(
        float(np.mean(grid_lats_5deg)), float(np.mean(grid_lons_5deg)),
        grid_lats_5deg, grid_lons_5deg,
    )
    cen_5deg_idx = int(np.argmin(cen_5deg_dist))

    n_geo = prior_edp.shape[1]
    for g in range(n_geo):
        if g == cen_5deg_idx:
            continue  # drawn last so it sits on top
        ax_edp.plot(prior_edp[:, g], alt_grid, color="steelblue",
                    linewidth=0.7, alpha=0.4, zorder=3)
        ax_edp.plot(post_edp[:, g],  alt_grid, color="tomato",
                    linewidth=0.7, alpha=0.6, zorder=3)

    # Bold center profiles drawn last (highest zorder)
    ax_edp.plot(prior_edp[:, cen_5deg_idx], alt_grid, color="steelblue",
                linewidth=2.5, alpha=1.0, zorder=6)
    ax_edp.plot(post_edp[:, cen_5deg_idx],  alt_grid, color="tomato",
                linewidth=2.5, alpha=1.0, zorder=6)

    cen_1deg_dist = _haversine_km(
        float(np.mean(grid_lats_5deg)), float(np.mean(grid_lons_5deg)),
        grid_lats_1deg, grid_lons_1deg,
    )
    cen_1deg_idx = int(np.argmin(cen_1deg_dist))
    ax_edp.plot(truth_ne_1deg[:, cen_1deg_idx], alt_grid,
                color="yellow", linewidth=1.5, linestyle="--", zorder=7)

    edp_handles = [
        Line2D([0], [0], color="steelblue", lw=0.7, alpha=0.4, label="Prior Ne"),
        Line2D([0], [0], color="tomato",    lw=0.7, alpha=0.6, label="Posterior Ne"),
        Line2D([0], [0], color="steelblue", lw=2.5, alpha=1.0, label="Prior Ne (center)"),
        Line2D([0], [0], color="tomato",    lw=2.5, alpha=1.0, label="Posterior Ne (center)"),
        Line2D([0], [0], color="yellow",    lw=1.5, linestyle="--",
               label="Truth Ne (center)"),
    ]
    ax_edp.legend(handles=edp_handles, fontsize=7,
                  facecolor="#2b2b2b", labelcolor="lightgray",
                  loc="lower right", framealpha=0.8)
    ax_edp.set_ylim(bottom=0)

    if prior_mean_state is not None and post_mean_state is not None:
        _draw_param_boxes(
            ax_edp,
            [("Prior",     "steelblue", prior_mean_state[:, cen_5deg_idx]),
             ("Posterior", "tomato",    post_mean_state[:, cen_5deg_idx])],
            loc="upper left",
        )

    # ── Populate TEC and globe panels ─────────────────────────────────────────
    globe_handles: list = [
        Line2D([0], [0], color="lightgray", marker="o", ms=3,
               linestyle="none", label="1° truth grid"),
        Line2D([0], [0], color="steelblue", marker="s", ms=4,
               linestyle="none", label="5° model grid"),
    ]
    style_placed = False

    for arc, sl, col in zip(arc_truth_list, tec_slices, occ_colors):
        const = _resolve_conid(arc)
        label = _arc_label(arc)
        tang  = sl["tang_km"]
        ax    = tec_axes.get(const, tec_axes.get("G"))

        ax.plot(sl["tec_truth"], tang, color=col, linewidth=2.2,
                zorder=6, label=label)
        ax.plot(sl["prior_tec"], tang, color=col, linewidth=1.1,
                linestyle="--", alpha=0.6, zorder=4,
                label="Prior" if not style_placed else None)
        ax.plot(sl["post_tec"], tang, color=col, linewidth=1.4,
                linestyle=":", alpha=0.9, zorder=5,
                label="Posterior" if not style_placed else None)
        style_placed = True

        ax_globe.scatter(arc["tp_lons"], arc["tp_lats"],
                         s=4, color=col, alpha=0.6,
                         transform=ccrs.PlateCarree(), zorder=4)
        peak_idx = int(np.argmin(tang))
        ax_globe.scatter(arc["tp_lons"][peak_idx], arc["tp_lats"][peak_idx],
                         s=60, color=col, edgecolors="white", linewidth=0.5,
                         marker="o", transform=ccrs.PlateCarree(), zorder=5)
        globe_handles.append(
            Line2D([0], [0], color=col, marker="o", ms=5,
                   linestyle="none", label=label)
        )

    for ax in tec_axes.values():
        if ax.lines:
            _capped_legend(ax, fontsize=5, facecolor="#2b2b2b",
                           labelcolor="lightgray", loc="best", framealpha=0.7)

    _capped_legend(
        ax_globe,
        handles=globe_handles,
        fontsize=6, facecolor="#2b2b2b", labelcolor="lightgray",
        loc="lower left", framealpha=0.7, markerscale=1.2,
    )

    fig.suptitle(suptitle, color="white", fontsize=10, y=0.98)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")
def plot_kf_tec_edp(
    arc_truth_list: list[dict],
    kf_result: dict,
    truth_ne_1deg: np.ndarray,
    grid_lats_1deg: np.ndarray,
    grid_lons_1deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
    truth_time: pd.Timestamp,
    save_path: str,
) -> None:
    """TEC + globe + EDP summary figure for the gridded Ne KF."""
    _plot_tec_edp_figure(
        arc_truth_list,
        kf_result["tec_slices"],
        kf_result["prior_edp"],
        kf_result["posterior_edp"],
        truth_ne_1deg,
        grid_lats_1deg, grid_lons_1deg,
        grid_lats_5deg, grid_lons_5deg,
        alt_grid,
        suptitle=(
            f"Gridded Ne KF retrieval — truth ionosphere "
            f"{truth_time.strftime('%Y-%m-%d %H:%M')} UTC  "
            f"(+{TRUTH_HOUR_OFFSET} h, F10.7+{TRUTH_F107_DELTA:.0f})\n"
            f"Prior RMSE {kf_result['prior_rmse']:.2f} TECU  →  "
            f"Posterior RMSE {kf_result['post_rmse']:.2f} TECU"
        ),
        save_path=save_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §11  EDP spatial error — 5×2 orthographic
# ─────────────────────────────────────────────────────────────────────────────

def _meshgrid_interp(
    lats: np.ndarray,
    lons: np.ndarray,
    vals: np.ndarray,
    n_pts: int      = 120,
    margin_deg: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearly interpolate scattered lat/lon values onto a regular meshgrid.

    Returns
    -------
    LO, LA    : (n_pts, n_pts) regular lon/lat coordinate grids (degrees).
    vals_grid : (n_pts, n_pts) interpolated values; NaN outside convex hull.
    """
    from scipy.interpolate import griddata as _gd
    lat_min = lats.min() - margin_deg;  lat_max = lats.max() + margin_deg
    lon_min = lons.min() - margin_deg;  lon_max = lons.max() + margin_deg
    lats_r  = np.linspace(lat_min, lat_max, n_pts)
    lons_r  = np.linspace(lon_min, lon_max, n_pts)
    LO, LA  = np.meshgrid(lons_r, lats_r)
    fin     = np.isfinite(vals)
    if fin.sum() < 3:
        return LO, LA, np.full_like(LO, np.nan)
    vg = _gd(
        np.column_stack([lons[fin], lats[fin]]),
        vals[fin],
        (LO, LA),
        method="linear",
    )
    return LO, LA, vg


def _ax_pcolormesh(
    ax, LO, LA, vg,
    cmap: str, vmin: float, vmax: float,
    fig,
    label: str,
    grid_lons: np.ndarray,
    grid_lats: np.ndarray,
) -> None:
    """Shared helper: pcolormesh + grid-point dots + colorbar on a cartopy ax."""
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)
    pm = ax.pcolormesh(
        LO, LA, vg,
        cmap=cmap_obj, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(), zorder=3, shading="auto",
    )
    # Overlay original grid points as small markers for reference
    ax.scatter(
        grid_lons, grid_lats, c="k", s=5, alpha=0.6,
        transform=ccrs.PlateCarree(), zorder=5,
    )
    cb = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(label, fontsize=8)


def plot_edp_spatial_error(
    truth_ne_5deg: np.ndarray,
    posterior_ne_5deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
    error_altitudes: list = None,
    truth_time: pd.Timestamp = None,
    show_truth_col: bool = False,
    save_path: str = "./Figures/test_param_iono/edp_spatial_error.png",
) -> None:
    """
    5×N orthographic meshgrid plots of EDP retrieval error at five altitudes.

    Rows are ordered highest-to-lowest altitude (top → bottom).
    All panels in the same column share the same colorbar limits, computed
    from the 98th percentile across all displayed altitudes.

    Col 0  : [if show_truth_col] truth Ne  (m⁻³, viridis)
    Col +0 : absolute Ne error  |Ne_post − Ne_truth|  (m⁻³, magma)
    Col +1 : signed percent error  100·(Ne_post − Ne_truth)/Ne_truth  (%, RdBu_r)
    """
    if error_altitudes is None:
        error_altitudes = ERROR_ALTITUDES

    # Highest altitude at the top of the figure
    error_altitudes_sorted = sorted(error_altitudes, reverse=True)

    n_alt_plot = len(error_altitudes_sorted)
    n_cols     = 3 if show_truth_col else 2
    # cen_lat    = float(np.mean(grid_lats_5deg))
    # cen_lon    = float(np.mean(grid_lons_5deg))
    # proj       = ccrs.Orthographic(central_longitude=cen_lon,))
    #                                central_latitude=cen_lat)
    
    # ── Safe Extent and Projection Calculations ───────────────────────────
    # Fallback to global defaults if coordinates are missing or all-NaN
    if len(grid_lats_5deg) == 0 or np.isnan(grid_lats_5deg).all():
        ext_lat_min, ext_lat_max = -90.0, 90.0
        cen_lat = 0.0
    else:
        ext_lat_min = float(np.nanmin(grid_lats_5deg) - 3.0)
        ext_lat_max = float(np.nanmax(grid_lats_5deg) + 3.0)
        cen_lat = float(np.nanmean(grid_lats_5deg)) # Using nanmean just in case

    # Longitude is circular — arithmetic min/max/mean break for grids that
    # wrap around the pole (Fibonacci nodes near the pole have lons uniformly
    # spanning [0, 360°]). Use a unit-vector mean to get a sensible centre;
    # if the coverage is nearly global in longitude, fall back to a
    # centre-aligned +/-90° window that Orthographic can display.
    if len(grid_lons_5deg) == 0 or np.isnan(grid_lons_5deg).all():
        ext_lon_min, ext_lon_max = -180.0, 180.0
        cen_lon = 0.0
        _lon_span_wide = True
    else:
        _rad = np.radians(grid_lons_5deg[np.isfinite(grid_lons_5deg)])
        cen_lon = float(np.degrees(np.arctan2(
            np.mean(np.sin(_rad)), np.mean(np.cos(_rad))
        )))
        # Shift longitudes into (cen_lon - 180, cen_lon + 180] so they're
        # contiguous, then take min/max to get a wrap-safe extent.
        _shifted = ((grid_lons_5deg - cen_lon + 180.0) % 360.0) - 180.0
        _fin     = np.isfinite(_shifted)
        _lon_lo  = float(np.min(_shifted[_fin])) if _fin.any() else -3.0
        _lon_hi  = float(np.max(_shifted[_fin])) if _fin.any() else  3.0
        ext_lon_min = cen_lon + _lon_lo - 3.0
        ext_lon_max = cen_lon + _lon_hi + 3.0
        _lon_span_wide = (_lon_hi - _lon_lo) > 160.0

    # If lon coverage is nearly global (typical near-pole clusters), the
    # PlateCarree extent has corners beyond Orthographic's visible hemisphere
    # and set_extent returns NaN.  Collapse to a hemispheric window in that
    # case and rely on set_global() below at plot time as the final fallback.
    if _lon_span_wide:
        ext_lon_min = cen_lon - 90.0
        ext_lon_max = cen_lon + 90.0

    # Move projection instantiation here so it uses the safe cen_lon/cen_lat
    proj = ccrs.Orthographic(central_longitude=cen_lon, central_latitude=cen_lat)

    fig_w  = 6 * n_cols
    fig, axes = plt.subplots(
        n_alt_plot, n_cols,
        figsize=(fig_w, 4 * n_alt_plot),
        subplot_kw={"projection": proj},
    )
    if n_alt_plot == 1:
        axes = axes[np.newaxis, :]

    title_str = "EDP spatial error — posterior vs truth"
    if show_truth_col:
        title_str = "EDP — truth / absolute error / percent error"
    if truth_time is not None:
        title_str += f"\nTruth: {truth_time.strftime('%Y-%m-%d %H:%M')} UTC"
    fig.suptitle(title_str, fontsize=11, y=1.01)


    # Column metadata (order matches show_truth_col branches below)
    if show_truth_col:
        col_cmaps    = ["viridis", "magma",  "RdBu_r"]
        col_symm     = [False,     False,     True   ]
        col_cb_labels= ["Truth Ne  (m⁻³)", "|ΔNe|  (m⁻³)", "ΔNe  (%)"]
    else:
        col_cmaps    = ["magma",  "RdBu_r"]
        col_symm     = [False,     True   ]
        col_cb_labels= ["|ΔNe|  (m⁻³)", "ΔNe  (%)"]

    # First pass — collect all finite values per column to find shared limits
    col_vals_global = [[] for _ in range(n_cols)]
    for alt_km in error_altitudes_sorted:
        alt_idx = int(np.argmin(np.abs(alt_grid - alt_km)))
        ne_tru  = truth_ne_5deg[alt_idx, :]
        ne_pst  = posterior_ne_5deg[alt_idx, :]
        abs_err = np.abs(ne_pst - ne_tru)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct_err = 100.0 * (ne_pst - ne_tru) / np.where(ne_tru > 0, ne_tru, np.nan)
        row_vals = ([ne_tru] if show_truth_col else []) + [abs_err, pct_err]
        for c, v in enumerate(row_vals):
            fin = np.isfinite(v)
            if fin.any():
                col_vals_global[c].append(v[fin])

    col_limits = []
    for c in range(n_cols):
        if col_vals_global[c]:
            all_v = np.concatenate(col_vals_global[c])
            v_abs = float(np.nanpercentile(np.abs(all_v), 98)) or 1.0
            col_limits.append((-v_abs, v_abs) if col_symm[c] else (0.0, v_abs))
        else:
            col_limits.append((0.0, 1.0))

    # Second pass — plot with shared limits
    for row_i, alt_km in enumerate(error_altitudes_sorted):
        alt_idx = int(np.argmin(np.abs(alt_grid - alt_km)))
        ne_tru  = truth_ne_5deg[alt_idx, :]
        ne_pst  = posterior_ne_5deg[alt_idx, :]
        abs_err = np.abs(ne_pst - ne_tru)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct_err = 100.0 * (ne_pst - ne_tru) / np.where(ne_tru > 0, ne_tru, np.nan)
        row_vals = ([ne_tru] if show_truth_col else []) + [abs_err, pct_err]

        for col_i, vals in enumerate(row_vals):
            ax      = axes[row_i, col_i]
            cmap    = col_cmaps[col_i]
            vmin, vmax = col_limits[col_i]
            cb_label = col_cb_labels[col_i]
            ax_title = f"~{alt_grid[alt_idx]:.0f} km — {cb_label}"

            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555")
            ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#444")
            ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4)
            ax.set_facecolor("#d8d8d8")
            try:
                ax.set_extent(
                    [ext_lon_min, ext_lon_max, ext_lat_min, ext_lat_max],
                    crs=ccrs.PlateCarree(),
                )
            except (ValueError, Exception):
                # Corner of the requested PlateCarree box projects to NaN in
                # Orthographic (extent crosses the visible hemisphere edge —
                # typical for pole-wrapping clusters).  Fall back to the
                # full projection view.
                ax.set_global()

            if not np.isfinite(vals).any():
                ax.set_title(f"{ax_title} — no data", fontsize=8)
                continue

            LO, LA, vg = _meshgrid_interp(grid_lats_5deg, grid_lons_5deg, vals)
            _ax_pcolormesh(ax, LO, LA, vg, cmap, vmin, vmax,
                           fig, cb_label, grid_lons_5deg, grid_lats_5deg)
            ax.set_title(ax_title, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §12a  Per-site & regional EDP-RMSE across time windows
#
# Both plots below aggregate the per-window ``filter_result`` dicts returned by
# ``_run_enkf_retrieval_experiment`` — one row per window — into cross-window
# diagnostics that summarise how well the filters recovered the synthetic
# truth EDP over the assimilation day.
#
# The KF and EKF assimilate on the 5-deg model grid; the truth ionosphere is
# evaluated on both the 1-deg (site comparison) and 5-deg (regional aggregation)
# Fibonacci grids.  Both grids are rebuilt per window from the arc tangent
# tracks, so the "nearest node" lookup runs independently for every window.
# ─────────────────────────────────────────────────────────────────────────────

def _rmse_1d(est: np.ndarray, ref: np.ndarray,
             mask: np.ndarray | None = None) -> float:
    """RMSE over a single 1-D vector, ignoring NaNs.

    mask : optional boolean array selecting which elements to include.
    """
    est  = np.asarray(est,  dtype=float)
    ref  = np.asarray(ref,  dtype=float)
    if mask is not None:
        est = est[mask]
        ref = ref[mask]
    diff = est - ref
    if not np.isfinite(diff).any():
        return float("nan")
    return float(np.sqrt(np.nanmean(diff ** 2)))


def _collect_window_records(
    all_results: dict[str, dict],
) -> list[dict]:
    """
    Extract the successful, filter-produced per-window records from
    ``all_results`` (as built by ``main()``), sorted chronologically.

    Returned entries carry the filter EDPs, truth EDPs, both grids, and the
    HHMM label used for axis ticks.  Windows with no filter_result (errored
    or filters disabled) are skipped.
    """
    recs: list[dict] = []
    for wkey, r in all_results.items():
        if "error" in r:
            continue
        fr = r.get("filter_result") or {}
        if not fr:
            continue
        kf_r  = fr.get("kf_result")
        ekf_r = fr.get("ekf_param")
        if kf_r is None and ekf_r is None:
            continue
        if fr.get("truth_ne_5deg") is None:
            continue
        recs.append(dict(
            window_key    = wkey,
            hhmm          = r.get("hhmm", wkey[-4:]),
            time_dt       = r.get("time_dt"),
            grid_lats_1deg= fr["grid_lats_1deg"],
            grid_lons_1deg= fr["grid_lons_1deg"],
            grid_lats_5deg= fr["grid_lats_5deg"],
            grid_lons_5deg= fr["grid_lons_5deg"],
            truth_ne_1deg = fr["truth_ne_1deg"],
            truth_ne_5deg = fr["truth_ne_5deg"],
            kf_result     = kf_r,
            ekf_param     = ekf_r,
        ))
    # Chronological sort — window_key is "YYYY-MM-DD_HHMM", so lexical == time.
    recs.sort(key=lambda d: d["window_key"])
    return recs


def plot_edp_site_rmse_across_windows(
    all_results: dict[str, dict],
    alt_grid: np.ndarray,
    sites: tuple[str, ...] = ISR_SITES,
    save_path: str = "./Figures/test_param_iono/edp_site_rmse_across_windows.png",
) -> None:
    """
    Per-site prior/posterior EDP RMSE (m⁻³) as a function of assimilation
    time window, for KF and EKF_Param.

    Layout — one row per site (top: highest-latitude site first)
    ─────────────────────────────────────────────────────────────
    Row s  : site s
             x = window HHMM (chronological)
             y = EDP RMSE (m⁻³), altitude-averaged at the nearest 1-deg truth
                 grid point to the site's (lat, lon)
             Lines:
               • Prior           (grey, dashed)
               • KF posterior    (steelblue, solid)
               • EKF_P posterior (seagreen, solid)

    The prior EDP is taken from the KF's 5-deg model baseline (which is the
    same IRI mean_5deg used by EKF_Param, so a single prior line suffices).
    The truth EDP is looked up on the 1-deg truth grid at the point nearest
    the site coordinates — matching the "nearest grid point in
    truth_state_1deg / truth_ne_1deg" spec.
    """
    recs = _collect_window_records(all_results)
    if not recs:
        print(f"  [plot_edp_site_rmse_across_windows] no eligible windows — "
              f"skipping plot.")
        return

    n_sites = len(sites)
    n_win   = len(recs)
    xs      = np.arange(n_win, dtype=float)
    hhmm    = [r["hhmm"] for r in recs]

    # Pre-allocate per-site × per-window RMSE arrays for each series.
    prior_rmse    = np.full((n_sites, n_win), np.nan)
    kf_post_rmse  = np.full((n_sites, n_win), np.nan)
    ekf_post_rmse = np.full((n_sites, n_win), np.nan)

    for s_i, site in enumerate(sites):
        inst = INSTRUMENTS[site]
        slat, slon = float(inst["lat"]), float(inst["lon"])
        for w_i, rec in enumerate(recs):
            g1lat = rec["grid_lats_1deg"]
            g1lon = rec["grid_lons_1deg"]
            g5lat = rec["grid_lats_5deg"]
            g5lon = rec["grid_lons_5deg"]
            # Nearest 1-deg truth-grid point for the site's truth profile.
            idx1  = int(np.argmin(_haversine_km(slat, slon, g1lat, g1lon)))
            # Nearest 5-deg model-grid point for the filter posterior column.
            idx5  = int(np.argmin(_haversine_km(slat, slon, g5lat, g5lon)))
            truth_prof = rec["truth_ne_1deg"][:, idx1]

            kf_r  = rec["kf_result"]
            ekf_r = rec["ekf_param"]

            # F2 peak from the shared IRI prior; restrict RMSE to below-peak altitudes.
            _prior_src = (kf_r or ekf_r or {}).get("prior_ne_5deg")
            if _prior_src is not None:
                _, _hmF2 = extract_robust_f2_peak(np.asarray(_prior_src)[:, idx5], alt_grid)
            else:
                _hmF2 = np.nan
            _below = (alt_grid <= _hmF2) if np.isfinite(_hmF2) else np.ones(len(alt_grid), dtype=bool)

            if kf_r is not None:
                prior_rmse[s_i, w_i]   = _rmse_1d(kf_r["prior_ne_5deg"][:, idx5],
                                                   truth_prof, _below)
                kf_post_rmse[s_i, w_i] = _rmse_1d(kf_r["posterior_ne_5deg"][:, idx5],
                                                   truth_prof, _below)
            elif ekf_r is not None:
                prior_rmse[s_i, w_i]   = _rmse_1d(ekf_r["prior_ne_5deg"][:, idx5],
                                                   truth_prof, _below)
            if ekf_r is not None:
                ekf_post_rmse[s_i, w_i] = _rmse_1d(ekf_r["posterior_ne_5deg"][:, idx5],
                                                    truth_prof, _below)

    fig, axes = plt.subplots(
        n_sites, 1,
        figsize=(max(9.0, 0.55 * n_win + 4.0), 3.4 * n_sites),
        facecolor="#1e1e1e",
        sharex=True,
    )
    if n_sites == 1:
        axes = np.array([axes])
    fig.patch.set_facecolor("#1e1e1e")

    for s_i, site in enumerate(sites):
        inst  = INSTRUMENTS[site]
        label = inst.get("label", site)
        ax    = axes[s_i]
        ax.set_facecolor("#2b2b2b")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")

        ax.plot(xs, prior_rmse[s_i],    color="gray",     lw=1.4,
                linestyle="--", marker="o", markersize=4,
                label="Prior (IRI baseline)")
        ax.plot(xs, kf_post_rmse[s_i],  color="steelblue", lw=1.6,
                marker="s", markersize=4, label="KF posterior")
        ax.plot(xs, ekf_post_rmse[s_i], color="seagreen",  lw=1.6,
                marker="^", markersize=4, label="EKF_P posterior")

        ax.set_title(
            f"{site} — {label}  "
            f"({slat_str(inst)}, altitude-averaged EDP RMSE)",
            color="white", fontsize=9, fontweight="bold",
        )
        ax.set_ylabel("EDP RMSE  (m⁻³)", color="lightgray", fontsize=8)
        ax.tick_params(colors="lightgray", labelsize=7)
        ax.grid(True, axis="y", lw=0.3, alpha=0.35, color="#888")
        ax.legend(fontsize=7, facecolor="#2b2b2b", labelcolor="lightgray",
                  loc="best", framealpha=0.8)

    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(hhmm, rotation=45, ha="right",
                             fontsize=7, color="lightgray")
    axes[-1].set_xlabel("Window (UTC, HHMM)", color="lightgray", fontsize=8)

    fig.suptitle(
        f"Per-site EDP RMSE across windows — {YYYY}.{DOY:03d}  "
        f"(prior vs posterior vs 1-deg truth)",
        color="white", fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


def slat_str(inst: dict) -> str:
    """Compact 'lat°N, lon°E' string for site titles."""
    lat, lon = float(inst["lat"]), float(inst["lon"])
    lat_hem  = "N" if lat >= 0 else "S"
    lon_hem  = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{lat_hem}, {abs(lon):.2f}°{lon_hem}"


def plot_edp_regional_rmse_across_windows(
    all_results: dict[str, dict],
    alt_grid: np.ndarray,
    save_path: str = "./Figures/test_param_iono/edp_regional_rmse_across_windows.png",
) -> None:
    """
    Regional (whole-grid) EDP-RMSE summary across the day.

    Layout — 1×2
    ────────────────────────────────────────────────────────────
    Left  : RMSE vs altitude — for each altitude bin, pooled RMSE over
            all 1-deg truth grid points AND all windows.
            Three thick lines: prior / KF post / EKF_P post.
            Thin per-window lines behind (light alpha, colour-matched) show
            window-to-window spread.

    Right : Regional RMSE vs time window — one scalar per window (RMSE
            pooled over all altitudes and all 1-deg grid points).  Three
            lines with markers, chronological x-axis (HHMM ticks).

    Aggregation notes
    -----------------
    Filter posteriors live on the 5-deg model grid.  For each window we
    nearest-neighbour interpolate every 5-deg posterior column onto the
    1-deg truth grid (each 1-deg node → its nearest 5-deg node), then
    subtract the 1-deg truth EDP.  This gives whole-region coverage that
    matches the truth grid rather than the coarser model grid.
    """
    recs = _collect_window_records(all_results)
    if not recs:
        print(f"  [plot_edp_regional_rmse_across_windows] no eligible windows — "
              f"skipping plot.")
        return

    n_alt = len(alt_grid)
    n_win = len(recs)
    hhmm  = [r["hhmm"] for r in recs]

    # Per-window, per-altitude MSE + count for pooled statistics.
    prior_sqerr_alt   = np.zeros((n_win, n_alt))
    kf_sqerr_alt      = np.zeros((n_win, n_alt))
    ekf_sqerr_alt     = np.zeros((n_win, n_alt))
    prior_count_alt   = np.zeros((n_win, n_alt), dtype=int)
    kf_count_alt      = np.zeros((n_win, n_alt), dtype=int)
    ekf_count_alt     = np.zeros((n_win, n_alt), dtype=int)

    # Per-window scalar RMSE (all altitudes × all 1-deg nodes pooled).
    prior_scalar = np.full(n_win, np.nan)
    kf_scalar    = np.full(n_win, np.nan)
    ekf_scalar   = np.full(n_win, np.nan)

    for w_i, rec in enumerate(recs):
        g1lat = rec["grid_lats_1deg"]
        g1lon = rec["grid_lons_1deg"]
        g5lat = rec["grid_lats_5deg"]
        g5lon = rec["grid_lons_5deg"]
        truth = rec["truth_ne_1deg"]                        # (n_alt, n_geo_1deg)

        # Nearest-5-deg-node lookup for every 1-deg node.
        tree5   = cKDTree(np.column_stack([g5lat, g5lon]))
        _, nn5  = tree5.query(np.column_stack([g1lat, g1lon]))

        kf_r  = rec["kf_result"]
        ekf_r = rec["ekf_param"]

        # Prior is shared (same IRI baseline).  Prefer KF's, fall back to EKF's.
        prior_edp_5 = None
        if kf_r is not None:
            prior_edp_5 = kf_r["prior_ne_5deg"]
        elif ekf_r is not None:
            prior_edp_5 = ekf_r["prior_ne_5deg"]

        if prior_edp_5 is not None:
            prior_at_1 = prior_edp_5[:, nn5]                # (n_alt, n_geo_1deg)
            # Per-column F2 peak from prior; restrict RMSE to below-peak altitudes.
            _n1 = prior_at_1.shape[1]
            _hmF2_cols = np.array([
                extract_robust_f2_peak(prior_at_1[:, j], alt_grid)[1]
                for j in range(_n1)
            ])
            _peak_lim = np.where(np.isfinite(_hmF2_cols), _hmF2_cols, np.inf)
            # below_mask[k, j] = True when alt_grid[k] is below prior F2 peak at node j
            below_mask = alt_grid[:, None] <= _peak_lim[None, :]   # (n_alt, n_1deg)

            diff = prior_at_1 - truth
            fin  = np.isfinite(diff) & below_mask
            prior_sqerr_alt[w_i] = np.where(fin, diff ** 2, 0.0).sum(axis=1)
            prior_count_alt[w_i] = fin.sum(axis=1)
            if fin.any():
                prior_scalar[w_i] = float(np.sqrt(
                    np.sum(diff ** 2 * fin) / max(int(fin.sum()), 1)))
        else:
            below_mask = np.ones((len(alt_grid), len(nn5)), dtype=bool)

        if kf_r is not None:
            kf_at_1  = kf_r["posterior_ne_5deg"][:, nn5]
            diff     = kf_at_1 - truth
            fin      = np.isfinite(diff) & below_mask
            kf_sqerr_alt[w_i] = np.where(fin, diff ** 2, 0.0).sum(axis=1)
            kf_count_alt[w_i] = fin.sum(axis=1)
            if fin.any():
                kf_scalar[w_i] = float(np.sqrt(
                    np.sum(diff ** 2 * fin) / max(int(fin.sum()), 1)))

        if ekf_r is not None:
            ekf_at_1 = ekf_r["posterior_ne_5deg"][:, nn5]
            diff     = ekf_at_1 - truth
            fin      = np.isfinite(diff) & below_mask
            ekf_sqerr_alt[w_i] = np.where(fin, diff ** 2, 0.0).sum(axis=1)
            ekf_count_alt[w_i] = fin.sum(axis=1)
            if fin.any():
                ekf_scalar[w_i] = float(np.sqrt(
                    np.sum(diff ** 2 * fin) / max(int(fin.sum()), 1)))

    # Pooled RMSE curves (all windows combined, per altitude).
    def _pooled_alt(sqerr: np.ndarray, cnt: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            tot_sq  = sqerr.sum(axis=0)
            tot_cnt = cnt.sum(axis=0)
            return np.where(tot_cnt > 0, np.sqrt(tot_sq / np.maximum(tot_cnt, 1)),
                            np.nan)

    prior_rmse_alt = _pooled_alt(prior_sqerr_alt, prior_count_alt)
    kf_rmse_alt    = _pooled_alt(kf_sqerr_alt,    kf_count_alt)
    ekf_rmse_alt   = _pooled_alt(ekf_sqerr_alt,   ekf_count_alt)

    # Per-window RMSE curves vs altitude (thin, for spread).
    def _per_win_alt(sqerr: np.ndarray, cnt: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, np.sqrt(sqerr / np.maximum(cnt, 1)), np.nan)

    prior_alt_pw = _per_win_alt(prior_sqerr_alt, prior_count_alt)
    kf_alt_pw    = _per_win_alt(kf_sqerr_alt,    kf_count_alt)
    ekf_alt_pw   = _per_win_alt(ekf_sqerr_alt,   ekf_count_alt)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, (ax_alt, ax_time) = plt.subplots(
        1, 2, figsize=(max(15.0, 0.55 * n_win + 8.0), 6.2),
        facecolor="#1e1e1e",
        gridspec_kw={"width_ratios": [1.0, 1.4]},
    )
    fig.patch.set_facecolor("#1e1e1e")
    for ax in (ax_alt, ax_time):
        ax.set_facecolor("#2b2b2b")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        ax.tick_params(colors="lightgray", labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.35, color="#888")

    # Left panel — RMSE vs altitude
    for w_i in range(n_win):
        ax_alt.plot(prior_alt_pw[w_i], alt_grid, color="gray",
                    lw=0.5, alpha=0.25)
        ax_alt.plot(kf_alt_pw[w_i],    alt_grid, color="steelblue",
                    lw=0.5, alpha=0.25)
        ax_alt.plot(ekf_alt_pw[w_i],   alt_grid, color="seagreen",
                    lw=0.5, alpha=0.25)

    ax_alt.plot(prior_rmse_alt, alt_grid, color="gray",      lw=2.2,
                linestyle="--", label="Prior (pooled)")
    ax_alt.plot(kf_rmse_alt,    alt_grid, color="steelblue", lw=2.2,
                label="KF post (pooled)")
    ax_alt.plot(ekf_rmse_alt,   alt_grid, color="seagreen",  lw=2.2,
                label="EKF_P post (pooled)")

    ax_alt.set_title(
        f"Regional EDP RMSE vs altitude  ({n_win} windows pooled, "
        f"1-deg truth grid)",
        color="white", fontsize=9, fontweight="bold",
    )
    ax_alt.set_xlabel("EDP RMSE  (m⁻³)", color="lightgray", fontsize=8)
    ax_alt.set_ylabel("Altitude  (km)",  color="lightgray", fontsize=8)
    ax_alt.set_ylim(alt_grid.min(), alt_grid.max())
    ax_alt.legend(fontsize=7, facecolor="#2b2b2b", labelcolor="lightgray",
                  loc="best", framealpha=0.8)

    # Right panel — RMSE vs time window
    xs = np.arange(n_win, dtype=float)
    ax_time.plot(xs, prior_scalar, color="gray",      lw=1.6,
                 linestyle="--", marker="o", markersize=4,
                 label="Prior")
    ax_time.plot(xs, kf_scalar,    color="steelblue", lw=1.8,
                 marker="s", markersize=4, label="KF posterior")
    ax_time.plot(xs, ekf_scalar,   color="seagreen",  lw=1.8,
                 marker="^", markersize=4, label="EKF_P posterior")

    ax_time.set_title(
        "Regional EDP RMSE vs time window "
        "(pooled over altitude × 1-deg grid)",
        color="white", fontsize=9, fontweight="bold",
    )
    ax_time.set_xlabel("Window (UTC, HHMM)", color="lightgray", fontsize=8)
    ax_time.set_ylabel("EDP RMSE  (m⁻³)",    color="lightgray", fontsize=8)
    ax_time.set_xticks(xs)
    ax_time.set_xticklabels(hhmm, rotation=45, ha="right",
                            fontsize=7, color="lightgray")
    ax_time.legend(fontsize=7, facecolor="#2b2b2b", labelcolor="lightgray",
                   loc="best", framealpha=0.8)

    fig.suptitle(
        f"Regional EDP RMSE summary — {YYYY}.{DOY:03d}",
        color="white", fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §12  Parameter spatial error — 8×2 orthographic
# ─────────────────────────────────────────────────────────────────────────────

def plot_parameter_spatial_error(
    truth_mean_5deg: np.ndarray,
    posterior_mean_5deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    truth_time: pd.Timestamp = None,
    save_path: str = "./Figures/test_param_iono/param_spatial_error.png",
) -> None:
    """
    8×2 orthographic meshgrid plots of IRI parameter retrieval.

    Row k  : PARAM_NAMES[k]
    Col 0  : posterior parameter value (pcolormesh, viridis)
    Col 1  : signed error  posterior_param − truth_param  (pcolormesh, RdBu_r)

    Values are linearly interpolated onto a regular lat/lon meshgrid
    (scipy.interpolate.griddata) and displayed with pcolormesh.  The original
    5-deg Fibonacci grid points are overlaid as small black dots for reference.
    """
    cen_lat = float(np.nanmean(grid_lats_5deg))
    # Wrap-safe longitude centre (unit-vector mean) so pole-wrapping clusters
    # don't collapse to a meaningless arithmetic mean.
    _rad    = np.radians(grid_lons_5deg[np.isfinite(grid_lons_5deg)])
    cen_lon = float(np.degrees(np.arctan2(
        np.mean(np.sin(_rad)), np.mean(np.cos(_rad))
    ))) if _rad.size else 0.0
    proj    = ccrs.Orthographic(central_longitude=cen_lon, central_latitude=cen_lat)

    fig, axes = plt.subplots(
        N_STATE, 2,
        figsize=(12, 4 * N_STATE),
        subplot_kw={"projection": proj},
    )

    title_str = "Parameter retrieval — posterior value and error vs truth"
    if truth_time is not None:
        title_str += f"\nTruth: {truth_time.strftime('%Y-%m-%d %H:%M')} UTC"
    fig.suptitle(title_str, fontsize=11, y=1.005)

    ext_lat_min = float(np.nanmin(grid_lats_5deg)) - 3.0
    ext_lat_max = float(np.nanmax(grid_lats_5deg)) + 3.0
    # Longitude extent using cen_lon-relative shift so a wrapping grid stays
    # contiguous; collapse to a +/-90° hemispheric window when coverage is
    # nearly global to keep Orthographic corners inside the visible disc.
    _shifted = ((grid_lons_5deg - cen_lon + 180.0) % 360.0) - 180.0
    _fin     = np.isfinite(_shifted)
    _lon_lo  = float(np.min(_shifted[_fin])) if _fin.any() else -3.0
    _lon_hi  = float(np.max(_shifted[_fin])) if _fin.any() else  3.0
    if (_lon_hi - _lon_lo) > 160.0:
        ext_lon_min = cen_lon - 90.0
        ext_lon_max = cen_lon + 90.0
    else:
        ext_lon_min = cen_lon + _lon_lo - 3.0
        ext_lon_max = cen_lon + _lon_hi + 3.0

    for k, pname in enumerate(PARAM_NAMES):
        pst_vals = posterior_mean_5deg[k, :]    # (n_geo,)
        tru_vals = truth_mean_5deg[k, :]        # (n_geo,)
        err_vals = pst_vals - tru_vals

        for col_i, (vals, label, cmap, symm) in enumerate([
            (pst_vals, f"{pname}  posterior",          "viridis", False),
            (err_vals, f"{pname}  error (post−truth)", "RdBu_r",  True),
        ]):
            ax = axes[k, col_i]
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555")
            ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#444")
            ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4)
            ax.set_facecolor("#d8d8d8")
            try:
                ax.set_extent(
                    [ext_lon_min, ext_lon_max, ext_lat_min, ext_lat_max],
                    crs=ccrs.PlateCarree(),
                )
            except (ValueError, Exception):
                ax.set_global()

            finite_mask = np.isfinite(vals)
            if not finite_mask.any():
                ax.set_title(f"{label} — no data", fontsize=8)
                continue

            v_abs = float(np.nanpercentile(np.abs(vals[finite_mask]), 98)) or 1e-6
            if symm:
                vmin, vmax = -v_abs, v_abs
            else:
                vmin = float(np.nanmin(vals[finite_mask]))
                vmax = float(np.nanmax(vals[finite_mask]))
                if np.isclose(vmin, vmax):
                    vmax = vmin + 1e-6

            LO, LA, vg = _meshgrid_interp(grid_lats_5deg, grid_lons_5deg, vals)
            _ax_pcolormesh(ax, LO, LA, vg, cmap, vmin, vmax,
                           fig, label, grid_lons_5deg, grid_lats_5deg)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
def run_gridded_ne_kf(
    arc_truth_list: list[dict],
    model_state_prior: "IonosphericState",
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
    sigma_obs: float     = ENKF_SIGMA_OBS,
    max_update_rays: int = ENKF_MAX_UPDATE_RAYS,
) -> dict:
    """
    Standard linear Kalman Filter assimilation directly in Ne-space.

    State vector  : x  (n_alt * n_geo,)   — Ne at every altitude × grid point.
                    Layout: x[k * n_geo + g] = Ne(alt_k, gp_g)  [m⁻³]
    Forward model : H  (n_obs, n_alt * n_geo) — trapezoidal arc integration
                    H[i, k*n_geo+g] = phi_k(ray_i) × w_ig  [TECU / m⁻³]
    Update        : P_xy  S⁻¹  (y − H x̄)   — single-step linear update.

    Expects ``model_state_prior`` built with ``build_truth_state(mean_5deg)``:
      member 0 = IRI baseline at current conditions (no perturbation)
      members 1…N_STATE = single-parameter ±σ shifts

    Member 0's Ne profile is used as the prior mean x̄.  The full ensemble
    spread provides the background-error covariance.

    Returns
    -------
    dict with the same keys as ``run_gridded_ne_kf`` for direct comparison.
    """
    n_occ    = len(arc_truth_list)
    n_geo    = len(grid_lats_5deg)
    n_alt    = len(alt_grid)
    n_state  = n_alt * n_geo
    n_members = model_state_prior.n_members

    print(f"  [KF] State: {n_alt} alt × {n_geo} geo = {n_state} elements | "
          f"{n_members}-member prior ensemble (member 0 = IRI baseline)")

    # ── 1. Convert parametric ensemble → Ne-space ─────────────────────────────
    ne_ens = _parametric_to_edp_ensemble(
        model_state_prior, model_state_prior.ensemble, alt_grid,
    )  # (n_alt, n_geo, n_members)
    # Flatten: X[k * n_geo + g, m] = ne_ens[k, g, m]   (C-order)
    X = ne_ens.reshape(n_state, n_members)

    # Prior mean = member 0, the unperturbed IRI baseline at current conditions
    x_bar = X[:, 0].copy()                            # (n_state,)
    X_c   = X - x_bar[:, np.newaxis]                  # centered   (n_state, n_members)

    # ── 2. Decimate update rays (same stride as EnKF) ─────────────────────────
    rep_rays:          list  = []
    rep_tp_lats_list:  list  = []
    rep_tp_lons_list:  list  = []
    rep_tec_obs_list:  list  = []
    arc_update_counts: list  = []
    ray_counts:        list  = []
    arc_all_tec:       list  = []
    per_arc_tp_lats:   list  = []
    per_arc_tp_lons:   list  = []

    for arc in arc_truth_list:
        rays    = arc["rays"]
        tec     = arc["tec_truth"]
        tp_lats = arc["tp_lats"]
        tp_lons = arc["tp_lons"]
        n_s     = len(rays)

        per_arc_tp_lats.append(tp_lats)
        per_arc_tp_lons.append(tp_lons)
        ray_counts.append(n_s)
        arc_all_tec.append(tec)

        if n_s > max_update_rays:
            stride = int(np.ceil(n_s / max_update_rays))
            chosen = list(range(0, n_s, stride))
            if n_s - 1 not in chosen:
                chosen.append(n_s - 1)
        else:
            chosen = list(range(n_s))

        for idx in chosen:
            rep_rays.append(rays[idx])
            rep_tp_lats_list.append(float(tp_lats[idx]))
            rep_tp_lons_list.append(float(tp_lons[idx]))
            rep_tec_obs_list.append(float(tec[idx]))
        arc_update_counts.append(len(chosen))

    y_obs_arc = np.array(rep_tec_obs_list)
    y_obs_all = np.concatenate(arc_all_tec)
    n_obs     = len(rep_rays)
    print(f"  [KF] {n_obs} update rays across {n_occ} arcs")

    if n_obs >= n_members:
        print(f"  [KF] WARNING: n_obs ({n_obs}) >= n_members ({n_members}); "
              f"S = Y_c Y_c^T + R may be rank-deficient in ensemble block.")

    # ── 3. Build H matrix (n_obs, n_state) ───────────────────────────────────
    # H[i, k*n_geo+g] = phi_k(ray_i) × w_ig
    # Each row = np.outer(phi_i, w_i).ravel()  (shape: n_alt × n_geo → n_state)
    print(f"  [KF] Building H ({n_obs} × {n_state}) …")
    H = np.zeros((n_obs, n_state), dtype=float)
    for i in range(n_obs):
        phi_i = _precompute_ray_phi(rep_rays[i], alt_grid)      # (n_alt,)
        w_i   = _idw_weights(rep_tp_lats_list[i], rep_tp_lons_list[i],
                             grid_lats_5deg, grid_lons_5deg)     # (n_geo,)
        H[i]  = np.outer(phi_i, w_i).ravel()                    # (n_state,)

    # ── 4. Prior TEC and innovations ──────────────────────────────────────────
    y_prior_arc = H @ x_bar
    prior_inno  = y_obs_arc - y_prior_arc
    print(f"  [KF] Prior innovations  "
          f"mean={prior_inno.mean():.2f}  std={prior_inno.std():.2f}  "
          f"max_abs={np.abs(prior_inno).max():.2f} TECU")

    # ── 5. Ensemble-based KF update ───────────────────────────────────────────
    # Avoids forming P (n_state × n_state) explicitly.
    # P_yy = Y_c Y_c^T / (N−1),   P_xy = X_c Y_c^T / (N−1)
    # S    = P_yy + R
    # K    = P_xy S⁻¹
    # x_post = x_bar + K (y − H x_bar)
    Y_c   = H @ X_c                                   # (n_obs, n_members)
    nm1   = max(n_members - 1, 1)
    P_yy  = Y_c @ Y_c.T / nm1                         # (n_obs, n_obs)
    P_xy  = X_c @ Y_c.T / nm1                         # (n_state, n_obs)
    R_mat = (sigma_obs ** 2) * np.eye(n_obs)
    S     = P_yy + R_mat                              # (n_obs, n_obs)
    # Solve S x = P_xy^T, then K = result^T
    K     = np.linalg.solve(S.T, P_xy.T).T            # (n_state, n_obs)

    x_post  = x_bar + K @ prior_inno                  # (n_state,)
    ne_post = np.maximum(x_post.reshape(n_alt, n_geo), 0.0)

    # Linearized posterior ensemble anomaly for covariance visualization:
    # X_post_c ≈ (I − K H) X_c  (ignores obs-noise contribution K R K^T)
    X_post_c = X_c - K @ (H @ X_c)                  # (n_state, n_members)

    post_inno_arc = y_obs_arc - H @ x_post
    print(f"  [KF] Posterior innovations  "
          f"mean={post_inno_arc.mean():.2f}  std={post_inno_arc.std():.2f}  "
          f"max_abs={np.abs(post_inno_arc).max():.2f} TECU")

    # ── 6. Build H_all for all sample rays ───────────────────────────────────
    all_tp_lats_flat = np.concatenate([a.tolist() for a in per_arc_tp_lats])
    all_tp_lons_flat = np.concatenate([a.tolist() for a in per_arc_tp_lons])
    all_sample_rays  = [r for arc in arc_truth_list for r in arc["rays"]]
    n_all = len(all_sample_rays)

    print(f"  [KF] Building H_all ({n_all} × {n_state}) for RMSE …")
    H_all = np.zeros((n_all, n_state), dtype=float)
    for i in range(n_all):
        phi_i    = _precompute_ray_phi(all_sample_rays[i], alt_grid)
        w_i      = _idw_weights(float(all_tp_lats_flat[i]),
                               float(all_tp_lons_flat[i]),
                               grid_lats_5deg, grid_lons_5deg)
        H_all[i] = np.outer(phi_i, w_i).ravel()

    y_prior_all = H_all @ x_bar
    y_post_all  = H_all @ x_post

    prior_rmse = float(np.sqrt(np.nanmean((y_obs_all - y_prior_all) ** 2)))
    post_rmse  = float(np.sqrt(np.nanmean((y_obs_all - y_post_all) ** 2)))
    print(f"  [KF] Prior RMSE {prior_rmse:.3f} TECU  →  "
          f"Post RMSE {post_rmse:.3f} TECU")

    # ── 7. Per-arc stats and tec_slices ──────────────────────────────────────
    tec_slices:     list[dict] = []
    all_prior_resid = y_obs_all - y_prior_all
    all_post_resid  = y_obs_all - y_post_all

    arc_prior_mean_l, arc_post_mean_l = [], []
    arc_prior_rmse_l, arc_post_rmse_l = [], []
    arc_lats_l, arc_lons_l, arc_lbl_l = [], [], []

    soff = 0
    for i, arc in enumerate(arc_truth_list):
        n_s = ray_counts[i]
        sl  = slice(soff, soff + n_s)
        tec_slices.append(dict(
            tec_truth = arc["tec_truth"],
            prior_tec = y_prior_all[sl].copy(),
            post_tec  = y_post_all[sl].copy(),
            tang_km   = arc["tang_km"],
        ))
        rp = all_prior_resid[sl]
        ra = all_post_resid[sl]
        arc_prior_mean_l.append(float(np.nanmean(rp)))
        arc_post_mean_l.append(float(np.nanmean(ra)))
        arc_prior_rmse_l.append(float(np.sqrt(np.nanmean(rp ** 2))))
        arc_post_rmse_l.append(float(np.sqrt(np.nanmean(ra ** 2))))
        arc_lats_l.append(float(arc["tp_lats"].mean()))
        arc_lons_l.append(float(arc["tp_lons"].mean()))
        arc_lbl_l.append(f"{arc['conid']}{arc['prn_id']}")
        soff += n_s

    return dict(
        prior_ne_5deg       = x_bar.reshape(n_alt, n_geo),
        posterior_ne_5deg   = ne_post,
        prior_edp           = x_bar.reshape(n_alt, n_geo),
        posterior_edp       = ne_post,
        tec_slices          = tec_slices,
        prior_rmse          = prior_rmse,
        post_rmse           = post_rmse,
        all_prior_resid     = all_prior_resid,
        all_post_resid      = all_post_resid,
        arc_prior_mean      = np.array(arc_prior_mean_l),
        arc_post_mean       = np.array(arc_post_mean_l),
        arc_prior_rmse      = np.array(arc_prior_rmse_l),
        arc_post_rmse       = np.array(arc_post_rmse_l),
        arc_lats            = np.array(arc_lats_l),
        arc_lons            = np.array(arc_lons_l),
        arc_labels          = arc_lbl_l,
        grid_lats           = grid_lats_5deg,
        grid_lons           = grid_lons_5deg,
        mda_arc_means_list  = None,
        mda_flat_list       = None,
        prior_Xc            = X_c,       # (n_state, n_members) centered prior ensemble
        post_Xc             = X_post_c,  # (n_state, n_members) linearized posterior ensemble
    )


# ─────────────────────────────────────────────────────────────────────────────
# §15b  Covariance structure panels — KF and EnKF
# ─────────────────────────────────────────────────────────────────────────────

def plot_kf_covariance_panels(
    prior_Xc: np.ndarray,
    post_Xc: np.ndarray,
    prior_edp: np.ndarray,
    alt_grid: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    truth_time: "pd.Timestamp | None" = None,
    save_path: str = "./Figures/test_param_iono/kf_covariance.png",
) -> None:
    """
    Four-panel covariance structure figure for the gridded Ne KF.

    Layout (2 rows × 2 cols):
      Row 0 — Prior:     Alt-Alt Ne correlation  |  Horizontal Ne correlation at hmF2
      Row 1 — Posterior: Alt-Alt Ne correlation  |  Horizontal Ne correlation at hmF2

    The alt-alt panels average the grid-point pair covariance over all geo
    vertices and normalise to a Pearson correlation matrix.

    The horizontal panels fix the centre grid point at the prior F2-peak
    altitude and plot the Pearson correlation of that state element with every
    other grid vertex at the same altitude, projected on an orthographic globe.

    Parameters
    ----------
    prior_Xc : (n_state, n_members) centered prior ensemble anomaly.
               Layout: x[k * n_geo + g] = Ne(alt_k, gp_g).
    post_Xc  : (n_state, n_members) linearized posterior ensemble anomaly,
               computed as (I − K H) X_c (ignores K R K^T obs-noise term).
    prior_edp : (n_alt, n_geo) prior-mean Ne profiles — used to locate hmF2.
    """
    import warnings

    n_state, n_members = prior_Xc.shape
    n_alt = len(alt_grid)
    n_geo = len(grid_lats_5deg)
    nm1   = max(n_members - 1, 1)

    # Centre grid point (nearest to spatial centroid)
    cen_lat = float(np.mean(grid_lats_5deg))
    cen_lon = float(np.mean(grid_lons_5deg))
    cen_idx = int(np.argmin(_haversine_km(cen_lat, cen_lon,
                                          grid_lats_5deg, grid_lons_5deg)))

    # F2-peak reference altitude from the prior-mean centre profile
    alt_ref_idx  = int(np.argmax(prior_edp[:, cen_idx]))
    true_alt_ref = float(alt_grid[alt_ref_idx])

    def _alt_corr(Xc):
        # Average the alt-alt covariance over all geo points.
        # Xc shape: (n_alt * n_geo, n_members) → reshape to (n_alt, n_geo*n_members)
        # so that a single matrix product gives the geo-averaged alt-alt cov.
        Xm  = Xc.reshape(n_alt, n_geo * n_members)
        cov = (Xm @ Xm.T) / (n_geo * nm1)          # (n_alt, n_alt)
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        outer = np.outer(std, std)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return cov / np.where(outer == 0, 1e-10, outer)

    def _horiz_corr(Xc):
        Xc4 = Xc.reshape(n_alt, n_geo, n_members)
        ref  = Xc4[alt_ref_idx, cen_idx, :]         # (n_members,)
        std_ref = float(np.sqrt(max(ref @ ref / nm1, 0.0)))
        horiz = np.empty(n_geo)
        for g in range(n_geo):
            gp      = Xc4[alt_ref_idx, g, :]
            cov_g   = float(ref @ gp / nm1)
            std_g   = float(np.sqrt(max(gp @ gp / nm1, 0.0)))
            denom   = std_ref * std_g
            horiz[g] = cov_g / denom if denom > 1e-30 else 0.0
        return horiz

    prior_alt_corr = _alt_corr(prior_Xc)
    post_alt_corr  = _alt_corr(post_Xc)
    prior_horiz    = _horiz_corr(prior_Xc)
    post_horiz     = _horiz_corr(post_Xc)

    clon = float(np.nanmean(grid_lons_5deg))
    clat = float(np.nanmean(grid_lats_5deg))
    proj = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)
    alt_extent = [float(alt_grid[0]), float(alt_grid[-1]),
                  float(alt_grid[0]), float(alt_grid[-1])]

    title_str = "Gridded Ne KF — Prior and Posterior Covariance Structure"
    if truth_time is not None:
        title_str += f"\n{truth_time.strftime('%Y-%m-%d %H:%M')} UTC truth"

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        title_str + f"\nHorizontal slice at {true_alt_ref:.0f} km  ·  ★ = centre vertex",
        fontsize=12,
    )
    gs = GridSpec(2, 2, figure=fig,
                  left=0.06, right=0.97, top=0.88, bottom=0.07,
                  wspace=0.30, hspace=0.35)

    for row, (row_lbl, alt_corr, horiz_corr) in enumerate([
        ("Prior",     prior_alt_corr, prior_horiz),
        ("Posterior", post_alt_corr,  post_horiz),
    ]):
        # ── Alt-Alt correlation ───────────────────────────────────────────────
        ax_aa = fig.add_subplot(gs[row, 0])
        pcm = ax_aa.imshow(
            alt_corr, cmap="coolwarm", vmin=-1, vmax=1,
            extent=alt_extent, origin="lower", aspect="auto",
        )
        ax_aa.axhline(true_alt_ref, color="gold", lw=1.0, ls="--", alpha=0.8)
        ax_aa.axvline(true_alt_ref, color="gold", lw=1.0, ls="--", alpha=0.8)
        ax_aa.set_xlabel("Altitude (km)", fontsize=9)
        ax_aa.set_ylabel("Altitude (km)", fontsize=9)
        ax_aa.set_title(f"{row_lbl} — Alt-Alt Ne Correlation", fontsize=10)
        fig.colorbar(pcm, ax=ax_aa, label="Pearson r", fraction=0.046, pad=0.04)

        # ── Horizontal correlation globe ──────────────────────────────────────
        ax_gl = fig.add_subplot(gs[row, 1], projection=proj)
        ax_gl.set_global()
        ax_gl.add_feature(cfeature.LAND,  facecolor="whitesmoke", zorder=0)
        ax_gl.add_feature(cfeature.OCEAN, facecolor="aliceblue",  zorder=0)
        ax_gl.add_feature(cfeature.COASTLINE.with_scale("110m"),
                          lw=0.4, edgecolor="gray")
        ax_gl.gridlines(lw=0.2, alpha=0.3)

        sc = ax_gl.scatter(
            grid_lons_5deg, grid_lats_5deg,
            c=horiz_corr, cmap="coolwarm", vmin=-1, vmax=1,
            s=80, transform=ccrs.Geodetic(), zorder=3,
        )
        cb = fig.colorbar(sc, ax=ax_gl, orientation="horizontal",
                          shrink=0.75, pad=0.04, fraction=0.04)
        cb.set_label("Pearson r", fontsize=8)

        ax_gl.plot(
            float(grid_lons_5deg[cen_idx]), float(grid_lats_5deg[cen_idx]),
            transform=ccrs.Geodetic(),
            marker="*", color="gold", ms=14, mec="black", mew=0.8, zorder=8,
        )
        ax_gl.set_title(
            f"{row_lbl} — Horizontal Ne Correlation at {true_alt_ref:.0f} km",
            fontsize=10,
        )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# §15  KF vs EKF_Param comparison figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_kf_ekf_comparison(
    arc_truth_list: list[dict],
    kf_result: dict,
    ekf_param_result: dict,
    truth_ne_1deg: np.ndarray,
    grid_lats_1deg: np.ndarray,
    grid_lons_1deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    alt_grid: np.ndarray,
    truth_time: "pd.Timestamp",
    save_path: str,
) -> None:
    """
    Side-by-side KF vs EKF_Param comparison figure.

    Layout — GridSpec(2, 3)
    ────────────────────────────────────────────────────────
    [0,0] GPS TEC      [0,1] GLONASS TEC    [0,2] EDP spaghetti
    [1,0] Galileo TEC  [1,1] BeiDou TEC     [1,2] RMSE comparison

    TEC panels
    ----------
    • Thick solid   : truth TEC (from 1-deg truth ionosphere)
    • Dashed        : prior model ensemble mean
    • Dash-dot blue : gridded Ne KF posterior
    • Dotted green  : EKF_Param posterior

    EDP panel
    ---------
    • Grey dashed  : prior Ne profiles (5-deg grid)
    • Blue solid   : KF posterior Ne profiles
    • Green solid  : EKF_Param posterior Ne profiles
    • Red dashed   : truth Ne at 5-deg grid centroid (1-deg grid nearest)

    RMSE panel
    ----------
    Grouped bar chart per arc plus global RMSE (prior / KF / EKF_Param).
    """
    from matplotlib.lines import Line2D

    occ_colors = _occ_colors(
        [{"conid": a["conid"], "prn_id": a["prn_id"]} for a in arc_truth_list]
    )

    fig = plt.figure(figsize=(22, 9), facecolor="#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    gs  = plt.GridSpec(2, 3, figure=fig,
                       width_ratios=[1, 1, 1.3],
                       wspace=0.38, hspace=0.45,
                       left=0.06, right=0.97, top=0.91, bottom=0.10)

    # ── TEC panels ───────────────────────────────────────────────────────────
    tec_axes: dict[str, plt.Axes] = {}
    first_ax = None
    for const, (row, col) in _CONST_POS.items():
        ax = fig.add_subplot(gs[row, col],
                             sharey=first_ax if first_ax is not None else None)
        ax.set_facecolor("#2b2b2b")
        cfg = CONSTELLATION_CONFIG.get(const, {"name": const, "title_color": "white"})
        ax.set_title(cfg["name"], color=cfg.get("title_color", "white"),
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("sTEC (TECU)", color="lightgray", fontsize=7)
        ax.set_ylabel("Tang. alt. (km)", color="lightgray", fontsize=7)
        ax.tick_params(colors="lightgray", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        tec_axes[const] = ax
        if first_ax is None:
            first_ax = ax

    # ── EDP spaghetti ─────────────────────────────────────────────────────────
    ax_edp = fig.add_subplot(gs[0, 2])
    ax_edp.set_facecolor("#2b2b2b")
    ax_edp.set_title("EDP — prior / KF / EKF_Param / truth", color="white", fontsize=8)
    ax_edp.set_xlabel("Ne (m⁻³)", color="lightgray", fontsize=7)
    ax_edp.set_ylabel("Altitude (km)", color="lightgray", fontsize=7)
    ax_edp.tick_params(colors="lightgray", labelsize=6)
    for sp in ax_edp.spines.values():
        sp.set_edgecolor("#555")

    kf_prior_edp    = kf_result["prior_edp"]           # (n_alt, n_geo)
    kf_post_edp     = kf_result["posterior_edp"]        # (n_alt, n_geo)
    ekf_post_edp    = ekf_param_result["posterior_edp"] # (n_alt, n_geo)
    n_geo           = kf_prior_edp.shape[1]

    for g in range(n_geo):
        ax_edp.plot(kf_prior_edp[:, g],  alt_grid, color="gray",
                    linewidth=0.6, alpha=0.4, linestyle="--")
        ax_edp.plot(kf_post_edp[:, g],   alt_grid, color="steelblue",
                    linewidth=0.8, alpha=0.7)
        ax_edp.plot(ekf_post_edp[:, g],  alt_grid, color="seagreen",
                    linewidth=0.8, alpha=0.7)

    # Truth at centroid of 5-deg grid
    cen_lat = float(np.mean(grid_lats_5deg))
    cen_lon = float(np.mean(grid_lons_5deg))
    dist_1deg = _haversine_km(cen_lat, cen_lon, grid_lats_1deg, grid_lons_1deg)
    cen_idx   = int(np.argmin(dist_1deg))
    ax_edp.plot(truth_ne_1deg[:, cen_idx], alt_grid,
                color="red", linewidth=1.8, linestyle="--",
                label="Truth (centroid)")

    edp_handles = [
        Line2D([0], [0], color="gray",      lw=1.2, linestyle="--", label="Prior Ne"),
        Line2D([0], [0], color="steelblue", lw=1.4, alpha=0.8,      label="KF posterior"),
        Line2D([0], [0], color="seagreen",  lw=1.4, alpha=0.8,      label="EKF_Param posterior"),
        Line2D([0], [0], color="red",       lw=1.8, linestyle="--", label="Truth Ne (centroid)"),
    ]
    ax_edp.legend(handles=edp_handles, fontsize=6,
                  facecolor="#2b2b2b", labelcolor="lightgray",
                  loc="upper right", framealpha=0.8)
    ax_edp.set_ylim(bottom=0)

    # Parameter readout for EKF_Param
    ekf_prior_state = ekf_param_result.get("prior_mean_state")
    ekf_post_state  = ekf_param_result.get("posterior_mean_5deg")
    if ekf_prior_state is not None and ekf_post_state is not None:
        cen5_idx = int(np.argmin(_haversine_km(
            cen_lat, cen_lon, grid_lats_5deg, grid_lons_5deg)))
        _draw_param_boxes(
            ax_edp,
            [("EKF prior",     "gray",     ekf_prior_state[:, cen5_idx]),
             ("EKF posterior", "seagreen", ekf_post_state[:, cen5_idx])],
            loc="lower left",
        )

    # ── RMSE comparison bar chart ─────────────────────────────────────────────
    ax_rmse = fig.add_subplot(gs[1, 2])
    ax_rmse.set_facecolor("#2b2b2b")
    ax_rmse.set_title("Per-arc RMSE: prior / KF / EKF_Param", color="white", fontsize=8)
    ax_rmse.set_xlabel("Arc", color="lightgray", fontsize=7)
    ax_rmse.set_ylabel("RMSE (TECU)", color="lightgray", fontsize=7)
    ax_rmse.tick_params(colors="lightgray", labelsize=6)
    for sp in ax_rmse.spines.values():
        sp.set_edgecolor("#555")

    arc_labels_kf  = kf_result["arc_labels"]
    kf_prior_rmse  = kf_result["arc_prior_rmse"]
    kf_post_rmse   = kf_result["arc_post_rmse"]
    ekf_post_rmse  = ekf_param_result["arc_post_rmse"]
    n_arcs         = len(arc_labels_kf)

    x_pos = np.arange(n_arcs, dtype=float)
    bw    = 0.24
    ax_rmse.bar(x_pos - bw, kf_prior_rmse, width=bw, color="#4c72b0", alpha=0.85,
                label="Prior (KF)")
    ax_rmse.bar(x_pos,      kf_post_rmse,  width=bw, color="#55a868", alpha=0.85,
                label="KF post")
    ax_rmse.bar(x_pos + bw, ekf_post_rmse, width=bw, color="#2ca02c", alpha=0.85,
                label="EKF_Param post")
    ax_rmse.set_xticks(x_pos)
    ax_rmse.set_xticklabels(arc_labels_kf, rotation=45, ha="right",
                             fontsize=6, color="lightgray")
    ax_rmse.legend(fontsize=6, facecolor="#2b2b2b", labelcolor="lightgray",
                   loc="upper right", framealpha=0.8)

    gkf_rmse  = kf_result["post_rmse"]
    gekf_rmse = ekf_param_result["post_rmse"]
    gpr_rmse  = kf_result["prior_rmse"]
    ax_rmse.text(0.02, 0.97,
                 f"Global:  Prior {gpr_rmse:.2f}  KF {gkf_rmse:.2f}  EKF_P {gekf_rmse:.2f} TECU",
                 transform=ax_rmse.transAxes, fontsize=6.5, color="lightgray",
                 va="top", ha="left")
    ax_rmse.grid(axis="y", lw=0.3, alpha=0.4)

    # ── TEC panels: truth / prior / KF / EKF_Param ────────────────────────────
    style_placed = False
    kf_slices  = kf_result["tec_slices"]
    ekf_slices = ekf_param_result["tec_slices"]

    for i, (arc, col) in enumerate(zip(arc_truth_list, occ_colors)):
        const = _resolve_conid(arc)
        label = _arc_label(arc)
        tang  = arc["tang_km"]
        ax    = tec_axes.get(const, tec_axes.get("G"))
        ksl   = kf_slices[i]
        esl   = ekf_slices[i]

        ax.plot(ksl["tec_truth"], tang, color=col,
                linewidth=2.2, zorder=6, label=label)
        ax.plot(ksl["prior_tec"], tang, color=col,
                linewidth=1.0, linestyle="--", alpha=0.55, zorder=3,
                label="Prior" if not style_placed else None)
        ax.plot(ksl["post_tec"],  tang, color="steelblue",
                linewidth=1.4, linestyle="-.", alpha=0.9, zorder=4,
                label="KF post" if not style_placed else None)
        ax.plot(esl["post_tec"],  tang, color="seagreen",
                linewidth=1.4, linestyle=":", alpha=0.9, zorder=5,
                label="EKF_P post" if not style_placed else None)
        style_placed = True

    for ax in tec_axes.values():
        if ax.lines:
            _capped_legend(ax, fontsize=5, facecolor="#2b2b2b",
                           labelcolor="lightgray", loc="best", framealpha=0.7)

    fig.suptitle(
        f"KF vs EKF_Param comparison — truth {truth_time.strftime('%Y-%m-%d %H:%M')} UTC  "
        f"(+{TRUTH_HOUR_OFFSET} h, F10.7+{TRUTH_F107_DELTA:.0f})\n"
        f"Prior {gpr_rmse:.2f} TECU  →  KF {gkf_rmse:.2f} TECU  |  "
        f"EKF_Param {gekf_rmse:.2f} TECU",
        color="white", fontsize=10, y=0.98,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")

def _run_enkf_retrieval_experiment(
    parsed_list: list[dict],
    meta_list: list[dict],
    mean_5deg: np.ndarray,
    grid_lats_1deg: np.ndarray,
    grid_lons_1deg: np.ndarray,
    grid_lats_5deg: np.ndarray,
    grid_lons_5deg: np.ndarray,
    time_dt: pd.Timestamp,
    save_dir: str = SAVE_DIR,
    save_suffix: str = "",
    mode: str = "ro_igs",
    igs_sim_geometry: list[dict] | None = None,
) -> dict:
    """
    Execute the filter retrieval experiment (gridded Ne KF + EKF_Param)
    and produce all diagnostic plots for one observation mode.

    Parameters
    ----------
    mode : {"ro_only", "ro_igs", "igs_only"}
        Selects which observations to feed the filters:
          - "ro_only":  arc_truth_list = RO arcs only          (parsed_list).
          - "ro_igs":   arc_truth_list = RO + simulated IGS.
          - "igs_only": arc_truth_list = simulated IGS only    (no RO).
        The caller is responsible for supplying grids/mean_5deg that make
        sense for the mode — for igs_only, those should be built from IGS
        pierce points rather than RO tangent tracks.
    igs_sim_geometry : list[dict] | None
        Pre-built (geometry-only) simulated IGS arcs from _build_igs_sim_arcs.
        Required for "ro_igs" and "igs_only"; ignored for "ro_only".

    Steps
    -----
    7   . Build truth ionosphere on the 1×1 Fibonacci grid (+1 h, F10.7+Δ).
          Also evaluate the truth at the 5×5 model grid for error comparison.
    8   . Generate synthetic sTEC measurements from the truth state.
    9b  . Run Gridded Ne KF (linear, single-step) — assimilate into Ne-space state.
    9c  . Plot: KF covariance panels (alt-alt + horizontal).
    9f  . Run EKF_Param (iterative EKF on parametric IRI state).
    10b . Plot: KF TEC profiles + globe + EDP spaghetti.
    10c . Plot: EKF_Param TEC profiles + globe + EDP spaghetti.
    11b . Plot: KF per-arc innovation diagnostic.
    11c . Plot: EKF_Param per-arc innovation diagnostic.
    12b . Plot: KF 5×3 EDP spatial error.
    12c . Plot: EKF_Param 5×2 EDP spatial error.
    14  . Plot: KF vs EKF_Param comparison (TEC, EDP, RMSE bar chart).
    """
    if mode not in FILTER_MODES:
        raise ValueError(f"unknown mode={mode!r}; expected one of {FILTER_MODES}")
    print(f"\n  [mode={mode}]  save_suffix={save_suffix!r}")
    # ── Step 7: truth ionosphere ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 7: Building truth ionosphere "
          f"(1×1 grid, +{TRUTH_HOUR_OFFSET} h, F10.7+{TRUTH_F107_DELTA:.0f}) …")

    truth_time, truth_sdf = _truth_solar_conditions(time_dt)
    print(f"  Base time:  {time_dt}")
    print(f"  Truth time: {truth_time}  (F10.7 → {truth_sdf['f107'].iloc[0]:.1f})")

    truth_state_1deg, truth_ne_1deg, truth_mean_1deg = build_truth_iri_grid(
        truth_time, truth_sdf,
        grid_lats_1deg, grid_lons_1deg, ALT_GRID,
        label="1-deg truth",
    )

    # Truth at the 5-deg model grid for error comparison
    truth_state_5deg, truth_ne_5deg, truth_mean_5deg = build_truth_iri_grid(
        truth_time, truth_sdf,
        grid_lats_5deg, grid_lons_5deg, ALT_GRID,
        label="5-deg truth (model grid)",
    )
    print(f"  Truth 1-deg: {truth_state_1deg.n_grid_points} grid points")
    print(f"  Truth 5-deg: {truth_state_5deg.n_grid_points} grid points")

    # ── Step 8: truth TEC measurements ───────────────────────────────────────
    # Mode selects which arcs enter arc_truth_list; the KF and EKF see it as a
    # plain arc list without any special handling per mode.
    arc_truth_list: list[dict] = []

    if mode in ("ro_only", "ro_igs"):
        print(f"\nStep 8: Generating truth sTEC from {len(parsed_list)} RO arcs …")
        arc_truth_list = generate_truth_tec(
            parsed_list, truth_state_1deg,
            grid_lats_1deg, grid_lons_1deg, ALT_GRID,
        )
    else:
        print(f"\nStep 8: [mode=igs_only] skipping RO truth TEC "
              f"({len(parsed_list)} RO arcs available but not assimilated).")

    # ── Step 8b: augment with simulated IGS ground-station arcs (per mode) ───
    # Real IGS station coordinates + real GNSS satellite ECEF positions from
    # the day's broadcast ephemeris, forward-modelled through the same truth
    # state as the RO arcs.  For "igs_only" this fully replaces the RO block;
    # for "ro_igs" it appends.
    if mode in ("ro_igs", "igs_only"):
        if not igs_sim_geometry:
            print(f"\nStep 8b: [mode={mode}] no IGS-sim geometry supplied — "
                  f"arc_truth_list will be empty for this pass.")
        else:
            print(f"\nStep 8b: Forward-modelling {len(igs_sim_geometry)} "
                  f"simulated IGS arcs through the truth state …")
            igs_arcs = generate_simulated_igs_tec(
                igs_sim_geometry, truth_state_1deg,
                grid_lats_1deg, grid_lons_1deg, ALT_GRID,
            )
            n_ro_arcs  = len(arc_truth_list)
            n_ro_rays  = sum(len(a["rays"]) for a in arc_truth_list)
            arc_truth_list.extend(igs_arcs)
            n_igs_rays = sum(len(a["rays"]) for a in igs_arcs)
            print(f"  [IGS-sim] Added {len(igs_arcs)} arcs / {n_igs_rays} rays  "
                  f"→ arc_truth_list: {n_ro_arcs} RO + {len(igs_arcs)} IGS = "
                  f"{len(arc_truth_list)} arcs  "
                  f"({n_ro_rays + n_igs_rays} total rays).")

    if not arc_truth_list:
        print(f"  [mode={mode}] arc_truth_list is empty — skipping filters "
              f"for this pass.")
        return {
            "mode":             mode,
            "prior_rmse":       None,
            "kf_result":        None,
            "ekf_param":        None,
            "arc_truth_list":   [],
            "truth_time":       truth_time,
            "truth_ne_1deg":    truth_ne_1deg,
            "truth_ne_5deg":    truth_ne_5deg,
            "truth_mean_5deg":  truth_mean_5deg,
            "grid_lats_1deg":   grid_lats_1deg,
            "grid_lons_1deg":   grid_lons_1deg,
            "grid_lats_5deg":   grid_lats_5deg,
            "grid_lons_5deg":   grid_lons_5deg,
        }

    kf_result   = None

    # ── Step 9b: Gridded Ne KF assimilation ──────────────────────────────────
    if RUN_KF:
        print(f"\nStep 9b: Running gridded Ne KF (linear update, IRI baseline prior) …")
        # build_model_ensemble pins member 0 to the unperturbed IRI baseline;
        # the remaining members are drawn stochastically for the covariance.
        model_state_kf = build_model_ensemble(
            mean_5deg, grid_lats_5deg, grid_lons_5deg,
            n_members=ENKF_N_MEMBERS, corr_length_km=CORR_LENGTH_KM,
        )
        kf_result = run_gridded_ne_kf(
            arc_truth_list, model_state_kf,
            grid_lats_5deg, grid_lons_5deg, ALT_GRID,
        )
    else:
        print("\n  [skip] KF disabled (RUN_KF = False)")

    # ── Step 9f: EKF_Param assimilation ──────────────────────────────────────
    if RUN_EKF_PARAM:
        print(f"\nStep 9f: Running EKF_Param "
              f"(alpha={EKF_PARAM_ALPHA}, tol={EKF_PARAM_TOL:.1e}, "
              f"max_iter={EKF_PARAM_MAX_ITER}) …")
        model_state_ekf = build_model_ensemble(
            mean_5deg, grid_lats_5deg, grid_lons_5deg,
            n_members=ENKF_N_MEMBERS, corr_length_km=CORR_LENGTH_KM,
        )
        ekf_param_result = EKF_Param(
            arc_truth_list, model_state_ekf,
            grid_lats_5deg, grid_lons_5deg, ALT_GRID,
            sigma_obs      = EKF_PARAM_SIGMA_OBS,
            max_update_rays= EKF_PARAM_UPDATE_RAYS,
            alpha                = EKF_PARAM_ALPHA,
            tol                  = EKF_PARAM_TOL,
            max_iter             = EKF_PARAM_MAX_ITER,
            apply_bounds         = EKF_PARAM_APPLY_BOUNDS,
            eps_jac              = EKF_PARAM_EPS_JAC,
            n_workers            = EKF_PARAM_N_WORKERS,
            jacobian_analytical  = EKF_PARAM_JAC_ANALYTICAL,
        )
    else:
        ekf_param_result = None
        print("\n  [skip] EKF_Param disabled (RUN_EKF_PARAM = False)")

    # ── Step 9c: KF covariance panels ────────────────────────────────────────
    if kf_result is not None:
        print("\nStep 9c: Plotting KF covariance panels (alt-alt + horizontal) …")
        plot_kf_covariance_panels(
            prior_Xc       = kf_result["prior_Xc"],
            post_Xc        = kf_result["post_Xc"],
            prior_edp      = kf_result["prior_edp"],
            alt_grid       = ALT_GRID,
            grid_lats_5deg = grid_lats_5deg,
            grid_lons_5deg = grid_lons_5deg,
            truth_time     = truth_time,
            save_path      = os.path.join(save_dir, f"kf_covariance_{YYYY}_{DOY:03d}{save_suffix}.png"),
        )

    # ── Step 10b: TEC + globe + EDP plot (KF) ────────────────────────────────
    if kf_result is not None:
        print("\nStep 10b: Plotting KF TEC profiles, globe, and EDP spaghetti …")
        plot_kf_tec_edp(
            arc_truth_list, kf_result, truth_ne_1deg,
            grid_lats_1deg, grid_lons_1deg,
            grid_lats_5deg, grid_lons_5deg,
            ALT_GRID, truth_time,
            save_path=os.path.join(save_dir, f"kf_tec_edp_{YYYY}_{DOY:03d}{save_suffix}.png"),
        )

    # ── Step 10c: TEC + globe + EDP plot (EKF_Param) ─────────────────────────
    if ekf_param_result is not None:
        print("\nStep 10c: Plotting EKF_Param TEC profiles, globe, and EDP spaghetti …")
        _plot_tec_edp_figure(
            arc_truth_list,
            ekf_param_result["tec_slices"],
            ekf_param_result["prior_edp"],
            ekf_param_result["posterior_edp"],
            truth_ne_1deg,
            grid_lats_1deg, grid_lons_1deg,
            grid_lats_5deg, grid_lons_5deg,
            ALT_GRID,
            suptitle=(
                f"EKF_Param retrieval — truth ionosphere "
                f"{truth_time.strftime('%Y-%m-%d %H:%M')} UTC  "
                f"(+{TRUTH_HOUR_OFFSET} h, F10.7+{TRUTH_F107_DELTA:.0f})\n"
                f"Prior RMSE {ekf_param_result['prior_rmse']:.2f} TECU  →  "
                f"Posterior RMSE {ekf_param_result['post_rmse']:.2f} TECU  "
                f"({ekf_param_result['n_iterations']} iters, "
                f"{'converged' if ekf_param_result['converged'] else 'not converged'})"
            ),
            save_path=os.path.join(save_dir, f"ekf_param_tec_edp_{YYYY}_{DOY:03d}{save_suffix}.png"),
            prior_mean_state=ekf_param_result["prior_mean_state"],
            post_mean_state=ekf_param_result["posterior_mean_5deg"],
        )

    # ── Step 11b: arc innovation diagnostic (KF) ──────────────────────────────
    if kf_result is not None:
        print("\nStep 11b: Plotting KF per-arc innovation diagnostic …")
        _plot_arc_innovation_diagnostic(
            arc_labels          = kf_result["arc_labels"],
            arc_prior_mean      = kf_result["arc_prior_mean"],
            arc_post_mean       = kf_result["arc_post_mean"],
            arc_prior_rmse      = kf_result["arc_prior_rmse"],
            arc_post_rmse       = kf_result["arc_post_rmse"],
            arc_lats            = kf_result["arc_lats"],
            arc_lons            = kf_result["arc_lons"],
            all_prior           = kf_result["all_prior_resid"],
            all_post_main       = kf_result["all_post_resid"],
            group_key           = f"{YYYY}_{DOY:03d}{save_suffix}_kf",
            save_dir            = save_dir,
            filter_name         = "GriddedNeKF",
            prior_rmse          = kf_result["prior_rmse"],
            post_rmse           = kf_result["post_rmse"],
            mda_arc_means_list  = None,
            mda_flat_list       = None,
        )

    # ── Step 11c: arc innovation diagnostic (EKF_Param) ─────────────────────
    if ekf_param_result is not None:
        print("\nStep 11c: Plotting EKF_Param per-arc innovation diagnostic …")
        _plot_arc_innovation_diagnostic(
            arc_labels         = ekf_param_result["arc_labels"],
            arc_prior_mean     = ekf_param_result["arc_prior_mean"],
            arc_post_mean      = ekf_param_result["arc_post_mean"],
            arc_prior_rmse     = ekf_param_result["arc_prior_rmse"],
            arc_post_rmse      = ekf_param_result["arc_post_rmse"],
            arc_lats           = ekf_param_result["arc_lats"],
            arc_lons           = ekf_param_result["arc_lons"],
            all_prior          = ekf_param_result["all_prior_resid"],
            all_post_main      = ekf_param_result["all_post_resid"],
            group_key          = f"{YYYY}_{DOY:03d}{save_suffix}_ekf_param",
            save_dir           = save_dir,
            filter_name        = "EKF_Param",
            prior_rmse         = ekf_param_result["prior_rmse"],
            post_rmse          = ekf_param_result["post_rmse"],
            mda_arc_means_list = None,
            mda_flat_list      = None,
        )

    # ── Step 12b: EDP spatial error (5×3) — KF with truth column ─────────────
    if kf_result is not None:
        print("\nStep 12b: Plotting KF EDP spatial error (5×3 orthographic) …")
        plot_edp_spatial_error(
            truth_ne_5deg, kf_result["posterior_ne_5deg"],
            grid_lats_5deg, grid_lons_5deg, ALT_GRID,
            truth_time=truth_time,
            save_path=os.path.join(save_dir, f"edp_spatial_error_kf_{YYYY}_{DOY:03d}{save_suffix}.png"),
        )

    # ── Step 12c: EDP spatial error — EKF_Param ──────────────────────────────
    if ekf_param_result is not None:
        print("\nStep 12c: Plotting EKF_Param EDP spatial error (5×2 orthographic) …")
        plot_edp_spatial_error(
            truth_ne_5deg, ekf_param_result["posterior_ne_5deg"],
            grid_lats_5deg, grid_lons_5deg, ALT_GRID,
            truth_time=truth_time,
            save_path=os.path.join(save_dir, f"edp_spatial_error_ekf_param_{YYYY}_{DOY:03d}{save_suffix}.png"),
        )

    # ── Step 14: KF vs EKF_Param comparison plot ─────────────────────────────
    if kf_result is not None and ekf_param_result is not None:
        print("\nStep 14: Plotting KF vs EKF_Param comparison …")
        plot_kf_ekf_comparison(
            arc_truth_list    = arc_truth_list,
            kf_result         = kf_result,
            ekf_param_result  = ekf_param_result,
            truth_ne_1deg     = truth_ne_1deg,
            grid_lats_1deg    = grid_lats_1deg,
            grid_lons_1deg    = grid_lons_1deg,
            grid_lats_5deg    = grid_lats_5deg,
            grid_lons_5deg    = grid_lons_5deg,
            alt_grid          = ALT_GRID,
            truth_time        = truth_time,
            save_path         = os.path.join(
                save_dir, f"kf_vs_ekf_param_comparison_{YYYY}_{DOY:03d}{save_suffix}.png"
            ),
        )


    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Filter comparison summary")
    print("=" * 60)
    _ref = kf_result or ekf_param_result
    prior_rmse = None
    if _ref is not None:
        prior_rmse = float(_ref["prior_rmse"])
        print(f"  Prior  RMSE:  {prior_rmse:.3f} TECU")
    if kf_result is not None:
        print(f"  KF     RMSE:  {kf_result['post_rmse']:.3f} TECU"
              f"  (Δ = {kf_result['post_rmse'] - kf_result['prior_rmse']:+.3f})")
    if ekf_param_result is not None:
        conv_tag = f"converged in {ekf_param_result['n_iterations']} iter" \
                   if ekf_param_result['converged'] \
                   else f"{ekf_param_result['n_iterations']} iter (not converged)"
        print(f"  EKF_P  RMSE:  {ekf_param_result['post_rmse']:.3f} TECU"
              f"  (Δ = {ekf_param_result['post_rmse'] - ekf_param_result['prior_rmse']:+.3f})"
              f"  [{conv_tag}]")
    print(f"\n✓ Retrieval experiment complete.  Figures in: {save_dir}")

    # Per-window metrics returned to the caller for later comparison plots.
    # arc_truth_list carries per-arc geometry + labels; truth_ne_{1,5}deg
    # and the two grids feed plot_edp_site_rmse_across_windows and
    # plot_edp_regional_rmse_across_windows in main().
    return {
        "mode":             mode,
        "prior_rmse":       prior_rmse,
        "kf_result":        kf_result,
        "ekf_param":        ekf_param_result,
        "arc_truth_list":   arc_truth_list,
        "truth_time":       truth_time,
        "truth_ne_1deg":    truth_ne_1deg,
        "truth_ne_5deg":    truth_ne_5deg,
        "truth_mean_5deg":  truth_mean_5deg,   # (N_STATE, n_geo) 8-param truth state
        "grid_lats_1deg":   grid_lats_1deg,
        "grid_lons_1deg":   grid_lons_1deg,
        "grid_lats_5deg":   grid_lats_5deg,
        "grid_lons_5deg":   grid_lons_5deg,
    }


def _resolve_sweep_base_path(yyyy: int, doy: int) -> "str | None":
    """Return the podTc2 day directory for (yyyy, doy), or None if absent.

    Tries both zero-padded ("2025.001") and bare ("2025.1") DOY spellings so
    the sweep works regardless of how the day folders are named on disk.
    """
    for doy_str in (f"{doy:03d}", str(doy)):
        p = f"{SWEEP_PODTC2_ROOT}/{yyyy}.{doy_str}/"
        if os.path.isdir(p):
            return p
    return None


def _sweep_prepare_window(window: dict) -> "dict | None":
    """
    Build the truth ionosphere, RO/IGS arc truth lists, model prior, and grids
    for one time window — everything run_occ_count_sweep needs.  Mirrors the
    per-window setup of _process_time_window / _run_enkf_retrieval_experiment
    but stops short of the plotting-heavy filter suite.

    Returns None if the window has no parsable RO arcs.
    """
    time_dt = window["time_dt"]
    records = window["records"]

    # Parse RO arc files.
    parsed_list: list[dict] = []
    for rec in records:
        try:
            data = parse_podTc2_nc_file(rec["path"])
            data["conid"]  = rec["conid"]
            data["prn_id"] = rec["prn_id"]
            data["leo_id"] = rec["leo_id"]
            parsed_list.append(data)
        except Exception as exc:
            warnings.warn(f"Could not parse {rec['path']}: {exc}")
    if not parsed_list:
        return None

    # Fibonacci grids (anchored on the RO tangent tracks).
    tp_lats_all, tp_lons_all = _arc_tangent_tracks(parsed_list)
    grid_lats_1deg, grid_lons_1deg = _make_fibonacci_grid(
        tp_lats_all, tp_lons_all, spacing_deg=1.0)
    grid_lats_5deg, grid_lons_5deg = _make_fibonacci_grid(
        tp_lats_all, tp_lons_all, spacing_deg=5.0)

    # 5-deg IRI background mean → model prior ensemble (nominal solar conditions).
    mean_5deg, _ = build_iri_state_grid_cached(
        time_dt, grid_lats_5deg, grid_lons_5deg, ALT_GRID,
        spacing_deg=5.0,
        lat_min=float(grid_lats_5deg.min()), lat_max=float(grid_lats_5deg.max()),
        lon_min=float(grid_lons_5deg.min()), lon_max=float(grid_lons_5deg.max()),
    )
    model_state = build_model_ensemble(
        mean_5deg, grid_lats_5deg, grid_lons_5deg,
        n_members=ENKF_N_MEMBERS, corr_length_km=CORR_LENGTH_KM,
    )

    # Shifted-IRI truth (matches the real retrieval experiment: +TRUTH_HOUR_OFFSET
    # hours, F10.7 + TRUTH_F107_DELTA) on the 1-deg grid.
    truth_time, truth_sdf = _truth_solar_conditions(time_dt)
    truth_state_1deg, truth_ne_1deg, _ = build_truth_iri_grid(
        truth_time, truth_sdf, grid_lats_1deg, grid_lons_1deg, ALT_GRID,
        label="1-deg truth (sweep)",
    )

    # RO arcs forward-modelled through the truth state.
    all_ro_arcs = generate_truth_tec(
        parsed_list, truth_state_1deg, grid_lats_1deg, grid_lons_1deg, ALT_GRID)

    # Simulated IGS arcs (optional), forward-modelled through the same truth.
    igs_arcs: list[dict] = []
    if USE_SIMULATED_IGS:
        igs_stations = _load_igs_sim_stations(
            IGS_SIM_STATIONS_JSON, IGS_SIM_STATIONS, roi_max_km=ISR_ROI_MAX_KM)
        if igs_stations:
            ephem = _load_broadcast_ephemeris(time_dt)
            if ephem is not None:
                igs_geom = _build_igs_sim_arcs(
                    igs_stations, time_dt, ephem, window_minutes=WINDOW_MINUTES)
                if igs_geom:
                    igs_arcs = generate_simulated_igs_tec(
                        igs_geom, truth_state_1deg,
                        grid_lats_1deg, grid_lons_1deg, ALT_GRID)

    return dict(
        model_state     = model_state,
        grid_lats_5deg  = grid_lats_5deg,
        grid_lons_5deg  = grid_lons_5deg,
        grid_lats_1deg  = grid_lats_1deg,
        grid_lons_1deg  = grid_lons_1deg,
        truth_ne_1deg   = truth_ne_1deg,
        truth_time      = truth_time,
        all_ro_arcs     = all_ro_arcs,
        igs_arcs        = igs_arcs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §sweep-plots  N_OCC sweep results plotting
# ─────────────────────────────────────────────────────────────────────────────

_SWEEP_THRESHOLDS_MHZ = (0.5, 0.2, 0.1)


def _sweep_thr_str(thr: float) -> str:
    return str(thr).replace(".", "")


def plot_occ_sweep_results(
    csv_path: str = SWEEP_RESULTS_CSV,
    save_dir: str = SWEEP_SAVE_DIR,
) -> None:
    """
    Read *csv_path* (the row-per-N_OCC/mode/site output of run_occ_count_sweep,
    accumulated across dates by main_sweep) and render cross-date, cross-mode
    summary figures of retrieval accuracy vs. N_OCC:

        occ_sweep_foF2_vs_nocc.png          — median |foF2 error| vs. N_OCC
        occ_sweep_foE_vs_nocc.png           — median |foE error|  vs. N_OCC
        occ_sweep_profile_rmse_vs_nocc.png  — median profile fp RMSE vs. N_OCC
        occ_sweep_threshold_fractions.png   — fraction-within-threshold vs. N_OCC
        occ_sweep_hf_propagation.png        — truth vs. posterior foF2/foE scatter

    All figures use posterior (post-assimilation) metrics, since the sweep's
    purpose is to show how N_OCC affects retrieval skill; each panel is one
    observation mode (ro_only / ro_igs / igs_only).
    """
    if not os.path.exists(csv_path):
        print(f"[plot_occ_sweep_results] CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[plot_occ_sweep_results] CSV is empty: {csv_path}")
        return
    os.makedirs(save_dir, exist_ok=True)

    modes   = list(FILTER_MODES)
    seasons = sorted(df["date_label"].dropna().unique().tolist()) if "date_label" in df else []
    season_cmap   = plt.get_cmap("tab10")
    season_colors = {s: season_cmap(i % 10) for i, s in enumerate(seasons)}

    def _median_by_season(mode: str, col: str, abs_val: bool = False) -> dict:
        sub = df[df["mode"] == mode]
        out = {}
        if sub.empty or col not in sub.columns:
            return out
        for season, g in sub.groupby("date_label"):
            s = g.groupby("n_occ")[col].apply(
                lambda v: float(np.nanmedian(np.abs(v))) if abs_val
                          else float(np.nanmedian(v))
            ).sort_index()
            out[season] = s
        return out

    def _line_panel(ax, mode: str, col: str, abs_val: bool, ylabel: str,
                     hlines: tuple = ()) -> None:
        series_by_season = _median_by_season(mode, col, abs_val=abs_val)
        for h in hlines:
            ax.axhline(h, color="k", ls="--", lw=0.8, alpha=0.5)
        if not series_by_season:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="0.5")
        else:
            for season, s in series_by_season.items():
                ax.plot(s.index, s.values, marker="o", ms=4,
                        color=season_colors.get(season, "0.3"), label=season)
        ax.set_title(mode)
        ax.set_xlabel("N_OCC")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    def _three_panel_line_fig(col: str, abs_val: bool, ylabel: str,
                               suptitle: str, fname: str) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, mode in zip(axes, modes):
            _line_panel(ax, mode, col, abs_val=abs_val, ylabel=ylabel,
                        hlines=_SWEEP_THRESHOLDS_MHZ)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, fontsize=7, loc="upper right")
        fig.suptitle(suptitle)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, fname), dpi=150)
        plt.close(fig)

    # ── Figure 1: N_OCC vs foF2 error ─────────────────────────────────────
    _three_panel_line_fig(
        "post_foF2_err", abs_val=True, ylabel="median |foF2 error| (MHz)",
        suptitle="N_OCC vs. posterior foF2 error (median across ISR sites)",
        fname="occ_sweep_foF2_vs_nocc.png",
    )

    # ── Figure 2: N_OCC vs foE error ──────────────────────────────────────
    _three_panel_line_fig(
        "post_foE_err", abs_val=True, ylabel="median |foE error| (MHz)",
        suptitle="N_OCC vs. posterior foE error (median across ISR sites)",
        fname="occ_sweep_foE_vs_nocc.png",
    )

    # ── Figure 3: N_OCC vs profile fp RMSE ────────────────────────────────
    _three_panel_line_fig(
        "post_fp_rmse", abs_val=False, ylabel="median profile fp RMSE (MHz)",
        suptitle="N_OCC vs. posterior profile plasma-frequency RMSE "
                  "(median across ISR sites)",
        fname="occ_sweep_profile_rmse_vs_nocc.png",
    )

    # ── Figure 4: threshold fractions (3 modes × 3 thresholds) ────────────
    fig, axes = plt.subplots(len(modes), len(_SWEEP_THRESHOLDS_MHZ),
                              figsize=(13, 3.6 * len(modes)),
                              sharex=True, sharey=True, squeeze=False)
    metric_style = [("foF2", "-"), ("foE", "--"), ("profile", ":")]
    for row, mode in enumerate(modes):
        sub_mode = df[df["mode"] == mode]
        for col_i, thr in enumerate(_SWEEP_THRESHOLDS_MHZ):
            ax = axes[row][col_i]
            if sub_mode.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="0.5")
            else:
                for metric, ls in metric_style:
                    colname = f"within_{_sweep_thr_str(thr)}mhz_{metric}_post"
                    if colname not in sub_mode.columns:
                        continue
                    frac = sub_mode.groupby("n_occ")[colname].mean().sort_index()
                    ax.plot(frac.index, frac.values, ls, marker="o", ms=3,
                            label=metric)
            if row == 0:
                ax.set_title(f"±{thr} MHz")
            if col_i == 0:
                ax.set_ylabel(f"{mode}\nfraction within threshold")
            if row == len(modes) - 1:
                ax.set_xlabel("N_OCC")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][0].legend(handles, labels, fontsize=7, loc="lower right")
    fig.suptitle("Fraction of cases within frequency threshold vs. N_OCC "
                 "(posterior)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "occ_sweep_threshold_fractions.png"),
                dpi=150)
    plt.close(fig)

    # ── Figure 5: HF propagation perspective (truth vs. posterior scatter) ─
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    n_occ_vals = pd.to_numeric(df["n_occ"], errors="coerce").dropna()
    if len(n_occ_vals):
        norm = mpl.colors.Normalize(vmin=float(n_occ_vals.min()),
                                     vmax=float(n_occ_vals.max()))
    else:
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
    scat_cmap  = plt.get_cmap("viridis")
    scat_proxy = None
    req_cols = ("truth_foF2", "post_foF2", "truth_foE", "post_foE")
    for ax, mode in zip(axes, modes):
        sub = df[df["mode"] == mode]
        if sub.empty or not all(c in sub.columns for c in req_cols):
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="0.5")
            ax.set_title(mode)
            continue
        scat_proxy = ax.scatter(sub["truth_foF2"], sub["post_foF2"],
                                 c=sub["n_occ"], cmap=scat_cmap, norm=norm,
                                 marker="o", s=22, edgecolor="none",
                                 label="foF2", alpha=0.8)
        ax.scatter(sub["truth_foE"], sub["post_foE"],
                   c=sub["n_occ"], cmap=scat_cmap, norm=norm,
                   marker="^", s=22, edgecolor="none",
                   label="foE", alpha=0.8)
        vals = pd.concat([sub["truth_foF2"], sub["post_foF2"],
                           sub["truth_foE"], sub["post_foE"]]).dropna()
        if not vals.empty:
            lo, hi = float(vals.min()), float(vals.max())
            pad = 0.05 * max(hi - lo, 1e-6)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    "k--", lw=1.0, label="perfect retrieval")
        ax.set_title(mode)
        ax.set_xlabel("truth frequency (MHz)")
        ax.set_ylabel("posterior frequency (MHz)")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    if scat_proxy is not None:
        fig.colorbar(scat_proxy, ax=axes, label="N_OCC", shrink=0.85)
    fig.suptitle("HF propagation perspective: posterior vs. truth foF2/foE, "
                 "colored by N_OCC")
    fig.savefig(os.path.join(save_dir, "occ_sweep_hf_propagation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_occ_sweep_results] wrote 5 figures to {save_dir}")


def main_sweep() -> None:
    """
    Entry point for the N_OCC sweep across multiple dates/seasons.
    Invoked with:  python test_param_iono.py --sweep
    Saves per-(date, window, N_OCC, mode, ISR-site) rows to SWEEP_RESULTS_CSV.
    """
    os.makedirs(SWEEP_SAVE_DIR, exist_ok=True)
    os.makedirs(IRI_CACHE_DIR, exist_ok=True)
    csv_dir = os.path.dirname(SWEEP_RESULTS_CSV)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    print("=" * 60)
    print("N_OCC sweep")
    print(f"  dates      : {[f'{y}.{d:03d} ({lbl})' for y, d, lbl in SWEEP_DATES]}")
    print(f"  N_OCC grid : {SWEEP_N_OCC_VALUES}")
    print(f"  modes      : {FILTER_MODES}")
    print(f"  output CSV : {SWEEP_RESULTS_CSV}")
    print("=" * 60)

    all_rows: list[dict] = []

    for (yyyy, doy, label) in SWEEP_DATES:
        base_path = _resolve_sweep_base_path(yyyy, doy)
        if base_path is None:
            print(f"\n[skip] {yyyy}.{doy:03d} ({label}) — directory not found "
                  f"under {SWEEP_PODTC2_ROOT}.")
            continue

        print("\n" + "#" * 70)
        print(f"#  {yyyy}.{doy:03d} ({label})  —  {base_path}")
        print("#" * 70)

        try:
            windows = scan_and_select_files_per_window(base_path)
        except Exception as exc:
            print(f"[skip] scan failed for {base_path}: "
                  f"{type(exc).__name__}: {exc}")
            continue
        if not windows:
            print(f"[skip] no windows retained for {yyyy}.{doy:03d}.")
            continue

        for wi, window in enumerate(windows):
            print("\n" + "-" * 66)
            print(f"  Window {wi+1}/{len(windows)}: {window['window_key']} "
                  f"({len(window['records'])} arcs)")
            print("-" * 66)
            try:
                prep = _sweep_prepare_window(window)
            except Exception as exc:
                print(f"  ✗ window prep FAILED: {type(exc).__name__}: {exc}")
                continue
            if prep is None:
                print("  [skip] no parsable RO arcs.")
                continue

            try:
                rows = run_occ_count_sweep(
                    window          = window,
                    model_state     = prep["model_state"],
                    grid_lats       = prep["grid_lats_5deg"],
                    grid_lons       = prep["grid_lons_5deg"],
                    alt_grid        = ALT_GRID,
                    all_ro_arcs     = prep["all_ro_arcs"],
                    igs_arcs        = prep["igs_arcs"],
                    truth_ne_1deg   = prep["truth_ne_1deg"],
                    grid_lats_truth = prep["grid_lats_1deg"],
                    grid_lons_truth = prep["grid_lons_1deg"],
                    truth_time      = prep["truth_time"],
                )
            except Exception as exc:
                print(f"  ✗ sweep FAILED: {type(exc).__name__}: {exc}")
                continue

            for r in rows:
                r["date_label"] = label
            all_rows.extend(rows)
            print(f"  ✓ window contributed {len(rows)} rows "
                  f"({len(all_rows)} total).")

            # Persist incrementally so a mid-run crash still leaves partial data.
            pd.DataFrame(all_rows).to_csv(SWEEP_RESULTS_CSV, index=False)

    if not all_rows:
        print("\nNo sweep rows produced — nothing written.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(SWEEP_RESULTS_CSV, index=False)
    print("\n" + "=" * 60)
    print(f"✓ N_OCC sweep complete: {len(df)} rows → {SWEEP_RESULTS_CSV}")
    print("=" * 60)

    print("\nPlotting sweep results …")
    try:
        plot_occ_sweep_results(SWEEP_RESULTS_CSV, SWEEP_SAVE_DIR)
    except Exception as exc:
        print(f"  [warn] plot_occ_sweep_results failed: "
              f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        main_sweep()
    else:
        main()
