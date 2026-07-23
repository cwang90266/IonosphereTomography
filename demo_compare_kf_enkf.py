#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
demo_compare_kf_enkf.py — Side-by-side comparison of the standard voxel-grid
Kalman Filter (KF) retrieval against the new parametric Ensemble Kalman Filter
(EnKF) developed in Tasks 1–3.

Both filters assimilate the same GNSS-RO sTEC observations over the Millstone
Hill ISR verification region and are compared against the same ISR ground-truth.

Output figures (per orbit group)
─────────────────────────────────
  group_<key>_joint_kf.png     — standard 4-column joint results plot (KF)
  group_<key>_joint_enkf.png   — same layout for parametric EnKF
  compare_<key>_2x2.png        — 2×2 direct comparison:
      [0,0] Prior TEC    — measured vs. KF/EnKF prior TEC profiles
      [0,1] Posterior TEC — measured vs. KF/EnKF posterior TEC profiles
      [1,0] Prior EDP    — IRI prior, KF prior, EnKF prior, ISR truth at MH vertex
      [1,1] Posterior EDP — KF posterior, EnKF posterior, ISR truth at MH vertex

Usage
─────
    python demo_compare_kf_enkf.py
    
    Use this command to remove KF cache
    rm -rf ./Figures/CompareKF_EnKF/.kf_cache/
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EDPSamples" / "Locate in mesh" / "outputs"))
sys.path.insert(0, str(ROOT / "iri2020_new" / "src"))

import warnings
import datetime
import hashlib
import pickle

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D

import pyproj
from scipy.optimize import minimize
from scipy.spatial import cKDTree

import netCDF4

# ── project imports ────────────────────────────────────────────────────────────
from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
from EDPSamples.edp_samples import EDPSamples
from demo import build_daily_global_edps, extract_robust_f2_peak
from EDPSamples.edp_samples import get_IRI2020_EDP
from IRI_Sample_Inputs.IRI_Sample_inputs import IRI_Sample_Inputs

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import demo_group as _demo_group
from demo_group import (
    scan_metadata,
    process_group,
    _make_tec_slices,
    _ECEF_TO_LL,
    _plot_group,
    WINDOW_MINUTES,
    CONSTELLATION_CONFIG,
    _CONST_FALLBACK_CMAP,
    _save_stats_csv,
    GAUSSIAN_COV_SIGMA,
    _draw_raypath,
    _draw_leo_path,
    _draw_terminator,
    _draw_roi_boundary,
    _roi_centre_idx,
    _parse_time_window,
    plot_igs_station_tec,
)

from demo_verification import (
    ISR_LAT, ISR_LON, ISR_LON_W,
    VERIF_LAT_MIN, VERIF_LAT_MAX, VERIF_LON_MIN, VERIF_LON_MAX,
    HALF_LAT, HALF_LON,
    filter_to_verif_region,
    assign_orbit_groups,
    load_isr_profiles,
    millstone_vertex_idx,
    compute_isr_tec,
    _patched_region_bounding_box,
    _isr_profiles_for_patch,
)

# Plotting functions consolidated into plotIonosphereTomography.py. Imported
# back here (top-level, not deferred) so the internal call sites in this
# module keep working as normal module-level references.
from plotIonosphereTomography import (
    _plot_arc_innovation_diagnostic, _plot_covariance_panels_labeled,
    _plot_ekf_param_covariance_panels, plot_kf_enkf_comparison,
)

from Ionosphere_Tomography_Inverter.ionospheric_state import (
    IonosphericState, N_STATE, PARAM_NAMES,
    I_LOG_NMF2, I_HMF2, I_H0, I_GAMMA, I_B0, I_B1, I_LOG_NME, I_HME,
)

try:
    from TEC_model.igs_tec_pipeline import (
        IGSTECPipeline, igs_obs_to_clean_entry, process_igs_station,
    )
    _IGS_PIPELINE_AVAILABLE = True
except ImportError as _igs_err:
    _IGS_PIPELINE_AVAILABLE = False
    print(f"[warn] IGS TEC pipeline not available: {_igs_err}")
from Ionosphere_Tomography_Inverter.observation_operator import (
    ObservationOperator, _ne_profile_ensemble,
)
from Ionosphere_Tomography_Inverter.enkf_update import (
    ParametricEnKF, build_localisation_matrix,
    build_ray_localisation_matrix,
)

# ── patch demo_group region handler (same as demo_verification.py) ─────────────
_demo_group.region_bounding_box = _patched_region_bounding_box

# ── ECEF → geodetic transformer (already defined in demo_group, re-import) ─────
_TRANSFORMER = pyproj.Transformer.from_crs(
    pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
    pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
    always_xy=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# §A  Ray geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_gnss_to_leo_ray(
    gnss_pt_km: np.ndarray,
    leo_pt_km: np.ndarray,
    n_pts: int = 800,
    alt_min_km: float = 80.0,
    alt_max_km: float = 1000.0,
) -> np.ndarray:
    """
    Build a single ray trajectory for one epoch by sampling the straight-line
    path from one GNSS satellite position to one LEO satellite position.

    Parameters
    ----------
    gnss_pt_km : (3,)  GNSS satellite ECEF position in km.
    leo_pt_km  : (3,)  LEO satellite ECEF position in km.
    n_pts      : Number of sample points along the path (before altitude filter).
    alt_min_km : Minimum altitude to retain (km).
    alt_max_km : Maximum altitude to retain (km).

    Returns
    -------
    traj : ndarray, shape (n_iono, 3) — [lat_deg, lon_deg, alt_km] for every
           point that falls within the ionospheric altitude band.  At least one
           point is always returned (the tangent point closest approach).

    Notes
    -----
    The parametric line  p(t) = gnss + t*(leo - gnss),  t ∈ [0, 1]  runs from
    GNSS to LEO in Cartesian ECEF.  Converting each sample point to geodetic
    coordinates and filtering to the ionospheric band gives the true
    line-of-sight integration path used by the observation operator.
    """
    t_vals = np.linspace(0.0, 1.0, n_pts)                       # (n_pts,)
    # Broadcast: pts[k] = gnss + t_k * (leo - gnss)
    pts = gnss_pt_km[:, np.newaxis] + t_vals * (               # (3, n_pts)
        leo_pt_km[:, np.newaxis] - gnss_pt_km[:, np.newaxis]
    )

    lons, lats, alts_m = _TRANSFORMER.transform(
        pts[0] * 1e3, pts[1] * 1e3, pts[2] * 1e3,
    )
    alts_km = alts_m / 1000.0

    mask = (alts_km >= alt_min_km) & (alts_km <= alt_max_km)
    if not np.any(mask):
        # Entire ray is outside the band (e.g. geometry error): return the
        # single minimum-altitude point so the integration at least runs.
        best = int(np.argmin(np.abs(alts_km - alt_min_km)))
        return np.array([[lats[best], lons[best], max(alts_km[best], alt_min_km)]])

    return np.column_stack([lats[mask], lons[mask], alts_km[mask]])


def _tangent_latlon_single(
    gnss_pt_km: np.ndarray,
    leo_pt_km: np.ndarray,
) -> tuple[float, float]:
    """
    Return the geodetic (lat, lon) of the closest-approach (tangent) point
    for one GNSS→LEO ray.  Both inputs are ECEF in km, shape (3,).
    """
    d     = leo_pt_km - gnss_pt_km
    denom = float(np.dot(d, d))
    if denom < 1e-12:
        r   = float(np.linalg.norm(leo_pt_km))
        lat = float(np.degrees(np.arcsin(np.clip(leo_pt_km[2] / r, -1, 1))))
        lon = float(np.degrees(np.arctan2(leo_pt_km[1], leo_pt_km[0])))
        return lat, lon
    t_tp  = -float(np.dot(gnss_pt_km, d)) / denom
    tp    = gnss_pt_km + np.clip(t_tp, 0.0, 1.0) * d
    r     = float(np.linalg.norm(tp))
    lat   = float(np.degrees(np.arcsin(np.clip(tp[2] / r, -1, 1))))
    lon   = float(np.degrees(np.arctan2(tp[1], tp[0])))
    return lat, lon


def _arc_tangent_point_latlons(
    leo_ecef: np.ndarray,
    gnss_ecef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample tangent-point (lat, lon) for an entire occultation arc.

    Parameters
    ----------
    leo_ecef  : (3, n_samples) LEO positions in km.
    gnss_ecef : (3, n_samples) GNSS positions in km.

    Returns
    -------
    tp_lats : (n_samples,)  tangent-point latitudes (degrees).
    tp_lons : (n_samples,)  tangent-point longitudes (degrees).
    """
    n = leo_ecef.shape[1]
    tp_lats = np.empty(n)
    tp_lons = np.empty(n)
    for i in range(n):
        tp_lats[i], tp_lons[i] = _tangent_latlon_single(
            gnss_ecef[:, i], leo_ecef[:, i]
        )
    return tp_lats, tp_lons


def _arc_representative_tangent(
    leo_ecef: np.ndarray,
    gnss_ecef: np.ndarray,
) -> tuple[float, float]:
    """
    Return (lat, lon) of the tangent point at the TEC-maximum epoch — used as
    the representative pierce-point for localisation weight assignment.

    The TEC-maximum sample is approximated as the epoch whose ray has the
    minimum closest-approach altitude (deepest tangent point in the
    ionosphere), which corresponds to the maximum columnar electron content.
    """
    n = leo_ecef.shape[1]
    tang_alts = np.empty(n)
    for i in range(n):
        d     = leo_ecef[:, i] - gnss_ecef[:, i]
        denom = float(np.dot(d, d))
        if denom < 1e-12:
            tang_alts[i] = float(np.linalg.norm(leo_ecef[:, i])) - 6371.0
            continue
        t_tp = -float(np.dot(gnss_ecef[:, i], d)) / denom
        tp   = gnss_ecef[:, i] + np.clip(t_tp, 0.0, 1.0) * d
        tang_alts[i] = float(np.linalg.norm(tp)) - 6371.0

    # The epoch with the lowest tangent altitude is the TEC-max proxy
    tec_max_idx = int(np.argmin(tang_alts))
    return _tangent_latlon_single(gnss_ecef[:, tec_max_idx], leo_ecef[:, tec_max_idx])


# ─────────────────────────────────────────────────────────────────────────────
# §A1b  WES2 IGS ground-station TEC pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_igs_arcs(
    date:          pd.Timestamp,
    stations:      list,
    cache_dir:     str  = "./Data/IGS_RINEX/",
    rinex_version: int  = 2,
    local_obs:     str  = None,
    local_nav:     str  = None,
    local_dcb:     str  = None,
    use_iri:       bool = False,
    max_rays:      int  = 200,
) -> list:
    """Download and process IGS RINEX data for a cascade of ground stations.

    Tries each station in ``stations`` order (closest to the region first).
    Arcs are collected from every station that produces valid data — all valid
    arcs are returned combined so the assimilation benefits from as much
    ground-truth coverage as possible.

    Each station's result is pickled separately so reruns skip already-processed
    stations.

    Parameters
    ----------
    date          : UTC date to process.
    stations      : Ordered list of 4-char IGS station codes, e.g.
                    ``["WES2", "BARH", "HLFX", "FRDN"]``.
    cache_dir     : Directory for RINEX downloads and processed-arc cache files.
    rinex_version : 2 or 3 (default 2 — most IGS stations upload RINEX-2).
    local_obs / local_nav / local_dcb : Optional pre-downloaded file paths
                    (applied to the *first* station only; set per-station if
                    needed by calling this function in a loop instead).
    use_iri       : Compute IRI-2020 baseline STEC for each arc.
    max_rays      : Maximum rays per arc after decimation.

    Returns
    -------
    Combined list of clean-list dicts from all stations that yielded data.
    """
    if not _IGS_PIPELINE_AVAILABLE:
        print("  [IGS] Pipeline not available — skipping ground-station data.")
        return []

    os.makedirs(cache_dir, exist_ok=True)
    date_str = date.strftime("%Y%m%d")

    all_entries: list = []

    for sta in stations:
        cache_path = os.path.join(cache_dir, f"{sta}_{date_str}_clean.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as fh:
                    entries = pickle.load(fh)
                print(f"  [IGS/{sta}] Loaded {len(entries)} arc(s) from cache.")
                all_entries.extend(entries)
                continue
            except Exception as exc:
                print(f"  [IGS/{sta}] Cache load failed ({exc}); re-processing …")

        print(f"  [IGS/{sta}] Running TEC pipeline for {date.date()} …")
        try:
            obs_list = process_igs_station(
                station       = sta,
                date          = date,
                rinex_version = rinex_version,
                cache_dir     = cache_dir,
                use_iri       = use_iri,
                local_obs     = local_obs if sta == stations[0] else None,
                local_nav     = local_nav if sta == stations[0] else None,
                local_dcb     = local_dcb if sta == stations[0] else None,
            )
        except Exception as exc:
            print(f"  [IGS/{sta}] Pipeline failed: {exc}")
            continue

        entries = []
        for obs in obs_list:
            ce = igs_obs_to_clean_entry(obs, max_rays=max_rays)
            if ce is not None:
                entries.append(ce)

        print(f"  [IGS/{sta}] {len(obs_list)} raw arcs → {len(entries)} valid entries")

        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(entries, fh, protocol=4)
        except Exception as exc:
            print(f"  [IGS/{sta}] Cache save failed: {exc}")

        all_entries.extend(entries)

    return all_entries


def _filter_igs_for_window(
    igs_entries:    list,
    time_window:    str,
    window_minutes: int = WINDOW_MINUTES,
) -> list:
    """Filter IGS clean-list entries to a 30-minute KF time window.

    Parameters
    ----------
    igs_entries    : list of clean-list dicts (from ``_load_igs_arcs``).
    time_window    : string key ``"YYYY-MM-DD_HHMM"`` from ``scan_metadata``.
    window_minutes : window width in minutes (same as the KF group width).

    Returns
    -------
    Filtered list of entries whose ``date`` field falls within
    ``[t_centre − window/2, t_centre + window/2)``.
    """
    if not igs_entries:
        return []

    try:
        t_win = _parse_time_window(time_window)   # tz-naive UTC Timestamp
    except Exception as exc:
        print(f"  [WES2] Could not parse time_window '{time_window}': {exc}")
        return []

    dt_half = pd.Timedelta(minutes=window_minutes / 2)
    t_start = t_win - dt_half
    t_end   = t_win + dt_half

    # _parse_time_window returns a tz-naive Timestamp (interpreted as UTC).
    # Strip timezone from arc dates to keep comparisons homogeneous.
    out = []
    for ce in igs_entries:
        arc_date = ce.get("date")
        if arc_date is None:
            continue
        if not isinstance(arc_date, pd.Timestamp):
            arc_date = pd.Timestamp(arc_date)
        if arc_date.tzinfo is not None:
            arc_date = arc_date.tz_convert("UTC").tz_localize(None)
        if t_start <= arc_date < t_end:
            out.append(ce)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# §A2  IDW grid-point weights for smooth forward-model TEC profiles
# ─────────────────────────────────────────────────────────────────────────────

def _idw_weights(
    tp_lats: np.ndarray,
    tp_lons: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    k: int = 4,
    power: float = 2.0,
    min_dist_km: float = 1.0,
) -> np.ndarray:
    """
    Inverse-distance weights from each ray's tangent point to the k nearest
    grid points.  Prevents discontinuous TEC profiles caused by hard Voronoi
    boundaries in the nearest-neighbour assignment.

    Parameters
    ----------
    tp_lats, tp_lons : (n_rays,)  tangent-point positions
    grid_lats, grid_lons : (n_grid,)  grid-point positions
    k : number of nearest grid points to blend
    power : IDW exponent (2 = standard inverse-square)
    min_dist_km : distance floor to avoid division by zero for on-node rays

    Returns
    -------
    W : ndarray, shape (n_rays, n_grid)  — rows sum to 1.
    """
    from scipy.spatial import cKDTree

    k = min(k, len(grid_lats))
    pts_grid = np.column_stack([grid_lats, grid_lons])
    pts_ray  = np.column_stack([tp_lats,   tp_lons  ])

    # Convert degree separation to approximate km (1° ≈ 111 km)
    tree = cKDTree(pts_grid * 111.0)
    dists_deg, idxs = tree.query(pts_ray * 111.0, k=k)   # (n_rays, k)

    dists_km = np.maximum(dists_deg, min_dist_km)
    w_k = 1.0 / dists_km ** power                        # (n_rays, k)
    w_k /= w_k.sum(axis=1, keepdims=True)                # normalise rows

    n_rays = len(tp_lats)
    n_grid = len(grid_lats)
    W = np.zeros((n_rays, n_grid), dtype=float)
    W[np.arange(n_rays)[:, None], idxs] = w_k
    return W


# ─────────────────────────────────────────────────────────────────────────────
# §B  Fit IRI parametric state to a Ne profile
# ─────────────────────────────────────────────────────────────────────────────

def _fit_iri_params(
    ne_profile: np.ndarray,
    alt_grid: np.ndarray,
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
    _lo = np.array([8.0, 100.0, 5.0, 0.05, 20.0, 0.5, 7.0, 80.0])
    _hi = np.array([13.0, 600.0, 300.0, 2.0, 300.0, 4.0, 12.0, 180.0])

    def _residual(x):
        p = np.minimum(np.maximum(x, _lo), _hi)
        params_lin = np.array([
            10.0 ** p[0], p[1], p[2], p[3], p[4], p[5], 10.0 ** p[6], p[7]
        ])[:, np.newaxis]  # (8, 1)
        ne_model = _ne_profile_ensemble(alts, params_lin)[:, 0]
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
    x0[0] = np.clip(x0[0], 9.0, 13.0)    # log10(NmF2)
    x0[1] = np.clip(x0[1], 100.0, 600.0) # hmF2
    x0[2] = np.clip(x0[2], 10.0, 300.0)  # H0
    x0[3] = np.clip(x0[3], 0.05, 2.0)    # gamma
    x0[4] = np.clip(x0[4], 20.0, 300.0)  # B0
    x0[5] = np.clip(x0[5], 0.5, 4.0)     # B1
    x0[6] = np.clip(x0[6], 7.0, 12.0)    # log10(NmE)
    x0[7] = np.clip(x0[7], 80.0, 180.0)  # hmE

    return x0   # (N_STATE,) in log-space convention


def _parametric_to_edp(
    state: IonosphericState,
    ensemble: np.ndarray,
    alt_grid: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the ensemble-mean Ne profile at every grid point.

    Returns
    -------
    ne_3d : ndarray, shape (n_alt, n_grid)   electron density in m⁻³.
    """
    params_lin = state.to_linear_densities(ensemble)     # (N_STATE, n_grid, n_members)
    mean_lin   = params_lin.mean(axis=2)                  # (N_STATE, n_grid)

    n_alt  = len(alt_grid)
    n_grid = state.n_grid_points

    # Vectorise: treat grid points as "members" for profile evaluation
    ne_3d = _ne_profile_ensemble(alt_grid, mean_lin)      # (n_alt, n_grid)
    return np.maximum(ne_3d, 0.0)


def _parametric_to_edp_ensemble(
    state: IonosphericState,
    ensemble: np.ndarray,
    alt_grid: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the Ne profile for every ensemble member at every grid point.

    Returns
    -------
    ne_ens : ndarray, shape (n_alt, n_grid, n_members)   electron density in m⁻³.
    """
    params_lin = state.to_linear_densities(ensemble)  # (N_STATE, n_grid, n_members)
    n_alt      = len(alt_grid)
    n_grid     = state.n_grid_points
    n_members  = ensemble.shape[2]

    ne_ens = np.empty((n_alt, n_grid, n_members), dtype=float)
    for m in range(n_members):
        ne_ens[:, :, m] = np.maximum(
            _ne_profile_ensemble(alt_grid, params_lin[:, :, m]), 0.0
        )
    return ne_ens


def _fit_log_rmse(
    params_log: np.ndarray,
    ne_profile: np.ndarray,
    alt_grid: np.ndarray,
) -> float:
    """
    Log₁₀-RMSE between a fitted parametric profile and the original Ne(h).

    Parameters
    ----------
    params_log : (N_STATE,)  fitted parameters in log/linear convention.
    ne_profile : (n_alt,)   original Ne in m⁻³.
    alt_grid   : (n_alt,)   altitude grid in km.

    Returns
    -------
    float : sqrt( mean( (log10(Ne_fit) - log10(Ne_orig))² ) )
    """
    lin = params_log.copy()
    lin[I_LOG_NMF2] = 10.0 ** params_log[I_LOG_NMF2]
    lin[I_LOG_NME]  = 10.0 ** params_log[I_LOG_NME]
    ne_fit = _ne_profile_ensemble(alt_grid, lin[:, np.newaxis])[:, 0]
    ne_fit = np.maximum(ne_fit, 1.0)
    ne_ref = np.maximum(ne_profile, 1.0)
    return float(np.sqrt(np.nanmean((np.log10(ne_fit) - np.log10(ne_ref)) ** 2)))


# Feature indices matching EDPSamples.FEATURE_LABEL order:
# ("nmf2","hmf2","nmf1","hmf1","nme","hme","nmd","hmd","hhalf","b0","valley_base","valley_top","b1")
_FEAT_NMF2 = 0
_FEAT_HMF2 = 1
_FEAT_NME  = 4
_FEAT_HME  = 5
_FEAT_B0   = 9
_FEAT_B1   = 12


def _solar_sampling_df(time_in: pd.Timestamp) -> pd.DataFrame:
    """
    Build the single-row sampling_parameters DataFrame for a given time using
    the IRI_Sample_Inputs solar index lookup.  Call once per time window and
    reuse across grid points.
    """
    inp = IRI_Sample_Inputs(time_in.strftime("%Y-%m-%d %H:%M:%S"))
    return pd.DataFrame([{
        "hour": float(time_in.hour) + time_in.minute / 60.0,
        "f107": float(inp.apf107["f107"][inp.current_idx_f107]),
        "ap":   float(inp.apf107["iapda"][inp.current_idx_f107]),
        "ig12": float(inp.ig_rz["ig"][inp.current_idx_igrz]),
        "rz12": float(inp.ig_rz["rz"][inp.current_idx_igrz]),
    }])


def _get_iri_edp_and_features(
    time_in: pd.Timestamp,
    lat: float,
    lon: float,
    alt_grid: np.ndarray,
    sampling_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Call the Fortran IRI2020 driver for a single point and time.

    Parameters
    ----------
    sampling_df : pre-built solar-index DataFrame from _solar_sampling_df.
                  When None it is built internally (one extra lookup per call).

    Returns
    -------
    ne_profile  : (n_alt,) electron density in m⁻³
    feature_vec : (13,)    IRI scalar outputs ordered by EDPSamples.FEATURE_LABEL
    """
    if sampling_df is None:
        sampling_df = _solar_sampling_df(time_in)

    geoloc = np.array([[lon, lat]])
    edps, feature_edps = get_IRI2020_EDP(
        DateTime            = time_in.strftime("%Y-%m-%d %H:%M:%S"),
        altitude            = alt_grid,
        geolocation         = geoloc,
        sampling_parameters = sampling_df,
    )
    ne_profile  = edps[:, 0, 0].astype(float)
    feature_vec = feature_edps[:, 0, 0].astype(float)
    return ne_profile, feature_vec


def _get_iri_edp_and_features_batch(
    time_in: pd.Timestamp,
    lats: np.ndarray,
    lons: np.ndarray,
    alt_grid: np.ndarray,
    sampling_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Call the Fortran IRI2020 driver for all grid points in a single subprocess.

    Parameters
    ----------
    lats, lons  : (n_geo,) geographic coordinates
    sampling_df : pre-built from _solar_sampling_df; built internally if None.

    Returns
    -------
    ne_profiles  : (n_alt, n_geo) electron density in m⁻³
    feature_vecs : (13, n_geo)    IRI scalar outputs per grid point
    """
    if sampling_df is None:
        sampling_df = _solar_sampling_df(time_in)

    geoloc = np.column_stack([lons, lats])   # (n_geo, 2): col0=lon, col1=lat
    edps, feature_edps = get_IRI2020_EDP(
        DateTime            = time_in.strftime("%Y-%m-%d %H:%M:%S"),
        altitude            = alt_grid,
        geolocation         = geoloc,
        sampling_parameters = sampling_df,
    )
    # edps shape: (n_alt, n_geo, 1);  feature_edps: (13, n_geo, 1)
    ne_profiles  = edps[:, :, 0].astype(float)
    feature_vecs = feature_edps[:, :, 0].astype(float)
    return ne_profiles, feature_vecs


def _get_pyiri_params(
    time_in: pd.Timestamp,
    lat: float,
    lon: float,
    alt_grid: np.ndarray,
    f107: float,
) -> dict:
    """
    Call PyIRI for a single point to get the topside and bottomside thickness
    parameters that the Fortran driver does not expose.

    Returns a dict with scalar floats:
        B_top : topside half-thickness at hmF2 (km) — used as H0
        B_bot : bottomside half-thickness (km) — cross-check for B0
        NmF2, hmF2, NmE, hmE from PyIRI
        EDP   : (n_alt,) PyIRI electron density profile in m⁻³
    """
    import PyIRI
    import PyIRI.main_library as lib

    F2, F1, E, _Es, _sun, _mag, EDP = lib.IRI_density_1day(
        time_in.year, time_in.month, time_in.day,
        np.array([float(time_in.hour) + time_in.minute / 60.0]),
        np.array([lon]),
        np.array([lat]),
        np.asarray(alt_grid, dtype=float),
        f107,
        PyIRI.coeff_dir,
    )
    return {
        "B_top": float(F2["B_top"].flat[0]),   # topside scale height at hmF2 (km)
        "B_bot": float(F2["B_bot"].flat[0]),   # bottomside thickness (km)
        "NmF2":  float(F2["Nm"].flat[0]),
        "hmF2":  float(F2["hm"].flat[0]),
        "NmE":   float(E["Nm"].flat[0]),
        "hmE":   float(E["hm"].flat[0]),
        "EDP":   np.asarray(EDP).reshape(-1).astype(float),
    }


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


def _state_from_iri_direct(
    ne_profile: np.ndarray,
    feature_vec: np.ndarray,
    alt_grid: np.ndarray,
) -> np.ndarray:
    """
    Build the 8-parameter state vector from IRI Fortran driver outputs.

    Sources for each parameter
    --------------------------
    NmF2, hmF2, B0, B1, NmE, hmE : IRI feature scalars (feature_vec)
    H0    : seeded from the topside half-power point of the ne profile
            (Epstein z=1.317 → H0 ≈ Δh_half / 1.317), then jointly refined
            with gamma via a 2-parameter L-BFGS-B fit against the valid
            topside IRI ne.
    gamma : jointly fitted with H0 (see above).

    Parameters
    ----------
    ne_profile  : (n_alt,) electron density in m⁻³ on alt_grid.
    feature_vec : (13,) from feature_edps[:, g, s] (EDPSamples.FEATURE_LABEL).
    alt_grid    : (n_alt,) altitude grid in km.

    Returns
    -------
    params_log : (N_STATE,) [log10(NmF2), hmF2, H0, gamma, B0, B1, log10(NmE), hmE]
    """
    nm_f2 = max(float(feature_vec[_FEAT_NMF2]), 1.0)
    hm_f2 = float(feature_vec[_FEAT_HMF2])
    b0    = float(feature_vec[_FEAT_B0])
    b1    = float(feature_vec[_FEAT_B1])
    nm_e  = float(feature_vec[_FEAT_NME])
    hm_e  = float(feature_vec[_FEAT_HME])

    hm_f2 = np.clip(hm_f2, 100.0, 600.0)
    b0    = np.clip(b0,    20.0,  300.0)
    b1    = np.clip(b1,    0.5,   4.0)
    nm_e  = max(nm_e, 1.0) if nm_e > 0 else max(nm_f2 * 0.05, 1e9)
    hm_e  = np.clip(hm_e, 80.0, 180.0) if hm_e > 0 else 110.0

    ne   = np.asarray(ne_profile)
    alts = np.asarray(alt_grid)
    top_mask = alts > hm_f2

    # ── H0 seed from topside half-power point (no external call needed) ──────
    H0_seed = _h0_seed_from_profile(ne[top_mask], alts[top_mask], nm_f2, hm_f2)

    # ── Jointly fit H0 and gamma from the valid topside ne points ────────────
    H0, gamma = H0_seed, 0.5
    if top_mask.sum() >= 5:
        try:
            H0, gamma = _fit_topside_H0_gamma(
                ne[top_mask], alts[top_mask], nm_f2, hm_f2, H0_seed
            )
        except Exception:
            pass

    # ── Refit B0, B1 to the IRI bottomside in the model's own basis ──────────
    # The IRI feature B0/B1 parameterise IRI's bottomside, not the Epstein
    # exp(-x**B1)/cosh(x) form _ne_profile_ensemble uses; copying them leaves a
    # large bottomside error (audit: F-bottom log10-RMSE ~0.15 -> ~0.04 when
    # refitted). Fit over the F-region bottomside ONLY (150 km -> hmF2): the
    # 100-150 km E/valley band is F1-filled in IRI but structurally unrepresent-
    # able by our 4-region model, and including it distorts the fit toward an
    # over-broad bottomside (B1 collapses, near-peak shape degrades).
    bot_mask = (alts > 150.0) & (alts < hm_f2)
    if bot_mask.sum() >= 4:
        try:
            b0, b1 = _fit_bottomside_B0_B1(
                ne[bot_mask], alts[bot_mask], nm_f2, hm_f2, b0, b1
            )
        except Exception:
            pass

    params_log = np.array([
        np.log10(nm_f2),
        hm_f2,
        H0,
        gamma,
        b0,
        b1,
        np.log10(nm_e),
        hm_e,
    ])

    params_log[0] = np.clip(params_log[0], 9.0,   13.0)
    params_log[1] = np.clip(params_log[1], 100.0, 600.0)
    params_log[2] = np.clip(params_log[2], 10.0,  200.0)
    params_log[3] = np.clip(params_log[3], 0.01,  2.0)
    params_log[4] = np.clip(params_log[4], 20.0,  300.0)
    params_log[5] = np.clip(params_log[5], 0.5,   4.0)
    params_log[6] = np.clip(params_log[6], 7.0,   12.0)
    params_log[7] = np.clip(params_log[7], 80.0,  180.0)

    return params_log


def plot_iri_param_fit_diagnostics(
    prior_edp:  np.ndarray,
    mean_state: np.ndarray,
    alt_grid:   np.ndarray,
    grid_lats:  np.ndarray,
    grid_lons:  np.ndarray,
    save_dir:   str,
    group_key:  str,
    mh_idx:     int | None = None,
    max_profiles: int = 40,
) -> str:
    """
    Three-panel diagnostic figure comparing the IRI prior Ne(h) profile at
    every grid point against the Ne(h) profile re-synthesised from the fitted
    8-parameter state vector.

    Panel layout
    ────────────
    Left   — Profile overlay: prior (dashed) vs fitted (solid) for every grid
             point, colour-coded by index.  The Millstone Hill vertex (if
             supplied) is highlighted in red.  Log x-axis.
    Centre — Absolute error |Ne_fit − Ne_prior| vs altitude for every grid
             point (thin, semi-transparent) plus the mean across all grid
             points (bold black line).  Log x-axis.
    Right  — Relative error 100·|Ne_fit − Ne_prior|/Ne_prior (%) vs altitude.
             A vertical dashed line at 10 % marks the acceptable threshold.

    A summary text box reports the mean log₁₀-RMSE and the worst-fitting
    grid point so that poorly-converged fits are immediately obvious.

    Parameters
    ----------
    prior_edp   : (n_alt, n_geo)  IRI prior electron density in m⁻³.
    mean_state  : (N_STATE, n_geo) fitted parameters in mixed log/linear space.
    alt_grid    : (n_alt,)  altitude grid in km.
    grid_lats/lons : (n_geo,)  grid-point coordinates for the legend.
    save_dir    : output directory.
    group_key   : used for the figure title and filename.
    mh_idx      : grid-point index of the Millstone Hill vertex (highlighted).
    max_profiles: cap the number of light profile lines to avoid over-crowding.

    Returns
    -------
    str : path to the saved PNG.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_alt, n_geo = prior_edp.shape

    # ── Reconstruct fitted Ne profiles ───────────────────────────────────────
    # Convert the fitted mean_state (log/linear convention) to linear densities
    # for all grid points simultaneously, then evaluate the IRI profile.
    lin_params = mean_state.copy()
    lin_params[I_LOG_NMF2] = 10.0 ** mean_state[I_LOG_NMF2]
    lin_params[I_LOG_NME]  = 10.0 ** mean_state[I_LOG_NME]
    # lin_params is now (N_STATE, n_geo) with density rows in m⁻³
    fitted_edp = _ne_profile_ensemble(alt_grid, lin_params)   # (n_alt, n_geo)
    fitted_edp = np.maximum(fitted_edp, 1.0)

    prior_safe  = np.maximum(prior_edp, 1.0)

    # ── Per-grid-point log-RMSE ───────────────────────────────────────────────
    log_rmse = np.sqrt(
        np.nanmean(
            (np.log10(fitted_edp) - np.log10(prior_safe)) ** 2,
            axis=0,
        )
    )   # (n_geo,)
    mean_log_rmse = float(np.nanmean(log_rmse))
    worst_gp      = int(np.nanargmax(log_rmse))

    # ── Errors ───────────────────────────────────────────────────────────────
    abs_err = np.abs(fitted_edp - prior_safe)               # (n_alt, n_geo)
    rel_err = 100.0 * abs_err / prior_safe                  # (n_alt, n_geo)  %
    mean_abs_err = np.nanmean(abs_err, axis=1)              # (n_alt,)
    mean_rel_err = np.nanmean(rel_err, axis=1)              # (n_alt,)

    # ── Colour map — one colour per grid point ────────────────────────────────
    cmap_gp  = mpl.colormaps.get_cmap("viridis")
    gp_cols  = [cmap_gp(i / max(n_geo - 1, 1)) for i in range(n_geo)]

    # Limit the number of light overlay lines to avoid over-crowding
    step  = max(1, n_geo // max_profiles)
    shown = list(range(0, n_geo, step))

    ne_fmt = ScalarFormatter(useMathText=True)
    ne_fmt.set_powerlimits((-2, 2))

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, (ax_prof, ax_abs, ax_rel) = plt.subplots(
        1, 3, figsize=(18, 8), sharey=True
    )
    safe_key = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    fig.suptitle(
        f"IRI Parameter Fit Diagnostics — {group_key}\n"
        f"Mean log₁₀-RMSE: {mean_log_rmse:.4f}   "
        f"Worst grid point: {worst_gp} "
        f"({grid_lats[worst_gp]:.1f}°N, {grid_lons[worst_gp]:.1f}°E)  "
        f"log₁₀-RMSE = {log_rmse[worst_gp]:.4f}",
        fontsize=12, y=1.01,
    )

    # ── Panel 1: Profile overlay ──────────────────────────────────────────────
    for g in shown:
        col = gp_cols[g]
        lbl = f"gp{g} ({grid_lats[g]:.1f}°N)"
        ax_prof.plot(prior_safe[:, g],  alt_grid,
                     color=col, lw=1.2, ls="--", alpha=0.55)
        ax_prof.plot(fitted_edp[:, g],  alt_grid,
                     color=col, lw=1.2, ls="-",  alpha=0.55)

    # Highlight Millstone Hill vertex if supplied
    if mh_idx is not None and 0 <= mh_idx < n_geo:
        ax_prof.plot(prior_safe[:, mh_idx],  alt_grid,
                     color="crimson", lw=2.2, ls="--",
                     label=f"MH prior (gp{mh_idx})", zorder=6)
        ax_prof.plot(fitted_edp[:, mh_idx],  alt_grid,
                     color="crimson", lw=2.2, ls="-",
                     label=f"MH fitted (gp{mh_idx})", zorder=6)

    # Highlight worst-fitting grid point
    ax_prof.plot(prior_safe[:, worst_gp], alt_grid,
                 color="darkorange", lw=2.0, ls="--",
                 label=f"Worst prior (gp{worst_gp})", zorder=5)
    ax_prof.plot(fitted_edp[:, worst_gp], alt_grid,
                 color="darkorange", lw=2.0, ls="-",
                 label=f"Worst fitted (gp{worst_gp})", zorder=5)

    # Style legend entries for line types
    style_handles = [
        Line2D([0], [0], color="gray", lw=1.5, ls="--", label="IRI prior"),
        Line2D([0], [0], color="gray", lw=1.5, ls="-",  label="Fitted params"),
    ]
    ax_prof.legend(
        handles=style_handles + (
            [Line2D([0], [0], color="crimson",    lw=2.2, ls="-",  label=f"MH (gp{mh_idx})"),
             Line2D([0], [0], color="darkorange", lw=2.0, ls="-",  label=f"Worst (gp{worst_gp})")]
            if mh_idx is not None else
            [Line2D([0], [0], color="darkorange", lw=2.0, ls="-",  label=f"Worst (gp{worst_gp})")]
        ),
        fontsize=8, loc="upper right", framealpha=0.85,
    )
    ax_prof.set_xscale("log")
    ax_prof.xaxis.set_major_formatter(ne_fmt)
    ax_prof.set_xlabel("Electron Density (m⁻³)", fontsize=10)
    ax_prof.set_ylabel("Altitude (km)", fontsize=10)
    ax_prof.set_title("Prior vs. Fitted Ne Profile", fontsize=10)
    ax_prof.grid(True, alpha=0.3, ls=":")

    # Colourbar for grid-point index
    sm = mpl.cm.ScalarMappable(
        cmap=cmap_gp, norm=mpl.colors.Normalize(vmin=0, vmax=n_geo - 1)
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax_prof, label="Grid-point index",
                 fraction=0.046, pad=0.04)

    # ── Panel 2: Absolute error ───────────────────────────────────────────────
    for g in shown:
        ax_abs.plot(abs_err[:, g], alt_grid,
                    color=gp_cols[g], lw=0.8, alpha=0.35)
    ax_abs.plot(mean_abs_err, alt_grid,
                color="black", lw=2.5, label="Mean |error|", zorder=5)
    if mh_idx is not None:
        ax_abs.plot(abs_err[:, mh_idx], alt_grid,
                    color="crimson", lw=2.0, label=f"MH (gp{mh_idx})", zorder=4)
    ax_abs.plot(abs_err[:, worst_gp], alt_grid,
                color="darkorange", lw=2.0, label=f"Worst (gp{worst_gp})", zorder=4)

    ax_abs.set_xscale("log")
    ax_abs.xaxis.set_major_formatter(ne_fmt)
    ax_abs.set_xlabel("|Ne_fit − Ne_prior|  (m⁻³)", fontsize=10)
    ax_abs.set_title("Absolute Fit Error vs. Altitude", fontsize=10)
    ax_abs.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_abs.grid(True, alpha=0.3, ls=":")

    # ── Panel 3: Relative error ───────────────────────────────────────────────
    for g in shown:
        ax_rel.plot(rel_err[:, g], alt_grid,
                    color=gp_cols[g], lw=0.8, alpha=0.35)
    ax_rel.plot(mean_rel_err, alt_grid,
                color="black", lw=2.5, label="Mean rel. error", zorder=5)
    if mh_idx is not None:
        ax_rel.plot(rel_err[:, mh_idx], alt_grid,
                    color="crimson", lw=2.0, label=f"MH (gp{mh_idx})", zorder=4)
    ax_rel.plot(rel_err[:, worst_gp], alt_grid,
                color="darkorange", lw=2.0, label=f"Worst (gp{worst_gp})", zorder=4)

    ax_rel.axvline(10.0, color="red", lw=1.2, ls=":", alpha=0.7,
                   label="10 % threshold")
    ax_rel.axvline(30.0, color="darkred", lw=1.0, ls=":", alpha=0.5,
                   label="30 % threshold")
    ax_rel.set_xlabel("Relative Error (%)", fontsize=10)
    ax_rel.set_title("Relative Fit Error vs. Altitude", fontsize=10)
    ax_rel.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_rel.grid(True, alpha=0.3, ls=":")

    # ── Per-panel log-RMSE annotation box ────────────────────────────────────
    rmse_text = (
        f"log₁₀-RMSE\n"
        + "\n".join(
            f"  gp{g}: {log_rmse[g]:.4f}"
            for g in sorted(range(n_geo), key=lambda x: -log_rmse[x])[:min(6, n_geo)]
        )
    )
    ax_rel.text(
        0.97, 0.03, rmse_text, transform=ax_rel.transAxes,
        fontsize=7, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.80),
    )

    # ── Shared y-axis limits ──────────────────────────────────────────────────
    ax_prof.set_ylim(0, float(alt_grid[-1]) + 20)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = os.path.join(save_dir, f"iri_fit_diag_{safe_key}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  IRI fit diagnostic plot saved → {plot_path}")
    return plot_path


def _spatial_corr_matrix(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    corr_length_km: float,
    nugget: float = 1e-6,
) -> np.ndarray:
    """
    Isotropic exponential spatial correlation matrix for the grid points.

        C[i, j] = exp( −d(i,j) / corr_length_km )

    Parameters
    ----------
    grid_lats, grid_lons : (n_grid,)
    corr_length_km : e-folding distance in km.  A value of ~300–500 km is
        typical for mid-latitude ionospheric variability.
    nugget : small diagonal regularisation added for numerical stability.

    Returns
    -------
    C : ndarray, shape (n_grid, n_grid) — symmetric, PD, diagonal = 1.
    """
    from Ionosphere_Tomography_Inverter.enkf_update import _haversine_km

    n = len(grid_lats)
    dist = _haversine_km(
        grid_lats[:, np.newaxis], grid_lons[:, np.newaxis],
        grid_lats[np.newaxis, :], grid_lons[np.newaxis, :],
    )  # (n_grid, n_grid)

    C = np.exp(-dist / corr_length_km)
    C += nugget * np.eye(n)           # ensure positive-definiteness
    C /= C.diagonal()[:, np.newaxis]  # keep diagonal exactly 1
    return C


def _covariance_from_edp_samples(
    eds_occ,
    alt_grid: np.ndarray,
    min_samples: int = 8,
    nugget: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the EnKF background covariance from the IRI EDP ensemble using
    exactly the same STATISTIC as the standard KF's Ne-space covariance:

        P_grid = np.cov(edps_flat)   # (n_height*n_geo, n_height*n_geo)
        C_spatial = P_grid.reshape(n_h,n_geo,n_h,n_geo).mean(axis=(0,2))  # marginal

    conceptually -- C_spatial is the (n_geo, n_geo) mean of P_grid over
    altitude pairs, normalised to Pearson r. This is identical in spirit to
    how the KF encodes horizontal structure through its full Ne-space P_grid.
    The implementation, however, never materialises P_grid: that dense matrix
    is O((n_height*n_geo)^2) (e.g. 77.6 GiB at n_height=55, n_geo=1855) even
    though only its altitude-marginalised (n_geo, n_geo) block is ever used.
    A closed-form identity (see the Step 1+2 comment below) computes that
    marginal directly in O(n_geo*n_s + n_geo^2), so grid size is no longer
    memory-bound by this function.

    P_b is separately estimated in parameter space by fitting the 8-parameter
    IRI state to each sample and computing np.cov across samples.

    Parameters
    ----------
    eds_occ : EDPSamples
    alt_grid : ndarray — altitude grid in km.
    min_samples : int  — fallback threshold.
    nugget : float     — diagonal regularisation for C_spatial.

    Returns
    -------
    P_b : ndarray, shape (N_STATE, N_STATE)
    C_spatial : ndarray, shape (n_geo, n_geo) — symmetric, PD, diagonal = 1.
    """
    physical_floor = 1e8   # m^-3, same floor the KF applies before np.cov

    ne_all   = np.asarray(eds_occ.edps,         dtype=float)  # (n_height, n_geo, n_s)
    feat_all = np.asarray(eds_occ.feature_edps, dtype=float)  # (n_feat,   n_geo, n_s)
    alt_eds  = np.asarray(eds_occ.altitude,     dtype=float)  # (n_height,)

    n_height, n_geo, n_s = ne_all.shape

    if n_s < min_samples:
        print(f"  [EnKF] Only {n_s} EDP samples — falling back to default covariance.")
        return _default_background_covariance(), np.eye(n_geo)

    print(f"  [EnKF] Estimating background covariance from {n_s} EDP samples "
          f"x {n_geo} grid points ...")

    # ── Step 1+2: C_spatial — Ne-space sample covariance, marginalised over
    # altitude pairs — WITHOUT ever forming the dense (n_height*n_geo)^2 Ne-space
    # covariance matrix the KF conceptually uses (P_grid = np.cov(edps_flat)).
    # That matrix is (n_height*n_geo)^2 * 8 bytes -- e.g. 77.6 GiB at n_height=55,
    # n_geo=1855 -- yet the only thing ever extracted from it is the (n_geo,n_geo)
    # mean over altitude-pairs: cov_geo[g1,g2] = mean_{h1,h2} Cov(ne[h1,g1,:],
    # ne[h2,g2,:]). That mean has a closed form that skips the (n_h*n_geo)^2
    # intermediate entirely:
    #   Xc          = ne_all - mean_over_samples(ne_all)            (n_h,n_geo,n_s)
    #   A[g,s]      = sum_h Xc[h,g,s]                                (n_geo,n_s)
    #   cov_geo     = (A @ A.T) / (n_height^2 * (n_s - 1))           (n_geo,n_geo)
    # Verified numerically identical (to float64 precision) to the old
    # P_grid.reshape(n_h,n_geo,n_h,n_geo).mean(axis=(0,2)) result. Memory drops
    # from O((n_height*n_geo)^2) to O(n_geo*n_s) + O(n_geo^2) -- e.g. ~13 MB
    # instead of 77.6 GB at n_geo=1855 -- so the uncapped/large Fibonacci grid no
    # longer OOMs here.
    ne_clip = np.nan_to_num(ne_all, nan=physical_floor)
    ne_clip = np.clip(ne_clip, physical_floor, None)

    if n_geo == 1:
        C_spatial = np.ones((1, 1))
    else:
        Xc      = ne_clip - ne_clip.mean(axis=2, keepdims=True)   # (n_h,n_geo,n_s)
        A       = Xc.sum(axis=0)                                  # (n_geo, n_s)
        cov_geo = (A @ A.T) / (n_height ** 2 * (n_s - 1))          # (n_geo, n_geo)
        std_geo = np.sqrt(np.maximum(np.diag(cov_geo), 0.0))
        outer   = np.outer(std_geo, std_geo)
        with np.errstate(invalid="ignore", divide="ignore"):
            C_spatial = np.where(outer > 0, cov_geo / outer, 0.0)
        C_spatial = np.clip(C_spatial, -1.0, 1.0)
        C_spatial = 0.5 * (C_spatial + C_spatial.T)
        C_spatial += nugget * np.eye(n_geo)
        d = np.sqrt(np.diag(C_spatial))
        C_spatial /= np.outer(d, d)

    # ── Step 3: P_b — parameter-space covariance from the same IRI ensemble ──
    param_ens = np.full((N_STATE, n_geo, n_s), np.nan)
    n_fail = 0
    for g in range(n_geo):
        for s in range(n_s):
            try:
                param_ens[:, g, s] = _state_from_iri_direct(
                    ne_all[:, g, s], feat_all[:, g, s], alt_eds
                )
            except Exception:
                n_fail += 1

    if n_fail:
        print(f"  [EnKF] {n_fail} fit failures — replacing with column means.")
    for g in range(n_geo):
        bad = np.any(np.isnan(param_ens[:, g, :]), axis=0)
        if bad.all():
            param_ens[:, g, :] = np.nan_to_num(
                param_ens[:, g, :], nan=_default_background_covariance().diagonal() ** 0.5
            )
        elif bad.any():
            good_mean = np.nanmean(param_ens[:, g, :], axis=1, keepdims=True)
            param_ens[:, g, bad] = good_mean

    P_b = np.zeros((N_STATE, N_STATE))
    for g in range(n_geo):
        P_b += np.cov(param_ens[:, g, :])
    P_b /= n_geo
    P_b = 0.5 * (P_b + P_b.T)

    # Variance floor: use the full climatological default as the minimum.
    # The 50-sample IRI ensemble is too smooth to reflect real day-to-day
    # variability (σ_NmF2 ≈ 0.075 from IRI vs. ≈ 0.15 from climatology).
    # Clamping to the default ensures the ensemble has enough spread to bridge
    # the typical 20–50 TECU innovations seen in real occultation data.
    P_b_default = _default_background_covariance()
    np.fill_diagonal(P_b, np.maximum(np.diag(P_b), np.diag(P_b_default)))
    P_b += nugget * np.eye(N_STATE)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    sigmas = np.sqrt(np.diag(P_b))
    print(f"  [EnKF] Parameter sigma: " +
          "  ".join(f"{PARAM_NAMES[i]}={sigmas[i]:.3g}" for i in range(N_STATE)))
    if n_geo > 1:
        off_diag = C_spatial[np.triu_indices(n_geo, k=1)]
        print(f"  [EnKF] Spatial corr (Ne-space KF method): "
              f"mean={off_diag.mean():.3f}  max={off_diag.max():.3f}  "
              f"min={off_diag.min():.3f}")

    return P_b, C_spatial


def _default_background_covariance() -> np.ndarray:
    """
    Diagonal empirical background error covariance for the 8-parameter
    IRI state vector.

    Standard deviations are based on typical day-to-day ionospheric
    variability derived from IRI climatology studies:
        log10(NmF2) : ±0.15 (factor ~1.4)
        hmF2        : ±20 km
        H0          : ±15 km
        gamma       : ±0.15
        B0          : ±20 km
        B1          : ±0.3
        log10(NmE)  : ±0.2
        hmE         : ±8 km
    """
    sigma = np.array([0.15, 20.0, 15.0, 0.15, 20.0, 0.3, 0.2, 8.0])
    return np.diag(sigma ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# §B2  Per-arc innovation diagnostic plot  (shared by KF and EnKF)
# ─────────────────────────────────────────────────────────────────────────────

def _arc_stats_from_tec_slices(
    tec_slices: list,
    clean_list: list,
    sat_ids:    list,
) -> dict:
    """
    Compute per-arc innovation statistics for the KF result.

    Parameters
    ----------
    tec_slices : list of dicts, each with 'measured', 'prior_tec', 'post_tec'.
        Typically ``res_kf["joint_tec_slices"]``.
    clean_list : list of arc metadata dicts (same order as tec_slices).
        Each entry has 'LEO' (3, n_rays) and 'GNSS' (3, n_rays) in ECEF km.
    sat_ids    : list of (leo_id, prn_id) tuples aligned with clean_list.

    Returns
    -------
    dict with keys:
        arc_labels      (n_arcs,) list[str]   — PRN strings, e.g. "G15 / LEO3"
        arc_prior_mean  (n_arcs,) ndarray      — mean prior residual per arc
        arc_post_mean   (n_arcs,) ndarray      — mean post  residual per arc
        arc_prior_rmse  (n_arcs,) ndarray      — RMSE prior per arc
        arc_post_rmse   (n_arcs,) ndarray      — RMSE post  per arc
        arc_lats        (n_arcs,) ndarray      — tangent centroid lat
        arc_lons        (n_arcs,) ndarray      — tangent centroid lon
        all_prior       (n_total,) ndarray     — flat prior residuals
        all_post        (n_total,) ndarray     — flat post  residuals
    """
    arc_labels     = []
    arc_prior_mean = []
    arc_post_mean  = []
    arc_prior_rmse = []
    arc_post_rmse  = []
    arc_lats_out   = []
    arc_lons_out   = []
    all_prior_list = []
    all_post_list  = []

    for i, sl in enumerate(tec_slices):
        meas   = np.asarray(sl["measured"],  dtype=float)
        prior  = np.asarray(sl["prior_tec"], dtype=float)
        post   = np.asarray(sl["post_tec"],  dtype=float)
        resid_prior = meas - prior
        resid_post  = meas - post
        all_prior_list.append(resid_prior)
        all_post_list.append(resid_post)

        arc_prior_mean.append(float(np.nanmean(resid_prior)))
        arc_post_mean.append( float(np.nanmean(resid_post)))
        arc_prior_rmse.append(float(np.sqrt(np.nanmean(resid_prior ** 2))))
        arc_post_rmse.append( float(np.sqrt(np.nanmean(resid_post  ** 2))))

        # PRN label from sat_ids; fall back to arc index if missing
        if i < len(sat_ids) and sat_ids[i]:
            _, prn = sat_ids[i]
            arc_labels.append(str(prn))
        else:
            arc_labels.append(f"arc{i:02d}")

        # Geographic centroid: mean tangent-point across all arc samples
        cl = clean_list[i]
        leo_ecef  = np.asarray(cl["LEO"],  dtype=float)   # (3, n_rays)
        gnss_ecef = np.asarray(cl["GNSS"], dtype=float)   # (3, n_rays)
        n_rays    = leo_ecef.shape[1]
        tp_lats_i, tp_lons_i = [], []
        for j in range(n_rays):
            lat_j, lon_j = _tangent_latlon_single(gnss_ecef[:, j], leo_ecef[:, j])
            tp_lats_i.append(lat_j)
            tp_lons_i.append(lon_j)
        arc_lats_out.append(float(np.mean(tp_lats_i)))
        arc_lons_out.append(float(np.mean(tp_lons_i)))

    return dict(
        arc_labels     = arc_labels,
        arc_prior_mean = np.array(arc_prior_mean),
        arc_post_mean  = np.array(arc_post_mean),
        arc_prior_rmse = np.array(arc_prior_rmse),
        arc_post_rmse  = np.array(arc_post_rmse),
        arc_lats       = np.array(arc_lats_out),
        arc_lons       = np.array(arc_lons_out),
        all_prior      = np.concatenate(all_prior_list),
        all_post       = np.concatenate(all_post_list),
    )


# ─────────────────────────────────────────────────────────────────────────────
# §C  Run the parametric EnKF on the data from a processed group result
# ─────────────────────────────────────────────────────────────────────────────

def _run_parametric_enkf(
    res_kf: dict,
    alt_grid: np.ndarray,
    save_dir: str = "./Figures/CompareKF_EnKF/",
    group_key: str = "group",
    n_members: int = 100,
    loc_radius_km: float = 600.0,
    corr_length_km: float = 600.0,
    inflation: float = 1.0,
    sigma_obs_tecu: float = 10.0,
    max_update_step: float = 1.0,
    n_mda_iterations: int = 4,
    max_update_rays_per_arc: int = 200,
) -> dict:
    """
    Run the parametric EnKF on the same observations used by the standard KF,
    and return a result dict with the same shape as process_group's output so
    that it can be passed directly to _plot_group.

    Parameters
    ----------
    res_kf : dict
        Output of process_group / _process_verif_group.
    alt_grid : ndarray
        Altitude grid shared by both filters.
    save_dir : str
        Output directory; the IRI fit diagnostic plot is written here.
    group_key : str
        Group identifier used in figure titles and filenames.
    n_members : int
        EnKF ensemble size.
    loc_radius_km : float
        Gaspari-Cohn half-support radius for ray-path localisation.
        Observations influence grid points within 2×loc_radius_km of the
        closest point on the ray path.
    corr_length_km : float
        Exponential spatial correlation e-folding length for the prior ensemble
        (km).  Adjacent grid points separated by this distance will have ~37%
        correlated perturbations; at 2× the distance they are ~14% correlated.
        Typical mid-latitude ionospheric correlation lengths are 200–500 km.
    inflation : float
        Multiplicative prior inflation.
    sigma_obs_tecu : float
        Assumed observation noise standard deviation (TECU).
    max_update_step : float
        Per-element log-space increment clip in enkf_update.
    n_mda_iterations : int
        Number of ES-MDA iterations (Emerick & Reynolds 2013).  Each step
        uses R_mda = n_mda_iterations * R and re-evaluates the forward model
        at the current state, keeping each linear approximation near-valid.
        Set to 1 to recover the standard single-step EnKF.
    max_update_rays_per_arc : int
        Maximum number of rays per arc used for the EnKF update.  The KF
        already stores at most 200 decimated points per arc in clean_list
        (stride = ceil(n_valid / 200)); setting this to 200 (default) reuses
        exactly those same points.  A uniform-stride sub-decimation is applied
        if the arc still exceeds this limit after the KF decimation.
    Notes
    -----
    C_spatial (from ``_covariance_from_edp_samples``) is used *only* to seed
    the Kronecker prior ensemble via ``generate_ensemble_spatial``.  It plays
    no role in the ES-MDA update step.  Localisation during the update is
    handled exclusively by the Gaspari-Cohn taper on ray–grid distances,
    controlled by ``loc_radius_km``.

    Returns
    -------
    res_enkf : dict with keys matching _plot_group expectations.
    """
    eds_occ    = res_kf["eds_occ"]
    clean_list = res_kf["clean_list"]
    sat_ids    = res_kf.get("sat_ids", [])  # (leo_id, prn_id) per arc
    prior_edp  = res_kf["prior_edp_3d"]    # (n_alt, n_geo)
    verts_geo  = eds_occ.geolocation        # (n_geo, 2): col0=lon, col1=lat

    n_alt   = len(alt_grid)
    n_geo   = verts_geo.shape[0]
    n_occ   = len(clean_list)

    grid_lats = verts_geo[:, 1].astype(float)
    grid_lons = verts_geo[:, 0].astype(float)

    # Parse the centre time and build solar indices once for all grid points
    t_centre   = _parse_time_window(res_kf.get("time_window", group_key))
    sampling_df = _solar_sampling_df(t_centre)

    print(f"  [EnKF] Building IRI state at {n_geo} grid points (batch call) …")
    mean_state = np.zeros((N_STATE, n_geo), dtype=float)
    try:
        ne_all, feat_all = _get_iri_edp_and_features_batch(
            t_centre, grid_lats, grid_lons, alt_grid, sampling_df
        )
        for g in range(n_geo):
            mean_state[:, g] = _state_from_iri_direct(
                ne_all[:, g], feat_all[:, g], alt_grid
            )
            log_rmse_g = _fit_log_rmse(mean_state[:, g], prior_edp[:, g], alt_grid)
            print(f"    gp {g:3d}  ({grid_lats[g]:+6.2f}°N {grid_lons[g]:+7.2f}°E)"
                  f"  log₁₀-RMSE = {log_rmse_g:.4f}"
                  f"  H0={mean_state[2,g]:.1f} km  γ={mean_state[3,g]:.3f}")
    except Exception as _exc:
        print(f"  [EnKF] Batch IRI call failed ({_exc}); falling back to profile fit")
        for g in range(n_geo):
            mean_state[:, g] = _fit_iri_params(prior_edp[:, g], alt_grid)
            log_rmse_g = _fit_log_rmse(mean_state[:, g], prior_edp[:, g], alt_grid)
            print(f"    gp {g:3d}  ({grid_lats[g]:+6.2f}°N {grid_lons[g]:+7.2f}°E)"
                  f"  log₁₀-RMSE = {log_rmse_g:.4f}  [profile fit]")

    # ── Fit diagnostic plot ───────────────────────────────────────────────────
    _mh_idx = millstone_vertex_idx(verts_geo) if n_geo > 1 else 0
    plot_iri_param_fit_diagnostics(
        prior_edp  = prior_edp,
        mean_state = mean_state,
        alt_grid   = alt_grid,
        grid_lats  = grid_lats,
        grid_lons  = grid_lons,
        save_dir   = save_dir,
        group_key  = group_key,
        mh_idx     = _mh_idx,
    )

    # Build background covariance from the same IRI ensemble the KF uses.
    # Falls back to hardcoded defaults only if the sample count is too small.
    P_b, C_s_edp = _covariance_from_edp_samples(eds_occ, alt_grid)

    # C_s is already derived from np.cov(edps_flat) the same way the KF builds
    # P_grid — no further blending needed.
    C_s = C_s_edp

    state = IonosphericState(n_grid_points=n_geo, n_members=n_members)
    if n_geo > 1:
        state.generate_ensemble_spatial(mean_state, P_b, C_s, n_members=n_members)
    else:
        state.generate_ensemble(mean_state, P_b, n_members=n_members)

    # ─────────────────────────────────────────────────────────────────────────
    # Build per-sample ray trajectories
    # ─────────────────────────────────────────────────────────────────────────
    # For each occultation arc we need TWO sets of rays:
    #
    #  (A) Per-sample rays  — one GNSS→LEO integration per epoch, giving the
    #      sTEC vs. tangent-altitude profile used for tec_slices and the
    #      visualisation figures.
    #
    #  (B) Per-arc representative ray  — one ray per occultation (at the
    #      deepest-tangent-point epoch, which is the TEC-maximum proxy) used
    #      for the EnKF state update.  This keeps the observation vector
    #      compact (n_occ scalars) and prevents the Kalman gain from being
    #      dominated by over-counted near-duplicate rays.
    #
    # (A) and (B) share the same IonosphericState so the posterior from (B)
    # is immediately used to re-evaluate (A) post-update.
    # ─────────────────────────────────────────────────────────────────────────
    print(f"  [EnKF] Building per-sample ray trajectories "
          f"for {n_occ} occultation(s) …")

    # ── (A) Per-sample rays (one per epoch) ───────────────────────────────────
    # Outer list index = occultation; inner list index = epoch within that arc.
    per_arc_sample_rays: list[list[np.ndarray]] = []
    per_arc_tp_lats:     list[np.ndarray]       = []  # tangent-point lats per arc
    per_arc_tp_lons:     list[np.ndarray]       = []  # tangent-point lons per arc
    ray_counts: list[int] = []                        # epochs per arc

    # ── (B) Full-profile rays reusing the KF's own decimation ────────────────────
    # clean_list already stores at most 200 points per arc (the same uniform-
    # stride decimation the KF uses).  We sort those points by tangent altitude
    # and, if the arc still exceeds max_update_rays_per_arc, apply one further
    # uniform-stride sub-decimation — matching the KF's own logic exactly.
    rep_rays:          list[np.ndarray] = []
    rep_tp_lats:       list[float]      = []
    rep_tp_lons:       list[float]      = []
    rep_tec_obs:       list[float]      = []
    arc_update_counts: list[int]        = []

    for cl in clean_list:
        leo  = cl["LEO"]    # (3, n_kf_decimated)
        gnss = cl["GNSS"]
        n_s  = leo.shape[1]

        # Sort by tangent altitude (ascending) — same ordering convention as KF
        tang_alts  = np.asarray(cl["tangent_km"], dtype=float)
        sorted_idx = np.argsort(tang_alts)

        # Optional further sub-decimation: ceil(n_s / max_update_rays_per_arc)
        if n_s > max_update_rays_per_arc:
            stride     = int(np.ceil(n_s / max_update_rays_per_arc))
            chosen_pos = list(range(0, len(sorted_idx), stride))
            # Always keep the deepest tangent point (last in sorted order)
            if len(sorted_idx) - 1 not in chosen_pos:
                chosen_pos.append(len(sorted_idx) - 1)
        else:
            chosen_pos = list(range(len(sorted_idx)))

        chosen = [sorted_idx[p] for p in chosen_pos]

        for idx in chosen:
            traj = _build_gnss_to_leo_ray(gnss[:, idx], leo[:, idx])
            tp_lat, tp_lon = _tangent_latlon_single(gnss[:, idx], leo[:, idx])
            rep_rays.append(traj)
            rep_tp_lats.append(tp_lat)
            rep_tp_lons.append(tp_lon)
            rep_tec_obs.append(float(cl["tec"][idx]))

        arc_update_counts.append(len(chosen))

        # ── Per-sample rays (all epochs, for tec_slices & RMSE) ──────────────
        arc_rays   = []
        arc_lats_i = []
        arc_lons_i = []
        for i in range(n_s):
            traj   = _build_gnss_to_leo_ray(gnss[:, i], leo[:, i])
            tp_lat, tp_lon = _tangent_latlon_single(gnss[:, i], leo[:, i])
            arc_rays.append(traj)
            arc_lats_i.append(tp_lat)
            arc_lons_i.append(tp_lon)

        per_arc_sample_rays.append(arc_rays)
        per_arc_tp_lats.append(np.array(arc_lats_i))
        per_arc_tp_lons.append(np.array(arc_lons_i))
        ray_counts.append(n_s)

    rep_tp_lats = np.array(rep_tp_lats)
    rep_tp_lons = np.array(rep_tp_lons)

    # Flatten all per-sample rays + tangent positions for IDW weight calculation
    all_sample_rays: list[np.ndarray] = []
    all_tp_lats:     list[float]      = []
    all_tp_lons:     list[float]      = []
    for arc_rays, arc_lats, arc_lons in zip(
            per_arc_sample_rays, per_arc_tp_lats, per_arc_tp_lons):
        all_sample_rays.extend(arc_rays)
        all_tp_lats.extend(arc_lats.tolist())
        all_tp_lons.extend(arc_lons.tolist())
    all_tp_lats = np.array(all_tp_lats)
    all_tp_lons = np.array(all_tp_lons)
    n_total_samples = len(all_sample_rays)

    # IDW weights: (n_rays, n_grid) — smooth blend across cell boundaries
    _idw_k = min(4, n_geo)
    all_sample_W = _idw_weights(all_tp_lats, all_tp_lons,
                                grid_lats,   grid_lons, k=_idw_k)
    rep_W        = _idw_weights(rep_tp_lats, rep_tp_lons,
                                grid_lats,   grid_lons, k=_idw_k)

    n_update_obs = len(rep_rays)
    avg_levels   = n_update_obs / max(n_occ, 1)
    print(f"  [EnKF] Total integration rays: {n_total_samples} per-sample + "
          f"{n_update_obs} update rays ({n_occ} arcs, ~{avg_levels:.1f} levels/arc, "
          f"max_per_arc={max_update_rays_per_arc})")

    # ── Prior forward model ───────────────────────────────────────────────────
    # (B) One sTEC per arc  — used for EnKF update
    op = ObservationOperator(state, alt_grid)
    Y_rep_prior_ens  = op.compute_stec_ensemble(
        rep_rays, grid_point_weights=rep_W
    )  # (n_occ, n_members)
    Y_rep_prior_mean = Y_rep_prior_ens.mean(axis=1)   # (n_occ,)

    # (A) One sTEC per sample  — used for tec_slices & RMSE diagnostics
    Y_all_prior_ens  = op.compute_stec_ensemble(
        all_sample_rays, grid_point_weights=all_sample_W
    )  # (n_total_samples, n_members)
    Y_all_prior_mean = Y_all_prior_ens.mean(axis=1)   # (n_total_samples,)

    # ── Observations ─────────────────────────────────────────────────────────
    # Use the TEC at the representative epoch (same geometry as rep_rays) so
    # observation and forward model are exactly matched.  Using the arc mean
    # creates a systematic bias because the representative ray is the TEC-peak
    # (deepest tangent point) which is always higher than the arc average.
    y_obs_arc = np.array(rep_tec_obs)   # (n_update_obs,) — decimated profile levels per arc

    # For RMSE diagnostics: concatenated per-sample measured TEC
    y_obs_all = np.concatenate([cl["tec"] for cl in clean_list])

    prior_innovation = y_obs_arc - Y_rep_prior_mean
    print(f"  [EnKF] Prior innovations: "
          f"mean={prior_innovation.mean():.2f}  "
          f"std={prior_innovation.std():.2f}  "
          f"max_abs={np.abs(prior_innovation).max():.2f} TECU")

    # Observation error covariance — diagonal, one entry per update ray
    R = (sigma_obs_tecu ** 2) * np.eye(n_update_obs)

    # ── EnKF update ───────────────────────────────────────────────────────────
    # Build ray-path localisation: weights based on minimum distance from each
    # grid point to any point along the representative ray, not just the tangent.
    if n_geo > 1 and np.isfinite(loc_radius_km) and loc_radius_km > 0:
        L_ray = build_ray_localisation_matrix(
            grid_lats, grid_lons, rep_rays, loc_radius_km
        )
        print(f"  [EnKF] GC ray-path localisation  loc_radius={loc_radius_km:.0f} km")
    else:
        L_ray = None
        print("  [EnKF] No localisation (loc_radius_km not set or single grid point)")

    # Save the prior ensemble before the update so we can build P_prior later
    prior_ensemble_snapshot = state.ensemble.copy()   # (N_STATE, n_geo, n_members)

    # ── Apply prior inflation ONCE before the ES-MDA loop ─────────────────────
    if inflation > 1.0:
        print(f"  [EnKF] Applying prior covariance inflation (factor {inflation})")
        X_mean = state.ensemble.mean(axis=2, keepdims=True)
        state.ensemble = X_mean + (state.ensemble - X_mean) * inflation

    enkf = ParametricEnKF(
        state         = state,
        grid_lats     = grid_lats,
        grid_lons     = grid_lons,
        loc_radius_km = loc_radius_km,
        inflation     = 1.0,  # <-- FIX: Set to 1.0 so it doesn't double-dip inside assimilate()
    )

    # ── Helper: print per-arc mean TEC residual ───────────────────────────────
    def _print_arc_innovations(label: str, inno: np.ndarray) -> None:
        """Print aggregate stats and a per-arc mean residual table."""
        print(f"  [{label}] innovation  "
              f"mean={inno.mean():.2f}  std={inno.std():.2f}  "
              f"max_abs={np.abs(inno).max():.2f} TECU")
        print(f"  [{label}] per-arc mean residual (obs − model, TECU):")
        offset = 0
        for arc_i, cnt in enumerate(arc_update_counts):
            arc_inno = inno[offset : offset + cnt]
            # Prefer constellation PRN from sat_ids; fall back to arc index
            if arc_i < len(sat_ids) and sat_ids[arc_i]:
                _, prn_lbl = sat_ids[arc_i]
            else:
                prn_lbl = f"arc{arc_i:02d}"
            print(f"          arc {arc_i+1:2d} ({prn_lbl:>6s}):  "
                  f"{arc_inno.mean():+7.2f}  (n={cnt})")
            offset += cnt

    # ── Raw single-step EnKF on a frozen prior snapshot ──────────────────────
    # Run one unregularised update (R, not R_mda) on a copy of the prior so we
    # can compare the single-shot correction against the iterated ES-MDA result.
    print(f"\n  [EnKF] Raw single-step EnKF ({n_update_obs} obs, {n_members} members) …")
    raw_state          = IonosphericState(n_geo, n_members)
    raw_state.ensemble = prior_ensemble_snapshot.copy()
    raw_enkf = ParametricEnKF(
        state         = raw_state,
        grid_lats     = grid_lats,
        grid_lons     = grid_lons,
        loc_radius_km = loc_radius_km,
        inflation     = inflation,
    )
    raw_op     = ObservationOperator(raw_state, alt_grid)
    Y_raw_ens  = raw_op.compute_stec_ensemble(rep_rays, grid_point_weights=rep_W)
    inno_raw   = y_obs_arc - Y_raw_ens.mean(axis=1)
    _print_arc_innovations("Raw prior", inno_raw)
    raw_enkf.assimilate(
        Y_f                 = Y_raw_ens,
        y_obs               = y_obs_arc,
        R                   = R,              # un-inflated — full correction in one step
        localisation_matrix = L_ray,
        max_update_step     = max_update_step,
        deterministic       = False,
    )
    Y_raw_post_ens  = ObservationOperator(raw_state, alt_grid).compute_stec_ensemble(
        rep_rays, grid_point_weights=rep_W
    )
    inno_raw_post = y_obs_arc - Y_raw_post_ens.mean(axis=1)
    _print_arc_innovations("Raw post ", inno_raw_post)

    # ── ES-MDA: inflate R by n_mda_iterations and re-linearize each step ─────
    # With n_mda_iterations=1 this degenerates to a single-step deterministic
    # square-root EnKF (R_mda = R, one iteration).  With n_mda_iterations > 1
    # it becomes the full Emerick & Reynolds (2013) smoother.
    _update_label = (
        f"ES-MDA ({n_mda_iterations} iter)"
        if n_mda_iterations > 1
        else "EnKF (single step)"
    )
    print(f"\n  [EnKF] {_update_label}, "
          f"{n_update_obs} obs, {n_members} members …")
    R_mda = n_mda_iterations * R
    _mda_inno_list: list[np.ndarray] = []   # per-step update-ray innovations
    for mda_i in range(n_mda_iterations):
        Y_mda_ens  = op.compute_stec_ensemble(rep_rays, grid_point_weights=rep_W)
        inno_mda   = y_obs_arc - Y_mda_ens.mean(axis=1)
        _mda_inno_list.append(inno_mda.copy())
        _step_label = (
            f"ES-MDA {mda_i+1}/{n_mda_iterations}"
            if n_mda_iterations > 1
            else "EnKF update"
        )
        _print_arc_innovations(_step_label, inno_mda)
        
        # Only enforce physical clamps on the very last MDA iteration
        # is_final_step = (mda_i == n_mda_iterations - 1)
        
        analysis_mean, diag = enkf.assimilate(
            Y_f                  = Y_mda_ens,
            y_obs                = y_obs_arc,
            R                    = R_mda,
            localisation_matrix  = L_ray,
            max_update_step      = max_update_step,
            deterministic        = False,           # <-- FIX: Use Stochastic Path A
            apply_bounds         = True    # <-- FIX: Delay physical clamping
        )
    # state.ensemble now holds the ES-MDA posterior

    # ── Posterior forward model ───────────────────────────────────────────────
    # Re-evaluate the full per-sample profile with the posterior ensemble
    Y_all_post_ens  = op.compute_stec_ensemble(
        all_sample_rays, grid_point_weights=all_sample_W
    )  # (n_total_samples, n_members)
    Y_all_post_mean = Y_all_post_ens.mean(axis=1)   # (n_total_samples,)

    # Also re-evaluate rep rays for RMSE consistency and innovation diagnostics
    Y_rep_post_ens  = op.compute_stec_ensemble(
        rep_rays, grid_point_weights=rep_W
    )
    Y_rep_post_mean = Y_rep_post_ens.mean(axis=1)
    # Posterior innovation on update rays (needed for arc innovation plot)
    inno_post_enkf = y_obs_arc - Y_rep_post_mean

    # ── Convert parametric state → 3D EDP grids ──────────────────────────────
    prior_state_enc          = IonosphericState(n_geo, 1)
    prior_state_enc.ensemble = mean_state[:, :, np.newaxis]

    prior_edp_enkf     = _parametric_to_edp(prior_state_enc, prior_state_enc.ensemble, alt_grid)
    posterior_edp_enkf = _parametric_to_edp(state, state.ensemble, alt_grid)

    # ── Ne-space ensemble covariances (for covariance plot) ───────────────────
    # Build a dummy IonosphericState to forward-model the prior ensemble.
    prior_ens_state          = IonosphericState(n_geo, n_members)
    prior_ens_state.ensemble = prior_ensemble_snapshot

    prior_ne_ens = _parametric_to_edp_ensemble(
        prior_ens_state, prior_ensemble_snapshot, alt_grid
    )   # (n_alt, n_geo, n_members)
    post_ne_ens  = _parametric_to_edp_ensemble(
        state, state.ensemble, alt_grid
    )   # (n_alt, n_geo, n_members)

    n_alt_  = len(alt_grid)
    # Flatten to (n_alt*n_geo, n_members) — same layout as the KF P_grid block
    def _ne_ens_cov(ne_ens):
        flat = ne_ens.reshape(n_alt_ * n_geo, n_members)
        cov  = np.cov(flat)                       # (n_alt*n_geo, n_alt*n_geo)
        # Augment with a zero bias block to match KF augmented-P shape
        n_sv_ = n_alt_ * n_geo
        P_aug = np.block([
            [cov,                        np.zeros((n_sv_, n_geo))],
            [np.zeros((n_geo, n_sv_)),   np.zeros((n_geo, n_geo))],
        ])
        return P_aug

    enkf_prior_P = _ne_ens_cov(prior_ne_ens)
    enkf_post_P  = _ne_ens_cov(post_ne_ens)

    # ── TEC RMSE metrics (per-sample, comparable to the KF's H@x residuals) ──
    prior_rmse = float(np.sqrt(np.nanmean((y_obs_all - Y_all_prior_mean) ** 2)))
    post_rmse  = float(np.sqrt(np.nanmean((y_obs_all - Y_all_post_mean) ** 2)))
    print(f"  [EnKF] Prior RMSE: {prior_rmse:.3f} TECU  →  "
          f"Post RMSE: {post_rmse:.3f} TECU  (per-sample)")

    # ── Arc innovation diagnostic figure ──────────────────────────────────────
    # Build per-arc stats from the full per-sample profile (not just the
    # 7-level update rays) so RMSE is directly comparable to the KF figure.
    _all_prior_resid = y_obs_all - Y_all_prior_mean   # (n_total_samples,)
    _all_post_resid  = y_obs_all - Y_all_post_mean    # (n_total_samples,)

    _arc_prior_mean, _arc_post_mean   = [], []
    _arc_prior_rmse, _arc_post_rmse   = [], []
    _arc_lats_e,     _arc_lons_e      = [], []
    _arc_labels_e                     = []
    _soff = 0
    for _i, (_cl, _ns) in enumerate(zip(clean_list, ray_counts)):
        _sl = slice(_soff, _soff + _ns)
        _rp = _all_prior_resid[_sl]
        _ra = _all_post_resid[_sl]
        _arc_prior_mean.append(float(np.nanmean(_rp)))
        _arc_post_mean.append( float(np.nanmean(_ra)))
        _arc_prior_rmse.append(float(np.sqrt(np.nanmean(_rp ** 2))))
        _arc_post_rmse.append( float(np.sqrt(np.nanmean(_ra ** 2))))
        _arc_lats_e.append(float(per_arc_tp_lats[_i].mean()))
        _arc_lons_e.append(float(per_arc_tp_lons[_i].mean()))
        if _i < len(sat_ids) and sat_ids[_i]:
            _, _prn = sat_ids[_i]
            _arc_labels_e.append(str(_prn))
        else:
            _arc_labels_e.append(f"arc{_i:02d}")
        _soff += _ns

    # Per-arc mean residuals at each MDA step (from update rays)
    _arc_upd_offsets = np.concatenate([[0], np.cumsum(arc_update_counts)])
    _mda_arc_means_list: list[np.ndarray] = []
    for _inno_step in _mda_inno_list:
        _step_means = []
        for _ai in range(len(clean_list)):
            _sl = slice(int(_arc_upd_offsets[_ai]), int(_arc_upd_offsets[_ai + 1]))
            _step_means.append(float(np.nanmean(_inno_step[_sl])))
        _mda_arc_means_list.append(np.array(_step_means))

    _plot_arc_innovation_diagnostic(
        arc_labels          = _arc_labels_e,
        arc_prior_mean      = np.array(_arc_prior_mean),
        arc_post_mean       = np.array(_arc_post_mean),
        arc_prior_rmse      = np.array(_arc_prior_rmse),
        arc_post_rmse       = np.array(_arc_post_rmse),
        arc_lats            = np.array(_arc_lats_e),
        arc_lons            = np.array(_arc_lons_e),
        all_prior           = _all_prior_resid,
        all_post_main       = _all_post_resid,
        group_key           = group_key,
        save_dir            = save_dir,
        filter_name         = "EnKF",
        prior_rmse          = prior_rmse,
        post_rmse           = post_rmse,
        mda_arc_means_list  = _mda_arc_means_list,
        mda_flat_list       = _mda_inno_list,
    )

    # ── Build tec_slices — one profile per occultation ────────────────────────
    # Split the flat per-sample arrays back into per-arc slices.  Each slice
    # carries the full sTEC vs. tangent-altitude variation — no flat lines.
    tec_slices_enkf = []
    sample_offset = 0
    for i, cl in enumerate(clean_list):
        n_s = ray_counts[i]
        sl  = slice(sample_offset, sample_offset + n_s)

        tec_slices_enkf.append({
            "measured":   np.asarray(cl["tec"]),
            "prior_tec":  Y_all_prior_mean[sl].copy(),
            "post_tec":   Y_all_post_mean[sl].copy(),
            "tangent_km": np.asarray(cl["tangent_km"]),
        })
        sample_offset += n_s

    # ── Assemble result dict ──────────────────────────────────────────────────
    res_enkf = dict(res_kf)            # copy all shared fields
    res_enkf["prior_edp_3d"]         = prior_edp_enkf          # (n_alt, n_geo)
    res_enkf["post_edp_3d"]          = posterior_edp_enkf
    res_enkf["joint_post_edp_3d"]    = posterior_edp_enkf
    res_enkf["tec_slices"]           = tec_slices_enkf
    res_enkf["prior_tec_rmse"]       = prior_rmse
    res_enkf["post_tec_rmse"]        = post_rmse
    res_enkf["joint_post_tec_rmse"]  = post_rmse
    res_enkf["prior_P"]              = enkf_prior_P   # Ne-space augmented covariance
    res_enkf["post_P"]               = enkf_post_P
    res_enkf["enkf_state"]           = state      # keep for diagnostics
    res_enkf["enkf_diag"]            = diag
    res_enkf["prior_mean_state"]     = mean_state              # (N_STATE, n_geo)
    res_enkf["post_mean_state"]      = state.ensemble.mean(axis=2)  # (N_STATE, n_geo)

    return res_enkf


# ─────────────────────────────────────────────────────────────────────────────
# §C2  Parametric optimization — scipy.optimize.minimize-based TEC inversion
# ─────────────────────────────────────────────────────────────────────────────

#: Physical bounds on the 8-parameter IRI state vector (log/linear convention).
#: Order matches PARAM_NAMES: [log10(NmF2), hmF2, H0, gamma, B0, B1, log10(NmE), hmE]
_OPT_BOUNDS_PER_PARAM: list[tuple[float, float]] = [
    (9.0,   13.0),   # log10(NmF2)
    (100.0, 600.0),  # hmF2
    (10.0,  300.0),  # H0
    (0.05,  2.0),    # gamma
    (20.0,  300.0),  # B0
    (0.5,   4.0),    # B1
    (7.0,   12.0),   # log10(NmE)
    (80.0,  180.0),  # hmE
]

#: Style map for the 5 optimisation methods — (colour, linestyle, marker)
_OPT_METHOD_STYLES: dict[str, tuple[str, str, str]] = {
    "BFGS":        ("tab:purple",  "--",  "v"),
    "Nelder-Mead": ("tab:brown",   "-.",  "^"),
    "Newton-CG":   ("tab:pink",    ":",   "<"),
    "SLSQP":       ("tab:cyan",    "--",  ">"),
    "trust-constr":("tab:olive",   "-.",  "P"),
}


def _draw_param_boxes(
    ax, entries: list[tuple[str, str, np.ndarray]],
    loc: str = "lower left", fontsize: float = 6.5, dy: float = 0.15,
) -> None:
    """
    Annotate a parameterized-EDP axes with one colour-coded parameter readout
    per filter/method, stacked vertically.

    entries : list of (label, colour, state_vector), where state_vector is the
              8-element Chapman/Epstein state (log10-density convention,
              ordered per PARAM_NAMES: [log10(NmF2), hmF2, H0, gamma, B0, B1,
              log10(NmE), hmE]).
    """
    if not entries:
        return
    anchors = {
        "upper left":  (0.02, 0.98, "left",  "top"),
        "lower left":  (0.02, 0.02, "left",  "bottom"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x0, y0, ha, va = anchors.get(loc, anchors["lower left"])
    step = -dy if va == "top" else dy
    y = y0
    for label, col, pvec in entries:
        nmf2 = 10.0 ** pvec[I_LOG_NMF2]
        nme  = 10.0 ** pvec[I_LOG_NME]
        txt = (f"{label}\n"
               f"NmF2={nmf2:.2e}  hmF2={pvec[I_HMF2]:.0f} km\n"
               f"H0={pvec[I_H0]:.1f}  γ={pvec[I_GAMMA]:.2f}  "
               f"B0={pvec[I_B0]:.1f}  B1={pvec[I_B1]:.2f}\n"
               f"NmE={nme:.2e}  hmE={pvec[I_HME]:.0f} km")
        ax.text(x0, y, txt, transform=ax.transAxes, fontsize=fontsize,
                color=col, ha=ha, va=va,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col,
                          lw=1.2, alpha=0.88))
        y += step


# ── Ray-integration precomputation ────────────────────────────────────────────

def _precompute_ray_phi(ray: np.ndarray, alt_grid: np.ndarray) -> np.ndarray:
    """
    Precompute a (n_alt,) integration weight vector φ such that

        TEC_g  =  φ · Ne(alt_grid)   [TECU]

    for a single-grid-point forward model.  Uses linear interpolation from the
    Ne profile on alt_grid onto the ray's actual altitude sample points, then
    applies the same trapezoidal arc-length integration as ObservationOperator.

    Parameters
    ----------
    ray      : (n_pts, 3)  [lat_deg, lon_deg, alt_km] along the ray.
    alt_grid : (n_alt,)    altitude grid used for Ne evaluation (km).

    Returns
    -------
    phi : (n_alt,)  integration weights in TECU / (m⁻³).
    """
    from Ionosphere_Tomography_Inverter.observation_operator import (
        ObservationOperator as _OO,
    )

    alts_km  = ray[:, 2]                     # (n_pts,)
    arc_lens = _OO._arc_length_km(ray)        # (n_pts,) cumulative km

    n_pts = len(alts_km)
    n_alt = len(alt_grid)

    # Trapezoid quadrature weights (km)
    trap_w = np.zeros(n_pts)
    if n_pts >= 2:
        trap_w[0]  = (arc_lens[1]  - arc_lens[0])  / 2.0
        trap_w[-1] = (arc_lens[-1] - arc_lens[-2]) / 2.0
        if n_pts > 2:
            trap_w[1:-1] = (arc_lens[2:] - arc_lens[:-2]) / 2.0
    elif n_pts == 1:
        trap_w[0] = 1.0

    # Scatter trap_w onto alt_grid via linear interpolation coefficients
    phi = np.zeros(n_alt)
    for k in range(n_pts):
        alt_k = alts_km[k]
        if alt_k <= alt_grid[0]:
            phi[0] += trap_w[k]
        elif alt_k >= alt_grid[-1]:
            phi[-1] += trap_w[k]
        else:
            i     = int(np.searchsorted(alt_grid, alt_k)) - 1
            alpha = (alt_k - alt_grid[i]) / (alt_grid[i + 1] - alt_grid[i])
            phi[i]     += trap_w[k] * (1.0 - alpha)
            phi[i + 1] += trap_w[k] * alpha

    # km → m (×1e3) then m⁻³·m → TECU (÷1e16)
    phi *= 1e3 / 1e16
    return phi


def _optimize_grid_point(args: tuple) -> dict:
    """
    Optimise one grid point's 8-parameter IRI state using multiple scipy
    methods, using the precomputed integration matrix for speed.

    This is a module-level function so it is picklable and can be dispatched
    to worker processes / threads.

    Parameters (packed into ``args``)
    ----------------------------------
    g            : int     — grid-point index (for progress display only)
    n_geo        : int     — total number of grid points
    x0_g         : (8,)   — initial IRI state (log/linear convention)
    Phi          : (n_rays, n_alt) — precomputed integration weight matrix
    W_g          : (n_rays,) — normalised IDW weight of this grid point
    TEC0_g       : (n_rays,) — prior TEC contribution from this grid point
    innov        : (n_rays,) — TEC innovation = y_obs − TEC_prior_total
    sigma        : float   — observation noise std-dev (TECU)
    alt_grid     : (n_alt,) — altitude grid (km)
    bounds_lo    : (8,)    — lower bounds per parameter
    bounds_hi    : (8,)    — upper bounds per parameter
    maxiter_unc  : int     — iteration budget (unconstrained methods)
    maxiter_con  : int     — iteration budget (constrained methods)
    verbose_gp   : bool    — whether to print per-iteration progress

    Returns
    -------
    dict keyed by method name → (x_opt_g, J_opt_g, TEC_post_g, converged, n_iters, n_fev)
    """
    from scipy.optimize import minimize, Bounds

    (g, n_geo, x0_g, Phi, W_g, TEC0_g, innov,
     sigma, alt_grid, bounds_lo, bounds_hi,
     maxiter_unc, maxiter_con, verbose_gp) = args

    sigma2 = max(sigma ** 2, 1e-12)
    n_alt  = Phi.shape[1]
    scipy_bounds = Bounds(lb=bounds_lo, ub=bounds_hi)

    def _ne_from_x(x):
        """Evaluate Ne(alt_grid) from the 8-parameter log/linear state."""
        p = np.clip(x, bounds_lo, bounds_hi)
        p_lin = p.copy()
        p_lin[I_LOG_NMF2] = 10.0 ** p[I_LOG_NMF2]
        p_lin[I_LOG_NME]  = 10.0 ** p[I_LOG_NME]
        return _ne_profile_ensemble(alt_grid, p_lin[:, np.newaxis])[:, 0]

    def _J_g(x):
        """Per-grid-point TEC residual objective."""
        Ne       = _ne_from_x(x)                      # (n_alt,)
        TEC_g    = Phi @ Ne                            # (n_rays,)
        delta    = innov + W_g * (TEC0_g - TEC_g)     # (n_rays,)
        return float(np.dot(delta, delta)) / sigma2

    def _J_and_grad_g(x):
        """Combined objective and gradient for gradient-based scipy methods.

        Computes the Jacobian of the sTEC residual objective via the chain rule:

            J(x)      = ||δ||² / σ²,       δ = innov + W_g ⊙ (TEC0_g − Φ Ne(x))
            ∂J/∂x_k   = −(2/σ²) (W_g ⊙ δ)ᵀ Φ  ∂Ne/∂x_k

        ∂Ne/∂x_k is approximated by a single forward finite-difference step on
        the Ne profile (8 profile evaluations total), which is roughly twice as
        efficient as scipy's "3-point" scheme that finite-differences the full
        scalar objective (16 objective evaluations per gradient call).

        Returns
        -------
        J    : float   objective value
        grad : (8,)    gradient w.r.t. the 8 IRI parameters
        """
        Ne    = _ne_from_x(x)
        TEC_g = Phi @ Ne                           # (n_rays,)
        delta = innov + W_g * (TEC0_g - TEC_g)    # (n_rays,)
        J     = float(np.dot(delta, delta)) / sigma2

        # Effective gradient direction in Ne-space: d_eff = -(2/σ²)(W_g ⊙ δ)ᵀ Φ
        d_eff = (-2.0 / sigma2) * (W_g * delta) @ Phi   # (n_alt,)

        grad = np.empty(8, dtype=float)
        for k in range(8):
            x_p     = x.copy()
            dx      = max(abs(x[k]) * 1e-5, 1e-8)
            x_p[k] += dx
            grad[k] = d_eff @ ((_ne_from_x(x_p) - Ne) / dx)

        return J, grad

    prior_J_g = _J_g(x0_g)

    # Per-iteration callback — printed only if verbose_gp=True
    def _make_cb(method, J_prior_g):
        _it = [0]
        _EVERY = 20

        def _cb_std(xk):
            _it[0] += 1
            if verbose_gp and _it[0] % _EVERY == 0:
                J_cur = _J_g(xk)
                print(f"      gp{g:3d}/{n_geo} [{method}] "
                      f"iter {_it[0]:>4d}  J={J_cur:.4f}  "
                      f"Δ={J_prior_g - J_cur:+.4f}")

        def _cb_tc(xk, state):
            _it[0] += 1
            if verbose_gp and _it[0] % _EVERY == 0:
                J_cur = float(state.fun) if hasattr(state, "fun") else _J_g(xk)
                print(f"      gp{g:3d}/{n_geo} [{method}] "
                      f"iter {_it[0]:>4d}  J={J_cur:.4f}  "
                      f"nfev={getattr(state,'nfev','?')}")

        return _cb_tc if method == "trust-constr" else _cb_std

    configs = [
        # Unconstrained — no bounds object; clip handled inside _J_g / _ne_from_x
        # jac=True → scipy expects fun to return (f, grad); uses _J_and_grad_g below
        ("BFGS",        {"method": "BFGS",
                         "jac": True,
                         "options": {"maxiter": maxiter_unc, "gtol": 1e-5}}),
        ("Nelder-Mead", {"method": "Nelder-Mead",
                         "options": {"maxiter": maxiter_unc * 20,
                                     "xatol": 1e-4, "fatol": 1e-4,
                                     "adaptive": True}}),
        ("Newton-CG",   {"method": "Newton-CG",
                         "jac": True,
                         "options": {"maxiter": maxiter_unc, "xtol": 1e-5}}),
        # Constrained
        ("SLSQP",       {"method": "SLSQP",
                         "jac": True,
                         "bounds": scipy_bounds,
                         "options": {"maxiter": maxiter_con, "ftol": 1e-9}}),
        ("trust-constr",{"method": "trust-constr",
                         "jac": True,
                         "bounds": scipy_bounds,
                         "options": {"maxiter": maxiter_con, "gtol": 1e-6,
                                     "verbose": 0}}),
    ]

    out = {}
    for method_name, kw in configs:
        cb  = _make_cb(method_name, prior_J_g)
        # Nelder-Mead is gradient-free; all other methods use _J_and_grad_g
        # which returns (J, grad) together, avoiding double Ne evaluation.
        fun = _J_g if method_name == "Nelder-Mead" else _J_and_grad_g
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(fun, x0_g.copy(), callback=cb, **kw)
            x_opt  = np.clip(res.x, bounds_lo, bounds_hi)
            J_opt  = float(res.fun)
            ok     = res.success
            n_it   = getattr(res, "nit",  "?")
            n_fev  = getattr(res, "nfev", "?")
        except Exception as exc:
            x_opt  = x0_g.copy()
            J_opt  = prior_J_g
            ok     = False
            n_it   = 0
            n_fev  = 0
            if verbose_gp:
                print(f"      gp{g:3d}/{n_geo} [{method_name}] FAILED: {exc}")

        Ne_post  = _ne_from_x(x_opt)
        TEC_post = Phi @ Ne_post    # (n_rays,)
        out[method_name] = (x_opt, J_opt, TEC_post, ok, n_it, n_fev)

    return out


def _run_parametric_optimization(
    res_kf: dict,
    alt_grid: np.ndarray,
    save_dir: str = "./Figures/CompareKF_EnKF/",
    group_key: str = "group",
    sigma_obs_tecu: float = 1.0,
    n_update_rays: int = 5,
    maxiter_unconstrained: int = 300,
    maxiter_constrained: int = 600,
    n_workers: int = 4,
) -> dict[str, dict]:
    """
    Minimise forward-model sTEC residuals using ``scipy.optimize.minimize``.

    **Acceleration strategy — per-grid-point factorisation**

    Instead of a single joint problem over all N_STATE × n_geo parameters,
    the domain is decomposed into n_geo independent 8-parameter sub-problems
    (one per grid point), each solved with all five scipy methods in parallel
    across threads.  Three mechanisms combine to make this fast:

    1. *Precomputed integration matrices* — a (n_rays, n_alt) matrix Φ is
       built once so that TEC_g = Φ @ Ne(alt_grid) is a pure matrix-vector
       multiply, replacing the Python loop over ray samples on every call.

    2. *8-parameter sub-problems* — an explicit chain-rule Jacobian
       ``_J_and_grad_g`` computes the gradient in 9 Ne-profile evaluations
       per step (1 baseline + 8 single-param perturbations), roughly half the
       cost of scipy's ``"3-point"`` which needed 16 full-objective evaluations.
       Each sub-problem converges in ~20–50 iters rather than 500+ for the
       joint problem.

    3. *Thread parallelism* — grid-point sub-problems are dispatched to a
       ThreadPoolExecutor; numpy releases the GIL during array operations so
       workers overlap substantially.

    Solvers applied to each grid-point sub-problem:

    Unconstrained  — BFGS, Nelder-Mead, Newton-CG
    Constrained    — SLSQP, trust-constr  (with scipy.optimize.Bounds)

    Parameters
    ----------
    res_kf         : dict    Output of process_group / _process_verif_group.
    alt_grid       : ndarray Altitude grid (km).
    sigma_obs_tecu : float   Observation noise std-dev (TECU).
    n_update_rays  : int     Representative rays per arc (same as EnKF).
    maxiter_unconstrained : int  Iteration cap for BFGS / Nelder-Mead / Newton-CG.
    maxiter_constrained   : int  Iteration cap for SLSQP / trust-constr.
    n_workers      : int     Thread-pool size for grid-point parallelism.

    Returns
    -------
    dict[str, dict]  — keyed by method name; values match _run_parametric_enkf
                       output structure so they drop straight into plotting.
    """
    import concurrent.futures

    eds_occ    = res_kf["eds_occ"]
    clean_list = res_kf["clean_list"]
    prior_edp  = res_kf["prior_edp_3d"]   # (n_alt, n_geo)
    verts_geo  = eds_occ.geolocation       # (n_geo, 2): col0=lon, col1=lat

    n_alt = len(alt_grid)
    n_geo = verts_geo.shape[0]
    n_occ = len(clean_list)

    grid_lats = verts_geo[:, 1].astype(float)
    grid_lons = verts_geo[:, 0].astype(float)

    # ── Parameter bounds (per grid point) ────────────────────────────────────
    bounds_lo = np.array([b[0] for b in _OPT_BOUNDS_PER_PARAM], dtype=float)
    bounds_hi = np.array([b[1] for b in _OPT_BOUNDS_PER_PARAM], dtype=float)

    # ── IRI prior state ───────────────────────────────────────────────────────
    t_centre    = _parse_time_window(res_kf.get("time_window", group_key))
    sampling_df = _solar_sampling_df(t_centre)

    print(f"  [Opt] Building IRI prior state at {n_geo} grid points …")
    prior_mean = np.zeros((N_STATE, n_geo), dtype=float)
    try:
        ne_all, feat_all = _get_iri_edp_and_features_batch(
            t_centre, grid_lats, grid_lons, alt_grid, sampling_df
        )
        for g in range(n_geo):
            prior_mean[:, g] = _state_from_iri_direct(
                ne_all[:, g], feat_all[:, g], alt_grid
            )
    except Exception as _exc:
        print(f"  [Opt] Batch IRI call failed ({_exc}); falling back to profile fit")
        for g in range(n_geo):
            prior_mean[:, g] = _fit_iri_params(prior_edp[:, g], alt_grid)

    # ── Build representative rays (same as EnKF) ──────────────────────────────
    print(f"  [Opt] Building ray trajectories …")
    rep_rays:    list[np.ndarray] = []
    rep_tp_lats: list[float]      = []
    rep_tp_lons: list[float]      = []
    rep_tec_obs: list[float]      = []

    per_arc_sample_rays: list[list[np.ndarray]] = []
    per_arc_tp_lats:     list[np.ndarray]       = []
    per_arc_tp_lons:     list[np.ndarray]       = []
    ray_counts: list[int] = []

    for cl in clean_list:
        leo  = cl["LEO"]
        gnss = cl["GNSS"]
        n_s  = leo.shape[1]

        tang_alts = np.array([
            float(np.linalg.norm(
                gnss[:, i] + np.clip(
                    -np.dot(gnss[:, i], leo[:, i] - gnss[:, i])
                    / max(float(np.dot(leo[:, i] - gnss[:, i],
                                       leo[:, i] - gnss[:, i])), 1e-12),
                    0.0, 1.0
                ) * (leo[:, i] - gnss[:, i])
            )) - 6371.0
            for i in range(n_s)
        ])
        sorted_idx = np.argsort(tang_alts)
        k      = min(n_update_rays, n_s)
        chosen = ([sorted_idx[0]] if k == 1 else
                  [sorted_idx[int(round(j * (len(sorted_idx) - 1) / (k - 1)))]
                   for j in range(k)])

        for idx in chosen:
            traj = _build_gnss_to_leo_ray(gnss[:, idx], leo[:, idx])
            tp_lat, tp_lon = _tangent_latlon_single(gnss[:, idx], leo[:, idx])
            rep_rays.append(traj)
            rep_tp_lats.append(tp_lat)
            rep_tp_lons.append(tp_lon)
            rep_tec_obs.append(float(cl["tec"][idx]))

        arc_rays, arc_lats_i, arc_lons_i = [], [], []
        for i in range(n_s):
            traj = _build_gnss_to_leo_ray(gnss[:, i], leo[:, i])
            tp_lat, tp_lon = _tangent_latlon_single(gnss[:, i], leo[:, i])
            arc_rays.append(traj); arc_lats_i.append(tp_lat); arc_lons_i.append(tp_lon)
        per_arc_sample_rays.append(arc_rays)
        per_arc_tp_lats.append(np.array(arc_lats_i))
        per_arc_tp_lons.append(np.array(arc_lons_i))
        ray_counts.append(n_s)

    rep_tp_lats_arr = np.array(rep_tp_lats)
    rep_tp_lons_arr = np.array(rep_tp_lons)
    y_obs_arc       = np.array(rep_tec_obs)

    all_sample_rays: list[np.ndarray] = []
    all_tp_lats: list[float] = []
    all_tp_lons: list[float] = []
    for arc_rays, arc_lats, arc_lons in zip(
            per_arc_sample_rays, per_arc_tp_lats, per_arc_tp_lons):
        all_sample_rays.extend(arc_rays)
        all_tp_lats.extend(arc_lats.tolist())
        all_tp_lons.extend(arc_lons.tolist())
    all_tp_lats_arr = np.array(all_tp_lats)
    all_tp_lons_arr = np.array(all_tp_lons)
    y_obs_all = np.concatenate([cl["tec"] for cl in clean_list])

    _idw_k       = min(4, n_geo)
    rep_W        = _idw_weights(rep_tp_lats_arr, rep_tp_lons_arr,
                                grid_lats, grid_lons, k=_idw_k)
    all_sample_W = _idw_weights(all_tp_lats_arr, all_tp_lons_arr,
                                grid_lats, grid_lons, k=_idw_k)

    # Normalise IDW weights the same way the observation operator does (row-wise)
    rep_W_norm = rep_W / rep_W.sum(axis=1, keepdims=True).clip(1e-12)
    all_W_norm = all_sample_W / all_sample_W.sum(axis=1, keepdims=True).clip(1e-12)

    n_update_obs = len(rep_rays)
    n_all_rays   = len(all_sample_rays)
    print(f"  [Opt] {n_update_obs} update rays  ({n_occ} arcs × {n_update_rays} levels)  "
          f"  {n_all_rays} diagnostic rays")

    # ── Precompute integration matrices Φ once ────────────────────────────────
    # Φ_rep  : (n_update_obs, n_alt) — for update rays
    # Φ_all  : (n_all_rays, n_alt)   — for diagnostic (per-sample) rays
    print(f"  [Opt] Precomputing ray integration matrices …")
    Phi_rep = np.stack([_precompute_ray_phi(r, alt_grid) for r in rep_rays])
    Phi_all = np.stack([_precompute_ray_phi(r, alt_grid) for r in all_sample_rays])
    print(f"  [Opt] Φ_rep={Phi_rep.shape}  Φ_all={Phi_all.shape}")

    # ── Prior Ne profiles and TEC ─────────────────────────────────────────────
    # ne0_grid : (n_alt, n_geo) — prior Ne for every grid point
    ne0_grid = np.zeros((n_alt, n_geo), dtype=float)
    for g in range(n_geo):
        p = prior_mean[:, g].copy()
        p[I_LOG_NMF2] = 10.0 ** prior_mean[I_LOG_NMF2, g]
        p[I_LOG_NME]  = 10.0 ** prior_mean[I_LOG_NME,  g]
        ne0_grid[:, g] = np.maximum(
            _ne_profile_ensemble(alt_grid, p[:, np.newaxis])[:, 0], 0.0
        )

    # TEC0_g_rep[r, g] = Φ_rep[r] · Ne0[:,g] — prior single-gp TEC for each ray
    TEC0_g_rep = Phi_rep @ ne0_grid       # (n_update_obs, n_geo)
    TEC0_g_all = Phi_all @ ne0_grid       # (n_all_rays,   n_geo)

    # Prior total TEC (weighted blend across grid points)
    TEC0_rep = (rep_W_norm * TEC0_g_rep).sum(axis=1)   # (n_update_obs,)
    TEC0_all = (all_W_norm * TEC0_g_all).sum(axis=1)   # (n_all_rays,)

    # Innovations (fixed — computed from prior, used by all grid-point sub-problems)
    innov_rep = y_obs_arc - TEC0_rep   # (n_update_obs,)

    prior_rmse = float(np.sqrt(np.nanmean((y_obs_all - TEC0_all) ** 2)))
    prior_J    = float(np.dot(innov_rep, innov_rep)) / max(sigma_obs_tecu ** 2, 1e-12)
    print(f"  [Opt] Prior — J={prior_J:.3f}  RMSE={prior_rmse:.3f} TECU")

    # ── Prior EDP ─────────────────────────────────────────────────────────────
    _prior_state_enc           = IonosphericState(n_geo, 1)
    _prior_state_enc.ensemble  = prior_mean[:, :, np.newaxis]
    prior_edp_opt = _parametric_to_edp(
        _prior_state_enc, _prior_state_enc.ensemble, alt_grid
    )

    # ── Per-grid-point sub-problems ───────────────────────────────────────────
    # Each sub-problem optimises 8 parameters for grid point g, with all other
    # grid points held at the IRI prior.  The objective is:
    #
    #   J_g(x_g) = Σ_r [innov_r + W_norm[r,g] * (TEC0_g_r − TEC_g_r(x_g))]² / σ²
    #
    # where W_norm[r,g]*TEC_g_r(x_g) is this grid point's contribution to ray r.
    # For rays with zero weight the innovation term cancels exactly, so they do
    # not pull the solution.

    print(f"  [Opt] Dispatching {n_geo} grid-point sub-problems "
          f"across {n_workers} threads …")

    worker_args = []
    for g in range(n_geo):
        worker_args.append((
            g, n_geo,
            prior_mean[:, g].copy(),   # x0_g
            Phi_rep,                   # (n_update_obs, n_alt) — shared, read-only
            rep_W_norm[:, g],          # W_g (n_update_obs,)
            TEC0_g_rep[:, g],          # TEC0_g (n_update_obs,)
            innov_rep,                 # (n_update_obs,)
            sigma_obs_tecu,
            alt_grid,
            bounds_lo,
            bounds_hi,
            maxiter_unconstrained,
            maxiter_constrained,
            True,                      # verbose_gp
        ))

    # Run grid points in a thread pool — numpy releases the GIL for heavy ops
    gp_results: list[dict] = [None] * n_geo   # type: ignore[assignment]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_optimize_grid_point, a): a[0]
                   for a in worker_args}
        for fut in concurrent.futures.as_completed(futures):
            g_idx = futures[fut]
            try:
                gp_results[g_idx] = fut.result()
            except Exception as exc:
                print(f"  [Opt] gp{g_idx} FAILED: {exc}")
                # Fall back: all methods use the prior for this grid point
                gp_results[g_idx] = {
                    m: (prior_mean[:, g_idx].copy(),
                        prior_J / max(n_geo, 1),
                        TEC0_g_rep[:, g_idx],
                        False, 0, 0)
                    for m in _OPT_METHOD_STYLES
                }

    # ── Reconstruct results per method ────────────────────────────────────────
    all_method_names = list(_OPT_METHOD_STYLES.keys())
    results: dict[str, dict] = {}

    for method_name in all_method_names:
        # Assemble posterior state (N_STATE, n_geo) from per-gp results
        post_mean = np.zeros((N_STATE, n_geo), dtype=float)
        TEC_post_g_rep = np.zeros((n_update_obs, n_geo), dtype=float)
        total_iters = 0
        total_fev   = 0
        any_converged = False

        for g in range(n_geo):
            x_opt_g, J_opt_g, tec_post_g, ok, n_it, n_fev = gp_results[g][method_name]
            post_mean[:, g]       = x_opt_g
            TEC_post_g_rep[:, g]  = tec_post_g
            total_iters  += (n_it  if isinstance(n_it,  int) else 0)
            total_fev    += (n_fev if isinstance(n_fev, int) else 0)
            if ok:
                any_converged = True

        # Posterior total TEC (update rays)
        TEC_post_rep = (rep_W_norm * TEC_post_g_rep).sum(axis=1)

        # Posterior Ne on alt_grid for all grid points (for Φ_all forward model)
        ne_post_grid = np.zeros((n_alt, n_geo), dtype=float)
        for g in range(n_geo):
            p = post_mean[:, g].copy()
            p[I_LOG_NMF2] = 10.0 ** post_mean[I_LOG_NMF2, g]
            p[I_LOG_NME]  = 10.0 ** post_mean[I_LOG_NME,  g]
            ne_post_grid[:, g] = np.maximum(
                _ne_profile_ensemble(alt_grid, p[:, np.newaxis])[:, 0], 0.0
            )
        TEC_post_g_all = Phi_all @ ne_post_grid          # (n_all_rays, n_geo)
        TEC_post_all   = (all_W_norm * TEC_post_g_all).sum(axis=1)

        post_rmse   = float(np.sqrt(np.nanmean((y_obs_all - TEC_post_all) ** 2)))
        inno_post   = y_obs_arc - TEC_post_rep
        inno_prior  = innov_rep
        post_J      = float(np.dot(inno_post, inno_post)) / max(sigma_obs_tecu**2, 1e-12)

        print(f"\n  [Opt] {method_name}  "
              f"iters={total_iters}  nfev={total_fev}  "
              f"any_converged={any_converged}  J: {prior_J:.3f}→{post_J:.3f}")
        print(f"    Prior innov  mean={inno_prior.mean():.2f}  std={inno_prior.std():.2f} TECU")
        print(f"    Post  innov  mean={inno_post.mean():.2f}   std={inno_post.std():.2f} TECU")
        print(f"    RMSE (all samples): {prior_rmse:.3f} → {post_rmse:.3f} TECU")

        # Posterior EDP
        _post_state           = IonosphericState(n_geo, 1)
        _post_state.ensemble  = post_mean[:, :, np.newaxis]
        post_edp_opt = _parametric_to_edp(
            _post_state, _post_state.ensemble, alt_grid
        )

        # tec_slices (split flat arrays back into per-arc)
        tec_slices_opt = []
        sample_offset  = 0
        for i, cl in enumerate(clean_list):
            n_s = ray_counts[i]
            sl  = slice(sample_offset, sample_offset + n_s)
            tec_slices_opt.append({
                "measured":   np.asarray(cl["tec"]),
                "prior_tec":  TEC0_all[sl].copy(),
                "post_tec":   TEC_post_all[sl].copy(),
                "tangent_km": np.asarray(cl["tangent_km"]),
            })
            sample_offset += n_s

        # Minimal placeholder covariance (single-member — not meaningful)
        def _ne_flat_cov(ne_2d):
            ne_flat = ne_2d.ravel()
            return np.diag(np.concatenate([ne_flat ** 2 * 0.01, np.ones(n_geo)]))

        res_opt = dict(res_kf)
        res_opt["prior_edp_3d"]        = prior_edp_opt
        res_opt["post_edp_3d"]         = post_edp_opt
        res_opt["joint_post_edp_3d"]   = post_edp_opt
        res_opt["tec_slices"]          = tec_slices_opt
        res_opt["prior_tec_rmse"]      = prior_rmse
        res_opt["post_tec_rmse"]       = post_rmse
        res_opt["joint_post_tec_rmse"] = post_rmse
        res_opt["prior_P"]             = _ne_flat_cov(prior_edp_opt)
        res_opt["post_P"]              = _ne_flat_cov(post_edp_opt)
        res_opt["opt_J_prior"]         = prior_J
        res_opt["opt_J_post"]          = post_J
        res_opt["opt_method"]          = method_name
        res_opt["prior_mean_state"]    = prior_mean          # (N_STATE, n_geo)
        res_opt["post_mean_state"]     = post_mean           # (N_STATE, n_geo)
        results[method_name] = res_opt

    return results


def plot_optimization_methods_comparison(
    res_kf:      dict,
    res_enkf:    dict,
    opt_results: dict[str, dict],
    isr_profiles: list[dict],
    alt_grid:    np.ndarray,
    group_key:   str,
    save_dir:    str,
    n_tec_shown: int = 3,
) -> str:
    """
    Multi-panel comparison of all five optimisation methods against KF and EnKF.

    Layout  (3 rows × 2 cols)
    ─────────────────────────
    [0,0]  Prior TEC profiles          [0,1]  Posterior TEC profiles
    [1,0]  Prior EDP at Millstone Hill [1,1]  Posterior EDP at MH (+ ISR truth)
    [2,0]  TEC RMSE bar chart          [2,1]  NmF2 / hmF2 bias vs. ISR

    Parameters
    ----------
    res_kf, res_enkf : result dicts from process_group / _run_parametric_enkf.
    opt_results      : keyed by method name, each value from _run_parametric_optimization.
    isr_profiles     : list of ISR sweep dicts (may be empty).
    alt_grid         : shared altitude grid (km).
    group_key        : label for title and filename.
    save_dir         : output directory.
    n_tec_shown      : number of occultation arcs in TEC panels.

    Returns
    -------
    str : path to the saved PNG.
    """
    os.makedirs(save_dir, exist_ok=True)

    ne_fmt = ScalarFormatter(useMathText=True)
    ne_fmt.set_powerlimits((-2, 2))

    verts_geo = res_kf["eds_occ"].geolocation
    n_geo     = verts_geo.shape[0]
    n_alt     = len(alt_grid)
    idx_mh    = millstone_vertex_idx(verts_geo)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _mh_edp(res, key="prior_edp_3d"):
        arr = np.asarray(res[key]).reshape(n_alt, n_geo)
        return arr[:, idx_mh]

    def _post_mh(res):
        k = res.get("joint_post_edp_3d")
        return _mh_edp(res, "post_edp_3d") if k is None else \
               np.asarray(k).reshape(n_alt, n_geo)[:, idx_mh]

    # ── Method ordering: KF, EnKF, then each optimizer ────────────────────────
    method_order = ["KF", "EnKF"] + list(opt_results.keys())
    res_by_name  = {"KF": res_kf, "EnKF": res_enkf, **opt_results}

    style_map: dict[str, tuple[str, str, str]] = {
        "KF":   ("royalblue",  "--", "D"),
        "EnKF": ("darkorange", "-.", "s"),
        **{k: v for k, v in _OPT_METHOD_STYLES.items()},
    }

    # TEC arc colours
    n_occ    = len(res_kf["tec_slices"])
    shown    = list(range(min(n_tec_shown, n_occ)))
    cmap_occ = mpl.colormaps.get_cmap("tab10")
    occ_cols = [cmap_occ(i % 10) for i in range(n_occ)]

    ISR_COLOR = "mediumseagreen"

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22))
    safe_key = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    fig.suptitle(
        f"Parametric Optimisation Methods Comparison\n{group_key}",
        fontsize=13, y=0.99,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.35, hspace=0.50,
                           height_ratios=[1, 1, 0.9])

    ax_tec_pr = fig.add_subplot(gs[0, 0])
    ax_tec_po = fig.add_subplot(gs[0, 1], sharey=ax_tec_pr, sharex=ax_tec_pr)
    ax_edp_pr = fig.add_subplot(gs[1, 0])
    ax_edp_po = fig.add_subplot(gs[1, 1], sharey=ax_edp_pr, sharex=ax_edp_pr)
    ax_rmse   = fig.add_subplot(gs[2, 0])
    ax_bias   = fig.add_subplot(gs[2, 1])

    # ── [0,0] Prior TEC ───────────────────────────────────────────────────────
    ax = ax_tec_pr
    sat_ids = res_kf.get("sat_ids", [])
    for i in shown:
        col = occ_cols[i]
        prn = sat_ids[i][1] if i < len(sat_ids) else f"Occ {i+1}"
        sl  = res_kf["tec_slices"][i]
        ax.plot(sl["measured"], sl["tangent_km"], color=col, lw=2.0, label=prn)

    # Draw prior for each method (all identical — IRI prior — but show one each)
    for mname in method_order:
        res  = res_by_name[mname]
        col, ls, _ = style_map.get(mname, ("gray", "--", "."))
        sl = res["tec_slices"][shown[0]] if shown else None
        if sl is not None:
            ax.plot(sl["prior_tec"], sl["tangent_km"],
                    color=col, lw=1.2, ls=ls, alpha=0.75,
                    label=f"{mname} prior" if mname in ("KF", "EnKF") else "_")

    legend_obs = [Line2D([0], [0], color="gray", lw=2.0, label="Measured")]
    legend_mth = [
        Line2D([0], [0], color=style_map[m][0], lw=1.4,
               ls=style_map[m][1], label=f"{m} prior")
        for m in ("KF", "EnKF")
    ]
    ax.legend(handles=legend_obs + legend_mth +
              [Line2D([0], [0], color=occ_cols[i], lw=2.0,
                      label=sat_ids[i][1] if i < len(sat_ids) else f"Occ {i+1}")
               for i in shown],
              fontsize=7, loc="upper right", framealpha=0.85)
    ax.set_xlabel("TEC (TECU)", fontsize=10)
    ax.set_ylabel("Tangent Altitude (km)", fontsize=10)
    ax.set_title("Prior TEC (all methods share IRI prior)", fontsize=10)
    ax.grid(True, alpha=0.3, ls=":")

    # ── [0,1] Posterior TEC ───────────────────────────────────────────────────
    ax = ax_tec_po
    for i in shown:
        col = occ_cols[i]
        sl_kf = res_kf["tec_slices"][i]
        ax.plot(sl_kf["measured"], sl_kf["tangent_km"], color=col, lw=2.0)

    for mname in method_order:
        res = res_by_name[mname]
        col, ls, _ = style_map.get(mname, ("gray", "--", "."))
        for i in shown:
            sl = res["tec_slices"][i]
            ax.plot(sl["post_tec"], sl["tangent_km"],
                    color=col, lw=1.4, ls=ls, alpha=0.8)

    legend_po = [Line2D([0], [0], color="gray", lw=2.0, label="Measured")] + [
        Line2D([0], [0], color=style_map.get(m, ("gray",))[0],
               lw=1.4, ls=style_map.get(m, ("gray", "--"))[1], label=m)
        for m in method_order
    ]
    ax.legend(handles=legend_po, fontsize=7, loc="upper right", framealpha=0.85)
    ax.set_xlabel("TEC (TECU)", fontsize=10)
    ax.set_title("Posterior TEC", fontsize=10)
    ax.grid(True, alpha=0.3, ls=":")

    # ── [1,0] Prior EDP at MH ─────────────────────────────────────────────────
    ax = ax_edp_pr
    for prof in isr_profiles:
        ax.plot(prof["ne"], prof["alt_km"],
                color=ISR_COLOR, lw=1.0, alpha=0.7,
                label="ISR truth" if prof is isr_profiles[0] else "_")

    for mname in ("KF", "EnKF"):
        col, ls, _ = style_map[mname]
        ax.plot(_mh_edp(res_by_name[mname]), alt_grid,
                color=col, lw=2.2, ls=ls, label=f"{mname} prior (IRI)")

    ax.set_xlabel("Electron Density (m⁻³)", fontsize=10)
    ax.set_ylabel("Altitude (km)", fontsize=10)
    ax.set_title("Prior EDP at Millstone Hill", fontsize=10)
    ax.xaxis.set_major_formatter(ne_fmt)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.3, ls=":")
    ax.set_ylim(bottom=0)

    # Parameter readout — prior is the shared IRI fit (identical for every
    # method), so show it once per method colour for direct comparison with
    # the posterior box below.
    prior_entries = [
        (f"{mname} prior", style_map[mname][0],
         res_by_name[mname]["prior_mean_state"][:, idx_mh])
        for mname in ("KF", "EnKF") if "prior_mean_state" in res_by_name[mname]
    ]
    _draw_param_boxes(ax, prior_entries, loc="lower left")

    # ── [1,1] Posterior EDP at MH ─────────────────────────────────────────────
    ax = ax_edp_po
    for prof in isr_profiles:
        ax.plot(prof["ne"], prof["alt_km"],
                color=ISR_COLOR, lw=1.0, alpha=0.7,
                label="ISR truth" if prof is isr_profiles[0] else "_")

    for mname in method_order:
        col, ls, _ = style_map.get(mname, ("gray", "--", "."))
        lw = 2.5 if mname in ("KF", "EnKF") else 1.8
        ax.plot(_post_mh(res_by_name[mname]), alt_grid,
                color=col, lw=lw, ls=ls, label=mname)

    ax.set_xlabel("Electron Density (m⁻³)", fontsize=10)
    ax.set_title("Posterior EDP at Millstone Hill vertex", fontsize=10)
    ax.xaxis.set_major_formatter(ne_fmt)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.3, ls=":")
    ax.set_ylim(bottom=0)

    # Parameter readout — one colour-coded box per parametric method (the
    # gridded KF has no Chapman/Epstein state, so it is excluded here).
    post_entries = [
        (mname, style_map.get(mname, ("gray",))[0],
         res_by_name[mname]["post_mean_state"][:, idx_mh])
        for mname in method_order
        if mname != "KF" and "post_mean_state" in res_by_name[mname]
    ]
    _draw_param_boxes(ax, post_entries, loc="lower left")

    # ── [2,0] TEC RMSE bar chart ──────────────────────────────────────────────
    ax    = ax_rmse
    names = method_order
    prior_rmses = [res_by_name[m]["prior_tec_rmse"] for m in names]
    post_rmses  = [
        res_by_name[m].get("joint_post_tec_rmse", res_by_name[m]["post_tec_rmse"])
        for m in names
    ]

    x_pos   = np.arange(len(names))
    bar_w   = 0.35
    bar_col = [style_map.get(m, ("gray",))[0] for m in names]

    ax.bar(x_pos - bar_w / 2, prior_rmses, bar_w,
           color="lightgray", edgecolor="black", lw=0.8, label="Prior")
    ax.bar(x_pos + bar_w / 2, post_rmses,  bar_w,
           color=bar_col,    edgecolor="black", lw=0.8, label="Posterior", alpha=0.85)

    for xi, (pr, po) in enumerate(zip(prior_rmses, post_rmses)):
        ax.text(xi - bar_w / 2, pr + 0.05, f"{pr:.2f}", ha="center",
                va="bottom", fontsize=7)
        ax.text(xi + bar_w / 2, po + 0.05, f"{po:.2f}", ha="center",
                va="bottom", fontsize=7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("TEC RMSE (TECU)", fontsize=10)
    ax.set_title("Prior vs. Posterior TEC RMSE by Method", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, ls=":")

    # ── [2,1] NmF2 / hmF2 bias vs. ISR ───────────────────────────────────────
    ax = ax_bias
    if isr_profiles:
        isr_nm = np.nanmean([p["nm_f2"] for p in isr_profiles])
        isr_hm = np.nanmean([p["hm_f2"] for p in isr_profiles])

        nm_biases: list[float] = []
        hm_biases: list[float] = []
        for m in names:
            nm_post, hm_post = extract_robust_f2_peak(_post_mh(res_by_name[m]), alt_grid)
            nm_biases.append(float(nm_post - isr_nm) if not np.isnan(nm_post) else np.nan)
            hm_biases.append(float(hm_post - isr_hm) if not np.isnan(hm_post) else np.nan)

        ax2 = ax.twinx()

        # hmF2 bias (left axis, km)
        ax.bar(x_pos - bar_w / 2, hm_biases, bar_w,
               color=bar_col, edgecolor="black", lw=0.8, alpha=0.85,
               label="Δhm F2 (km)")
        ax.axhline(0.0, color="black", lw=0.8, ls="--")
        ax.set_ylabel("hmF2 bias (km)", fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_title("Posterior F2 Bias vs. ISR Truth", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3, ls=":")

        # NmF2 bias (right axis, m⁻³)
        ax2.bar(x_pos + bar_w / 2, nm_biases, bar_w,
                color=bar_col, edgecolor="black", lw=0.8, alpha=0.55,
                label="ΔNm F2 (m⁻³)")
        ax2.set_ylabel("NmF2 bias (m⁻³)", fontsize=10)
        ax2.legend(loc="upper right", fontsize=9)

        # Print bias table
        print("\n  [Opt] Posterior bias vs. ISR at Millstone Hill:")
        print(f"  {'Method':<15} {'ΔNmF2 (m⁻³)':>16} {'ΔhmF2 (km)':>12}")
        print("  " + "─" * 46)
        for m, dnm, dhm in zip(names, nm_biases, hm_biases):
            print(f"  {m:<15} {dnm:>+16.3e} {dhm:>+12.2f}")
    else:
        ax.text(0.5, 0.5, "No ISR profiles available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Posterior F2 Bias vs. ISR Truth (no data)", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plot_path = os.path.join(save_dir, f"opt_methods_{safe_key}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Optimisation comparison plot saved → {plot_path}")
    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# §D  Covariance plot (shared by KF and EnKF)
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# §E  ISR patch (same as demo_verification.py)
# ─────────────────────────────────────────────────────────────────────────────

_isr_profiles_compare: list[dict] = []
# Current group's IGS clean entries — set per group before calling process_group
# so the monkey-patched plot function can attach them to the summary figure.
_igs_entries_compare: list[dict] = []


def _patched_plot_group_compare(
    result, save_dir, group_key, *, suffix="", mode_label="Joint KF"
):
    isr_arg = None
    if suffix == "_joint" and _isr_profiles_compare:
        win = result.get("time_window", "")
        try:
            hhmm  = win.split("_")[-1]
            h_mid = int(hhmm[:2]) + int(hhmm[2:]) / 60.0
            half  = WINDOW_MINUTES / 120.0
            isr_arg = [
                p for p in _isr_profiles_compare
                if abs(p["hour_utc"] - h_mid) < half
            ]
            if not isr_arg:
                isr_arg = [min(
                    _isr_profiles_compare,
                    key=lambda p: min(
                        abs(p["hour_utc"] - h_mid),
                        24 - abs(p["hour_utc"] - h_mid),
                    ),
                )]
        except Exception:
            isr_arg = _isr_profiles_compare[:1]
    return _plot_group(
        result, save_dir, group_key,
        suffix=suffix, mode_label=mode_label,
        isr_profiles=isr_arg,
        isr_site=(ISR_LON_W, ISR_LAT) if isr_arg else None,
        igs_entries=_igs_entries_compare or None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §F  KF result caching
# ─────────────────────────────────────────────────────────────────────────────

def _kf_cache_path(cache_dir: str, group_key: str, kf_config: dict,
                   extra_tag: str = "") -> str:
    """Return the pickle path for a KF result, incorporating a config hash."""
    cfg_hash = hashlib.md5(
        pickle.dumps(kf_config, protocol=4)
    ).hexdigest()[:8]
    fname = f"kf_{group_key}_{cfg_hash}{extra_tag}.pkl"
    return os.path.join(cache_dir, fname)


def _load_kf_cache(path: str) -> dict | None:
    """Load a cached KF result dict, or return None if missing / corrupt."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            res = pickle.load(fh)
        print(f"  [KF cache] Loaded from {path}")
        return res
    except Exception as exc:
        print(f"  [KF cache] Could not load {path}: {exc} — re-running KF.")
        return None


def _save_kf_cache(path: str, res_kf: dict) -> None:
    """Persist a KF result dict to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(res_kf, fh, protocol=4)
    print(f"  [KF cache] Saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# §G  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def demo_compare_kf_enkf() -> None:
    """
    Run the KF vs. parametric EnKF comparison pipeline.

    Toggle flags
    ────────────
    RUN_SCIPY_OPT : bool  – set False to skip the 5-method scipy optimisation
                            (steps f & g) and save significant runtime.

    Steps
    ─────
    1.  Scan podTc2 files, filter to MH verification region.
    2.  Build the global IRI prior cache.
    3.  Load ISR truth profiles.
    4.  For each orbit group:
        a. Run standard joint KF  (process_group).
        b. Run parametric EnKF    (_run_parametric_enkf).
        c. Produce joint KF summary plot   (group_…_joint_kf.png).
        d. Produce joint EnKF summary plot (group_…_joint_enkf.png).
        e. Produce 2×2 comparison plot     (compare_…_2x2.png).
    5.  Print comparison statistics.
    """
    # ── User-configurable settings ─────────────────────────────────────────────
    RUN_SCIPY_OPT = False   # ← set False to skip scipy optimisation (steps f & g)

    DOY  = 154
    YYYY = 2025
    base_path = (
        f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/"
        f"{YYYY}.{DOY}/"
    )
    alt_grid    = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)
    TYPE        = "log"
    save_dir    = "./Figures/CompareKF_EnKF/"
    num_workers = 12
    kf_config   = {"measurement_err": 1.0, "relaxation": 0.99, "topside_follow_f2": True}

    # Parametric EnKF hyper-parameters
    # C_spatial (derived from IRI EDP ensemble) is used ONLY to seed the
    # Kronecker prior ensemble.  The ES-MDA update uses a pure GC taper on
    # ray–grid distances, set by loc_radius_km.
    
    # Assuming you have 8 parameters per grid point
    # Adjust this array to match the order of parameters in your state vector
    # Order: [NmF2(log), hmF2, H0, gamma, B0, B1, NmE(log), hmE]
    max_step_array = np.array([0.2, 20.0, 10.0, 0.1, 5.0, 0.1, 0.2, 5.0])
    
    # If you have n_grid points, tile this to the full state size
    n_grid = state.n_grid_points
    full_max_step_array = np.tile(max_step_array, n_grid)

    enkf_config = {
            "n_members":               200,
            "loc_radius_km":           100.0,
            "corr_length_km":          200.0,
            "inflation":               1.0, # <-- Handled manually below
            "sigma_obs_tecu":          10.0,
            "n_mda_iterations":        4,   # 4 iterations is standard for ES-MDA
            "max_update_step":         0.15, # <-- FIX: Restrict to ~40% max change per step
            "max_update_rays_per_arc": 100,
            "max_update_step": full_max_step_array, # Pass the array, not a float
        }

    isr_files = [
        "./DataFiles/EDPS/mlh250603m.002.nc",
    ]

    # ── IGS ground-station settings ───────────────────────────────────────────
    # Stations ordered closest-to-farthest from Millstone Hill ISR (~42.6°N,
    # ~288.5°E).  Absolute TEC arcs from all stations that succeed are pooled
    # and injected into both the standard KF and parametric EnKF alongside RO
    # occultations.  Set USE_IGS = False to disable this data source entirely.
    USE_IGS         = True
    IGS_STATIONS    = ["WES2", "BARH", "HLFX", "FRDN"]
    IGS_RINEX_VER   = 2   # these stations upload RINEX-2 Hatanaka (.{yy}d.gz)
    IGS_CACHE_DIR   = "./Data/IGS_RINEX/"
    IGS_LOCAL_OBS   = None   # set to a path to skip CDDIS download
    IGS_LOCAL_NAV   = None
    IGS_LOCAL_DCB   = None
    IGS_USE_IRI     = False
    IGS_MAX_RAYS    = 200    # max epochs per arc after decimation
    # ──────────────────────────────────────────────────────────────────────────

    if not os.path.exists(base_path):
        print(f"ERROR: base_path not found: {base_path}")
        return

    print("=" * 70)
    print("  demo_compare_kf_enkf.py — KF vs. Parametric EnKF Comparison")
    print("=" * 70)

    # ── Step 1: Scan and filter metadata ─────────────────────────────────────
    meta = scan_metadata(base_path)
    if meta.empty:
        print("No valid podTc2 files found.  Exiting.")
        return

    meta_verif = filter_to_verif_region(meta)
    if meta_verif.empty:
        print("No occultations in verification region.  Exiting.")
        return

    meta_verif = assign_orbit_groups(meta_verif)

    # ── Step 2: Global EDP prior cache ───────────────────────────────────────
    batch_date = pd.Timestamp(
        datetime.date(YYYY, 1, 1) + datetime.timedelta(days=DOY - 1)
    )
    global_edp_data_dir = f"./Data/Global_EDPS_{DOY}_{TYPE}/"
    print(f"\nBuilding global EDP cache for {batch_date.date()} …")
    global_edp_cache = build_daily_global_edps(
        batch_date, alt_grid,
        dLat=5.0, dLon=5.0,
        num_workers=num_workers,
        data_dir=global_edp_data_dir,
    )
    print("Global EDP cache ready.\n")

    # ── Step 3: Load ISR truth ────────────────────────────────────────────────
    isr_profiles: list[dict] = []
    if isr_files:
        isr_profiles = load_isr_profiles(isr_files)
    else:
        print("  [ISR] No ISR files configured — skipping ISR comparison.")

    global _isr_profiles_compare, _igs_entries_compare
    _isr_profiles_compare = isr_profiles
    _igs_entries_compare  = []          # will be updated per-group in the loop
    _demo_group._plot_group = _patched_plot_group_compare

    # ── Step 3b: Load IGS ground-station arcs for the full day ───────────────
    igs_all_arcs: list = []
    if USE_IGS:
        print(f"\nLoading IGS arcs for {batch_date.date()} "
              f"(stations: {', '.join(IGS_STATIONS)}) …")
        igs_all_arcs = _load_igs_arcs(
            date          = batch_date,
            stations      = IGS_STATIONS,
            cache_dir     = IGS_CACHE_DIR,
            rinex_version = IGS_RINEX_VER,
            local_obs     = IGS_LOCAL_OBS,
            local_nav     = IGS_LOCAL_NAV,
            local_dcb     = IGS_LOCAL_DCB,
            use_iri       = IGS_USE_IRI,
            max_rays      = IGS_MAX_RAYS,
        )
        print(f"IGS: {len(igs_all_arcs)} valid arcs for the day.\n")

        # Quick standalone sTEC + IPP plot for the full day's ground-station data.
        if igs_all_arcs:
            try:
                plot_igs_station_tec(
                    igs_entries = igs_all_arcs,
                    save_dir    = save_dir,
                    tag         = f"_{batch_date.strftime('%Y%j')}",
                )
            except Exception as _igs_plot_exc:
                print(f"  [warn] IGS sTEC plot failed: {_igs_plot_exc}")
    else:
        print("\n  [IGS] Ground-station data disabled (USE_IGS = False).\n")

    # ── Step 4: Process each orbit group ─────────────────────────────────────
    groups     = meta_verif.groupby("group_key", sort=True)
    group_keys = list(groups.groups.keys())
    os.makedirs(save_dir, exist_ok=True)

    comparison_stats: list[dict] = []

    for g_idx, gk in enumerate(group_keys):   # limit to first 2 groups
        print(f"\n{'─'*70}")
        print(f"  [{g_idx+1}/{len(group_keys)}]  {gk}")
        print(f"{'─'*70}")

        gm = groups.get_group(gk)

        # ── Filter IGS arcs to this group's time window ───────────────────────
        win_key            = gm["time_window"].iloc[0]
        igs_clean_window: list = []
        if USE_IGS and igs_all_arcs:
            igs_clean_window = _filter_igs_for_window(
                igs_all_arcs, win_key, window_minutes=WINDOW_MINUTES
            )
            print(f"  [IGS] {len(igs_clean_window)} arc(s) in window {win_key}")

        # ── (a) Standard KF (cached) ─────────────────────────────────────────
        # Cache key includes a tag when IGS arcs are present so runs with and
        # without ground-station data do not overwrite each other.
        _igs_tag   = f"_igsn{len(igs_clean_window)}" if igs_clean_window else ""
        _kf_pkl = _kf_cache_path(
            os.path.join(save_dir, ".kf_cache"), gk, kf_config,
            extra_tag=_igs_tag,
        )
        # Expose current IGS window to the monkey-patched plot function.
        _igs_entries_compare = igs_clean_window

        res_kf = _load_kf_cache(_kf_pkl)
        if res_kf is None:
            print("\n  Running standard joint KF …")
            from demo_verification import _process_verif_group
            res_kf = _process_verif_group(
                group_key        = gk,
                group_meta       = gm,
                alt_grid         = alt_grid,
                global_edp_cache = global_edp_cache,
                generate_plots   = False,
                save_dir         = save_dir,
                extra_clean_list = igs_clean_window or None,
                **kf_config,
            )
            if res_kf.get("status") == "Success":
                _save_kf_cache(_kf_pkl, res_kf)

        if res_kf.get("status") != "Success":
            print(f"  [warn] KF failed for group {gk}: {res_kf.get('status')}")
            continue

        # ── (a2) KF arc innovation diagnostic ────────────────────────────────
        try:
            _kf_stats = _arc_stats_from_tec_slices(
                tec_slices = res_kf.get("joint_tec_slices", res_kf["tec_slices"]),
                clean_list = res_kf["clean_list"],
                sat_ids    = res_kf.get("sat_ids", []),
            )
            _plot_arc_innovation_diagnostic(
                arc_labels     = _kf_stats["arc_labels"],
                arc_prior_mean = _kf_stats["arc_prior_mean"],
                arc_post_mean  = _kf_stats["arc_post_mean"],
                arc_prior_rmse = _kf_stats["arc_prior_rmse"],
                arc_post_rmse  = _kf_stats["arc_post_rmse"],
                arc_lats       = _kf_stats["arc_lats"],
                arc_lons       = _kf_stats["arc_lons"],
                all_prior      = _kf_stats["all_prior"],
                all_post_main  = _kf_stats["all_post"],
                group_key      = gk,
                save_dir       = save_dir,
                filter_name    = "KF",
                prior_rmse     = float(res_kf.get("prior_tec_rmse", np.nan)),
                post_rmse      = float(res_kf.get("joint_post_tec_rmse",
                                                   res_kf.get("post_tec_rmse", np.nan))),
            )
        except Exception as _exc:
            print(f"  [warn] KF arc diagnostic failed: {_exc}")

        # ── (b) Parametric EnKF ──────────────────────────────────────────────
        print("\n  Running parametric EnKF …")
        try:
            res_enkf = _run_parametric_enkf(
                res_kf    = res_kf,
                alt_grid  = alt_grid,
                save_dir  = save_dir,
                group_key = gk,
                **enkf_config,
            )
        except Exception as exc:
            print(f"  [warn] EnKF failed: {exc}")
            import traceback; traceback.print_exc()
            continue

        # ── Select ISR sweeps for this time window ────────────────────────────
        isr_win: list[dict] = []
        if isr_profiles:
            win = res_kf.get("time_window", "")
            try:
                hhmm  = win.split("_")[-1]
                h_mid = int(hhmm[:2]) + int(hhmm[2:]) / 60.0
                half  = WINDOW_MINUTES / 120.0
                isr_win = [p for p in isr_profiles if abs(p["hour_utc"] - h_mid) < half]
                if not isr_win:
                    isr_win = [min(
                        isr_profiles,
                        key=lambda p: min(
                            abs(p["hour_utc"] - h_mid),
                            24 - abs(p["hour_utc"] - h_mid),
                        ),
                    )]
            except Exception:
                isr_win = isr_profiles[:1]

        # ── (c) Joint KF summary plot ─────────────────────────────────────────
        print("\n  Writing joint KF summary plot …")
        # Temporarily set post_tec_rmse to joint value for the title
        res_kf_jnt = dict(res_kf)
        res_kf_jnt["post_tec_rmse"]  = res_kf.get("joint_post_tec_rmse",
                                                    res_kf["post_tec_rmse"])
        res_kf_jnt["post_edp_3d"]    = res_kf.get("joint_post_edp_3d",
                                                    res_kf["post_edp_3d"])
        res_kf_jnt["tec_slices"]     = res_kf.get("tec_slices_joint",
                                                    res_kf["tec_slices"])
        try:
            kf_plot_path = _plot_group(
                result     = res_kf_jnt,
                save_dir   = save_dir,
                group_key  = f"{gk}_kf",
                suffix     = "_joint",
                mode_label = "Joint KF",
                isr_profiles = isr_win or None,
                isr_site   = (ISR_LON_W, ISR_LAT) if isr_win else None,
                igs_entries  = igs_clean_window or None,
            )
            print(f"  KF joint plot → {kf_plot_path}")
        except Exception as exc:
            print(f"  [warn] KF joint plot failed: {exc}")

        # ── (d) Joint EnKF summary plot ───────────────────────────────────────
        print("\n  Writing joint EnKF summary plot …")
        try:
            enkf_plot_path = _plot_group(
                result     = res_enkf,
                save_dir   = save_dir,
                group_key  = f"{gk}_enkf",
                suffix     = "_joint",
                mode_label = "Parametric EnKF",
                isr_profiles = isr_win or None,
                isr_site   = (ISR_LON_W, ISR_LAT) if isr_win else None,
                igs_entries  = igs_clean_window or None,
            )
            print(f"  EnKF joint plot → {enkf_plot_path}")
        except Exception as exc:
            print(f"  [warn] EnKF joint plot failed: {exc}")

        # ── (e) 2×2 comparison plot ───────────────────────────────────────────
        print("\n  Writing 2×2 comparison plot …")
        try:
            cmp_path = plot_kf_enkf_comparison(
                res_kf       = res_kf_jnt,
                res_enkf     = res_enkf,
                isr_profiles = isr_win,
                alt_grid     = alt_grid,
                group_key    = gk,
                save_dir     = save_dir,
            )
        except Exception as exc:
            print(f"  [warn] Comparison plot failed: {exc}")
            import traceback; traceback.print_exc()

        # ── (f) Parametric optimisation (5 scipy methods) ────────────────────
        opt_results_gk: dict[str, dict] = {}
        if RUN_SCIPY_OPT:
            print("\n  Running parametric scipy optimisation (5 methods) …")
            try:
                opt_results_gk = _run_parametric_optimization(
                    res_kf         = res_kf,
                    alt_grid       = alt_grid,
                    save_dir       = save_dir,
                    group_key      = gk,
                    sigma_obs_tecu = enkf_config.get("sigma_obs_tecu", 1.0),
                    n_update_rays  = enkf_config.get("n_update_rays", 5),
                )
            except Exception as exc:
                print(f"  [warn] Optimisation failed: {exc}")
                import traceback; traceback.print_exc()
        else:
            print("\n  [skip] scipy optimisation disabled (RUN_SCIPY_OPT = False)")

        # ── (g) Optimisation methods comparison plot ──────────────────────────
        if RUN_SCIPY_OPT and opt_results_gk:
            print("\n  Writing optimisation methods comparison plot …")
            try:
                opt_plot_path = plot_optimization_methods_comparison(
                    res_kf       = res_kf_jnt,
                    res_enkf     = res_enkf,
                    opt_results  = opt_results_gk,
                    isr_profiles = isr_win,
                    alt_grid     = alt_grid,
                    group_key    = gk,
                    save_dir     = save_dir,
                )
                print(f"  Optimisation plot → {opt_plot_path}")
            except Exception as exc:
                print(f"  [warn] Optimisation comparison plot failed: {exc}")
                import traceback; traceback.print_exc()

        # ── Collect statistics ────────────────────────────────────────────────
        verts_geo = res_kf["eds_occ"].geolocation
        n_alt_    = len(alt_grid)
        n_geo_    = verts_geo.shape[0]
        idx_mh    = millstone_vertex_idx(verts_geo)

        _jnt = res_kf.get("joint_post_edp_3d")
        kf_post_mh = np.asarray(
            _jnt if _jnt is not None else res_kf["post_edp_3d"]
        ).reshape(n_alt_, n_geo_)[:, idx_mh]
        enkf_post_mh = np.asarray(res_enkf["post_edp_3d"]).reshape(n_alt_, n_geo_)[:, idx_mh]

        nm_kf,  hm_kf  = extract_robust_f2_peak(kf_post_mh,   alt_grid)
        nm_en,  hm_en  = extract_robust_f2_peak(enkf_post_mh, alt_grid)

        isr_nm = np.nanmean([p["nm_f2"] for p in isr_win]) if isr_win else np.nan
        isr_hm = np.nanmean([p["hm_f2"] for p in isr_win]) if isr_win else np.nan

        # Collect optimisation RMSEs for the summary table
        opt_post_rmses: dict[str, float] = {
            m: opt_results_gk[m]["post_tec_rmse"]
            for m in opt_results_gk
        }

        comparison_stats.append({
            "group_key":        gk,
            "kf_prior_rmse":    res_kf["prior_tec_rmse"],
            "kf_post_rmse":     res_kf.get("joint_post_tec_rmse", res_kf["post_tec_rmse"]),
            "enkf_prior_rmse":  res_enkf["prior_tec_rmse"],
            "enkf_post_rmse":   res_enkf["post_tec_rmse"],
            "opt_post_rmses":   opt_post_rmses,
            "kf_nm_bias":       nm_kf  - isr_nm if not np.isnan(nm_kf)  else np.nan,
            "kf_hm_bias":       hm_kf  - isr_hm if not np.isnan(hm_kf)  else np.nan,
            "enkf_nm_bias":     nm_en  - isr_nm if not np.isnan(nm_en)  else np.nan,
            "enkf_hm_bias":     hm_en  - isr_hm if not np.isnan(hm_en)  else np.nan,
        })

    # ── Step 5: Print comparison table ───────────────────────────────────────
    print("\n" + "=" * 78)
    print("  KF vs. EnKF vs. Optimisation — Comparison Summary")
    print("=" * 78)
    opt_method_names = list(_OPT_METHOD_STYLES.keys())
    hdr_opt = "".join(f" {m[:10]:>12}" for m in opt_method_names)
    print(f"  {'Group':<40} {'KF prior':>9} {'KF post':>9} "
          f"{'EnKF post':>10}{hdr_opt}")
    print(f"  {'':40} {'RMSE':>9} {'RMSE':>9} {'RMSE':>10}"
          + "".join(f" {'RMSE':>12}" for _ in opt_method_names) + "  (TECU)")
    print("  " + "─" * 76)
    for s in comparison_stats:
        gk_short = s["group_key"][-40:] if len(s["group_key"]) > 40 else s["group_key"]
        opt_vals = "".join(
            f" {s['opt_post_rmses'].get(m, float('nan')):>12.3f}"
            for m in opt_method_names
        )
        print(
            f"  {gk_short:<40}"
            f"  {s['kf_prior_rmse']:>7.3f}"
            f"  {s['kf_post_rmse']:>7.3f}"
            f"  {s['enkf_post_rmse']:>9.3f}"
            + opt_vals
        )

    if any(not np.isnan(s["kf_nm_bias"]) for s in comparison_stats):
        print("\n  F2-peak biases vs. ISR (posterior at Millstone Hill vertex):")
        print(f"  {'Group':<45} {'KF ΔNmF2':>10} {'EnKF ΔNmF2':>12} "
              f"{'KF Δhm':>8} {'EnKF Δhm':>10}")
        print("  " + "─" * 68)
        for s in comparison_stats:
            gk_short = s["group_key"][-45:] if len(s["group_key"]) > 45 else s["group_key"]
            print(
                f"  {gk_short:<45}"
                f"  {s['kf_nm_bias']:>+10.2e}"
                f"  {s['enkf_nm_bias']:>+12.2e}"
                f"  {s['kf_hm_bias']:>+8.1f}"
                f"  {s['enkf_hm_bias']:>+10.1f}"
            )

    print("\nAll comparison figures written.  Done.")

    # Restore the real _plot_group so repeated calls don't accumulate patches
    _demo_group._plot_group = _plot_group


# ─────────────────────────────────────────────────────────────────────────────
def compare_iri_vs_parametric(
    time_in: pd.Timestamp | str,
    lat: float,
    lon: float,
    alt_min_km: float = 80.0,
    alt_max_km: float = 800.0,
    n_alt: int = 200,
    save_path: str | None = None,
) -> None:
    """
    Call IRI for a single location/time, extract the parametric state via
    _state_from_iri_direct, reconstruct the profile with _ne_profile_ensemble,
    and plot the two EDPs side-by-side.

    Parameters
    ----------
    time_in   : datetime-like or "YYYY-MM-DD HH:MM"
    lat, lon  : geographic latitude / longitude (degrees)
    alt_min_km, alt_max_km : altitude range for the comparison
    n_alt     : number of altitude points
    save_path : if given, save the figure here; otherwise show interactively
    """
    if isinstance(time_in, str):
        time_in = pd.Timestamp(time_in)

    alt_grid = np.linspace(alt_min_km, alt_max_km, n_alt)

    # ── 1. Fortran IRI2020 (same driver as EDPSamples) ────────────────────────
    print(f"Calling Fortran IRI2020  t={time_in}  lat={lat:.2f}  lon={lon:.2f} …")
    sampling_df = _solar_sampling_df(time_in)
    ne_iri, feat = _get_iri_edp_and_features(time_in, lat, lon, alt_grid, sampling_df)
    f107 = float(sampling_df["f107"].iloc[0])

    feat_labels = ("NmF2","hmF2","NmF1","hmF1","NmE","hmE","NmD","hmD",
                   "hhalf","B0","valley_base","valley_top","B1")
    print("  IRI Fortran features:")
    for label, val in zip(feat_labels, feat):
        print(f"    {label:<14} {val:.4g}")

    # ── 2. Build parametric state (H0 seeded from half-power point, γ fitted) ─
    params_log = _state_from_iri_direct(ne_iri, feat, alt_grid)

    params_lin = params_log.copy()
    params_lin[I_LOG_NMF2] = 10.0 ** params_log[I_LOG_NMF2]
    params_lin[I_LOG_NME]  = 10.0 ** params_log[I_LOG_NME]
    ne_param = _ne_profile_ensemble(alt_grid, params_lin[:, np.newaxis])[:, 0]

    # ── 4. Diagnostics (skip NaN IRI points — IRI clips to NaN above ~700 km) ─
    valid_all = np.isfinite(ne_iri) & (ne_iri > 0)
    log_rmse_all = float(np.sqrt(np.nanmean(
        (np.log10(np.maximum(ne_param[valid_all], 1))
         - np.log10(np.maximum(ne_iri[valid_all], 1)))**2
    )))
    hmF2 = params_log[I_HMF2]
    valid_top = (alt_grid > hmF2) & valid_all
    valid_bot = (alt_grid > 100) & (alt_grid <= hmF2) & valid_all
    def _rmse_valid(mask):
        if mask.sum() < 2:
            return float("nan")
        return float(np.sqrt(np.mean(
            (np.log10(np.maximum(ne_param[mask], 1))
             - np.log10(np.maximum(ne_iri[mask], 1)))**2
        )))
    rmse_top = _rmse_valid(valid_top)
    rmse_bot = _rmse_valid(valid_bot)

    print(f"\n  Parametric state vector:")
    for name, val in zip(PARAM_NAMES, params_log):
        print(f"    {name:<18} {val:.4f}")
    print(f"\n  log₁₀-RMSE  overall={log_rmse_all:.4f}"
          f"  topside={rmse_top:.4f}  bottomside={rmse_bot:.4f}")

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    fig.suptitle(
        f"IRI ne  vs  parametric (H0 from PyIRI, γ fitted)\n"
        f"{time_in.strftime('%Y-%m-%d %H:%M UT')}   "
        f"({lat:+.1f}°N  {lon:+.1f}°E)   F10.7={f107:.0f}",
        fontsize=10,
    )

    kw_iri   = dict(color="tab:blue",   lw=1.8)
    kw_param = dict(color="tab:orange", lw=1.5, ls="--")

    # Left panel — linear scale
    ax = axes[0]
    ax.plot(ne_iri   / 1e12, alt_grid, label="IRI (Fortran)", **kw_iri)
    ax.plot(ne_param / 1e12, alt_grid, label="parametric",    **kw_param)
    ax.axhline(hmF2, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Ne  [×10¹² m⁻³]")
    ax.set_ylabel("Altitude  [km]")
    ax.set_title("Linear scale")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Right panel — log scale
    ax = axes[1]
    ax.semilogx(np.maximum(ne_iri,   1), alt_grid, label="IRI (Fortran)", **kw_iri)
    ax.semilogx(np.maximum(ne_param, 1), alt_grid, label="parametric",    **kw_param)
    ax.axhline(hmF2, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Ne  [m⁻³]  (log scale)")
    ax.set_title(f"Log  RMSE top={rmse_top:.3f}  bot={rmse_bot:.3f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(bottom=0)

    # Colour-coded parameter readout for the fitted parametric state.
    _draw_param_boxes(ax, [("parametric", kw_param["color"], params_log)],
                       loc="lower right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    # Uncomment to run the IRI vs parametric diagnostic instead of the full comparison:
    compare_iri_vs_parametric(
        time_in = "2025-01-01 12:00",
        lat     = 42.6,    # Millstone Hill
        lon     = -71.5,
        save_path = "./Figures/iri_vs_parametric_test.png",
    )
    demo_compare_kf_enkf()
