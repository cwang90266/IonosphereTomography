#!/usr/bin/env python3
"""
demo_ground_station_kf.py
=========================
Kalman Filter tomographic assimilation of ground-station GNSS sTEC from IGS RINEX data.

Pipeline
--------
1.  Load IGS arcs (runs demo_ground_station.py acquisition pipeline, with cache).
2.  Define a regular lat/lon grid of ionospheric state columns over the region.
3.  For each 90-minute assimilation window:
      a)  IRI mean state + background covariance at the window centre time.
      b)  Filter arcs to this window; build GNSS→receiver ray paths.
      c)  Build linear observation operator H; compute prior sTEC.
      d)  Run standard Kalman Filter update (single step).
      e)  Compute posterior sTEC and EDP ensemble.
4.  Produce four figure types:
      (a) Regional map — NmF2 change (prior → posterior) at each grid point.
      (b) 2×2 constellation TEC time series (measured / prior / posterior).
      (c) Pass-by-pass arc innovation diagnostic (bar / scatter / map / KDE).
      (d) Prior / posterior / ΔNe EDP profiles with ±1σ uncertainty bands.

Usage
-----
    python demo_ground_station_kf.py

NASA Earthdata credentials in ~/.netrc are required for CDDIS downloads.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "iri2020_new" / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
import pyproj
import scipy.linalg as la
from scipy.spatial import cKDTree

# ── Project infrastructure ────────────────────────────────────────────────────
from TEC_model.igs_tec_pipeline import (
    RinexDownloader,
    DCBCorrector,
    IGSTECPipeline,
    BroadcastEphemeris,
    igs_obs_to_clean_entry,
    report_freq_dcb_coverage,
)
from EDPSamples.edp_samples import get_IRI2020_EDP
from IRI_Sample_Inputs.IRI_Sample_inputs import IRI_Sample_Inputs
from Ionosphere_Tomography_Inverter.enkf_update import _haversine_km
from Ionosphere_Tomography_Inverter.srif_batch_update import SRIFBatchUpdate
from Ionosphere_Tomography_Inverter.info_batch_update import InfoBatchUpdate

import cartopy.crs as ccrs
import cartopy.feature as cfeature
_CARTOPY = True

# ── ECEF → geodetic transformer ───────────────────────────────────────────────
_TRANSFORMER = pyproj.Transformer.from_crs(
    pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
    pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
    always_xy=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# §0  Configuration
# ─────────────────────────────────────────────────────────────────────────────

CAMPAIGN_DATE      = datetime(2026, 6, 2)
POI_LAT            = 50.0          # °N — centre of region
POI_LON            = 10.0          # °E
SEARCH_RADIUS_DEG  = 3.0          # great-circle search radius for stations

WINDOW_MIN         = 90            # assimilation window width (minutes)
RINEX_VERSION      = 3
RINEX_CACHE        = str(ROOT / "Data" / "RINEX_Cache")
IGS_JSON           = str(ROOT / "Data" / "IGS_Stations" / "IGSNetwork.json")
SAVE_DIR           = str(ROOT / "Figures" / "Demo_Ground_Station_KF")

MAX_RAYS_PER_ARC   = 20           # downsample per arc before KF
MIN_VALID_RAYS     = 20            # reject arcs shorter than this
NUM_SV_WORKERS     = 10
VERBOSE            = False         # suppress per-SV output during data load
EPHEM_STRIDE       = 0             # auto

# DCB product selection — controls which GNSS DCB SINEX file is downloaded.
# 'CODE' : CODE MGEX Final/Rapid (COD0MGXFIN / COD0MGXRAP) from the
#          cddis.nasa.gov/archive/gnss/products/mgex/{gpsweek}/ tree.
#          Better GPS–Galileo inter-constellation calibration; Final product
#          has ~4-week latency, Rapid ~1–2 days.  Falls back to CAS RAPID
#          automatically if CODE files are not yet available.
# 'CAS'  : CAS RAPID (CAS0OPSRAP) from products/bias/{year}/ — legacy default.
DCB_PRODUCT        = 'CODE'

# Kalman Filter parameters
CORR_LENGTH_KM     = 400.0         # horizontal spatial correlation length (km)
V_CORR_KM          = 100.0         # vertical altitude correlation length (km)
SIGMA_NE_FLOOR     = 0.20          # minimum fractional Ne background uncertainty
SIGMA_OBS_TECU     = 5.0           # observation noise std-dev (TECU)

# Solver selection
# USE_SRIF = True  → SRIF batch update (RAM-efficient; never forms full H)
# USE_SRIF = False → standard information-form Kalman update (run_kf_window)
USE_SRIF           = False
SRIF_CHUNK_SIZE    = 512           # max H rows per Householder QR step

# Grid parameters — regular lat/lon grid covering the region
GRID_DLAT          = 2           # grid spacing (degrees)
GRID_DLON          = 2
GRID_PAD_DEG       = 2.5           # extra pad beyond search radius

# Altitude grid for IRI / Ne profile evaluation
ALT_GRID_KM        = np.arange(80.0, 1010.0, 20.0)   # 80–1000 km, 20 km step

# Constellation display config — 2×2 panel layout
CONST_CFG: dict[str, dict] = {
    "G": {"name": "GPS",     "color": "steelblue",    "panel": (0, 0)},
    "E": {"name": "Galileo", "color": "darkorange",   "panel": (0, 1)},
    "R": {"name": "GLONASS", "color": "mediumpurple", "panel": (1, 0)},
    "C": {"name": "BeiDou",  "color": "seagreen",     "panel": (1, 1)},
}


# ─────────────────────────────────────────────────────────────────────────────
# §1  Ground-station TEC data acquisition (mirrors demo_ground_station.py)
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return math.degrees(2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def _find_nearby_stations(poi_lat, poi_lon, radius_deg, json_path):
    with open(json_path) as fh:
        network = json.load(fh)
    nearby = []
    for name, info in network.items():
        try:
            lat = float(info["Latitude"])
            lon = float(info["Longitude"])
        except (KeyError, ValueError):
            continue
        dist = _haversine_deg(poi_lat, poi_lon, lat, lon)
        if dist <= radius_deg:
            nearby.append({"name": name, "code": name[:4], "lat": lat, "lon": lon,
                           "dist_deg": dist, "info": info})
    return sorted(nearby, key=lambda s: s["dist_deg"])


def _resample_clean_entry(ce: dict, max_rays: int) -> dict:
    """
    Uniformly subsample all per-epoch arrays in a clean entry to at most
    ``max_rays`` samples.

    Any ndarray whose last axis length equals the arc's epoch count is
    resampled; scalars and shorter arrays are passed through unchanged.
    If the arc already has ≤ ``max_rays`` epochs the dict is returned as-is
    (no copy).

    This is applied to cache-loaded arcs so that changing ``MAX_RAYS_PER_ARC``
    between runs takes effect without having to re-run the full pipeline.
    """
    n_s = len(ce["tec"])
    if n_s <= max_rays:
        return ce

    stride = int(np.ceil(n_s / max_rays))
    idx    = np.arange(0, n_s, stride)

    ce_new = {}
    for key, val in ce.items():
        if isinstance(val, np.ndarray):
            if val.ndim == 1 and val.shape[0] == n_s:
                ce_new[key] = val[idx]
            elif val.ndim == 2 and val.shape[-1] == n_s:
                ce_new[key] = val[:, idx]
            else:
                ce_new[key] = val          # scalar-like or different shape
        else:
            ce_new[key] = val              # non-array (str, float, Timestamp…)
    return ce_new


def load_igs_arcs(cache_pickle: str | None = None) -> tuple[list, list]:
    """
    Load (obs_all, clean_all) from a pickle cache if it exists, else run
    the full acquisition pipeline (download + carrier-phase levelling + DCB).

    Returns
    -------
    obs_all   : list of raw IGS obs dicts (one per accepted arc).
    clean_all : list of clean_entry dicts (same order).
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    pkl = cache_pickle or os.path.join(SAVE_DIR, "_igs_arcs_cache.pkl")

    if os.path.exists(pkl):
        print(f"  Loading IGS arcs from cache: {pkl}")
        with open(pkl, "rb") as fh:
            obs_all, clean_all = pickle.load(fh)
        # Re-apply the current MAX_RAYS_PER_ARC cap in case the cache was built
        # with a different value or the setting has changed since caching.
        n_before = sum(len(ce["tec"]) for ce in clean_all)
        clean_all = [_resample_clean_entry(ce, MAX_RAYS_PER_ARC)
                     for ce in clean_all]
        n_after = sum(len(ce["tec"]) for ce in clean_all)
        print(f"  Loaded {len(clean_all)} arcs "
              f"({n_before} → {n_after} total epochs after resampling "
              f"to MAX_RAYS_PER_ARC={MAX_RAYS_PER_ARC}).")
        return obs_all, clean_all

    print("  IGS cache not found — running acquisition pipeline …")
    stations = _find_nearby_stations(POI_LAT, POI_LON, SEARCH_RADIUS_DEG, IGS_JSON)
    print(f"  Found {len(stations)} stations within {SEARCH_RADIUS_DEG}° of "
          f"({POI_LAT:.0f}°N, {POI_LON:.0f}°E)")

    dl = RinexDownloader(cache_dir=RINEX_CACHE)
    dcb_path = dl.dcb_sinex(CAMPAIGN_DATE, product=DCB_PRODUCT)
    dcb = DCBCorrector(dcb_path) if dcb_path else None
    if dcb:
        sta_keys = {k.upper() for k in dcb._sta_dcb}
        stations = [s for s in stations
                    if s["code"].upper() in sta_keys or s["name"].upper() in sta_keys]
        print(f"  {len(stations)} stations have Rx DCB entries")

    # ── Mixed nav file: download once and share across all stations ───────────
    # The BRDM/BRDC mixed nav file is day-specific and identical for every
    # station.  Build BroadcastEphemeris once so the O(station) re-parses
    # of the large nav file are avoided.
    shared_ephem: "BroadcastEphemeris | None" = None
    try:
        print("  Downloading mixed nav file (shared across all stations) …",
              flush=True)
        nav_path_shared = dl.nav_file("BRDM", CAMPAIGN_DATE, RINEX_VERSION)
        print(f"  Building BroadcastEphemeris from {nav_path_shared.name} …",
              flush=True)
        shared_ephem = BroadcastEphemeris(nav_path_shared)
        print(f"  Shared ephemeris ready: {len(shared_ephem._cache)} SVs cached",
              flush=True)
    except Exception as exc:
        print(f"  [warn] Could not build shared ephemeris ({exc}); "
              "each station will fall back to its own nav file.", flush=True)

    obs_all, clean_all = [], []
    _dcb_coverage_printed = False   # print diagnostic once, for the first station
    for i, sta in enumerate(stations, 1):
        code = sta["code"]
        print(f"\n[{i}/{len(stations)}] {code} …", flush=True)
        try:
            obs_path = dl.obs_file(code, CAMPAIGN_DATE, RINEX_VERSION)
            # Only download a per-station nav file when the shared ephem is
            # unavailable (graceful fallback).
            if shared_ephem is None:
                nav_path = dl.nav_file(code, CAMPAIGN_DATE, RINEX_VERSION)
            else:
                nav_path = nav_path_shared   # already cached; used only if needed
        except Exception as exc:
            print(f"  SKIP (download failed): {exc}"); continue

        # ── Frequency / DCB coverage diagnostic (printed once) ────────────────
        if not _dcb_coverage_printed:
            print(f"\n  [Coverage diagnostic] Checking obs codes vs DCB pairs "
                  f"for representative station {code} …", flush=True)
            try:
                report_freq_dcb_coverage(obs_path, dcb, station=code)
            except Exception as _exc:
                print(f"  [Coverage diagnostic] Failed: {_exc}")
            _dcb_coverage_printed = True
        try:
            pipe = IGSTECPipeline(
                station        = code,
                date           = CAMPAIGN_DATE,
                rinex_version  = RINEX_VERSION,
                cache_dir      = RINEX_CACHE,
                use_iri        = False,
                local_obs      = str(obs_path),
                local_nav      = str(nav_path),
                local_dcb      = str(dcb_path) if dcb_path else None,
                dcb_product    = DCB_PRODUCT,
                ephem_stride   = EPHEM_STRIDE,
                show_progress  = True,
                verbose        = VERBOSE,
                num_sv_workers = NUM_SV_WORKERS,
                shared_ephem   = shared_ephem,   # None → pipeline builds its own
            )
            arcs = pipe.run()
        except Exception as exc:
            print(f"  SKIP (pipeline failed): {exc}"); continue

        n_acc = 0
        for obs in arcs:
            entry = igs_obs_to_clean_entry(obs, max_rays=MAX_RAYS_PER_ARC,
                                            min_valid=MIN_VALID_RAYS)
            if entry is not None:
                obs_all.append(obs)
                clean_all.append(entry)
                n_acc += 1
        print(f"  {n_acc}/{len(arcs)} arcs accepted")

    print(f"\nTotal arcs accepted: {len(clean_all)}")
    with open(pkl, "wb") as fh:
        pickle.dump((obs_all, clean_all), fh, protocol=4)
    print(f"  Saved to {pkl}")
    return obs_all, clean_all


# ─────────────────────────────────────────────────────────────────────────────
# §1b  GLONASS diagnostic summary
# ─────────────────────────────────────────────────────────────────────────────

def print_glonass_diagnostics(obs_all: list[dict]) -> None:
    """Print a per-(station, PRN) summary for every GLONASS arc.

    Columns:
      STA   PRN   k   f1 (G1) MHz   f2 (G2) MHz  code_A  code_B   Tx DCB   Rx DCB  Total
    k is the FDMA channel number back-computed from f1_hz:
      k = round((f1_hz - 1602e6) / 0.5625e6)
    DCBs are in TECU.
    """
    glo_arcs = [o for o in obs_all if o.get("conid") == "R"]
    if not glo_arcs:
        print("  No GLONASS arcs to report.")
        return

    seen: dict[tuple, dict] = {}
    for obs in glo_arcs:
        sta = obs.get("station_id", "????")
        prn = "R" + str(obs.get("prn_id", "??"))
        key = (sta, prn)
        if key not in seen:
            seen[key] = obs

    print(f"\n  {'STA':<6} {'PRN':<5} {'k':>3}  "
          f"{'f1 (G1) MHz':>12}  {'f2 (G2) MHz':>12}  "
          f"{'code_A':<7} {'code_B':<7}  "
          f"{'Tx DCB':>9}  {'Rx DCB':>9}  {'Total':>9}")
    print("  " + "-" * 86)

    for (sta, prn), obs in sorted(seen.items()):
        f1 = float(obs.get("f1_hz") or 0.0)
        f2 = float(obs.get("f2_hz") or 0.0)
        k  = round((f1 - 1602e6) / 0.5625e6) if f1 else 0
        cA = obs.get("code_obs_A", "?")
        cB = obs.get("code_obs_B", "?")
        tx = float(obs.get("dcb_sv_tecu") or 0.0)
        rx = float(obs.get("dcb_rx_tecu") or 0.0)
        print(f"  {sta:<6} {prn:<5} {k:>+3}  "
              f"{f1/1e6:>12.4f}  {f2/1e6:>12.4f}  "
              f"{cA:<7} {cB:<7}  "
              f"{tx:>+9.4f}  {rx:>+9.4f}  {tx+rx:>+9.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# §2  Grid and time-window utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(poi_lat: float, poi_lon: float,
               radius_deg: float, pad_deg: float,
               dlat: float, dlon: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Regular lat/lon grid covering the region.
    Returns (grid_lats, grid_lons), each shape (n_grid,).
    """
    lat_min = poi_lat - radius_deg - pad_deg
    lat_max = poi_lat + radius_deg + pad_deg
    lon_min = poi_lon - radius_deg - pad_deg
    lon_max = poi_lon + radius_deg + pad_deg
    lats = np.arange(lat_min, lat_max + dlat * 0.5, dlat)
    lons = np.arange(lon_min, lon_max + dlon * 0.5, dlon)
    lon_g, lat_g = np.meshgrid(lons, lats)
    return lat_g.ravel(), lon_g.ravel()


def build_grid_from_bounds(lat_min: float, lat_max: float,
                            lon_min: float, lon_max: float,
                            dlat: float, dlon: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Regular lat/lon grid covering explicit bounds.
    Returns (grid_lats, grid_lons), each shape (n_grid,).
    """
    lats = np.arange(lat_min, lat_max + dlat * 0.5, dlat)
    lons = np.arange(lon_min, lon_max + dlon * 0.5, dlon)
    lon_g, lat_g = np.meshgrid(lons, lats)
    return lat_g.ravel(), lon_g.ravel()


def window_centres(date: datetime, window_min: int = WINDOW_MIN
                   ) -> list[pd.Timestamp]:
    """
    Return list of UTC Timestamps at the centre of each 90-min window for *date*.
    """
    half = pd.Timedelta(minutes=window_min / 2)
    day  = pd.Timestamp(date.year, date.month, date.day)
    centres = []
    t = day + half
    while t < day + pd.Timedelta(days=1):
        centres.append(t)
        t += pd.Timedelta(minutes=window_min)
    return centres


def filter_arcs_for_window(clean_all: list, t_centre: pd.Timestamp,
                            window_min: int = WINDOW_MIN) -> list:
    """Return arcs whose 'date' falls within ±window_min/2 of t_centre."""
    half    = pd.Timedelta(minutes=window_min / 2)
    t_start = t_centre - half
    t_end   = t_centre + half
    out = []
    for ce in clean_all:
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


def filter_arcs_for_region(
    clean_list: list,
    grid_lats:  np.ndarray,
    grid_lons:  np.ndarray,
    pad_deg:    float = 0.5,
) -> tuple[list, int]:
    """
    Keep only arcs whose ionospheric pierce-point (IPP) falls inside the
    grid bounding box (with an optional pad).

    An arc is accepted when the **mean** of its finite IPP samples lies within:

        [lat_min − pad, lat_max + pad] × [lon_min − pad, lon_max + pad]

    The mean IPP is drawn from the per-epoch ``ipp_lat`` / ``ipp_lon`` arrays
    if present; otherwise the arc-level ``lat_tecmax_tangent`` /
    ``lon_tecmax_tangent`` scalar is used as a fallback.

    Arcs with no recoverable IPP coordinates are silently dropped.

    Parameters
    ----------
    clean_list : list of clean-entry dicts (e.g. from filter_arcs_for_window).
    grid_lats  : (n_grid,) array — latitude coordinates of all grid columns.
    grid_lons  : (n_grid,) array — longitude coordinates of all grid columns.
    pad_deg    : extra margin beyond the strict grid bbox (degrees).  Default 0.5°.

    Returns
    -------
    (accepted, n_rejected)
        accepted    — filtered list, subset of clean_list.
        n_rejected  — number of arcs dropped for being outside the region.
    """
    lat_min = float(grid_lats.min()) - pad_deg
    lat_max = float(grid_lats.max()) + pad_deg
    lon_min = float(grid_lons.min()) - pad_deg
    lon_max = float(grid_lons.max()) + pad_deg

    accepted   = []
    n_rejected = 0

    for ce in clean_list:
        # Prefer the per-epoch IPP arrays (more representative for long arcs)
        ipp_lat = ce.get("ipp_lat")
        ipp_lon = ce.get("ipp_lon")

        if (ipp_lat is not None and len(ipp_lat) > 0
                and np.any(np.isfinite(ipp_lat))):
            mean_lat = float(np.nanmean(ipp_lat))
            mean_lon = float(np.nanmean(ipp_lon))
        else:
            # Fallback: arc-level tangent/IPP scalar
            mean_lat = float(ce.get("lat_tecmax_tangent", np.nan))
            mean_lon = float(ce.get("lon_tecmax_tangent", np.nan))

        if not (np.isfinite(mean_lat) and np.isfinite(mean_lon)):
            n_rejected += 1
            continue

        if lat_min <= mean_lat <= lat_max and lon_min <= mean_lon <= lon_max:
            accepted.append(ce)
        else:
            n_rejected += 1

    return accepted, n_rejected


# ─────────────────────────────────────────────────────────────────────────────
# §3  Ray geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_ground_ray(
    gnss_pt_km: np.ndarray,
    rx_pt_km:   np.ndarray,
    n_pts:      int   = 80,
    alt_min_km: float = 80.0,
    alt_max_km: float = 1000.0,
) -> np.ndarray:
    """
    Sample the line from GNSS satellite (km ECEF) to ground receiver (km ECEF),
    convert to geodetic, and keep only points inside the ionospheric band.

    Parameters
    ----------
    gnss_pt_km : (3,)  GNSS satellite ECEF position in km.
    rx_pt_km   : (3,)  Ground receiver ECEF position in km.

    Returns
    -------
    traj : (n_iono, 3)  — [lat_deg, lon_deg, alt_km], at least one point.
    """
    t_vals = np.linspace(0.0, 1.0, n_pts)
    pts    = gnss_pt_km[:, None] + t_vals * (rx_pt_km[:, None] - gnss_pt_km[:, None])
    lons, lats, alts_m = _TRANSFORMER.transform(
        pts[0] * 1e3, pts[1] * 1e3, pts[2] * 1e3
    )
    alts_km = alts_m / 1e3
    mask    = (alts_km >= alt_min_km) & (alts_km <= alt_max_km)
    if not np.any(mask):
        best = int(np.argmin(np.abs(alts_km - (alt_min_km + alt_max_km) / 2)))
        return np.array([[lats[best], lons[best],
                          float(np.clip(alts_km[best], alt_min_km, alt_max_km))]])
    return np.column_stack([lats[mask], lons[mask], alts_km[mask]])


def _ipp_latlon(gnss_km: np.ndarray, rx_km: np.ndarray,
                h_ipp: float = 350.0) -> tuple[float, float]:
    """
    Compute the ionospheric pierce point (IPP) at altitude h_ipp (km) for one
    GNSS→receiver pair.  Searches along the parameterised line for the point
    whose geodetic altitude is closest to h_ipp.

    Returns (lat_deg, lon_deg) of the IPP.
    """
    t_vals  = np.linspace(0.0, 1.0, 200)
    pts     = gnss_km[:, None] + t_vals * (rx_km[:, None] - gnss_km[:, None])
    lons, lats, alts_m = _TRANSFORMER.transform(
        pts[0] * 1e3, pts[1] * 1e3, pts[2] * 1e3
    )
    alts_km = alts_m / 1e3
    idx     = int(np.argmin(np.abs(alts_km - h_ipp)))
    return float(lats[idx]), float(lons[idx])


def _idw_weights(tp_lats: np.ndarray, tp_lons: np.ndarray,
                 grid_lats: np.ndarray, grid_lons: np.ndarray,
                 k: int = 4, power: float = 2.0,
                 min_dist_km: float = 1.0) -> np.ndarray:
    """IDW weights from each ray tangent point to the k nearest grid points."""
    k    = min(k, len(grid_lats))
    pts_g = np.column_stack([grid_lats, grid_lons])
    pts_r = np.column_stack([tp_lats,   tp_lons  ])
    tree  = cKDTree(pts_g * 111.0)
    dists_deg, idxs = tree.query(pts_r * 111.0, k=k)
    dists_km = np.maximum(dists_deg, min_dist_km)
    w_k  = 1.0 / dists_km ** power
    w_k /= w_k.sum(axis=1, keepdims=True)
    n_rays, n_grid = len(tp_lats), len(grid_lats)
    W    = np.zeros((n_rays, n_grid), dtype=float)
    W[np.arange(n_rays)[:, None], idxs] = w_k
    return W


# ─────────────────────────────────────────────────────────────────────────────
# §4  IRI prior and background covariance
# ─────────────────────────────────────────────────────────────────────────────

def _solar_sampling_df(t: pd.Timestamp) -> pd.DataFrame:
    inp = IRI_Sample_Inputs(t.strftime("%Y-%m-%d %H:%M:%S"))
    return pd.DataFrame([{
        "hour": float(t.hour) + t.minute / 60.0,
        "f107": float(inp.apf107["f107"][inp.current_idx_f107]),
        "ap":   float(inp.apf107["iapda"][inp.current_idx_f107]),
        "ig12": float(inp.ig_rz["ig"][inp.current_idx_igrz]),
        "rz12": float(inp.ig_rz["rz"][inp.current_idx_igrz]),
    }])


def _get_iri_prior(t_centre: pd.Timestamp,
                   grid_lats: np.ndarray, grid_lons: np.ndarray,
                   alt_grid: np.ndarray) -> np.ndarray:
    """
    Call IRI2020 at all grid points for t_centre.

    Returns
    -------
    ne_prior : (n_alt, n_grid)  IRI Ne profile in m⁻³, clipped to ≥ 1 m⁻³.
    """
    sdf    = _solar_sampling_df(t_centre)
    geoloc = np.column_stack([grid_lons, grid_lats])
    edps, _ = get_IRI2020_EDP(
        DateTime            = t_centre.strftime("%Y-%m-%d %H:%M:%S"),
        altitude            = alt_grid,
        geolocation         = geoloc,
        sampling_parameters = sdf,
    )
    ne = edps[:, :, 0].astype(float)
    # IRI can return NaN or negative values at the edges of its validity range.
    # Replace non-finite values with a 1 m⁻³ floor before use.
    ne = np.where(np.isfinite(ne), ne, 1.0)
    return np.maximum(ne, 1.0)   # (n_alt, n_grid)


def _build_prior_covariance(
    t_centre:        pd.Timestamp,
    grid_lats:       np.ndarray,
    grid_lons:       np.ndarray,
    alt_grid:        np.ndarray,
    n_iri_samples:   int   = 12,
    corr_length_km:  float = CORR_LENGTH_KM,
    v_corr_km:       float = V_CORR_KM,
    sigma_ne_floor:  float = SIGMA_NE_FLOOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate background error statistics from diurnal IRI variability.

    Returns
    -------
    sigma_v : (n_alt,)         Per-altitude fractional Ne uncertainty
                               (fraction of IRI prior mean), ≥ sigma_ne_floor.
    C_v     : (n_alt, n_alt)   Vertical altitude correlation (exponential kernel).
    C_s     : (n_grid, n_grid) Horizontal spatial correlation (exponential kernel).
    """
    n_alt  = len(alt_grid)
    n_geo  = len(grid_lats)
    day0   = pd.Timestamp(t_centre.date())
    geoloc = np.column_stack([grid_lons, grid_lats])

    ne_samples = []
    dt_step    = pd.Timedelta(hours=24 / n_iri_samples)
    for s in range(n_iri_samples):
        t_s = day0 + s * dt_step
        try:
            sdf = _solar_sampling_df(t_s)
            edps, _ = get_IRI2020_EDP(
                DateTime            = t_s.strftime("%Y-%m-%d %H:%M:%S"),
                altitude            = alt_grid,
                geolocation         = geoloc,
                sampling_parameters = sdf,
            )
            ne = edps[:, :, 0].astype(float)
            ne = np.where(np.isfinite(ne), ne, 1.0)
            ne_samples.append(np.maximum(ne, 1.0))
        except Exception as exc:
            print(f"    [IRI cov] sample {s} failed: {exc}")

    if len(ne_samples) >= 4:
        ne_stack   = np.stack(ne_samples, axis=2)          # (n_alt, n_geo, n_samples)
        ne_mean_gp = ne_stack.mean(axis=2)                 # (n_alt, n_geo)
        ne_std_gp  = ne_stack.std(axis=2)                  # (n_alt, n_geo)
        # Fractional variability averaged over grid columns, floored.
        # Guard against zero or non-finite mean to avoid NaN ratios.
        frac     = np.where(ne_mean_gp > 1.0, ne_std_gp / ne_mean_gp, sigma_ne_floor)
        frac     = np.where(np.isfinite(frac), frac, sigma_ne_floor)
        sigma_v  = np.maximum(frac.mean(axis=1), sigma_ne_floor)   # (n_alt,)
        sigma_v  = np.where(np.isfinite(sigma_v), sigma_v, sigma_ne_floor)
    else:
        print("    [IRI cov] Too few samples — using default sigma_v=0.50.")
        sigma_v = np.full(n_alt, 0.50)

    # Vertical correlation: exponential kernel in altitude
    dalt = np.abs(alt_grid[:, None] - alt_grid[None, :])   # (n_alt, n_alt)
    C_v  = np.exp(-dalt / v_corr_km)
    C_v += 1e-6 * np.eye(n_alt)

    # Horizontal correlation: exponential kernel in great-circle distance
    dist = _haversine_km(
        grid_lats[:, None], grid_lons[:, None],
        grid_lats[None, :], grid_lons[None, :],
    )
    C_s  = np.exp(-dist / corr_length_km)
    C_s += 1e-6 * np.eye(n_geo)
    C_s /= C_s.diagonal()[:, None]   # normalise so diagonal = 1

    return sigma_v, C_v, C_s


# ─────────────────────────────────────────────────────────────────────────────
# §4b  Linear TEC observation operator H
# ─────────────────────────────────────────────────────────────────────────────

def _build_H_matrix(
    rays:       list[np.ndarray],
    alt_grid:   np.ndarray,
    grid_lats:  np.ndarray,
    grid_lons:  np.ndarray,
    k_idw:      int   = 4,
) -> np.ndarray:
    """
    Build the linear observation operator H.

    State indexing: ``x[i * n_grid + g] = Ne(alt_grid[i], grid_col[g])``  (m⁻³).
    H integrates Ne along each ray using the trapezoid rule and bi-linear
    interpolation in altitude (from alt_grid levels) and IDW in the horizontal
    (from the nearest k_idw grid columns).

    Parameters
    ----------
    rays      : list of (n_pts, 3) arrays  — [lat_deg, lon_deg, alt_km] per ray.
    alt_grid  : (n_alt,)  altitude levels in km.
    grid_lats, grid_lons : (n_grid,)  grid column positions.
    k_idw     : number of nearest grid columns for IDW blending.

    Returns
    -------
    H : (n_rays, n_alt * n_grid)  — units TECU per (m⁻³).
    """
    n_rays  = len(rays)
    n_alt   = len(alt_grid)
    n_grid  = len(grid_lats)
    H       = np.zeros((n_rays, n_alt * n_grid), dtype=float)
    _CONV   = 1.0e3 / 1.0e16   # km·m⁻³  →  TECU   (×1e3 m/km, ÷1e16 TECU)
    _R_E    = 6371.0            # km
    k_idw   = min(k_idw, n_grid)
    g_idx   = np.arange(n_grid, dtype=int)

    # KD-tree on grid columns (flat-degree approximation, scaled to ~km)
    tree = cKDTree(np.column_stack([grid_lats, grid_lons]) * 111.0)

    for i_ray, ray in enumerate(rays):
        n_pts   = len(ray)
        lat_r   = ray[:, 0]
        lon_r   = ray[:, 1]
        alt_r   = ray[:, 2]

        # ── Arc-length trapezoid weights ──────────────────────────────────────
        lat_rad = np.radians(lat_r)
        lon_rad = np.radians(lon_r)
        r_vec   = _R_E + alt_r
        xyz     = np.stack([
            r_vec * np.cos(lat_rad) * np.cos(lon_rad),
            r_vec * np.cos(lat_rad) * np.sin(lon_rad),
            r_vec * np.sin(lat_rad),
        ], axis=1)                                                # (n_pts, 3)
        segs    = np.linalg.norm(np.diff(xyz, axis=0), axis=1)   # (n_pts-1,)
        if n_pts == 1:
            ds = np.array([1.0])
        else:
            ds      = np.empty(n_pts)
            ds[0]   = 0.5 * segs[0]
            ds[-1]  = 0.5 * segs[-1]
            if n_pts > 2:
                ds[1:-1] = 0.5 * (segs[:-1] + segs[1:])

        # ── IDW weights at every sample point → (n_pts, n_grid) ──────────────
        query    = np.column_stack([lat_r, lon_r]) * 111.0
        dists, nb_idx = tree.query(query, k=k_idw)               # (n_pts, k_idw)
        dists    = np.maximum(dists, 1.0)
        w_raw    = 1.0 / dists ** 2
        w_raw   /= w_raw.sum(axis=1, keepdims=True)
        W_pts    = np.zeros((n_pts, n_grid), dtype=float)
        W_pts[np.arange(n_pts)[:, None], nb_idx] = w_raw         # (n_pts, n_grid)

        # ── Altitude linear interpolation weights ─────────────────────────────
        alt_c    = np.clip(alt_r, alt_grid[0], alt_grid[-1])
        k_lo     = np.searchsorted(alt_grid, alt_c, side='right') - 1
        k_lo     = np.clip(k_lo, 0, n_alt - 2)
        k_hi     = k_lo + 1
        dalt     = alt_grid[k_hi] - alt_grid[k_lo]
        alpha_hi = np.where(dalt > 0, (alt_c - alt_grid[k_lo]) / dalt, 0.5)
        alpha_lo = 1.0 - alpha_hi

        # ── Scatter contributions into H ──────────────────────────────────────
        contrib    = (ds[:, None] * _CONV) * W_pts   # (n_pts, n_grid)
        contrib_lo = alpha_lo[:, None] * contrib      # (n_pts, n_grid)
        contrib_hi = alpha_hi[:, None] * contrib

        idx_lo = k_lo[:, None] * n_grid + g_idx[None, :]   # (n_pts, n_grid) int
        idx_hi = k_hi[:, None] * n_grid + g_idx[None, :]

        np.add.at(H[i_ray], idx_lo, contrib_lo)
        np.add.at(H[i_ray], idx_hi, contrib_hi)

    return H   # (n_rays, n_alt * n_grid)


# ─────────────────────────────────────────────────────────────────────────────
# §5  Standard Kalman Filter update
# ─────────────────────────────────────────────────────────────────────────────

def run_kf_window(
    clean_window:     list,
    t_centre:         pd.Timestamp,
    grid_lats:        np.ndarray,
    grid_lons:        np.ndarray,
    alt_grid:         np.ndarray,
    ne_prior:         np.ndarray,   # (n_alt, n_grid)  IRI prior Ne in m⁻³
    sigma_v:          np.ndarray,   # (n_alt,)          fractional uncertainty per alt
    C_v:              np.ndarray,   # (n_alt, n_alt)    vertical correlation
    C_s:              np.ndarray,   # (n_grid, n_grid)  horizontal correlation
    sigma_obs:        float = SIGMA_OBS_TECU,
    max_rays_per_arc: int   = MAX_RAYS_PER_ARC,
) -> dict | None:
    """
    Standard (deterministic) Kalman Filter update for one 90-minute window.

    State
    -----
    x  : (n_state,)  where n_state = n_alt * n_grid.
         Indexed as  x[i * n_grid + g] = Ne(alt_grid[i], grid_col[g])  (m⁻³).
    x_f = ne_prior.ravel()  — IRI prior as the background state.

    Background covariance
    ---------------------
    P_f = diag(σ) @ kron(C_v, C_s) @ diag(σ)
    where  σ[i*n_grid+g] = sigma_v[i] * ne_prior[i, g]  (absolute Ne uncertainty).

    Observation operator
    --------------------
    H  : (n_obs, n_state)  — linear ray-integration matrix built by _build_H_matrix.
    y  = H x + ε,  ε ~ N(0, R),  R = σ_obs² I.

    Kalman update
    -------------
    S   = H P_f Hᵀ + R
    K   = P_f Hᵀ S⁻¹
    x_a = x_f + K (y − H x_f)
    P_a = (I − K H) P_f

    Returns a result dict, or None if there are no usable observations.
    """
    if not clean_window:
        return None

    n_geo   = len(grid_lats)
    n_alt   = len(alt_grid)
    n_state = n_alt * n_geo

    # ── 1. Prior state vector ─────────────────────────────────────────────────
    x_f = ne_prior.ravel().copy()   # (n_state,)  row-major: x[i*n_geo+g]=ne[i,g]

    # ── 2. Background covariance P_f ─────────────────────────────────────────
    # Absolute Ne uncertainty per state element: σ[i*n_geo+g] = sigma_v[i]*ne[i,g]
    sigma_abs = (sigma_v[:, None] * ne_prior).ravel()   # (n_state,)
    # Kronecker-structured correlation: kron(C_v, C_s)
    # kron[i*n_geo+g, j*n_geo+h] = C_v[i,j] * C_s[g,h]
    K_corr = np.kron(C_v, C_s)                           # (n_state, n_state)
    # Scaled background covariance
    P_f    = sigma_abs[:, None] * K_corr * sigma_abs[None, :]  # (n_state, n_state)

    # ── 3. Build ray trajectories ─────────────────────────────────────────────
    all_rays:    list[np.ndarray] = []
    all_tp_lats: list[float]      = []
    all_tp_lons: list[float]      = []
    all_tec_obs: list[float]      = []
    arc_sizes:   list[int]        = []   # one entry per element of clean_window

    for ce in clean_window:
        gnss     = ce["GNSS"]    # (3, n_s) km ECEF
        leo      = ce["LEO"]     # (3, n_s) km ECEF — ground receiver
        tec_arc  = ce["tec"]     # (n_s,)   TECU
        n_s      = gnss.shape[1]
        ipp_lats = ce.get("ipp_lat", np.full(n_s, np.nan))
        ipp_lons = ce.get("ipp_lon", np.full(n_s, np.nan))

        stride = max(1, int(np.ceil(n_s / max_rays_per_arc)))
        idx    = np.arange(0, n_s, stride)

        arc_ray_list: list[np.ndarray] = []
        arc_lats_i:   list[float]      = []
        arc_lons_i:   list[float]      = []
        arc_tec_i:    list[float]      = []

        for j in idx:
            traj = _build_ground_ray(gnss[:, j], leo[:, j])
            ilat = float(ipp_lats[j]) if j < len(ipp_lats) else np.nan
            ilon = float(ipp_lons[j]) if j < len(ipp_lons) else np.nan
            if not (np.isfinite(ilat) and np.isfinite(ilon)):
                ilat, ilon = _ipp_latlon(gnss[:, j], leo[:, j])
            arc_ray_list.append(traj)
            arc_lats_i.append(ilat)
            arc_lons_i.append(ilon)
            arc_tec_i.append(float(tec_arc[j]))

        ns = len(arc_ray_list)
        arc_sizes.append(ns)
        if ns == 0:
            continue

        all_rays.extend(arc_ray_list)
        all_tp_lats.extend(arc_lats_i)
        all_tp_lons.extend(arc_lons_i)
        all_tec_obs.extend(arc_tec_i)

    n_total = len(all_rays)
    if n_total == 0:
        return None

    all_tp_lats_arr = np.array(all_tp_lats)
    all_tp_lons_arr = np.array(all_tp_lons)
    y_obs_all       = np.array(all_tec_obs)

    # Filter to finite observations
    valid_obs = np.isfinite(y_obs_all)
    if valid_obs.sum() == 0:
        return None
    rays_v    = [all_rays[i] for i in range(n_total) if valid_obs[i]]
    tp_lats_v = all_tp_lats_arr[valid_obs]
    tp_lons_v = all_tp_lons_arr[valid_obs]
    y_obs_v   = y_obs_all[valid_obs]
    n_obs     = len(y_obs_v)

    # ── 4. Build H matrices ───────────────────────────────────────────────────
    print(f"    Building H  ({n_obs} obs × {n_state} state) …", flush=True)
    H_v   = _build_H_matrix(rays_v,   alt_grid, grid_lats, grid_lons)   # (n_obs,   n_state)
    H_all = _build_H_matrix(all_rays, alt_grid, grid_lats, grid_lons)   # (n_total, n_state)

    # ── 5. Prior simulated observations ──────────────────────────────────────
    y_f_v       = H_v   @ x_f    # (n_obs,)
    y_f_all     = H_all @ x_f    # (n_total,)
    prior_innov = y_obs_v - y_f_v

    print(f"    [Prior] innovation  mean={prior_innov.mean():.2f}  "
          f"std={prior_innov.std():.2f}  n_obs={n_obs}", flush=True)

    # ── 6. Information-form Kalman update ────────────────────────────────────
    # Standard form  S = H P H^T + R  requires an (n_obs × n_obs) matrix which
    # is infeasible when n_obs is O(10 000).  The information form stays in
    # (n_state × n_state) space regardless of how many observations there are.
    #
    #   Prior information   : I_f  = P_f^{-1}          (n_state × n_state)
    #   Obs information     : I_y  = H^T R^{-1} H       (n_state × n_state)
    #   Posterior info      : I_a  = I_f + I_y
    #   Posterior state     : x_a  = I_a^{-1} (I_f x_f + H^T R^{-1} y)
    #   Posterior covariance: P_a  = I_a^{-1}
    #
    # With R = r² I  →  H^T R^{-1} H = H^T H / r²  and
    #                    H^T R^{-1} y = H^T y / r².
    r2     = sigma_obs ** 2

    # Regularise P_f before inversion (guards against near-singular rows where
    # ne_prior was clamped to the 1 m⁻³ floor and sigma is essentially zero).
    P_f_reg = P_f + 1e-6 * np.diag(np.maximum(np.diag(P_f), 1.0))
    try:
        L_f = la.cholesky(P_f_reg, lower=True)
        I_f = la.cho_solve((L_f, True), np.eye(n_state))   # P_f^{-1}
    except la.LinAlgError:
        I_f = np.linalg.solve(P_f_reg, np.eye(n_state))

    I_y    = (H_v.T @ H_v) / r2                            # (n_state, n_state)
    I_a    = I_f + I_y
    I_a    = 0.5 * (I_a + I_a.T)                           # enforce symmetry
    # Regularise: add a small fraction of the diagonal to ensure positive definiteness
    # even when some state elements are unobserved (H rows all zero for that column).
    I_a   += 1e-10 * np.diag(np.diag(I_a))

    z      = H_v.T @ (y_obs_v / r2)                        # (n_state,)
    b_f    = I_f @ x_f                                      # (n_state,)

    # Solve via Cholesky — reuse the factor for both x_a and diag(P_a)
    L_a = None
    try:
        L_a  = la.cholesky(I_a, lower=True)
        x_a  = la.cho_solve((L_a, True), b_f + z)          # (n_state,)
    except la.LinAlgError:
        L_a  = None
        x_a  = la.solve(I_a, b_f + z)
    x_a = np.maximum(x_a, 1.0)                             # clip to physical minimum

    # Diagonal of P_a = I_a^{-1}; reuse L_a if available
    if L_a is not None:
        try:
            L_a_inv  = la.solve_triangular(L_a, np.eye(n_state), lower=True)
            post_var = np.sum(L_a_inv ** 2, axis=0)        # diag(L_a^{-T} L_a^{-1})
        except la.LinAlgError:
            post_var = sigma_abs ** 2
    else:
        post_var = sigma_abs ** 2                           # fallback: prior variance

    # ── 7. Posterior simulated observations ──────────────────────────────────
    y_a_v          = H_v   @ x_a
    y_a_all        = H_all @ x_a
    post_innov     = y_obs_v - y_a_v
    prior_rmse     = float(np.sqrt(np.nanmean((y_obs_all - y_f_all) ** 2)))
    post_rmse      = float(np.sqrt(np.nanmean((y_obs_all - y_a_all) ** 2)))

    print(f"    [Post ] innovation  mean={post_innov.mean():.2f}  "
          f"std={post_innov.std():.2f}", flush=True)
    print(f"    RMSE:  prior={prior_rmse:.3f} TECU  →  post={post_rmse:.3f} TECU")

    # ── 8. Reshape state and uncertainty to (n_alt, n_grid) ──────────────────
    prior_edp   = ne_prior                                       # (n_alt, n_grid)
    post_edp    = x_a.reshape(n_alt, n_geo)                     # (n_alt, n_grid)
    prior_sigma = sigma_abs.reshape(n_alt, n_geo)               # ±1σ prior
    post_sigma  = np.sqrt(np.maximum(post_var, 0.0)).reshape(n_alt, n_geo)  # ±1σ posterior

    return {
        "t_centre":         t_centre,
        "clean_window":     clean_window,
        "n_obs":            n_obs,
        # Full set of rays (NaN-inclusive)
        "all_rays":         all_rays,
        "all_tp_lats":      all_tp_lats_arr,
        "all_tp_lons":      all_tp_lons_arr,
        "y_obs_all":        y_obs_all,          # (n_total,)
        "Y_all_prior_mean": y_f_all,            # (n_total,)
        "Y_all_post_mean":  y_a_all,            # (n_total,)
        # Finite-observation subset
        "y_obs_v":          y_obs_v,
        "rep_tp_lats":      tp_lats_v,
        "rep_tp_lons":      tp_lons_v,
        "Y_prior_mean":     y_f_v,
        "Y_post_mean":      y_a_v,
        # Residual statistics
        "prior_innov":      prior_innov,
        "post_innov":       post_innov,
        "prior_rmse":       prior_rmse,
        "post_rmse":        post_rmse,
        # EDP grids (n_alt, n_grid)
        "prior_edp":        prior_edp,
        "post_edp":         post_edp,
        # 1-σ uncertainty grids (n_alt, n_grid)
        "prior_sigma":      prior_sigma,
        "post_sigma":       post_sigma,
        # arc_sizes[i] is the number of subsampled epochs for clean_window[i];
        # 0 for skipped arcs.  Slice flat arrays as [offset : offset+arc_sizes[i]].
        "arc_sizes":        arc_sizes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# §5b  SRIF batch update — RAM-efficient alternative to run_kf_window
# ─────────────────────────────────────────────────────────────────────────────

def run_srif_window(
    clean_window:     list,
    t_centre:         pd.Timestamp,
    grid_lats:        np.ndarray,
    grid_lons:        np.ndarray,
    alt_grid:         np.ndarray,
    ne_prior:         np.ndarray,   # (n_alt, n_grid)  IRI prior Ne in m⁻³
    sigma_v:          np.ndarray,   # (n_alt,)          fractional uncertainty per alt
    C_v:              np.ndarray,   # (n_alt, n_alt)    vertical correlation
    C_s:              np.ndarray,   # (n_grid, n_grid)  horizontal correlation
    sigma_obs:        float = SIGMA_OBS_TECU,
    max_rays_per_arc: int   = MAX_RAYS_PER_ARC,
    srif_chunk_size:  int   = 512,
) -> dict | None:
    """
    SRIF (Square Root Information Filter) batch update for one 90-minute window.

    Identical external interface to :func:`run_kf_window` and returns the same
    result dict, but avoids forming the full  H  matrix  (n_obs × n_state)  in
    memory.  Instead observations are fed arc-by-arc into :class:`SRIFBatchUpdate`
    which maintains only the  n_state × n_state  information square root  R̄.

    Memory profile
    --------------
    • Prior covariance P_f is never densified: the SRIF is initialised via
      :meth:`SRIFBatchUpdate.from_kron_prior`, which factorises only the small
      C_v (n_alt × n_alt) and C_s (n_grid × n_grid) blocks in closed form.
    • SRIF state  (R̄, z) :  n_state² + n_state  ≈ 40 MB  (n_state = 2254)
    • Per-arc H chunk    :  MAX_RAYS_PER_ARC × n_state × 8 ≈ 0.9 MB
    • Full H never assembled — peak RAM dominated by the R̄ buffer above.

    For comparison, the standard form assembles H at once:
    14 524 obs × 2 254 state × 8 bytes ≈ 261 MB, before any further algebra.

    SRIF update mathematics
    -----------------------
    Λ = P_f^{-1} + H^T W H   (Batch information matrix)
    N = P_f^{-1} x_f + H^T W y

    SRIF factors:  R̄_a s.t.  R̄_a^T R̄_a = Λ
    Posterior:      x_a = R̄_a^{-1} (R̄_a^{-T} N)  via back-substitution
    Uncertainty:    diag(P_a) = row-wise ||R̄_a^{-1}||²

    At each arc, an augmented Householder step adds the arc's H rows without
    ever accumulating the full H matrix:

        A_aug = [R̄  | z ]   ←  n rows   (current SRIF state)
                [H/σ | y/σ]  ←  m rows   (arc's scaled observations)

        [R̄_new | z_new] = upper block of QR(A_aug)

    See :mod:`Ionosphere_Tomography_Inverter.srif_batch_update` for details.
    """
    if not clean_window:
        return None

    n_geo   = len(grid_lats)
    n_alt   = len(alt_grid)
    n_state = n_alt * n_geo

    # ── 1. Prior state vector ─────────────────────────────────────────────────
    x_f = ne_prior.ravel().copy()        # (n_state,)

    # ── 2. Background covariance (Kronecker-structured, never densified) ────
    sigma_abs = (sigma_v[:, None] * ne_prior).ravel()        # (n_state,)

    # ── 3. Initialise SRIF from the prior ────────────────────────────────────
    # Uses the closed-form Kronecker fast path: factorises only the small
    # C_v (n_alt × n_alt) and C_s (n_grid × n_grid) blocks, so the O(n_state³)
    # dense Cholesky/inversion that previously caused RAM/swap thrashing for
    # large n_state is never performed.
    print(f"    [SRIF] Initialising information square root "
          f"(n_state={n_state}) …", flush=True)
    srif = SRIFBatchUpdate.from_kron_prior(
        x_prior    = x_f,
        sigma_abs  = sigma_abs,
        C_v        = C_v,
        C_s        = C_s,
        obs_sigma  = sigma_obs,
        chunk_size = srif_chunk_size,
    )

    # ── 4. Build rays arc-by-arc and stream into SRIF ─────────────────────────
    # We never assemble the full H matrix; each arc is processed and discarded.
    all_rays:    list[np.ndarray] = []
    all_tp_lats: list[float]      = []
    all_tp_lons: list[float]      = []
    all_tec_obs: list[float]      = []
    arc_sizes:   list[int]        = []

    for ce in clean_window:
        gnss     = ce["GNSS"]
        leo      = ce["LEO"]
        tec_arc  = ce["tec"]
        n_s      = gnss.shape[1]
        ipp_lats = ce.get("ipp_lat", np.full(n_s, np.nan))
        ipp_lons = ce.get("ipp_lon", np.full(n_s, np.nan))

        stride = max(1, int(np.ceil(n_s / max_rays_per_arc)))
        idx    = np.arange(0, n_s, stride)

        arc_rays: list[np.ndarray] = []
        arc_lats: list[float]      = []
        arc_lons: list[float]      = []
        arc_tec:  list[float]      = []

        for j in idx:
            traj = _build_ground_ray(gnss[:, j], leo[:, j])
            ilat = float(ipp_lats[j]) if j < len(ipp_lats) else np.nan
            ilon = float(ipp_lons[j]) if j < len(ipp_lons) else np.nan
            if not (np.isfinite(ilat) and np.isfinite(ilon)):
                ilat, ilon = _ipp_latlon(gnss[:, j], leo[:, j])
            arc_rays.append(traj)
            arc_lats.append(ilat)
            arc_lons.append(ilon)
            arc_tec.append(float(tec_arc[j]))

        ns = len(arc_rays)
        arc_sizes.append(ns)
        if ns == 0:
            continue

        all_rays.extend(arc_rays)
        all_tp_lats.extend(arc_lats)
        all_tp_lons.extend(arc_lons)
        all_tec_obs.extend(arc_tec)

        # Build H for this arc only  (ns × n_state) — then discard
        y_arc = np.array(arc_tec, dtype=float)
        fin   = np.isfinite(y_arc)
        if fin.sum() == 0:
            continue

        H_arc = _build_H_matrix(arc_rays, alt_grid, grid_lats, grid_lons)
        # Feed finite rows into SRIF; H_arc is discarded after this call
        srif.update(H_arc[fin], y_arc[fin])

    n_total = len(all_rays)
    if n_total == 0 or srif.n_obs == 0:
        return None

    all_tp_lats_arr = np.array(all_tp_lats)
    all_tp_lons_arr = np.array(all_tp_lons)
    y_obs_all       = np.array(all_tec_obs)
    valid_obs       = np.isfinite(y_obs_all)
    n_obs           = int(valid_obs.sum())

    diag_info = srif.innovation_stats()
    print(f"    [SRIF] arcs ingested = {srif.n_arcs}/{len(clean_window)}  |  "
          f"obs rows = {srif.n_obs}  |  "
          f"R̄ cond ≈ {diag_info['condition_number_estimate']:.2e}", flush=True)

    # ── 5. Solve for posterior ────────────────────────────────────────────────
    print(f"    [SRIF] Solving …", flush=True)
    x_a_raw, post_var = srif.solve()
    x_a = np.maximum(x_a_raw, 1.0)   # clip to physical Ne floor

    # ── 6. Prior and posterior simulated observations ─────────────────────────
    # Build the full H only for diagnostics / plotting — but do it in arc chunks
    # so we still avoid the full n_obs × n_state allocation at once.
    y_f_all = np.empty(n_total)
    y_a_all = np.empty(n_total)
    ray_offset = 0
    for ce_i, ce in enumerate(clean_window):
        ns = arc_sizes[ce_i]
        if ns == 0:
            continue
        rays_slice = all_rays[ray_offset: ray_offset + ns]
        H_chunk    = _build_H_matrix(rays_slice, alt_grid, grid_lats, grid_lons)
        sl         = slice(ray_offset, ray_offset + ns)
        y_f_all[sl] = H_chunk @ x_f
        y_a_all[sl] = H_chunk @ x_a
        ray_offset  += ns

    # Residual statistics
    fin_all         = valid_obs
    prior_innov     = (y_obs_all - y_f_all)[fin_all]
    post_innov      = (y_obs_all - y_a_all)[fin_all]
    prior_rmse      = float(np.sqrt(np.nanmean((y_obs_all - y_f_all)[fin_all] ** 2)))
    post_rmse       = float(np.sqrt(np.nanmean((y_obs_all - y_a_all)[fin_all] ** 2)))

    print(f"    [Prior] innovation  mean={prior_innov.mean():.2f}  "
          f"std={prior_innov.std():.2f}  n_obs={n_obs}", flush=True)
    print(f"    [Post ] innovation  mean={post_innov.mean():.2f}  "
          f"std={post_innov.std():.2f}", flush=True)
    print(f"    RMSE:  prior={prior_rmse:.3f} TECU  →  post={post_rmse:.3f} TECU")

    # Subsets for finite observations (parallel to rays_v in run_kf_window)
    rays_v    = [all_rays[i] for i in range(n_total) if valid_obs[i]]
    tp_lats_v = all_tp_lats_arr[valid_obs]
    tp_lons_v = all_tp_lons_arr[valid_obs]
    y_obs_v   = y_obs_all[valid_obs]
    y_f_v     = y_f_all[valid_obs]
    y_a_v     = y_a_all[valid_obs]

    # ── 7. Reshape to (n_alt, n_grid) grids ──────────────────────────────────
    prior_edp   = ne_prior
    post_edp    = x_a.reshape(n_alt, n_geo)
    prior_sigma = sigma_abs.reshape(n_alt, n_geo)
    post_sigma  = np.sqrt(np.maximum(post_var, 0.0)).reshape(n_alt, n_geo)

    return {
        "t_centre":         t_centre,
        "clean_window":     clean_window,
        "n_obs":            n_obs,
        "all_rays":         all_rays,
        "all_tp_lats":      all_tp_lats_arr,
        "all_tp_lons":      all_tp_lons_arr,
        "y_obs_all":        y_obs_all,
        "Y_all_prior_mean": y_f_all,
        "Y_all_post_mean":  y_a_all,
        "y_obs_v":          y_obs_v,
        "rep_tp_lats":      tp_lats_v,
        "rep_tp_lons":      tp_lons_v,
        "Y_prior_mean":     y_f_v,
        "Y_post_mean":      y_a_v,
        "prior_innov":      prior_innov,
        "post_innov":       post_innov,
        "prior_rmse":       prior_rmse,
        "post_rmse":        post_rmse,
        "prior_edp":        prior_edp,
        "post_edp":         post_edp,
        "prior_sigma":      prior_sigma,
        "post_sigma":       post_sigma,
        "arc_sizes":        arc_sizes,
        # SRIF-specific diagnostics
        "srif_n_arcs":      srif.n_arcs,
        "srif_n_obs":       srif.n_obs,
        "srif_weighted_rss": srif.weighted_rss,
        "srif_R_cond":      diag_info["condition_number_estimate"],
    }


def run_info_window(
    clean_window:     list,
    t_centre:         pd.Timestamp,
    grid_lats:        np.ndarray,
    grid_lons:        np.ndarray,
    alt_grid:         np.ndarray,
    ne_prior:         np.ndarray,   # (n_alt, n_grid)  IRI prior Ne in m⁻³
    sigma_v:          np.ndarray,   # (n_alt,)          fractional uncertainty per alt
    C_v:              np.ndarray,   # (n_alt, n_alt)    vertical correlation
    C_s:              np.ndarray,   # (n_grid, n_grid)  horizontal correlation
    sigma_obs:        float = SIGMA_OBS_TECU,
    max_rays_per_arc: int   = MAX_RAYS_PER_ARC,
) -> dict | None:
    """
    Plain information-form (normal-equations) batch update for one 90-minute
    window.

    Identical external interface and return dict to :func:`run_srif_window`,
    but replaces the Householder-QR SRIF accumulation with the simpler
    recursion:

        Λ = P_f^{-1} + H^T W H
        N = P_f^{-1} x_f + H^T W y

    accumulated per-arc as plain BLAS matrix products (``H^T H``, ``H^T y``)
    rather than a QR re-triangularisation of the full n_state × n_state
    factor on every arc. Mathematically equivalent to the SRIF/covariance-form
    updates for isotropic R, but noticeably faster per arc since it avoids
    Householder reflections over the full state dimension — at the cost of
    squaring H's condition number relative to the square-root (SRIF) form,
    the standard information-filter tradeoff. Use this when SRIF's per-arc
    QR cost is the bottleneck; fall back to :func:`run_srif_window` if
    conditioning becomes a concern (see ``Lam_cond`` diagnostic below).

    See :mod:`Ionosphere_Tomography_Inverter.info_batch_update` for details.
    """
    if not clean_window:
        return None

    n_geo   = len(grid_lats)
    n_alt   = len(alt_grid)
    n_state = n_alt * n_geo

    # ── 1. Prior state vector ─────────────────────────────────────────────────
    x_f = ne_prior.ravel().copy()        # (n_state,)

    # ── 2. Background covariance (Kronecker-structured, never densified) ────
    sigma_abs = (sigma_v[:, None] * ne_prior).ravel()        # (n_state,)

    # ── 3. Initialise the information accumulator from the prior ────────────
    print(f"    [Info] Initialising information matrix "
          f"(n_state={n_state}) …", flush=True)
    info = InfoBatchUpdate.from_kron_prior(
        x_prior   = x_f,
        sigma_abs = sigma_abs,
        C_v       = C_v,
        C_s       = C_s,
        obs_sigma = sigma_obs,
    )

    # ── 4. Build rays arc-by-arc and stream into the accumulator ─────────────
    # We never assemble the full H matrix; each arc is processed and discarded.
    all_rays:    list[np.ndarray] = []
    all_tp_lats: list[float]      = []
    all_tp_lons: list[float]      = []
    all_tec_obs: list[float]      = []
    arc_sizes:   list[int]        = []

    for ce in clean_window:
        gnss     = ce["GNSS"]
        leo      = ce["LEO"]
        tec_arc  = ce["tec"]
        n_s      = gnss.shape[1]
        ipp_lats = ce.get("ipp_lat", np.full(n_s, np.nan))
        ipp_lons = ce.get("ipp_lon", np.full(n_s, np.nan))

        stride = max(1, int(np.ceil(n_s / max_rays_per_arc)))
        idx    = np.arange(0, n_s, stride)

        arc_rays: list[np.ndarray] = []
        arc_lats: list[float]      = []
        arc_lons: list[float]      = []
        arc_tec:  list[float]      = []

        for j in idx:
            traj = _build_ground_ray(gnss[:, j], leo[:, j])
            ilat = float(ipp_lats[j]) if j < len(ipp_lats) else np.nan
            ilon = float(ipp_lons[j]) if j < len(ipp_lons) else np.nan
            if not (np.isfinite(ilat) and np.isfinite(ilon)):
                ilat, ilon = _ipp_latlon(gnss[:, j], leo[:, j])
            arc_rays.append(traj)
            arc_lats.append(ilat)
            arc_lons.append(ilon)
            arc_tec.append(float(tec_arc[j]))

        ns = len(arc_rays)
        arc_sizes.append(ns)
        if ns == 0:
            continue

        all_rays.extend(arc_rays)
        all_tp_lats.extend(arc_lats)
        all_tp_lons.extend(arc_lons)
        all_tec_obs.extend(arc_tec)

        # Build H for this arc only  (ns × n_state) — then discard
        y_arc = np.array(arc_tec, dtype=float)
        fin   = np.isfinite(y_arc)
        if fin.sum() == 0:
            continue

        H_arc = _build_H_matrix(arc_rays, alt_grid, grid_lats, grid_lons)
        # Feed finite rows into the accumulator; H_arc is discarded after this call
        info.update(H_arc[fin], y_arc[fin])

    n_total = len(all_rays)
    if n_total == 0 or info.n_obs == 0:
        return None

    all_tp_lats_arr = np.array(all_tp_lats)
    all_tp_lons_arr = np.array(all_tp_lons)
    y_obs_all       = np.array(all_tec_obs)
    valid_obs       = np.isfinite(y_obs_all)
    n_obs           = int(valid_obs.sum())

    diag_info = info.innovation_stats()
    print(f"    [Info] arcs ingested = {info.n_arcs}/{len(clean_window)}  |  "
          f"obs rows = {info.n_obs}  |  "
          f"Λ cond ≈ {diag_info['condition_number_estimate']:.2e}", flush=True)

    # ── 5. Solve for posterior ────────────────────────────────────────────────
    print(f"    [Info] Solving …", flush=True)
    x_a_raw, post_var = info.solve()
    x_a = np.maximum(x_a_raw, 1.0)   # clip to physical Ne floor

    # ── 6. Prior and posterior simulated observations ─────────────────────────
    # Build the full H only for diagnostics / plotting — but do it in arc chunks
    # so we still avoid the full n_obs × n_state allocation at once.
    y_f_all = np.empty(n_total)
    y_a_all = np.empty(n_total)
    ray_offset = 0
    for ce_i, ce in enumerate(clean_window):
        ns = arc_sizes[ce_i]
        if ns == 0:
            continue
        rays_slice = all_rays[ray_offset: ray_offset + ns]
        H_chunk    = _build_H_matrix(rays_slice, alt_grid, grid_lats, grid_lons)
        sl         = slice(ray_offset, ray_offset + ns)
        y_f_all[sl] = H_chunk @ x_f
        y_a_all[sl] = H_chunk @ x_a
        ray_offset  += ns

    # Residual statistics
    fin_all         = valid_obs
    prior_innov     = (y_obs_all - y_f_all)[fin_all]
    post_innov      = (y_obs_all - y_a_all)[fin_all]
    prior_rmse      = float(np.sqrt(np.nanmean((y_obs_all - y_f_all)[fin_all] ** 2)))
    post_rmse       = float(np.sqrt(np.nanmean((y_obs_all - y_a_all)[fin_all] ** 2)))

    print(f"    [Prior] innovation  mean={prior_innov.mean():.2f}  "
          f"std={prior_innov.std():.2f}  n_obs={n_obs}", flush=True)
    print(f"    [Post ] innovation  mean={post_innov.mean():.2f}  "
          f"std={post_innov.std():.2f}", flush=True)
    print(f"    RMSE:  prior={prior_rmse:.3f} TECU  →  post={post_rmse:.3f} TECU")

    # Subsets for finite observations (parallel to rays_v in run_kf_window)
    rays_v    = [all_rays[i] for i in range(n_total) if valid_obs[i]]
    tp_lats_v = all_tp_lats_arr[valid_obs]
    tp_lons_v = all_tp_lons_arr[valid_obs]
    y_obs_v   = y_obs_all[valid_obs]
    y_f_v     = y_f_all[valid_obs]
    y_a_v     = y_a_all[valid_obs]

    # ── 7. Reshape to (n_alt, n_grid) grids ──────────────────────────────────
    prior_edp   = ne_prior
    post_edp    = x_a.reshape(n_alt, n_geo)
    prior_sigma = sigma_abs.reshape(n_alt, n_geo)
    post_sigma  = np.sqrt(np.maximum(post_var, 0.0)).reshape(n_alt, n_geo)

    return {
        "t_centre":         t_centre,
        "clean_window":     clean_window,
        "n_obs":            n_obs,
        "all_rays":         all_rays,
        "all_tp_lats":      all_tp_lats_arr,
        "all_tp_lons":      all_tp_lons_arr,
        "y_obs_all":        y_obs_all,
        "Y_all_prior_mean": y_f_all,
        "Y_all_post_mean":  y_a_all,
        "y_obs_v":          y_obs_v,
        "rep_tp_lats":      tp_lats_v,
        "rep_tp_lons":      tp_lons_v,
        "Y_prior_mean":     y_f_v,
        "Y_post_mean":      y_a_v,
        "prior_innov":      prior_innov,
        "post_innov":       post_innov,
        "prior_rmse":       prior_rmse,
        "post_rmse":        post_rmse,
        "prior_edp":        prior_edp,
        "post_edp":         post_edp,
        "prior_sigma":      prior_sigma,
        "post_sigma":       post_sigma,
        # Grid actually used to compute prior_edp/post_edp above -- callers
        # that rebuild an eds_occ/geolocation mesh from this result (e.g.
        # demo_isr_da_comparison.py's igs_only adapters) must key off THIS
        # grid rather than whatever grid_lats/grid_lons happen to be in
        # scope at call time, since a cached result dict can be older than
        # the ROI/grid-construction code that produced the caller's current
        # grid_lats/grid_lons (different vertex count -> geolocation and
        # prior_edp/post_edp go out of sync).
        "grid_lats":        np.asarray(grid_lats, dtype=float),
        "grid_lons":        np.asarray(grid_lons, dtype=float),
        "arc_sizes":        arc_sizes,
        # Info-form-specific diagnostics
        "info_n_arcs":      info.n_arcs,
        "info_n_obs":       info.n_obs,
        "info_weighted_rss": info.weighted_rss,
        "info_Lam_cond":    diag_info["condition_number_estimate"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# §6  Plot (a) — F2 peak density change map (prior → posterior)
# ─────────────────────────────────────────────────────────────────────────────

def plot_f2_change_map(results: list[dict],
                       grid_lats: np.ndarray, grid_lons: np.ndarray,
                       save_dir: str) -> None:
    """
    Regional map showing ΔNmF2 (posterior − prior) at each grid point for
    every assimilation window.  One subplot per window in a grid layout,
    rendered on a Cartopy Orthographic globe centred on (POI_LON, POI_LAT).

    Matches the globe style of demo_group.py _plot_group:
      - set_global() orthographic projection
      - LAND / OCEAN / COASTLINE / BORDERS features
      - tripcolor with Delaunay triangulation over the grid points
      - coolwarm diverging colourmap, per-panel symmetric limits
      - horizontal colourbar with scientific-notation formatter
    """
    import matplotlib.tri as mtri

    active = [r for r in results if r is not None]
    if not active:
        print("  [Map] No assimilation results — skipping F2 change map.")
        return

    n_wins = len(active)
    n_cols = min(4, n_wins)
    n_rows = math.ceil(n_wins / n_cols)

    if _CARTOPY:
        proj   = ccrs.Orthographic(central_longitude=POI_LON,
                                   central_latitude=POI_LAT)
        geo    = ccrs.Geodetic()
        fig    = plt.figure(figsize=(5.5 * n_cols, 4.8 * n_rows))

        # Build Delaunay triangulation from the (fixed) grid points once
        triang = mtri.Triangulation(grid_lons, grid_lats)

        for idx, res in enumerate(active):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection=proj)
            t  = res["t_centre"]
            dNmF2 = res["post_edp"].max(axis=0) - res["prior_edp"].max(axis=0)

            # ── Globe background (demo_group.py _plot_group style) ─────────
            ax.set_global()
            ax.add_feature(cfeature.LAND,  facecolor="whitesmoke", zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor="aliceblue",  zorder=0)
            ax.add_feature(
                cfeature.COASTLINE.with_scale("110m"), lw=0.5, edgecolor="gray")
            ax.add_feature(
                cfeature.BORDERS.with_scale("110m"),   lw=0.3, edgecolor="lightgray")
            ax.gridlines(lw=0.3, alpha=0.4)

            # ── ΔNmF2 filled triangulation ──────────────────────────────────
            max_delta = float(np.nanmax(np.abs(dNmF2)))
            if max_delta > 0:
                tc = ax.tripcolor(
                    triang, dNmF2,
                    transform=geo,
                    cmap="coolwarm", shading="flat",
                    vmin=-max_delta, vmax=max_delta,
                    zorder=1,
                )
                cbar = fig.colorbar(tc, ax=ax, orientation="horizontal",
                                    shrink=0.75, pad=0.04, fraction=0.04)
                cbar.set_label("ΔNmF2 (m⁻³)", fontsize=6)
                cbar.ax.tick_params(labelsize=6)
                cbar.formatter.set_powerlimits((-2, 2))
                cbar.update_ticks()

            # Mark the POI centre
            ax.scatter([POI_LON], [POI_LAT], marker="+", s=120,
                       color="black", lw=1.2, transform=geo, zorder=5)

            ax.set_title(f"{t.strftime('%H:%M')} UTC\n"
                         f"n_obs={res['n_obs']}", fontsize=8)

    else:
        # Fallback: plain axes when Cartopy is not installed
        # Per-panel symmetric colour limits
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 3.5 * n_rows),
                                 squeeze=False)
        for idx, res in enumerate(active):
            r, c   = divmod(idx, n_cols)
            ax     = axes[r][c]
            t      = res["t_centre"]
            dNmF2  = res["post_edp"].max(axis=0) - res["prior_edp"].max(axis=0)
            max_delta = max(float(np.nanmax(np.abs(dNmF2))), 1e9)

            triang_f = mtri.Triangulation(grid_lons, grid_lats)
            tc = ax.tripcolor(triang_f, dNmF2,
                              cmap="coolwarm", shading="flat",
                              vmin=-max_delta, vmax=max_delta)
            ax.axhline(POI_LAT, color="gray", lw=0.6, ls=":")
            ax.axvline(POI_LON, color="gray", lw=0.6, ls=":")
            ax.set_title(f"{t.strftime('%H:%M')} UTC\nn_obs={res['n_obs']}",
                         fontsize=8)
            ax.set_xlabel("Lon (°E)", fontsize=7)
            ax.set_ylabel("Lat (°N)", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3, lw=0.3)
            cbar = fig.colorbar(tc, ax=ax, fraction=0.04, pad=0.03,
                                orientation="horizontal", shrink=0.75)
            cbar.set_label("ΔNmF2 (m⁻³)", fontsize=6)
            cbar.ax.tick_params(labelsize=6)
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.update_ticks()

        for j in range(len(active), n_rows * n_cols):
            r, c = divmod(j, n_cols)
            axes[r][c].set_visible(False)

    fig.suptitle(
        f"NmF2 Change (Posterior − Prior) — {CAMPAIGN_DATE.strftime('%Y-%m-%d')}\n"
        f"POI: ({POI_LAT:.0f}°N, {POI_LON:.0f}°E)  "
        f"Grid: Δ{GRID_DLAT:.1f}°×{GRID_DLON:.1f}°",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    fpath = os.path.join(save_dir, "f2_change_map.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# §7  Plot (b) — TEC time series: measured / prior / posterior (2×2 grid)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tec_2x2(results: list[dict], obs_all: list[dict],
                 clean_all: list[dict], save_dir: str) -> None:
    """
    2×2 constellation panel showing measured (circles), prior (dashed),
    and posterior (solid) sTEC vs UTC time for the single ground station
    closest to POI.  Only arcs that fall inside an assimilation window are
    plotted; the x-axis is zoomed to the data extent.
    """
    active = [r for r in results if r is not None]

    # Build a fast lookup: clean_entry → obs dict (for constellation + station)
    ce_to_obs = {}
    for obs, ce in zip(obs_all, clean_all):
        ce_to_obs[id(ce)] = obs

    # ── Find the ground station closest to POI ────────────────────────────────
    # obs dicts from IGSTECPipeline.run() / _compute_sv_tec use "station_id",
    # "station_lat", "station_lon" (not "station" / "rx_lat" / "rx_lon").
    sta_pos: dict[str, tuple[float, float]] = {}
    for obs in obs_all:
        code = obs.get("station_id", "?")
        if code not in sta_pos:
            sta_pos[code] = (float(obs.get("station_lat", np.nan)),
                             float(obs.get("station_lon", np.nan)))

    closest_sta: str | None = None
    closest_dist = np.inf
    for code, (lat, lon) in sta_pos.items():
        if np.isfinite(lat) and np.isfinite(lon):
            d = _haversine_deg(POI_LAT, POI_LON, lat, lon)
            if d < closest_dist:
                closest_dist = d
                closest_sta  = code

    def _sta(ce: dict) -> str:
        obs = ce_to_obs.get(id(ce))
        return obs.get("station_id", "?") if obs else "?"

    def _keep(ce: dict) -> bool:
        return closest_sta is None or _sta(ce) == closest_sta

    # ── Determine x-axis bounds from the closest-station window arcs only ─────
    all_t: list[float] = []
    for res in active:
        for ce in res["clean_window"]:
            if not _keep(ce):
                continue
            t = ce.get("time_utc_h", np.array([]))
            if len(t):
                all_t.extend(t.tolist())
    if all_t:
        t_lo = max(0.0,  float(np.min(all_t)) - 0.15)
        t_hi = min(24.0, float(np.max(all_t)) + 0.15)
    else:
        t_lo, t_hi = 0.0, 24.0
    # Adaptive tick spacing
    t_span = t_hi - t_lo
    tick_step = 3.0 if t_span > 6 else (1.0 if t_span > 2 else 0.25)

    sta_label = (f"{closest_sta}  ({closest_dist:.2f}° from POI)"
                 if closest_sta else "all stations")

    fig, ax_arr = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Ground-Station sTEC — {CAMPAIGN_DATE.strftime('%Y-%m-%d')}  ·  {sta_label}\n"
        f"● Measured  ── Posterior (KF)  - - Prior (IRI)",
        fontsize=11,
    )
    axes: dict[str, plt.Axes] = {}
    for conid, cfg in CONST_CFG.items():
        r, c = cfg["panel"]
        ax   = ax_arr[r, c]
        ax.set_title(cfg["name"], fontsize=11, color=cfg["color"], fontweight="bold")
        ax.set_xlabel("UTC (hours)", fontsize=9)
        ax.set_ylabel("sTEC (TECU)", fontsize=9)
        ax.set_xlim(t_lo, t_hi)
        ticks = np.arange(math.ceil(t_lo / tick_step) * tick_step,
                          t_hi + 1e-9, tick_step)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: f"{v:.2g}"))
        ax.grid(True, alpha=0.25, ls=":")
        # Highlight each assimilation window
        for res in active:
            tc  = res["t_centre"]
            wh  = WINDOW_MIN / 60.0
            w0  = tc.hour + tc.minute / 60.0 - wh / 2
            w1  = w0 + wh
            ax.axvspan(w0, w1, color="lightyellow", alpha=0.6, zorder=0)
        axes[conid] = ax

    # Arc colour map — built from closest-station window arcs only
    all_prn_ids = sorted({ce["prn_id"]
                          for res in active for ce in res["clean_window"]
                          if _keep(ce)})
    cmap_arc    = mpl.colormaps.get_cmap("tab20")
    prn_col     = {prn: cmap_arc(i % 20) for i, prn in enumerate(all_prn_ids)}

    # Keep track of legend handles per constellation
    legend_handles: dict[str, list] = {c: [] for c in CONST_CFG}
    plotted_prns:   dict[str, set]  = {c: set() for c in CONST_CFG}

    # Plot measured TEC — closest station only
    for res in active:
        for ce in res["clean_window"]:
            if not _keep(ce):
                continue
            obs = ce_to_obs.get(id(ce))
            if obs is None:
                continue
            conid  = obs.get("conid", "?")
            if conid not in axes:
                continue
            ax     = axes[conid]
            prn    = ce["prn_id"]
            col    = prn_col.get(prn, "gray")
            t_utc  = ce.get("time_utc_h", np.array([]))
            tec    = ce.get("tec", np.array([]))
            if len(t_utc) < 2:
                continue
            ax.scatter(t_utc, tec, s=3, color=col, alpha=0.50, zorder=3)
            if prn not in plotted_prns[conid]:
                legend_handles[conid].append(
                    Line2D([0], [0], color=col, lw=1.5, label=prn)
                )
                plotted_prns[conid].add(prn)

    # Overlay prior and posterior — closest station only
    for res in active:
        ce_w      = res["clean_window"]
        arc_sizes = res["arc_sizes"]   # parallel to ce_w; 0 for skipped arcs
        offset    = 0
        for i, ce in enumerate(ce_w):
            ns = arc_sizes[i] if i < len(arc_sizes) else 0
            sl = slice(offset, offset + ns)
            offset += ns
            if ns == 0:
                continue
            if not _keep(ce):
                continue
            obs = ce_to_obs.get(id(ce))
            if obs is None:
                continue
            conid = obs.get("conid", "?")
            if conid not in axes:
                continue
            ax    = axes[conid]
            prn   = ce["prn_id"]
            col   = prn_col.get(prn, "gray")
            t_utc = ce.get("time_utc_h", np.array([]))
            if len(t_utc) < 2:
                continue
            # Subsampled time axis (same stride as rays)
            gnss_arr = ce.get("GNSS")
            if gnss_arr is not None:
                n_s    = gnss_arr.shape[1]
                stride = max(1, int(np.ceil(n_s / MAX_RAYS_PER_ARC)))
                idx_s  = np.arange(0, n_s, stride)
                t_sub  = t_utc[idx_s] if len(t_utc) == n_s else t_utc[:ns]
            else:
                t_sub = t_utc[:ns]
            t_plot = t_sub[:ns]
            y_pri  = res["Y_all_prior_mean"][sl]
            y_pos  = res["Y_all_post_mean"][sl]
            ax.plot(t_plot, y_pri, color="gray", lw=0.7, ls="--", alpha=0.55, zorder=4)
            ax.plot(t_plot, y_pos, color=col,    lw=1.0, ls="-",  alpha=0.80, zorder=5)

    # Style legend entries
    style_h = [
        Line2D([0], [0], color="gray", lw=1.2, ls="--", label="Prior (IRI)"),
        Line2D([0], [0], color="gray", lw=1.2, ls="-",  label="Posterior (KF)"),
    ]
    for conid, cfg in CONST_CFG.items():
        ax = axes.get(conid)
        if ax is None:
            continue
        h = legend_handles.get(conid, [])
        if h:
            ax.legend(handles=style_h + h[:12], fontsize=5.5,
                      loc="upper right", framealpha=0.85,
                      ncol=max(1, len(h) // 8))

    plt.tight_layout()
    fpath = os.path.join(save_dir, "tec_2x2_prior_posterior.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def _window_edges(window_min: int = WINDOW_MIN) -> list[tuple[float, float]]:
    step = window_min / 60.0
    edges, t = [], 0.0
    while t < 24.0:
        edges.append((t, min(t + step, 24.0)))
        t += step
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# §8  Plot (c) — pass-by-pass arc innovation diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def plot_arc_innovation_diagnostic(
    results:   list[dict],
    clean_all: list[dict],
    obs_all:   list[dict],
    save_dir:  str,
) -> None:
    """
    5-panel figure (bar chart / RMSE scatter / geographic map / KDE histogram
    / constellation RMSE) aggregated across all assimilation windows.

    Only arcs from the ground station closest to POI are included, so the
    diagnostic isolates a single receiver's geometry.
    """
    from scipy.stats import gaussian_kde as _kde

    # ── Find the ground station closest to POI ────────────────────────────────
    obs_by_ce_id: dict[int, dict] = {id(ce): obs
                                      for obs, ce in zip(obs_all, clean_all)}

    # obs dicts from IGSTECPipeline.run() / _compute_sv_tec use "station_id",
    # "station_lat", "station_lon" (not "station" / "rx_lat" / "rx_lon").
    sta_pos: dict[str, tuple[float, float]] = {}   # station → (lat, lon)
    for obs in obs_all:
        code = obs.get("station_id", "?")
        if code not in sta_pos:
            sta_pos[code] = (float(obs.get("station_lat", np.nan)),
                             float(obs.get("station_lon", np.nan)))

    closest_sta: str | None = None
    closest_dist = np.inf
    for code, (lat, lon) in sta_pos.items():
        if np.isfinite(lat) and np.isfinite(lon):
            d = _haversine_deg(POI_LAT, POI_LON, lat, lon)
            if d < closest_dist:
                closest_dist = d
                closest_sta  = code

    if closest_sta:
        print(f"  [Innovation] Using arcs from closest station: "
              f"{closest_sta}  ({closest_dist:.2f}° from POI)")
    else:
        print("  [Innovation] Could not determine closest station — "
              "using all arcs.")

    def _arc_sta(ce: dict) -> str:
        obs = obs_by_ce_id.get(id(ce))
        return obs.get("station_id", "?") if obs else "?"

    # ── Collect per-arc, per-obs, and per-constellation statistics ───────────
    arc_labels     = []
    arc_prior_mean = []
    arc_post_mean  = []
    arc_prior_rmse = []
    arc_post_rmse  = []
    arc_lats       = []
    arc_lons       = []
    all_prior_resid  = []
    all_post_resid   = []
    # constellation letter → flat list of residuals
    con_prior_resid: dict[str, list] = {}
    con_post_resid:  dict[str, list] = {}

    for res in results:
        if res is None:
            continue
        y_all   = res["y_obs_all"]
        y_pri_m = res["Y_all_prior_mean"]
        y_pos_m = res["Y_all_post_mean"]

        arc_sizes = res["arc_sizes"]   # parallel to clean_window
        offset    = 0
        for i, ce in enumerate(res["clean_window"]):
            ns = arc_sizes[i] if i < len(arc_sizes) else 0
            sl = slice(offset, offset + ns)
            offset += ns
            if ns == 0:
                continue
            # Filter to closest station only
            if closest_sta is not None and _arc_sta(ce) != closest_sta:
                continue
            fin_sl = np.isfinite(y_all[sl])
            if fin_sl.sum() == 0:
                continue
            rp = (y_all[sl] - y_pri_m[sl])[fin_sl]
            ra = (y_all[sl] - y_pos_m[sl])[fin_sl]
            # Accumulate per-obs residuals (for KDE histogram and global RMSE)
            all_prior_resid.extend(rp.tolist())
            all_post_resid.extend(ra.tolist())
            # Per-arc statistics
            prn_label = str(ce.get("prn_id", f"arc{i}"))
            arc_labels.append(prn_label)
            arc_prior_mean.append(float(np.nanmean(rp)))
            arc_post_mean.append(float(np.nanmean(ra)))
            arc_prior_rmse.append(float(np.sqrt(np.nanmean(rp ** 2))))
            arc_post_rmse.append(float(np.sqrt(np.nanmean(ra ** 2))))
            arc_lats.append(float(np.nanmean(res["all_tp_lats"][sl])))
            arc_lons.append(float(np.nanmean(res["all_tp_lons"][sl])))
            # Per-constellation accumulation (first character of prn_id)
            con = prn_label[0] if prn_label and prn_label[0].isalpha() else "?"
            con_prior_resid.setdefault(con, []).extend(rp.tolist())
            con_post_resid.setdefault(con, []).extend(ra.tolist())

    if not arc_labels:
        print("  [Innovation] No arc data — skipping.")
        return

    arc_prior_mean = np.array(arc_prior_mean)
    arc_post_mean  = np.array(arc_post_mean)
    arc_prior_rmse = np.array(arc_prior_rmse)
    arc_post_rmse  = np.array(arc_post_rmse)
    arc_lats       = np.array(arc_lats)
    arc_lons       = np.array(arc_lons)
    all_prior_resid = np.array(all_prior_resid)
    all_post_resid  = np.array(all_post_resid)

    n_arcs = len(arc_labels)
    prior_rmse_g = float(np.sqrt(np.nanmean(all_prior_resid ** 2)))
    post_rmse_g  = float(np.sqrt(np.nanmean(all_post_resid  ** 2)))

    imp_mean = np.abs(arc_post_mean) < np.abs(arc_prior_mean)

    # ── Stratified selection for the bar chart (max 20 arcs) ─────────────────
    # Sort the full arc set by prior_mean (ascending, negative → positive bias)
    # then pick evenly-spaced quantile positions.  This guarantees the 20 shown
    # span the full error distribution rather than clustering at the extremes.
    MAX_BAR_ARCS = 20
    if n_arcs > MAX_BAR_ARCS:
        sorted_by_mean  = np.argsort(arc_prior_mean)          # ascending by bias
        sel_positions   = np.round(
            np.linspace(0, n_arcs - 1, MAX_BAR_ARCS)
        ).astype(int)
        bar_idx         = sorted_by_mean[sel_positions]        # representative subset
        bar_note        = (f"  [{MAX_BAR_ARCS} of {n_arcs} arcs shown, "
                           f"stratified by prior bias]")
    else:
        bar_idx  = np.arange(n_arcs)
        bar_note = ""
    n_bar = len(bar_idx)

    # Within those n_bar arcs, still sort descending by |prior_mean| for display
    bar_sort_idx = bar_idx[np.argsort(np.abs(arc_prior_mean[bar_idx]))[::-1]]

    fig = plt.figure(figsize=(18, max(12, 0.45 * n_bar + 4)))
    gs  = fig.add_gridspec(4, 2, width_ratios=[1.5, 1],
                            height_ratios=[1, 1, 1, 0.72], hspace=0.56, wspace=0.42)
    ax_bar  = fig.add_subplot(gs[:, 0])    # full left column
    ax_scat = fig.add_subplot(gs[0, 1])
    ax_map  = fig.add_subplot(gs[1, 1])
    ax_hist = fig.add_subplot(gs[2, 1])
    ax_con  = fig.add_subplot(gs[3, 1])   # constellation RMSE panel

    # ── Panel A: signed mean residual bar chart (representative 20) ───────────
    bh    = 0.28
    y_pos = np.arange(n_bar, dtype=float)
    for k, si in enumerate(bar_sort_idx):
        y   = y_pos[k]
        imp = bool(imp_mean[si])
        ax_bar.barh(y + bh, arc_prior_mean[si], height=bh * 1.85,
                    color="#2166ac", alpha=0.88,
                    label="Prior  mean(obs−model)" if k == 0 else "")
        bar_col = "#1a9641" if imp else "#d7191c"
        ax_bar.barh(y - bh, arc_post_mean[si], height=bh * 1.85,
                    color=bar_col, alpha=0.84,
                    label=("Post  ↓ improved" if (k == 0 and imp)
                           else ("Post  ↑ degraded" if (k == 0 and not imp) else "")))
    ax_bar.axvline(0, color="k", lw=0.9)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([arc_labels[bar_sort_idx[k]] for k in range(n_bar)],
                            fontsize=7, fontfamily="monospace")
    ax_bar.set_xlabel("Mean residual  obs − model  (TECU)", fontsize=9)
    ax_bar.set_title(
        f"Per-arc mean TEC error — KF  ·  {CAMPAIGN_DATE.strftime('%Y-%m-%d')}\n"
        f"Global RMSE: Prior {prior_rmse_g:.2f} → Post {post_rmse_g:.2f} TECU"
        + bar_note,
        fontsize=9, fontweight="bold",
    )
    handles = [
        mpatches.Patch(color="#2166ac", alpha=0.88, label="Prior  mean(obs−model)"),
        mpatches.Patch(color="#1a9641", alpha=0.84, label="Post  ↓ |bias| reduced"),
        mpatches.Patch(color="#d7191c", alpha=0.84, label="Post  ↑ |bias| increased"),
    ]
    ax_bar.legend(handles=handles, fontsize=8, loc="lower right")
    ax_bar.grid(axis="x", lw=0.4, alpha=0.5)

    # ── Panel B: prior vs posterior RMSE scatter ──────────────────────────────
    delta_rmse = arc_post_rmse - arc_prior_rmse
    v_sc = max(float(np.percentile(np.abs(delta_rmse), 95)), 2.0)
    from matplotlib.colors import Normalize
    norm_sc = Normalize(-v_sc, v_sc)
    sc = ax_scat.scatter(arc_prior_rmse, arc_post_rmse,
                         c=delta_rmse, cmap="RdYlGn_r", norm=norm_sc,
                         s=60, edgecolors="k", linewidths=0.4, zorder=4)
    lim = max(np.concatenate([arc_prior_rmse, arc_post_rmse]).max() * 1.08, 5.0)
    ax_scat.plot([0, lim], [0, lim], "--", color="0.5", lw=0.9)
    ax_scat.set_xlim(0, lim); ax_scat.set_ylim(0, lim)
    ax_scat.set_xlabel("Prior RMSE (TECU)", fontsize=8)
    ax_scat.set_ylabel("Post RMSE (TECU)",  fontsize=8)
    ax_scat.set_title("Prior → Posterior RMSE per arc", fontsize=8)
    fig.colorbar(sc, ax=ax_scat, fraction=0.05, pad=0.02).set_label("ΔRMSE (TECU)", fontsize=7)
    for k in range(min(n_arcs, 40)):
        ax_scat.annotate(arc_labels[k], (arc_prior_rmse[k], arc_post_rmse[k]),
                         fontsize=5, ha="center", va="bottom",
                         xytext=(0, 3), textcoords="offset points")

    # ── Panel C: geographic map ────────────────────────────────────────────────
    _sz_scale = 5.0
    sz_prior  = 20 + _sz_scale * arc_prior_rmse
    sz_post   = 20 + _sz_scale * arc_post_rmse
    v_map     = max(float(np.percentile(np.abs(delta_rmse), 95)), 2.0)
    norm_map  = Normalize(-v_map, v_map)
    ax_map.scatter(arc_lons, arc_lats, s=sz_prior, facecolors="none",
                   edgecolors="#555555", linewidths=1.6, zorder=3)
    sc_map = ax_map.scatter(arc_lons, arc_lats, s=sz_post,
                             c=delta_rmse, cmap="RdYlGn_r", norm=norm_map,
                             alpha=0.82, edgecolors="k", linewidths=0.35, zorder=4)
    for k in range(min(n_arcs, 40)):
        ax_map.annotate(arc_labels[k], (arc_lons[k], arc_lats[k]),
                        fontsize=5, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")
    cb_map = fig.colorbar(sc_map, ax=ax_map, fraction=0.05, pad=0.02)
    cb_map.set_label("ΔRMSE  post−prior (TECU)\n← improved   degraded →", fontsize=7)
    ax_map.set_xlabel("Longitude (°E)", fontsize=8)
    ax_map.set_ylabel("Latitude (°N)",  fontsize=8)
    ax_map.set_title("Prior ○ vs Posterior ● RMSE per arc", fontsize=8)
    ax_map.grid(lw=0.3, alpha=0.4)

    # ── Shared constellation metadata (used by Panels D and E) ──────────────
    _CON_FULL   = {'G': 'GPS', 'R': 'GLONASS', 'E': 'Galileo', 'C': 'BeiDou'}
    _CON_COLORS = {'G': '#E07B00', 'R': '#CC0022', 'E': '#7B2D8B', 'C': '#8B5E3C'}

    # ── Panel D: residual histogram + KDE, with per-constellation post KDEs ──
    finite_all = np.concatenate([all_prior_resid[np.isfinite(all_prior_resid)],
                                  all_post_resid[np.isfinite(all_post_resid)]])
    lo = float(np.percentile(finite_all,  1)) - 5
    hi = float(np.percentile(finite_all, 99)) + 5
    bins = np.linspace(lo, hi, 45)
    x_k  = np.linspace(bins[0], bins[-1], 300)

    # Aggregate prior and posterior histograms + bold KDE lines
    for arr, col, lbl in [
        (all_prior_resid, "#2166ac",
         f"Prior (all)  μ={np.nanmean(all_prior_resid):+.1f}  σ={np.nanstd(all_prior_resid):.1f}"),
        (all_post_resid,  "#1a9641",
         f"Post  (all)  μ={np.nanmean(all_post_resid):+.1f}  σ={np.nanstd(all_post_resid):.1f}"),
    ]:
        fin = arr[np.isfinite(arr)]
        ax_hist.hist(fin, bins=bins, density=True, alpha=0.35, color=col, label=lbl)
        try:
            ax_hist.plot(x_k, _kde(fin)(x_k), color=col, lw=1.8)
        except Exception:
            pass

    # Per-constellation posterior KDE curves (dashed, colour-coded)
    for c in sorted(con_post_resid.keys()):
        fin_c = np.array([v for v in con_post_resid[c] if np.isfinite(v)])
        if len(fin_c) < 10:
            continue
        col_c  = _CON_COLORS.get(c, "#555555")
        name_c = _CON_FULL.get(c, c)
        try:
            ax_hist.plot(x_k, _kde(fin_c)(x_k), color=col_c, lw=1.2, ls="--",
                         label=(f"Post {name_c}"
                                f"  μ={np.nanmean(fin_c):+.1f}"
                                f"  σ={np.nanstd(fin_c):.1f}"))
        except Exception:
            pass

    ax_hist.axvline(0, color="k", lw=0.8, ls="--")
    ax_hist.set_xlabel("Residual  obs − model  (TECU)", fontsize=8)
    ax_hist.set_ylabel("Density", fontsize=8)
    ax_hist.set_title("KF residual distribution — aggregate + post by constellation",
                      fontsize=8)
    ax_hist.legend(fontsize=6.5, ncol=2, loc="upper right")
    ax_hist.grid(lw=0.3, alpha=0.4)

    # ── Panel E: per-constellation prior → posterior RMSE ─────────────────────
    con_keys = sorted(con_prior_resid.keys())
    n_con    = len(con_keys)

    if n_con > 0:
        con_prior_v = np.array([
            np.sqrt(np.nanmean(np.array(con_prior_resid[c]) ** 2))
            for c in con_keys
        ])
        con_post_v = np.array([
            np.sqrt(np.nanmean(np.array(con_post_resid[c]) ** 2))
            for c in con_keys
        ])
        con_n_obs = np.array([len(con_prior_resid[c]) for c in con_keys])

        x_con = np.arange(n_con)
        bw    = 0.32
        ax_con.bar(x_con - bw / 2, con_prior_v, width=bw,
                   color="#2166ac", alpha=0.85, label="Prior RMSE", zorder=3)

        for k, (pv, av) in enumerate(zip(con_prior_v, con_post_v)):
            imp     = av <= pv
            bar_col = "#1a9641" if imp else "#d7191c"
            ax_con.bar(k + bw / 2, av, width=bw,
                       color=bar_col, alpha=0.85,
                       label=("Post ↓ improved" if (k == 0 and imp)
                              else ("Post ↑ degraded" if (k == 0 and not imp) else "")),
                       zorder=3)

            # Value annotation above each bar
            ax_con.text(k - bw / 2, pv + 0.03, f"{pv:.2f}",
                        ha="center", va="bottom", fontsize=7, color="#2166ac")
            pct = 100.0 * (pv - av) / pv if pv > 0 else 0.0
            arrow = "↓" if pct >= 0 else "↑"
            ax_con.text(k + bw / 2, av + 0.03,
                        f"{av:.2f}\n{arrow}{abs(pct):.0f}%",
                        ha="center", va="bottom", fontsize=6.5, color=bar_col)

        ax_con.set_xticks(x_con)
        ax_con.set_xticklabels(
            [f"{_CON_FULL.get(c, c)}\n(n={con_n_obs[k]:,})"
             for k, c in enumerate(con_keys)],
            fontsize=8,
        )
        ax_con.set_ylabel("RMSE (TECU)", fontsize=8)
        ax_con.set_title("Prior → Posterior RMSE by constellation", fontsize=8)
        ax_con.set_ylim(bottom=0,
                        top=max(con_prior_v.max(), con_post_v.max()) * 1.45)
        ax_con.legend(fontsize=7, loc="upper right")
        ax_con.grid(axis="y", lw=0.3, alpha=0.4, zorder=0)
    else:
        ax_con.set_visible(False)

    plt.tight_layout()
    fpath = os.path.join(save_dir, "arc_innovation_diagnostic.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# §8b  Plot (c2) — GLONASS R08 / R26 arc innovation across all stations
# ─────────────────────────────────────────────────────────────────────────────

_GLO_TARGET_PRNS = ("R08", "R26")

def plot_glo_prn_innovation(
    results:     list[dict],
    obs_all:     list[dict],
    clean_all:   list[dict],
    save_dir:    str,
    target_prns: tuple[str, ...] = _GLO_TARGET_PRNS,
) -> None:
    """
    Same 5-panel layout as plot_arc_innovation_diagnostic, filtered to the
    nominated GLONASS PRNs across all ground stations (not just the closest).

    Panel A (left, full height) — horizontal bar chart of mean arc residuals.
                                   Y-labels: "PRN@STA"; sorted by |prior bias|.
    Panel B (top right)         — prior vs posterior RMSE scatter.
    Panel C (middle right)      — geographic map of IPP locations.
    Panel D (lower-mid right)   — KDE residual histogram (prior / posterior /
                                   per-PRN posterior curves).
    Panel E (bottom right)      — per-PRN prior → posterior RMSE bar chart.
    """
    from scipy.stats import gaussian_kde as _kde
    from matplotlib.colors import Normalize

    obs_by_ce_id: dict[int, dict] = {id(ce): obs
                                      for obs, ce in zip(obs_all, clean_all)}

    def _arc_sta(ce: dict) -> str:
        obs = obs_by_ce_id.get(id(ce))
        return obs.get("station_id", "?") if obs else "?"

    print(f"  [GLO Innovation] PRNs: {', '.join(target_prns)} — all stations")

    # ── Collect per-arc statistics ────────────────────────────────────────────
    arc_labels     = []
    arc_prior_mean = []
    arc_post_mean  = []
    arc_prior_rmse = []
    arc_post_rmse  = []
    arc_lats       = []
    arc_lons       = []
    all_prior_resid: list[float] = []
    all_post_resid:  list[float] = []
    prn_prior_resid: dict[str, list] = {}
    prn_post_resid:  dict[str, list] = {}

    for res in results:
        if res is None:
            continue
        y_all   = res["y_obs_all"]
        y_pri_m = res["Y_all_prior_mean"]
        y_pos_m = res["Y_all_post_mean"]
        arc_sizes = res["arc_sizes"]

        offset = 0
        for i, ce in enumerate(res["clean_window"]):
            ns = arc_sizes[i] if i < len(arc_sizes) else 0
            sl = slice(offset, offset + ns)
            offset += ns
            if ns == 0:
                continue

            prn = ce.get("prn_id", "")
            if prn not in target_prns:
                continue

            fin_sl = np.isfinite(y_all[sl])
            if fin_sl.sum() == 0:
                continue

            rp = (y_all[sl] - y_pri_m[sl])[fin_sl]
            ra = (y_all[sl] - y_pos_m[sl])[fin_sl]

            all_prior_resid.extend(rp.tolist())
            all_post_resid.extend(ra.tolist())

            sta   = _arc_sta(ce)
            label = f"{prn}@{sta}"
            arc_labels.append(label)
            arc_prior_mean.append(float(np.nanmean(rp)))
            arc_post_mean.append(float(np.nanmean(ra)))
            arc_prior_rmse.append(float(np.sqrt(np.nanmean(rp ** 2))))
            arc_post_rmse.append(float(np.sqrt(np.nanmean(ra ** 2))))
            arc_lats.append(float(np.nanmean(res["all_tp_lats"][sl])))
            arc_lons.append(float(np.nanmean(res["all_tp_lons"][sl])))

            prn_prior_resid.setdefault(prn, []).extend(rp.tolist())
            prn_post_resid.setdefault(prn,  []).extend(ra.tolist())

    if not arc_labels:
        print(f"  [GLO Innovation] No arc data for {target_prns} — skipping.")
        return

    arc_prior_mean  = np.array(arc_prior_mean)
    arc_post_mean   = np.array(arc_post_mean)
    arc_prior_rmse  = np.array(arc_prior_rmse)
    arc_post_rmse   = np.array(arc_post_rmse)
    arc_lats        = np.array(arc_lats)
    arc_lons        = np.array(arc_lons)
    all_prior_resid = np.array(all_prior_resid)
    all_post_resid  = np.array(all_post_resid)

    n_arcs       = len(arc_labels)
    prior_rmse_g = float(np.sqrt(np.nanmean(all_prior_resid ** 2)))
    post_rmse_g  = float(np.sqrt(np.nanmean(all_post_resid  ** 2)))
    imp_mean     = np.abs(arc_post_mean) < np.abs(arc_prior_mean)

    # ── Stratified bar-chart selection (max 20 arcs) ──────────────────────────
    MAX_BAR_ARCS = 20
    if n_arcs > MAX_BAR_ARCS:
        sorted_by_mean = np.argsort(arc_prior_mean)
        sel_positions  = np.round(
            np.linspace(0, n_arcs - 1, MAX_BAR_ARCS)
        ).astype(int)
        bar_idx  = sorted_by_mean[sel_positions]
        bar_note = (f"  [{MAX_BAR_ARCS} of {n_arcs} arcs shown, "
                    f"stratified by prior bias]")
    else:
        bar_idx  = np.arange(n_arcs)
        bar_note = ""
    n_bar = len(bar_idx)

    bar_sort_idx = bar_idx[np.argsort(np.abs(arc_prior_mean[bar_idx]))[::-1]]

    # ── Figure (mirrors plot_arc_innovation_diagnostic exactly) ──────────────
    fig = plt.figure(figsize=(18, max(12, 0.45 * n_bar + 4)))
    gs  = fig.add_gridspec(4, 2, width_ratios=[1.5, 1],
                            height_ratios=[1, 1, 1, 0.72],
                            hspace=0.56, wspace=0.42)
    ax_bar  = fig.add_subplot(gs[:, 0])
    ax_scat = fig.add_subplot(gs[0, 1])
    ax_map  = fig.add_subplot(gs[1, 1])
    ax_hist = fig.add_subplot(gs[2, 1])
    ax_prn  = fig.add_subplot(gs[3, 1])

    # ── Panel A: signed mean residual horizontal bar chart ────────────────────
    bh    = 0.28
    y_pos = np.arange(n_bar, dtype=float)
    for k, si in enumerate(bar_sort_idx):
        y   = y_pos[k]
        imp = bool(imp_mean[si])
        ax_bar.barh(y + bh, arc_prior_mean[si], height=bh * 1.85,
                    color="#2166ac", alpha=0.88,
                    label="Prior  mean(obs−model)" if k == 0 else "")
        bar_col = "#1a9641" if imp else "#d7191c"
        ax_bar.barh(y - bh, arc_post_mean[si], height=bh * 1.85,
                    color=bar_col, alpha=0.84,
                    label=("Post  ↓ improved" if (k == 0 and imp)
                           else ("Post  ↑ degraded" if (k == 0 and not imp)
                                 else "")))
    ax_bar.axvline(0, color="k", lw=0.9)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([arc_labels[bar_sort_idx[k]] for k in range(n_bar)],
                            fontsize=7, fontfamily="monospace")
    ax_bar.set_xlabel("Mean residual  obs − model  (TECU)", fontsize=9)
    ax_bar.set_title(
        f"Per-arc mean TEC error — KF  ·  {CAMPAIGN_DATE.strftime('%Y-%m-%d')}\n"
        f"GLONASS {' / '.join(target_prns)} — all stations\n"
        f"Global RMSE: Prior {prior_rmse_g:.2f} → Post {post_rmse_g:.2f} TECU"
        + bar_note,
        fontsize=9, fontweight="bold",
    )
    handles = [
        mpatches.Patch(color="#2166ac", alpha=0.88, label="Prior  mean(obs−model)"),
        mpatches.Patch(color="#1a9641", alpha=0.84, label="Post  ↓ |bias| reduced"),
        mpatches.Patch(color="#d7191c", alpha=0.84, label="Post  ↑ |bias| increased"),
    ]
    ax_bar.legend(handles=handles, fontsize=8, loc="lower right")
    ax_bar.grid(axis="x", lw=0.4, alpha=0.5)

    # ── Panel B: prior vs posterior RMSE scatter ──────────────────────────────
    delta_rmse = arc_post_rmse - arc_prior_rmse
    v_sc    = max(float(np.percentile(np.abs(delta_rmse), 95)), 2.0)
    norm_sc = Normalize(-v_sc, v_sc)
    sc = ax_scat.scatter(arc_prior_rmse, arc_post_rmse,
                         c=delta_rmse, cmap="RdYlGn_r", norm=norm_sc,
                         s=60, edgecolors="k", linewidths=0.4, zorder=4)
    lim = max(np.concatenate([arc_prior_rmse, arc_post_rmse]).max() * 1.08, 5.0)
    ax_scat.plot([0, lim], [0, lim], "--", color="0.5", lw=0.9)
    ax_scat.set_xlim(0, lim); ax_scat.set_ylim(0, lim)
    ax_scat.set_xlabel("Prior RMSE (TECU)", fontsize=8)
    ax_scat.set_ylabel("Post RMSE (TECU)",  fontsize=8)
    ax_scat.set_title("Prior → Posterior RMSE per arc", fontsize=8)
    fig.colorbar(sc, ax=ax_scat, fraction=0.05, pad=0.02).set_label("ΔRMSE (TECU)", fontsize=7)
    for k in range(min(n_arcs, 40)):
        ax_scat.annotate(arc_labels[k], (arc_prior_rmse[k], arc_post_rmse[k]),
                         fontsize=5, ha="center", va="bottom",
                         xytext=(0, 3), textcoords="offset points")

    # ── Panel C: geographic map ───────────────────────────────────────────────
    _sz_scale = 5.0
    sz_prior  = 20 + _sz_scale * arc_prior_rmse
    sz_post   = 20 + _sz_scale * arc_post_rmse
    v_map     = max(float(np.percentile(np.abs(delta_rmse), 95)), 2.0)
    norm_map  = Normalize(-v_map, v_map)
    ax_map.scatter(arc_lons, arc_lats, s=sz_prior, facecolors="none",
                   edgecolors="#555555", linewidths=1.6, zorder=3)
    sc_map = ax_map.scatter(arc_lons, arc_lats, s=sz_post,
                             c=delta_rmse, cmap="RdYlGn_r", norm=norm_map,
                             alpha=0.82, edgecolors="k", linewidths=0.35, zorder=4)
    for k in range(min(n_arcs, 40)):
        ax_map.annotate(arc_labels[k], (arc_lons[k], arc_lats[k]),
                        fontsize=5, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")
    cb_map = fig.colorbar(sc_map, ax=ax_map, fraction=0.05, pad=0.02)
    cb_map.set_label("ΔRMSE  post−prior (TECU)\n← improved   degraded →", fontsize=7)
    ax_map.set_xlabel("Longitude (°E)", fontsize=8)
    ax_map.set_ylabel("Latitude (°N)",  fontsize=8)
    ax_map.set_title("Prior ○ vs Posterior ● RMSE per arc", fontsize=8)
    ax_map.grid(lw=0.3, alpha=0.4)

    # ── Panel D: residual KDE histogram ──────────────────────────────────────
    _PRN_COLORS = {
        "R08": "#CC0022", "R26": "#7B2D8B",
        "R01": "#E07B00", "R02": "#1A6B3C",
    }
    finite_all = np.concatenate([all_prior_resid[np.isfinite(all_prior_resid)],
                                  all_post_resid[np.isfinite(all_post_resid)]])
    lo   = float(np.percentile(finite_all,  1)) - 5
    hi   = float(np.percentile(finite_all, 99)) + 5
    bins = np.linspace(lo, hi, 45)
    x_k  = np.linspace(bins[0], bins[-1], 300)

    for arr, col, lbl in [
        (all_prior_resid, "#2166ac",
         f"Prior (all)  μ={np.nanmean(all_prior_resid):+.1f}  σ={np.nanstd(all_prior_resid):.1f}"),
        (all_post_resid,  "#1a9641",
         f"Post  (all)  μ={np.nanmean(all_post_resid):+.1f}  σ={np.nanstd(all_post_resid):.1f}"),
    ]:
        fin = arr[np.isfinite(arr)]
        ax_hist.hist(fin, bins=bins, density=True, alpha=0.35, color=col, label=lbl)
        try:
            ax_hist.plot(x_k, _kde(fin)(x_k), color=col, lw=1.8)
        except Exception:
            pass

    for p in sorted(prn_post_resid.keys()):
        fin_p = np.array([v for v in prn_post_resid[p] if np.isfinite(v)])
        if len(fin_p) < 10:
            continue
        col_p = _PRN_COLORS.get(p, "#555555")
        try:
            ax_hist.plot(x_k, _kde(fin_p)(x_k), color=col_p, lw=1.2, ls="--",
                         label=(f"Post {p}"
                                f"  μ={np.nanmean(fin_p):+.1f}"
                                f"  σ={np.nanstd(fin_p):.1f}"))
        except Exception:
            pass

    ax_hist.axvline(0, color="k", lw=0.8, ls="--")
    ax_hist.set_xlabel("Residual  obs − model  (TECU)", fontsize=8)
    ax_hist.set_ylabel("Density", fontsize=8)
    ax_hist.set_title("KF residual distribution — aggregate + post by PRN", fontsize=8)
    ax_hist.legend(fontsize=6.5, ncol=2, loc="upper right")
    ax_hist.grid(lw=0.3, alpha=0.4)

    # ── Panel E: per-PRN prior → posterior RMSE bar chart ────────────────────
    prn_keys  = sorted(prn_prior_resid.keys())
    n_prn_bar = len(prn_keys)

    if n_prn_bar > 0:
        prn_prior_v = np.array([
            np.sqrt(np.nanmean(np.array(prn_prior_resid[p]) ** 2))
            for p in prn_keys
        ])
        prn_post_v = np.array([
            np.sqrt(np.nanmean(np.array(prn_post_resid[p]) ** 2))
            for p in prn_keys
        ])
        prn_n_obs = np.array([len(prn_prior_resid[p]) for p in prn_keys])

        x_prn = np.arange(n_prn_bar)
        bw    = 0.32
        ax_prn.bar(x_prn - bw / 2, prn_prior_v, width=bw,
                   color="#2166ac", alpha=0.85, label="Prior RMSE", zorder=3)

        for k, (pv, av) in enumerate(zip(prn_prior_v, prn_post_v)):
            imp     = av <= pv
            bar_col = "#1a9641" if imp else "#d7191c"
            ax_prn.bar(k + bw / 2, av, width=bw, color=bar_col, alpha=0.85,
                       label=("Post ↓ improved" if (k == 0 and imp)
                              else ("Post ↑ degraded" if (k == 0 and not imp)
                                    else "")),
                       zorder=3)
            ax_prn.text(k - bw / 2, pv + 0.03, f"{pv:.2f}",
                        ha="center", va="bottom", fontsize=7, color="#2166ac")
            pct   = 100.0 * (pv - av) / pv if pv > 0 else 0.0
            arrow = "↓" if pct >= 0 else "↑"
            ax_prn.text(k + bw / 2, av + 0.03,
                        f"{av:.2f}\n{arrow}{abs(pct):.0f}%",
                        ha="center", va="bottom", fontsize=6.5, color=bar_col)

        ax_prn.set_xticks(x_prn)
        ax_prn.set_xticklabels(
            [f"{p}\n(n={prn_n_obs[k]:,})" for k, p in enumerate(prn_keys)],
            fontsize=8,
        )
        ax_prn.set_ylabel("RMSE (TECU)", fontsize=8)
        ax_prn.set_title("Prior → Posterior RMSE by PRN", fontsize=8)
        ax_prn.set_ylim(bottom=0,
                        top=max(prn_prior_v.max(), prn_post_v.max()) * 1.45)
        ax_prn.legend(fontsize=7, loc="upper right")
        ax_prn.grid(axis="y", lw=0.3, alpha=0.4, zorder=0)
    else:
        ax_prn.set_visible(False)

    plt.tight_layout()
    fname = "glo_" + "_".join(p.lower() for p in target_prns) + "_innovation.png"
    fpath = os.path.join(save_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# §9  Plot (d) — prior / posterior / ΔNe EDP profiles with ±1σ bands
# ─────────────────────────────────────────────────────────────────────────────

def plot_edp_profiles(results: list[dict],
                      grid_lats: np.ndarray, grid_lons: np.ndarray,
                      alt_grid: np.ndarray,
                      save_dir: str) -> None:
    """
    Three-column figure showing the Kalman-filter prior and posterior Ne(h)
    profiles for each assimilation window (one row per window).

    Left   — Prior Ne(h): one coloured line per grid column,
              global mean as bold black dashed, ±1σ shaded.
    Centre — Posterior Ne(h): same layout, ±1σ from posterior covariance diagonal.
    Right  — ΔNe(h) = posterior − prior per grid column,
              global mean delta as bold black solid.
    """
    active = [r for r in results if r is not None]
    if not active:
        print("  [EDP] No results — skipping EDP profile plot.")
        return

    n_wins = len(active)
    n_geo  = len(grid_lats)

    fig, ax_mat = plt.subplots(n_wins, 3,
                                figsize=(14, max(4, 3.5 * n_wins)),
                                squeeze=False, sharey=True)
    fig.suptitle(
        f"EDP Profiles — Prior / Posterior / ΔNe\n"
        f"{CAMPAIGN_DATE.strftime('%Y-%m-%d')}  "
        f"POI: ({POI_LAT:.0f}°N, {POI_LON:.0f}°E)",
        fontsize=12, y=1.01,
    )

    # Centre grid column is highlighted bold; all others are faint
    ctr_g      = n_geo // 2
    _COL_LIGHT = "#7EB6D9"   # light steel-blue for non-centre lines
    _COL_BOLD  = "#1A3A5C"   # dark navy for the centre line
    _COL_DELTA = "#A0522D"   # sienna for ΔNe centre highlight

    for row, res in enumerate(active):
        ax_pri  = ax_mat[row][0]
        ax_post = ax_mat[row][1]
        ax_del  = ax_mat[row][2]
        label   = res["t_centre"].strftime("%H:%M UTC")

        prior_edp   = res["prior_edp"]    # (n_alt, n_grid)
        post_edp    = res["post_edp"]
        prior_sigma = res["prior_sigma"]  # (n_alt, n_grid)
        post_sigma  = res["post_sigma"]

        pri_mean  = prior_edp.mean(axis=1)
        post_mean = post_edp.mean(axis=1)
        pri_sig_m = prior_sigma.mean(axis=1)
        pos_sig_m = post_sigma.mean(axis=1)

        # ── Prior ─────────────────────────────────────────────────────────────
        # Draw non-centre columns first (behind), then the centre column on top
        for g in range(n_geo):
            if g == ctr_g:
                continue
            ax_pri.plot(np.maximum(prior_edp[:, g], 1.0), alt_grid,
                        color=_COL_LIGHT, lw=0.6, alpha=0.35, zorder=2)
        ax_pri.plot(np.maximum(prior_edp[:, ctr_g], 1.0), alt_grid,
                    color=_COL_BOLD, lw=2.2, alpha=1.0, zorder=4,
                    label=f"Centre col ({ctr_g})")
        ax_pri.plot(np.maximum(pri_mean, 1.0), alt_grid,
                    color="black", lw=1.4, ls="--", zorder=5, label="Mean")
        # ax_pri.fill_betweenx(alt_grid,
        #                       np.maximum(pri_mean - pri_sig_m, 1.0),
        #                       pri_mean + pri_sig_m,
        #                       color="black", alpha=0.10, zorder=3)
        # ax_pri.set_xscale("log")
        ax_pri.set_title(f"{label}\nPrior Ne(h)", fontsize=8)
        ax_pri.set_xlabel("Ne (m⁻³)", fontsize=7)
        ax_pri.set_ylabel("Alt (km)", fontsize=7)
        ax_pri.tick_params(labelsize=6)
        ax_pri.grid(True, alpha=0.3, ls=":")
        ax_pri.legend(fontsize=6, loc="upper right")

        # ── Posterior ─────────────────────────────────────────────────────────
        for g in range(n_geo):
            if g == ctr_g:
                continue
            ax_post.plot(np.maximum(post_edp[:, g], 1.0), alt_grid,
                         color=_COL_LIGHT, lw=0.6, alpha=0.35, zorder=2)
        ax_post.plot(np.maximum(post_edp[:, ctr_g], 1.0), alt_grid,
                     color=_COL_BOLD, lw=2.2, alpha=1.0, zorder=4,
                     label=f"Centre col ({ctr_g})")
        ax_post.plot(np.maximum(post_mean, 1.0), alt_grid,
                     color="black", lw=1.4, ls="--", zorder=5, label="Mean")
        # ax_post.fill_betweenx(alt_grid,
        #                        np.maximum(post_mean - pos_sig_m, 1.0),
        #                        post_mean + pos_sig_m,
        #                        color="black", alpha=0.10, zorder=3)
        # ax_post.set_xscale("log")
        ax_post.set_title(f"{label}\nPosterior Ne(h)", fontsize=8)
        ax_post.set_xlabel("Ne (m⁻³)", fontsize=7)
        ax_post.tick_params(labelsize=6)
        ax_post.grid(True, alpha=0.3, ls=":")
        ax_post.legend(fontsize=6, loc="upper right")

        # ── ΔNe ───────────────────────────────────────────────────────────────
        delta_mean = np.zeros(len(alt_grid))
        for g in range(n_geo):
            delta_g = post_edp[:, g] - prior_edp[:, g]
            delta_mean += delta_g
            if g == ctr_g:
                continue
            ax_del.plot(delta_g, alt_grid,
                        color=_COL_LIGHT, lw=0.6, alpha=0.35, zorder=2)
        delta_ctr  = post_edp[:, ctr_g] - prior_edp[:, ctr_g]
        delta_mean /= max(n_geo, 1)
        ax_del.plot(delta_ctr, alt_grid,
                    color=_COL_DELTA, lw=2.2, alpha=1.0, zorder=4,
                    label=f"Centre col ({ctr_g})")
        ax_del.plot(delta_mean, alt_grid,
                    color="black", lw=1.4, ls="--", zorder=5, label="Mean Δ")
        ax_del.axvline(0, color="gray", lw=0.8, ls=":")
        ax_del.set_title(f"{label}\nΔNe (post − prior)", fontsize=8)
        ax_del.set_xlabel("ΔNe (m⁻³)", fontsize=7)
        ax_del.tick_params(labelsize=6)
        ax_del.grid(True, alpha=0.3, ls=":")
        ax_del.legend(fontsize=6, loc="upper right")

    ax_mat[0][0].set_ylim(bottom=0)   # shared y-axis: applies to every panel

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fpath = os.path.join(save_dir, "edp_profiles.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# §10  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Step 1: Load TEC arcs ─────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════════")
    print("  §1  Loading IGS ground-station TEC arcs")
    print("══════════════════════════════════════════════════════")
    obs_all, clean_all = load_igs_arcs()

    if not clean_all:
        print("ERROR: No valid TEC arcs available.  Exiting.")
        return

    print("\n──── §1b  GLONASS satellite diagnostics ────────────────────────────")
    print_glonass_diagnostics(obs_all)

    # ── Step 2: Build grid ────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════════")
    print("  §2  Building ionospheric grid")
    print("══════════════════════════════════════════════════════")
    grid_lats, grid_lons = build_grid(
        POI_LAT, POI_LON, SEARCH_RADIUS_DEG,
        GRID_PAD_DEG, GRID_DLAT, GRID_DLON,
    )
    n_geo = len(grid_lats)
    print(f"  Grid: {n_geo} points  (Δlat={GRID_DLAT}°, Δlon={GRID_DLON}°)")
    print(f"  Lat: {grid_lats.min():.1f}°–{grid_lats.max():.1f}°  "
          f"Lon: {grid_lons.min():.1f}°–{grid_lons.max():.1f}°")

    # ── Step 3: Build 90-minute windows ───────────────────────────────────────
    wins    = window_centres(CAMPAIGN_DATE, WINDOW_MIN)
    results: list[dict | None] = []

    print("\n══════════════════════════════════════════════════════")
    print(f"  §3  Assimilation loop  ({len(wins)} windows × {WINDOW_MIN} min)")
    print("══════════════════════════════════════════════════════")

    # Compute background covariance structure once for the day
    print("\n  Building background covariance from diurnal IRI variability …")
    t_noon = pd.Timestamp(CAMPAIGN_DATE.year, CAMPAIGN_DATE.month, CAMPAIGN_DATE.day,
                          12, 0, 0)
    try:
        sigma_v, C_v, C_s = _build_prior_covariance(
            t_noon, grid_lats, grid_lons, ALT_GRID_KM,
            n_iri_samples=12,
        )
        print(f"  sigma_v range: {sigma_v.min():.2f} – {sigma_v.max():.2f}  "
              f"(fraction of prior Ne)")
    except Exception as exc:
        print(f"  IRI covariance failed ({exc}) — using defaults.")
        sigma_v = np.full(len(ALT_GRID_KM), SIGMA_NE_FLOOR)
        dalt    = np.abs(ALT_GRID_KM[:, None] - ALT_GRID_KM[None, :])
        C_v     = np.exp(-dalt / V_CORR_KM) + 1e-6 * np.eye(len(ALT_GRID_KM))
        dist    = _haversine_km(
            grid_lats[:, None], grid_lons[:, None],
            grid_lats[None, :], grid_lons[None, :],
        )
        C_s  = np.exp(-dist / CORR_LENGTH_KM)
        C_s += 1e-6 * np.eye(n_geo)
        C_s /= C_s.diagonal()[:, None]

    for w_idx, t_centre in enumerate(wins[:1]):
        print(f"\n  Window {w_idx+1:2d}/{len(wins)}: {t_centre.strftime('%Y-%m-%d %H:%M')} UTC",
              flush=True)

        clean_w = filter_arcs_for_window(clean_all, t_centre, WINDOW_MIN)

        # Geographic check: drop arcs whose mean IPP is outside the grid bbox.
        # Low-elevation GNSS links can have pierce points far from the receiver,
        # outside the region the grid can actually represent.
        clean_w, n_geo_rej = filter_arcs_for_region(
            clean_w, grid_lats, grid_lons, pad_deg=0.5
        )
        print(f"    {len(clean_w)} arcs in window and region"
              + (f"  ({n_geo_rej} outside grid bbox — skipped)"
                 if n_geo_rej else ""))

        if not clean_w:
            results.append(None)
            continue

        # IRI prior at window centre
        try:
            ne_prior = _get_iri_prior(
                t_centre, grid_lats, grid_lons, ALT_GRID_KM
            )
        except Exception as exc:
            print(f"    IRI prior failed: {exc} — skipping window.")
            results.append(None)
            continue

        # Run the selected solver
        _solver_name = "SRIF" if USE_SRIF else "KF"
        try:
            if USE_SRIF:
                res = run_srif_window(
                    clean_window     = clean_w,
                    t_centre         = t_centre,
                    grid_lats        = grid_lats,
                    grid_lons        = grid_lons,
                    alt_grid         = ALT_GRID_KM,
                    ne_prior         = ne_prior,
                    sigma_v          = sigma_v,
                    C_v              = C_v,
                    C_s              = C_s,
                    sigma_obs        = SIGMA_OBS_TECU,
                    srif_chunk_size  = SRIF_CHUNK_SIZE,
                )
            else:
                res = run_kf_window(
                    clean_window = clean_w,
                    t_centre     = t_centre,
                    grid_lats    = grid_lats,
                    grid_lons    = grid_lons,
                    alt_grid     = ALT_GRID_KM,
                    ne_prior     = ne_prior,
                    sigma_v      = sigma_v,
                    C_v          = C_v,
                    C_s          = C_s,
                    sigma_obs    = SIGMA_OBS_TECU,
                )
            results.append(res)
        except Exception as exc:
            import traceback
            print(f"    {_solver_name} failed: {exc}")
            traceback.print_exc()
            results.append(None)
            continue

        if res is None:
            continue

        # ── Per-window figures ─────────────────────────────────────────────
        win_tag  = t_centre.strftime("%H%M")
        win_dir  = os.path.join(SAVE_DIR, f"window_{win_tag}")
        os.makedirs(win_dir, exist_ok=True)

        print(f"\n    Generating figures for window {win_tag} …", flush=True)

        print("      (a) F2 peak density change map …")
        plot_f2_change_map([res], grid_lats, grid_lons, win_dir)

        print("      (b) TEC 2×2 constellation panel …")
        plot_tec_2x2([res], obs_all, clean_all, win_dir)

        print("      (c) Arc innovation diagnostic …")
        plot_arc_innovation_diagnostic([res], clean_all, obs_all, win_dir)

        print("      (c2) GLONASS R08 / R26 arc innovation — all stations …")
        plot_glo_prn_innovation([res], obs_all, clean_all, win_dir)

        print("      (d) EDP profiles …")
        plot_edp_profiles([res], grid_lats, grid_lons, ALT_GRID_KM, win_dir)

    n_active = sum(1 for r in results if r is not None)
    print(f"\n  Assimilation complete: {n_active}/{len(wins)} windows produced results.")

    # ── Step 4: Aggregate figures across all windows ──────────────────────────
    print("\n══════════════════════════════════════════════════════")
    print("  §4  Generating aggregate figures (all windows combined)")
    print("══════════════════════════════════════════════════════")

    print("\n  (a) F2 peak density change map (all windows) …")
    plot_f2_change_map(results, grid_lats, grid_lons, SAVE_DIR)

    print("\n  (b) TEC 2×2 constellation panel (all windows) …")
    plot_tec_2x2(results, obs_all, clean_all, SAVE_DIR)

    print("\n  (c) Arc innovation diagnostic (all windows) …")
    plot_arc_innovation_diagnostic(results, clean_all, obs_all, SAVE_DIR)

    print("\n  (c2) GLONASS R08 / R26 arc innovation — all stations …")
    plot_glo_prn_innovation(results, obs_all, clean_all, SAVE_DIR)

    print("\n  (d) EDP profiles (all windows) …")
    plot_edp_profiles(results, grid_lats, grid_lons, ALT_GRID_KM, SAVE_DIR)

    print(f"\n  All figures saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
