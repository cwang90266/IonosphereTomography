#!/usr/bin/env python3
"""
demo_suborbital_igs.py
======================
Joint Kalman Filter assimilation of suborbital GNSS-TEC observations
(from Data/STEC_Suborbital/) and absolute sTEC from three IGS ground
stations (PIE1, SGPO, MDO1).

Pipeline
--------
1.  Load all 27 suborbital arcs from the CSV export.
2.  Download / cache RINEX data for PIE1, SGPO, MDO1 via the IGS TEC
    pipeline and convert each satellite arc to the standard clean_list
    format used by the Kalman Filter.
3.  Build an IRI-2020 regional prior (southwestern USA / Mexico)
    for the UTC hour of the campaign.
4.  Run a joint Kalman Filter update over all observations.
5.  Write diagnostic figures to ./Figures/Demo_Suborbital_IGS/.

Figures produced
----------------
obs_geometry_map.png   — map of suborbital TEC-max tangent points + IGS IPPs
ray_counts.png         — bars of rays per arc coloured by source
tec_fits_time.png      — per-arc sTEC-vs-UTC-hour panels (meas / prior / post)
tec_by_prn.png         — 2×2 constellation grid, one curve per PRN
edp_prior_vs_posterior.png  — Ne altitude profiles
regional_delta_ne.png  — ΔNe coolwarm map at F2-peak + TEC residual IPPs

Usage
-----
    python demo_suborbital_igs.py

NASA Earthdata credentials (for CDDIS RINEX download) must be stored in
~/.netrc:
    machine urs.earthdata.nasa.gov login <user> password <pass>

If the credentials are absent the IGS download will fail with a warning;
the script will continue with only the suborbital observations.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EDPSamples" / "Locate in mesh" / "outputs"))
sys.path.insert(0, str(ROOT / "iri2020_new" / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd

from EDPSamples.edp_samples import EDPSamples
from Ionosphere_Tomography_Inverter.Ionophy_Tomography_Inverter import (
    Ionosphere_Tomography_Inverter,
)
from TEC_model.igs_tec_pipeline import (
    process_igs_station,
    igs_obs_to_clean_entry,
)
from demo import build_daily_global_edps, extract_robust_f2_peak

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CAMPAIGN_DATE = datetime(2025, 9, 18)   # date matching the suborbital dataset
PROFILE_HOUR  = 13                      # IRI prior hour (12:55–13:03 UTC event)

IGS_STATIONS   = ["PIE1", "SGPO", "MDO1"]
RINEX_VERSION  = 3
RINEX_CACHE    = str(ROOT / "Data" / "RINEX_Cache")

SUBORBITAL_CSV = ROOT / "Data" / "STEC_Suborbital" / "stec_suborbital_export.csv"

# time_reference in the suborbital CSV is seconds_since_2025-09-18T12:55:43Z
CAMPAIGN_START_SOD = 12 * 3600.0 + 55 * 60.0 + 43.0  # 46543 s from midnight

# Altitude grid — 60 to 800 km in 10-km steps
ALT_GRID = np.arange(60.0, 801.0, 10.0, dtype=float)

# Region: covers the suborbital TEC-max footprints + the three IGS stations
#   PIE1: 34.3°N, -108.1°E  |  SGPO: ~20°N, -98.7°E  |  MDO1: 30.7°N, -104°E
LAT_MIN, LAT_MAX =  15.0,  58.0
LON_MIN, LON_MAX = -140.0, -78.0

GLOBAL_EDP_DIR   = str(ROOT / "Data" / "Section20_Global_EDPS")
SAVE_DIR         = str(ROOT / "Figures" / "Demo_Suborbital_IGS")

# KF hyper-parameters (mirror demo_group.py defaults)
MEASUREMENT_ERR      = 10.0       # TECU² observation noise variance
RELAXATION           = 0.99       # Gauss-Markov relaxation
NUM_RAY_SEGMENTS     = 300        # integration points per ray in H
GAUSSIAN_COV_SIGMA   = (20, 500)  # (sigma_h_km, sigma_latlon_km)
ALTITUDE_TAPER_KM    = 100.0
ALTITUDE_TAPER_SCALE = 0.05
TOPSIDE_FOLLOW_F2    = True

MAX_RAYS_PER_ARC = 400   # downsample dense arcs before assimilation
MIN_VALID_RAYS   = 20    # drop arcs with fewer valid observations

# Constellation colour config — mirrors demo_group.py
CONSTELLATION_CONFIG = {
    "G": {"name": "GPS",     "cmap": "Blues",   "title_color": "steelblue"},
    "R": {"name": "GLONASS", "cmap": "Purples", "title_color": "mediumpurple"},
    "E": {"name": "Galileo", "cmap": "Oranges", "title_color": "darkorange"},
    "C": {"name": "BeiDou",  "cmap": "Greens",  "title_color": "seagreen"},
}
_CONST_FALLBACK_CMAP = "Greys"
# Fixed (row, col) in the 2×2 PRN panel GridSpec
_CONST_POS = {"G": (0, 0), "R": (1, 0), "E": (0, 1), "C": (1, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# §1  Load suborbital STEC arcs from the CSV export
# ─────────────────────────────────────────────────────────────────────────────

def load_suborbital_arcs(csv_path: Path,
                          max_rays: int = MAX_RAYS_PER_ARC,
                          min_valid: int = MIN_VALID_RAYS) -> list[dict]:
    """Parse stec_suborbital_export.csv and return clean_list dicts.

    Groups rows by (prn_id, conid, leo_id), filters for finite positive TEC,
    and decimates to at most `max_rays` points using a uniform stride that
    preserves the full temporal span of each arc.

    The ``time_s`` field in the CSV is seconds since 2025-09-18T12:55:43Z
    (CAMPAIGN_START_SOD).  We keep that raw value as ``time_s`` for arc-
    relative plots and compute ``time_utc_h`` for cross-arc UTC-hour plots.
    """
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows from {csv_path.name}")

    arc_list: list[dict] = []
    groups = df.groupby(["conid", "prn_id", "leo_id"], sort=False)
    print(f"  Found {len(groups)} unique satellite arcs")

    for (conid, prn_id, leo_id), grp in groups:
        grp = grp.reset_index(drop=True)
        tec = grp["TEC_podTc2_TECU"].values.astype(float)

        valid    = np.isfinite(tec) & (tec > 0.0)
        n_valid  = int(valid.sum())
        if n_valid < min_valid:
            print(f"  [skip] {leo_id}/{conid}{prn_id} — {n_valid} valid rays")
            continue

        # Uniform-stride decimation preserving arc endpoints
        idx_valid = np.where(valid)[0]
        if n_valid > max_rays:
            stride   = int(np.ceil(n_valid / max_rays))
            idx_keep = idx_valid[::stride]
        else:
            idx_keep = idx_valid

        sub = grp.iloc[idx_keep]

        leo_km  = np.vstack([sub["x_LEO_km"].values,
                              sub["y_LEO_km"].values,
                              sub["z_LEO_km"].values])   # (3, N_k) km
        gnss_km = np.vstack([sub["x_GNSS_km"].values,
                              sub["y_GNSS_km"].values,
                              sub["z_GNSS_km"].values])  # (3, N_k) km

        # Use the first row's stored TEC-max tangent location for the map
        lat_tm = float(grp["lat_tecmax_deg"].iloc[0])
        lon_tm = float(grp["lon_tecmax_deg"].iloc[0])

        # Infer arc start datetime from the date columns
        yr = int(grp["year"].iloc[0])
        mo = int(grp["month"].iloc[0])
        dy = int(grp["day"].iloc[0])
        arc_date = pd.Timestamp(yr, mo, dy)

        # Time arrays — CSV time_s is seconds since CAMPAIGN_START_SOD
        t_s_raw   = sub["time_s"].values.astype(float)  # secs from campaign epoch
        time_utc_h = (CAMPAIGN_START_SOD + t_s_raw) / 3600.0  # UTC decimal hours

        arc: dict = {
            # Core fields required by get_observation_operator_batch
            "LEO":       leo_km,
            "GNSS":      gnss_km,
            "tec":       tec[idx_keep],
            "tec_type":  "absolute",
            # Tangent altitude (actual ray tangent, not IPP approximation)
            "tangent_km": sub["tangent_alt_km"].values.astype(float),
            # Time axes
            "time_s":       t_s_raw,          # seconds from campaign epoch (UTC 12:55:43)
            "time_utc_h":   time_utc_h,       # UTC decimal hours for each epoch
            "arc_start_sod": CAMPAIGN_START_SOD + float(t_s_raw[0]),
            # Metadata
            "leo_id":    leo_id,
            "prn_id":    f"{conid}{prn_id}",
            "label":     f"{leo_id}/{conid}{prn_id}",
            "obs_source": "suborbital",
            "lat_tecmax_tangent": lat_tm,
            "lon_tecmax_tangent": lon_tm,
            "date":      arc_date,
        }
        arc_list.append(arc)
        print(f"  [ok] {leo_id}/{conid}{prn_id}: "
              f"{n_valid} valid → {len(idx_keep)} kept  "
              f"({time_utc_h[0]:.4f}–{time_utc_h[-1]:.4f} UTC h)")

    print(f"  → {len(arc_list)} suborbital arcs ready")
    return arc_list


# ─────────────────────────────────────────────────────────────────────────────
# §2  Load IGS ground-station TEC arcs
# ─────────────────────────────────────────────────────────────────────────────

# Per-epoch array keys that must all be masked together when cropping an arc
# to a time window.  LEO / GNSS are (3, N) and handled separately.
_PER_EPOCH_KEYS = (
    "tec", "tangent_km", "time_s", "arc_time_sec", "time_utc_h",
    "ipp_lat", "ipp_lon",
)


def _crop_arc_to_time_window(entry: dict,
                              t_min_h: float,
                              t_max_h: float,
                              min_valid: int = MIN_VALID_RAYS) -> "dict | None":
    """Restrict all per-epoch arrays in *entry* to epochs within [t_min_h, t_max_h].

    Parameters
    ----------
    entry     : clean_list dict with a ``time_utc_h`` array (required).
    t_min_h   : start of keep window, UTC decimal hours.
    t_max_h   : end   of keep window, UTC decimal hours.
    min_valid : return None if fewer than this many epochs survive the crop.

    Returns
    -------
    A new dict with all per-epoch arrays masked, or None if the arc has no
    overlap with the window or too few surviving epochs.
    """
    t_utc = entry.get("time_utc_h")
    if t_utc is None or len(t_utc) == 0:
        return None

    mask = (t_utc >= t_min_h) & (t_utc <= t_max_h)
    n_keep = int(mask.sum())
    if n_keep < min_valid:
        return None

    result = dict(entry)  # shallow copy — we'll replace arrays in-place

    # 1-D per-epoch arrays
    n_orig = len(t_utc)
    for key in _PER_EPOCH_KEYS:
        val = result.get(key)
        if val is not None and hasattr(val, "__len__") and len(val) == n_orig:
            result[key] = val[mask]

    # (3, N) position arrays
    for key in ("LEO", "GNSS"):
        val = result.get(key)
        if val is not None and getattr(val, "ndim", 0) == 2 and val.shape[1] == n_orig:
            result[key] = val[:, mask]

    return result


def load_igs_arcs(stations: list[str],
                  date: datetime,
                  cache_dir: str = RINEX_CACHE,
                  max_rays: int  = MAX_RAYS_PER_ARC,
                  time_window_h: "tuple[float, float] | None" = None) -> list[dict]:
    """Run IGSTECPipeline for each station and return clean_list entries.

    Parameters
    ----------
    stations       : list of IGS station 4-character codes.
    date           : campaign date.
    cache_dir      : local RINEX cache directory.
    max_rays       : maximum epochs kept per arc after decimation.
    time_window_h  : optional (t_min, t_max) in UTC decimal hours.  When
                     supplied every arc is cropped to this window before being
                     accepted, so only epochs that overlap the suborbital
                     campaign contribute to the joint assimilation.

    Failed stations are silently skipped so the rest of the script can
    proceed with whatever data is available.
    """
    os.makedirs(cache_dir, exist_ok=True)
    all_entries: list[dict] = []

    if time_window_h is not None:
        t_min_h, t_max_h = time_window_h
        print(f"  Time window: {t_min_h:.4f} – {t_max_h:.4f} UTC h  "
              f"({(t_max_h - t_min_h) * 60:.1f} min)")

    for sta in stations:
        print(f"\n  Processing IGS station: {sta}")
        try:
            obs_list = process_igs_station(
                station       = sta,
                date          = date,
                rinex_version = RINEX_VERSION,
                cache_dir     = cache_dir,
                use_iri       = False,
                max_rays      = max_rays,
            )
            n_arcs = len(obs_list)
            print(f"  → {n_arcs} raw arcs from {sta}")
            n_accepted = 0
            for obs in obs_list:
                entry = igs_obs_to_clean_entry(obs,
                                               max_rays  = max_rays,
                                               min_valid = MIN_VALID_RAYS)
                if entry is None:
                    continue

                # Crop to the suborbital time window before accepting
                if time_window_h is not None:
                    entry = _crop_arc_to_time_window(
                        entry, t_min_h, t_max_h, min_valid=MIN_VALID_RAYS
                    )
                    if entry is None:
                        continue  # no overlap or too few epochs after crop

                entry["obs_source"] = "IGS_ground"
                all_entries.append(entry)
                n_accepted += 1
            print(f"  → {n_accepted}/{n_arcs} arcs accepted (≥{MIN_VALID_RAYS} valid rays)")
        except Exception as exc:
            print(f"  [warn] {sta} failed: {exc}")

    print(f"\n  → {len(all_entries)} IGS arcs ready")
    return all_entries


# ─────────────────────────────────────────────────────────────────────────────
# §3  Build regional IRI-2020 prior
# ─────────────────────────────────────────────────────────────────────────────

def build_iri_prior(date: datetime,
                    hour: int,
                    alt_grid: np.ndarray,
                    lat_min: float, lat_max: float,
                    lon_min: float, lon_max: float) -> EDPSamples:
    """Generate (or load from NetCDF cache) the global EDP grids for `date`,
    then subset to the assimilation region.
    """
    ts = pd.Timestamp(date)
    print(f"\n  Building global IRI EDP cache for {ts.date()} …")
    global_cache = build_daily_global_edps(
        date        = ts,
        alt_grid    = alt_grid,
        dLat        = 5.0,
        dLon        = 5.0,
        n_mc        = 50,
        data_dir    = GLOBAL_EDP_DIR,
        num_workers = 8,
    )
    eds_global = global_cache[hour]
    print(f"  Subsetting region: lat [{lat_min:.1f}, {lat_max:.1f}]  "
          f"lon [{lon_min:.1f}, {lon_max:.1f}]")
    eds_region = eds_global.subset_region(lat_min, lat_max, lon_min, lon_max)
    n_geo = eds_region.geolocation.shape[0]
    print(f"  Regional prior: {n_geo} spatial nodes × {len(alt_grid)} altitude levels")
    return eds_region


# ─────────────────────────────────────────────────────────────────────────────
# §4  Internal colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _arc_colour(prn_id: str, idx_in: int, n_in: int) -> tuple:
    """Return an RGBA colour for an arc, shaded within its constellation family."""
    const = prn_id[0].upper() if prn_id else "?"
    cfg   = CONSTELLATION_CONFIG.get(const, {})
    cmap  = mpl.colormaps.get_cmap(cfg.get("cmap", _CONST_FALLBACK_CMAP))
    t     = (0.40 + 0.50 * (idx_in / max(n_in - 1, 1))) if n_in > 1 else 0.70
    return cmap(t)


# ─────────────────────────────────────────────────────────────────────────────
# §5  Diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

def _obs_map(sub_arcs: list[dict],
             igs_arcs: list[dict],
             save_dir: str) -> None:
    """Globe map: suborbital and IGS TEC-max tangent / IPP points."""
    fig = plt.figure(figsize=(12, 8))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([LON_MIN - 5, LON_MAX + 5, LAT_MIN - 5, LAT_MAX + 5],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle=":")
    ax.add_feature(cfeature.LAND,      facecolor="wheat",     alpha=0.5)
    ax.add_feature(cfeature.OCEAN,     facecolor="lightcyan", alpha=0.5)
    ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                 linewidth=0.4, color="gray", alpha=0.6)

    # Suborbital TEC-max points
    lats_s = [a["lat_tecmax_tangent"] for a in sub_arcs]
    lons_s = [a["lon_tecmax_tangent"] for a in sub_arcs]
    ax.scatter(lons_s, lats_s, c="steelblue", s=45, marker="^",
               transform=ccrs.PlateCarree(), zorder=6, alpha=0.85,
               label=f"Suborbital ({len(sub_arcs)} arcs)")

    # IGS IPP points
    lats_i = [a.get("lat_tecmax_tangent", np.nan) for a in igs_arcs]
    lons_i = [a.get("lon_tecmax_tangent", np.nan) for a in igs_arcs]
    ax.scatter(lons_i, lats_i, c="darkorange", s=20, marker="o",
               transform=ccrs.PlateCarree(), zorder=5, alpha=0.70,
               label=f"IGS ({len(igs_arcs)} arcs)")

    # Station markers
    sta_coords = {"PIE1": (34.30, -108.12),
                  "SGPO": (20.03,  -98.67),
                  "MDO1": (30.68, -104.02)}
    for sta, (lat, lon) in sta_coords.items():
        ax.plot(lon, lat, "rv", ms=10, transform=ccrs.PlateCarree(), zorder=7)
        ax.text(lon + 0.5, lat + 0.5, sta, fontsize=8, color="darkred",
                transform=ccrs.PlateCarree(), zorder=8)

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"Observation geometry — {CAMPAIGN_DATE.strftime('%Y-%m-%d')}  "
        f"~{PROFILE_HOUR:02d}:00 UTC\n"
        f"SOUNDING_ROCKET_01  ×  IGS: {', '.join(IGS_STATIONS)}",
        fontsize=10,
    )
    fpath = os.path.join(save_dir, "obs_geometry_map.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def _tec_panel(clean_list: list[dict],
               prior_tec_slices: list[np.ndarray],
               post_tec_slices:  list[np.ndarray],
               prior_rmse: float,
               post_rmse:  float,
               save_dir: str) -> None:
    """Per-arc sTEC-vs-UTC-time panels (measured / IRI prior / KF posterior).

    x-axis : UTC decimal hours  (time_utc_h from each arc dict)
    y-axis : sTEC (TECU)

    Each sub-panel title shows the source [SUB] or [IGS], label, and
    UTC hour span.  Colours mirror the constellation families.
    """
    n = len(clean_list)
    if n == 0:
        return
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))

    # Track per-constellation arc index so shading is consistent with
    # the PRN-panel figure.
    const_counts: dict[str, int] = defaultdict(int)
    for arc in clean_list:
        const_counts[arc.get("prn_id", "?")[0].upper()] += 1
    const_counter: dict[str, int] = defaultdict(int)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * n_cols, 3.0 * n_rows),
                              squeeze=False)
    fig.suptitle(
        f"sTEC vs UTC time  —  joint KF assimilation\n"
        f"{CAMPAIGN_DATE.strftime('%Y-%m-%d')}  ~{PROFILE_HOUR:02d}:00 UTC  |  "
        f"Prior RMSE = {prior_rmse:.2f} TECU    "
        f"Posterior RMSE = {post_rmse:.2f} TECU",
        fontsize=11,
    )

    for k, (arc, ax) in enumerate(zip(clean_list, axes.flat)):
        src   = arc.get("obs_source", "unknown")
        lbl   = arc.get("label", f"arc {k}")
        prn   = arc.get("prn_id", "?")
        const = prn[0].upper() if prn else "?"
        idx_in = const_counter[const]
        n_in   = const_counts[const]
        col    = _arc_colour(prn, idx_in, n_in)
        const_counter[const] += 1

        meas  = arc["tec"]
        prior = prior_tec_slices[k]
        post  = post_tec_slices[k]
        t_utc = arc.get("time_utc_h", np.arange(len(meas)) / 3600.0)

        ax.plot(t_utc, meas,  ".", color=col,   ms=2.5, alpha=0.65, label="Meas")
        ax.plot(t_utc, prior, "--", color="royalblue", lw=0.9,  label="Prior")
        ax.plot(t_utc, post,  "-",  color="firebrick", lw=1.1,  label="Post")

        ax.set_xlabel("UTC (hours)", fontsize=6)
        ax.set_ylabel("sTEC (TECU)", fontsize=6)
        ax.tick_params(labelsize=6)
        short_lbl = lbl.replace("SOUNDING_ROCKET_01/", "SR01/")
        t0_str    = f"{t_utc[0]:.3f}" if len(t_utc) else ""
        ax.set_title(f"[{src[:3].upper()}] {short_lbl}  @{t0_str}h",
                     fontsize=6.0, color=CONSTELLATION_CONFIG.get(const, {})
                     .get("title_color", "black"))
        if k == 0:
            ax.legend(fontsize=5.5, loc="upper right")

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    fpath = os.path.join(save_dir, "tec_fits_time.png")
    fig.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def _prn_tec_panels(clean_list: list[dict],
                     prior_slices: list[np.ndarray],
                     post_slices:  list[np.ndarray],
                     save_dir: str) -> None:
    """2×2 constellation figure: one panel per GNSS family, one curve per PRN.

    Layout (mirrors _plot_group / _plot_igs_stec_section in demo_group.py):
        (0,0) GPS     — Blues
        (1,0) GLONASS — Purples
        (0,1) Galileo — Oranges
        (1,1) BeiDou  — Greens

    x-axis : UTC decimal hours
    y-axis : sTEC (TECU)
    Lines  : solid = measured, dashed = prior, dotted = posterior
    Source annotations: suborbital arcs labelled "[SR01]", IGS "[STA]".
    """
    fig_tec = plt.figure(figsize=(14, 10))
    fig_tec.suptitle(
        f"sTEC by PRN — {CAMPAIGN_DATE.strftime('%Y-%m-%d')}  ~{PROFILE_HOUR:02d}:00 UTC\n"
        f"Solid = measured  |  Dashed = IRI prior  |  Dotted = KF posterior",
        fontsize=11,
    )
    gs_tec  = GridSpec(2, 2, figure=fig_tec, wspace=0.38, hspace=0.48)

    # Build axes and per-constellation arc counters
    ax_by_const: dict[str, plt.Axes] = {}
    for const, (row, col) in _CONST_POS.items():
        cfg = CONSTELLATION_CONFIG.get(const, {"name": const, "title_color": "black"})
        ax  = fig_tec.add_subplot(gs_tec[row, col])
        ax.set_title(cfg["name"], fontsize=10, color=cfg["title_color"],
                     fontweight="bold")
        ax.set_xlabel("UTC (hours)", fontsize=8)
        ax.set_ylabel("sTEC (TECU)", fontsize=8)
        ax.grid(True, alpha=0.3, ls=":")
        ax_by_const[const] = ax

    # Count arcs per constellation for shade mapping
    const_counts: dict[str, int] = defaultdict(int)
    for arc in clean_list:
        const_counts[arc.get("prn_id", "?")[0].upper()] += 1

    const_counter: dict[str, int] = defaultdict(int)
    legend_handles: dict[str, list] = defaultdict(list)

    for k, arc in enumerate(clean_list):
        prn   = arc.get("prn_id", "?")
        const = prn[0].upper() if prn else "?"
        src   = arc.get("obs_source", "?")
        sta   = arc.get("leo_id", "?")
        idx_in = const_counter[const]
        n_in   = const_counts[const]
        col    = _arc_colour(prn, idx_in, n_in)
        const_counter[const] += 1

        meas   = arc["tec"]
        prior  = prior_slices[k]
        post   = post_slices[k]
        t_utc  = arc.get("time_utc_h", np.arange(len(meas)) / 3600.0)

        src_tag = f"SR01" if src == "suborbital" else sta
        lbl_str = f"{src_tag}/{prn}"

        ax = ax_by_const.get(const) or ax_by_const.get("G") or next(iter(ax_by_const.values()))

        # Three style passes (solid/dashed/dotted) — share colour to link them
        ax.plot(t_utc, meas,  color=col, lw=1.5, ls="-",  alpha=0.85)
        ax.plot(t_utc, prior, color=col, lw=0.9, ls="--", alpha=0.65)
        ax.plot(t_utc, post,  color=col, lw=1.0, ls=":",  alpha=0.80)

        legend_handles[const].append(
            Line2D([0], [0], color=col, lw=1.6, label=lbl_str)
        )

    # Add legends to each panel (per-const arcs + style key on first populated panel)
    _style_legend_added = False
    for const, ax in ax_by_const.items():
        handles = legend_handles.get(const, [])
        if handles:
            if not _style_legend_added:
                style_handles = [
                    Line2D([0], [0], color="gray", lw=1.5, ls="-",  label="Measured"),
                    Line2D([0], [0], color="gray", lw=0.9, ls="--", label="IRI prior"),
                    Line2D([0], [0], color="gray", lw=1.0, ls=":",  label="KF posterior"),
                ]
                ax.legend(handles=handles + style_handles,
                          fontsize=6.5, loc="upper right", framealpha=0.85)
                _style_legend_added = True
            else:
                ax.legend(handles=handles, fontsize=6.5,
                          loc="upper right", framealpha=0.85)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="lightgray",
                    fontsize=12, style="italic")

    fpath = os.path.join(save_dir, "tec_by_prn.png")
    fig_tec.savefig(fpath, dpi=130, bbox_inches="tight")
    plt.close(fig_tec)
    print(f"  Saved: {fpath}")


def _regional_delta_map(eds_region: EDPSamples,
                         prior_flat: np.ndarray,
                         post_flat:  np.ndarray,
                         alt_grid:   np.ndarray,
                         clean_list: list[dict],
                         prior_slices: list[np.ndarray],
                         post_slices:  list[np.ndarray],
                         save_dir: str) -> None:
    """Regional ΔNe coolwarm map at the F2-peak altitude.

    Left panel  — PlateCarree map of the assimilation region:
        • Coolwarm scatter at every mesh node: colour = ΔNe at posterior hmF2
        • Semi-transparent circles at each IPP (TEC-max tangent point) sized
          by |TEC residual| (posterior – measured), coloured by sign.
    Right panel — histogram of TEC residuals (prior and posterior).
    """
    n_height = len(alt_grid)
    n_geo    = eds_region.geolocation.shape[0]
    prior_3d = prior_flat[:n_height * n_geo].reshape(n_height, n_geo)
    post_3d  = post_flat[:n_height  * n_geo].reshape(n_height, n_geo)
    delta_3d = post_3d - prior_3d   # ΔNe in m⁻³

    # Find the mean hmF2 across all mesh nodes (use prior for stability)
    hmF2_km_per_node = np.full(n_geo, np.nan)
    for i_g in range(n_geo):
        _, hm = extract_robust_f2_peak(prior_3d[:, i_g], alt_grid)
        hmF2_km_per_node[i_g] = hm

    mean_hmF2 = float(np.nanmedian(hmF2_km_per_node))
    if np.isnan(mean_hmF2):
        mean_hmF2 = 300.0   # fallback
    alt_idx  = int(np.argmin(np.abs(alt_grid - mean_hmF2)))
    hmF2_lbl = f"{alt_grid[alt_idx]:.0f} km"

    delta_slice = delta_3d[alt_idx, :]   # (n_geo,) m⁻³
    # geolocation: (n_geo, 2) with col0=lon, col1=lat (EDPSamples convention)
    geo_lon = eds_region.geolocation[:, 0]
    geo_lat = eds_region.geolocation[:, 1]

    # Symmetrical colour scale
    vmax = float(np.nanpercentile(np.abs(delta_slice), 97))
    if vmax == 0:
        vmax = 1e4

    fig, (ax_map, ax_hist) = plt.subplots(
        1, 2, figsize=(15, 7),
        gridspec_kw={"width_ratios": [2.5, 1]},
    )
    # ── Map panel ────────────────────────────────────────────────────────────
    # Use a plain 2-D axes with gridlines so the map renders without projection
    # artefacts on lightweight installs.  For full cartopy renders replace
    # subplot with projection=ccrs.PlateCarree() and add features.
    try:
        from matplotlib.axes import Axes as _mpl_Axes
        import cartopy.crs as _ccrs
        import cartopy.feature as _cfeature
        # Re-create the axes as cartopy if available
        fig.delaxes(ax_map)
        ax_map = fig.add_subplot(1, 2, 1, projection=_ccrs.PlateCarree())
        ax_map.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                          crs=_ccrs.PlateCarree())
        ax_map.add_feature(_cfeature.COASTLINE, linewidth=0.8, zorder=2)
        ax_map.add_feature(_cfeature.BORDERS,   linewidth=0.5, ls=":", zorder=2)
        ax_map.add_feature(_cfeature.LAND,      facecolor="whitesmoke", alpha=0.6, zorder=1)
        ax_map.add_feature(_cfeature.OCEAN,     facecolor="aliceblue",  alpha=0.4, zorder=1)
        ax_map.gridlines(draw_labels=True, dms=False,
                         x_inline=False, y_inline=False,
                         linewidth=0.4, color="gray", alpha=0.5, zorder=1)
        _use_cartopy = True
    except Exception:
        # Fallback plain axes
        _use_cartopy = False
        ax_map.set_xlim(LON_MIN, LON_MAX)
        ax_map.set_ylim(LAT_MIN, LAT_MAX)
        ax_map.set_xlabel("Longitude (°E)")
        ax_map.set_ylabel("Latitude (°N)")

    _transform = _ccrs.PlateCarree() if _use_cartopy else None
    _sc_kw = {"transform": _transform} if _use_cartopy else {}

    # Mesh-node ΔNe scatter
    sc_ne = ax_map.scatter(
        geo_lon, geo_lat,
        c=delta_slice * 1e-6,   # convert m⁻³ → ×10⁶ m⁻³ for display
        cmap="coolwarm",
        vmin=-vmax * 1e-6,
        vmax= vmax * 1e-6,
        s=60, alpha=0.80, zorder=5,
        **_sc_kw,
    )
    cb_ne = fig.colorbar(sc_ne, ax=ax_map, orientation="vertical",
                         fraction=0.025, pad=0.04)
    cb_ne.set_label(f"ΔNe at ~{hmF2_lbl}  (×10⁶ m⁻³)", fontsize=9)

    # TEC residuals at TEC-max IPP locations
    const_counts: dict[str, int] = defaultdict(int)
    for arc in clean_list:
        const_counts[arc.get("prn_id", "?")[0].upper()] += 1
    const_counter: dict[str, int] = defaultdict(int)

    for k, arc in enumerate(clean_list):
        prn   = arc.get("prn_id", "?")
        const = prn[0].upper() if prn else "?"
        lat_ipp = arc.get("lat_tecmax_tangent", np.nan)
        lon_ipp = arc.get("lon_tecmax_tangent", np.nan)
        if not (np.isfinite(lat_ipp) and np.isfinite(lon_ipp)):
            continue

        idx_in = const_counter[const]
        n_in   = const_counts[const]
        col    = _arc_colour(prn, idx_in, n_in)
        const_counter[const] += 1

        # Mean TEC residual (posterior – measured)
        resid  = float(np.nanmean(post_slices[k] - arc["tec"]))
        marker = "^" if arc.get("obs_source") == "suborbital" else "o"
        ms     = max(40.0, min(200.0, abs(resid) * 15.0))
        edge   = "firebrick" if resid > 0 else "royalblue"

        ax_map.scatter(
            lon_ipp, lat_ipp,
            s=ms, marker=marker,
            facecolors=col, edgecolors=edge, linewidths=1.2,
            alpha=0.85, zorder=8,
            **_sc_kw,
        )

    ax_map.set_title(
        f"ΔNe (posterior − prior) at hmF2 ≈ {hmF2_lbl}\n"
        f"Markers: ▲ suborbital  ● IGS   (size ∝ |TEC residual|, "
        f"edge colour: red=+TEC, blue=−TEC)",
        fontsize=9,
    )

    # ── Histogram panel ───────────────────────────────────────────────────────
    meas_all  = np.concatenate([a["tec"] for a in clean_list])
    prior_all = np.concatenate(prior_slices)
    post_all  = np.concatenate(post_slices)

    bins = np.linspace(-20, 20, 61)
    ax_hist.hist(prior_all - meas_all, bins=bins,
                 color="royalblue", alpha=0.60, label="Prior residual")
    ax_hist.hist(post_all  - meas_all, bins=bins,
                 color="firebrick", alpha=0.60, label="Post residual")
    ax_hist.axvline(0, color="black", lw=0.8, ls="--")
    ax_hist.set_xlabel("TEC residual (TECU)", fontsize=9)
    ax_hist.set_ylabel("Count",               fontsize=9)
    ax_hist.set_title("TEC fit residuals",    fontsize=10)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    fig.suptitle(
        f"Regional ionosphere update — {CAMPAIGN_DATE.strftime('%Y-%m-%d')} "
        f"~{PROFILE_HOUR:02d}:00 UTC\n"
        f"joint suborbital + IGS KF assimilation  |  "
        f"{n_geo} mesh nodes  |  "
        f"lat [{LAT_MIN:.0f}°, {LAT_MAX:.0f}°]  "
        f"lon [{LON_MIN:.0f}°, {LON_MAX:.0f}°]",
        fontsize=10,
    )
    plt.tight_layout()
    fpath = os.path.join(save_dir, "regional_delta_ne.png")
    fig.savefig(fpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def _edp_profiles(eds_region: EDPSamples,
                  prior_flat: np.ndarray,
                  post_flat:  np.ndarray,
                  alt_grid:   np.ndarray,
                  save_dir:   str) -> None:
    """Prior vs posterior Ne profiles at all mesh nodes, plus the mean delta."""
    n_height = len(alt_grid)
    n_geo    = eds_region.geolocation.shape[0]
    prior_3d = prior_flat.reshape(n_height, n_geo)   # m^-3
    post_3d  = post_flat.reshape(n_height, n_geo)    # m^-3
    diff_3d  = post_3d - prior_3d

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    fig.suptitle(
        "IRI Prior vs KF Posterior EDP — suborbital + IGS joint assimilation\n"
        f"{CAMPAIGN_DATE.strftime('%Y-%m-%d')}  ~{PROFILE_HOUR:02d}:00 UTC  |  "
        f"{n_geo} spatial nodes",
        fontsize=10,
    )

    cmap = cm.plasma
    for i_g in range(n_geo):
        axes[0].plot(prior_3d[:, i_g] * 1e-6, alt_grid, color="lightblue", lw=0.4, alpha=0.5)
        axes[1].plot(post_3d[:,  i_g] * 1e-6, alt_grid, color="lightcoral", lw=0.4, alpha=0.5)

    axes[0].plot(prior_3d.mean(axis=1) * 1e-6, alt_grid, "b-",  lw=2.0, label="mean")
    axes[1].plot(post_3d.mean(axis=1)  * 1e-6, alt_grid, "r-",  lw=2.0, label="mean")
    axes[2].plot(diff_3d.mean(axis=1)  * 1e-6, alt_grid, "k-",  lw=2.0, label="mean Δ")
    axes[2].axvline(0, color="gray", lw=0.8, ls="--")

    titles = ["IRI Prior  Ne\n(×10⁶ m⁻³)",
              "KF Posterior  Ne\n(×10⁶ m⁻³)",
              "Posterior − Prior\n(×10⁶ m⁻³)"]
    for ax, title in zip(axes, titles):
        ax.set_xlabel("Ne (10⁶ m⁻³)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(alt_grid[0], alt_grid[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Altitude (km)", fontsize=10)
    plt.tight_layout()
    fpath = os.path.join(save_dir, "edp_prior_vs_posterior.png")
    fig.savefig(fpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def _obs_count_bar(clean_list: list[dict], save_dir: str) -> None:
    """Bar chart showing number of valid rays per arc, colour-coded by source."""
    labels  = [a.get("label", f"arc {k}").replace("SOUNDING_ROCKET_01/", "SR01/")
               for k, a in enumerate(clean_list)]
    counts  = [len(a["tec"]) for a in clean_list]
    colours = ["steelblue" if a.get("obs_source") == "suborbital" else "darkorange"
               for a in clean_list]

    fig, ax = plt.subplots(figsize=(max(8, len(clean_list) * 0.4), 5))
    ax.bar(range(len(clean_list)), counts, color=colours, edgecolor="white", lw=0.5)
    ax.set_xticks(range(len(clean_list)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Number of rays", fontsize=10)
    ax.set_title("Rays per arc after decimation", fontsize=11)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="steelblue",  label="Suborbital"),
                        Patch(facecolor="darkorange", label="IGS ground")],
              fontsize=9)
    plt.tight_layout()
    fpath = os.path.join(save_dir, "ray_counts.png")
    fig.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    os.makedirs(SAVE_DIR,       exist_ok=True)
    os.makedirs(RINEX_CACHE,    exist_ok=True)
    os.makedirs(GLOBAL_EDP_DIR, exist_ok=True)

    print("=" * 68)
    print("  demo_suborbital_igs.py — Suborbital + IGS Joint KF Assimilation")
    print(f"  Date     : {CAMPAIGN_DATE.strftime('%Y-%m-%d')}  "
          f"~{PROFILE_HOUR:02d}:00 UTC")
    print(f"  Stations : {', '.join(IGS_STATIONS)}")
    print(f"  Region   : lat [{LAT_MIN:.0f}, {LAT_MAX:.0f}]  "
          f"lon [{LON_MIN:.0f}, {LON_MAX:.0f}]")
    print(f"  Alt grid : {ALT_GRID[0]:.0f}–{ALT_GRID[-1]:.0f} km "
          f"({len(ALT_GRID)} levels)")
    print("=" * 68)

    # ── §1  Suborbital data ──────────────────────────────────────────────────
    print("\n──── §1  Suborbital STEC arcs ────────────────────────────────────")
    sub_arcs = load_suborbital_arcs(SUBORBITAL_CSV)

    # ── §2  IGS data — cropped to the suborbital time window ─────────────────
    print("\n──── §2  IGS ground-station TEC ──────────────────────────────────")
    # Derive the UTC hour span from the suborbital arcs so IGS observations
    # are restricted to the same ~15-minute window as the campaign event.
    if sub_arcs:
        sub_t_min = float(min(a["time_utc_h"][ 0] for a in sub_arcs
                              if len(a["time_utc_h"]) > 0))
        sub_t_max = float(max(a["time_utc_h"][-1] for a in sub_arcs
                              if len(a["time_utc_h"]) > 0))
        print(f"  Suborbital window : {sub_t_min:.4f} – {sub_t_max:.4f} UTC h  "
              f"({(sub_t_max - sub_t_min) * 60:.1f} min)")
        sub_window_h: "tuple[float, float] | None" = (sub_t_min, sub_t_max)
    else:
        sub_window_h = None

    igs_arcs = load_igs_arcs(IGS_STATIONS, CAMPAIGN_DATE,
                              time_window_h=sub_window_h)

    # ── §3  IRI prior ─────────────────────────────────────────────────────────
    print("\n──── §3  IRI-2020 prior ──────────────────────────────────────────")
    eds_region = build_iri_prior(
        CAMPAIGN_DATE, PROFILE_HOUR, ALT_GRID,
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    )

    # ── §4  Assemble observation pool ─────────────────────────────────────────
    print("\n──── §4  Observation pool ────────────────────────────────────────")
    clean_list: list[dict] = sub_arcs + igs_arcs
    print(f"  Suborbital arcs : {len(sub_arcs)}")
    print(f"  IGS arcs        : {len(igs_arcs)}")
    print(f"  Total arcs      : {len(clean_list)}")
    total_rays = sum(len(a["tec"]) for a in clean_list)
    print(f"  Total rays      : {total_rays:,}")

    if len(clean_list) < 1:
        print("ERROR: no valid observations — aborting.")
        return

    # Observation map — before KF
    print("\n──── §5  Observation geometry plot ───────────────────────────────")
    _obs_map(sub_arcs, igs_arcs, SAVE_DIR)
    _obs_count_bar(clean_list, SAVE_DIR)

    # ── §6  Initialise KF inverter ─────────────────────────────────────────
    print("\n──── §6  Initialising Ionosphere_Tomography_Inverter ─────────────")
    inverter = Ionosphere_Tomography_Inverter(
        EDPSam                   = eds_region,
        meanscale                = 1,
        topside_prior_floor_tecu = 1.0,
        n_rel_arcs               = 0,          # all observations are absolute TEC
        topside_alpha            = 0.0,
        gaussian_cov_sigma       = GAUSSIAN_COV_SIGMA,
        altitude_taper_km        = ALTITUDE_TAPER_KM,
        altitude_taper_min_scale = ALTITUDE_TAPER_SCALE,
        topside_follow_f2        = TOPSIDE_FOLLOW_F2,
    )
    n_sv     = inverter.attrs["n_state_vars"]
    n_sv_aug = inverter.attrs["n_state_vars_aug"]
    n_geo    = inverter.attrs["n_geo"]
    print(f"  State vector: {n_sv} grid + {n_geo} topside = {n_sv_aug} total components")

    # ── §7  Build H matrices ────────────────────────────────────────────────
    print("\n──── §7  Building observation operators ──────────────────────────")
    H_blocks = inverter.get_observation_operator_batch(
        clean_list, num_segments=NUM_RAY_SEGMENTS
    )
    H_joint   = np.vstack(H_blocks).astype(np.float32)
    obs_joint = np.concatenate([cl["tec"] for cl in clean_list]).astype(np.float64)
    ray_counts = [len(cl["tec"]) for cl in clean_list]
    print(f"  H_joint shape: {H_joint.shape}   "
          f"(total rays = {len(obs_joint):,})")

    # ── §8  Prior TEC ───────────────────────────────────────────────────────
    print("\n──── §8  Computing IRI prior TEC ─────────────────────────────────")
    prior_mean  = inverter.attrs["initial_edps_mean"]   # (n_sv, 1) m^-3
    x_top_prior = inverter.attrs["x_top_prior"]          # (n_geo,) TECU
    prior_tec   = (
        H_joint[:, :n_sv]       @ prior_mean
        + H_joint[:, n_sv:n_sv_aug] @ x_top_prior[:, None]
    ).flatten()

    # ── §9  Joint KF assimilation ────────────────────────────────────────────
    print("\n──── §9  Joint Kalman Filter assimilation ────────────────────────")
    t_kf = time.time()
    posterior_flat = inverter.assimilate(
        obs             = obs_joint,
        obs_operator    = H_joint.astype(np.float32),
        relaxation      = RELAXATION,
        measurement_err = MEASUREMENT_ERR,
    )
    print(f"  KF update completed in {time.time() - t_kf:.1f} s")

    # ── §10  Posterior TEC and statistics ────────────────────────────────────
    print("\n──── §10  Posterior TEC statistics ───────────────────────────────")
    post_tec = (
        H_joint[:, :n_sv]       @ np.asarray(posterior_flat)
        + H_joint[:, n_sv:n_sv_aug] @ inverter.x_top_tecu
    ).flatten()

    prior_rmse = float(np.sqrt(np.mean((obs_joint - prior_tec) ** 2)))
    post_rmse  = float(np.sqrt(np.mean((obs_joint - post_tec)  ** 2)))
    reduction  = 100.0 * (prior_rmse - post_rmse) / prior_rmse if prior_rmse > 0 else 0.0
    print(f"  Prior RMSE     : {prior_rmse:.3f} TECU")
    print(f"  Posterior RMSE : {post_rmse:.3f} TECU")
    print(f"  RMSE reduction : {reduction:.1f} %")

    # Per-source breakdown
    for tag, src_label in [("suborbital", "Suborbital"),
                            ("IGS_ground", "IGS ground")]:
        mask = np.concatenate([
            np.ones(ray_counts[k], dtype=bool)
            if clean_list[k].get("obs_source") == tag
            else np.zeros(ray_counts[k], dtype=bool)
            for k in range(len(clean_list))
        ])
        if mask.any():
            r_prior = float(np.sqrt(np.mean((obs_joint[mask] - prior_tec[mask]) ** 2)))
            r_post  = float(np.sqrt(np.mean((obs_joint[mask] - post_tec[mask])  ** 2)))
            print(f"  [{src_label}]  prior {r_prior:.3f} TECU  →  "
                  f"post {r_post:.3f} TECU")

    # ── §11  Diagnostic figures ──────────────────────────────────────────────
    print("\n──── §11  Generating diagnostic figures ──────────────────────────")

    # Slice prior/posterior TEC back into per-arc arrays
    prior_slices: list[np.ndarray] = []
    post_slices:  list[np.ndarray] = []
    ptr = 0
    for n_k in ray_counts:
        prior_slices.append(prior_tec[ptr : ptr + n_k])
        post_slices.append(post_tec[ptr : ptr + n_k])
        ptr += n_k

    # sTEC-vs-UTC-time per-arc panels
    _tec_panel(clean_list, prior_slices, post_slices,
               prior_rmse, post_rmse, SAVE_DIR)

    # Per-PRN 2×2 constellation figure
    _prn_tec_panels(clean_list, prior_slices, post_slices, SAVE_DIR)

    # Regional ΔNe map at F2-peak altitude + TEC residual IPP scatter
    _regional_delta_map(
        eds_region,
        prior_mean.flatten(),
        np.asarray(posterior_flat).flatten(),
        ALT_GRID,
        clean_list,
        prior_slices,
        post_slices,
        SAVE_DIR,
    )

    # EDP spaghetti
    _edp_profiles(eds_region,
                  prior_mean.flatten(),
                  np.asarray(posterior_flat).flatten(),
                  ALT_GRID, SAVE_DIR)

    gc.collect()
    elapsed = time.time() - t0
    print(f"\n{'=' * 68}")
    print(f"  Completed in {elapsed:.1f} s  —  figures written to {SAVE_DIR}/")
    print("=" * 68)


if __name__ == "__main__":
    main()
