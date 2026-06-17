#!/usr/bin/env python3
"""
demo_ground_station.py
======================
Ground-station-only ionospheric TEC data acquisition using IGS RINEX data.

Pipeline (stops before assimilation)
-------------------------------------
1.  Load IGSNetwork.json and find all stations within SEARCH_RADIUS_DEG of
    the point of interest (POI_LAT, POI_LON).
2.  Download the DCB SINEX file for the campaign date; parse Tx and Rx DCBs.
3.  Filter to stations that have Rx DCB entries in the SINEX file.
4.  For each qualifying station download RINEX obs + nav files and run the
    carrier-phase levelling + DCB correction pipeline.
5.  Plot:
      a) 2×2 constellation grid (GPS / GAL / GLO / BDS) — DCB-corrected sTEC
         vs UTC time, with 90-min windows shaded; raw pre-DCB TEC overlaid
         as a faint dashed line so the correction magnitude is visible.
      b) 2×2 DCB summary bar chart per constellation.
      c) Regional map of station locations and IPP ground-tracks.
6.  Stop here — assimilation is a subsequent step.

NASA Earthdata credentials in ~/.netrc are required for CDDIS downloads:
    machine urs.earthdata.nasa.gov login <user> password <pass>

Usage
-----
    python demo_ground_station.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from TEC_model.igs_tec_pipeline import (
    RinexDownloader,
    DCBCorrector,
    IGSTECPipeline,
    igs_obs_to_clean_entry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CAMPAIGN_DATE     = datetime(2026, 6, 2)
POI_LAT           = 50.0          # °N — centre of search region
POI_LON           = 10.0          # °E
SEARCH_RADIUS_DEG = 1.0          # great-circle search radius
WINDOW_MIN        = 90            # minutes per display window

RINEX_VERSION  = 3
RINEX_CACHE    = str(ROOT / "Data" / "RINEX_Cache")
IGS_JSON       = str(ROOT / "Data" / "IGS_Stations" / "IGSNetwork.json")
SAVE_DIR       = str(ROOT / "Figures" / "Demo_Ground_Station")

MAX_RAYS_PER_ARC = 100
MIN_VALID_RAYS   = 20
NUM_SV_WORKERS   = 10    # parallel SV workers per station (1 = serial)
VERBOSE          = True # verbose per-SV timing output
EPHEM_STRIDE     = 0    # 0 = auto-detect from sample rate (~150 s target)

# Constellation display config — order matches 2×2 panel layout
CONSTELLATION_CONFIG: dict[str, dict] = {
    "G": {"name": "GPS",     "color": "steelblue",    "panel": (0, 0)},
    "E": {"name": "Galileo", "color": "darkorange",   "panel": (0, 1)},
    "R": {"name": "GLONASS", "color": "mediumpurple", "panel": (1, 0)},
    "C": {"name": "BeiDou",  "color": "seagreen",     "panel": (1, 1)},
}


# ─────────────────────────────────────────────────────────────────────────────
# §1  Find nearby IGS stations
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in degrees."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return math.degrees(2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def find_nearby_stations(poi_lat: float, poi_lon: float,
                          radius_deg: float,
                          json_path: str) -> list[dict]:
    """Return station info dicts sorted by distance to the POI.

    Each dict has: name (9-char), code (4-char), lat, lon, dist_deg, info.
    """
    with open(json_path) as fh:
        network: dict = json.load(fh)

    nearby: list[dict] = []
    for name, info in network.items():
        try:
            lat = float(info["Latitude"])
            lon = float(info["Longitude"])
        except (KeyError, ValueError):
            continue
        dist = _haversine_deg(poi_lat, poi_lon, lat, lon)
        if dist <= radius_deg:
            nearby.append({
                "name":     name,       # e.g. "FFMJ00DEU"
                "code":     name[:4],   # 4-char pipeline code
                "lat":      lat,
                "lon":      lon,
                "dist_deg": dist,
                "info":     info,
            })
    nearby.sort(key=lambda s: s["dist_deg"])
    return nearby


# ─────────────────────────────────────────────────────────────────────────────
# §2  Download SINEX and check for Rx DCBs
# ─────────────────────────────────────────────────────────────────────────────

def download_sinex(date: datetime,
                   cache_dir: str) -> tuple[DCBCorrector | None, Path | None]:
    """Download the DCB SINEX for *date* and return (DCBCorrector, path)."""
    dl = RinexDownloader(cache_dir=cache_dir)
    dcb_path = dl.dcb_sinex(date)
    if dcb_path is None:
        print("  [warn] No SINEX file found — DCB corrections will be zero.")
        return None, None
    dcb = DCBCorrector(dcb_path)
    n_sv  = len(dcb._sv_dcb)
    n_sta = len(dcb._sta_dcb)
    print(f"  Loaded : {dcb_path.name}")
    print(f"    Satellite Tx DCBs : {n_sv} SVs")
    print(f"    Receiver  Rx DCBs : {n_sta} stations")
    return dcb, dcb_path


def filter_stations_with_rx_dcb(stations: list[dict],
                                  dcb: DCBCorrector | None) -> list[dict]:
    """Keep only stations whose 4-char code (or 9-char name) is in the SINEX.

    If *dcb* is None (no SINEX available) all stations pass through.
    """
    if dcb is None:
        print("  No SINEX — accepting all nearby stations.")
        return stations

    sta_keys = {k.upper() for k in dcb._sta_dcb.keys()}
    accepted, skipped = [], []
    for sta in stations:
        code4 = sta["code"].upper()
        code9 = sta["name"].upper()
        if code4 in sta_keys or code9 in sta_keys:
            accepted.append(sta)
        else:
            skipped.append(sta["name"])

    if skipped:
        print(f"  Skipped (no Rx DCB): {', '.join(skipped)}")
    return accepted


# ─────────────────────────────────────────────────────────────────────────────
# §3  Process RINEX for each qualified station
# ─────────────────────────────────────────────────────────────────────────────

# ── §3a  Pre-download helper ─────────────────────────────────────────────────

def predownload_station_files(stations: list[dict],
                               date: datetime,
                               cache_dir: str) -> dict[str, tuple]:
    """Download obs + nav RINEX for every station into the cache (sequential).

    Returns a dict mapping 4-char station code → (obs_path, nav_path).
    Stations that fail to download are excluded from the returned dict.
    Downloads are sequential so CDDIS never sees concurrent requests from the
    same IP; each file is skipped on cache hit.
    """
    dl = RinexDownloader(cache_dir=cache_dir)
    paths: dict[str, tuple] = {}
    n = len(stations)
    for i, sta in enumerate(stations, 1):
        code = sta["code"]
        print(f"  [{i:2d}/{n}] {code}  ", end="", flush=True)
        try:
            obs = dl.obs_file(code, date, RINEX_VERSION)
            nav = dl.nav_file(code, date, RINEX_VERSION)
            paths[code] = (obs, nav)
            print("OK", flush=True)
        except Exception as exc:
            print(f"SKIP — {exc}", flush=True)
    return paths


# ── §3b  Orchestrator (serial stations, parallel SVs inside each) ────────────

def process_all_stations(stations:      list[dict],
                          date:          datetime,
                          dcb_path:      "Path | None",
                          cache_dir:     str,
                          file_paths:    "dict[str, tuple] | None" = None,
                          ephem_stride:  int = EPHEM_STRIDE,
                          num_sv_workers: int = NUM_SV_WORKERS,
                          verbose:       bool = VERBOSE,
                          ) -> tuple[list[dict], list[dict]]:
    """Run the IGS TEC pipeline for each station sequentially.

    Within each station, per-SV TEC computations run in parallel across
    *num_sv_workers* processes.

    Returns
    -------
    obs_all, clean_all : paired lists (element k corresponds to clean_all[k]).
    """
    obs_all:   list[dict] = []
    clean_all: list[dict] = []
    n = len(stations)

    for i, sta in enumerate(stations, 1):
        code = sta["code"]
        t0   = time.time()
        print(f"\n[{i}/{n}] Processing {code} …", flush=True)

        obs_p = nav_p = None
        if file_paths and code in file_paths:
            obs_p, nav_p = file_paths[code]

        try:
            pipe = IGSTECPipeline(
                station         = code,
                date            = date,
                rinex_version   = RINEX_VERSION,
                cache_dir       = cache_dir,
                use_iri         = False,
                local_obs       = str(obs_p) if obs_p else None,
                local_nav       = str(nav_p) if nav_p else None,
                local_dcb       = str(dcb_path) if dcb_path else None,
                ephem_stride    = ephem_stride,
                show_progress   = not verbose,
                verbose         = verbose,
                num_sv_workers  = num_sv_workers,
            )
            arcs = pipe.run()

            n_arcs = len(arcs)
            n_acc  = 0
            for obs in arcs:
                entry = igs_obs_to_clean_entry(obs, max_rays=MAX_RAYS_PER_ARC,
                                               min_valid=MIN_VALID_RAYS)
                if entry is not None:
                    obs_all.append(obs)
                    clean_all.append(entry)
                    n_acc += 1

            n_rays = sum(len(e["tec"]) for e in clean_all[-n_acc:]) if n_acc else 0
            print(f"  [{code}] {n_acc}/{n_arcs} arcs accepted  "
                  f"{n_rays:,} rays  ({time.time()-t0:.1f}s)", flush=True)

        except Exception as exc:
            print(f"  [{code}] ERROR: {exc}", flush=True)

    return obs_all, clean_all


# ─────────────────────────────────────────────────────────────────────────────
# §4  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _window_edges(duration_min: int = 90) -> list[tuple[float, float]]:
    """Return list of (t_start, t_end) in UTC decimal hours for a full day."""
    step = duration_min / 60.0
    edges = []
    t = 0.0
    while t < 24.0:
        edges.append((t, min(t + step, 24.0)))
        t += step
    return edges


def _station_colour_map(clean_all: list[dict]) -> dict[str, tuple]:
    sta_names = sorted({e.get("leo_id", "?") for e in clean_all})
    cmap = cm.get_cmap("tab20", max(len(sta_names), 1))
    return {s: cmap(i) for i, s in enumerate(sta_names)}


# ─────────────────────────────────────────────────────────────────────────────

def plot_tec_constellations(obs_all: list[dict],
                             clean_all: list[dict],
                             save_dir: str) -> None:
    """2×2 constellation panel: DCB-corrected sTEC (solid) vs raw TEC (dashed).

    Alternating 90-minute windows are shaded grey.  One colour per station.
    """
    fig = plt.figure(figsize=(15, 10))
    gs  = GridSpec(2, 2, figure=fig, wspace=0.33, hspace=0.48)
    fig.suptitle(
        f"IGS Ground-Station sTEC — {CAMPAIGN_DATE.strftime('%Y-%m-%d')}\n"
        f"POI: {POI_LAT:.0f}°N, {POI_LON:.0f}°E  |  search radius {SEARCH_RADIUS_DEG:.0f}°  "
        f"|  solid = DCB-corrected, dashed = pre-DCB",
        fontsize=11,
    )

    axes: dict[str, plt.Axes] = {}
    for conid, cfg in CONSTELLATION_CONFIG.items():
        r, c = cfg["panel"]
        ax   = fig.add_subplot(gs[r, c])
        ax.set_title(cfg["name"], fontsize=11, color=cfg["color"], fontweight="bold")
        ax.set_xlabel("UTC (hours)", fontsize=9)
        ax.set_ylabel("sTEC (TECU)", fontsize=9)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.grid(True, alpha=0.25, ls=":")

        # Alternating 90-min shaded bands
        for i, (t0, t1) in enumerate(_window_edges(WINDOW_MIN)):
            if i % 2 == 1:
                ax.axvspan(t0, t1, color="lightgray", alpha=0.35, zorder=0)

        axes[conid] = ax

    sta_colour = _station_colour_map(clean_all)

    for obs, entry in zip(obs_all, clean_all):
        conid = obs.get("conid", "?")
        if conid not in axes:
            continue
        ax     = axes[conid]
        sta    = entry.get("leo_id", "?")
        col    = sta_colour.get(sta, "gray")
        t_utc  = entry.get("time_utc_h", np.array([]))
        tec    = entry.get("tec",         np.array([]))
        if len(t_utc) < 2:
            continue

        # Raw pre-DCB TEC (add back the DCB correction)
        dcb_sv  = float(obs.get("dcb_sv_tecu", 0.0) or 0.0)
        dcb_rx  = float(obs.get("dcb_rx_tecu", 0.0) or 0.0)
        tec_raw = tec + dcb_sv + dcb_rx

        ax.plot(t_utc, tec_raw, color=col, lw=0.5, alpha=0.40, ls="--", zorder=2)
        ax.plot(t_utc, tec,     color=col, lw=0.8, alpha=0.80, ls="-",  zorder=3)

    # Per-station legend on the GPS panel
    legend_handles = [
        Line2D([0], [0], color=col, lw=1.8, label=sta)
        for sta, col in sta_colour.items()
    ]
    style_handles = [
        Line2D([0], [0], color="gray", lw=1.2, ls="-",  label="DCB-corrected"),
        Line2D([0], [0], color="gray", lw=0.8, ls="--", label="Pre-DCB"),
    ]
    ax_gps = axes.get("G")
    if ax_gps is not None and legend_handles:
        ax_gps.legend(
            handles=legend_handles + style_handles,
            fontsize=5.5, loc="upper right",
            framealpha=0.85, ncol=max(1, len(legend_handles) // 8),
        )

    fpath = os.path.join(save_dir, "tec_constellations.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────

def plot_dcb_summary(obs_all: list[dict], save_dir: str) -> None:
    """2×2 bar chart of Tx and Rx DCB values, one bar group per arc.

    X-axis labels show PRN@Station.  Bars are grouped by Tx (satellite) and
    Rx (receiver) DCB so the relative contribution is immediately visible.
    """
    # Collect per-constellation arc lists
    by_const: dict[str, list[dict]] = {k: [] for k in CONSTELLATION_CONFIG}
    for obs in obs_all:
        conid = obs.get("conid", "?")
        if conid in by_const:
            by_const[conid].append(obs)

    fig, ax_arr = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(
        f"Tx (satellite) and Rx (receiver) DCB Corrections — "
        f"{CAMPAIGN_DATE.strftime('%Y-%m-%d')}\n"
        f"(applied to carrier-levelled sTEC to obtain absolute TEC)",
        fontsize=11,
    )

    for ax, (conid, cfg) in zip(ax_arr.flatten(), CONSTELLATION_CONFIG.items()):
        ax.set_title(cfg["name"], fontsize=10, color=cfg["color"], fontweight="bold")
        ax.set_ylabel("DCB (TECU)", fontsize=9)
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.grid(True, alpha=0.25, axis="y")

        arcs = by_const[conid]
        if not arcs:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="lightgray",
                    style="italic")
            continue

        # Aggregate: one entry per unique (PRN, station) pair — mean DCB
        from collections import defaultdict
        agg: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for obs in arcs:
            prn  = obs.get("conid", "?") + str(obs.get("prn_id", "?"))
            sta  = obs.get("station_id", "?")
            key  = f"{prn}\n@{sta}"
            agg[key].append((
                float(obs.get("dcb_sv_tecu", 0.0) or 0.0),
                float(obs.get("dcb_rx_tecu", 0.0) or 0.0),
            ))

        labels   = list(agg.keys())
        dcb_tx   = [float(np.mean([v[0] for v in agg[k]])) for k in labels]
        dcb_rx   = [float(np.mean([v[1] for v in agg[k]])) for k in labels]
        x        = np.arange(len(labels))
        width    = 0.38

        ax.bar(x - width / 2, dcb_tx, width, label="Tx DCB (satellite)",
               color=cfg["color"], alpha=0.80)
        ax.bar(x + width / 2, dcb_rx, width, label="Rx DCB (receiver)",
               color="dimgray",    alpha=0.65)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fpath = os.path.join(save_dir, "dcb_summary.png")
    fig.savefig(fpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────

def plot_station_map(stations_used: list[dict],
                      clean_all: list[dict],
                      save_dir: str) -> None:
    """Regional map: station markers + IPP ground-tracks coloured by sTEC."""
    pad     = SEARCH_RADIUS_DEG + 4
    lon_min = POI_LON - pad
    lon_max = POI_LON + pad
    lat_min = POI_LAT - pad
    lat_max = POI_LAT + pad

    fig = plt.figure(figsize=(11, 9))

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE,  linewidth=0.8)
        ax.add_feature(cfeature.BORDERS,    linewidth=0.5, linestyle=":")
        ax.add_feature(cfeature.LAND,       facecolor="wheat",     alpha=0.50)
        ax.add_feature(cfeature.OCEAN,      facecolor="lightcyan", alpha=0.50)
        ax.gridlines(draw_labels=True, dms=False,
                     x_inline=False, y_inline=False,
                     linewidth=0.4, color="gray", alpha=0.55)
        _tr = ccrs.PlateCarree()
    except Exception:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.grid(True, alpha=0.3)
        _tr = None

    _kw = {"transform": _tr} if _tr is not None else {}

    # Search-radius circle (approximate) around POI
    theta = np.linspace(0, 2 * np.pi, 360)
    deg_per_km = 1.0 / 111.0
    r_lat = SEARCH_RADIUS_DEG
    r_lon = SEARCH_RADIUS_DEG / max(math.cos(math.radians(POI_LAT)), 0.01)
    circle_lat = POI_LAT + r_lat * np.sin(theta)
    circle_lon = POI_LON + r_lon * np.cos(theta)
    ax.plot(circle_lon, circle_lat, "k--", lw=0.8, alpha=0.4, zorder=4, **_kw)

    # POI marker
    ax.plot(POI_LON, POI_LAT, "r*", ms=14, zorder=11,
            label=f"POI ({POI_LAT:.0f}°N, {POI_LON:.0f}°E)", **_kw)

    # Station markers — colour per station
    sta_colour = _station_colour_map(clean_all)
    for sta in stations_used:
        code = sta["code"]
        col  = sta_colour.get(code, "gray")
        ax.plot(sta["lon"], sta["lat"], "^",
                color=col, ms=10, zorder=10,
                markeredgecolor="black", markeredgewidth=0.6, **_kw)
        if _tr is not None:
            ax.text(sta["lon"] + 0.25, sta["lat"] + 0.25, code,
                    fontsize=6.5, color="black", zorder=12,
                    transform=_tr, clip_on=True)
        else:
            ax.text(sta["lon"] + 0.25, sta["lat"] + 0.25, code,
                    fontsize=6.5, color="black", zorder=12)

    # IPP ground-tracks coloured by TEC
    sc = None
    if clean_all:
        tec_vals = np.concatenate([e["tec"] for e in clean_all if len(e["tec"]) > 0])
        vmin = float(np.nanpercentile(tec_vals, 2))
        vmax = float(np.nanpercentile(tec_vals, 98))
        for entry in clean_all:
            ipp_lat = entry.get("ipp_lat", np.array([]))
            ipp_lon = entry.get("ipp_lon", np.array([]))
            tec     = entry.get("tec",     np.array([]))
            if len(ipp_lat) < 2:
                continue
            sc = ax.scatter(ipp_lon, ipp_lat, c=tec,
                            cmap="plasma", vmin=vmin, vmax=vmax,
                            s=2, alpha=0.55, zorder=5, **_kw)
        if sc is not None:
            cb = fig.colorbar(sc, ax=ax, orientation="vertical",
                              fraction=0.03, pad=0.04)
            cb.set_label("sTEC (TECU)", fontsize=9)

    # Legend
    sta_handles = [
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor=sta_colour.get(s["code"], "gray"),
               markeredgecolor="black", ms=8, label=s["code"])
        for s in stations_used
    ]
    if sta_handles:
        ax.legend(handles=sta_handles, fontsize=6.5, loc="lower right",
                  framealpha=0.85, ncol=max(1, len(sta_handles) // 10))

    ax.set_title(
        f"Station locations & IPP ground-tracks\n"
        f"{CAMPAIGN_DATE.strftime('%Y-%m-%d')}  |  "
        f"{len(stations_used)} stations  |  "
        f"IPP altitude = 350 km",
        fontsize=10,
    )

    fpath = os.path.join(save_dir, "station_map.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    os.makedirs(SAVE_DIR,    exist_ok=True)
    os.makedirs(RINEX_CACHE, exist_ok=True)

    print("=" * 68)
    print("  demo_ground_station.py — IGS Ground-Station TEC Acquisition")
    print(f"  Date     : {CAMPAIGN_DATE.strftime('%Y-%m-%d')}")
    print(f"  POI      : {POI_LAT:.1f}°N, {POI_LON:.1f}°E")
    print(f"  Radius   : {SEARCH_RADIUS_DEG:.0f}°")
    print(f"  Windows  : {WINDOW_MIN}-minute display intervals")
    print("=" * 68)

    # ── §1  Nearby IGS stations ───────────────────────────────────────────────
    print(f"\n──── §1  Nearby IGS stations (radius = {SEARCH_RADIUS_DEG:.0f}°) ─────────────")
    all_nearby = find_nearby_stations(
        POI_LAT, POI_LON, SEARCH_RADIUS_DEG, IGS_JSON
    )
    if not all_nearby:
        print("  ERROR: no stations found — check IGS_JSON path or radius.")
        return
    print(f"  Found {len(all_nearby)} stations:")
    for sta in all_nearby:
        print(f"    {sta['name']}  ({sta['lat']:.2f}°N, {sta['lon']:.2f}°E)  "
              f"dist = {sta['dist_deg']:.2f}°")

    # ── §2  Download SINEX ────────────────────────────────────────────────────
    print(f"\n──── §2  Downloading DCB SINEX ─────────────────────────────────────")
    dcb, dcb_path = download_sinex(CAMPAIGN_DATE, RINEX_CACHE)

    # ── §3  Filter to stations with Rx DCBs ──────────────────────────────────
    print(f"\n──── §3  Filtering to stations with Rx DCBs in SINEX ───────────────")
    qualified = filter_stations_with_rx_dcb(all_nearby, dcb)
    if not qualified:
        print("  No qualifying stations — proceeding with all nearby stations.")
        qualified = all_nearby
    else:
        print(f"  {len(qualified)} / {len(all_nearby)} stations qualify:")
        for sta in qualified:
            print(f"    {sta['name']}")

    # ── §4a  Pre-download RINEX files (sequential, avoids concurrent CDDIS hits)
    print(f"\n──── §4a  Downloading RINEX files for {len(qualified)} stations ────────────")
    print(f"  ephem_stride = {EPHEM_STRIDE or 'auto'}  |  sv_workers = {NUM_SV_WORKERS}  |  verbose = {VERBOSE}")
    file_paths = predownload_station_files(qualified, CAMPAIGN_DATE, RINEX_CACHE)

    # ── §4b  Process stations (serial) with per-SV parallelism ───────────────
    print(f"\n──── §4b  Processing {len(qualified)} stations (serial, {NUM_SV_WORKERS} SV workers each) ───")
    obs_all, clean_all = process_all_stations(
        qualified, CAMPAIGN_DATE, dcb_path, RINEX_CACHE,
        file_paths     = file_paths,
        ephem_stride   = EPHEM_STRIDE,
        num_sv_workers = NUM_SV_WORKERS,
        verbose        = VERBOSE,
    )

    total_rays = sum(len(e["tec"]) for e in clean_all)
    print(f"\n  Arcs accepted  : {len(clean_all)}")
    print(f"  Observations   : {total_rays:,}")

    if not clean_all:
        print("\n  ERROR: No usable observations — check credentials and network.")
        return

    # ── §5  Diagnostic plots ──────────────────────────────────────────────────
    print(f"\n──── §5  Generating plots ──────────────────────────────────────────")
    plot_tec_constellations(obs_all, clean_all, SAVE_DIR)
    plot_dcb_summary(obs_all, SAVE_DIR)
    plot_station_map(qualified, clean_all, SAVE_DIR)

    elapsed = time.time() - t0
    print(f"\n{'=' * 68}")
    print(f"  Stopping before assimilation — {elapsed:.1f} s elapsed.")
    print(f"  Figures written to:  {SAVE_DIR}/")
    print(f"    tec_constellations.png  — 2×2 sTEC per constellation")
    print(f"    dcb_summary.png         — Tx / Rx DCB bar chart")
    print(f"    station_map.png         — station locations + IPP tracks")
    print("=" * 68)


if __name__ == "__main__":
    main()
