#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
demo_occultation_availability.py

Reads every podTc2 occultation file for a single day and characterises how many
occultations are available near the two EISCAT ISR sites (Svalbard / ESR and
Tromso / TRO) over the course of that day.

For each occultation the *time of the occultation* is taken to be the time of
its TEC-max point (the peak of the raw TEC profile), and its location is the
TEC-max tangent point (lat_tecmax_tangent / lon_tecmax_tangent).  An occultation
counts as "available" when that tangent point falls inside the same ISR ROI used
by demo_isr_da_comparison.py -- i.e. within ISR_ROI_MAX_KM great-circle distance
of ESR or TRO.

Two timeline plots are produced (stacked, shared x-axis over 00:00-24:00 UTC):

    (1) The total number of in-ROI occultations (all satellites combined) inside
        a rolling 1-hour window, evaluated across the whole day.
    (2) A scatter of each in-ROI occultation's TEC-max tangent-point distance to
        its closest ISR site, coloured by which site is closest -- the relative
        proximity of the occultations to the radars over time.

ROI constants (ISR sites, ISR_ROI_MAX_KM, PODTC_BASE) and the haversine helper
are imported from demo_isr_da_comparison so this stays a single source of truth.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4
import pyproj
from scipy.signal import find_peaks
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.path as mpath
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic

from demo_isr_da_comparison import (
    INSTRUMENTS, ISR_SITES, ISR_ROI_MAX_KM, PODTC_BASE, _haversine_km,
)
from demo_esr_isr import load_edps
from demo_isr_initial_conditions import _identify_instrument

# ── Longitude-shift experiment ────────────────────────────────────────────────
# Shift the simulated ISR ground stations (ESR / TRO) by this many degrees of
# longitude to see how station longitude alone affects GNSS-RO occultation
# availability. Set to 0.0 to restore the true Svalbard/Tromso coordinates.
# Figures from a shifted run are written to their own subfolder so they never
# overwrite the baseline (unshifted) results.
LON_SHIFT_DEG = 180.0
SAVE_SUBDIR = "North_America" if LON_SHIFT_DEG else None

if LON_SHIFT_DEG:
    INSTRUMENTS = {
        site: ({**inst,
                "lon": ((inst["lon"] + LON_SHIFT_DEG + 180.0) % 360.0) - 180.0}
               if site in ISR_SITES else inst)
        for site, inst in INSTRUMENTS.items()
    }

FIGURES_DIR = Path(__file__).parent / "Figures" / "Occultation_Availability"
if SAVE_SUBDIR:
    FIGURES_DIR = FIGURES_DIR / SAVE_SUBDIR

# ISR ground-station marker shape (colour now carries the satellite instead --
# see _SAT_COLOR below). Consistent naming with demo_isr_initial_conditions.
_INST_LABEL  = {"ESR": "ESR (Svalbard)", "TRO": "TRO (Tromso)"}
_SITE_MARKER = {"ESR": "o", "TRO": "^"}
_SITE_FALLBACK_MARKER = "D"

# Default nested proximity thresholds for the rolling-count panel (km). Ordered
# largest-first so the wider (higher-count) areas are drawn behind the tighter
# ones; the counts are nested (500 ⊂ 1500 ⊂ 2500 km). This is a *default* --
# plot_occultation_availability(roi_thresholds_km=...) can override it with any
# number of thresholds, and shades of green are derived to match at call time.
ROI_THRESHOLDS_KM = (2500.0, 1500.0, 500.0)


def _rolling_threshold_colors(thresholds) -> dict:
    """
    Derive a shade of green per rolling-window proximity threshold, so
    ROI_THRESHOLDS_KM (or any override) stays a freely adjustable parameter
    while the panel keeps the same green colour family regardless of how many
    thresholds are supplied. Largest threshold -> lightest green (drawn
    behind); smallest -> darkest (drawn on top).
    """
    cmap = plt.get_cmap("Greens")
    ordered = sorted(set(float(t) for t in thresholds), reverse=True)
    n = len(ordered)
    return {t: cmap(0.35 + 0.55 * i / max(n - 1, 1)) for i, t in enumerate(ordered)}


# Per-satellite colour / marker (podTc2 spacecraft code). Colour now encodes
# the LEO satellite on the proximity scatter and ROI map (panels 2 and 3);
# ISR-station identity on those panels is instead encoded by marker shape
# (see _SITE_MARKER above).
_SAT_MARKER = {"GN04": "o", "GN05": "s", "YM08": "^"}
_SAT_COLOR  = {"GN04": "darkorange", "GN05": "mediumpurple", "YM08": "teal"}
_SAT_LABEL  = {"GN04": "GNOMES-4", "GN05": "GNOMES-5", "YM08": "YAM-8"}
_SAT_FALLBACK_MARKER = "D"
_SAT_FALLBACK_COLOR  = "gray"

PODTC_SUFFIX = ".0001_nc"


# ─────────────────────────────────────────────────────────────────────────────
# Reading occultations + TEC-max time
# ─────────────────────────────────────────────────────────────────────────────

def _tecmax_time_and_point(fpath: Path) -> dict | None:
    """
    Read one podTc2 file and return its TEC-max timing/location metadata.

    The occultation time is the TEC-max time: the file's own datetime attributes
    (year/month/day/hour/minute/second, which mark the first sample) advanced by
    the elapsed seconds from the first sample to the raw-TEC peak
    (time[argmax(TEC)] - time[0]).  This reproduces the file's `tecmax` attribute
    (the raw TEC maximum) and its accompanying lat/lon_tecmax_tangent point.

    Returns None if the file is unreadable or fails a basic validity check.
    """
    try:
        with netCDF4.Dataset(str(fpath), "r") as nc:
            tec  = np.asarray(nc.variables["TEC"][:], dtype=float)
            tsec = np.asarray(nc.variables["time"][:], dtype=float)
            lat  = float(nc.getncattr("lat_tecmax_tangent"))
            lon  = float(nc.getncattr("lon_tecmax_tangent"))
            yr   = int(nc.getncattr("year"))
            mo   = int(nc.getncattr("month"))
            dy   = int(nc.getncattr("day"))
            hh   = int(nc.getncattr("hour"))
            mm   = int(nc.getncattr("minute"))
            ss   = int(nc.getncattr("second"))
    except Exception:
        return None

    if abs(lat) > 90 or tec.size == 0 or not np.isfinite(tec).any():
        return None

    idx = int(np.nanargmax(tec))
    offset_s = float(tsec[idx] - tsec[0])
    start = pd.Timestamp(yr, mo, dy, hh, mm, ss)
    tecmax_time = start + pd.to_timedelta(offset_s, unit="s")

    spacecraft = fpath.name.split(".")[0].replace("podTc2_", "")
    return {
        "filename":    fpath.name,
        "spacecraft":  spacecraft,
        "tecmax_time": tecmax_time,
        "lat":         lat,
        "lon":         lon,
        "tecmax":      float(np.nanmax(tec)),
    }


def read_day_occultations(podtc_dir: Path) -> pd.DataFrame:
    """
    Read every podTc2 file in `podtc_dir`, compute each occultation's TEC-max
    time and tangent point, and tag it with distances to the ISR sites.

    Columns: filename, spacecraft, tecmax_time, lat, lon, tecmax,
             dist_<site>_km (one per ISR site), nearest_site, nearest_km, in_roi
    """
    files = sorted(podtc_dir.glob(f"*{PODTC_SUFFIX}"))
    print(f"  Scanning {len(files)} podTc2 files in {podtc_dir} ...")

    rows = [r for f in files if (r := _tecmax_time_and_point(f)) is not None]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("tecmax_time").reset_index(drop=True)

    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    dist_cols = []
    for site in ISR_SITES:
        inst = INSTRUMENTS[site]
        col = f"dist_{site}_km"
        df[col] = _haversine_km(inst["lat"], inst["lon"], lat, lon)
        dist_cols.append(col)

    dist_mat = df[dist_cols].to_numpy()                 # (n_occ, n_site)
    nearest_idx = np.argmin(dist_mat, axis=1)
    df["nearest_site"] = [ISR_SITES[i] for i in nearest_idx]
    df["nearest_km"]   = dist_mat[np.arange(len(df)), nearest_idx]
    df["in_roi"]       = df["nearest_km"] <= ISR_ROI_MAX_KM

    print(f"  {len(df)} valid occultations, "
          f"{int(df['in_roi'].sum())} within {ISR_ROI_MAX_KM:.0f} km of an ISR site.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Rolling window count
# ─────────────────────────────────────────────────────────────────────────────

def rolling_window_count(
    times: pd.Series,
    day: pd.Timestamp,
    window_hours: float = 1.0,
    step_minutes: float = 1.0,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Count occultations inside a centred sliding window of width `window_hours`,
    evaluated on a `step_minutes` grid spanning 00:00-24:00 of `day`.

    Returns (grid_times, counts).
    """
    day0 = pd.Timestamp(day.year, day.month, day.day)
    grid = pd.date_range(day0, day0 + pd.Timedelta(days=1),
                         freq=f"{step_minutes}min")
    t = np.sort(times.to_numpy().astype("datetime64[ns]"))
    half = pd.Timedelta(hours=window_hours) / 2

    lo = np.searchsorted(t, (grid - half).to_numpy().astype("datetime64[ns]"), side="left")
    hi = np.searchsorted(t, (grid + half).to_numpy().astype("datetime64[ns]"), side="right")
    return grid, (hi - lo).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_occultation_availability(
    df: pd.DataFrame,
    day: pd.Timestamp,
    window_hours: float = 1.0,
    roi_thresholds_km=ROI_THRESHOLDS_KM,
    alt_min: float | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    2×2 panel summary of the in-ROI occultations for one day:
      (top-left)     rolling `window_hours` count at each of `roi_thresholds_km`
                     (all satellites combined), shown as nested green areas
      (bottom-left)  TEC-max tangent-point distance to the closest ISR site vs
                     time, coloured by satellite, marker-shaped by ISR station
      (top-right)    orthographic map with rings at each of `roi_thresholds_km`
      (bottom-right) horizontal histogram of the in-ROI distance distribution
                     (stacked by satellite), sharing the distance (y) axis with
                     the bottom-left panel

    `roi_thresholds_km` is freely adjustable (any number of thresholds) --
    shades of green for panels 1/3/4 are derived from it at call time via
    _rolling_threshold_colors so the palette always matches whatever
    thresholds are passed in.
    """
    roi = df[df["in_roi"]].copy()
    day0 = pd.Timestamp(day.year, day.month, day.day)
    thresh_colors = _rolling_threshold_colors(roi_thresholds_km)
    thresholds_sorted = sorted(thresh_colors.keys(), reverse=True)   # largest→smallest

    # Orthographic view centred between the two ISR sites.
    _clat = float(np.mean([INSTRUMENTS[s]["lat"] for s in ISR_SITES]))
    _clon = float(np.mean([INSTRUMENTS[s]["lon"] for s in ISR_SITES]))

    fig = plt.figure(figsize=(17, 9))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[3.0, 1.15],
                  height_ratios=[1.0, 1.2], hspace=0.12, wspace=0.14)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_map = fig.add_subplot(gs[0, 1], projection=ccrs.Orthographic(_clon, _clat))
    ax_hist = fig.add_subplot(gs[1, 1], sharey=ax_bot)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # ── (1) rolling count at nested proximity thresholds ───────────────────
    # Overlapping filled areas: for each threshold, count occultations whose
    # TEC-max tangent point is within that many km of *either* ISR site.
    peak = 0
    for thresh in thresholds_sorted:                       # largest → smallest
        sub = df[df["nearest_km"] <= thresh]
        grid, counts = rolling_window_count(sub["tecmax_time"], day, window_hours)
        color = thresh_colors[thresh]
        ax_top.fill_between(grid, counts, step="mid", alpha=0.55, color=color,
                            label=f"≤ {thresh:.0f} km (n={len(sub)})")
        ax_top.step(grid, counts, where="mid", color=color, lw=1.2)
        peak = max(peak, int(counts.max()) if counts.size else 0)
    ax_top.set_ylabel(f"# occultations in\n{window_hours:g} h rolling window")
    ax_top.set_ylim(bottom=0)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper right", fontsize=8, framealpha=0.9,
                  title="within of ESR/TRO")
    _gate = (f" · limb reaching ≤ {alt_min:g} km only"
             if alt_min is not None else "")
    ax_top.set_title(
        f"GNSS-RO occultation availability near EISCAT ISR sites — "
        f"{day0.date()}\n"
        f"{len(roi)} occultations within {ISR_ROI_MAX_KM:.0f} km of ESR/TRO · "
        f"peak {peak} per {window_hours:g} h window{_gate}",
        fontsize=12,
    )

    # ── (2) proximity to nearest ISR site ──────────────────────────────────
    # Colour encodes the LEO satellite; marker shape encodes the closest ISR
    # ground station.
    for site, site_grp in roi.groupby("nearest_site"):
        marker = _SITE_MARKER.get(site, _SITE_FALLBACK_MARKER)
        ax_bot.scatter(
            site_grp["tecmax_time"], site_grp["nearest_km"],
            s=34, marker=marker,
            c=[_SAT_COLOR.get(sat, _SAT_FALLBACK_COLOR) for sat in site_grp["spacecraft"]],
            edgecolor="k", linewidth=0.3, alpha=0.85,
        )
    ax_bot.axhline(ISR_ROI_MAX_KM, color="k", ls="--", lw=1.0)
    ax_bot.set_ylabel("TEC-max tangent point →\nnearest ISR site (km)")
    ax_bot.set_ylim(0, ISR_ROI_MAX_KM * 1.05)
    ax_bot.set_xlabel("Time of TEC-max point (UTC)")
    ax_bot.grid(True, alpha=0.3)

    # Two-part legend: colour = satellite, marker = closest ISR ground station.
    sat_handles = [
        Line2D([], [], marker="o", ls="none", color=_SAT_COLOR.get(sat, _SAT_FALLBACK_COLOR),
               markeredgecolor="k", markeredgewidth=0.3, markersize=7,
               label=f"{_SAT_LABEL.get(sat, sat)} (n={int((roi['spacecraft'] == sat).sum())})")
        for sat in sorted(roi["spacecraft"].unique())
    ]
    site_handles = [
        Line2D([], [], marker=_SITE_MARKER.get(s, _SITE_FALLBACK_MARKER),
               ls="none", color="0.4", markeredgecolor="k", markeredgewidth=0.3,
               markersize=7,
               label=f"closest to {_INST_LABEL.get(s, s)} "
                     f"(n={int((roi['nearest_site'] == s).sum())})")
        for s in ISR_SITES
    ]
    roi_handle = Line2D([], [], color="k", ls="--", lw=1.0,
                        label=f"ROI limit {ISR_ROI_MAX_KM:.0f} km")
    ax_bot.legend(handles=sat_handles + site_handles + [roi_handle],
                  loc="upper right", fontsize=8, framealpha=0.9, ncol=2)

    ax_bot.set_xlim(day0, day0 + pd.Timedelta(days=1))
    ax_bot.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for lbl in ax_bot.get_xticklabels():
        lbl.set_rotation(0)
        lbl.set_ha("center")

    # ── (3) orthographic map with the ROI rings ────────────────────────────
    ax_map.add_feature(cfeature.LAND, facecolor="0.92", zorder=0)
    ax_map.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax_map.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.6", zorder=1)
    ax_map.gridlines(color="0.85", lw=0.4)
    # ax_map.set_extent([-40, 75, 54, 90], crs=ccrs.PlateCarree())

    # In-ROI occultation tangent points for context: colour = satellite,
    # marker shape = closest ISR ground station (same encoding as panel 2).
    for site, site_grp in roi.groupby("nearest_site"):
        marker = _SITE_MARKER.get(site, _SITE_FALLBACK_MARKER)
        ax_map.scatter(
            site_grp["lon"], site_grp["lat"], transform=ccrs.PlateCarree(),
            s=16, marker=marker,
            c=[_SAT_COLOR.get(sat, _SAT_FALLBACK_COLOR) for sat in site_grp["spacecraft"]],
            edgecolor="k", linewidth=0.2, alpha=0.75, zorder=2,
        )

    # Geodesic ROI rings around each ISR site, coloured to match the top-left
    # area plot; drawn largest-first so tighter rings sit on top.
    geod = Geodesic()
    for thresh in thresholds_sorted:
        for site in ISR_SITES:
            inst = INSTRUMENTS[site]
            ring = np.asarray(
                geod.circle(lon=inst["lon"], lat=inst["lat"],
                            radius=thresh * 1000.0, n_samples=181)
            )
            # Geodetic (not PlateCarree): these rings can wrap over the pole
            # (radius exceeds the station's distance to the pole), and a
            # PlateCarree transform draws straight lon/lat-space chords
            # between samples there, faceting the ring into a rosette/pentagon.
            ax_map.plot(ring[:, 0], ring[:, 1], transform=ccrs.Geodetic(),
                        color=thresh_colors[thresh], lw=1.6, zorder=3)

    for site in ISR_SITES:
        inst = INSTRUMENTS[site]
        ax_map.plot(inst["lon"], inst["lat"], transform=ccrs.PlateCarree(),
                    marker="*", markersize=13, color="gold",
                    markeredgecolor="k", markeredgewidth=0.5, zorder=4)
        ax_map.text(inst["lon"], inst["lat"] + 1.2, site,
                    transform=ccrs.PlateCarree(), fontsize=8, fontweight="bold",
                    ha="center", va="bottom", zorder=5)
    ax_map.set_title(
        f"ROI rings ({' / '.join(f'{t:.0f}' for t in thresholds_sorted)} km)",
        fontsize=10,
    )

    ring_handles = [Line2D([], [], color=thresh_colors[t], lw=1.6, label=f"{t:.0f} km")
                    for t in thresholds_sorted]
    map_sat_handles = [
        Line2D([], [], marker="o", ls="none", color=_SAT_COLOR.get(sat, _SAT_FALLBACK_COLOR),
               markeredgecolor="k", markeredgewidth=0.3, markersize=6,
               label=_SAT_LABEL.get(sat, sat))
        for sat in sorted(roi["spacecraft"].unique())
    ]
    map_site_handles = [
        Line2D([], [], marker=_SITE_MARKER.get(s, _SITE_FALLBACK_MARKER), ls="none",
               color="0.4", markeredgecolor="k", markeredgewidth=0.3, markersize=6,
               label=s)
        for s in ISR_SITES
    ]
    ax_map.legend(handles=ring_handles + map_sat_handles + map_site_handles,
                 loc="lower left", fontsize=6.5, framealpha=0.9, ncol=1)

    # ── (4) distance-distribution histogram (shares distance axis w/ panel 2)─
    # Stacked by satellite (rather than by ISR site) to match the marker-shape
    # encoding of the bottom-left proximity panel.
    bins = np.arange(0.0, ISR_ROI_MAX_KM + 100.0, 100.0)
    sats       = sorted(roi["spacecraft"].unique())
    sat_data   = [roi.loc[roi["spacecraft"] == sat, "nearest_km"] for sat in sats]
    sat_colors = [_SAT_COLOR.get(sat, _SAT_FALLBACK_COLOR) for sat in sats]
    ax_hist.hist(sat_data, bins=bins, orientation="horizontal", stacked=True,
                 color=sat_colors, edgecolor="k", linewidth=0.3,
                 label=[_SAT_LABEL.get(sat, sat) for sat in sats])
    for thresh in thresholds_sorted:
        ax_hist.axhline(thresh, color=thresh_colors[thresh], ls="--", lw=1.0)
    ax_hist.set_xlabel("# occultations")
    ax_hist.set_title(f"Distance distribution\n(≤ {ISR_ROI_MAX_KM:.0f} km, "
                      f"{len(roi)} occ)", fontsize=10)
    ax_hist.grid(True, axis="x", alpha=0.3)
    ax_hist.legend(loc="lower right", fontsize=7, framealpha=0.9)
    plt.setp(ax_hist.get_yticklabels(), visible=False)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LEO ground tracks (per-sample sub-satellite lat/lon) + animation
# ─────────────────────────────────────────────────────────────────────────────
#
# The podTc2 files store per-sample LEO positions as ECEF kilometres
# (x_LEO/y_LEO/z_LEO). Converting them with the ECEF→geodetic transform below
# and dropping the height gives the sub-satellite ground track. NOTE: the files'
# start_time/stop_time/time variables are NOT unix seconds (interpreting them as
# such lands in 2015, ~10 yr off); the authoritative absolute clock is the
# year/month/day/hour/minute/second attributes (= arc START), advanced by the
# per-sample elapsed offset (time[i] - time[0]). This mirrors the TEC-max timing
# in _tecmax_time_and_point so the ground tracks and the tangent points share one
# consistent clock.

# ECEF (EPSG:4978) → geodetic lon/lat/height (EPSG:4979). Built once and reused.
_ECEF_TO_GEODETIC = pyproj.Transformer.from_crs(
    "EPSG:4978", "EPSG:4979", always_xy=True)

DEFAULT_TRACK_CACHE_DIR = Path(__file__).parent / "Data" / "GroundTrack_Cache"


def _leo_track_points(fpath: Path, decimate_s: float = 15.0,
                      check: bool = False) -> dict | None:
    """
    Read one podTc2 file and return its decimated LEO sub-satellite ground track.

    Returns a dict with the spacecraft code and equal-length numpy arrays
    utc_time / lat / lon (decimated to roughly `decimate_s` spacing), or None if
    the file is unreadable. When `check` is True the first-sample ECEF→geodetic
    conversion is asserted against the file's lat_start/lon_start attributes.
    """
    try:
        with netCDF4.Dataset(str(fpath), "r") as nc:
            x    = np.asarray(nc.variables["x_LEO"][:], dtype=float)
            y    = np.asarray(nc.variables["y_LEO"][:], dtype=float)
            z    = np.asarray(nc.variables["z_LEO"][:], dtype=float)
            tsec = np.asarray(nc.variables["time"][:], dtype=float)
            yr   = int(nc.getncattr("year"));   mo = int(nc.getncattr("month"))
            dy   = int(nc.getncattr("day"));     hh = int(nc.getncattr("hour"))
            mm   = int(nc.getncattr("minute"));  ss = int(nc.getncattr("second"))
            lat_start = float(nc.getncattr("lat_start"))
            lon_start = float(nc.getncattr("lon_start"))
    except Exception:
        return None

    n = x.size
    if n == 0 or tsec.size != n:
        return None

    # Decimate on the sample index. Native cadence is ~1 Hz, so the step is just
    # decimate_s samples; clamp to >= 1 and always keep the first sample.
    dt = float(np.median(np.diff(tsec))) if n > 1 else 1.0
    step = max(1, int(round(decimate_s / dt))) if dt > 0 else 1
    sel = np.arange(0, n, step)

    lon, lat, _h = _ECEF_TO_GEODETIC.transform(
        x[sel] * 1000.0, y[sel] * 1000.0, z[sel] * 1000.0)
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    if check:
        assert abs(lat[0] - lat_start) < 1e-4 and abs(lon[0] - lon_start) < 1e-4, (
            f"ECEF→geodetic self-check failed for {fpath.name}: "
            f"got ({lat[0]:.4f},{lon[0]:.4f}) vs attr ({lat_start:.4f},{lon_start:.4f})")

    start = pd.Timestamp(yr, mo, dy, hh, mm, ss)
    utc = start + pd.to_timedelta(tsec[sel] - tsec[0], unit="s")

    spacecraft = fpath.name.split(".")[0].replace("podTc2_", "")
    return {"spacecraft": spacecraft, "utc_time": utc.to_numpy(),
            "lat": lat, "lon": lon}


def read_day_ground_tracks(podtc_dir: Path, decimate_s: float = 15.0,
                           self_check_n: int = 3) -> pd.DataFrame:
    """
    Read every podTc2 file in `podtc_dir` and assemble a tidy per-sample LEO
    ground-track table with columns [sat, utc_time, lat, lon], decimated to
    ~`decimate_s` spacing. The first `self_check_n` readable files are converted
    with the ECEF→geodetic self-check enabled.
    """
    files = sorted(podtc_dir.glob(f"*{PODTC_SUFFIX}"))
    print(f"  Building LEO ground tracks from {len(files)} podTc2 files "
          f"(decimate ~{decimate_s:g}s) ...")

    sats, times, lats, lons = [], [], [], []
    checked = 0
    for i, f in enumerate(files):
        do_check = checked < self_check_n
        rec = _leo_track_points(f, decimate_s=decimate_s, check=do_check)
        if rec is None:
            continue
        if do_check:
            checked += 1
        k = rec["lat"].size
        sats.append(np.repeat(rec["spacecraft"], k))
        times.append(rec["utc_time"])
        lats.append(rec["lat"])
        lons.append(rec["lon"])
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(files)} files ...")

    if not sats:
        return pd.DataFrame(columns=["sat", "utc_time", "lat", "lon"])

    df = pd.DataFrame({
        "sat":      np.concatenate(sats),
        "utc_time": pd.to_datetime(np.concatenate(times)),
        "lat":      np.concatenate(lats),
        "lon":      np.concatenate(lons),
    }).sort_values("utc_time").reset_index(drop=True)
    print(f"  {len(df)} ground-track samples across {df['sat'].nunique()} satellites.")
    return df


def read_day_ground_tracks_cached(
    podtc_dir: Path, day: pd.Timestamp, decimate_s: float = 15.0,
    cache_dir: Path | None = None, use_cache: bool = True,
) -> pd.DataFrame:
    """
    Memoised read_day_ground_tracks (one pickle per day+decimation). The full
    per-sample read of a busy day (~9,500 arcs) is far heavier than the
    tecmax-only scan, so caching keeps repeat/animation runs fast.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_TRACK_CACHE_DIR
    cache_path = cache_dir / f"{day.strftime('%Y%m%d')}_tracks_d{int(round(decimate_s))}.pkl"

    if use_cache and cache_path.exists():
        return pd.read_pickle(cache_path)

    df = read_day_ground_tracks(podtc_dir, decimate_s=decimate_s)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
    return df


def _polar_circular_boundary(ax) -> None:
    """Clip a polar-stereographic axes to a circle (nicer than the square frame)."""
    theta = np.linspace(0, 2 * np.pi, 200)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    ax.set_boundary(mpath.Path(verts * radius + center), transform=ax.transAxes)


def animate_ground_tracks(
    day,
    step_minutes: float = 5.0,
    trail_minutes: float = 25.0,
    window_hours: float = 1.0,
    decimate_s: float = 15.0,
    lat_min: float = 40.0,
    fps: int = 12,
    alt_cap: float = 700.0,
    alt_min: float | None = None,
    save_path: Path | None = None,
    cache_dir: Path | None = None,
    track_cache_dir: Path | None = None,
    minalt_cache_dir: Path | None = None,
    use_cache: bool = True,
) -> Path | None:
    """
    Render an animated GIF/MP4 of the day's LEO ground tracks over the polar cap
    together with the GNSS-RO tangent points as they occur, so the orbital
    geometry funnelling soundings into the ESR/TRO ROI is visually obvious.

    Each frame (spaced `step_minutes`) at UTC time t shows, on a NorthPolarStereo
    map covering `lat_min`..90°N:
      • per-satellite current sub-satellite marker (nearest track sample ≤ t) plus
        a fading trail over the trailing `trail_minutes`, coloured by _SAT_COLOR;
      • occultation TEC-max tangent points with tecmax_utc ≤ t — those inside the
        trailing `window_hours` drawn large/opaque, older ones faded — coloured by
        satellite; in-ROI points ringed to tie them to the availability metric;
      • ESR/TRO gold stars and the 500/1500/2500 km geodesic ROI rings (always on);
      • a clock title with the current UTC and the trailing-`window_hours` in-ROI
        occultation count (via rolling_window_count).

    Returns the written file path (or None if the day has no data).
    """
    day = pd.Timestamp(day)
    day0 = pd.Timestamp(day.year, day.month, day.day)

    podtc_dir = _resolve_podtc_dir(day)
    if podtc_dir is None:
        print(f"[skip] No podTc2 directory with data for {day.date()} under {PODTC_BASE}")
        return None

    # ── data: per-sample ground tracks + per-occultation tangent points ────────
    tracks = read_day_ground_tracks_cached(
        podtc_dir, day, decimate_s=decimate_s,
        cache_dir=track_cache_dir, use_cache=use_cache)
    occ = read_day_occultations_cached(
        podtc_dir, day, cache_dir=cache_dir, use_cache=use_cache)
    if tracks.empty:
        print(f"[skip] No ground-track samples for {day.date()}.")
        return None

    # Gate occultations to those whose limb sounding reaches ≤ alt_min km, using
    # the same accept test (and alt_cap) as the TEC-profile panels, so the movie's
    # trailing in-ROI count matches the plotted-profile count.
    if alt_min is not None and not occ.empty:
        occ = attach_min_tan_alt(occ, podtc_dir, day, scope="all", alt_cap=alt_cap,
                                 cache_dir=minalt_cache_dir, use_cache=use_cache)
        before = len(occ)
        occ = occ[occ["min_tan_alt_km"] <= alt_min].reset_index(drop=True)
        print(f"  alt_min gate: {len(occ)}/{before} occultations reach ≤ "
              f"{alt_min:g} km (alt_cap {alt_cap:g} km).")

    # Integer-ns time axes for fast per-frame windowing.
    trk_t = tracks["utc_time"].to_numpy().astype("datetime64[ns]").astype("int64")
    trk_order = np.argsort(trk_t)
    trk_t = trk_t[trk_order]
    trk_sat = tracks["sat"].to_numpy()[trk_order]
    trk_lat = tracks["lat"].to_numpy()[trk_order]
    trk_lon = tracks["lon"].to_numpy()[trk_order]

    has_occ = not occ.empty
    if has_occ:
        occ = occ.sort_values("tecmax_time").reset_index(drop=True)
        occ_t = occ["tecmax_time"].to_numpy().astype("datetime64[ns]").astype("int64")
        occ_sat = occ["spacecraft"].to_numpy()
        occ_lat = occ["lat"].to_numpy()
        occ_lon = occ["lon"].to_numpy()
        occ_inroi = occ["in_roi"].to_numpy()

    trail_ns = int(trail_minutes * 60 * 1e9)
    window_ns = int(window_hours * 3600 * 1e9)

    frames = pd.date_range(day0, day0 + pd.Timedelta(days=1),
                           freq=f"{step_minutes:g}min")
    frame_ns = frames.to_numpy().astype("datetime64[ns]").astype("int64")

    # Trailing in-ROI count for the title, on the same frame grid.
    if has_occ:
        roi_times = occ.loc[occ["in_roi"], "tecmax_time"]
        _grid, roi_counts = rolling_window_count(
            roi_times, day, window_hours=window_hours, step_minutes=step_minutes)
    else:
        roi_counts = np.zeros(len(frames), dtype=int)

    sats = sorted(set(np.unique(trk_sat)) | ({*occ_sat.tolist()} if has_occ else set()))

    # ── figure ─────────────────────────────────────────────────────────────────
    _clon = float(np.mean([INSTRUMENTS[s]["lon"] for s in ISR_SITES]))
    proj = ccrs.NorthPolarStereo(central_longitude=_clon)
    fig = plt.figure(figsize=(8.4, 8.8))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-180, 180, lat_min, 90], crs=ccrs.PlateCarree())
    _polar_circular_boundary(ax)
    ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.6", zorder=1)
    ax.gridlines(color="0.85", lw=0.4, zorder=1)

    # Static overlay: ISR sites + geodesic ROI rings (always on).
    geod = Geodesic()
    thresholds_sorted = sorted(ROI_THRESHOLDS_KM, reverse=True)
    thresh_colors = _rolling_threshold_colors(ROI_THRESHOLDS_KM)
    for thresh in thresholds_sorted:
        for site in ISR_SITES:
            inst = INSTRUMENTS[site]
            ring = np.asarray(geod.circle(lon=inst["lon"], lat=inst["lat"],
                                          radius=thresh * 1000.0, n_samples=181))
            # Geodetic (not PlateCarree): see plot_occultation_availability for why.
            ax.plot(ring[:, 0], ring[:, 1], transform=ccrs.Geodetic(),
                    color=thresh_colors[thresh], lw=1.4, zorder=4)
    for site in ISR_SITES:
        inst = INSTRUMENTS[site]
        ax.plot(inst["lon"], inst["lat"], transform=ccrs.PlateCarree(),
                marker="*", markersize=15, color="gold",
                markeredgecolor="k", markeredgewidth=0.6, zorder=6)
        ax.text(inst["lon"], inst["lat"] + 1.3, site, transform=ccrs.PlateCarree(),
                fontsize=9, fontweight="bold", ha="center", va="bottom", zorder=7)

    # Legend: satellites, ROI rings, tangent-point / ISR-site glyphs.
    legend_handles = [
        Line2D([], [], marker=_SAT_MARKER.get(s, _SAT_FALLBACK_MARKER), ls="none",
               color=_SAT_COLOR.get(s, _SAT_FALLBACK_COLOR), markeredgecolor="k",
               markeredgewidth=0.3, markersize=8, label=_SAT_LABEL.get(s, s))
        for s in sats
    ]
    legend_handles += [
        Line2D([], [], color=thresh_colors[t], lw=1.4, label=f"ROI {t:.0f} km")
        for t in thresholds_sorted
    ]
    legend_handles += [
        Line2D([], [], marker="*", ls="none", color="gold", markeredgecolor="k",
               markeredgewidth=0.5, markersize=11, label="ESR / TRO"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="none",
               markeredgecolor="k", markersize=9, label="in-ROI tangent pt"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=7.5,
              framealpha=0.9, ncol=1, bbox_to_anchor=(-0.02, -0.02))

    title = ax.set_title("", fontsize=12)

    # Dynamic artists get removed & rebuilt each frame (blit=False; simplest and
    # robust with cartopy transforms).
    dynamic: list = []

    def _rgba(color, alpha):
        r, g, b, _ = matplotlib.colors.to_rgba(color)
        return (r, g, b, float(alpha))

    def update(i):
        for art in dynamic:
            art.remove()
        dynamic.clear()
        t = frame_ns[i]

        # LEO trails + current sub-satellite marker, per satellite.
        lo = np.searchsorted(trk_t, t - trail_ns, side="left")
        hi = np.searchsorted(trk_t, t, side="right")
        if hi > lo:
            seg_t = trk_t[lo:hi]; seg_sat = trk_sat[lo:hi]
            seg_lat = trk_lat[lo:hi]; seg_lon = trk_lon[lo:hi]
            age_frac = (t - seg_t) / max(trail_ns, 1)          # 0 newest → 1 oldest
            for s in sats:
                m = seg_sat == s
                if not m.any():
                    continue
                base = _SAT_COLOR.get(s, _SAT_FALLBACK_COLOR)
                alphas = 0.12 + 0.75 * (1.0 - age_frac[m])
                colors = [_rgba(base, a) for a in alphas]
                sc = ax.scatter(seg_lon[m], seg_lat[m], transform=ccrs.PlateCarree(),
                                s=7, marker=".", c=colors, linewidths=0, zorder=3)
                dynamic.append(sc)
                j = np.nonzero(m)[0][-1]                        # newest ≤ t
                cur = ax.scatter(seg_lon[m][-1], seg_lat[m][-1],
                                 transform=ccrs.PlateCarree(), s=110,
                                 marker=_SAT_MARKER.get(s, _SAT_FALLBACK_MARKER),
                                 c=[base], edgecolor="k", linewidth=0.8, zorder=6)
                dynamic.append(cur)

        # Occultation tangent points with tecmax ≤ t (recent big, older faded).
        if has_occ:
            k = np.searchsorted(occ_t, t, side="right")
            if k > 0:
                pt_t = occ_t[:k]; recent = (t - pt_t) <= window_ns
                for s in sats:
                    m = occ_sat[:k] == s
                    if not m.any():
                        continue
                    base = _SAT_COLOR.get(s, _SAT_FALLBACK_COLOR)
                    rec_m = m & recent; old_m = m & ~recent
                    if old_m.any():
                        art = ax.scatter(occ_lon[:k][old_m], occ_lat[:k][old_m],
                                         transform=ccrs.PlateCarree(), s=12,
                                         c=[_rgba(base, 0.18)], linewidths=0, zorder=4)
                        dynamic.append(art)
                    if rec_m.any():
                        ec = np.where(occ_inroi[:k][rec_m], "k", "none")
                        art = ax.scatter(occ_lon[:k][rec_m], occ_lat[:k][rec_m],
                                         transform=ccrs.PlateCarree(), s=48,
                                         c=[base], edgecolor=list(ec),
                                         linewidth=0.9, alpha=0.95, zorder=5)
                        dynamic.append(art)

        t_utc = pd.Timestamp(t).tz_localize("UTC")
        _gate = f"   ·   limb ≤ {alt_min:g} km" if alt_min is not None else ""
        title.set_text(
            f"GNSS-RO LEO ground tracks & occultation tangent points — "
            f"{day0.date()}\n{t_utc:%H:%M} UTC   ·   "
            f"in-ROI occ in trailing {window_hours:g} h: {int(roi_counts[i])}   ·   "
            f"trail {trail_minutes:g} min{_gate}")
        return dynamic

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / max(fps, 1),
                         blit=False)

    if save_path is None:
        save_path = FIGURES_DIR / "Movies" / f"ground_tracks_{day.strftime('%Y%m%d')}.gif"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer MP4 (FFMpeg) but fall back to an animated GIF (Pillow) when ffmpeg
    # isn't installed.
    wrote = None
    if FFMpegWriter.isAvailable():
        try:
            mp4 = save_path.with_suffix(".mp4")
            print(f"  Encoding MP4 → {mp4} ({len(frames)} frames @ {fps} fps) ...")
            anim.save(str(mp4), writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=110)
            wrote = mp4
        except Exception as exc:
            print(f"  FFMpeg encode failed ({exc}); falling back to GIF.")
    if wrote is None:
        gif = save_path.with_suffix(".gif")
        print(f"  Encoding GIF → {gif} ({len(frames)} frames @ {fps} fps) ...")
        anim.save(str(gif), writer=PillowWriter(fps=fps), dpi=100)
        wrote = gif

    plt.close(fig)
    print(f"  Saved movie → {wrote}")
    return wrote


# ─────────────────────────────────────────────────────────────────────────────
# Altitude-vs-TEC occultation profiles, windowed by rolling-count local minima
# ─────────────────────────────────────────────────────────────────────────────
#
# Each podTc2 arc carries a per-sample slant-TEC series. The tangent (impact)
# height of the LEO→GNSS ray descends/rises through the ionosphere over the arc,
# so plotting TEC against that tangent altitude gives the classic occultation
# electron-content profile (peaking in the F2 region ~250-350 km). We render
# these as a 2×2 grid — one panel per GNSS constellation (GPS / GLONASS /
# Galileo / BeiDou) — matching the TEC-profile styling in
# plotIonosphereTomography._plot_group (CONSTELLATION_CONFIG colour families,
# shade deepening with occultation order; _CONST_PANEL layout).
#
# The day is partitioned into time windows at the *local minima* of the rolling
# 1-hour occultation count (the same availability metric as the summary figure).
# Those minima fall ~every 90 min — the LEO orbital cadence — so each window is
# one "burst" of soundings between two availability lulls, and the minima are
# found adaptively (scipy.signal.find_peaks on the negated count) rather than at
# hard-coded boundaries.

DEFAULT_PROFILE_CACHE_DIR = Path(__file__).parent / "Data" / "Profile_Cache"

# GNSS PRN token in a podTc2 filename, e.g. "E29"/"G03"/"R13"/"C22"/"J07"/"S28".
_PRN_RE = re.compile(r"^[GRECJIS]\d{2}$")

# Constellation panel layout (mirrors demo_group.CONSTELLATION_CONFIG /
# plotIonosphereTomography._CONST_POS): GPS top-left, GLONASS bottom-left,
# Galileo top-right, BeiDou bottom-right.
_CONST_PANEL = {"G": (0, 0), "R": (1, 0), "E": (0, 1), "C": (1, 1)}
_CONST_FALLBACK = {"name": "GPS", "cmap": "Blues", "title_color": "steelblue"}


def _constellation_config() -> dict:
    """demo_group.CONSTELLATION_CONFIG when importable (single source of truth),
    otherwise a local copy so this module stays runnable on its own."""
    try:
        from demo_group import CONSTELLATION_CONFIG
        return CONSTELLATION_CONFIG
    except Exception:
        return {
            "G": {"name": "GPS",     "cmap": "Blues",   "title_color": "steelblue"},
            "R": {"name": "GLONASS", "cmap": "Purples", "title_color": "mediumpurple"},
            "E": {"name": "Galileo", "cmap": "Oranges", "title_color": "darkorange"},
            "C": {"name": "BeiDou",  "cmap": "Greens",  "title_color": "seagreen"},
        }


def _parse_prn(filename: str) -> str:
    """Extract the GNSS transmitter PRN (e.g. 'E29') from a podTc2 filename."""
    for tok in filename.split("."):
        if _PRN_RE.match(tok):
            return tok
    return "?"


def _occultation_tec_profile(fpath: Path, alt_cap: float = 700.0,
                             alt_min: float = 400.0,
                             max_pts: int = 250) -> dict | None:
    """
    Read one podTc2 arc and return its TEC-vs-tangent-altitude limb profile.

    The tangent point is the foot of the perpendicular from Earth's centre to the
    straight LEO→GNSS ray; its geodetic height (via EPSG:4978→4979) is the impact
    altitude. Only the physical limb-sounding portion is kept — samples whose
    tangent point lies between the two spacecraft (0 ≤ t ≤ 1) and at 0..alt_cap km
    with finite TEC. Points are sorted by altitude and decimated to ≤ max_pts.

    The arc is *accepted* only if its tangent point descends to at least
    `alt_min` km (the lowest kept tangent altitude ≤ alt_min); passes that only
    graze the topside above alt_min are rejected, so every retained profile
    actually sounds down through the requested altitude.

    Returns {tan_alt, tec} (equal-length arrays, ascending altitude) or None if
    the arc has too few valid limb samples or never reaches alt_min.
    """
    try:
        with netCDF4.Dataset(str(fpath), "r") as nc:
            tec = np.asarray(nc.variables["TEC"][:], dtype=float)
            A = np.stack([np.asarray(nc.variables[v][:], dtype=float)
                          for v in ("x_GPS", "y_GPS", "z_GPS")], axis=1)  # GNSS, km
            B = np.stack([np.asarray(nc.variables[v][:], dtype=float)
                          for v in ("x_LEO", "y_LEO", "z_LEO")], axis=1)  # LEO, km
    except Exception:
        return None
    if tec.size == 0 or A.shape[0] != tec.size:
        return None

    AB = B - A
    denom = np.sum(AB * AB, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tpar = -np.sum(A * AB, axis=1) / denom
    P = A + tpar[:, None] * AB
    _lon, _lat, h = _ECEF_TO_GEODETIC.transform(
        P[:, 0] * 1000.0, P[:, 1] * 1000.0, P[:, 2] * 1000.0)
    h = np.asarray(h, dtype=float) / 1000.0

    valid = (tpar >= 0) & (tpar <= 1) & (h >= 0) & (h <= alt_cap) & np.isfinite(tec)
    if int(valid.sum()) < 5:
        return None

    hv, tv = h[valid], tec[valid]
    if float(np.nanmin(hv)) > alt_min:        # never descends to the required depth
        return None
    order = np.argsort(hv)
    hv, tv = hv[order], tv[order]
    if hv.size > max_pts:
        sel = np.linspace(0, hv.size - 1, max_pts).round().astype(int)
        hv, tv = hv[sel], tv[sel]
    return {"tan_alt": hv, "tec": tv}


def read_day_profiles(podtc_dir: Path, occ_df: pd.DataFrame,
                      alt_cap: float = 700.0, alt_min: float = 400.0,
                      max_pts: int = 250) -> dict:
    """
    Read the TEC-vs-tangent-altitude limb profile for every file listed in
    `occ_df["filename"]`, returning {filename: {tan_alt, tec}} (files with too
    few valid limb samples, or whose tangent point never descends to alt_min,
    are omitted).
    """
    files = list(occ_df["filename"])
    print(f"  Reading TEC/altitude profiles for {len(files)} occultations "
          f"(alt {alt_min:g}..{alt_cap:g} km) ...")
    prof: dict = {}
    for i, fname in enumerate(files):
        rec = _occultation_tec_profile(podtc_dir / fname, alt_cap=alt_cap,
                                       alt_min=alt_min, max_pts=max_pts)
        if rec is not None:
            prof[fname] = rec
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(files)} profiles ...")
    print(f"  {len(prof)} usable limb profiles reaching ≤ {alt_min:g} km.")
    return prof


def read_day_profiles_cached(
    podtc_dir: Path, day: pd.Timestamp, occ_df: pd.DataFrame,
    roi_only: bool = True, alt_cap: float = 700.0, alt_min: float = 400.0,
    max_pts: int = 250, cache_dir: Path | None = None, use_cache: bool = True,
) -> dict:
    """Memoised read_day_profiles (one pickle per day+roi_only+alt_cap+alt_min)."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_PROFILE_CACHE_DIR
    tag = "roi" if roi_only else "all"
    cache_path = (cache_dir / f"{day.strftime('%Y%m%d')}_prof_{tag}"
                  f"_a{int(round(alt_cap))}_m{int(round(alt_min))}.pkl")

    if use_cache and cache_path.exists():
        return pd.read_pickle(cache_path)

    prof = read_day_profiles(podtc_dir, occ_df, alt_cap=alt_cap, alt_min=alt_min,
                             max_pts=max_pts)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(prof, cache_path)
    return prof


# ─────────────────────────────────────────────────────────────────────────────
# Min tangent altitude gate (shared accept test for availability ⇄ profiles)
# ─────────────────────────────────────────────────────────────────────────────
#
# The availability figure/movie count *all* in-ROI occultations, whereas the
# TEC-profile panels only draw arcs whose limb sounding descends deep enough to
# be usable (_occultation_tec_profile). To make the two counts agree, gate the
# availability occultations on the same accept test: keep a pass iff its lowest
# tangent altitude is ≤ alt_min. _occultation_min_tan_alt reproduces
# _occultation_tec_profile's valid-sample mask (0 ≤ t ≤ 1, 0 ≤ h ≤ alt_cap,
# finite TEC, ≥ 5 samples) exactly, so `min_tan_alt_km <= alt_min` accepts
# precisely the arcs that yield a plotted profile.

DEFAULT_MINALT_CACHE_DIR = Path(__file__).parent / "Data" / "MinTanAlt_Cache"


def _occultation_min_tan_alt(fpath: Path, alt_cap: float = 700.0) -> float:
    """
    Lowest tangent (impact) altitude in km that one podTc2 arc's limb sounding
    reaches, using the identical valid-sample gate as _occultation_tec_profile.
    Returns np.nan when the arc has too few valid limb samples, so that the test
    `min_tan_alt_km <= alt_min` reproduces that function's accept/reject decision
    (nan fails the test just like a rejected profile).
    """
    try:
        with netCDF4.Dataset(str(fpath), "r") as nc:
            tec = np.asarray(nc.variables["TEC"][:], dtype=float)
            A = np.stack([np.asarray(nc.variables[v][:], dtype=float)
                          for v in ("x_GPS", "y_GPS", "z_GPS")], axis=1)  # GNSS, km
            B = np.stack([np.asarray(nc.variables[v][:], dtype=float)
                          for v in ("x_LEO", "y_LEO", "z_LEO")], axis=1)  # LEO, km
    except Exception:
        return np.nan
    if tec.size == 0 or A.shape[0] != tec.size:
        return np.nan

    AB = B - A
    denom = np.sum(AB * AB, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tpar = -np.sum(A * AB, axis=1) / denom
    P = A + tpar[:, None] * AB
    _lon, _lat, h = _ECEF_TO_GEODETIC.transform(
        P[:, 0] * 1000.0, P[:, 1] * 1000.0, P[:, 2] * 1000.0)
    h = np.asarray(h, dtype=float) / 1000.0

    valid = (tpar >= 0) & (tpar <= 1) & (h >= 0) & (h <= alt_cap) & np.isfinite(tec)
    if int(valid.sum()) < 5:
        return np.nan
    return float(np.nanmin(h[valid]))


def read_day_min_tan_alt(podtc_dir: Path, filenames,
                         alt_cap: float = 700.0) -> dict:
    """{filename: lowest tangent altitude km} for every file (np.nan when unusable)."""
    filenames = list(filenames)
    print(f"  Computing min tangent altitude for {len(filenames)} occultations "
          f"(≤ {alt_cap:g} km gate) ...")
    out: dict = {}
    for i, fname in enumerate(filenames):
        out[fname] = _occultation_min_tan_alt(podtc_dir / fname, alt_cap=alt_cap)
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{len(filenames)} ...")
    return out


def read_day_min_tan_alt_cached(
    podtc_dir: Path, day: pd.Timestamp, filenames, scope: str = "roi",
    alt_cap: float = 700.0, cache_dir: Path | None = None, use_cache: bool = True,
) -> dict:
    """Memoised read_day_min_tan_alt (one pickle per day+scope+alt_cap)."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_MINALT_CACHE_DIR
    cache_path = (cache_dir / f"{day.strftime('%Y%m%d')}_minalt_{scope}"
                  f"_a{int(round(alt_cap))}.pkl")
    if use_cache and cache_path.exists():
        return pd.read_pickle(cache_path)
    out = read_day_min_tan_alt(podtc_dir, filenames, alt_cap=alt_cap)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(out, cache_path)
    return out


def attach_min_tan_alt(
    occ_df: pd.DataFrame, podtc_dir: Path, day: pd.Timestamp, scope: str = "roi",
    alt_cap: float = 700.0, cache_dir: Path | None = None, use_cache: bool = True,
) -> pd.DataFrame:
    """
    Return a copy of `occ_df` with a `min_tan_alt_km` column (lowest tangent
    altitude each arc reaches, np.nan when unusable), so callers can gate
    occultations on `min_tan_alt_km <= alt_min` — the same accept test
    _occultation_tec_profile applies — making the availability count equal the
    number of plotted TEC profiles.
    """
    min_map = read_day_min_tan_alt_cached(
        podtc_dir, day, occ_df["filename"], scope=scope, alt_cap=alt_cap,
        cache_dir=cache_dir, use_cache=use_cache)
    out = occ_df.copy()
    out["min_tan_alt_km"] = out["filename"].map(min_map).astype(float)
    return out


def availability_minima_windows(
    times: pd.Series, day: pd.Timestamp, window_hours: float = 1.0,
    step_minutes: float = 5.0, min_sep_minutes: float = 60.0,
    prominence: float = 3.0,
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], pd.DatetimeIndex,
           np.ndarray, np.ndarray]:
    """
    Partition a day into time windows bounded by the local minima of the rolling
    `window_hours` occultation count.

    The count is evaluated on a `step_minutes` grid (rolling_window_count); minima
    are the peaks of its negation, spaced ≥ `min_sep_minutes` apart and with the
    given `prominence`, so spurious adjacent dips are ignored while the true
    ~90-min availability lulls are picked out. Day start/end are added as the
    outer edges.

    Returns (windows, grid, counts, minima_idx) where windows is a list of
    (lo, hi) Timestamp pairs.
    """
    day0 = pd.Timestamp(day.year, day.month, day.day)
    grid, counts = rolling_window_count(times, day, window_hours=window_hours,
                                        step_minutes=step_minutes)
    distance = max(1, int(round(min_sep_minutes / step_minutes)))
    minima_idx, _props = find_peaks(-counts.astype(float), distance=distance,
                                    prominence=prominence)

    edges = [day0] + [grid[m] for m in minima_idx] + [day0 + pd.Timedelta(days=1)]
    edges = sorted(set(pd.Timestamp(e) for e in edges))
    windows = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    return windows, grid, counts, minima_idx


def _shade_by_order(idx: int, n: int, cmap):
    """Constellation shade deepening with occultation order (matches
    plotIonosphereTomography): index → [0.40, 0.90], singletons → 0.70."""
    t = (0.40 + 0.50 * (idx / max(n - 1, 1))) if n > 1 else 0.70
    return cmap(t)


def plot_window_tec_profiles(
    prof_map: dict, occ_df: pd.DataFrame, day: pd.Timestamp,
    lo: pd.Timestamp, hi: pd.Timestamp, roi_only: bool = True,
    alt_cap: float = 700.0, alt_min: float = 400.0,
    save_path: Path | None = None,
) -> plt.Figure | None:
    """
    2×2 constellation grid of TEC-vs-tangent-altitude occultation profiles for the
    occultations whose TEC-max time falls in [lo, hi). One panel per GNSS
    constellation (GPS / GLONASS / Galileo / BeiDou); within a panel each arc is a
    line coloured by its constellation family, shade deepening with time through
    the window. Returns the figure, or None if the window has no usable profiles.
    """
    cfg_all = _constellation_config()
    sub = occ_df[(occ_df["tecmax_time"] >= lo) & (occ_df["tecmax_time"] < hi)].copy()
    sub = sub.sort_values("tecmax_time").reset_index(drop=True)
    sub = sub[sub["filename"].isin(prof_map.keys())]
    if sub.empty:
        return None

    sub["const"] = [_parse_prn(f)[0] for f in sub["filename"]]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10), sharex=False, sharey=True)
    ax_by_const = {c: axes[r][cc] for c, (r, cc) in _CONST_PANEL.items()}

    y_top = 0.0
    for const, (row, col) in _CONST_PANEL.items():
        ax = ax_by_const[const]
        cfg = cfg_all.get(const, _CONST_FALLBACK)
        cmap = matplotlib.colormaps.get_cmap(cfg.get("cmap", "Greys"))
        grp = sub[sub["const"] == const].reset_index(drop=True)
        n = len(grp)
        legend_handles = []
        for idx, r in grp.iterrows():
            rec = prof_map[r["filename"]]
            color = _shade_by_order(idx, n, cmap)
            ax.plot(rec["tec"], rec["tan_alt"], color=color, lw=1.0, alpha=0.85)
            y_top = max(y_top, float(np.nanmax(rec["tan_alt"])))
            if len(legend_handles) < 8:
                prn = _parse_prn(r["filename"])
                legend_handles.append(Line2D(
                    [0], [0], color=color, lw=1.6,
                    label=f"{prn}  {pd.Timestamp(r['tecmax_time']):%H:%M}"))

        ax.set_title(f"{cfg.get('name', const)}  (n={n})", fontsize=10,
                     color=cfg.get("title_color", "black"), fontweight="bold")
        ax.grid(True, alpha=0.3, ls=":")
        if n == 0:
            ax.text(0.5, 0.5, "No occultations", transform=ax.transAxes,
                    ha="center", va="center", color="lightgray", fontsize=11,
                    style="italic")
        else:
            extra = f"  (+{n - len(legend_handles)} more)" if n > len(legend_handles) else ""
            ax.legend(handles=legend_handles, fontsize=6.5, loc="upper right",
                      framealpha=0.85, title=(f"PRN  @TEC-max{extra}" if extra else "PRN  @TEC-max"),
                      title_fontsize=6.5)
        if col == 0:
            ax.set_ylabel("Tangent altitude (km)")
        if row == 1:
            ax.set_xlabel("TEC (TECU)")

    for ax in ax_by_const.values():
        ax.set_ylim(0, max(y_top + 30.0, 300.0))
        ax.set_xlim(left=0)
        ax.axhline(alt_min, color="0.5", ls="--", lw=0.8, alpha=0.7, zorder=0)

    n_tot = len(sub)
    roi_note = f"within {ISR_ROI_MAX_KM:.0f} km of ESR/TRO" if roi_only else "all tangent points"
    fig.suptitle(
        f"GNSS-RO TEC vs tangent altitude — {pd.Timestamp(day).date()}   "
        f"{lo:%H:%M}–{hi:%H:%M} UTC\n{n_tot} occultation(s) · {roi_note} · "
        f"reaching ≤ {alt_min:g} km (dashed) · "
        f"shade lightens→darkens with time through the window",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved → {save_path.name}")
    return fig


def _plot_windows_overview(
    day: pd.Timestamp, grid: pd.DatetimeIndex, counts: np.ndarray,
    minima_idx: np.ndarray, windows, window_hours: float,
    roi_only: bool, save_path: Path,
) -> None:
    """Context strip: the rolling count with its local minima (window edges)
    marked and alternating windows shaded, so the partitioning is auditable."""
    day0 = pd.Timestamp(day.year, day.month, day.day)
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.fill_between(grid, counts, step="mid", alpha=0.35, color="seagreen")
    ax.step(grid, counts, where="mid", color="seagreen", lw=1.2)
    for k, (lo, hi) in enumerate(windows):
        if k % 2 == 0:
            ax.axvspan(lo, hi, color="0.85", alpha=0.35, zorder=0)
    for m in minima_idx:
        ax.axvline(grid[m], color="crimson", ls="--", lw=1.0, alpha=0.8)
    ax.set_xlim(day0, day0 + pd.Timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f"# occ / {window_hours:g} h\nrolling window")
    ax.set_xlabel("Time (UTC)")
    roi_note = "in-ROI (≤ ESR/TRO gate)" if roi_only else "all"
    ax.set_title(
        f"Availability-minima window partition — {day0.date()} · {roi_note} · "
        f"{len(windows)} windows (dashed = minima)", fontsize=11)
    ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved window-overview → {save_path}")


def run_tec_profile_windows(
    day, roi_only: bool = True, window_hours: float = 1.0,
    step_minutes: float = 5.0, min_sep_minutes: float = 60.0,
    prominence: float = 3.0, alt_cap: float = 700.0, alt_min: float = 400.0,
    save_dir: Path | None = None, cache_dir: Path | None = None,
    profile_cache_dir: Path | None = None, use_cache: bool = True,
    show: bool = False,
) -> list[Path]:
    """
    End-to-end: read a day's occultations, partition it at the rolling-count
    local minima (~every 90 min), and write one 2×2 constellation
    altitude-vs-TEC figure per window plus a window-partition overview.

    `roi_only` (default True) restricts both the window-defining count and the
    plotted profiles to occultations within ISR_ROI_MAX_KM of ESR/TRO. Only
    passes whose tangent point descends to at least `alt_min` km are plotted.
    Returns the list of written figure paths.
    """
    day = pd.Timestamp(day)
    if save_dir is None:
        save_dir = FIGURES_DIR / "TEC_Profiles" / day.strftime("%Y%m%d")
    save_dir = Path(save_dir)

    podtc_dir = _resolve_podtc_dir(day)
    if podtc_dir is None:
        print(f"[skip] No podTc2 directory with data for {day.date()} under {PODTC_BASE}")
        return []

    occ = read_day_occultations_cached(podtc_dir, day, cache_dir=cache_dir,
                                       use_cache=use_cache)
    if occ.empty:
        print(f"[skip] No occultations for {day.date()}.")
        return []

    sel = occ[occ["in_roi"]].copy() if roi_only else occ.copy()
    if sel.empty:
        print(f"[skip] No {'in-ROI ' if roi_only else ''}occultations for {day.date()}.")
        return []

    windows, grid, counts, minima_idx = availability_minima_windows(
        sel["tecmax_time"], day, window_hours=window_hours, step_minutes=step_minutes,
        min_sep_minutes=min_sep_minutes, prominence=prominence)
    print(f"  {len(minima_idx)} availability minima → {len(windows)} windows.")

    prof_map = read_day_profiles_cached(
        podtc_dir, day, sel, roi_only=roi_only, alt_cap=alt_cap, alt_min=alt_min,
        cache_dir=profile_cache_dir, use_cache=use_cache)

    _plot_windows_overview(day, grid, counts, minima_idx, windows, window_hours,
                           roi_only, save_dir / "windows_overview.png")

    saved: list[Path] = []
    for lo, hi in windows:
        save_path = save_dir / f"tec_profiles_{lo:%H%M}-{hi:%H%M}.png"
        fig = plot_window_tec_profiles(prof_map, sel, day, lo, hi,
                                       roi_only=roi_only, alt_cap=alt_cap,
                                       alt_min=alt_min, save_path=save_path)
        if fig is None:
            continue
        saved.append(save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    print(f"  Wrote {len(saved)} window figure(s) → {save_dir}")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Day → directory resolution + CLI
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_podtc_dir(day: pd.Timestamp) -> Path | None:
    """Locate the podTc2 directory for a day, following demo_isr_da_comparison's
    `{year}.{doy:03d}` convention (with a 4-digit fallback for odd names)."""
    doy = day.timetuple().tm_yday
    for name in (f"{day.year}.{doy:03d}", f"{day.year}.{doy:04d}"):
        cand = PODTC_BASE / name
        if cand.is_dir() and any(cand.glob(f"*{PODTC_SUFFIX}")):
            return cand
    return None


def run_day(day, window_hours: float = 1.0, roi_thresholds_km=ROI_THRESHOLDS_KM,
            alt_cap: float = 700.0, alt_min: float | None = None,
            save_dir: Path | None = None, minalt_cache_dir: Path | None = None,
            use_cache: bool = True, show: bool = True):
    """
    Process one day end-to-end: locate its podTc2 directory, read occultations,
    and draw the availability figure.

    `day` may be a pd.Timestamp or anything pd.Timestamp() accepts (e.g. the
    string "2024-10-10"), so this is convenient to call directly from Spyder.
    `roi_thresholds_km` is passed straight through to
    plot_occultation_availability (see its docstring) -- any number of
    thresholds, green shades are derived to match.

    When `alt_min` is set, the in-ROI occultations are gated to those whose limb
    sounding descends to at least `alt_min` km (using the same accept test, at
    the same `alt_cap`, as the TEC-profile panels) so the figure's in-ROI count
    matches the number of plotted profiles in run_tec_profile_windows.
    Returns (df, fig) — or (df, None) / (None, None) when there is nothing to
    plot — so results can be inspected interactively.
    """
    day = pd.Timestamp(day)
    if save_dir is None:
        save_dir = FIGURES_DIR
    save_dir = Path(save_dir)

    podtc_dir = _resolve_podtc_dir(day)
    if podtc_dir is None:
        print(f"[skip] No podTc2 directory with data for {day.date()} under {PODTC_BASE}")
        return None, None

    df = read_day_occultations(podtc_dir)
    if df.empty or not df["in_roi"].any():
        print(f"[skip] No in-ROI occultations for {day.date()}.")
        return df, None

    if alt_min is not None:
        roi_sub = attach_min_tan_alt(
            df[df["in_roi"]], podtc_dir, day, scope="roi", alt_cap=alt_cap,
            cache_dir=minalt_cache_dir, use_cache=use_cache)
        keep_files = set(roi_sub.loc[roi_sub["min_tan_alt_km"] <= alt_min, "filename"])
        before = int(df["in_roi"].sum())
        df = df[(~df["in_roi"]) | df["filename"].isin(keep_files)].reset_index(drop=True)
        print(f"  alt_min gate: {len(keep_files)}/{before} in-ROI occultations "
              f"reach ≤ {alt_min:g} km (alt_cap {alt_cap:g} km).")
        if not df["in_roi"].any():
            print(f"[skip] No in-ROI occultations reach ≤ {alt_min:g} km for {day.date()}.")
            return df, None

    save_path = save_dir / f"occultation_availability_{day.strftime('%Y%m%d')}.png"
    fig = plot_occultation_availability(df, day, window_hours=window_hours,
                                        roi_thresholds_km=roi_thresholds_km,
                                        alt_min=alt_min, save_path=save_path)
    if show:
        plt.show()
    return df, fig


# ─────────────────────────────────────────────────────────────────────────────
# Multi-day batch processing + cross-day violin summary
# ─────────────────────────────────────────────────────────────────────────────

_DAY_DIR_RE = re.compile(r"^(\d{4})\.(\d{3,4})$")
DEFAULT_CACHE_DIR = Path(__file__).parent / "Data" / "Occultation_Cache"


def list_available_days() -> list[pd.Timestamp]:
    """
    Enumerate every day under PODTC_BASE that actually has podTc2 files, by
    parsing `{year}.{doy[3 or 4 digits]}` directory names (e.g. 2025.254,
    2025.0254). Non-day entries (e.g. STORM_DATA) are silently skipped.
    """
    days = []
    for cand in sorted(PODTC_BASE.iterdir()):
        if not cand.is_dir():
            continue
        m = _DAY_DIR_RE.match(cand.name)
        if not m:
            continue
        if not any(cand.glob(f"*{PODTC_SUFFIX}")):
            continue
        year, doy = int(m.group(1)), int(m.group(2))
        days.append(pd.Timestamp(year, 1, 1) + pd.Timedelta(days=doy - 1))
    return sorted(set(days))


def read_day_occultations_cached(
    podtc_dir: Path,
    day: pd.Timestamp,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Same as read_day_occultations, but memoised to a pickle per day (scanning
    ~7000 files costs ~12s, and PODTC_BASE has ~140 day directories, so this
    keeps repeated/iterative batch runs fast).
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_path = cache_dir / f"{day.strftime('%Y%m%d')}.pkl"

    if use_cache and cache_path.exists():
        return pd.read_pickle(cache_path)

    df = read_day_occultations(podtc_dir)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
    return df


def run_all_days(
    days=None,
    window_hours: float = 1.0,
    roi_thresholds_km=ROI_THRESHOLDS_KM,
    alt_cap: float = 700.0,
    alt_min: float | None = None,
    save_dir: Path | None = None,
    cache_dir: Path | None = None,
    minalt_cache_dir: Path | None = None,
    use_cache: bool = True,
    make_figures: bool = True,
    show: bool = False,
) -> dict[pd.Timestamp, np.ndarray]:
    """
    Batch-process every available podTc2 day (or a caller-supplied subset),
    saving each day's full 2x2 availability figure and collecting the 1-hour
    rolling-window in-ROI occultation-count series needed for the cross-day
    violin summary. Figures are closed right after saving so memory stays flat
    across the full PODTC_BASE history (~140 days).

    `roi_thresholds_km` is used both for each day's figure and to pick the
    rolling-count series returned here: the *largest* threshold (the full ISR
    ROI, ISR_ROI_MAX_KM by default) is what feeds the violin plot.

    When `alt_min` is set, each day's in-ROI occultations are gated to those
    whose limb sounding descends to at least `alt_min` km (the same accept test,
    at the same `alt_cap`, used by run_tec_profile_windows), so both the daily
    figures and the returned counts (and therefore the violin summary) reflect
    only occultations that would actually yield a usable TEC profile.

    Returns daily_counts: {day: per-minute rolling-window count array}, one
    entry per day that had at least one (gated) in-ROI occultation.
    """
    if days is None:
        days = list_available_days()
    else:
        days = sorted(pd.Timestamp(d) for d in days)

    if save_dir is None:
        save_dir = FIGURES_DIR
    save_dir = Path(save_dir)
    outer_threshold = max(roi_thresholds_km)

    daily_counts: dict[pd.Timestamp, np.ndarray] = {}
    n = len(days)
    for i, day in enumerate(days, 1):
        print(f"[{i}/{n}] {day.date()}")
        podtc_dir = _resolve_podtc_dir(day)
        if podtc_dir is None:
            print(f"  [skip] no podTc2 directory for {day.date()}")
            continue

        df = read_day_occultations_cached(podtc_dir, day, cache_dir=cache_dir,
                                           use_cache=use_cache)
        if df.empty or not df["in_roi"].any():
            print(f"  [skip] no in-ROI occultations for {day.date()}")
            continue

        if alt_min is not None:
            roi_sub = attach_min_tan_alt(
                df[df["in_roi"]], podtc_dir, day, scope="roi", alt_cap=alt_cap,
                cache_dir=minalt_cache_dir, use_cache=use_cache)
            keep_files = set(roi_sub.loc[roi_sub["min_tan_alt_km"] <= alt_min, "filename"])
            before = int(df["in_roi"].sum())
            df = df[(~df["in_roi"]) | df["filename"].isin(keep_files)].reset_index(drop=True)
            print(f"  alt_min gate: {len(keep_files)}/{before} in-ROI occultations "
                  f"reach ≤ {alt_min:g} km (alt_cap {alt_cap:g} km).")
            if not df["in_roi"].any():
                print(f"  [skip] no in-ROI occultations reach ≤ {alt_min:g} km for {day.date()}")
                continue

        roi = df[df["nearest_km"] <= outer_threshold]
        _, counts = rolling_window_count(roi["tecmax_time"], day, window_hours)
        daily_counts[day] = counts

        if make_figures:
            save_path = save_dir / f"occultation_availability_{day.strftime('%Y%m%d')}.png"
            fig = plot_occultation_availability(df, day, window_hours=window_hours,
                                                roi_thresholds_km=roi_thresholds_km,
                                                alt_min=alt_min, save_path=save_path)
            if show:
                plt.show()
            plt.close(fig)

    return daily_counts


# Quarter row boundaries for the violin summary: (doy_lo, doy_hi, label).
# ~91-93 days each; DOY 366 covers the Dec 31 of leap years too.
QUARTER_BOUNDS = (
    (1,   91,  "Jan – Mar"),
    (92,  182, "Apr – Jun"),
    (183, 273, "Jul – Sep"),
    (274, 366, "Oct – Dec"),
)


def isr_availability_by_day(edps: list[dict] | None = None) -> dict:
    """
    Map each date with ISR data to the set of {"ESR", "TRO"} sites that have
    at least one EDP profile that day (JRO is excluded -- this script only
    covers the two sites in ISR_SITES). Backed by demo_esr_isr's own on-disk
    EDP cache (esr_edp_cache.pkl), so repeat calls are fast.
    """
    if edps is None:
        edps = load_edps()
    avail: dict = {}
    for e in edps:
        inst = _identify_instrument(e["lat"])
        if inst not in ISR_SITES:
            continue
        day = e["time"].date()
        avail.setdefault(day, set()).add(inst)
    return avail


def plot_availability_violin(
    daily_counts: dict[pd.Timestamp, np.ndarray],
    isr_availability: dict | None = None,
    threshold_km: float = ISR_ROI_MAX_KM,
    window_hours: float = 1.0,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    One violin per day summarising the distribution (across that day's
    per-minute grid) of in-ROI occultation counts inside a `window_hours`
    rolling window. Split into 4 rows, one per calendar quarter (see
    QUARTER_BOUNDS), so each row only has to span ~91-93 days of x-axis and
    the violins render noticeably wider than a single 365-day-wide row.

    X-axis is day-of-year; since PODTC_BASE spans multiple years
    (2024/2025/2026), each year gets its own colour and a small horizontal
    offset so same-DOY violins from different years don't overlap.

    A thin strip beneath each quarter row marks, per day, whether ISR EDP
    data exists at ESR and/or TRO (isr_availability_by_day() is used if
    `isr_availability` isn't supplied) -- this shows which podTc-covered days
    also have ground-truth ISR data to compare against.
    """
    days = sorted(daily_counts.keys())
    years = sorted({d.year for d in days})
    year_offset = {yr: (i - (len(years) - 1) / 2) * 0.28 for i, yr in enumerate(years)}
    cmap = plt.get_cmap("tab10")
    year_color = {yr: cmap(i % 10) for i, yr in enumerate(years)}

    if isr_availability is None:
        isr_availability = isr_availability_by_day()

    global_max = max((float(np.max(c)) for c in daily_counts.values() if c.size), default=1.0)

    fig = plt.figure(figsize=(16.0, 4.6 * len(QUARTER_BOUNDS)))
    gs = GridSpec(len(QUARTER_BOUNDS), 1, figure=fig, hspace=0.55)

    n_plotted = 0
    for row, (doy_lo, doy_hi, label) in enumerate(QUARTER_BOUNDS):
        inner = gs[row].subgridspec(2, 1, height_ratios=[4.0, 0.9], hspace=0.08)
        ax = fig.add_subplot(inner[0])
        ax_isr = fig.add_subplot(inner[1], sharex=ax)

        positions, datasets, colors = [], [], []
        for d in days:
            doy = d.timetuple().tm_yday
            if not (doy_lo <= doy <= doy_hi):
                continue
            counts = daily_counts[d]
            if counts.size == 0 or not np.any(counts):
                continue
            positions.append(doy + year_offset[d.year])
            datasets.append(counts.astype(float))
            colors.append(year_color[d.year])

        if datasets:
            parts = ax.violinplot(datasets, positions=positions, widths=2.2,
                                  showmedians=True, showextrema=True)
            for body, color in zip(parts["bodies"], colors):
                body.set_facecolor(color)
                body.set_edgecolor("k")
                body.set_linewidth(0.4)
                body.set_alpha(0.65)
            for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                if key in parts:
                    parts[key].set_edgecolor("0.3")
                    parts[key].set_linewidth(0.7)
            n_plotted += len(datasets)

        ax.set_xlim(doy_lo - 2, doy_hi + 2)
        ax.set_ylim(0, global_max * 1.05)
        ax.set_ylabel(f"# occ. / {window_hours:g} h\nwindow", fontsize=8.5)
        ax.set_title(label, loc="left", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), visible=False)

        # ── ISR ESR/TRO data-availability strip for this quarter ───────────
        esr_x = [doy + year_offset[d.year] for d in days
                 if doy_lo <= (doy := d.timetuple().tm_yday) <= doy_hi
                 and "ESR" in isr_availability.get(d.date(), ())]
        tro_x = [doy + year_offset[d.year] for d in days
                 if doy_lo <= (doy := d.timetuple().tm_yday) <= doy_hi
                 and "TRO" in isr_availability.get(d.date(), ())]
        if esr_x:
            ax_isr.scatter(esr_x, [1] * len(esr_x), marker="o", s=24,
                           color="0.15", edgecolor="none")
        if tro_x:
            ax_isr.scatter(tro_x, [0] * len(tro_x), marker="^", s=24,
                           color="0.55", edgecolor="none")
        ax_isr.set_xlim(ax.get_xlim())
        ax_isr.set_ylim(-0.7, 1.7)
        ax_isr.set_yticks([0, 1])
        ax_isr.set_yticklabels(["TRO", "ESR"], fontsize=7)
        ax_isr.grid(True, axis="x", alpha=0.2)
        if row == len(QUARTER_BOUNDS) - 1:
            ax_isr.set_xlabel("Day of year")
        else:
            plt.setp(ax_isr.get_xticklabels(), visible=False)

    fig.suptitle(
        f"Occultation availability distribution across {n_plotted} days "
        f"({', '.join(str(y) for y in years)}) -- within {threshold_km:.0f} km of ESR/TRO",
        fontsize=13,
    )

    year_handles = [
        Line2D([], [], marker="s", ls="none", color=year_color[yr],
               markeredgecolor="k", markeredgewidth=0.3, markersize=9,
               label=f"{yr} (n={sum(1 for d in days if d.year == yr and daily_counts[d].any())})")
        for yr in years
    ]
    isr_handles = [
        Line2D([], [], marker="o", ls="none", color="0.15", markersize=7, label="ESR data available"),
        Line2D([], [], marker="^", ls="none", color="0.55", markersize=7, label="TRO data available"),
    ]
    fig.legend(handles=year_handles + isr_handles, loc="upper right",
              bbox_to_anchor=(0.99, 0.995), fontsize=8, framealpha=0.9, ncol=1)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure → {save_path}")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot rolling-window GNSS-RO occultation availability and "
                    "ISR proximity for a single day (TEC-max timing), or batch "
                    "process every available day and summarise as a violin plot.")
    parser.add_argument("--date", type=str, default=None,
                        help="Day to process, YYYY-MM-DD (e.g. 2024-10-10).")
    parser.add_argument("--year", type=int, default=None, help="Year (with --doy).")
    parser.add_argument("--doy", type=int, default=None,
                        help="Day-of-year (with --year).")
    parser.add_argument("--all-days", action="store_true",
                        help="Batch-process every day found under PODTC_BASE, "
                             "save each day's figure, and produce the cross-day "
                             "violin summary (ignores --date/--year/--doy).")
    parser.add_argument("--no-figures", action="store_true",
                        help="With --all-days, skip saving each day's 2x2 figure "
                             "and only compute the violin summary (faster).")
    parser.add_argument("--no-cache", action="store_true",
                        help="With --all-days, ignore/overwrite the per-day pickle cache.")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR),
                        help="Directory for the per-day occultation cache "
                             f"(default {DEFAULT_CACHE_DIR}).")
    parser.add_argument("--window-hours", type=float, default=1.0,
                        help="Rolling window width in hours (default 1.0).")
    parser.add_argument("--roi-thresholds-km", type=str, default=None,
                        help="Comma-separated proximity thresholds in km for "
                             "the rolling-count/ROI-ring/histogram panels "
                             f"(default {','.join(str(int(t)) for t in ROI_THRESHOLDS_KM)}).")
    parser.add_argument("--save-dir", type=str,
                        default=str(FIGURES_DIR),
                        help="Directory to write per-day figures into.")
    parser.add_argument("--movie", action="store_true",
                        help="Instead of the 2x2 figure, render an animated "
                             "GIF/MP4 of the day's LEO ground tracks + occultation "
                             "tangent points (requires --date or --year/--doy).")
    parser.add_argument("--step-minutes", type=float, default=5.0,
                        help="With --movie: frame spacing in minutes (default 5).")
    parser.add_argument("--trail-minutes", type=float, default=25.0,
                        help="With --movie: LEO trail length in minutes (default 25).")
    parser.add_argument("--decimate-s", type=float, default=15.0,
                        help="With --movie: ground-track sample spacing in seconds "
                             "(default 15).")
    parser.add_argument("--fps", type=int, default=12,
                        help="With --movie: output frames per second (default 12).")
    parser.add_argument("--tec-profiles", action="store_true",
                        help="Instead of the 2x2 summary, write 2x2-per-constellation "
                             "TEC-vs-tangent-altitude occultation profiles, one figure "
                             "per time window bounded by the rolling-count local "
                             "minima (~every 90 min). Requires --date or --year/--doy.")
    parser.add_argument("--all-occ", action="store_true",
                        help="With --tec-profiles: use ALL occultations (not just "
                             "those within the ESR/TRO ROI) for both the window "
                             "minima and the plotted profiles.")
    parser.add_argument("--min-sep-minutes", type=float, default=60.0,
                        help="With --tec-profiles: minimum spacing between detected "
                             "availability minima, in minutes (default 60).")
    parser.add_argument("--prominence", type=float, default=3.0,
                        help="With --tec-profiles: minimum prominence of a rolling-count "
                             "minimum for it to bound a window (default 3).")
    parser.add_argument("--alt-cap", type=float, default=700.0,
                        help="Max tangent altitude (km) kept in each limb profile "
                             "and used as the accept-gate ceiling (default 700). "
                             "Applies to --tec-profiles, the availability figure, "
                             "and --movie.")
    parser.add_argument("--alt-min", type=float, default=400.0,
                        help="A pass is kept only if its tangent point descends to "
                             "at least this altitude (km); passes that only graze "
                             "above it are dropped (default 400). Applied to "
                             "--tec-profiles, the availability figure, --movie, and "
                             "--all-days so their RO counts match; use --no-alt-min "
                             "to disable the gate.")
    parser.add_argument("--no-alt-min", action="store_true",
                        help="Disable the alt_min gate on the availability figure, "
                             "--movie, and --all-days (they then count all in-ROI "
                             "occultations). Does not affect --tec-profiles.")
    parser.add_argument("--show", action="store_true", help="Display the figure(s).")
    args = parser.parse_args()

    if args.roi_thresholds_km is not None:
        roi_thresholds_km = tuple(float(x) for x in args.roi_thresholds_km.split(","))
    else:
        roi_thresholds_km = ROI_THRESHOLDS_KM

    if not args.show:
        matplotlib.use("Agg")

    gate_alt_min = None if args.no_alt_min else args.alt_min

    if args.all_days:
        daily_counts = run_all_days(
            window_hours=args.window_hours, roi_thresholds_km=roi_thresholds_km,
            alt_cap=args.alt_cap, alt_min=gate_alt_min,
            save_dir=Path(args.save_dir), cache_dir=Path(args.cache_dir),
            use_cache=not args.no_cache, make_figures=not args.no_figures,
            show=args.show,
        )
        violin_path = Path(args.save_dir) / "occultation_availability_violin.png"
        plot_availability_violin(daily_counts, threshold_km=max(roi_thresholds_km),
                                 window_hours=args.window_hours, save_path=violin_path)
        if args.show:
            plt.show()
        return

    if args.date is not None:
        day = pd.Timestamp(args.date)
    elif args.year is not None and args.doy is not None:
        day = pd.Timestamp(args.year, 1, 1) + pd.Timedelta(days=args.doy - 1)
    else:
        parser.error("Provide either --date YYYY-MM-DD, both --year and --doy, or --all-days.")

    if args.movie:
        animate_ground_tracks(
            day, step_minutes=args.step_minutes, trail_minutes=args.trail_minutes,
            window_hours=args.window_hours, decimate_s=args.decimate_s, fps=args.fps,
            alt_cap=args.alt_cap, alt_min=gate_alt_min,
            use_cache=not args.no_cache,
            save_path=Path(args.save_dir) / "Movies" /
                      f"ground_tracks_{day.strftime('%Y%m%d')}.gif")
        return

    if args.tec_profiles:
        run_tec_profile_windows(
            day, roi_only=not args.all_occ, window_hours=args.window_hours,
            step_minutes=args.step_minutes, min_sep_minutes=args.min_sep_minutes,
            prominence=args.prominence, alt_cap=args.alt_cap, alt_min=args.alt_min,
            save_dir=Path(args.save_dir) / "TEC_Profiles" / day.strftime("%Y%m%d"),
            use_cache=not args.no_cache, show=args.show)
        return

    run_day(day, window_hours=args.window_hours, roi_thresholds_km=roi_thresholds_km,
            alt_cap=args.alt_cap, alt_min=gate_alt_min, save_dir=Path(args.save_dir),
            use_cache=not args.no_cache, show=args.show)


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────
#
# Two ways to run this file:
#
#   • Command line, single day:  python demo_occultation_availability.py --date 2024-10-10
#   • Command line, all days:    python demo_occultation_availability.py --all-days
#
#   • Spyder / IDE:   just press F5 (Run file), or put the cursor in the
#                     "% Spyder" cell below and press Ctrl+Enter. Edit the DATE
#                     / WINDOW_HOURS values in that cell first. The returned
#                     `df` and `fig` are left in the console namespace so you
#                     can inspect them (e.g. df[df.in_roi], fig.axes). Set
#                     RUN_ALL_DAYS = True to batch-process every day under
#                     PODTC_BASE and get the cross-day violin summary instead
#                     (`daily_counts` and `violin_fig` are left in the
#                     namespace). The per-day scan is cached under
#                     Data/Occultation_Cache/ so repeat runs are fast.
#
# When launched with command-line flags (sys.argv has extras) the CLI parser
# runs; otherwise the interactive defaults below are used.

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # %% Spyder — edit these, then Run file (F5) or run this cell (Ctrl+Enter)
        DATE         = "2025-9-22"   # day to plot, "YYYY-MM-DD"
        WINDOW_HOURS = 1.0            # rolling-window width in hours
        ROI_THRESHOLDS = (2500.0, 1500.0, 500.0)   # any # of km thresholds
        SAVE_DIR     = None           # None → FIGURES_DIR (Occultation_Availability[/North_America])

        RUN_ALL_DAYS = False   # True → batch-process every PODTC_BASE day + violin plot
        MAKE_FIGURES = True    # with RUN_ALL_DAYS: also save each day's 2x2 figure
        USE_CACHE    = True    # with RUN_ALL_DAYS: memoise per-day scans to disk

        MAKE_MOVIE   = True   # True → animate LEO ground tracks + tangent points for DATE
        STEP_MINUTES  = 5.0    # movie/profile frame/grid spacing (min)
        TRAIL_MINUTES = 25.0   # movie LEO trail length (min)
        DECIMATE_S    = 15.0   # movie ground-track sample spacing (s)
        FPS           = 12     # movie frames per second

        # 2×2-per-constellation TEC-vs-tangent-altitude profiles, one figure per
        # time window bounded by the rolling-count local minima (~every 90 min).
        MAKE_TEC_PROFILES = False    # True → write the per-window TEC/altitude figures
        PROFILE_ROI_ONLY  = True    # False → use every occultation, not just in-ROI
        MIN_SEP_MINUTES   = 60.0    # min spacing between detected availability minima
        PROMINENCE        = 3.0     # min prominence of a count minimum to bound a window
        ALT_CAP           = 900.0   # max tangent altitude (km) kept per profile
        ALT_MIN           = 400.0   # keep only passes descending to ≤ this altitude (km)

        if MAKE_TEC_PROFILES:
            profile_paths = run_tec_profile_windows(
                DATE, roi_only=PROFILE_ROI_ONLY, window_hours=WINDOW_HOURS,
                step_minutes=STEP_MINUTES, min_sep_minutes=MIN_SEP_MINUTES,
                prominence=PROMINENCE, alt_cap=ALT_CAP, alt_min=ALT_MIN,
                use_cache=USE_CACHE, show=False)
        elif MAKE_MOVIE:
            movie_path = animate_ground_tracks(
                DATE, step_minutes=STEP_MINUTES, trail_minutes=TRAIL_MINUTES,
                window_hours=WINDOW_HOURS, decimate_s=DECIMATE_S, fps=FPS,
                alt_cap=ALT_CAP, alt_min=ALT_MIN, use_cache=USE_CACHE)
        elif RUN_ALL_DAYS:
            daily_counts = run_all_days(window_hours=WINDOW_HOURS,
                                        roi_thresholds_km=ROI_THRESHOLDS,
                                        alt_cap=ALT_CAP, alt_min=ALT_MIN,
                                        save_dir=SAVE_DIR, use_cache=USE_CACHE,
                                        make_figures=MAKE_FIGURES, show=False)
            violin_dir = Path(SAVE_DIR) if SAVE_DIR else FIGURES_DIR
            violin_fig = plot_availability_violin(
                daily_counts, threshold_km=max(ROI_THRESHOLDS),
                window_hours=WINDOW_HOURS,
                save_path=violin_dir / "occultation_availability_violin.png")
            plt.show()
        else:
            df, fig = run_day(DATE, window_hours=WINDOW_HOURS,
                              roi_thresholds_km=ROI_THRESHOLDS,
                              alt_cap=ALT_CAP, alt_min=ALT_MIN,
                              save_dir=SAVE_DIR, use_cache=USE_CACHE, show=True)
