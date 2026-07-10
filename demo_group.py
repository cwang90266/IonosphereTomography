#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
demo_group.py — Grouped occultation Kalman update demonstration.

Reads all podTc2 files from a day directory (GN04, GN05, etc.) and:
  1. Parses metadata to locate each occultation on the globe.
  2. Groups occultations into 30-minute time windows, then into geographic
     regions:
       - Polar caps       : |lat| > 65°  (one north cap, one south cap)
       - Mid-latitude bins: 20° lat × 50° lon cells
  3. Processes each group with a *joint* Kalman Filter update — all
     occultation observation operators are stacked into a single H matrix
     so every ray in the group informs the same posterior EDP field.
  4. Generates two diagnostic figures:
       • Globe map  — all occultation TEC-max tangent points, colour-coded
                      by the number of occultations in their group.
       • TEC panels — measured / prior / posterior TEC profiles for every
                      group, individual lines colour-coded by occultation
                      index within the group.

Run from the project root:
    python demo_group.py

Or call demo_group_main() directly from another script.
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EDPSamples" / "Locate in mesh" / "outputs"))
sys.path.insert(0, str(ROOT / "iri2020_new" / "src"))

import numpy as np
import pandas as pd
import matplotlib

# NumPy >=2.0 removed np.trapz (renamed to np.trapezoid); NumPy <2.0 doesn't
# have np.trapezoid yet. Resolve once at import time so call sites work
# under either version.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
matplotlib.use("Agg")                    # non-interactive for batch runs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import time
import gc

import netCDF4                           # used by the lightweight metadata scan
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj
from scipy.spatial import ConvexHull

# ── Project imports ───────────────────────────────────────────────────────────
from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
from TEC_model.conPhs_file_processing import load_conPhs
from EDPSamples.edp_samples import EDPSamples
from Ionosphere_Tomography_Inverter.Ionophy_Tomography_Inverter import (
    Ionosphere_Tomography_Inverter,
)
# Re-use the global EDP builder and F2-peak extractor already in demo.py.
from demo import build_daily_global_edps, extract_robust_f2_peak

# Plotting functions consolidated into plotIonosphereTomography.py. Imported
# back here (top-level, not deferred) because process_group() calls these
# directly and several other modules monkey-patch _demo_group._plot_group at
# runtime, which requires this to remain a persistent module attribute.
from plotIonosphereTomography import (
    _draw_roi_boundary, _roi_centre_idx,
    _plot_altitude_slices, _plot_covariance_panels_tagged,
    _plot_group,
)

# ─────────────────────────────────────────────────────────────────────────────
# Grouping parameters (adjust as needed)
# ─────────────────────────────────────────────────────────────────────────────

POLAR_LAT_THRESHOLD = 60.0   # |lat| above this → polar cap
DLAT_MID            = 30.0   # mid-latitude bin height (degrees)
DLON_MID            = 60.0   # mid-latitude bin width  (degrees)
WINDOW_MINUTES      = 30     # time-bin width in minutes
MAX_MESH_VERTICES   = 300    # trim group until EDP mesh has ≤ this many vertices

# ── Gaussian-smoothed prior covariance (set to None to use raw ensemble cov) ──
# (sigma_h_km, sigma_latlon_km): vertical and horizontal smoothing scales.
# Example: (60, 500) gives ~60 km vertical / 500 km horizontal correlation.
GAUSSIAN_COV_SIGMA: tuple | None = (20,500)  # e.g. (60, 500)

# ── Multiplicative lognormal noise injected into the prior EDP ensemble ────────
# Ne_noisy = Ne * exp(ε),  ε ~ N(0, PRIOR_EDP_NOISE_SIGMA²)
# Keeps all densities strictly positive; noise magnitude is proportional to Ne.
# PRIOR_EDP_NOISE_SIGMA ≈ 0.1 adds ~10 % relative spread to each ensemble member,
# enriching the sample covariance and preventing ensemble collapse.
# Set to None (or 0.0) to disable.
PRIOR_EDP_NOISE_SIGMA: float | None = 0.0

# Filename tag appended to every saved figure when noise is active.
# e.g. "_noise0.1" so noisy and clean runs never overwrite each other.
_NOISE_SUFFIX: str = (
    f"_noise{PRIOR_EDP_NOISE_SIGMA}" if PRIOR_EDP_NOISE_SIGMA else ""
)

# Constellation → colour-family and 2×2 TEC-panel position.
# Panel positions: (row, col) with row ∈ {0,1}, col ∈ {0,1}.
#   GPS     → top-left      Blues
#   GLONASS → bottom-left   Purples
#   Galileo → top-right     Oranges
#   BeiDou  → bottom-right  Greens
CONSTELLATION_CONFIG = {
    "G": {"name": "GPS",     "cmap": "Blues",   "panel": (0, 0),
          "title_color": "steelblue"},
    "R": {"name": "GLONASS", "cmap": "Purples", "panel": (1, 0),
          "title_color": "mediumpurple"},
    "E": {"name": "Galileo", "cmap": "Oranges", "panel": (0, 1),
          "title_color": "darkorange"},
    "C": {"name": "BeiDou",  "cmap": "Greens",  "panel": (1, 1),
          "title_color": "seagreen"},
}
_CONST_FALLBACK_CMAP = "Greys"   # used for unknown constellations (J, S, …)


# ─────────────────────────────────────────────────────────────────────────────
# §A  Geographic region helpers
# ─────────────────────────────────────────────────────────────────────────────

def assign_region(lat: float, lon: float) -> str:
    """
    Map a (lat, lon) TEC-max tangent point to a region string key.

    Polar caps return "POLAR_N" or "POLAR_S".
    Mid-latitude bins are labelled by their SW corner, e.g. "LAT+20_LON-050".
    """
    if lat > POLAR_LAT_THRESHOLD:
        return "POLAR_N"
    if lat < -POLAR_LAT_THRESHOLD:
        return "POLAR_S"
    lat_bin = int(np.floor(lat / DLAT_MID) * DLAT_MID)
    lon_bin = int(np.floor(lon / DLON_MID) * DLON_MID)
    return f"LAT{lat_bin:+04d}_LON{lon_bin:+04d}"


def region_bounding_box(region_key: str) -> tuple[float, float, float, float]:
    """
    Return (lat_min, lat_max, lon_min, lon_max) for a region string key.
    Polar caps span all longitudes.
    """
    if region_key == "POLAR_N":
        return (POLAR_LAT_THRESHOLD, 90.0, -180.0, 180.0)
    if region_key == "POLAR_S":
        return (-90.0, -POLAR_LAT_THRESHOLD, -180.0, 180.0)
    parts   = region_key.split("_")
    lat0    = int(parts[0][3:])         # e.g. "+20" → 20
    lon0    = int(parts[1][3:])         # e.g. "-050" → -50
    return (float(lat0), float(lat0 + DLAT_MID),
            float(lon0), float(lon0 + DLON_MID))


def time_window_key(dt: pd.Timestamp, window_minutes: int = WINDOW_MINUTES) -> str:
    """
    Floor a timestamp to the nearest `window_minutes` boundary.
    Returns a compact string like "2024-10-10_0000" or "2024-10-10_0030".
    """
    total_min  = dt.hour * 60 + dt.minute
    floored    = (total_min // window_minutes) * window_minutes
    h, m       = divmod(floored, 60)
    return f"{dt.strftime('%Y-%m-%d')}_{h:02d}{m:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# §B  Fast metadata scan — reads only NC *attributes*, no array loading
# ─────────────────────────────────────────────────────────────────────────────

def scan_metadata(base_path: str, file_suffix: str = ".0001_nc") -> pd.DataFrame:
    """
    Scan all podTc2 files in `base_path` and return a DataFrame with one row
    per file containing the metadata needed for grouping:
        filename, full_path, date (Timestamp), lat, lon, spacecraft

    Only reads file *attributes*; no array data is loaded so this is fast
    even for hundreds of files.  Files with invalid latitudes are silently
    skipped.
    """
    rows = []
    files = sorted(f for f in os.listdir(base_path) if f.endswith(file_suffix))
    print(f"  Scanning {len(files)} files for metadata …")

    for fname in files:
        fpath = os.path.join(base_path, fname)
        try:
            with netCDF4.Dataset(fpath, "r") as nc:
                lat = float(nc.getncattr("lat_tecmax_tangent"))
                lon = float(nc.getncattr("lon_tecmax_tangent"))
                yr  = int(nc.getncattr("year"))
                mo  = int(nc.getncattr("month"))
                dy  = int(nc.getncattr("day"))
                hh  = int(nc.getncattr("hour"))
                mm  = int(nc.getncattr("minute"))
                ss  = int(nc.getncattr("second"))
        except Exception:
            continue

        if abs(lat) > 90:
            continue                     # guard against invalid metadata

        dt          = pd.Timestamp(yr, mo, dy, hh, mm, ss)
        spacecraft  = fname.split(".")[0].replace("podTc2_", "")  # e.g. "GN04"
        region      = assign_region(lat, lon)
        win_key     = time_window_key(dt)
        group_key   = f"{win_key}__{region}"                       # unique group id

        rows.append({
            "filename":   fname,
            "full_path":  fpath,
            "date":       dt,
            "lat":        lat,
            "lon":        lon,
            "spacecraft": spacecraft,
            "region":     region,
            "time_window": win_key,
            "group_key":   group_key,
        })

    meta = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(f"  Found {len(meta)} valid files across "
          f"{meta['group_key'].nunique()} geographic groups "
          f"({meta['time_window'].nunique()} 30-min windows).")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# §C  Combined bounding box for a set of occultations
# ─────────────────────────────────────────────────────────────────────────────

def _group_bounding_box(
    parsed_list: list[dict],
    alt_grid: np.ndarray,
    region_key: str,
    margin_deg: float = 10.0,
    extra_points: list[tuple[float, float]] | None = None,
) -> tuple[float, float, float, float]:
    """
    Compute a bounding box covering all occultations in `parsed_list` using
    the same approach as demo.py's section20():

      1. Call EDPSamples.get_occultation_extrema for each occultation to
         obtain three ray-path corner points (top, TEC-max, bottom tangent).
      2. Merge all corner lat/lon values across occultations, plus any
         `extra_points` (e.g. IGS ground-station pierce points) supplied by
         the caller.
      3. Determine polar vs. mid-lat from the midpoint of the *official*
         region bounding box (region_bounding_box(region_key)), not from the
         ray extrema, so the classification is always consistent with the
         grouping grid.
         - Polar  (|roi_center_lat| > 65°): poleward edge fixed at ±90°;
           equatorward edge = min/max of actual ray extrema ± margin_deg
           (expands when rays reach lower latitudes); full lon span.
         - Mid-lat: min/max of merged extrema ± margin_deg; antimeridian
           crossing handled by converting to 0–360 when raw span > 180°.

    Falls back to the region's nominal bounding box if extrema cannot be
    computed for any occultation and no tangent-point metadata is available.
    """
    all_pts_lat: list[float] = []
    all_pts_lon: list[float] = []

    for data in parsed_list:
        try:
            pt1, pt2, pt3 = EDPSamples.get_occultation_extrema(
                data["LEO"], data["GNSS"], alt_limit=700.0
            )
            for pt in (pt1, pt2, pt3):
                all_pts_lat.append(float(pt[0]))
                all_pts_lon.append(float(pt[1]))
        except Exception:
            # Fallback: use the TEC-max tangent point stored as metadata.
            all_pts_lat.append(float(data.get("lat_tecmax_tangent", 0.0)))
            all_pts_lon.append(float(data.get("lon_tecmax_tangent", 0.0)))

    if extra_points:
        for _lat, _lon in extra_points:
            all_pts_lat.append(float(_lat))
            all_pts_lon.append(float(_lon))

    if not all_pts_lat:
        return region_bounding_box(region_key)

    # Use the midpoint of the *official* region bounding box to decide whether
    # the group is polar or mid-latitude.  This is stable regardless of where
    # the individual ray extrema happen to fall within the bin.
    _roi = region_bounding_box(region_key)
    center_lat = (_roi[0] + _roi[1]) / 2.0

    if abs(center_lat) > 65.0:
        # Polar group — always extend to the geographic pole; let the
        # equatorward edge be driven by the actual ray extrema (+ margin)
        # so the mesh expands when occultations reach lower latitudes.
        # Full longitude span avoids antimeridian issues at high latitudes.
        if center_lat > 0:
            lat_min = max(min(all_pts_lat) - margin_deg, -90.0)
            lat_max = 90.0
        else:
            lat_min = -90.0
            lat_max = min(max(all_pts_lat) + margin_deg, 90.0)
        lon_min, lon_max = -180.0, 180.0
    else:
        lat_min = max(min(all_pts_lat) - margin_deg, -90.0)
        lat_max = min(max(all_pts_lat) + margin_deg,  90.0)
        lon_spread = max(all_pts_lon) - min(all_pts_lon)
        if lon_spread <= 180.0:
            lon_min = max(min(all_pts_lon) - margin_deg, -180.0)
            lon_max = min(max(all_pts_lon) + margin_deg,  180.0)
        else:
            # Antimeridian crossing: convert to 0–360, expand, shift back.
            lons_360 = [(l + 360) % 360 for l in all_pts_lon]
            raw_min  = min(lons_360) - margin_deg
            raw_max  = max(lons_360) + margin_deg
            lon_min  = raw_min - 360 if raw_min > 180 else raw_min
            lon_max  = raw_max - 360 if raw_max > 180 else raw_max

    return lat_min, lat_max, lon_min, lon_max


def _tight_bbox_from_points(
    lats: list[float],
    lons: list[float],
    margin_deg: float,
) -> tuple[float, float, float, float]:
    """
    Compute a bounding box from raw lat/lon points with a flat margin, using
    the same min/max ± margin_deg logic as the mid-lat branch of
    `_group_bounding_box` (including antimeridian handling), but WITHOUT the
    polar pole-extension special case — the box never extends to ±90° lat or
    the full lon span regardless of latitude.

    RO polar meshes deliberately span the full longitude circle for ray-
    geometry reasons (see `_group_bounding_box`), but using that same
    full-globe extent to size `igs_only`'s regular lat/lon grid would blow up
    its point count enormously for no benefit (ground stations are fixed
    points, not orbiting).  Use this helper instead for that purpose.
    """
    pts = [(float(a), float(b)) for a, b in zip(lats, lons)
           if np.isfinite(a) and np.isfinite(b)]
    if not pts:
        raise ValueError("_tight_bbox_from_points: no finite lat/lon points supplied")
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]

    lat_min = max(min(lats) - margin_deg, -90.0)
    lat_max = min(max(lats) + margin_deg,  90.0)
    lon_spread = max(lons) - min(lons)
    if lon_spread <= 180.0:
        lon_min = max(min(lons) - margin_deg, -180.0)
        lon_max = min(max(lons) + margin_deg,  180.0)
    else:
        # Antimeridian crossing: convert to 0–360, expand, shift back.
        lons_360 = [(l + 360) % 360 for l in lons]
        raw_min  = min(lons_360) - margin_deg
        raw_max  = max(lons_360) + margin_deg
        lon_min  = raw_min - 360 if raw_min > 180 else raw_min
        lon_max  = raw_max - 360 if raw_max > 180 else raw_max
        if lon_min > lon_max:
            # Points scattered across most/all of the longitude circle (e.g.
            # extrema very close to a pole, where longitude is nearly
            # undefined) — the antimeridian shift-back can't represent that
            # as a single non-wrapped [lon_min, lon_max] range.  Fall back to
            # the full circle rather than return an inverted (empty) range.
            lon_min, lon_max = -180.0, 180.0

    return lat_min, lat_max, lon_min, lon_max


# ─────────────────────────────────────────────────────────────────────────────
# §D  Core: process one geographic group with a joint KF update
# ─────────────────────────────────────────────────────────────────────────────

def process_group(
    group_key:              str,
    group_meta:             pd.DataFrame,
    alt_grid:               np.ndarray,
    global_edp_cache:       dict,
    measurement_err:        float = 10.0,
    relaxation:             float = 0.99,
    generate_plots:         bool  = True,
    save_dir:               str   = "./Figures/GroupKF/",
    run_sequential:         bool  = True,
    conphs_base_dir:        str   = None,
    conphs_max_rays:        int   = 200,
    podtc_max_rays:         int   = 200,
    num_ray_segments:       int   = 500,
    altitude_taper_km:      float = 100.0,
    altitude_taper_min_scale: float = 0.05,
    topside_follow_f2:      bool  = True,
    extra_clean_list:       list  = None,
    roi_extra_points:       list[tuple[float, float]] | None = None,
    filter_label:           str   = "",
) -> dict:
    """
    Process a geographic group of occultations with a single joint KF update.

    Parameters
    ----------
    group_key        : Unique string identifying this time-window + region.
    group_meta       : Rows of the metadata DataFrame for this group.
    alt_grid         : 1-D altitude array (km).
    global_edp_cache : Dict[int hour → EDPSamples] from build_daily_global_edps.
    measurement_err  : Diagonal measurement noise variance R (TECU²).
    relaxation       : Gauss-Markov relaxation applied before the KF update.
    generate_plots   : Whether to write per-group diagnostic plots to disk.
    save_dir         : Directory for figure output.
    run_sequential   : When False, skip the sequential KF loop and its associated
                       plots (_seq, _sequential, _comparison).  Only the joint
                       (batch) update is performed.  Saves significant compute
                       time when sequential estimates are not needed.
    conphs_base_dir  : If provided, attempt to find and load a matching conPhs
                       file for each podTc2 occultation.  Matching uses
                       load_conPhs() with this directory as the search root.
                       conPhs arcs contribute relative TEC; an unknown
                       carrier-phase bias per arc is jointly estimated with
                       the ionospheric state.  When None (default), only
                       absolute podTc2 TEC is assimilated.
    conphs_max_rays  : Maximum number of rays to retain from each conPhs arc
                       after validity filtering.  conPhs data is sampled at
                       100 Hz, so a typical arc contains several thousand
                       points.  Uniform-stride decimation is applied to bring
                       the count down to at most this many rays before the arc
                       is added to the H-matrix computation.  Default: 200.
    podtc_max_rays   : Maximum number of rays to retain from each podTc2 arc
                       after validity filtering.  Uniform-stride decimation is
                       applied so the altitude span of the arc is preserved.
                       Default: 200.
    num_ray_segments : Number of integration points used to discretise each
                       GNSS–LEO ray when building the H matrix.  Lower values
                       are faster; 500 (default) gives a good accuracy/speed
                       trade-off.  Reduce to 250 for a further ~2× speed-up.
    filter_label     : Optional short tag (e.g. "kf") inserted into the
                       generated "_seq"/"_joint" figure filenames as
                       "_{filter_label}_seq"/"_{filter_label}_joint" (and into
                       the covariance-plot filename as "_{filter_label}"
                       before "_covariance"), so callers that run more than
                       one filter type against the same group_key can tell
                       gridded-KF outputs apart from other filters' outputs
                       on disk.  Default "" preserves the original
                       "_seq"/"_joint"/"_covariance" filenames exactly, so
                       existing callers are unaffected.

    Returns
    -------
    dict with per-group statistics and references to the plot path.
    """
    t_start = time.time()
    n_occ   = len(group_meta)
    region  = group_meta["region"].iloc[0]
    win_key = group_meta["time_window"].iloc[0]

    print(f"\n{'─'*60}")
    print(f"  Group : {group_key}")
    print(f"  Window: {win_key}  |  Region: {region}  |  {n_occ} occultation(s)")
    print(f"{'─'*60}")

    result = {
        "group_key":      group_key,
        "region":         region,
        "time_window":    win_key,
        "n_occultations": n_occ,
        "files":          list(group_meta["filename"]),
        "lats":           list(group_meta["lat"]),
        "lons":           list(group_meta["lon"]),
        "status":         "Failed",
        "prior_tec_rmse":       np.nan,
        "post_tec_rmse":        np.nan,
        "joint_post_tec_rmse":  np.nan,
        "plot_path":            None,
        "joint_plot_path":      None,
        "comparison_plot_path": None,
    }

    try:
        # ── 1. Parse full data for every file in the group ─────────────────
        parsed_list  = []          # list of podTc_data dicts (after QC)
        file_labels  = []          # short label for each occultation (for plots)
        conphs_list  = []          # parallel list: conPhs data dict or None
        for _, row in group_meta.iterrows():
            data = parse_podTc2_nc_file(row["full_path"])
            if data is None:
                print(f"    [skip] {row['filename']} — parse returned None")
                continue
            parsed_list.append(data)
            _tec_vals = data.get("TEC_podTc2", data.get("TEC", None))
            if _tec_vals is not None and len(_tec_vals) > 0:
                _tmax_idx  = int(np.argmax(_tec_vals))
                _tmax_time = data["date"] + pd.to_timedelta(float(data["time"][_tmax_idx]), unit="s")
            else:
                _tmax_time = row["date"]
            file_labels.append(f"{row['spacecraft']} {_tmax_time.strftime('%H:%M')}")

            # Optionally load the companion conPhs file for relative TEC
            if conphs_base_dir is not None:
                cp = load_conPhs(row["full_path"], conPhs_base_dir=conphs_base_dir)
                conphs_list.append(cp)
                if cp is not None:
                    print(f"    [conPhs] Loaded relative TEC for {row['filename']}")
            else:
                conphs_list.append(None)

        if len(parsed_list) < 1:
            result["status"] = "No Valid Files"
            return result

        print(f"  Parsed {len(parsed_list)}/{n_occ} files successfully.")

        # ── 2. Build clean (NaN-free, positive TEC) arrays for each occ ────
        # Each entry in clean_list carries a 'tec_type' field:
        #   'absolute' — absolute TECU from podTc2 (no bias state needed)
        #   'relative' — relative TECU from conPhs  (bias estimated by KF)
        # When a conPhs arc is available it is added as an additional entry
        # alongside the absolute podTc2 arc for the same occultation, so the
        # KF benefits from both measurement types simultaneously.
        clean_list   = []    # list of dicts: {'tec', 'tangent_km', 'LEO', 'GNSS', 'tec_type'}
        clean_labels = []    # display label for each entry in clean_list
        clean_parsed = []    # full podTc_data dict for Abel inversion
        clean_sat_ids = []   # (leo_id, prn_id) per entry — receiver × transmitter
        for i, data in enumerate(parsed_list):
            _, _, tang_raw = rayTangent(data["LEO"], data["GNSS"], units="km")
            tang_km  = tang_raw * 1e-3
            meas_tec = data.get("TEC_podTc2", data.get("TEC", np.zeros_like(tang_km)))
            valid    = ~np.isnan(meas_tec) & (meas_tec > 0)

            leo_id   = str(data.get("leo_id", "??")).strip()
            con_id   = str(data.get("conid",  "?")).strip()
            prn_num  = str(data.get("prn_id", "??")).strip()
            full_prn = f"{con_id}{prn_num}"

            n_valid = int(valid.sum())
            if n_valid < 50:
                print(f"    [skip] {file_labels[i]} — only {n_valid} valid rays")
            else:
                if n_valid > podtc_max_rays:
                    _stride   = int(np.ceil(n_valid / podtc_max_rays))
                    _dec_idx  = np.where(valid)[0][::_stride]
                    _dec_mask = np.zeros(len(meas_tec), dtype=bool)
                    _dec_mask[_dec_idx] = True
                else:
                    _dec_mask = valid
                n_dec = int(_dec_mask.sum())
                print(f"    [podTc2] {file_labels[i]} — "
                      f"{n_valid} valid rays → decimated to {n_dec} "
                      f"(stride {int(np.ceil(n_valid / podtc_max_rays)) if n_valid > podtc_max_rays else 1})")
                clean_list.append({
                    "tec":        np.asarray(meas_tec[_dec_mask], dtype=np.float64).flatten(),
                    "tangent_km": tang_km[_dec_mask].flatten(),
                    "LEO":        data["LEO"][:,  _dec_mask],
                    "GNSS":       data["GNSS"][:, _dec_mask],
                    "tec_type":   "absolute",
                })
                clean_labels.append(file_labels[i])
                clean_parsed.append(data)
                clean_sat_ids.append((leo_id, full_prn))

            # Add conPhs relative-TEC arc if available (independent entry in clean_list)
            cp = conphs_list[i]
            if cp is not None:
                rel_tec  = cp.get("rel_TEC", np.array([]))
                cp_valid = np.isfinite(rel_tec) & (cp["tangent_alt_km"] > 0)
                n_cp_valid = int(cp_valid.sum())
                if n_cp_valid >= 50:
                    # Decimate to at most conphs_max_rays using a uniform stride.
                    # conPhs is sampled at 100 Hz; a stride > 1 thins the arc
                    # while preserving its full altitude span.
                    if n_cp_valid > conphs_max_rays:
                        stride   = int(np.ceil(n_cp_valid / conphs_max_rays))
                        dec_idx  = np.where(cp_valid)[0][::stride]
                        dec_mask = np.zeros(len(rel_tec), dtype=bool)
                        dec_mask[dec_idx] = True
                    else:
                        dec_mask = cp_valid
                    n_dec = int(dec_mask.sum())
                    print(f"    [conPhs] {file_labels[i]} — "
                          f"{n_cp_valid} valid rays → decimated to {n_dec} "
                          f"(stride {int(np.ceil(n_cp_valid / conphs_max_rays)) if n_cp_valid > conphs_max_rays else 1})")
                    clean_list.append({
                        "tec":        np.asarray(rel_tec[dec_mask], dtype=np.float64).flatten(),
                        "tangent_km": cp["tangent_alt_km"][dec_mask].flatten(),
                        "LEO":        cp["LEO"][:,  dec_mask],
                        "GNSS":       cp["GNSS"][:, dec_mask],
                        "tec_type":   "relative",
                    })
                    clean_labels.append(f"{file_labels[i]} [conPhs]")
                    clean_parsed.append(data)   # use podTc2 for Abel/plotting
                    clean_sat_ids.append((leo_id, full_prn))
                else:
                    print(f"    [conPhs skip] {file_labels[i]} — "
                          f"only {n_cp_valid} valid relative-TEC rays")

        if len(clean_list) < 1:
            result["status"] = "Insufficient Rays"
            return result

        # ── 3. Derive the EDP grid hour from the group's time window ────────
        # Use the median observation time to pick the closest cached hour.
        all_dates    = [d["date"] for d in parsed_list]
        median_ts    = pd.Timestamp(np.median([d.value for d in all_dates]))
        profile_hour = median_ts.hour

        # ── 4 & 5. Bounding box + mesh with vertex-count guard ──────────────
        # Uses the same get_occultation_extrema method as demo.py section20().
        # Two strategies depending on group type:
        #
        #   Polar   (POLAR_N / POLAR_S): shrink the equatorward latitude boundary
        #           in _SHRINK_STEP degree increments until the mesh is small
        #           enough.  No occultations are ever dropped.
        #
        #   Mid-lat (rectangular bins): keep the bounding box fixed to the ray
        #           extrema; remove the occultation whose centroid is farthest
        #           from the group centroid until the mesh fits.
        _BBOX_MARGIN = 100.0 / 111.32    # 100 km of padding around ray-path footprints
        _SHRINK_STEP = 1.0     # equatorward shrink per polar iteration (degrees)
        _is_polar    = region in ("POLAR_N", "POLAR_S")

        # Pre-compute extrema centre (mean of 3 corner points) per occultation.
        # Used by the mid-lat outlier loop; also stored for the trim diagnostic.
        occ_centers: list[tuple[float, float]] = []
        for _data in clean_parsed:
            try:
                _pt1, _pt2, _pt3 = EDPSamples.get_occultation_extrema(
                    _data["LEO"], _data["GNSS"], alt_limit=700.0
                )
                _clat = float(np.mean([_pt1[0], _pt2[0], _pt3[0]]))
                _clon = float(np.mean([_pt1[1], _pt2[1], _pt3[1]]))
            except Exception:
                _clat = float(_data.get("lat_tecmax_tangent", 0.0))
                _clon = float(_data.get("lon_tecmax_tangent", 0.0))
            occ_centers.append((_clat, _clon))

        # Trim-tracking mirrors clean_list for the diagnostic plot.
        _trim_info: list[dict] = [
            {
                "label":      clean_labels[i],
                "sat_id":     clean_sat_ids[i],
                "tec_lat":    float(clean_parsed[i].get(
                                  "lat_tecmax_tangent", occ_centers[i][0])),
                "tec_lon":    float(clean_parsed[i].get(
                                  "lon_tecmax_tangent", occ_centers[i][1])),
                "LEO":        clean_list[i]["LEO"],
                "GNSS":       clean_list[i]["GNSS"],
                "tec":        clean_list[i]["tec"],
                "tangent_km": clean_list[i]["tangent_km"],
                "tec_type":   clean_list[i].get("tec_type", "absolute"),
            }
            for i in range(len(clean_list))
        ]
        _trim_removed: list[dict] = []   # remains empty for polar groups

        # ── Build per-occultation triangle list (union mesh) ─────────────────
        def _build_occ_triangles(parsed_list):
            tris = []
            for _data in parsed_list:
                try:
                    pt1, pt2, pt3 = EDPSamples.get_occultation_extrema(
                        _data["LEO"], _data["GNSS"], alt_limit=700.0
                    )
                    tris.append((pt1, pt2, pt3))
                except Exception:
                    tris.append(None)
            return tris

        def _subset_union(bbox_eds, parsed_list, margin_deg=_BBOX_MARGIN):
            tris = [t for t in _build_occ_triangles(parsed_list) if t is not None]
            if not tris:
                return bbox_eds
            return bbox_eds.subset_union_triangles(tris, margin_deg=margin_deg)

        if _is_polar:
            # ── Polar: shrink equatorward boundary, then apply union mask ────
            lat_min, lat_max, lon_min, lon_max = _group_bounding_box(
                clean_parsed, alt_grid, region, margin_deg=_BBOX_MARGIN,
                extra_points=roi_extra_points,
            )
            while True:
                eds_bbox = global_edp_cache[profile_hour].subset_region(
                    lat_min, lat_max, lon_min, lon_max
                )
                try:
                    eds_occ = _subset_union(eds_bbox, clean_parsed)
                except ValueError:
                    eds_occ = eds_bbox
                n_geo = eds_occ.geolocation.shape[0]
                if n_geo <= MAX_MESH_VERTICES:
                    print(f"  [polar] Bbox lat [{lat_min:.1f}, {lat_max:.1f}]  "
                          f"lon [{lon_min:.1f}, {lon_max:.1f}]  |  "
                          f"union vertices: {n_geo}")
                    break
                if region == "POLAR_N":
                    new_eq = lat_min + _SHRINK_STEP
                    if new_eq >= 89.0:
                        print(f"  [polar-shrink] Hit poleward limit, accepting {n_geo}.")
                        break
                    lat_min = new_eq
                else:
                    new_eq = lat_max - _SHRINK_STEP
                    if new_eq <= -89.0:
                        print(f"  [polar-shrink] Hit poleward limit, accepting {n_geo}.")
                        break
                    lat_max = new_eq

        else:
            # ── Mid-lat: remove farthest outlier, then apply union mask ──────
            while True:
                lat_min, lat_max, lon_min, lon_max = _group_bounding_box(
                    clean_parsed, alt_grid, region, margin_deg=_BBOX_MARGIN,
                    extra_points=roi_extra_points,
                )
                eds_bbox = global_edp_cache[profile_hour].subset_region(
                    lat_min, lat_max, lon_min, lon_max
                )
                try:
                    eds_occ = _subset_union(eds_bbox, clean_parsed)
                except ValueError:
                    eds_occ = eds_bbox
                n_geo = eds_occ.geolocation.shape[0]
                print(f"  Bbox lat [{lat_min:.1f}, {lat_max:.1f}]  "
                      f"lon [{lon_min:.1f}, {lon_max:.1f}]  |  "
                      f"union vertices: {n_geo}")

                if n_geo <= MAX_MESH_VERTICES or len(clean_list) <= 1:
                    break

                _clats  = np.array([c[0] for c in occ_centers])
                _clons  = np.array([c[1] for c in occ_centers])
                _gc_lat = float(np.mean(_clats))
                _gc_lon = float(np.mean(_clons))
                _dist   = (_clats - _gc_lat) ** 2 + (_clons - _gc_lon) ** 2
                worst   = int(np.argmax(_dist))
                print(f"  [trim] {n_geo} > MAX_MESH_VERTICES={MAX_MESH_VERTICES} — "
                      f"dropping {clean_labels[worst]} "
                      f"({np.sqrt(_dist[worst]):.1f}° from centroid)")
                _trim_removed.append(_trim_info[worst])
                _trim_info.pop(worst)
                clean_list.pop(worst)
                clean_labels.pop(worst)
                clean_parsed.pop(worst)
                clean_sat_ids.pop(worst)
                occ_centers.pop(worst)

        n_height = len(alt_grid)
        print(f"  Final group: {len(clean_list)} occultation(s), "
              f"{n_geo} union-mesh vertices")

        # ── 5b. Trim diagnostic plot (before assimilation) ───────────────────
        if generate_plots:
            try:
                _plot_trim_diagnostic(
                    kept        = _trim_info,
                    removed     = _trim_removed,
                    lat_min     = lat_min,
                    lat_max     = lat_max,
                    lon_min     = lon_min,
                    lon_max     = lon_max,
                    region      = region,
                    save_dir    = save_dir,
                    group_key   = group_key,
                    time_window = win_key,
                    eds_occ     = eds_occ,
                )
            except Exception as _exc_trim:
                print(f"  [warn] Trim diagnostic plot failed: {_exc_trim}")

        # ── 5b. Inject lognormal noise into the prior EDP ensemble ───────────
        # Adds controlled spread to each ensemble member before the inverter
        # computes np.cov(edps_flat), enriching the sample covariance.
        # Mirror of the add_noise path in demo.py section20.
        if PRIOR_EDP_NOISE_SIGMA:
            _noise_rng  = np.random.default_rng(seed=None)  # fresh seed each run
            _edps_arr   = eds_occ["EDPs"].values            # (n_height, n_geo, n_sample)
            _eps        = _noise_rng.standard_normal(_edps_arr.shape).astype(_edps_arr.dtype)
            eds_occ["EDPs"].values[:] = _edps_arr * np.exp(PRIOR_EDP_NOISE_SIGMA * _eps)
            print(f"  [cov noise] Lognormal noise σ={PRIOR_EDP_NOISE_SIGMA} applied "
                  f"to prior ensemble ({_edps_arr.shape[2]} members).")

        # ── 5c. Inject pre-built ground-station arcs ─────────────────────────
        # Injected AFTER bounding-box / mesh construction so ground-station
        # rays do not distort the voxel-grid domain.  All injected arcs must
        # be tec_type='absolute' (IGS absolute TEC from igs_tec_pipeline.py).
        if extra_clean_list:
            n_before = len(clean_list)
            for _ec in extra_clean_list:
                clean_list.append(_ec)
                _eid = _ec.get("leo_id", "IGS")
                _prn = _ec.get("prn_id", "?")
                clean_labels.append(f"{_eid}/{_prn}")
                clean_sat_ids.append((_eid, _prn))
                clean_parsed.append({       # minimal dict for Abel-inversion fallback
                    "LEO":  _ec["LEO"],
                    "GNSS": _ec["GNSS"],
                    "lat_tecmax_tangent": float(_ec.get("lat_tecmax_tangent", 0.0)),
                    "lon_tecmax_tangent": float(_ec.get("lon_tecmax_tangent", 0.0)),
                })
            print(f"  [extra] Injected {len(clean_list) - n_before} ground-station "
                  f"arc(s) → total {len(clean_list)} arc(s) in assimilation pool")

        # ── 6. Build ONE inverter from the combined mesh ─────────────────────
        # Count relative-TEC arcs so the inverter can augment its state with
        # one carrier-phase bias per conPhs arc.
        n_rel_arcs = sum(1 for cl in clean_list if cl["tec_type"] == "relative")
        if n_rel_arcs > 0:
            print(f"  [{n_rel_arcs} relative-TEC arc(s) — bias states will be estimated]")

        # topside_prior_floor_tecu guards against IRI's lack of a plasmasphere:
        # when IRI clips ne to the physical floor at 800 km, x_top_prior collapses
        # to ~0.002 TECU, making the forward TEC zero for high-tangent-altitude rays.
        # topside_alpha=0 → pure single-exponential decay (simpler model,
        # appropriate now that the lat/lon coordinate error has been fixed).
        inverter = Ionosphere_Tomography_Inverter(
            EDPSam=eds_occ, meanscale=1, topside_prior_floor_tecu=1.0,
            n_rel_arcs=n_rel_arcs, topside_alpha=0.0,
            gaussian_cov_sigma=GAUSSIAN_COV_SIGMA,
            altitude_taper_km=altitude_taper_km,
            altitude_taper_min_scale=altitude_taper_min_scale,
            topside_follow_f2=topside_follow_f2,
        )
        _n_sv    = inverter.attrs["n_state_vars"]

        # ── 7. Build all H matrices in a single parallel sweep ──────────────
        # get_observation_operator_batch spawns the joblib worker pool exactly
        # once for all arcs combined, eliminating the per-arc pool-management
        # overhead that dominated runtime in the old per-arc loop.
        tec_obs    = [cl["tec"]        for cl in clean_list]
        ray_counts = [len(cl["tec"])   for cl in clean_list]
        H_blocks   = inverter.get_observation_operator_batch(
            clean_list, num_segments=num_ray_segments
        )

        # Stack all H matrices and obs for prior TEC computation and final
        # posterior TEC over all observations (stacking is still needed for the
        # RMSE stats and per-occ TEC slices in the summary figure).
        #   H_joint  : (sum_rays, n_state_aug)
        #   obs_joint: (sum_rays,)
        H_joint   = np.vstack(H_blocks).astype(np.float32)
        obs_joint = np.concatenate(tec_obs).astype(np.float64)
        print(f"  Combined H shape : {H_joint.shape}  "
              f"(total rays = {len(obs_joint)})")

        # ── 8. Prior TEC and bias pre-warming ────────────────────────────────
        prior_state_flat = inverter.attrs["initial_edps_mean"]   # (n_sv, 1)
        x_top_prior      = inverter.attrs["x_top_prior"]         # (n_geo,)
        _n_sv_aug        = inverter.attrs["n_state_vars_aug"]    # grid + topside (no bias)
        # Bias prior is zero, so it does not contribute to the prior TEC.
        prior_tec_joint  = (
            H_joint[:, :_n_sv] @ prior_state_flat
            + H_joint[:, _n_sv:_n_sv_aug] @ x_top_prior[:, None]
        ).flatten()

        # Pre-warm each relative-arc bias state to the mean prior-TEC residual.
        # Without this, the KF sees a huge innovation (the carrier-phase offset
        # can be tens of TECU) and tries to compensate by distorting the grid
        # state, causing catastrophic divergence.  Setting x[bias_idx] to the
        # mean(rel_TEC − modeled_TEC) means the KF innovation only contains
        # genuine ionospheric structure, not the arc offset.
        if n_rel_arcs > 0:
            x_prior_full = np.vstack([
                prior_state_flat,
                x_top_prior[:, None],
                np.zeros((n_rel_arcs, 1)),
            ])
            rel_arc_idx = 0
            for i, cl in enumerate(clean_list):
                if cl.get("tec_type") == "relative":
                    modeled = (H_blocks[i].astype(np.float64) @ x_prior_full).flatten()
                    bias_init = float(np.mean(cl["tec"] - modeled))
                    inverter.x[_n_sv_aug + rel_arc_idx, 0] = bias_init
                    print(f"  [bias init] Arc {rel_arc_idx} ({clean_labels[i]}): "
                          f"{bias_init:+.2f} TECU")
                    rel_arc_idx += 1

        # ── 9. Sequential Kalman Filter updates (optional) ───────────────────
        if run_sequential:
            # Each occultation is assimilated one at a time.  The inverter
            # carries self.x and self.P forward between calls, so the posterior
            # from step k automatically becomes the prior for step k+1.
            #
            # step_edp_snapshots : list of (label, edp_3d) tuples.
            #   Index 0 = prior; index k = posterior after step k.
            print("  Running sequential KF assimilation …")
            step_edp_snapshots: list[tuple[str, np.ndarray]] = [
                ("Prior", prior_state_flat.reshape(n_height, n_geo).copy())
            ]

            posterior_state_flat = prior_state_flat   # fallback if no occs survive
            _H_eff_m = inverter.attrs["topside_H_eff_m"]
            for k, (H_k, obs_k, lbl_k) in enumerate(
                zip(H_blocks, tec_obs, clean_labels)
            ):
                print(f"  [step {k+1}/{len(clean_list)}] Assimilating {lbl_k} "
                      f"({len(obs_k)} rays) …")
                posterior_state_flat = inverter.assimilate(
                    obs             = obs_k,
                    obs_operator    = H_k.astype(np.float32),
                    relaxation      = relaxation,
                    measurement_err = measurement_err,
                    podTc2_data     = clean_list[k],
                    distance_localization=True,
                    localization_radius_km=800.0,
                    localization_mode='inverse_distance'
                )
                x_top_tecu_cur = inverter.x_top_tecu.flatten()
                ne_top_post = np.asarray(posterior_state_flat).reshape(n_height, n_geo)[-1, :]
                x_top_prior_new = ne_top_post * inverter.attrs["topside_H_eff_m"] / 1e16
                # Re-apply the plasmasphere floor so the re-anchored prior does not
                # collapse to near-zero when IRI still gives floor-level density at
                # the grid top after the KF update.
                x_top_prior_new = np.maximum(
                    x_top_prior_new, inverter.attrs["topside_prior_floor_tecu"]
                )
                inverter.attrs["x_top_prior"] = x_top_prior_new
                inverter.x[_n_sv:_n_sv_aug] = (x_top_tecu_cur - x_top_prior_new)[:, None]
                step_edp_snapshots.append((
                    f"Step {k+1}: {lbl_k}",
                    np.asarray(posterior_state_flat).reshape(n_height, n_geo).copy(),
                ))
        else:
            print("  Skipping sequential KF assimilation (run_sequential=False).")
            posterior_state_flat = prior_state_flat   # unused but keeps namespace clean
            step_edp_snapshots   = []

        # ── 10. Joint (batch) KF update — fresh inverter, same prior ────────
        # A single call ingests all occultations simultaneously so each one
        # sees the original prior P, not a narrowed posterior.
        print("  Running joint (batch) KF assimilation …")
        inverter_jnt = Ionosphere_Tomography_Inverter(
            EDPSam=eds_occ, meanscale=1, topside_prior_floor_tecu=1.0,
            n_rel_arcs=n_rel_arcs, topside_alpha=0.0,
            gaussian_cov_sigma=GAUSSIAN_COV_SIGMA,
            altitude_taper_km=altitude_taper_km,
            altitude_taper_min_scale=altitude_taper_min_scale,
            topside_follow_f2=topside_follow_f2,
        )
        # Pre-warm bias states on the joint inverter with the same prior residuals.
        if n_rel_arcs > 0:
            x_prior_full_jnt = np.vstack([
                inverter_jnt.attrs["initial_edps_mean"],
                inverter_jnt.attrs["x_top_prior"][:, None],
                np.zeros((n_rel_arcs, 1)),
            ])
            _n_sv_aug_jnt_init = inverter_jnt.attrs["n_state_vars_aug"]
            rel_arc_idx_jnt = 0
            for i, cl in enumerate(clean_list):
                if cl.get("tec_type") == "relative":
                    modeled_jnt = (H_blocks[i].astype(np.float64) @ x_prior_full_jnt).flatten()
                    inverter_jnt.x[_n_sv_aug_jnt_init + rel_arc_idx_jnt, 0] = float(
                        np.mean(cl["tec"] - modeled_jnt)
                    )
                    rel_arc_idx_jnt += 1

        posterior_state_flat_jnt = inverter_jnt.assimilate(
            obs             = obs_joint,
            obs_operator    = H_joint.astype(np.float32),
            relaxation      = relaxation,
            measurement_err = measurement_err,
            use_info_form   = True,
        )

        # ── 11. Posterior TEC ─────────────────────────────────────────────────
        # Slice only grid and topside columns (bias columns don't map to TEC output).
        _n_sv_aug_jnt = inverter_jnt.attrs["n_state_vars_aug"]
        post_tec_jnt = (
            H_joint[:, :_n_sv] @ np.asarray(posterior_state_flat_jnt)
            + H_joint[:, _n_sv:_n_sv_aug_jnt] @ inverter_jnt.x_top_tecu
        ).flatten()

        if run_sequential:
            post_tec_seq = (
                H_joint[:, :_n_sv] @ np.asarray(posterior_state_flat)
                + H_joint[:, _n_sv:_n_sv_aug] @ inverter.x_top_tecu
            ).flatten()
        else:
            post_tec_seq = post_tec_jnt   # alias — seq stats not reported

        # ── 12. Per-group residual statistics ────────────────────────────────
        # Relative-TEC (conPhs) arcs carry an unknown carrier-phase offset that
        # would inflate RMSE figures if included.  Build a boolean mask that
        # selects only absolute-TEC rays for the error metrics; relative arcs
        # are still assimilated but excluded from the printed/stored statistics.
        abs_mask = np.concatenate([
            np.ones(n, dtype=bool) if cl.get("tec_type") != "relative"
            else np.zeros(n, dtype=bool)
            for cl, n in zip(clean_list, ray_counts)
        ])
        has_abs = abs_mask.any()

        prior_res    = obs_joint - prior_tec_joint
        post_res_jnt = obs_joint - post_tec_jnt
        result["prior_tec_rmse"] = float(
            np.sqrt(np.mean(prior_res[abs_mask] ** 2)) if has_abs else np.nan
        )
        result["joint_post_tec_rmse"] = float(
            np.sqrt(np.mean(post_res_jnt[abs_mask] ** 2)) if has_abs else np.nan
        )
        if run_sequential:
            post_res_seq = obs_joint - post_tec_seq
            result["post_tec_rmse"] = float(
                np.sqrt(np.mean(post_res_seq[abs_mask] ** 2)) if has_abs else np.nan
            )
            print(f"  Prior RMSE        : {result['prior_tec_rmse']:.3f} TECU")
            print(f"  Post  RMSE (seq)  : {result['post_tec_rmse']:.3f} TECU")
            print(f"  Post  RMSE (joint): {result['joint_post_tec_rmse']:.3f} TECU")
        else:
            result["post_tec_rmse"] = result["joint_post_tec_rmse"]
            print(f"  Prior RMSE  : {result['prior_tec_rmse']:.3f} TECU")
            print(f"  Post  RMSE  : {result['joint_post_tec_rmse']:.3f} TECU")

        # ── 13. Slice joint arrays back into per-occultation pieces ─────────
        tec_slices     = _make_tec_slices(obs_joint, prior_tec_joint,
                                          post_tec_seq, ray_counts, clean_list)
        tec_slices_jnt = _make_tec_slices(obs_joint, prior_tec_joint,
                                          post_tec_jnt, ray_counts, clean_list)

        # ── 13. Abel inversion — one per occultation ────────────────────────
        # Runs on the *full* (unmasked) podTc_data dict so the Abel profile
        # uses the complete TEC arc rather than the validity-filtered subset.
        from Abel_Inverter import run_abel_inversion
        abel_list = []
        n_ground_skipped = 0
        for i, data in enumerate(clean_parsed):
            if "TEC_podTc2" not in data:
                # Ground-station (IGS) arcs have no LEO-GNSS TEC-vs-tangent-
                # altitude profile -- Abel inversion is an RO-only technique,
                # so skip rather than hit run_abel_inversion's KeyError path.
                abel_list.append(None)
                n_ground_skipped += 1
                continue
            try:
                abel = run_abel_inversion(data)
                if abel is None or len(abel.get("Ne", [])) == 0:
                    abel = None
            except Exception as exc_a:
                print(f"    [Abel] {clean_labels[i]} failed: {exc_a}")
                abel = None
            abel_list.append(abel)
            if abel is not None:
                abel_nm, abel_hm = extract_robust_f2_peak(abel["Ne"], abel["alt_km"])
                print(f"    [Abel] {clean_labels[i]}: NmF2={abel_nm:.2e} m⁻³  hmF2={abel_hm:.1f} km")
        if n_ground_skipped:
            print(f"    [Abel] Skipped {n_ground_skipped} ground-station arc(s) "
                  f"(RO-only technique)")

        # Store for plotting
        result["tec_slices"]           = tec_slices           # sequential posterior TEC slices
        result["joint_tec_slices"]     = tec_slices_jnt       # joint posterior TEC slices
        result["file_labels"]          = clean_labels          # aligned with clean_list
        result["sat_ids"]              = clean_sat_ids         # (leo_id, prn_id) per occ
        result["clean_list"]           = clean_list            # LEO/GNSS arrays for raypaths
        result["abel_list"]            = abel_list             # Abel results per occultation
        result["eds_occ"]              = eds_occ
        result["prior_edp_3d"]         = prior_state_flat.reshape(n_height, n_geo)
        result["post_edp_3d"]          = np.asarray(posterior_state_flat).reshape(n_height, n_geo)
        result["joint_post_edp_3d"]    = np.asarray(posterior_state_flat_jnt).reshape(n_height, n_geo)
        result["step_edp_snapshots"]   = step_edp_snapshots   # prior + one per step
        result["alt_grid"]             = alt_grid
        result["prior_P"]              = inverter_jnt.attrs["initial_edps_cov"]
        result["post_P"]               = np.array(inverter_jnt.P)
        result["x_top_tecu_jnt"]       = np.array(inverter_jnt.x_top_tecu).flatten()
        result["status"]               = "Success"

        # ── 15. Optional per-group diagnostic plots ──────────────────────────
        if generate_plots:
            centre_idx_plt = _roi_centre_idx(eds_occ.geolocation, region)

            # filter_label ("" by default) is inserted right after group_key
            # so filenames stay unchanged for existing callers, while callers
            # running multiple filter types against the same group_key (e.g.
            # demo_isr_da_comparison.py) can pass filter_label="kf" to get
            # "_kf_seq"/"_kf_joint"/"_kf_covariance" filenames instead of the
            # bare "_seq"/"_joint"/"_covariance" ones.
            _seq_suffix   = f"_{filter_label}_seq"   if filter_label else "_seq"
            _joint_suffix = f"_{filter_label}_joint" if filter_label else "_joint"

            # ── Sequential summary figure (only when sequential was run) ─────
            if run_sequential:
                result["plot_path"] = _plot_group(
                    result, save_dir=save_dir, group_key=group_key,
                    suffix=_seq_suffix, mode_label="Sequential KF",
                    igs_entries=extra_clean_list,
                )

            # ── Joint summary figure ─────────────────────────────────────────
            result_jnt = dict(result)
            result_jnt["post_edp_3d"]   = result["joint_post_edp_3d"]
            result_jnt["tec_slices"]    = tec_slices_jnt
            result_jnt["post_tec_rmse"] = result["joint_post_tec_rmse"]
            result["joint_plot_path"] = _plot_group(
                result_jnt, save_dir=save_dir, group_key=group_key,
                suffix=_joint_suffix, mode_label="Joint KF",
                igs_entries=extra_clean_list,
            )

            # ── Altitude-slice ΔNe grid (joint posterior) ────────────────────
            try:
                result["alt_slice_plot_path"] = _plot_altitude_slices(
                    result_jnt, save_dir=save_dir, group_key=group_key,
                    suffix=_joint_suffix,
                )
            except Exception as _exc_alt:
                print(f"  [warn] Altitude-slice plot failed: {_exc_alt}")

            # ── Prior / posterior covariance panels ───────────────────────────
            try:
                result["cov_plot_path"] = _plot_covariance_panels_tagged(
                    result_jnt, save_dir=save_dir, group_key=group_key,
                    tag=filter_label,
                )
            except Exception as _exc_cov:
                print(f"  [warn] Covariance plot failed: {_exc_cov}")

            # ── Sequential-update centre-vertex figure (sequential only) ─────
            if run_sequential:
                result["seq_plot_path"] = _plot_sequential_centre(
                    step_edp_snapshots = step_edp_snapshots,
                    alt_grid           = alt_grid,
                    sat_ids            = clean_sat_ids,
                    centre_idx         = centre_idx_plt,
                    save_dir           = save_dir,
                    group_key          = group_key,
                )

            # ── Sequential vs Joint comparison figure (sequential only) ──────
            if run_sequential:
                try:
                    result["comparison_plot_path"] = _plot_comparison(
                        result, save_dir=save_dir, group_key=group_key
                    )
                except Exception as _exc_cmp:
                    print(f"  [warn] Comparison plot failed: {_exc_cmp}")

    except Exception as exc:
        print(f"  [!] Error in group {group_key}: {exc}")
        result["status"] = f"Error: {exc}"

    finally:
        result["processing_time_s"] = time.time() - t_start
        gc.collect()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# §D-helper  Slice joint arrays back into per-occultation pieces
# ─────────────────────────────────────────────────────────────────────────────

def _make_tec_slices(
    obs_joint: np.ndarray,
    prior_tec_joint: np.ndarray,
    post_tec_flat: np.ndarray,
    ray_counts: list[int],
    clean_list: list[dict],
) -> list[dict]:
    """Return per-occultation dict slices from joint arrays."""
    slices: list[dict] = []
    start = 0
    for i, n_rays in enumerate(ray_counts):
        slices.append({
            "measured":   obs_joint[start : start + n_rays],
            "prior_tec":  prior_tec_joint[start : start + n_rays],
            "post_tec":   post_tec_flat[start : start + n_rays],
            "tangent_km": clean_list[i]["tangent_km"],
        })
        start += n_rays
    return slices


# ─────────────────────────────────────────────────────────────────────────────
# §E  Per-group 3-panel diagnostic plot
# ─────────────────────────────────────────────────────────────────────────────

# Module-level ECEF→lon/lat transformer (initialised once, reused per call).
_ECEF_TO_LL = pyproj.Transformer.from_crs(
    pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
    pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
    always_xy=True,
)


def _draw_raypath(ax, LEO: np.ndarray, GNSS: np.ndarray,
                  ray_idx: int, color, ls: str, lw: float,
                  label: str | None, zorder: int = 6, TP: bool = False) -> None:
    """
    Convert one ECEF ray (GNSS→LEO parametric line) to lon/lat and draw
    the ionospheric portion (<800 km altitude) on a cartopy axes.
    The tangent point is marked with a circle.
    """
    t_ray  = np.linspace(0, 1, 120)
    leo_r  = LEO[:,  ray_idx]
    gnss_r = GNSS[:, ray_idx]
    pts    = gnss_r[:, None] + (leo_r[:, None] - gnss_r[:, None]) * t_ray
    r_lons, r_lats, r_alts_m = _ECEF_TO_LL.transform(
        pts[0] * 1e3, pts[1] * 1e3, pts[2] * 1e3
    )
    iono_mask = r_alts_m < 800_000
    if np.any(iono_mask):
        ax.plot(r_lons[iono_mask], r_lats[iono_mask],
                transform=ccrs.Geodetic(), color=color, lw=lw, ls=ls,
                zorder=zorder)
    if TP:
        # Closest-approach (tangent) point
        v   = leo_r - gnss_r
        t_s = np.clip(-np.dot(v, gnss_r) / np.dot(v, v), 0.0, 1.0)
        tp  = gnss_r + v * t_s
        tp_lon, tp_lat, _ = _ECEF_TO_LL.transform(tp[0]*1e3, tp[1]*1e3, tp[2]*1e3)
        ax.plot(tp_lon, tp_lat, transform=ccrs.Geodetic(),
                marker="o", color=color, ms=5, mec="black", mew=0.6,
                zorder=zorder + 1, label=label)


def _parse_time_window(time_window: str) -> pd.Timestamp:
    """
    Parse a time_window string of the form "YYYY-MM-DD_HHMM" (with optional
    trailing "__orbitNN__..." suffix) into a pd.Timestamp.

    Raises ValueError if the string cannot be parsed.
    """
    # Strip any "__..." suffixes so both standard and VERIF_MH keys work.
    base = time_window.split("__")[0]          # "YYYY-MM-DD_HHMM"
    return pd.to_datetime(base, format="%Y-%m-%d_%H%M")


def _draw_terminator(ax, dt: pd.Timestamp, zorder: int = 3) -> None:
    """
    Draw the solar terminator and shade the night side on a cartopy axes.

    Uses cartopy's built-in Nightshade feature, which correctly handles all
    projections (including orthographic) and antimeridian crossings.
    """
    from cartopy.feature.nightshade import Nightshade
    import datetime as _dt
    # Nightshade expects a naive UTC datetime
    naive_dt = _dt.datetime(dt.year, dt.month, dt.day,
                            dt.hour, dt.minute, dt.second)
    ax.add_feature(Nightshade(naive_dt, alpha=0.30), zorder=zorder)


def _draw_leo_path(ax, clean_list: list, occ_colours: list,
                   zorder: int = 5) -> None:
    """
    Draw the LEO satellite ground-track for each occultation in clean_list.

    The full arc of LEO ECEF positions is converted to lon/lat and plotted
    as a thin line in the occultation's colour, with a triangle marker at
    the LEO position at the TEC-max sample.
    """
    for cl, col in zip(clean_list, occ_colours):
        LEO  = cl["LEO"]    # (3, N) km
        tec  = cl["tec"]
        i_tm = int(np.argmax(tec))

        # Full LEO ground-track
        leo_lons, leo_lats, leo_alts_m = _ECEF_TO_LL.transform(
            LEO[0] * 1e3, LEO[1] * 1e3, LEO[2] * 1e3
        )
        ax.plot(leo_lons, leo_lats,
                transform=ccrs.Geodetic(),
                color='k', lw=3.0, ls="--", alpha=0.6, zorder=zorder)

        # # Triangle at TEC-max LEO position
        # ax.plot(leo_lons[i_tm], leo_lats[i_tm],
        #         transform=ccrs.Geodetic(),
        #         marker="^", ms=7, color=col,
        #         mec="black", mew=0.6, zorder=zorder + 1)


def _plot_trim_diagnostic(
    kept:        list[dict],
    removed:     list[dict],
    lat_min:     float,
    lat_max:     float,
    lon_min:     float,
    lon_max:     float,
    region:      str,
    save_dir:    str,
    group_key:   str,
    time_window: str = "",
    eds_occ=None,
) -> str:
    """
    Globe plot showing which occultations survived the vertex-count trim and
    which were dropped — produced *before* the assimilation so it is always
    available even if the KF step fails.

    Kept occultations   : filled circle + TEC-max raypath in constellation colour.
    Removed occultations: × marker + dashed raypath in shades of red, labelled
                          with removal order (✗1 = first dropped, ✗2 = second, …).
    The convex hull of the final EDP mesh vertices is drawn as a dashed gray
    outline; the official ROI boundary is drawn in lime green.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_kept    = len(kept)
    n_removed = len(removed)
    _roi_bb = region_bounding_box(region)
    clat = (_roi_bb[0] + _roi_bb[1]) / 2.0
    clon = (_roi_bb[2] + _roi_bb[3]) / 2.0
    proj = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={"projection": proj})
    fig.suptitle(
        f"Trim Diagnostic — {group_key}\n"
        f"{n_kept} kept  ·  {n_removed} removed  "
        f"(MAX_MESH_VERTICES = {MAX_MESH_VERTICES})",
        fontsize=11,
    )

    ax.set_global()
    ax.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.5, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS.with_scale("110m"),   lw=0.3, edgecolor="lightgray")
    ax.gridlines(lw=0.3, alpha=0.4)

    # ── Terminator at the group time window ──────────────────────────────────
    try:
        _ts_tr = _parse_time_window(time_window or group_key)
        _draw_terminator(ax, _ts_tr, zorder=2)
    except Exception:
        pass

    # ── Final EDP mesh boundary (convex hull of actual mesh vertices) ────────
    if eds_occ is not None and eds_occ.geolocation.shape[0] >= 3:
        _verts = eds_occ.geolocation          # (n_geo, 2): col0=lon, col1=lat
        try:
            _hull = ConvexHull(_verts)
            _hull_verts = _verts[np.append(_hull.vertices, _hull.vertices[0])]
            ax.plot(
                _hull_verts[:, 0], _hull_verts[:, 1],
                transform=ccrs.Geodetic(),
                color="dimgray", lw=1.2, ls="--", zorder=2,
                label="Final EDP mesh",
            )
        except Exception:
            ax.plot(
                [lon_min, lon_max, lon_max, lon_min, lon_min],
                [lat_min, lat_min, lat_max, lat_max, lat_min],
                transform=ccrs.Geodetic(),
                color="dimgray", lw=1.2, ls="--", zorder=2,
                label="Final EDP bbox",
            )
    else:
        ax.plot(
            [lon_min, lon_max, lon_max, lon_min, lon_min],
            [lat_min, lat_min, lat_max, lat_max, lat_min],
            transform=ccrs.Geodetic(),
            color="dimgray", lw=1.2, ls="--", zorder=2,
            label="Final EDP bbox",
        )

    # ── Official ROI boundary (lime green) ────────────────────────────────────
    _draw_roi_boundary(ax, region)

    # ── Inner helper: draw TEC-max raypath + tangent-point marker ─────────────
    def _draw_occ(entry: dict, color, marker: str, ms: float,
                  ray_ls: str, ray_zord: int, pt_zord: int) -> None:
        LEO  = entry.get("LEO")
        GNSS = entry.get("GNSS")
        tec  = entry.get("tec")
        if LEO is not None and GNSS is not None and tec is not None and len(tec) > 0:
            idx_tm = int(np.argmax(tec))
            _draw_raypath(ax, LEO, GNSS, idx_tm,
                          color=color, ls=ray_ls, lw=1.4,
                          label=None, zorder=ray_zord)
        ax.plot(entry["tec_lon"], entry["tec_lat"],
                transform=ccrs.Geodetic(),
                marker=marker, ms=ms, color=color,
                mec="black", mew=0.7, zorder=pt_zord, linestyle="None")

    # ── Kept occultations ─────────────────────────────────────────────────────
    for entry in kept:
        prn      = entry["sat_id"][1] if entry.get("sat_id") else "?"
        const    = prn[0].upper() if prn else "?"
        cfg      = CONSTELLATION_CONFIG.get(const, {})
        col      = mpl.colormaps.get_cmap(cfg.get("cmap", _CONST_FALLBACK_CMAP))(0.70)
        is_rel   = entry.get("tec_type") == "relative"
        pt_marker = "s" if is_rel else "o"   # square for conPhs, circle for podTc2
        _draw_occ(entry, color=col, marker=pt_marker, ms=9,
                  ray_ls="solid", ray_zord=4, pt_zord=6)
        ax.annotate(prn,
                    xy=(entry["tec_lon"], entry["tec_lat"]),
                    xycoords=ccrs.Geodetic()._as_mpl_transform(ax),
                    fontsize=7, color=col, fontweight="bold",
                    xytext=(5, 4), textcoords="offset points", zorder=7)

    # ── Removed occultations (shades of red, darker = later removal) ──────────
    rem_cmap = mpl.colormaps.get_cmap("Reds").resampled(max(n_removed + 2, 4))
    for k, entry in enumerate(removed):
        prn = entry["sat_id"][1] if entry.get("sat_id") else "?"
        col = rem_cmap(0.40 + 0.50 * k / max(n_removed - 1, 1))
        _draw_occ(entry, color=col, marker="x", ms=11,
                  ray_ls="dashed", ray_zord=3, pt_zord=5)
        ax.annotate(f"✗{k + 1} {prn}",   # ✗k PRN
                    xy=(entry["tec_lon"], entry["tec_lat"]),
                    xycoords=ccrs.Geodetic()._as_mpl_transform(ax),
                    fontsize=7, color=col, fontweight="bold",
                    xytext=(5, -9), textcoords="offset points", zorder=7)

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", mfc="steelblue",
                   mec="black", ms=9, label=f"Kept — absolute TEC (podTc2)"),
            Line2D([0], [0], marker="s", color="w", mfc="steelblue",
                   mec="black", ms=9, label=f"Kept — relative TEC (conPhs)"),
            Line2D([0], [0], marker="x", color="crimson", ms=10,
                   mew=2, label=f"Removed ({n_removed})"),
            Line2D([0], [0], color="dimgray", lw=1.2, ls="--",
                   label="Final EDP mesh"),
            Line2D([0], [0], color="limegreen", lw=1.8,
                   label="ROI boundary"),
        ],
        loc="lower left", fontsize=8, framealpha=0.85,
    )

    safe_key  = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    plot_path = os.path.join(save_dir, f"group_{safe_key}{_NOISE_SUFFIX}_trim.png")
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved trim diagnostic → {plot_path}")
    return plot_path


def _plot_sequential_centre(
    step_edp_snapshots: list,
    alt_grid:           np.ndarray,
    sat_ids:            list,
    centre_idx:         int,
    save_dir:           str,
    group_key:          str,
) -> str:
    """
    Figure tracing how the centre-vertex EDP evolves as each occultation is
    assimilated sequentially into the Kalman Filter.

    Layout (1 × 2):
      Left  — EDP profiles at every sequential step, colour-coded by
               assimilation order (viridis: early = dark purple, late = yellow).
               Prior = thick black dashed.  Final posterior = thick solid.
               F2-peak circles mark (NmF2, hmF2) at each step.
      Right — F2-peak trajectory: scatter of (NmF2, hmF2) connected by
               arrows, same colour coding.  Labels show the PRN code
               (e.g. G03, E22) assimilated at each step.

    step_edp_snapshots : list of (label, edp_3d (n_height, n_geo)).
                         Index 0 = prior; index k+1 = after step k.
    sat_ids            : list of (leo_id, prn_id) — one per occultation
                         (no entry for the prior).
    """
    os.makedirs(save_dir, exist_ok=True)

    n_snaps = len(step_edp_snapshots)   # prior + N occultations
    n_occs  = n_snaps - 1

    # ── Colour assignment ─────────────────────────────────────────────────────
    # Prior (index 0) = black; steps 1..N use viridis so early assimilations
    # are dark purple and later ones are bright yellow.
    step_cmap    = mpl.colormaps.get_cmap("viridis").resampled(max(n_occs, 2))
    step_colours = (
        ["black"]
        + [step_cmap(i / max(n_occs - 1, 1)) for i in range(n_occs)]
    )

    # ── Extract centre-vertex EDP and F2 peak at each snapshot ───────────────
    centre_edps = []
    nm_list     = []
    hm_list     = []
    for _, edp_3d in step_edp_snapshots:
        ce = edp_3d[:, centre_idx]
        nm, hm = extract_robust_f2_peak(ce, alt_grid)
        centre_edps.append(ce)
        nm_list.append(nm)
        hm_list.append(hm)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_edp, ax_f2) = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle(
        f"Sequential KF Update — Centre-Vertex EDP Evolution\n{group_key}",
        fontsize=11,
    )

    ne_formatter = ScalarFormatter(useMathText=True)
    ne_formatter.set_powerlimits((-2, 2))

    # ── Left panel: EDP profiles ──────────────────────────────────────────────
    for k in range(n_snaps):
        label_snap, _ = step_edp_snapshots[k]
        col      = step_colours[k]
        is_prior = (k == 0)
        is_final = (k == n_snaps - 1)
        lw       = 2.5 if is_prior or is_final else 1.3
        ls       = "--" if is_prior else "-"
        alpha    = 1.0 if is_prior or is_final else 0.70
        zord     = 5 if is_prior or is_final else 3

        # Build a readable legend label: "Prior" or the PRN code (e.g. "G03")
        if is_prior:
            leg_lbl = "Prior"
        else:
            occ_idx  = k - 1
            prn_code = (sat_ids[occ_idx][1]
                        if occ_idx < len(sat_ids) else f"Occ {k}")
            leg_lbl  = f"{prn_code}  [final]" if is_final else prn_code

        ax_edp.plot(centre_edps[k], alt_grid, color=col, lw=lw, ls=ls,
                    alpha=alpha, label=leg_lbl, zorder=zord)

        # F2 peak marker
        if not (np.isnan(nm_list[k]) or np.isnan(hm_list[k])):
            ax_edp.plot(nm_list[k], hm_list[k],
                        marker="o", ms=7, color=col, mec="black", mew=0.6,
                        zorder=zord + 1)

    ax_edp.set_xlabel("Electron Density (m⁻³)")
    ax_edp.set_ylabel("Altitude (km)")
    ax_edp.set_title("Centre-Vertex EDP  (prior → sequential posterior)")
    ax_edp.xaxis.set_major_formatter(ne_formatter)
    ax_edp.grid(True, alpha=0.3, ls=":")
    ax_edp.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_edp.set_ylim(bottom=0)

    # Colour bar alongside the EDP panel to show step number
    sm_edp = plt.cm.ScalarMappable(
        cmap=step_cmap, norm=plt.Normalize(vmin=1, vmax=max(n_occs, 1))
    )
    sm_edp.set_array([])
    cbar_edp = fig.colorbar(sm_edp, ax=ax_edp, orientation="vertical",
                             shrink=0.85, pad=0.02)
    cbar_edp.set_label("Assimilation step", fontsize=8)
    if n_occs > 0:
        cbar_edp.set_ticks(np.arange(1, n_occs + 1))

    # ── Right panel: F2-peak trajectory ──────────────────────────────────────
    for k in range(n_snaps):
        nm = nm_list[k]
        hm = hm_list[k]
        if np.isnan(nm) or np.isnan(hm):
            continue

        col      = step_colours[k]
        is_prior = (k == 0)
        is_final = (k == n_snaps - 1)
        ms       = 12 if (is_prior or is_final) else 8
        marker   = "D" if is_prior else ("*" if is_final else "o")

        ax_f2.scatter(nm, hm, color=col, s=ms ** 2,
                      edgecolors="black", linewidths=0.7, zorder=5,
                      marker=marker)

        # Arrow from previous valid step to this step
        if k > 0:
            # Walk backwards to find the last valid predecessor
            for kp in range(k - 1, -1, -1):
                nm0, hm0 = nm_list[kp], hm_list[kp]
                if not (np.isnan(nm0) or np.isnan(hm0)):
                    ax_f2.annotate(
                        "", xy=(nm, hm), xytext=(nm0, hm0),
                        arrowprops=dict(
                            arrowstyle="->", color=col,
                            lw=1.4, alpha=0.8,
                        ),
                    )
                    break

        # Text label: "Prior", PRN code, or "[final] PRN"
        if is_prior:
            txt = "Prior"
        else:
            occ_idx  = k - 1
            prn_code = (sat_ids[occ_idx][1]
                        if occ_idx < len(sat_ids) else f"Occ {k}")
            txt = f"[final]\n{prn_code}" if is_final else prn_code
        ax_f2.annotate(txt, (nm, hm), fontsize=7,
                       textcoords="offset points", xytext=(6, 4))

    ax_f2.set_xlabel("NmF2 (m⁻³)")
    ax_f2.set_ylabel("hmF2 (km)")
    ax_f2.set_title("F2-Peak Trajectory  (prior → sequential posterior)")
    ax_f2.xaxis.set_major_formatter(ne_formatter)
    ax_f2.grid(True, alpha=0.3, ls=":")

    plt.tight_layout()
    safe_key  = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    plot_path = os.path.join(save_dir, f"group_{safe_key}{_NOISE_SUFFIX}_sequential.png")
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved sequential plot → {plot_path}")
    return plot_path


def _isr_limb_tec(
    isr_profile: dict,
    tangent_km: np.ndarray,
    R_E_km: float = 6371.0,
    topside_scale_height_m: float = 150_000.0,
    topside_H_H_m: float = 1_000_000.0,
    topside_alpha: float = 0.05,
) -> np.ndarray:
    """
    Integrate the ISR vertical Ne profile along a spherical-limb geometry to
    produce a predicted TEC profile (TECU) as a function of ray tangent altitude.

    For each tangent altitude h_t, the slant TEC is approximated as:
        TEC(h_t) = 2 * ∫_{h_t}^{h_top} Ne(h) * (R_E + h) / sqrt((R_E+h)² - (R_E+h_t)²) dh
    evaluated numerically on a fine altitude grid, with the same analytic
    exponential topside extension used by compute_isr_tec.
    """
    alt_raw = np.asarray(isr_profile["alt_km"], dtype=float)
    ne_raw  = np.asarray(isr_profile["ne"],     dtype=float)
    order   = np.argsort(alt_raw)
    alt_raw = alt_raw[order]
    ne_raw  = ne_raw[order]
    valid   = np.isfinite(ne_raw)
    alt_raw = alt_raw[valid]
    ne_raw  = ne_raw[valid]

    if len(alt_raw) < 2:
        return np.full(len(tangent_km), np.nan)

    # Build a fine regular integration grid (1 km steps) from the lowest ISR
    # altitude up through the topside extension.  Interpolating the ISR data
    # onto this grid avoids inaccuracies from the coarse/irregular raw spacing.
    h_top  = float(alt_raw[-1])
    h_bot  = float(alt_raw[0])
    H_eff  = ((1.0 - topside_alpha) * topside_scale_height_m
              + topside_alpha * topside_H_H_m) / 1000.0  # km
    ne_top = float(ne_raw[-1])

    h_ext  = np.arange(h_top, h_top + 5.0 * H_eff, 1.0)
    ne_ext = ne_top * np.exp(-(h_ext - h_top) / H_eff)

    # Dense grid over the ISR measurement range
    h_iono = np.arange(h_bot, h_top, 1.0)
    ne_iono = np.interp(h_iono, alt_raw, ne_raw)

    h_fine  = np.concatenate([h_iono, h_ext[1:]])
    ne_fine = np.concatenate([ne_iono, ne_ext[1:]])

    tec_profile = np.empty(len(tangent_km))
    for k, h_t in enumerate(tangent_km):
        mask = h_fine >= h_t
        if mask.sum() < 2:
            tec_profile[k] = 0.0
            continue
        hh   = h_fine[mask]
        nn   = ne_fine[mask]
        R_h  = R_E_km + hh          # km
        R_t  = R_E_km + h_t         # km
        denom = np.sqrt(np.maximum(R_h**2 - R_t**2, 0.0))
        # avoid div-by-zero at tangent point itself
        denom = np.where(denom < 1e-6, 1e-6, denom)
        integrand = nn * R_h / denom  # m⁻³ * (dimensionless)
        # Trapezoid in km → convert to m (*1000) for TECU (/1e16), factor 2 for both sides
        tec_profile[k] = 2.0 * float(_trapz(integrand, hh)) * 1000.0 / 1e16
    return tec_profile


def _plot_igs_stec_section(
    fig,
    gs,
    igs_entries: list,
    region: str,
    proj,
    row_start:  int         = 2,
    tec_slices: list | None = None,
    clean_list: list | None = None,
    verts_geo:  np.ndarray | None = None,
    tris_geo:   np.ndarray | None = None,
) -> None:
    """
    Render the IGS ground-station sTEC section into rows R and R+1 of *gs*.

    Panel layout (R = row_start = 2 in the 4-row IGS figure)
    ----------------------------------------------------------
    (R,   0): GPS     sTEC vs time  (measured solid, prior dashed, post dotted)
    (R+1, 0): GLONASS sTEC vs time
    (R,   1): Galileo sTEC vs time
    (R+1, 1): BeiDou  sTEC vs time
    (R:R+2, 2): Lower globe — IPP ground tracks, EDP mesh, station markers
    (R,   3): Abel Ne profiles  [placed by _plot_group, not here]
    (R+1, 3): Arc legend / summary table

    Parameters
    ----------
    fig        : parent matplotlib Figure
    gs         : GridSpec with at least (row_start + 2) rows and 4 cols
    igs_entries: list of clean-list dicts with obs_source='IGS_ground'
    region     : region key (for ROI boundary)
    proj       : cartopy projection shared with the upper globe
    row_start  : first GridSpec row for this section (default 2)
    tec_slices : KF result tec_slices — used to overlay prior / posterior
    clean_list : KF result clean_list — maps entries to tec_slices indices
    verts_geo  : (n_geo, 2) array col0=lon, col1=lat of EDP mesh vertices
    tris_geo   : triangulation indices for the EDP mesh
    """
    # Build a mapping (leo_id, prn_id, date-str) → tec_slice index so we can
    # look up prior/posterior for each IGS entry without fragile index math.
    _slice_map: dict[tuple, int] = {}
    if tec_slices is not None and clean_list is not None:
        for _i, _cl in enumerate(clean_list):
            if _cl.get("obs_source") == "IGS_ground":
                _key = (
                    str(_cl.get("leo_id", "")),
                    str(_cl.get("prn_id", "")),
                    str(_cl.get("date", "")),
                )
                _slice_map[_key] = _i
    R = row_start   # shorthand

    # ── 2×2 TEC-vs-elevation bar panels — one per GNSS constellation ─────────
    # Each IGS arc contributes one clustered triplet of bars (measured / prior /
    # posterior, arc-mean over its valid epochs) positioned at its mean
    # elevation angle, rather than a continuous sTEC-vs-time curve — this
    # reads better now that the elevation cutoff was lowered to 10° and more
    # (lower-elevation) arcs pass the filter.
    _CONST_POS_IGS = {"G": (R, 0), "R": (R+1, 0), "E": (R, 1), "C": (R+1, 1)}

    ax_igs = {}
    for const, (row, col) in _CONST_POS_IGS.items():
        cfg = CONSTELLATION_CONFIG.get(const, {"name": const, "title_color": "black"})
        ax  = fig.add_subplot(gs[row, col])
        ax.set_title(
            f"IGS sTEC — {cfg['name']}",
            fontsize=9, color=cfg["title_color"], fontweight="bold",
        )
        ax.set_xlabel("Elevation angle (deg)", fontsize=8)
        ax.set_ylabel("sTEC (TECU)", fontsize=8)
        ax.grid(True, alpha=0.3, ls=":", axis="y")
        ax_igs[const] = ax

    _bar_w       = 0.6   # degrees
    _bar_offset  = {"measured": -0.7, "prior": 0.0, "post": 0.7}
    _bar_color   = {"measured": "dimgray", "prior": "tab:blue", "post": "tab:orange"}
    _bar_label   = {"measured": "Measured sTEC", "prior": "Prior sTEC (KF)",
                     "post": "Posterior sTEC (KF)"}
    _has_bars: dict[str, bool] = defaultdict(bool)

    for ce in igs_entries:
        prn   = ce.get("prn_id", "")
        const = prn[0].upper() if prn else "?"
        ax_t  = ax_igs.get(const) or ax_igs.get("G") or next(iter(ax_igs.values()))

        elev = np.asarray(ce.get("elev_deg", []), dtype=float)
        tec  = np.asarray(ce.get("tec", []), dtype=float)
        valid = np.isfinite(elev) & np.isfinite(tec)
        if valid.sum() == 0:
            continue

        elev_mean = float(np.mean(elev[valid]))
        meas_mean = float(np.mean(tec[valid]))

        _ce_key = (
            str(ce.get("leo_id", "")),
            str(ce.get("prn_id", "")),
            str(ce.get("date", "")),
        )
        _sl = tec_slices[_slice_map[_ce_key]] if _ce_key in _slice_map else None
        prior_mean = post_mean = np.nan
        if _sl is not None:
            prior_arr = np.asarray(_sl.get("prior_tec", []), dtype=float)
            post_arr  = np.asarray(_sl.get("post_tec",  []), dtype=float)
            if len(prior_arr) == len(tec):
                prior_mean = float(np.mean(prior_arr[valid]))
            if len(post_arr) == len(tec):
                post_mean = float(np.mean(post_arr[valid]))

        ax_t.bar(elev_mean + _bar_offset["measured"], meas_mean,
                 width=_bar_w, color=_bar_color["measured"], zorder=3)
        _has_bars[const] = True
        if np.isfinite(prior_mean):
            ax_t.bar(elev_mean + _bar_offset["prior"], prior_mean,
                     width=_bar_w, color=_bar_color["prior"], alpha=0.85, zorder=3)
        if np.isfinite(post_mean):
            ax_t.bar(elev_mean + _bar_offset["post"], post_mean,
                     width=_bar_w, color=_bar_color["post"], alpha=0.85, zorder=3)

    _igs_style_entries = [
        Patch(facecolor=_bar_color["measured"], label=_bar_label["measured"]),
        Patch(facecolor=_bar_color["prior"], alpha=0.85, label=_bar_label["prior"]),
        Patch(facecolor=_bar_color["post"],  alpha=0.85, label=_bar_label["post"]),
    ]
    for const, ax_t in ax_igs.items():
        if _has_bars[const]:
            ax_t.legend(handles=_igs_style_entries, fontsize=6.5,
                        loc="upper right", framealpha=0.85)
        else:
            ax_t.text(0.5, 0.5, "No data", transform=ax_t.transAxes,
                      ha="center", va="center", color="lightgray",
                      fontsize=11, style="italic")

    # ── Globe — IPP ground tracks (col 2, rows R and R+1) ────────────────────
    ax_ipp = fig.add_subplot(gs[R:R+2, 2], projection=proj)
    ax_ipp.set_global()
    ax_ipp.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
    ax_ipp.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
    ax_ipp.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.5, edgecolor="gray")
    ax_ipp.add_feature(cfeature.BORDERS.with_scale("110m"),   lw=0.3, edgecolor="lightgray")
    ax_ipp.gridlines(lw=0.3, alpha=0.4)

    # ROI boundary (same green rectangle as the main globe)
    _draw_roi_boundary(ax_ipp, region)

    # Track recorded stations to avoid duplicate station markers in the legend.
    _station_plotted: set[str] = set()

    # Shared TEC colour scale across every IPP point drawn on this globe, so
    # marker colour reads directly as measured sTEC magnitude.
    _all_tec = (np.concatenate([
        np.asarray(ce.get("tec", []), dtype=float) for ce in igs_entries
    ]) if igs_entries else np.array([]))
    _all_tec = _all_tec[np.isfinite(_all_tec)]
    _tec_vmin = float(_all_tec.min()) if len(_all_tec) else 0.0
    _tec_vmax = float(_all_tec.max()) if len(_all_tec) else 1.0
    if _tec_vmax <= _tec_vmin:
        _tec_vmax = _tec_vmin + 1.0
    _ipp_cmap = mpl.colormaps.get_cmap("plasma")
    _ipp_norm = mcolors.Normalize(vmin=_tec_vmin, vmax=_tec_vmax)
    _ipp_sm: object = None

    for ce in igs_entries:
        station    = ce.get("leo_id", "?")
        ipp_lat    = ce.get("ipp_lat")
        ipp_lon    = ce.get("ipp_lon")
        tec        = np.asarray(ce.get("tec", []), dtype=float)
        sta_lat    = ce.get("station_lat", np.nan)
        sta_lon    = ce.get("station_lon", np.nan)
        lat_tmax   = ce.get("lat_tecmax_tangent", np.nan)
        lon_tmax   = ce.get("lon_tecmax_tangent", np.nan)

        # Faint ground track for geometry — colour now conveys TEC magnitude.
        if ipp_lat is not None and ipp_lon is not None and len(ipp_lat) > 1:
            ax_ipp.plot(
                ipp_lon, ipp_lat,
                transform=ccrs.Geodetic(),
                color="gray", lw=0.8, ls=":", alpha=0.55, zorder=4,
            )
            if len(ipp_lat) == len(tec):
                _ipp_sm = ax_ipp.scatter(
                    ipp_lon, ipp_lat, c=tec,
                    transform=ccrs.Geodetic(),
                    cmap=_ipp_cmap, norm=_ipp_norm,
                    s=12, edgecolors="none", zorder=5,
                )

        # Pierce point at TEC max (●), colour-coded on the same TEC scale
        if np.isfinite(lat_tmax) and np.isfinite(lon_tmax):
            _tmax_val = float(np.nanmax(tec)) if len(tec) else np.nan
            _tmax_col = _ipp_cmap(_ipp_norm(_tmax_val)) if np.isfinite(_tmax_val) else "black"
            ax_ipp.plot(
                lon_tmax, lat_tmax,
                transform=ccrs.Geodetic(),
                marker="o", ms=7, color=_tmax_col, mec="black", mew=0.8,
                zorder=7,
            )

        # Station location (■) — one marker and label per station
        if station not in _station_plotted and np.isfinite(sta_lat) and np.isfinite(sta_lon):
            ax_ipp.plot(
                sta_lon, sta_lat,
                transform=ccrs.Geodetic(),
                marker="s", ms=8, color="black", mec="white", mew=0.8,
                zorder=8, label=station,
            )
            try:
                ax_ipp.text(
                    sta_lon + 0.5, sta_lat + 0.5, station,
                    transform=ccrs.Geodetic(),
                    fontsize=7, color="black", fontweight="bold", zorder=9,
                )
            except Exception:
                pass
            _station_plotted.add(station)

    if _ipp_sm is not None:
        cbar_ipp = fig.colorbar(_ipp_sm, ax=ax_ipp, orientation="horizontal",
                                 shrink=0.75, pad=0.04, fraction=0.04)
        cbar_ipp.set_label("sTEC (TECU)", fontsize=7)

    # ── EDP domain mesh on the lower globe ───────────────────────────────────
    if verts_geo is not None and tris_geo is not None:
        try:
            ax_ipp.triplot(
                verts_geo[:, 0], verts_geo[:, 1], tris_geo,
                transform=ccrs.Geodetic(),
                color="steelblue", lw=0.5, alpha=0.45, zorder=3,
            )
        except Exception:
            pass

    ax_ipp.set_title(
        "IGS IPP ground tracks coloured by sTEC\n"
        "( ● = TEC-max pierce point  ■ = station )",
        fontsize=8,
    )
    if _station_plotted:
        ax_ipp.legend(loc="lower left", fontsize=7, framealpha=0.75)

    # ── Arc legend / summary table (col 3, row R+1 only) ─────────────────────
    # Row R, col 3 is occupied by Abel Ne (placed in _plot_group).
    ax_leg = fig.add_subplot(gs[R + 1, 3])
    ax_leg.axis("off")
    ax_leg.set_title("IGS arc summary", fontsize=9, fontweight="bold")

    # Build table rows: station | PRN | UTC start | TECU range
    _tbl_rows = []
    for ce in igs_entries:
        sta   = ce.get("leo_id", "?")
        prn   = ce.get("prn_id", "?")
        dt    = ce.get("date")
        tec   = ce.get("tec", np.array([]))
        t_str = dt.strftime("%H:%M") if dt is not None else "—"
        if len(tec) > 0 and np.any(np.isfinite(tec)):
            tec_rng = f"{np.nanmin(tec):.1f}–{np.nanmax(tec):.1f}"
        else:
            tec_rng = "—"
        _tbl_rows.append([sta, prn, t_str, tec_rng])

    if _tbl_rows:
        col_labels = ["Station", "PRN", "UTC", "TECU range"]
        tbl = ax_leg.table(
            cellText   = _tbl_rows,
            colLabels  = col_labels,
            loc        = "upper center",
            cellLoc    = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1.0, 1.35)
        # Header row — slightly darker background
        for col_idx in range(len(col_labels)):
            tbl[0, col_idx].set_facecolor("#d0d8e8")
            tbl[0, col_idx].set_text_props(fontweight="bold")
    else:
        ax_leg.text(0.5, 0.5, "No IGS arcs in window",
                    transform=ax_leg.transAxes, ha="center", va="center",
                    fontsize=10, color="lightgray", style="italic")


def _plot_comparison(result: dict, save_dir: str, group_key: str) -> str:
    """
    Four-panel figure comparing Sequential KF vs Joint KF for one group.

    Layout — GridSpec(2, 2):
      (0, 0)  Centre-vertex EDP profiles — prior / sequential / joint with F2 peaks.
      (0, 1)  Centre-vertex EDP update   — (seq − prior) and (joint − prior);
              horizontal dotted lines mark each method's hmF2.
      (1, 0)  TEC fit — sequential: measured (solid), prior (dashed), seq post (:).
      (1, 1)  TEC fit — joint:      measured (solid), prior (dashed), joint post (:).

    Panels (0,0) and (0,1) share the altitude y-axis.
    Panels (1,0) and (1,1) share the tangent-altitude y-axis.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Plotting comparison {group_key}")

    # ── Unpack ────────────────────────────────────────────────────────────────
    alt_grid       = result["alt_grid"]
    prior_edp      = result["prior_edp_3d"]          # (n_height, n_geo)
    seq_edp        = result["post_edp_3d"]            # sequential posterior
    jnt_edp        = result["joint_post_edp_3d"]      # joint posterior
    tec_slices_seq = result["tec_slices"]             # sequential TEC slices
    tec_slices_jnt = result["joint_tec_slices"]       # joint TEC slices
    sat_ids        = result.get("sat_ids", [])
    file_labels    = result.get("file_labels", [])
    region         = result["region"]
    eds_occ        = result["eds_occ"]
    n_occ          = len(tec_slices_seq)

    prior_rmse = result["prior_tec_rmse"]
    seq_rmse   = result["post_tec_rmse"]
    jnt_rmse   = result["joint_post_tec_rmse"]

    # ── Centre-vertex profiles ────────────────────────────────────────────────
    centre_idx   = _roi_centre_idx(eds_occ.geolocation, region)
    prior_centre = prior_edp[:, centre_idx]
    seq_centre   = seq_edp[:,   centre_idx]
    jnt_centre   = jnt_edp[:,   centre_idx]
    seq_delta    = seq_centre - prior_centre
    jnt_delta    = jnt_centre - prior_centre

    pr_nm,  pr_hm  = extract_robust_f2_peak(prior_centre, alt_grid)
    seq_nm, seq_hm = extract_robust_f2_peak(seq_centre,   alt_grid)
    jnt_nm, jnt_hm = extract_robust_f2_peak(jnt_centre,   alt_grid)

    # ── Constellation colours (same logic as _plot_group) ────────────────────
    const_counts  = defaultdict(int)
    occ_const     = []
    for i in range(n_occ):
        prn   = sat_ids[i][1] if i < len(sat_ids) else ""
        const = prn[0].upper() if prn else "?"
        occ_const.append(const)
        const_counts[const] += 1

    const_counter = defaultdict(int)
    occ_colours   = []
    for const in occ_const:
        cfg       = CONSTELLATION_CONFIG.get(const, {})
        cmap_name = cfg.get("cmap", _CONST_FALLBACK_CMAP)
        cmap_obj  = mpl.colormaps.get_cmap(cmap_name)
        n_in      = const_counts[const]
        idx_in    = const_counter[const]
        t = (0.40 + 0.50 * (idx_in / max(n_in - 1, 1))) if n_in > 1 else 0.70
        occ_colours.append(cmap_obj(t))
        const_counter[const] += 1

    # ── Figure ────────────────────────────────────────────────────────────────
    unique_leos = list(dict.fromkeys(leo for leo, _ in sat_ids)) if sat_ids else []
    leo_str     = " / ".join(unique_leos) if unique_leos else "—"

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Sequential vs Joint KF — {result['time_window']}  |  "
        f"Region: {region}  |  GN: {leo_str}  |  {n_occ} occultation(s)\n"
        f"Prior RMSE: {prior_rmse:.3f} TECU     "
        f"Sequential RMSE: {seq_rmse:.3f} TECU     "
        f"Joint RMSE: {jnt_rmse:.3f} TECU",
        fontsize=11,
    )

    gs       = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)
    ax_edp   = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[0, 1], sharey=ax_edp)
    ax_tec_s = fig.add_subplot(gs[1, 0])
    ax_tec_j = fig.add_subplot(gs[1, 1], sharey=ax_tec_s)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))

    # ── (0,0): Centre-vertex EDP profiles ────────────────────────────────────
    ax_edp.plot(prior_centre, alt_grid, color="gray",       lw=2.0, ls="--",
                label="Prior")
    ax_edp.plot(seq_centre,   alt_grid, color="steelblue",  lw=2.0,
                label="Sequential")
    ax_edp.plot(jnt_centre,   alt_grid, color="darkorange", lw=2.0,
                label="Joint")

    for nm, hm, col in (
        (pr_nm,  pr_hm,  "gray"),
        (seq_nm, seq_hm, "steelblue"),
        (jnt_nm, jnt_hm, "darkorange"),
    ):
        if not np.isnan(nm):
            ax_edp.plot(nm, hm, marker="o", ms=9, color=col, mec="black",
                        zorder=5)

    ax_edp.legend(fontsize=9, loc="upper right")
    ax_edp.set_xlabel("Electron Density (m⁻³)")
    ax_edp.set_ylabel("Altitude (km)")
    ax_edp.set_title("Centre-Vertex EDP  (● = F2 peak)")
    ax_edp.xaxis.set_major_formatter(formatter)
    ax_edp.grid(True, alpha=0.3, ls=":")
    ax_edp.set_ylim(bottom=0)

    # ── (0,1): Centre-vertex EDP update (post − prior) ───────────────────────
    ax_delta.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax_delta.plot(seq_delta, alt_grid, color="steelblue",  lw=2.0,
                  label="Seq − Prior")
    ax_delta.plot(jnt_delta, alt_grid, color="darkorange", lw=2.0,
                  label="Joint − Prior")

    # Horizontal dotted line at each method's hmF2
    for hm, col in ((seq_hm, "steelblue"), (jnt_hm, "darkorange")):
        if not np.isnan(hm):
            ax_delta.axhline(hm, color=col, lw=0.9, ls=":", alpha=0.7)

    ax_delta.legend(fontsize=9, loc="upper right")
    ax_delta.set_xlabel("ΔNe = Posterior − Prior  (m⁻³)")
    ax_delta.set_title("Centre-Vertex EDP Update")
    ax_delta.xaxis.set_major_formatter(formatter)
    ax_delta.tick_params(labelleft=False)
    ax_delta.grid(True, alpha=0.3, ls=":")

    # ── (1,0) & (1,1): TEC fit — sequential and joint ────────────────────────
    style_legend = [
        Line2D([0], [0], color="gray", lw=2.0,          label="Measured"),
        # Line2D([0], [0], color="gray", lw=1.3, ls="--", label="Prior"),
        Line2D([0], [0], color="gray", lw=1.8, ls=":",  label="KF Posterior"),
    ]
    occ_legend = []
    for i, col in enumerate(occ_colours):
        prn_code = sat_ids[i][1] if i < len(sat_ids) else f"Occ {i + 1}"
        time_str = file_labels[i].split()[-1] if i < len(file_labels) else ""
        lbl      = f"{prn_code}  ({time_str})" if time_str else prn_code
        occ_legend.append(Line2D([0], [0], color=col, lw=2.0, label=lbl))

    for i, col in enumerate(occ_colours):
        sl_s = tec_slices_seq[i]
        sl_j = tec_slices_jnt[i]
        tang = sl_s["tangent_km"]

        # Measured and prior are identical between methods — draw on both panels
        for ax_t in (ax_tec_s, ax_tec_j):
            ax_t.plot(sl_s["measured"],  tang, color=col, lw=2.0)
            # ax_t.plot(sl_s["prior_tec"], tang, color=col, lw=1.3,
            #           ls="--", alpha=0.6)

        ax_tec_s.plot(sl_s["post_tec"], tang, color=col, lw=1.8,
                      ls=":", alpha=0.95)
        ax_tec_j.plot(sl_j["post_tec"], tang, color=col, lw=1.8,
                      ls=":", alpha=0.95)

    all_alts = np.concatenate([sl["tangent_km"] for sl in tec_slices_seq])
    alt_ylim = (0, max(float(np.nanmax(all_alts)) + 50, float(alt_grid[-1])))

    ax_tec_s.legend(handles=occ_legend + style_legend, fontsize=7,
                    loc="upper right", framealpha=0.85)
    ax_tec_j.legend(handles=style_legend, fontsize=7,
                    loc="upper right", framealpha=0.85)

    for ax_t, title in (
        (ax_tec_s, f"TEC Fit — Sequential KF  (RMSE {seq_rmse:.3f} TECU)"),
        (ax_tec_j, f"TEC Fit — Joint KF       (RMSE {jnt_rmse:.3f} TECU)"),
    ):
        ax_t.set_ylim(*alt_ylim)
        ax_t.set_xlabel("TEC (TECU)")
        ax_t.set_title(title)
        ax_t.grid(True, alpha=0.3, ls=":")

    ax_tec_s.set_ylabel("Tangent Altitude (km)")
    ax_tec_j.tick_params(labelleft=False)

    safe_key  = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    plot_path = os.path.join(save_dir, f"group_{safe_key}{_NOISE_SUFFIX}_comparison.png")
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison plot → {plot_path}")
    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# §F  Global summary figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_igs_station_tec(
    igs_entries:  list,
    save_dir:     str,
    *,
    tag:          str  = "",
    ipp_alt_km:   float = 350.0,
) -> str:
    """
    Standalone two-panel figure for IGS ground-station sTEC data.

    Left panel  — sTEC (TECU) vs UTC time.
        All arcs are shown on a single shared time axis.  Lines are coloured
        by GNSS constellation (GPS=Blues, GLONASS=Purples, Galileo=Oranges,
        BeiDou=Greens) and shaded by arc index within each constellation.
        Each arc is labelled "STATION/PRN  HH:MM UTC".

    Right panel — Ionospheric pierce-point (IPP) ground tracks at *ipp_alt_km*.
        The full IPP trace is drawn as a dotted line in the arc's colour.
        A filled circle marks the IPP at the epoch of maximum sTEC.
        Each ground station is shown with a black square (■) and a name label.

    Parameters
    ----------
    igs_entries : list of clean-list dicts produced by ``igs_obs_to_clean_entry``
                  (must contain ``tec``, ``arc_time_sec``, ``date``, ``leo_id``,
                  ``prn_id``, ``ipp_lat``, ``ipp_lon``, ``lat_tecmax_tangent``,
                  ``lon_tecmax_tangent``, ``station_lat``, ``station_lon``).
    save_dir    : directory for the output PNG (created if absent).
    tag         : optional string appended to the filename before ``.png``
                  (e.g. ``"_2025154"``).
    ipp_alt_km  : shell altitude used for the IPP (for the title label only;
                  the coordinates are already stored in the clean entries).

    Returns
    -------
    Absolute path of the saved PNG.
    """
    import matplotlib.dates as mdates

    if not igs_entries:
        print("  [plot_igs_station_tec] No IGS entries — nothing to plot.")
        return ""

    os.makedirs(save_dir, exist_ok=True)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 7))
    fig.suptitle(
        f"IGS Ground-Station sTEC  ·  IPP at {ipp_alt_km:.0f} km (F2 layer)",
        fontsize=13, fontweight="bold",
    )
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1.0],
                  wspace=0.08, left=0.06, right=0.97, top=0.88, bottom=0.10)

    ax_tec = fig.add_subplot(gs[0, 0])
    ax_tec.set_xlabel("UTC time", fontsize=10)
    ax_tec.set_ylabel("sTEC (TECU)", fontsize=10)
    ax_tec.grid(True, alpha=0.3, ls=":")
    ax_tec.set_title("Slant TEC vs. time  (one arc per GNSS satellite)",
                     fontsize=9)

    # ── Projection: Orthographic centred on mean station location ─────────────
    _sta_lats = [ce.get("station_lat", np.nan) for ce in igs_entries]
    _sta_lons = [ce.get("station_lon", np.nan) for ce in igs_entries]
    _valid_lats = [v for v in _sta_lats if np.isfinite(v)]
    _valid_lons = [v for v in _sta_lons if np.isfinite(v)]
    clon = float(np.mean(_valid_lons)) if _valid_lons else 0.0
    clat = float(np.mean(_valid_lats)) if _valid_lats else 0.0
    _proj = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)

    ax_map = fig.add_subplot(gs[0, 1], projection=_proj)
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
    ax_map.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
    ax_map.add_feature(cfeature.COASTLINE.with_scale("50m"),
                       lw=0.6, edgecolor="gray")
    ax_map.add_feature(cfeature.STATES.with_scale("50m"),
                       lw=0.3, edgecolor="lightgray")
    ax_map.add_feature(cfeature.BORDERS.with_scale("50m"),
                       lw=0.3, edgecolor="lightgray")
    ax_map.gridlines(lw=0.3, alpha=0.4)
    ax_map.set_title(
        f"IPP ground tracks  ({ipp_alt_km:.0f} km)\n"
        "( : = track  ● = TEC-max pierce point  ■ = station )",
        fontsize=8,
    )

    # ── Colour assignment ─────────────────────────────────────────────────────
    _const_counts: dict[str, int] = defaultdict(int)
    for ce in igs_entries:
        prn = ce.get("prn_id", "")
        _const_counts[prn[0].upper() if prn else "?"] += 1

    _const_counter: dict[str, int] = defaultdict(int)
    _station_plotted: set[str] = set()
    _legend_handles: list = []

    # ── Draw each arc ─────────────────────────────────────────────────────────
    for ce in igs_entries:
        prn     = ce.get("prn_id", "")
        const   = prn[0].upper() if prn else "?"
        cfg     = CONSTELLATION_CONFIG.get(const, {})
        cmap    = mpl.colormaps.get_cmap(cfg.get("cmap", _CONST_FALLBACK_CMAP))
        n_in    = _const_counts[const]
        idx_in  = _const_counter[const]
        t_frac  = (0.40 + 0.50 * (idx_in / max(n_in - 1, 1))) if n_in > 1 else 0.70
        col     = cmap(t_frac)
        _const_counter[const] += 1

        station = ce.get("leo_id", "?")
        tec     = ce.get("tec", np.array([]))
        arc_t   = ce.get("arc_time_sec", np.arange(len(tec), dtype=float))
        arc_dt  = ce.get("date")         # pd.Timestamp of arc start (UTC)

        if arc_dt is not None:
            # Strip tz for matplotlib compatibility
            if getattr(arc_dt, "tzinfo", None) is not None:
                arc_dt = arc_dt.tz_convert("UTC").tz_localize(None)
            utc_times = [arc_dt + pd.Timedelta(seconds=float(s)) for s in arc_t]
        else:
            utc_times = list(arc_t / 60.0)   # fall back to minutes

        # ── Left panel: sTEC vs UTC time ──────────────────────────────────────
        lbl = (
            f"{station}/{prn}  "
            f"{arc_dt.strftime('%H:%M') if arc_dt is not None else ''} UTC"
        )
        h, = ax_tec.plot(utc_times, tec, color=col, lw=1.8, label=lbl)
        _legend_handles.append(h)

        # TEC-max marker
        if len(tec) > 0:
            i_max = int(np.argmax(tec))
            ax_tec.plot(utc_times[i_max], tec[i_max],
                        marker="o", ms=5, color=col, mec="black", mew=0.5,
                        zorder=5)

        # ── Right panel: IPP tracks ───────────────────────────────────────────
        ipp_lat  = ce.get("ipp_lat")
        ipp_lon  = ce.get("ipp_lon")
        lat_tmax = ce.get("lat_tecmax_tangent", np.nan)
        lon_tmax = ce.get("lon_tecmax_tangent", np.nan)
        sta_lat  = ce.get("station_lat", np.nan)
        sta_lon  = ce.get("station_lon", np.nan)

        # Full IPP trace
        if ipp_lat is not None and len(ipp_lat) > 1:
            ax_map.plot(
                ipp_lon, ipp_lat,
                transform=ccrs.Geodetic(),
                color=col, lw=1.6, ls=":", alpha=0.85, zorder=5,
            )

        # Pierce point at TEC max
        if np.isfinite(lat_tmax) and np.isfinite(lon_tmax):
            ax_map.plot(
                lon_tmax, lat_tmax,
                transform=ccrs.Geodetic(),
                marker="o", ms=7, color=col, mec="black", mew=0.6,
                zorder=7,
            )

        # Station marker (once per unique station)
        if station not in _station_plotted and np.isfinite(sta_lat) and np.isfinite(sta_lon):
            ax_map.plot(
                sta_lon, sta_lat,
                transform=ccrs.Geodetic(),
                marker="s", ms=9, color="black", mec="white", mew=0.9,
                zorder=8, label=station,
            )
            try:
                ax_map.text(
                    sta_lon + 0.4, sta_lat + 0.5, station,
                    transform=ccrs.Geodetic(),
                    fontsize=7.5, color="black", fontweight="bold", zorder=9,
                    clip_on=True,
                )
            except Exception:
                pass
            _station_plotted.add(station)

    # ── Left panel formatting ─────────────────────────────────────────────────
    # Format x-axis as HH:MM if we have datetime objects
    if igs_entries and _legend_handles:
        try:
            ax_tec.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax_tec.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate(rotation=30, ha="right")
        except Exception:
            ax_tec.set_xlabel("Arc time (min)", fontsize=10)

    if _legend_handles:
        ax_tec.legend(
            handles=_legend_handles,
            fontsize=7, loc="upper left",
            framealpha=0.85, ncol=max(1, len(_legend_handles) // 12),
        )

    # Constellation colour patches for quick reference in the TEC panel
    _const_patch_handles = []
    for const, cfg in CONSTELLATION_CONFIG.items():
        if _const_counts.get(const, 0) > 0:
            from matplotlib.patches import Patch
            _const_patch_handles.append(
                Patch(facecolor=mpl.colormaps.get_cmap(cfg["cmap"])(0.65),
                      edgecolor="gray", lw=0.5,
                      label=cfg["name"])
            )
    if _const_patch_handles:
        ax_tec.legend(
            handles=_legend_handles + _const_patch_handles,
            fontsize=7, loc="upper left", framealpha=0.85,
            ncol=max(1, len(_legend_handles) // 10),
        )

    # ── Map legend ────────────────────────────────────────────────────────────
    if _station_plotted:
        ax_map.legend(loc="lower left", fontsize=7.5, framealpha=0.80)

    # ── Save ──────────────────────────────────────────────────────────────────
    safe_tag  = tag.replace("/", "_").replace(" ", "_").replace(":", "")
    plot_path = os.path.join(save_dir, f"igs_stec{safe_tag}.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved IGS sTEC plot → {plot_path}")
    return plot_path


def plot_globe_all_groups(
    meta:        pd.DataFrame,
    all_results: list[dict],
    save_path:   str = "./Figures/GroupKF/globe_all_groups.png",
) -> None:
    """
    Globe plot showing every occultation's TEC-max tangent point.

    Colour encodes the number of occultations in the same group (group size),
    making it easy to identify regions of dense vs. sparse coverage.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Build a lookup: group_key → occultation count (from results list)
    count_map = {}
    for res in all_results:
        if res["status"] == "Success":
            count_map[res["group_key"]] = res["n_occultations"]

    # Map each metadata row to its group's count
    meta = meta.copy()
    meta["group_count"] = meta["group_key"].map(count_map).fillna(1).astype(int)
    max_count = int(meta["group_count"].max())

    # ── Figure ────────────────────────────────────────────────────────────────
    proj = ccrs.Robinson()
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), subplot_kw={"projection": proj})
    ax.set_global()
    ax.add_feature(cfeature.LAND,      facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="aliceblue", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.5)
    ax.gridlines(lw=0.3, alpha=0.4, draw_labels=False)

    # Colour map: low count (blue/cool) → high count (red/warm)
    cmap_count = mpl.colormaps.get_cmap("plasma")
    norm       = plt.Normalize(vmin=1, vmax=max(max_count, 2))

    sc = ax.scatter(
        meta["lon"].values,
        meta["lat"].values,
        c=meta["group_count"].values,
        cmap=cmap_count,
        norm=norm,
        s=60,
        edgecolors="black",
        linewidths=0.4,
        transform=ccrs.Geodetic(),
        zorder=5,
    )

    # Draw mid-latitude region grid lines (20° lat, 50° lon cells)
    for lat_line in np.arange(-90, 91, DLAT_MID):
        if abs(lat_line) <= POLAR_LAT_THRESHOLD:
            ax.plot(
                [-180, 180], [lat_line, lat_line],
                transform=ccrs.PlateCarree(),
                lw=0.6, ls="--", color="steelblue", alpha=0.4, zorder=1,
            )
    for lon_line in np.arange(-180, 181, DLON_MID):
        ax.plot(
            [lon_line, lon_line], [-POLAR_LAT_THRESHOLD, POLAR_LAT_THRESHOLD],
            transform=ccrs.PlateCarree(),
            lw=0.6, ls="--", color="steelblue", alpha=0.4, zorder=1,
        )

    # Polar cap boundaries
    for pole_lat in (POLAR_LAT_THRESHOLD, -POLAR_LAT_THRESHOLD):
        ax.plot(
            np.linspace(-180, 180, 360), [pole_lat] * 360,
            transform=ccrs.PlateCarree(),
            lw=1.2, ls="-", color="darkorange", alpha=0.7, zorder=2,
        )

    cbar = fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.025, pad=0.04)
    cbar.set_label("Occultations in Group", fontsize=11)
    cbar.set_ticks(range(1, max_count + 1))

    n_groups  = len(set(meta["group_key"]))
    n_success = sum(1 for r in all_results if r["status"] == "Success")
    ax.set_title(
        f"All Occultation TEC-Max Tangent Points — Colour by Group Count\n"
        f"{len(meta)} occultations  |  {n_groups} groups  |  "
        f"{n_success} groups processed successfully\n"
        f"Grid: 20°lat × 50°lon mid-lat bins  |  orange lines = polar caps",
        fontsize=12,
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGlobe plot saved → {save_path}")


def plot_tec_all_groups(
    all_results: list[dict],
    save_path:   str = "./Figures/GroupKF/tec_all_groups.png",
    max_panels:  int = 16,
) -> None:
    """
    Grid of TEC profile panels — one panel per successfully processed group.

    Within each panel every occultation's measured TEC is drawn in a unique
    colour from the plasma colormap.  Prior and posterior traces are overlaid
    in dashed and solid lines respectively.  If there are more than `max_panels`
    groups the figure is split into pages (one PNG per page).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    success_results = [r for r in all_results if r.get("status") == "Success"
                       and r.get("tec_slices")]
    if not success_results:
        print("  No successful groups to plot.")
        return

    # ── Layout ────────────────────────────────────────────────────────────────
    n_total = len(success_results)
    n_cols  = min(4, n_total)
    n_rows  = int(np.ceil(min(n_total, max_panels) / n_cols))
    page    = 0
    idx     = 0

    while idx < n_total:
        chunk        = success_results[idx : idx + n_cols * n_rows]
        n_this_page  = len(chunk)
        n_cols_page  = min(n_cols, n_this_page)
        n_rows_page  = int(np.ceil(n_this_page / n_cols_page))

        fig, axes = plt.subplots(
            n_rows_page, n_cols_page,
            figsize=(5 * n_cols_page, 5 * n_rows_page),
            squeeze=False,
        )
        fig.suptitle(
            f"Grouped KF: TEC Profiles — Page {page + 1}\n"
            f"(measured = solid, prior = dashed, posterior = dotted)",
            fontsize=13,
        )

        for panel_idx, res in enumerate(chunk):
            row_i = panel_idx // n_cols_page
            col_i = panel_idx  % n_cols_page
            ax    = axes[row_i][col_i]

            tec_slices  = res["tec_slices"]
            file_labels = res.get("file_labels", [])
            n_occ       = len(tec_slices)
            cmap_occ    = mpl.colormaps.get_cmap("plasma").resampled(max(n_occ, 2))
            occ_colours = [cmap_occ(i / max(n_occ - 1, 1)) for i in range(n_occ)]

            for i, (sl, col) in enumerate(zip(tec_slices, occ_colours)):
                lbl = file_labels[i] if i < len(file_labels) else f"Occ {i+1}"
                ax.plot(sl["measured"],  sl["tangent_km"], color=col,  lw=1.8, label=lbl)
                ax.plot(sl["prior_tec"], sl["tangent_km"], color=col,  lw=1.0, ls="--", alpha=0.5)
                ax.plot(sl["post_tec"],  sl["tangent_km"], color=col,  lw=1.0, ls=":",  alpha=0.8)

            ax.set_title(
                f"{res['region']}\n{res['time_window']}  ({n_occ} occ)",
                fontsize=8,
            )
            ax.set_xlabel("TEC (TECU)", fontsize=8)
            ax.set_ylabel("Tang. Alt (km)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3, ls=":")

            all_alts = np.concatenate([sl["tangent_km"] for sl in tec_slices])
            ax.set_ylim(0, max(np.nanmax(all_alts) + 30, res["alt_grid"][-1]))

            if n_occ <= 6:
                ax.legend(fontsize=6, loc="upper right", framealpha=0.7)

        # Hide empty panels
        for empty in range(n_this_page, n_rows_page * n_cols_page):
            axes[empty // n_cols_page][empty % n_cols_page].set_visible(False)

        plt.tight_layout()
        base, ext = os.path.splitext(save_path)
        page_path = f"{base}_p{page + 1:02d}{ext}"
        fig.savefig(page_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  TEC panel page {page + 1} saved → {page_path}")

        idx  += n_cols * n_rows
        page += 1


# ─────────────────────────────────────────────────────────────────────────────
# §G  Main entry point — analogous to main() in demo.py
# ─────────────────────────────────────────────────────────────────────────────

def demo_group_main() -> None:
    """
    Run the full grouped-occultation Kalman pipeline on a day's podTc2 data.

    Workflow mirrors main() in demo.py:
      1.  Scan metadata from all podTc2 files in `base_path`.
      2.  Group by 30-minute time window × geographic region.
      3.  Build global EDP prior grids (reuses build_daily_global_edps from demo.py).
      4.  Process each group with a joint KF update (process_group).
      5.  Write globe plot and TEC panel grid to disk.
    """
    import datetime

    # ── User-configurable settings ─────────────────────────────────────────────
    DOY = 153
    YYYY = 2025
    base_path    = f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/{YYYY}.{DOY}/"
    # alt_grid     = np.arange(60.0, 600.0, 10.0, dtype=float)
    # TYPE = "linear"
    # Extended to 800 km so that high-tangent-altitude rays (LEO ~580-590 km,
    # tangent up to ~595 km) have many in-grid midpoints rather than relying
    # almost entirely on the single exponential topside surrogate.  With the
    # old 600 km ceiling only ~12/999 midpoints were in-grid for the top rays;
    # at 800 km that rises to ~100+ and the IRI density at 600-800 km is
    # captured directly in the state vector.
    alt_grid = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)
    TYPE = "log"
    save_dir     = "./Figures/GroupKF/"
    num_workers  = 12       # parallel workers for global EDP build
    max_groups   = None     # set to an int to process only the first N groups (None = all)
    kf_config    = {
        "measurement_err": 10.0,
        "relaxation":      0.99,
        "podtc_max_rays":  200,
        "conphs_max_rays": 200,
    }
    # ──────────────────────────────────────────────────────────────────────────

    if not os.path.exists(base_path):
        print(f"ERROR: base_path not found: {base_path}")
        return

    print("=" * 65)
    print("  demo_group.py — Grouped Occultation KF Processing")
    print("=" * 65)

    # ── Step 1: Scan metadata ─────────────────────────────────────────────────
    meta = scan_metadata(base_path)
    if meta.empty:
        print("No valid podTc2 files found.  Exiting.")
        return

    # ── Step 2: Derive calendar date from directory name ──────────────────────
    year_doy    = base_path.rstrip("/").split("/")[-1]
    year, doy   = map(int, year_doy.split("."))
    batch_date  = pd.Timestamp(
        datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
    )
    print(f"\nBatch date     : {batch_date.date()}")
    print(f"Time bins      : {WINDOW_MINUTES}-minute windows")
    print(f"Lat bin size   : {DLAT_MID}°  |  Lon bin size: {DLON_MID}°")
    print(f"Polar cap lat  : ±{POLAR_LAT_THRESHOLD}°")

    # ── Step 3: Build global EDP prior grids for the day ─────────────────────
    global_edp_data_dir = f"./Data/Global_EDPS_{DOY}_{TYPE}/"
    print(f"\nBuilding global EDP cache for {batch_date.date()} …")
    global_edp_cache = build_daily_global_edps(
        batch_date, alt_grid,
        dLat=5.0, dLon=5.0,
        num_workers=num_workers,
        data_dir=global_edp_data_dir,
    )
    print("Global EDP cache ready.\n")

    # ── Step 4: Process each group ────────────────────────────────────────────
    groups       = meta.groupby("group_key", sort=False)
    group_keys   = list(groups.groups.keys())
    print(f"Processing {len(group_keys)} groups …")

    if max_groups is not None:
        group_keys = group_keys[:max_groups]
        print(f"  (limited to first {max_groups} groups)")

    all_results = []
    for g_idx, gk in enumerate(group_keys[329:]):
        print(f"\n[{g_idx + 1}/{len(group_keys)}]", end="")
        gm  = groups.get_group(gk)
        res = process_group(
            group_key        = gk,
            group_meta       = gm,
            alt_grid         = alt_grid,
            global_edp_cache = global_edp_cache,
            generate_plots   = True,
            save_dir         = save_dir,
            **kf_config,
        )
        all_results.append(res)

        # Light rolling backup every 20 groups
        if (g_idx + 1) % 20 == 0:
            _save_stats_csv(all_results, year, doy)

    # ── Step 5: Summary statistics ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Batch complete.  Compiling statistics …")
    print("=" * 65)

    stats_csv = _save_stats_csv(all_results, year, doy)
    print(f"Stats CSV saved → {stats_csv}")

    success = [r for r in all_results if r["status"] == "Success"]
    print(f"\nSuccessfully processed {len(success)} / {len(all_results)} groups.")

    if success:
        rmse_prior = np.nanmean([r["prior_tec_rmse"] for r in success])
        rmse_post  = np.nanmean([r["post_tec_rmse"]  for r in success])
        imprv      = (rmse_prior - rmse_post) / rmse_prior * 100.0
        print(f"  Mean prior  RMSE : {rmse_prior:.3f} TECU")
        print(f"  Mean post   RMSE : {rmse_post:.3f} TECU")
        print(f"  Mean improvement : {imprv:.1f} %")
        occ_counts = [r["n_occultations"] for r in success]
        print(f"  Occultations/group: mean={np.mean(occ_counts):.1f}, "
              f"max={max(occ_counts)}, min={min(occ_counts)}")

    # ── Step 6: Summary figures ───────────────────────────────────────────────
    print("\nGenerating summary figures …")
    os.makedirs(save_dir, exist_ok=True)

    plot_globe_all_groups(
        meta,
        all_results,
        save_path=os.path.join(save_dir, "globe_all_groups.png"),
    )

    plot_tec_all_groups(
        all_results,
        save_path=os.path.join(save_dir, "tec_all_groups.png"),
        max_panels=16,
    )

    print("\nAll figures written.  Done.")


def _save_stats_csv(all_results: list[dict], year: int, doy: int) -> str:
    """Serialise a flat summary DataFrame to CSV and return the file path."""
    rows = []
    for r in all_results:
        rows.append({
            "group_key":        r["group_key"],
            "region":           r["region"],
            "time_window":      r["time_window"],
            "n_occultations":   r["n_occultations"],
            "files":            "|".join(r.get("files", [])),
            "status":           r["status"],
            "prior_tec_rmse":   r.get("prior_tec_rmse", np.nan),
            "post_tec_rmse":    r.get("post_tec_rmse",  np.nan),
            "processing_time_s": r.get("processing_time_s", np.nan),
        })
    df       = pd.DataFrame(rows)
    csv_name = f"GroupKF_Stats_{year}_{doy:03d}.csv"
    df.to_csv(csv_name, index=False)
    return csv_name


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_group_main()
