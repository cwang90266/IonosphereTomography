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
matplotlib.use("Agg")                    # non-interactive for batch runs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import time
import gc

import netCDF4                           # used by the lightweight metadata scan
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj

# ── Project imports ───────────────────────────────────────────────────────────
from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
from EDPSamples.edp_samples import EDPSamples
from Ionosphere_Tomography_Inverter.Ionophy_Tomography_Inverter import (
    Ionosphere_Tomography_Inverter,
)
# Re-use the global EDP builder and F2-peak extractor already in demo.py.
from demo import build_daily_global_edps, extract_robust_f2_peak

# ─────────────────────────────────────────────────────────────────────────────
# Grouping parameters (adjust as needed)
# ─────────────────────────────────────────────────────────────────────────────

POLAR_LAT_THRESHOLD = 60.0   # |lat| above this → polar cap
DLAT_MID            = 30.0   # mid-latitude bin height (degrees)
DLON_MID            = 60.0   # mid-latitude bin width  (degrees)
WINDOW_MINUTES      = 30     # time-bin width in minutes
MAX_MESH_VERTICES   = 300    # trim group until EDP mesh has ≤ this many vertices

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
) -> tuple[float, float, float, float]:
    """
    Compute a bounding box covering all occultations in `parsed_list` using
    the same approach as demo.py's section20():

      1. Call EDPSamples.get_occultation_extrema for each occultation to
         obtain three ray-path corner points (top, TEC-max, bottom tangent).
      2. Merge all corner lat/lon values across occultations.
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


# ─────────────────────────────────────────────────────────────────────────────
# §D  Core: process one geographic group with a joint KF update
# ─────────────────────────────────────────────────────────────────────────────

def process_group(
    group_key:        str,
    group_meta:       pd.DataFrame,
    alt_grid:         np.ndarray,
    global_edp_cache: dict,
    measurement_err:  float = 10.0,
    relaxation:       float = 0.99,
    generate_plots:   bool  = True,
    save_dir:         str   = "./Figures/GroupKF/",
    run_sequential:   bool  = True,
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
        for _, row in group_meta.iterrows():
            data = parse_podTc2_nc_file(row["full_path"])
            if data is None:
                print(f"    [skip] {row['filename']} — parse returned None")
                continue
            parsed_list.append(data)
            file_labels.append(f"{row['spacecraft']} {row['date'].strftime('%H:%M')}")

        if len(parsed_list) < 1:
            result["status"] = "No Valid Files"
            return result

        print(f"  Parsed {len(parsed_list)}/{n_occ} files successfully.")

        # ── 2. Build clean (NaN-free, positive TEC) arrays for each occ ────
        # Keep labels, original parsed dict, and satellite IDs all paired
        # with clean entries so later stages (Abel, plotting) have full info.
        clean_list   = []    # list of dicts: {'tec', 'tangent_km', 'LEO', 'GNSS'}
        clean_labels = []    # display label for each entry in clean_list
        clean_parsed = []    # full podTc_data dict for Abel inversion
        clean_sat_ids = []   # (leo_id, prn_id) per entry — receiver × transmitter
        for i, data in enumerate(parsed_list):
            _, _, tang_raw = rayTangent(data["LEO"], data["GNSS"], units="km")
            tang_km  = tang_raw * 1e-3
            meas_tec = data.get("TEC_podTc2", data.get("TEC", np.zeros_like(tang_km)))
            valid    = ~np.isnan(meas_tec) & (meas_tec > 0)

            if valid.sum() < 50:
                print(f"    [skip] {file_labels[i]} — only {valid.sum()} valid rays")
                continue

            clean_list.append({
                "tec":        np.asarray(meas_tec[valid], dtype=np.float64).flatten(),
                "tangent_km": tang_km[valid].flatten(),
                "LEO":        data["LEO"][:,  valid],
                "GNSS":       data["GNSS"][:, valid],
            })
            clean_labels.append(file_labels[i])
            clean_parsed.append(data)
            # Build full transmitter code: constellation letter (conid) + PRN number
            # e.g. conid="G", prn_id="03"  →  full_prn="G03"
            #      conid="E", prn_id="22"  →  full_prn="E22"
            leo_id   = str(data.get("leo_id", "??")).strip()
            con_id   = str(data.get("conid",  "?")).strip()
            prn_num  = str(data.get("prn_id", "??")).strip()
            full_prn = f"{con_id}{prn_num}"
            clean_sat_ids.append((leo_id, full_prn))

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
        _BBOX_MARGIN = 1.0    # degrees of padding around ray-path footprints
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
            }
            for i in range(len(clean_list))
        ]
        _trim_removed: list[dict] = []   # remains empty for polar groups

        if _is_polar:
            # ── Polar: shrink equatorward boundary, keep all occultations ────
            # Compute the initial bbox from the full occultation set.
            lat_min, lat_max, lon_min, lon_max = _group_bounding_box(
                clean_parsed, alt_grid, region, margin_deg=_BBOX_MARGIN
            )
            while True:
                eds_occ = global_edp_cache[profile_hour].subset_region(
                    lat_min, lat_max, lon_min, lon_max
                )
                n_geo = eds_occ.geolocation.shape[0]
                if n_geo <= MAX_MESH_VERTICES:
                    print(f"  [polar] Bounding box: lat [{lat_min:.1f}, {lat_max:.1f}]  "
                          f"lon [{lon_min:.1f}, {lon_max:.1f}]  |  vertices: {n_geo}")
                    break
                # Tighten the equatorward edge by _SHRINK_STEP
                if region == "POLAR_N":
                    new_eq = lat_min + _SHRINK_STEP
                    if new_eq >= 89.0:
                        print(f"  [polar-shrink] Hit poleward limit at lat_min=89°, "
                              f"accepting {n_geo} vertices.")
                        break
                    lat_min = new_eq
                else:  # POLAR_S
                    new_eq = lat_max - _SHRINK_STEP
                    if new_eq <= -89.0:
                        print(f"  [polar-shrink] Hit poleward limit at lat_max=-89°, "
                              f"accepting {n_geo} vertices.")
                        break
                    lat_max = new_eq
                # print(f"  [polar-shrink] n_geo={n_geo} > {MAX_MESH_VERTICES} — "
                #       f"tightening equatorward edge → "
                #       f"lat_min={lat_min:.1f}, lat_max={lat_max:.1f}")

        else:
            # ── Mid-lat: remove farthest-outlier occultation each iteration ──
            while True:
                lat_min, lat_max, lon_min, lon_max = _group_bounding_box(
                    clean_parsed, alt_grid, region, margin_deg=_BBOX_MARGIN
                )
                eds_occ = global_edp_cache[profile_hour].subset_region(
                    lat_min, lat_max, lon_min, lon_max
                )
                n_geo = eds_occ.geolocation.shape[0]
                print(f"  Bounding box: lat [{lat_min:.1f}, {lat_max:.1f}]  "
                      f"lon [{lon_min:.1f}, {lon_max:.1f}]  |  vertices: {n_geo}")

                if n_geo <= MAX_MESH_VERTICES or len(clean_list) <= 1:
                    break

                _clats  = np.array([c[0] for c in occ_centers])
                _clons  = np.array([c[1] for c in occ_centers])
                _gc_lat = float(np.mean(_clats))
                _gc_lon = float(np.mean(_clons))
                _dist   = (_clats - _gc_lat) ** 2 + (_clons - _gc_lon) ** 2
                worst   = int(np.argmax(_dist))
                print(f"  [trim] n_geo={n_geo} > MAX_MESH_VERTICES={MAX_MESH_VERTICES} — "
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
              f"{n_geo} mesh vertices")

        # ── 5b. Trim diagnostic plot (before assimilation) ───────────────────
        if generate_plots:
            try:
                _plot_trim_diagnostic(
                    kept      = _trim_info,
                    removed   = _trim_removed,
                    lat_min   = lat_min,
                    lat_max   = lat_max,
                    lon_min   = lon_min,
                    lon_max   = lon_max,
                    region    = region,
                    save_dir  = save_dir,
                    group_key = group_key,
                )
            except Exception as _exc_trim:
                print(f"  [warn] Trim diagnostic plot failed: {_exc_trim}")

        # ── 6. Build ONE inverter from the combined mesh ─────────────────────
        # All individual H matrices share the same state vector.
        # topside_prior_floor_tecu guards against IRI's lack of a plasmasphere:
        # when IRI clips ne to the physical floor at 800 km, x_top_prior collapses
        # to ~0.002 TECU, making the forward TEC zero for high-tangent-altitude rays.
        inverter = Ionosphere_Tomography_Inverter(
            EDPSam=eds_occ, meanscale=1, topside_prior_floor_tecu=1.0
        )
        _n_sv    = inverter.attrs["n_state_vars"]

        # ── 7. Build H matrix and observation vector for each occultation ───
        H_blocks  = []     # one (n_rays_i, n_state_aug) array per occultation
        tec_obs   = []     # concatenated measured TEC (TECU)
        ray_counts = []    # number of valid rays per occultation (for slicing later)

        for i, cl in enumerate(clean_list):
            print(f"  [{i+1}/{len(clean_list)}] Building H for {clean_labels[i]} "
                  f"({cl['tec'].shape[0]} rays) …")
            H_i = inverter.get_observation_operator({"LEO": cl["LEO"], "GNSS": cl["GNSS"]})
            H_blocks.append(H_i)
            tec_obs.append(cl["tec"])
            ray_counts.append(len(cl["tec"]))

        # Stack all H matrices and obs for prior TEC computation and final
        # posterior TEC over all observations (stacking is still needed for the
        # RMSE stats and per-occ TEC slices in the summary figure).
        #   H_joint  : (sum_rays, n_state_aug)
        #   obs_joint: (sum_rays,)
        H_joint   = np.vstack(H_blocks).astype(np.float32)
        obs_joint = np.concatenate(tec_obs).astype(np.float64)
        print(f"  Combined H shape : {H_joint.shape}  "
              f"(total rays = {len(obs_joint)})")

        # ── 8. Prior TEC (before any assimilation) ───────────────────────────
        prior_state_flat = inverter.attrs["initial_edps_mean"]   # (n_sv, 1)
        x_top_prior      = inverter.attrs["x_top_prior"]         # (n_geo,)
        prior_tec_joint  = (
            H_joint[:, :_n_sv] @ prior_state_flat
            + H_joint[:, _n_sv:] @ x_top_prior[:, None]
        ).flatten()

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
                inverter.x[_n_sv:] = (x_top_tecu_cur - x_top_prior_new)[:, None]
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
            EDPSam=eds_occ, meanscale=1, topside_prior_floor_tecu=1.0
        )
        posterior_state_flat_jnt = inverter_jnt.assimilate(
            obs             = obs_joint,
            obs_operator    = H_joint.astype(np.float32),
            relaxation      = relaxation,
            measurement_err = measurement_err,
        )

        # ── 11. Posterior TEC ─────────────────────────────────────────────────
        post_tec_jnt = (
            H_joint[:, :_n_sv] @ np.asarray(posterior_state_flat_jnt)
            + H_joint[:, _n_sv:] @ inverter_jnt.x_top_tecu
        ).flatten()

        if run_sequential:
            post_tec_seq = (
                H_joint[:, :_n_sv] @ np.asarray(posterior_state_flat)
                + H_joint[:, _n_sv:] @ inverter.x_top_tecu
            ).flatten()
        else:
            post_tec_seq = post_tec_jnt   # alias — seq stats not reported

        # ── 12. Per-group residual statistics ────────────────────────────────
        prior_res    = obs_joint - prior_tec_joint
        post_res_jnt = obs_joint - post_tec_jnt
        result["prior_tec_rmse"]      = float(np.sqrt(np.mean(prior_res**2)))
        result["joint_post_tec_rmse"] = float(np.sqrt(np.mean(post_res_jnt**2)))
        if run_sequential:
            post_res_seq = obs_joint - post_tec_seq
            result["post_tec_rmse"] = float(np.sqrt(np.mean(post_res_seq**2)))
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
        for i, data in enumerate(clean_parsed):
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
        result["status"]               = "Success"

        # ── 15. Optional per-group diagnostic plots ──────────────────────────
        if generate_plots:
            centre_idx_plt = _roi_centre_idx(eds_occ.geolocation, region)

            # ── Sequential summary figure (only when sequential was run) ─────
            if run_sequential:
                result["plot_path"] = _plot_group(
                    result, save_dir=save_dir, group_key=group_key,
                    suffix="_seq", mode_label="Sequential KF",
                )

            # ── Joint summary figure ─────────────────────────────────────────
            result_jnt = dict(result)
            result_jnt["post_edp_3d"]   = result["joint_post_edp_3d"]
            result_jnt["tec_slices"]    = tec_slices_jnt
            result_jnt["post_tec_rmse"] = result["joint_post_tec_rmse"]
            result["joint_plot_path"] = _plot_group(
                result_jnt, save_dir=save_dir, group_key=group_key,
                suffix="_joint", mode_label="Joint KF",
            )

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
                  label: str | None, zorder: int = 6) -> None:
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
    # Closest-approach (tangent) point
    v   = leo_r - gnss_r
    t_s = np.clip(-np.dot(v, gnss_r) / np.dot(v, v), 0.0, 1.0)
    tp  = gnss_r + v * t_s
    tp_lon, tp_lat, _ = _ECEF_TO_LL.transform(tp[0]*1e3, tp[1]*1e3, tp[2]*1e3)
    ax.plot(tp_lon, tp_lat, transform=ccrs.Geodetic(),
            marker="o", color=color, ms=5, mec="black", mew=0.6,
            zorder=zorder + 1, label=label)


def _draw_roi_boundary(ax, region_key: str) -> None:
    """
    Draw the geographic region of interest on a cartopy axes.

    Mid-latitude bins  : rectangle formed by four PlateCarree line segments.
    Polar caps         : latitude boundary circle at ±POLAR_LAT_THRESHOLD.
    """
    kw = dict(transform=ccrs.PlateCarree(),
              color="lime", lw=2.0, ls="-", zorder=3, alpha=0.9)

    if region_key == "POLAR_N":
        lons = np.linspace(-180, 180, 361)
        ax.plot(lons, [POLAR_LAT_THRESHOLD] * 361, **kw,
                label=f"ROI: North polar cap (>{POLAR_LAT_THRESHOLD:.0f}°N)")
    elif region_key == "POLAR_S":
        lons = np.linspace(-180, 180, 361)
        ax.plot(lons, [-POLAR_LAT_THRESHOLD] * 361, **kw,
                label=f"ROI: South polar cap (<{-POLAR_LAT_THRESHOLD:.0f}°N)")
    else:
        lat0, lat1, lon0, lon1 = region_bounding_box(region_key)
        n_pts = 60
        lats_lr = np.linspace(lat0, lat1, n_pts)
        lons_tb = np.linspace(lon0, lon1, n_pts)
        ax.plot(lons_tb,      [lat0] * n_pts, **kw)            # south edge
        ax.plot(lons_tb,      [lat1] * n_pts, **kw)            # north edge
        ax.plot([lon0] * n_pts, lats_lr,      **kw)            # west edge
        ax.plot([lon1] * n_pts, lats_lr,      **kw,            # east edge + label
                label=f"ROI: {lat0:.0f}–{lat1:.0f}°N  {lon0:.0f}–{lon1:.0f}°E")


def _roi_centre_idx(verts_geo: np.ndarray, region_key: str) -> int:
    """
    Return the index of the EDP mesh vertex closest to the geographic
    centre of the official region-of-interest bounding box.

    ``verts_geo`` has shape (n_geo, 2) with col-0 = longitude, col-1 = latitude.
    The ROI centre is the midpoint of ``region_bounding_box(region_key)``.
    Distance is measured in plain lat/lon degrees (adequate for a small patch).
    """
    lat_min, lat_max, lon_min, lon_max = region_bounding_box(region_key)
    roi_clat = (lat_min + lat_max) / 2.0
    roi_clon = (lon_min + lon_max) / 2.0
    dlat = verts_geo[:, 1] - roi_clat
    dlon = verts_geo[:, 0] - roi_clon
    # if region_key == "POLAR_N":
        
    # if region_key == "POLAR_S":    
        
    return int(np.argmin(dlat ** 2 + dlon ** 2))


def _plot_trim_diagnostic(
    kept:     list[dict],
    removed:  list[dict],
    lat_min:  float,
    lat_max:  float,
    lon_min:  float,
    lon_max:  float,
    region:   str,
    save_dir: str,
    group_key: str,
) -> str:
    """
    Globe plot showing which occultations survived the vertex-count trim and
    which were dropped — produced *before* the assimilation so it is always
    available even if the KF step fails.

    Kept occultations   : filled circle + TEC-max raypath in constellation colour.
    Removed occultations: × marker + dashed raypath in shades of red, labelled
                          with removal order (✗1 = first dropped, ✗2 = second, …).
    The final EDP bounding box is drawn as a dashed gray rectangle; the official
    ROI boundary is drawn in lime green.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_kept    = len(kept)
    n_removed = len(removed)
    clon = (lon_min + lon_max) / 2.0
    clat = (lat_min + lat_max) / 2.0
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

    # ── Final EDP bounding box (dashed gray rectangle) ────────────────────────
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
        prn   = entry["sat_id"][1] if entry.get("sat_id") else "?"
        const = prn[0].upper() if prn else "?"
        cfg   = CONSTELLATION_CONFIG.get(const, {})
        col   = mpl.colormaps.get_cmap(cfg.get("cmap", _CONST_FALLBACK_CMAP))(0.70)
        _draw_occ(entry, color=col, marker="o", ms=9,
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
                   mec="black", ms=9, label=f"Kept ({n_kept})"),
            Line2D([0], [0], marker="x", color="crimson", ms=10,
                   mew=2, label=f"Removed ({n_removed})"),
            Line2D([0], [0], color="dimgray", lw=1.2, ls="--",
                   label="Final EDP bbox"),
            Line2D([0], [0], color="limegreen", lw=1.8,
                   label="ROI boundary"),
        ],
        loc="lower left", fontsize=8, framealpha=0.85,
    )

    safe_key  = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    plot_path = os.path.join(save_dir, f"group_{safe_key}_trim.png")
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
    plot_path = os.path.join(save_dir, f"group_{safe_key}_sequential.png")
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved sequential plot → {plot_path}")
    return plot_path


def _plot_group(
    result: dict,
    save_dir: str,
    group_key: str,
    *,
    suffix: str = "",
    mode_label: str = "Sequential KF",
    isr_profiles: list | None = None,
    isr_site: tuple[float, float] | None = None,
) -> str:
    """
    Write a summary figure for one geographic group.

    Parameters
    ----------
    result      : dict returned / built by process_group.
    save_dir    : directory where the PNG is written.
    group_key   : human-readable group identifier (used in the filename).
    suffix      : appended to the base filename before ".png"  (e.g. "_seq", "_joint").
    mode_label  : KF mode string shown in the figure suptitle.

    Layout — GridSpec(2, 4):
        Cols 0–1, rows 0–1 : 2×2 TEC profile panels, one per GNSS constellation.
                             GPS (top-left / Blues), GLONASS (bottom-left / Purples),
                             Galileo (top-right / Oranges), BeiDou (bottom-right / Greens).
                             Within each panel the shade deepens with occultation index.
                             Legend entries use the GNSS PRN code (G03, E22, R13, C36).
        Col 2, rows 0–1    : Globe — ΔNe coolwarm map at posterior hmF2, ROI boundary,
                             per-occultation raypaths (top/TEC-max/bottom), centre ★.
        Col 3, rows 0–1    : EDP — prior/posterior spaghetti, centre-column profiles,
                             F2-peak markers, Abel Ne profiles in constellation colours.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Unpack result fields ──────────────────────────────────────────────────
    tec_slices  = result["tec_slices"]
    file_labels = result["file_labels"]
    sat_ids     = result.get("sat_ids", [])          # list of (leo_id, prn_id)
    alt_grid    = result["alt_grid"]
    prior_edp   = result["prior_edp_3d"]
    post_edp    = result["post_edp_3d"]
    eds_occ     = result["eds_occ"]
    clean_list  = result.get("clean_list", [])
    abel_list   = result.get("abel_list", [None] * len(tec_slices))
    region      = result["region"]
    n_occ       = len(tec_slices)

    # ── Pre-compute centre-column and F2 peaks (needed by Panels 2 & 3) ──────
    verts_geo  = eds_occ.geolocation               # (n_geo, 2): col0=lon, col1=lat
    tris_geo   = eds_occ.mesh
    n_verts    = verts_geo.shape[0]
    # Centre vertex = mesh point nearest the geographic midpoint of the ROI.
    centre_idx   = _roi_centre_idx(verts_geo, region)
    prior_centre = prior_edp[:, centre_idx]
    post_centre  = post_edp[:,  centre_idx]
    pr_nm, pr_hm = extract_robust_f2_peak(prior_centre, alt_grid)
    po_nm, po_hm = extract_robust_f2_peak(post_centre,  alt_grid)

    # ΔNe slice at the prior F2 peak altitude
    if not np.isnan(po_hm):
        alt_idx     = int(np.argmin(np.abs(alt_grid - pr_hm)))
        delta_slice = post_edp[alt_idx, :] - prior_edp[alt_idx, :]
        hmF2_label  = f"~{alt_grid[alt_idx]:.0f} km"
    else:
        delta_slice = np.zeros(n_verts)
        hmF2_label  = "F2 unavailable"

    # ── Per-occultation constellation & colour assignment ────────────────────
    # Each GNSS constellation gets its own colour family (Blues / Purples /
    # Oranges / Greens …).  Within a family the shade deepens by occultation
    # index, mapped to the range [0.40, 0.90] to avoid too-pale tones.
    const_counts  = defaultdict(int)   # total occs per constellation letter
    occ_const     = []                 # constellation letter per occ

    for i in range(n_occ):
        prn   = sat_ids[i][1] if i < len(sat_ids) else ""
        const = prn[0].upper() if prn else "?"
        occ_const.append(const)
        const_counts[const] += 1

    const_counter = defaultdict(int)   # running index within each constellation
    occ_colours   = []                 # final RGBA colour per occ

    for const in occ_const:
        cfg       = CONSTELLATION_CONFIG.get(const, {})
        cmap_name = cfg.get("cmap", _CONST_FALLBACK_CMAP)
        cmap      = mpl.colormaps.get_cmap(cmap_name)
        n_in      = const_counts[const]
        idx_in    = const_counter[const]
        # Map index to [0.40, 0.90]; single-occ constellations get shade 0.70
        t = (0.40 + 0.50 * (idx_in / max(n_in - 1, 1))) if n_in > 1 else 0.70
        occ_colours.append(cmap(t))
        const_counter[const] += 1

    # ── Globe centre-point — compute before figure creation for projection ────
    lats_c = result["lats"]
    lons_c = result["lons"]
    clon   = float(np.nanmean(lons_c)) if lons_c else 0.0
    clat   = float(np.nanmean(lats_c)) if lats_c else 0.0
    proj   = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)

    # ── Unique receiver codes for the title ───────────────────────────────────
    unique_leos = list(dict.fromkeys(leo for leo, _ in sat_ids)) if sat_ids else []
    leo_str     = " / ".join(unique_leos) if unique_leos else "—"

    # ── Build figure with GridSpec ────────────────────────────────────────────
    # Cols 0–1: 2×2 TEC grid  |  Col 2: Globe  |  Col 3: EDP
    fig = plt.figure(figsize=(26, 9))
    fig.suptitle(
        f"Group KF Update ({mode_label}): {result['time_window']}  |  "
        f"Region: {region}  |  GN: {leo_str}\n"
        f"{n_occ} occultation(s)  —  "
        f"Prior RMSE {result['prior_tec_rmse']:.2f} → "
        f"Post RMSE {result['post_tec_rmse']:.2f} TECU",
        fontsize=12,
    )
    gs = GridSpec(2, 4, figure=fig,
                  width_ratios=[1, 1, 1.5, 1.2],
                  wspace=0.40, hspace=0.50)

    # 2×2 TEC panels — fixed positions per constellation
    # (row, col) in the GridSpec left half
    _CONST_POS = {"G": (0, 0), "R": (1, 0), "E": (0, 1), "C": (1, 1)}
    ax_tec      = {}
    first_tec   = None           # used for shared y-axis
    for const, (row, col) in _CONST_POS.items():
        cfg = CONSTELLATION_CONFIG.get(const, {"name": const, "title_color": "black"})
        ax  = fig.add_subplot(gs[row, col],
                              sharey=first_tec if first_tec is not None else None)
        ax.set_title(cfg["name"], fontsize=9, color=cfg["title_color"], fontweight="bold")
        ax.grid(True, alpha=0.3, ls=":")
        ax_tec[const] = ax
        if first_tec is None:
            first_tec = ax

    # Globe (col 2, spans both rows — add after computing proj)
    ax2 = fig.add_subplot(gs[:, 2], projection=proj)

    # EDP col 3: two stacked panels.
    #   Top (row 0): KF prior / posterior spaghetti + centre profiles
    #   Bot (row 1): Abel Ne profiles
    # Both share the altitude y-axis (sharey=first_tec) and the Ne x-axis
    # (sharex links them so zoom / xlim changes propagate to both).
    ax3_kf   = fig.add_subplot(gs[0, 3], sharey=first_tec)
    ax3_abel = fig.add_subplot(gs[1, 3], sharey=first_tec, sharex=ax3_kf)

    # ── TEC panel drawing ─────────────────────────────────────────────────────
    # Line-style legend entries (drawn once in the first panel that has data)
    style_entries = [
        Line2D([0], [0], color="gray", lw=2.2,          label="Measured TEC"),
        Line2D([0], [0], color="gray", lw=1.3, ls="--", label="Prior TEC"),
        Line2D([0], [0], color="gray", lw=1.5, ls=":",  label="KF Posterior"),
        Line2D([0], [0], color="gray", lw=1.5, ls="-.", label="Abel Cal. TEC"),
    ]

    # Accumulate per-constellation legend handles  {const: [Line2D, …]}
    const_legend = defaultdict(list)

    for i, (sl, col) in enumerate(zip(tec_slices, occ_colours)):
        const = occ_const[i]
        # Route to the correct TEC panel; unknown constellations fall back to GPS
        ax_t  = ax_tec.get(const) or ax_tec.get("G") or next(iter(ax_tec.values()))

        ax_t.plot(sl["measured"],  sl["tangent_km"], color=col, lw=2.2)
        ax_t.plot(sl["prior_tec"], sl["tangent_km"], color=col, lw=1.3,
                  ls="--", alpha=0.6)
        ax_t.plot(sl["post_tec"],  sl["tangent_km"], color=col, lw=1.5,
                  ls=":",  alpha=0.9)
        # if i < len(abel_list) and abel_list[i] is not None:
        #     abel = abel_list[i]
        #     if len(abel.get("TEC_cal", [])) > 0:
        #         ax_t.plot(abel["TEC_cal"], abel["alt_km"], color=col, lw=1.5,
        #                   ls="-.", alpha=0.85)

        # Legend entry: PRN code  (e.g. "G03") + observation time
        prn_code = sat_ids[i][1] if i < len(sat_ids) else f"Occ {i + 1}"
        time_str = file_labels[i].split()[-1] if i < len(file_labels) else ""
        lbl      = f"{prn_code}  ({time_str})" if time_str else prn_code
        const_legend[const].append(Line2D([0], [0], color=col, lw=2.2, label=lbl))

    # Apply legends and y-limits; add line-style key to first panel with data
    all_alts      = np.concatenate([sl["tangent_km"] for sl in tec_slices])
    alt_ylim      = (0, max(float(np.nanmax(all_alts)) + 50, float(alt_grid[-1])))
    style_placed  = False

    for const, ax_t in ax_tec.items():
        entries = const_legend.get(const, [])

        if entries:
            leg_handles = entries + (style_entries if not style_placed else [])
            ax_t.legend(handles=leg_handles, fontsize=7, loc="upper right",
                        framealpha=0.85)
            style_placed = True
        else:
            ax_t.text(0.5, 0.5, "No data", transform=ax_t.transAxes,
                      ha="center", va="center", color="lightgray", fontsize=11,
                      style="italic")

        ax_t.set_ylim(*alt_ylim)
        # Y-axis label only on left column panels
        if const in ("G", "R"):
            ax_t.set_ylabel("Tangent Altitude (km)")
        else:
            ax_t.tick_params(labelleft=False)
        # X-axis label only on bottom row panels
        if const in ("R", "C"):
            ax_t.set_xlabel("TEC (TECU)")

    # ── Panel 2: Globe map — ΔNe at F2 peak + raypaths + ROI ────────────────
    ax2.set_global()
    ax2.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
    ax2.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
    ax2.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.5, edgecolor="gray")
    ax2.add_feature(cfeature.BORDERS.with_scale("110m"),   lw=0.3, edgecolor="lightgray")
    ax2.gridlines(lw=0.3, alpha=0.4)

    # ΔNe = posterior − prior at posterior hmF2.  Diverging coolwarm centred at 0.
    try:
        max_delta = float(np.nanmax(np.abs(delta_slice)))
        if max_delta > 0:
            tc = ax2.tripcolor(
                verts_geo[:, 0], verts_geo[:, 1], tris_geo,
                delta_slice,
                transform=ccrs.Geodetic(),
                cmap="coolwarm", shading="flat",
                vmin=-max_delta, vmax=max_delta,
                zorder=1,
            )
            cbar = fig.colorbar(tc, ax=ax2, orientation="horizontal",
                                shrink=0.75, pad=0.04, fraction=0.04)
            cbar.set_label(f"ΔNe at hmF2 ({hmF2_label}) [m⁻³]", fontsize=8)
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.update_ticks()
    except Exception:
        pass

    # Yellow star at the centre EDP vertex
    ctr_lon = float(verts_geo[centre_idx, 0])
    ctr_lat = float(verts_geo[centre_idx, 1])
    ax2.plot(ctr_lon, ctr_lat, transform=ccrs.Geodetic(),
             marker="*", color="yellow", ms=14, mec="black", mew=0.8,
             zorder=8, label="Centre EDP vertex")

    # Region-of-interest boundary (lime rectangle or polar latitude circle)
    _draw_roi_boundary(ax2, region)

    # Per-occultation raypaths: top (solid), TEC-max (dashed), bottom (dotted).
    # Raypath labels appear only on the first occultation for a clean legend.
    ray_defs = [
        ("top",     "solid",  2.0),
        ("tec-max", "dashed", 1.8),
        ("bottom",  "dotted", 1.5),
    ]
    for i, (cl, col) in enumerate(zip(clean_list, occ_colours)):
        LEO  = cl["LEO"]
        GNSS = cl["GNSS"]
        tec  = cl["tec"]
        tang = cl["tangent_km"]

        idx_top    = int(np.argmax(tang))
        idx_bottom = int(np.argmin(tang))
        idx_tecmax = int(np.argmax(tec))

        prn_code = sat_ids[i][1] if i < len(sat_ids) else f"Occ {i + 1}"

        for (rtype, ls, lw), ridx in zip(
            ray_defs, [idx_top, idx_tecmax, idx_bottom]
        ):
            lbl = f"{rtype}" if i == 0 else None
            _draw_raypath(ax2, LEO, GNSS, ridx,
                          color=col, ls=ls, lw=lw, label=lbl, zorder=6)

    ax2.set_title(
        f"ΔNe at hmF2 ({hmF2_label}) + Raypaths + ROI\n"
        "(solid=top, dashed=TEC-max, dotted=bottom  ★=centre)"
    )
    ax2.legend(loc="lower left", fontsize=7, framealpha=0.75)

    # ── Panel 3 (top): KF prior / posterior EDPs ─────────────────────────────
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))

    # Faint spaghetti — all grid columns
    ax3_kf.plot(prior_edp, alt_grid, color="tab:red",  alpha=0.07, lw=0.8)
    ax3_kf.plot(post_edp,  alt_grid, color="tab:blue", alpha=0.07, lw=0.8)

    # Bold centre-column profiles
    ax3_kf.plot(prior_centre, alt_grid, color="darkred",  lw=2.0, ls="--",
                label="Prior (centre)")
    ax3_kf.plot(post_centre,  alt_grid, color="darkblue", lw=2.0,
                label="Posterior (centre)")

    # F2 peak markers — circles on KF centre column
    if not np.isnan(pr_nm):
        ax3_kf.plot(pr_nm, pr_hm, marker="o", ms=8, color="darkred",
                    mec="black", zorder=5)
    if not np.isnan(po_nm):
        ax3_kf.plot(po_nm, po_hm, marker="o", ms=8, color="darkblue",
                    mec="black", zorder=5)

    kf_legend_lines = [
        Line2D([0], [0], color="tab:red",  lw=1.5, alpha=0.4, label="Prior (all cols)"),
        Line2D([0], [0], color="tab:blue", lw=1.5, alpha=0.4, label="Post (all cols)"),
        Line2D([0], [0], color="darkred",  lw=2.0, ls="--",   label="Prior (centre)"),
        Line2D([0], [0], color="darkblue", lw=2.0,            label="Post (centre)"),
        Line2D([0], [0], marker="o", color="w", mfc="gray", mec="black", ms=8,
               label="F2 Peak (KF)"),
    ]

    # ── ISR truth overlay on EDP panel ───────────────────────────────────────
    if isr_profiles:
        _isr_cmap = mpl.colormaps.get_cmap("viridis")
        _isr_norm = mpl.colors.Normalize(vmin=0, vmax=24)
        for _prof in isr_profiles:
            _col = _isr_cmap(_isr_norm(_prof["hour_utc"]))
            ax3_kf.plot(_prof["ne"], _prof["alt_km"],
                        color=_col, lw=1.2, alpha=0.65, zorder=3)
        # Mean ISR F2 peak marker
        _isr_nms = [p["nm_f2"] for p in isr_profiles if not np.isnan(p.get("nm_f2", np.nan))]
        _isr_hms = [p["hm_f2"] for p in isr_profiles if not np.isnan(p.get("hm_f2", np.nan))]
        if _isr_nms:
            _isr_nm_mean = float(np.nanmean(_isr_nms))
            _isr_hm_mean = float(np.nanmean(_isr_hms))
            ax3_kf.plot(_isr_nm_mean, _isr_hm_mean,
                        marker="^", ms=11, color="limegreen",
                        mec="black", mew=1.0, zorder=8,
                        label="ISR NmF2 (mean)")
        kf_legend_lines += [
            Line2D([0], [0], color="limegreen", lw=1.5, alpha=0.7,
                   label=f"ISR truth ({len(isr_profiles)} sweeps)"),
            Line2D([0], [0], marker="^", color="w", mfc="limegreen",
                   mec="black", ms=9, label="ISR NmF2 (mean)"),
        ]

        # ISR site marker on the globe
        if isr_site is not None:
            _isr_lon, _isr_lat = isr_site
            ax2.plot(_isr_lon, _isr_lat,
                     transform=ccrs.Geodetic(),
                     marker="^", ms=12, color="limegreen",
                     mec="black", mew=1.0, zorder=9,
                     label="Millstone Hill ISR")

    ax3_kf.legend(handles=kf_legend_lines, fontsize=7, loc="upper right")
    ax3_kf.set_title("EDP — Prior / Posterior\n(★ = centre vertex, ▲ = ISR truth)"
                     if isr_profiles else
                     "EDP — Prior / Posterior\n(★ = centre vertex)")
    ax3_kf.xaxis.set_major_formatter(formatter)
    ax3_kf.tick_params(labelbottom=False)
    ax3_kf.grid(True, alpha=0.3, ls=":")

    # ── Panel 3 (bottom): Abel Ne profiles ───────────────────────────────────
    max_edp_candidates = []
    idx_f2_mask = alt_grid < 450
    valid_post  = post_edp[idx_f2_mask].copy()
    valid_post[valid_post < 0] = np.nan
    if not np.all(np.isnan(valid_post)):
        max_edp_candidates.append(np.nanmax(valid_post))

    abel_legend_lines = []
    for i, (col, abel) in enumerate(zip(occ_colours, abel_list)):
        if abel is None or len(abel.get("Ne", [])) == 0:
            continue
        prn_code = sat_ids[i][1] if i < len(sat_ids) else f"Occ {i + 1}"
        ax3_abel.plot(abel["Ne"], abel["alt_km"], color=col, lw=1.8, ls="-.",
                      alpha=0.9)
        abel_legend_lines.append(
            Line2D([0], [0], color=col, lw=1.8, ls="-.", label=f"Abel {prn_code}")
        )
        abel_nm, abel_hm = extract_robust_f2_peak(abel["Ne"], abel["alt_km"])
        if not np.isnan(abel_nm):
            ax3_abel.plot(abel_nm, abel_hm, marker="^", ms=7,
                          color=col, mec="black", mew=0.6, zorder=5)
            max_edp_candidates.append(abel_nm)

    abel_legend_lines.append(
        Line2D([0], [0], marker="^", color="w", mfc="gray", mec="black", ms=7,
               label="F2 Peak (Abel)")
    )

    # ISR truth overlay on Abel panel (same colour coding as EDP panel)
    if isr_profiles:
        _isr_cmap2 = mpl.colormaps.get_cmap("viridis")
        _isr_norm2 = mpl.colors.Normalize(vmin=0, vmax=24)
        for _prof in isr_profiles:
            _col2 = _isr_cmap2(_isr_norm2(_prof["hour_utc"]))
            ax3_abel.plot(_prof["ne"], _prof["alt_km"],
                          color=_col2, lw=1.2, alpha=0.65, zorder=3)
            _isr_nm2, _isr_hm2 = extract_robust_f2_peak(_prof["ne"], _prof["alt_km"])
            if not np.isnan(_isr_nm2):
                ax3_abel.plot(_isr_nm2, _isr_hm2,
                              marker="^", ms=6, color=_col2,
                              mec="black", mew=0.6, zorder=6)
                max_edp_candidates.append(_isr_nm2)
        abel_legend_lines += [
            Line2D([0], [0], color="limegreen", lw=1.5, alpha=0.7,
                   label=f"ISR truth ({len(isr_profiles)} sweeps)"),
        ]

    ax3_abel.legend(handles=abel_legend_lines, fontsize=7, loc="upper right")
    ax3_abel.set_xlabel("Electron Density (m⁻³)")
    ax3_abel.set_title("Abel Ne Profiles  (▲ = ISR truth)" if isr_profiles
                       else "Abel Ne Profiles")
    ax3_abel.xaxis.set_major_formatter(formatter)
    ax3_abel.grid(True, alpha=0.3, ls=":")

    # Shared x-limit (propagates to ax3_kf via sharex)
    if max_edp_candidates:
        ax3_abel.set_xlim(left=0, right=max(max_edp_candidates) * 1.2)

    safe_key  = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    plot_path = os.path.join(save_dir, f"group_{safe_key}{suffix}.png")
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved group plot ({mode_label}) → {plot_path}")
    return plot_path


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
    plot_path = os.path.join(save_dir, f"group_{safe_key}_comparison.png")
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison plot → {plot_path}")
    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# §F  Global summary figures
# ─────────────────────────────────────────────────────────────────────────────

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
