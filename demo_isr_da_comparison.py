#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
demo_isr_da_comparison.py

Compares data-assimilation filter outputs (gridded KF vs. parametric EKF)
against ISR ground-truth electron density profiles, across GNSS-RO-only,
IGS-only, and combined observation modes.

Supports an occultation-count sensitivity study: within each occultation-
availability-minima window, sweeps the count of assimilated RO measurements
(OCC_COUNT_BINS) to study measurement-density effects on retrieval quality,
measured against real ISR electron density profiles.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4
import pyproj

from demo_esr_isr import load_edps, isr_days
from demo_isr_initial_conditions import (
    INSTRUMENTS, ALT_GRID,
    get_solar_conditions, build_voxel_grid, build_parametric_grid,
    _iri_at_instrument, _identify_instrument,
)
from demo_group import (
    scan_metadata, process_group, assign_region,
    time_window_key, _parse_time_window,
    _tight_bbox_from_points, MAX_MESH_VERTICES,
)
from demo_ground_station_kf import (
    run_info_window, build_grid_from_bounds,
    window_centres, filter_arcs_for_window, load_igs_arcs,
)
from TEC_model.igs_tec_pipeline import H_IPP_KM
from demo_compare_kf_enkf import (
    _load_igs_arcs, _filter_igs_for_window as _filter_igs_cmp,
    _get_iri_edp_and_features_batch,
    _solar_sampling_df, _state_from_iri_direct,
    _build_gnss_to_leo_ray, _tangent_latlon_single,
    _fit_iri_params, _covariance_from_edp_samples,
)
from plotIonosphereTomography import (
    ISR_MIN_VALID_GATES, _plot_group_all_modes, plot_isr_truth_comparison,
    plot_occultation_prior_post_truth,
)
from test_param_iono import (
    EKF_Param, select_arcs_by_count_bin, _get_reflection_height, _fit_power_law,
)
from Ionosphere_Tomography_Inverter.ionospheric_state import IonosphericState, N_STATE, PARAM_NAMES
from Ionosphere_Tomography_Inverter.observation_operator import _ne_profile_ensemble
from Ionosphere_Tomography_Inverter.enkf_update import _haversine_km
from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
from demo import _build_hourly_global_edp, extract_robust_f2_peak
from EDPSamples.edp_samples import EDPSamples, get_IRI2020_EDP

# ── Plasma-frequency helpers ─────────────────────────────────────────────────

def ne_to_mhz(ne_m3) -> float | np.ndarray:
    """Electron density [m⁻³] → plasma frequency [MHz]. foF2 ≈ 8.98 MHz at 1e12 m⁻³."""
    return 8.978e-6 * np.sqrt(np.maximum(np.asarray(ne_m3, dtype=float), 0.0))


def extract_e_layer_peak(ne_arr: np.ndarray, alt_arr: np.ndarray,
                          e_alt_min: float = 90.0,
                          e_alt_max: float = 150.0) -> tuple[float, float]:
    """
    Return (NmE [m⁻³], hmE [km]) — the E-layer peak in the 90–150 km band.
    Returns (nan, nan) if no valid data in band.
    """
    mask = (alt_arr >= e_alt_min) & (alt_arr <= e_alt_max) & np.isfinite(ne_arr) & (ne_arr > 1e6)
    if mask.sum() == 0:
        return np.nan, np.nan
    band_ne  = ne_arr[mask]
    band_alt = alt_arr[mask]
    idx = int(np.argmax(band_ne))
    return float(band_ne[idx]), float(band_alt[idx])


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
PODTC_BASE  = Path("/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2")
ISR_CACHE   = ROOT / "Data" / "ISR_Data" / "esr_edp_cache.pkl"
IC_CATALOG  = ROOT / "Data" / "ISR_IC" / "catalog.json"
RINEX_CACHE = ROOT / "Data" / "RINEX_Cache"
DA_CACHE    = ROOT / "Data" / "DA_Cache"
SAVE_DIR    = ROOT / "Figures" / "ISR_DA_Comparison"/ "OUTPUT"
ISR_METRICS_CSV  = DA_CACHE / "isr_metrics.csv"
PROGRESS_MANIFEST = DA_CACHE / "progress_manifest.json"
LOG_DIR     = ROOT / "Logs" / "ISR_DA_Comparison"

IGS_STATIONS_NORDIC = ["TRO1", "NYA1", "KIR0", "SOD3", "ALRT","SCOR", "HOFN", 'REYK']
OBS_MODES    = ["ro_only", "ro_igs", "igs_only"]
FILTER_TYPES = ["gridded_kf", "parametric_ekf"]
POLAR_LAT_THRESHOLD = 60.0   # matches demo_group.py

# ── Parametric-EKF tuning knobs ──────────────────────────────────────────────
# Single source of truth for the EKF_Param hyperparameters used by BOTH the
# ro/ro_igs and igs_only call sites in run_all_filters(). Defaults preserve the
# previous hardcoded call-site behavior exactly (alpha=0.5, sigma_obs=10 TECU,
# 100 update rays, 200-member prior, tol=5e-4, 20 iters, unscaled prior).
# Overridable per-run via the --ekf-* CLI flags in _main_impl() (mirrors the
# OCC_COUNT_BINS global-mutation pattern). NOTE: this supersedes the now-unused
# EKF_PARAM_* block in test_param_iono.py, which only feeds that module's own
# standalone __main__, not this ISR-comparison pipeline.
EKF_ALPHA        = 0.5     # Gauss-Newton step-size damping (0,1]; lower = safer
EKF_SIGMA_OBS    = 10.0    # observation-noise std-dev R = sigma^2 I (TECU)
EKF_MAX_RAYS     = 100     # representative update rays per arc
EKF_TOL          = 5e-7    # relative ||dP||/||P|| convergence tolerance
EKF_MAX_ITER     = 20      # maximum EKF iterations (hard cap; stop on either
                           # this OR the compound ΔP/P & TEC-RMSE gate below)
EKF_N_MEMBERS    = 200     # prior ensemble size factoring the covariance X_c
EKF_PRIOR_SCALE  = 1.0     # multiplier on prior param variances P_b

# ── Tuned-EKF knobs (Phase 2a/2c recommended config) ─────────────────────────
# Default OFF = legacy all-free, fixed-alpha (preserves prior pipeline output).
# Enable the tuned config via --ekf-free / --ekf-adapt-alpha / --ekf-alpha-max /
# --ekf-tec-rmse-tol. Recommended (dominates all-free on hmF2 + below-peak,
# self-regulates on inconsistent data): free="log10(NmF2)", adapt_alpha=True,
# alpha_max=1.0, tec_rmse_tol~45, alpha(start)=0.25, max_iter>=20.
EKF_FREE_PARAMS  = None     # None = all 8 params free; else list of free names
EKF_ADAPT_ALPHA  = False    # residual-merit adaptive step size (accelerate/damp)
EKF_ALPHA_MAX    = 1.0      # cap on adaptive-alpha growth
EKF_TEC_RMSE_TOL = 5.0      # compound TEC-RMSE convergence gate (TECU); None=off.
                            # Convergence requires ΔP/P<EKF_TOL AND RMSE<this.


def _parse_ekf_free_spec(spec: "str | None") -> "list | None":
    """"all" (or None) -> None (all params free); "log10(NmF2),hmF2" -> list."""
    if spec is None or spec.strip().lower() == "all":
        return None
    return [p.strip() for p in spec.split(",") if p.strip()]


# ── EKF-only dense-grid experiment knobs ─────────────────────────────────────
# Default preserves the legacy behaviour (gridded KF + single parametric EKF on
# the RO union mesh). The Aug/Sep/Oct-2025 ISR study enables:
#   --no-gridded-kf  : skip the gridded-KF joint solve (process_group
#                      skip_joint=True) and drop gridded_kf from scoring/plots.
#   --ekf-grid fibonacci --ekf-grid-km 200 : run the EKF on a dense
#                      ROI-restricted Fibonacci sphere instead of the union mesh.
#   --ekf-modes allfree,nmf2 : run the EKF once per named mode (distinct
#                      filter_type labels), instead of one "parametric_ekf".
RUN_GRIDDED_KF   = True      # False => --no-gridded-kf
EKF_GRID_MODE    = "mesh"    # "mesh" (union mesh) | "fibonacci" (dense ROI grid)
EKF_GRID_KM      = 200.0     # Fibonacci point spacing (km); tunable per run.
                             # 200 km keeps the uncapped ROI grid tractable: the
                             # EKF's dense (N_STATE*n_geo)^2 covariance scales as
                             # 1/spacing^4, so 200 km is ~16x cheaper than 100 km.
# Hard cap on Fibonacci grid points. The EKF's dense background covariance is
# (N_STATE * n_geo)^2, so n_geo must stay bounded; if the footprint disk at the
# chosen spacing exceeds this, the points closest to the centroid are kept.
# None (default) removes the cap -- rely on the ROI gate + 200 km spacing to
# bound n_geo instead. Set --ekf-grid-max-pts N (>0) to reinstate a hard cap.
EKF_GRID_MAX_PTS = None
# Domain gate (Fix 1): RO tangent points are selected by TIME window, so some
# land thousands of km from the ISR/IGS footprint (Siberia/Bering/Mongolia when
# the target is ESR at 78 N). Drop any RO anchor whose great-circle distance
# from the footprint reference (robust IGS-pierce/TECmax centroid) exceeds this,
# before it can seed the Fibonacci grid. 6000 km keeps the RO extrema out to a
# wide radius while the reference stays anchored on the TECmax footprint; lower
# it (e.g. 2500) to tighten the grid onto the core. None disables the gate.
EKF_ROI_GATE_KM  = 6000.0

# Registry of named EKF modes -> (filter_type label, free_params). "allfree"
# estimates all 8 IRI params; "nmf2" freezes everything but the amplitude.
_EKF_MODE_REGISTRY = {
    "allfree": ("ekf_allfree", None),
    "nmf2":    ("ekf_nmf2",    ["log10(NmF2)"]),
}
# None => legacy single EKF run labelled "parametric_ekf" (honours --ekf-free).
# Else a list of (label, free_params) tuples set from --ekf-modes.
EKF_MODES = None

ISR_SITES              = ("ESR", "TRO")
ISR_SITE_MATCH_DEG     = 0.5
ISR_WINDOW_HALF_MINUTES = 15

# When True (default) only process availability-minima windows that have a
# co-located ISR truth profile (within ISR_WINDOW_HALF_MINUTES of an ISR_SITES
# station); ISR-blind windows are dropped since they can never be scored.
# --all-windows sets this False to also run (and sort last) the blind windows.
REQUIRE_ISR_TRUTH      = True

# When True, investigate ONLY the single best window per day -- the one with the
# most RO occultations (tie-broken by most co-located ISR truth EDPs). The
# OCC_COUNT_BINS sweep still runs on that window so the occultation-count
# sensitivity study is preserved. Enabled by --best-window-only / --final.
BEST_WINDOW_ONLY       = False
ISR_ROI_MAX_KM         = 2500.0   # RO peak-tangent-point → ISR site gate (great-circle)

# Minimum tangent-altitude a ray must reach to count as a "full profile" for
# the per-occultation prior/post/truth diagnostic plot -- see the selection
# comment in _process_window_bin for why this matters (avoids picking a
# short, high-altitude-only arc fragment just because it's geographically
# close to the ISR site).
OCC_DIAG_FULL_PROFILE_MAX_ALT_KM = 100.0

OCC_COUNT_BINS = [None, 55, 45, 35, 25, 15, 5]
  # None = assimilate ALL available RO occultations in the window;
  # then decreasing counts to study measurement-density sensitivity.
WINDOW_ROLLING_HOURS = 1.0     # rolling window for availability_minima_windows
WINDOW_MIN_SEP_MIN   = 60.0    # min separation between detected minima
WINDOW_PROMINENCE    = 3.0     # peak prominence for minima detection
MIN_ARCS_PER_WINDOW  = 10      # skip windows thinner than this

# HF band frequencies [MHz] at which posterior/ISR radio reflection heights are
# compared in compute_isr_metrics (mirrors analyze_hf_reflection_heights() in
# test_param_iono.py, applied here to a single column vs a single ISR profile).
HF_REFLECTION_FREQS_MHZ = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


# ─────────────────────────────────────────────────────────────────────────────
# Progress manifest (resume support)
# ─────────────────────────────────────────────────────────────────────────────
#
# One JSON file recording, per group_key, whether that group's metrics/CSV row
# and figures have already been produced. This is separate from DA_CACHE's
# per-(group, obs_mode, filter_type) pickle cache (which already lets
# _run_or_load skip re-running an individual filter): the manifest instead
# lets a resumed run skip a group's *entire* metrics+plotting pass (the slow
# part -- cartopy/matplotlib figure generation re-runs on every invocation
# even when every underlying filter result is a cache hit) and gives a
# human-readable "how far did this get" status independent of scrolling
# through run logs.

def _load_progress_manifest() -> dict:
    if PROGRESS_MANIFEST.exists():
        with open(PROGRESS_MANIFEST, "r") as fh:
            return json.load(fh)
    return {}


def _save_progress_manifest(manifest: dict) -> None:
    """Atomic write (tmp file + rename) so a crash mid-write can't leave a
    truncated/corrupt manifest behind."""
    DA_CACHE.mkdir(parents=True, exist_ok=True)
    tmp_path = PROGRESS_MANIFEST.with_suffix(".json.tmp")
    with open(tmp_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    tmp_path.replace(PROGRESS_MANIFEST)


def print_progress_status(priority_days: "list[dict] | None" = None,
                           window_key_filter: "str | None" = None) -> None:
    """
    Print a human-readable summary of how much of the analysis has already
    been completed, from the progress manifest and the accumulated ISR
    metrics CSV -- without running anything except (when *priority_days* is
    given) the same podTc2 metadata scan / minima-window split main() itself
    uses, so the grid below reflects the exact units a real run would see.
    Used by `--status`.

    When *priority_days* is given (already filtered by --tier/--days/
    --start-date, as main() does before calling this), also prints a
    window x bin_count completion grid: for every (window, bin_count) cell,
    how many of the len(OBS_MODES) x len(FILTER_TYPES) units already have a
    DA_CACHE pickle on disk -- the exact same check _run_or_load() uses to
    decide what to skip on resume -- so an interrupted overnight run shows
    exactly what remains. *window_key_filter* narrows this grid to a single
    window_key (--window), matching what an actual run with --window would
    process.
    """
    manifest = _load_progress_manifest()
    print(f"[ISR-DA][status] Progress manifest: {PROGRESS_MANIFEST}")
    if not manifest:
        print("  No (group, bin) units recorded as complete yet.")
    else:
        dates = sorted({e.get("date") for e in manifest.values() if e.get("date")})
        date_range = f"{dates[0]} .. {dates[-1]}" if dates else "unknown"
        print(f"  {len(manifest)} (group, bin) unit(s) marked complete, spanning "
              f"{len(dates)} day(s) ({date_range})")

        from collections import Counter
        # entry.get("bin_label", "all") covers manifest entries written before
        # the OCC_COUNT_BINS sensitivity study added bin_count/bin_label keys.
        bin_counts: Counter = Counter(
            e.get("bin_label", "all") for e in manifest.values()
        )
        print("  Completed (group, bin) units per occultation-count bin:")
        for bin_label, n in sorted(bin_counts.items(),
                                    key=lambda kv: (kv[0] != "all", kv[0])):
            print(f"    bin={bin_label:<4} : {n}")

        status_counts: Counter = Counter()
        for entry in manifest.values():
            bin_label = entry.get("bin_label", "all")
            for obs_mode, per_filter in entry.get("obs_mode_status", {}).items():
                for filter_type, status in per_filter.items():
                    status_counts[(bin_label, obs_mode, filter_type, status)] += 1
        print("  Filter-run outcomes across completed (group, bin) units:")
        for (bin_label, obs_mode, filter_type, status), n in sorted(status_counts.items()):
            print(f"    bin={bin_label:<4} {obs_mode:<9} {filter_type:<14} {status:<45} : {n}")

    if ISR_METRICS_CSV.exists():
        df = pd.read_csv(ISR_METRICS_CSV, parse_dates=["t_centre"])
        print(f"\n  ISR metrics CSV: {ISR_METRICS_CSV}")
        print(f"    {len(df)} row(s), {df['group_key'].nunique()} group(s), "
              f"date range {df['date'].min()} .. {df['date'].max()}")
    else:
        print(f"\n  No ISR metrics CSV yet at {ISR_METRICS_CSV}")

    if not priority_days:
        return

    bin_labels = [_bin_label(b) for b in OCC_COUNT_BINS]
    n_per_cell = len(OBS_MODES) * len(FILTER_TYPES)
    window_filter_note = f" matching --window {window_key_filter}" if window_key_filter else ""
    if BEST_WINDOW_ONLY:
        isr_note = " (best ISR-truth window per day only)"
    elif REQUIRE_ISR_TRUTH:
        isr_note = " (ISR-truth windows only; --all-windows to include blind ones)"
    else:
        isr_note = ""
    print(f"\n  Window x bin_count completion grid{window_filter_note}{isr_note} "
          f"({len(bin_labels)} bin(s): {bin_labels}; "
          f"{n_per_cell} = {len(OBS_MODES)} obs_modes x {len(FILTER_TYPES)} filters per cell):")

    # Only load the ISR EDP catalogue when we actually need to gate windows on
    # truth availability -- keeps --all-windows / --status fast otherwise.
    _status_edps = load_edps() if REQUIRE_ISR_TRUTH else None

    grand_done = 0
    grand_total = 0
    any_windows = False
    for day_info in priority_days:
        date, podtc_dir = day_info["date"], day_info["podtc_dir"]
        if podtc_dir is None:
            continue
        windows = build_minima_windows_for_day(podtc_dir, date)
        if window_key_filter is not None:
            windows = [w for w in windows if w["window_key"] == window_key_filter]
        if REQUIRE_ISR_TRUTH and _status_edps is not None:
            # Mirror run_all_filters(): drop ISR-blind windows so the grid
            # reflects exactly the units a real run would process.
            _counts = {id(w): len(_isr_profiles_for_window(_status_edps,
                                                           w["t_centre"]))
                       for w in windows}
            windows = [w for w in windows if _counts[id(w)] > 0]
            if BEST_WINDOW_ONLY and windows:
                # Same selection as run_all_filters(): most occultations,
                # tie-broken by most ISR truth EDPs.
                windows = [max(windows,
                               key=lambda w: (w["n_occ"], _counts[id(w)]))]
        if not windows:
            continue
        any_windows = True

        print(f"\n  {date}  ({len(windows)} window(s)):")
        header = "    window  " + "".join(f"{lbl:>8}" for lbl in bin_labels) + "     total"
        print(header)
        for w in windows:
            group_key = w["t_centre"].strftime("%Y-%m-%d_%H%M")
            cells = []
            row_done = 0
            for bin_count in OCC_COUNT_BINS:
                n_done = sum(
                    1 for obs_mode in OBS_MODES for filter_type in FILTER_TYPES
                    if _cache_path(group_key, bin_count, obs_mode, filter_type).exists()
                )
                row_done += n_done
                cells.append(f"{n_done}/{n_per_cell}")
            row_total = n_per_cell * len(OCC_COUNT_BINS)
            grand_done += row_done
            grand_total += row_total
            print(f"    {w['window_key']:<8}" + "".join(f"{c:>8}" for c in cells) +
                  f"   {row_done:>4}/{row_total}")

    if not any_windows:
        _blind_note = " with ISR truth" if REQUIRE_ISR_TRUTH else ""
        print(f"  No windows found{_blind_note} for the selected day(s)"
              + window_filter_note + ".")
    else:
        pct = 100.0 * grand_done / grand_total if grand_total else 0.0
        print(f"\n  Grid total: {grand_done}/{grand_total} unit(s) done ({pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Day / group selection
# ─────────────────────────────────────────────────────────────────────────────

def select_priority_days(force: bool = False) -> list[dict]:
    """
    Select ISR days to process, prioritised by data availability/tier.

    Tier 1 : both TRO and ESR present on that day.
    Tier 2 : either TRO or ESR present (not both).
    Tier 3 : JRO only.

    Returns a list of dicts sorted tier-first, then by date:
        {date, tier, instruments, podtc_dir, doy, year}
    """
    edps = load_edps(force=force)
    days = isr_days(edps)

    entries: list[dict] = []
    for d in days:
        day_edps = [e for e in edps if e["time"].date() == d]
        instruments = sorted({_identify_instrument(e["lat"]) for e in day_edps})

        if "TRO" in instruments and "ESR" in instruments:
            tier = 1
        elif "TRO" in instruments or "ESR" in instruments:
            tier = 2
        else:
            tier = 3

        doy = d.timetuple().tm_yday
        year = d.year
        podtc_dir = PODTC_BASE / f"{year}.{doy:03d}"
        if not (podtc_dir.is_dir() and any(podtc_dir.glob("*.0001_nc"))):
            podtc_dir = None

        entries.append({
            "date":        d,
            "tier":        tier,
            "instruments": instruments,
            "podtc_dir":   podtc_dir,
            "doy":         doy,
            "year":        year,
        })

    entries.sort(key=lambda e: (e["tier"], e["date"]))
    return entries


def _near_isr_site_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    sites: tuple[str, ...] = ISR_SITES,
    max_km: float = ISR_ROI_MAX_KM,
) -> np.ndarray:
    """
    True where the RO peak (TEC-max) tangent point (lat, lon) is within
    `max_km` great-circle distance of at least one of the given ISR sites.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    within = np.zeros(lat.shape, dtype=bool)
    for site in sites:
        inst = INSTRUMENTS[site]
        dist_km = _haversine_km(inst["lat"], inst["lon"], lat, lon)
        within |= (dist_km <= max_km)
    return within


def load_ro_group_for_day(podtc_dir: Path) -> list[tuple[str, pd.DataFrame]]:
    """
    Load and group GNSS-RO occultations for a given ISR day's podTc2 directory.

    Selects occultations whose RO peak (TEC-max) tangent point — the
    scan_metadata() "lat"/"lon" columns, i.e. lat_tecmax_tangent /
    lon_tecmax_tangent — falls within ISR_ROI_MAX_KM (great-circle distance)
    of one of the ISR_SITES radar positions. This replaces the previous
    crude region == "POLAR_N" latitude-threshold proxy with an actual
    distance-based gate.

    Groups by demo_group's fixed 30-min time_window_key() clock grid
    (group_key = "<time_window>__<region>"). Superseded by
    build_minima_windows_for_day() below for the occultation-count
    sensitivity study, which instead partitions the day at local minima of
    occultation availability; kept intact here for backward compatibility
    with call sites that still want the fixed-grid grouping.

    Returns a sorted list of (group_key, group_meta_df) tuples.
    """
    if podtc_dir is None or not Path(podtc_dir).is_dir():
        return []

    meta = scan_metadata(str(podtc_dir))
    if meta.empty:
        return []

    near_mask = _near_isr_site_mask(meta["lat"].to_numpy(), meta["lon"].to_numpy())
    roi_meta = meta[near_mask]
    if roi_meta.empty:
        return []

    groups = [(key, grp) for key, grp in roi_meta.groupby("group_key")]
    groups.sort(key=lambda kv: kv[0])
    return groups


def build_minima_windows_for_day(podtc_dir: Path, date) -> list[dict]:
    """
    Partition *podtc_dir*'s ISR-ROI occultations for *date* into windows
    bounded by local minima of the rolling occultation-availability count
    (demo_occultation_availability.availability_minima_windows), replacing
    demo_group's fixed 30-min time_window_key() clock grid used by
    load_ro_group_for_day() above.

    Each occultation's timestamp is scan_metadata()'s "date" column, which
    is the RO peak (TEC-max) tangent-point time (built from the
    lat_tecmax_tangent / lon_tecmax_tangent netCDF attrs), not the file
    start time.

    Windows with fewer than MIN_ARCS_PER_WINDOW ISR-ROI occultations are
    dropped.

    Returns a list of dicts sorted by window start, one per retained window:
        {"window_key": "HHMM" of the window centre,
         "lo": window start Timestamp,
         "hi": window end Timestamp,
         "t_centre": window centre Timestamp,
         "group_meta": DataFrame subset of ISR-ROI occultations in [lo, hi),
         "n_occ": len(group_meta)}
    """
    if podtc_dir is None or not Path(podtc_dir).is_dir():
        return []

    meta = scan_metadata(str(podtc_dir))
    if meta.empty:
        return []

    near_mask = _near_isr_site_mask(meta["lat"].to_numpy(), meta["lon"].to_numpy())
    roi_meta = meta[near_mask]
    if roi_meta.empty:
        return []

    from demo_occultation_availability import availability_minima_windows

    date = pd.Timestamp(date)
    day = pd.Timestamp(date.year, date.month, date.day)
    minima_windows, _grid, _counts, _minima_idx = availability_minima_windows(
        roi_meta["date"], day,
        window_hours    = WINDOW_ROLLING_HOURS,
        min_sep_minutes = WINDOW_MIN_SEP_MIN,
        prominence       = WINDOW_PROMINENCE,
    )

    windows: list[dict] = []
    for lo, hi in minima_windows:
        subset = roi_meta[(roi_meta["date"] >= lo) & (roi_meta["date"] < hi)]
        n_occ = len(subset)
        if n_occ < MIN_ARCS_PER_WINDOW:
            continue
        t_centre = lo + (hi - lo) / 2
        window_key = f"{t_centre.hour:02d}{t_centre.minute:02d}"
        windows.append({
            "window_key": window_key,
            "lo": lo, "hi": hi, "t_centre": t_centre,
            "group_meta": subset, "n_occ": n_occ,
        })

    windows.sort(key=lambda w: w["lo"])

    print(f"[ISR-DA][minima-windows] {day.date()}: {len(windows)} window(s) "
          f"(>= {MIN_ARCS_PER_WINDOW} arcs each)")
    for w in windows:
        print(f"  {w['window_key']}  centre={w['t_centre']}  "
              f"n_occ={w['n_occ']}  span=[{w['lo']} .. {w['hi']})")

    return windows


def load_igs_for_day(date: pd.Timestamp) -> list:
    """
    Load IGS ground-station sTEC arcs co-located with an ISR day.

    Returns the combined list of clean-list dicts from all Nordic stations
    (may be empty if downloads/processing fail for every station).
    """
    return _load_igs_arcs(
        date,
        stations=IGS_STATIONS_NORDIC,
        cache_dir=str(RINEX_CACHE),
        rinex_version=3,
        use_iri=False,
        max_rays=200,
    )


# Per-epoch array fields in an IGS clean-list entry (see
# igs_obs_to_clean_entry(), TEC_model/igs_tec_pipeline.py) that need slicing
# down to a single epoch. LEO/GNSS are (3, n_s); the rest are (n_s,).
_IGS_ARC_PER_EPOCH_FIELDS = (
    "tec", "tangent_km", "ipp_lat", "ipp_lon",
    "arc_time_sec", "time_s", "time_utc_h", "elev_deg",
)


def _collapse_igs_arc_to_central_epoch(arc: dict) -> dict:
    """
    Collapse one IGS clean-list arc (many epochs spread across the ~30-minute
    assimilation window) down to a single epoch at the arc's central time
    index, so the arc contributes exactly one raypath/measurement instead of
    one per (decimated) epoch.

    Why: the filters assume a static ionosphere across the assimilation
    window, but IGS sTEC visibly changes over a 30-minute pass, so spreading
    one arc's epochs across the whole window as independent observations
    injects real ionospheric variability as if it were noise, degrading the
    filters below climatology. Picking the single raypath closest to the
    window's midpoint avoids that violated-stationarity assumption while
    still using real ground-truth TEC.

    Parameters
    ----------
    arc : one clean-list dict as returned by igs_obs_to_clean_entry() /
        _load_igs_arcs() (must have 'LEO', 'GNSS', and the fields in
        _IGS_ARC_PER_EPOCH_FIELDS, all aligned along the epoch axis).

    Returns
    -------
    A new dict with the same keys, but every per-epoch field sliced down to
    a single (central-index) sample. LEO/GNSS keep their (3, 1) shape so
    downstream code (which indexes by `.shape[1]`) is unaffected.
    """
    n_s = arc["LEO"].shape[1]
    if n_s <= 1:
        return arc
    idx = n_s // 2

    out = dict(arc)
    out["LEO"]  = arc["LEO"][:, idx:idx + 1]
    out["GNSS"] = arc["GNSS"][:, idx:idx + 1]
    for key in _IGS_ARC_PER_EPOCH_FIELDS:
        val = arc.get(key)
        if val is not None and hasattr(val, "__len__") and len(val) == n_s:
            out[key] = val[idx:idx + 1]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Filter execution
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_EDP_DATA_DIR = ROOT / "Data" / "Section20_Global_EDPS"


def _build_hour_edp_cache(t_centre: pd.Timestamp) -> dict:
    """
    Build (or load from NetCDF cache) a single hour's global EDPSamples grid,
    keyed by t_centre.hour, for use as process_group's global_edp_cache.

    Reuses the same per-hour worker as build_daily_global_edps (including its
    on-disk NetCDF cache) but builds only the one hour actually needed instead
    of all 24 — 30-min RO windows never straddle an hour boundary, so this is
    always the hour process_group's internal median-timestamp lookup expects.
    """
    _GLOBAL_EDP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    solar = get_solar_conditions(t_centre)
    args = (
        t_centre.hour, ALT_GRID,
        solar["f107"], solar["ap"], solar["ig12"], solar["rz12"],
        50, 5.0, 5.0, str(_GLOBAL_EDP_DATA_DIR),
        t_centre.strftime("%Y%m%d"), t_centre.strftime("%Y-%m-%d"),
    )
    hour, nc_path = _build_hourly_global_edp(args)
    return {hour: EDPSamples.fromNetCDF(nc_path)}


def _build_igs_eds_occ(t_centre: pd.Timestamp, grid_lats: np.ndarray,
                        grid_lons: np.ndarray, alt_grid: np.ndarray,
                        n_mc: int = 100) -> "SimpleNamespace":
    """
    Build an EDPSamples-compatible IRI Monte Carlo ensemble evaluated exactly
    at the IGS regular grid's own points (grid_lats/grid_lons from build_grid).

    EDPSamples(geo_type=...) always regenerates its own vertex layout, which
    would not line up with run_info_window's prior_edp/post_edp grids (indexed
    by that same grid_lats/grid_lons order). Calling get_IRI2020_EDP directly
    sidesteps that regeneration so the returned .geolocation matches 1:1.
    """
    from types import SimpleNamespace

    solar = get_solar_conditions(t_centre)
    rng   = np.random.default_rng(42)
    mc_df = pd.DataFrame({
        "hour": np.full(n_mc, float(t_centre.hour)),
        "f107": rng.normal(loc=solar["f107"], scale=10, size=n_mc).clip(70, 250),
        "ap":   rng.normal(loc=solar["ap"],   scale=5,  size=n_mc).clip(0,  400),
        "ig12": rng.normal(loc=solar["ig12"], scale=10, size=n_mc).clip(50, 200),
        "rz12": rng.normal(loc=solar["rz12"], scale=10, size=n_mc).clip(50, 200),
    })
    mc_df.iloc[0] = {
        "hour": float(t_centre.hour), "f107": solar["f107"], "ap": solar["ap"],
        "ig12": solar["ig12"], "rz12": solar["rz12"],
    }

    geolocation = np.column_stack([grid_lons, grid_lats])   # (n_geo, 2): col0=lon, col1=lat
    edps, feature_edps = get_IRI2020_EDP(
        t_centre.strftime("%Y-%m-%d %H:%M:%S"), np.asarray(alt_grid, dtype=float),
        geolocation, mc_df,
    )

    return SimpleNamespace(
        geolocation=geolocation,
        altitude=np.asarray(alt_grid, dtype=float),
        edps=edps,
        feature_edps=feature_edps,
        mesh=None,
        attrs={"geo_type": "Rectangle"},
    )


# ── Dense Fibonacci-sphere grid for the parametric EKF ───────────────────────
_EARTH_RADIUS_KM = 6371.0


def _fibonacci_sphere_latlon(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Golden-spiral (Fibonacci) lattice of n_points ~evenly spaced on a sphere.

    Returns (lat_deg, lon_deg) with lat in [-90, 90], lon in [-180, 180). The
    average nearest-neighbour spacing is ~R * sqrt(4*pi / n_points).
    """
    n = int(max(n_points, 1))
    i = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (i + 0.5) / n                   # ring heights in (-1, 1)
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    golden = np.pi * (3.0 - np.sqrt(5.0))           # golden angle (radians)
    lon = np.degrees((golden * i) % (2.0 * np.pi))
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


def _latlon_unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """(N,3) unit vectors on the sphere for arrays of lat/lon in degrees."""
    latr = np.radians(np.asarray(lat_deg, dtype=float))
    lonr = np.radians(np.asarray(lon_deg, dtype=float))
    cl = np.cos(latr)
    return np.column_stack([cl * np.cos(lonr), cl * np.sin(lonr), np.sin(latr)])


def _robust_sphere_centroid(V: np.ndarray,
                            trim_iters: int = 3,
                            trim_percentile: float = 90.0) -> np.ndarray:
    """Outlier-resistant unit centroid of (N,3) unit vectors.

    Iteratively computes the mean direction, drops the farthest points beyond
    the *trim_percentile* angular distance, and recomputes -- so a handful of
    stray anchors (e.g. RO tangent points thousands of km from the footprint)
    cannot drag the centre. Always keeps at least three points.
    """
    V = np.asarray(V, dtype=float)
    if V.shape[0] <= 3:
        c = V.mean(axis=0)
        n = float(np.linalg.norm(c))
        return V[0] if n < 1e-9 else c / n

    keep = np.ones(V.shape[0], dtype=bool)
    centroid = V.mean(axis=0)
    nrm = float(np.linalg.norm(centroid))
    centroid = V[0] if nrm < 1e-9 else centroid / nrm
    for _ in range(int(trim_iters)):
        cos_to = np.clip(V[keep] @ centroid, -1.0, 1.0)
        ang = np.arccos(cos_to)
        thr = np.percentile(ang, float(trim_percentile))
        idx = np.where(keep)[0]
        new_keep = keep.copy()
        new_keep[idx[ang > thr]] = False
        if new_keep.sum() < 3:                             # never over-trim
            break
        keep = new_keep
        c = V[keep].mean(axis=0)
        nrm = float(np.linalg.norm(c))
        if nrm < 1e-9:
            break
        centroid = c / nrm
    return centroid


def _fibonacci_roi_grid(roi_lats, roi_lons, spacing_km: float,
                        margin_km: float = 200.0,
                        max_pts: "int | None" = None,
                        radius_percentile: float = 95.0,
                        trim_iters: int = 3,
                        trim_percentile: float = 90.0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Fibonacci-sphere grid (~spacing_km apart) covering the RO/IGS footprint.

    A GLOBAL Fibonacci sphere is generated at the resolution needed for the
    requested spacing (N ~ 4*pi*R^2 / spacing^2), then restricted by
    GREAT-CIRCLE distance to the footprint -- NOT a lat/lon bounding box.

    Why great-circle, not bbox: near the pole a physically compact footprint
    (e.g. RO tangent points within a few 1000 km of ESR at 78 N) scatters
    across nearly the full longitude circle, so a lat/lon bbox degenerates to
    lon in [-180, 180] and the clipped grid becomes an entire latitude BAND
    around the globe (~18k points at 100 km) -- most of them on the far side
    of the pole, thousands of km from any data, which blows up both the IRI
    Monte Carlo and the EKF's dense (N_STATE * n_geo)^2 covariance. Distance
    from the footprint centroid keeps only the points that actually surround
    the data.

    Robustness (Fix 2): the centre is a trimmed unit centroid (see
    _robust_sphere_centroid) so stray anchors cannot drag it, and the disk
    radius is the *radius_percentile* of anchor distances (not the single
    farthest point) plus *margin_km* -- so one outlier that slips past the
    upstream gate can no longer balloon the footprint. If the result still
    exceeds *max_pts*, the points closest to the centroid are kept (a dense
    core centred on the footprint), bounding the EKF state size.
    """
    roi_lats = np.asarray(roi_lats, dtype=float)
    roi_lons = np.asarray(roi_lons, dtype=float)
    ok = np.isfinite(roi_lats) & np.isfinite(roi_lons)
    roi_lats, roi_lons = roi_lats[ok], roi_lons[ok]
    if roi_lats.size == 0:
        return np.array([]), np.array([])

    V = _latlon_unit_vectors(roi_lats, roi_lons)            # (n_roi, 3)
    centroid = _robust_sphere_centroid(
        V, trim_iters=trim_iters, trim_percentile=trim_percentile)

    # Angular radius: percentile of anchor distances (+ physical margin), so a
    # single outlier cannot set the radius the way arccos(min cos) would.
    cos_to_roi = np.clip(V @ centroid, -1.0, 1.0)
    ang_to_roi = np.arccos(cos_to_roi)
    pct_ang = float(np.percentile(ang_to_roi, float(radius_percentile)))
    radius_km = _EARTH_RADIUS_KM * pct_ang + float(margin_km)

    n_global = int(round(4.0 * np.pi * _EARTH_RADIUS_KM ** 2 / float(spacing_km) ** 2))
    n_global = max(n_global, 60)
    flat, flon = _fibonacci_sphere_latlon(n_global)
    F = _latlon_unit_vectors(flat, flon)
    ang = np.arccos(np.clip(F @ centroid, -1.0, 1.0))
    dist_km = _EARTH_RADIUS_KM * ang
    keep = dist_km <= radius_km
    klat, klon, kdist = flat[keep], flon[keep], dist_km[keep]

    if max_pts is not None and klat.size > int(max_pts):
        order = np.argsort(kdist)[:int(max_pts)]           # densest core first
        klat, klon = klat[order], klon[order]

    return klat, klon


def _gate_ro_anchors_to_footprint(ro_points, ref_points,
                                  gate_km: "float | None" = EKF_ROI_GATE_KM
                                  ) -> tuple[list, int]:
    """Drop RO anchors that sit far from the ISR/IGS footprint (Fix 1).

    ``ro_points`` and ``ref_points`` are lists of ``(lat, lon)``. The gate
    reference is the robust unit centroid of ``ref_points`` (the IGS pierce
    points, which are physically anchored to the ground stations); if there
    are none, it falls back to the robust centroid of the RO points themselves
    (so a well-clustered RO-only footprint is left untouched). Any RO anchor
    whose great-circle distance from that reference exceeds ``gate_km`` is
    removed, and the survivors are de-duplicated.

    Returns ``(kept_points, n_dropped)``.
    """
    ro_points = [p for p in (ro_points or [])
                 if np.isfinite(p[0]) and np.isfinite(p[1])]
    if not ro_points or gate_km is None:
        # De-dupe only.
        seen, kept = set(), []
        for p in ro_points:
            key = (round(float(p[0]), 4), round(float(p[1]), 4))
            if key not in seen:
                seen.add(key)
                kept.append(p)
        return kept, 0

    ref = [p for p in (ref_points or [])
           if np.isfinite(p[0]) and np.isfinite(p[1])]
    ref_src = ref if ref else ro_points
    Vref = _latlon_unit_vectors(np.array([p[0] for p in ref_src]),
                                np.array([p[1] for p in ref_src]))
    centroid = _robust_sphere_centroid(Vref)

    Vro = _latlon_unit_vectors(np.array([p[0] for p in ro_points]),
                               np.array([p[1] for p in ro_points]))
    ang = np.arccos(np.clip(Vro @ centroid, -1.0, 1.0))
    dist_km = _EARTH_RADIUS_KM * ang

    seen, kept, n_drop = set(), [], 0
    for p, d in zip(ro_points, dist_km):
        if d > float(gate_km):
            n_drop += 1
            continue
        key = (round(float(p[0]), 4), round(float(p[1]), 4))
        if key not in seen:
            seen.add(key)
            kept.append(p)
    return kept, n_drop


def _adapt_igs_kf_result_for_plotting(
    result:           dict,
    *,
    grid_lats:        np.ndarray,
    grid_lons:        np.ndarray,
    alt_grid:         np.ndarray,
    t_centre:         pd.Timestamp,
    region:           str,
    group_key:        str,
    C_v:              np.ndarray,
    C_s:              np.ndarray,
    max_rays_per_arc: int,
) -> dict:
    """
    Adapt run_info_window()'s igs_only gridded-KF return schema (clean_window,
    grid_lats/grid_lons, prior_edp/post_edp, prior_sigma/post_sigma, y_obs_all/
    Y_all_prior_mean/Y_all_post_mean, arc_sizes, ...) into the same
    _plot_group/_plot_altitude_slices/plot_kf_enkf_comparison-compatible
    schema already used by igs_only's parametric-EKF path (see _run_igs_ekf's
    res_kf_adapted, above in run_all_filters()).

    Returns a dict of NEW keys to merge onto ``result`` (does not mutate it) —
    the original run_info_window keys (clean_window, prior_edp, ...) are left
    untouched since _run_igs_ekf still reads them directly from the same
    object.

    Deliberately does NOT densify a full posterior covariance matrix: the
    Kronecker prior inputs (C_v/C_s) are carried through as-is under
    prior_C_v/prior_C_s, and the posterior is exposed as variance-only
    (post_sigma) — see Q1 in memory/project_isr_da_comparison_plan.md.
    A dense (n_alt*n_grid)^2 covariance here would risk the same O(n^3) RAM
    blowup the SRIF batch-update fix was written to avoid.

    eds_occ is built from ``result``'s own grid_lats/grid_lons (the grid that
    actually produced result["prior_edp"]/result["post_edp"]) rather than the
    grid_lats/grid_lons passed in by the caller, which reflect the CURRENT
    run's ROI/grid construction and can differ in vertex count from an older
    cached ``result`` (e.g. a stale igs_only/gridded_kf DA_CACHE pickle from
    before a ROI-construction code change) -- using the caller's grid in that
    case silently produces a geolocation mesh with a different vertex count
    than prior_edp_3d/post_edp_3d, which later crashes compute_isr_metrics's
    cKDTree lookup with an out-of-bounds column index. Falls back to the
    passed-in grid_lats/grid_lons for old cached results that predate the
    "grid_lats"/"grid_lons" keys being stored on the result dict.
    """
    clean_window     = result["clean_window"]
    result_grid_lats = result.get("grid_lats", grid_lats)
    result_grid_lons = result.get("grid_lons", grid_lons)
    eds_occ = _build_igs_eds_occ(t_centre, result_grid_lats, result_grid_lons, alt_grid)

    sat_ids     = [(arc.get("leo_id", "IGS"), arc.get("prn_id", "?")) for arc in clean_window]
    file_labels = [f"{leo}/{prn}" for leo, prn in sat_ids]
    lats = [float(arc.get("lat_tecmax_tangent", np.nan)) for arc in clean_window]
    lons = [float(arc.get("lon_tecmax_tangent", np.nan)) for arc in clean_window]

    arc_sizes   = result["arc_sizes"]
    y_obs_all   = result["y_obs_all"]
    y_prior_all = result["Y_all_prior_mean"]
    y_post_all  = result["Y_all_post_mean"]

    # Per-arc TEC slices, sliced from the flat joint arrays using arc_sizes
    # (one entry per clean_window arc, in the same order — see
    # run_info_window's own "arc_sizes.append(ns)" loop). "tangent_km" is a
    # constant (H_IPP_KM) for IGS ground arcs, not a real tangent-altitude
    # profile — kept for schema compatibility with the RO-derived tec_slices
    # consumed by _plot_group / plot_kf_enkf_comparison / _arc_stats_from_
    # tec_slices. arc_time_sec/time_utc_h are re-derived with the SAME
    # stride/idx logic run_info_window used internally (so they line up 1:1
    # with the measured/prior_tec/post_tec slices), for any downstream
    # time-vs-TEC use.
    tec_slices: list[dict] = []
    soff = 0
    for i, n_s in enumerate(arc_sizes):
        sl  = slice(soff, soff + n_s)
        arc = clean_window[i]
        n_s_full = arc["GNSS"].shape[1]
        stride   = max(1, int(np.ceil(n_s_full / max_rays_per_arc)))
        idx      = np.arange(0, n_s_full, stride)
        arc_time_sec = np.asarray(
            arc.get("arc_time_sec", np.arange(n_s_full, dtype=float)))[idx]
        time_utc_h = np.asarray(
            arc.get("time_utc_h", np.full(n_s_full, np.nan)))[idx]
        tec_slices.append(dict(
            measured     = y_obs_all[sl].copy(),
            prior_tec    = y_prior_all[sl].copy(),
            post_tec     = y_post_all[sl].copy(),
            tangent_km   = np.full(n_s, H_IPP_KM),
            arc_time_sec = arc_time_sec,
            time_utc_h   = time_utc_h,
        ))
        soff += n_s

    return dict(
        eds_occ             = eds_occ,
        clean_list          = clean_window,
        sat_ids             = sat_ids,
        file_labels         = file_labels,
        region              = region,
        time_window         = group_key,
        lats                = lats,
        lons                = lons,
        alt_grid            = alt_grid,
        prior_edp_3d        = result["prior_edp"],
        post_edp_3d         = result["post_edp"],
        # igs_only has no seq/joint distinction (single-shot batch update) —
        # the "joint" posterior is the same as the (only) posterior.
        joint_post_edp_3d   = result["post_edp"],
        tec_slices          = tec_slices,
        joint_tec_slices    = tec_slices,
        prior_tec_rmse      = result["prior_rmse"],
        post_tec_rmse       = result["post_rmse"],
        joint_post_tec_rmse = result["post_rmse"],
        prior_C_v           = C_v,
        prior_C_s           = C_s,
        post_sigma          = result["post_sigma"],
    )


def _plot_ekf_convergence(
    residual_history: list[float],
    update_norm_history: list[float],
    converged: bool,
    tol: float,
    save_dir: str,
    group_key: str,
) -> None:
    """
    Plot EKF_Param's per-iteration RMSE and ||ΔP||/||P|| convergence trace.

    Replaces the old per-iteration terminal prints (one "computing Jacobian"
    line + one "RMSE=...  ||ΔP||/||P||=..." line per iteration) -- EKF_Param
    now shows a live tqdm progress bar during the loop instead, and this plot
    is the durable record of the convergence behaviour.
    """
    if not residual_history:
        return
    iters = np.arange(1, len(residual_history) + 1)

    fig, (ax_rmse, ax_dp) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax_rmse.plot(iters, residual_history, marker="o", color="tab:blue")
    ax_rmse.set_ylabel("RMSE (TECU)")
    ax_rmse.set_yscale("log")
    ax_rmse.grid(True, which="both", alpha=0.3)
    ax_rmse.set_title(
        f"EKF convergence — {group_key}  "
        f"({'converged' if converged else 'NOT converged'} at iter {len(residual_history)})"
    )

    ax_dp.plot(iters, update_norm_history, marker="o", color="tab:orange")
    ax_dp.axhline(tol, color="k", linestyle="--", linewidth=1, label=f"tol={tol:.1e}")
    ax_dp.set_ylabel(r"$||\Delta P|| \, / \, ||P||$")
    ax_dp.set_xlabel("Iteration")
    ax_dp.set_yscale("log")
    ax_dp.grid(True, which="both", alpha=0.3)
    ax_dp.legend(loc="upper right")

    fig.tight_layout()
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{group_key}_ekf_convergence.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved EKF convergence plot → {out_path}")


def _run_parametric_ekf(
    res_kf: dict,
    alt_grid: np.ndarray,
    save_dir: str = "./Figures/ISR_DA_Comparison/",
    group_key: str = "group",
    n_members: int = 200,
    sigma_obs: float = 10.0,
    max_update_rays: int = 100,
    alpha: float = 0.5,
    tol: float = 5e-4,
    max_iter: int = 20,
    tec_rmse_tol: "float | None" = None,
    adapt_alpha: bool = False,
    alpha_max: float = 1.0,
    step_clip: "float | None" = None,
    prior_scale: float = 1.0,
    return_diagnostics: bool = False,
    free_params: "list | None" = None,
    param_stages: "list | None" = None,
) -> dict:
    """
    Run the iterative parametric EKF (EKF_Param, test_param_iono.py) on the
    same observations used by the standard KF, and return a result dict with
    the same shape as process_group's output so it can be passed directly to
    _plot_group / plot_kf_enkf_comparison.

    Builds model_state (an IonosphericState with a populated prior ensemble)
    and arc_truth_list (one dict per arc, in EKF_Param's expected schema) from
    res_kf's eds_occ/clean_list, mirroring the input-construction logic of
    _run_parametric_enkf in demo_compare_kf_enkf.py, then hands off to
    EKF_Param for the actual iterative EKF update.

    Parameters
    ----------
    res_kf : dict
        Output of process_group / run_info_window (or an IGS-adapted dict
        with the same required fields: eds_occ, clean_list, sat_ids,
        prior_edp_3d, time_window).
    alt_grid : ndarray
        Altitude grid shared by both filters.
    save_dir : str
        Unused by EKF_Param directly; kept for signature parity with the
        gridded-KF/EKF call sites.
    group_key : str
        Group identifier used in log messages.
    n_members : int
        Prior ensemble size used to factor the EKF's covariance (X_c X_c^T).
    sigma_obs : float
        Assumed observation noise standard deviation (TECU).
    max_update_rays : int
        Maximum representative rays per arc used in the EKF update.
    alpha, tol, max_iter :
        EKF_Param step-size, convergence tolerance, and iteration cap.
    prior_scale : float
        Multiplier on the prior parameter-error *variances* (the N_STATE x
        N_STATE background covariance P_b from _covariance_from_edp_samples,
        which factors the EKF gain via X_c X_c^T). >1 loosens the prior (larger
        steps, more data-following); <1 tightens it (more conservative, guards
        against TEC-fit aliasing that smears the F2 peak). 1.0 = current
        behavior. Applied as P_b *= prior_scale before ensemble generation.

    Returns
    -------
    res_ekf : dict with keys matching _plot_group / plot_kf_enkf_comparison
        expectations (prior_edp_3d, post_edp_3d, joint_post_edp_3d,
        tec_slices, prior_tec_rmse, post_tec_rmse, joint_post_tec_rmse).
    """
    eds_occ    = res_kf["eds_occ"]
    clean_list = res_kf["clean_list"]
    sat_ids    = res_kf.get("sat_ids", [])   # (leo_id, prn_id) per arc
    prior_edp  = res_kf["prior_edp_3d"]      # (n_alt, n_geo)
    verts_geo  = eds_occ.geolocation          # (n_geo, 2): col0=lon, col1=lat

    n_geo     = verts_geo.shape[0]
    grid_lats = verts_geo[:, 1].astype(float)
    grid_lons = verts_geo[:, 0].astype(float)

    t_centre    = _parse_time_window(res_kf.get("time_window", group_key))
    sampling_df = _solar_sampling_df(t_centre)

    print(f"  [EKF] Building IRI state at {n_geo} grid points (batch call) …")
    mean_state = np.zeros((N_STATE, n_geo), dtype=float)
    try:
        ne_all, feat_all = _get_iri_edp_and_features_batch(
            t_centre, grid_lats, grid_lons, alt_grid, sampling_df
        )
        for g in range(n_geo):
            mean_state[:, g] = _state_from_iri_direct(
                ne_all[:, g], feat_all[:, g], alt_grid
            )
    except Exception as _exc:
        print(f"  [EKF] Batch IRI call failed ({_exc}); falling back to profile fit")
        for g in range(n_geo):
            mean_state[:, g] = _fit_iri_params(prior_edp[:, g], alt_grid)

    # Background covariance from the same IRI ensemble the KF uses.
    P_b, C_s = _covariance_from_edp_samples(eds_occ, alt_grid)
    if prior_scale != 1.0:
        P_b = np.asarray(P_b, dtype=float) * float(prior_scale)

    model_state = IonosphericState(n_grid_points=n_geo, n_members=n_members)
    if n_geo > 1:
        model_state.generate_ensemble_spatial(mean_state, P_b, C_s, n_members=n_members)
    else:
        model_state.generate_ensemble(mean_state, P_b, n_members=n_members)

    # ── Build arc_truth_list (full per-epoch arrays) from clean_list ─────────
    # EKF_Param does its own internal decimation to max_update_rays, so the
    # arrays here should be full-length (i.e. as decimated by the KF already,
    # not further sub-sampled).
    arc_truth_list: list[dict] = []
    for i, cl in enumerate(clean_list):
        leo  = cl["LEO"]     # (3, n_s) ECEF km
        gnss = cl["GNSS"]    # (3, n_s) ECEF km
        n_s  = leo.shape[1]

        rays = [_build_gnss_to_leo_ray(gnss[:, k], leo[:, k]) for k in range(n_s)]
        tp_latlon = [_tangent_latlon_single(gnss[:, k], leo[:, k]) for k in range(n_s)]
        tp_lats = np.array([ll[0] for ll in tp_latlon])
        tp_lons = np.array([ll[1] for ll in tp_latlon])

        if i < len(sat_ids) and sat_ids[i]:
            leo_id, prn_id = sat_ids[i]
        else:
            leo_id, prn_id = "IGS", f"arc{i:02d}"

        arc_truth_list.append(dict(
            rays      = rays,
            tec_truth = np.asarray(cl["tec"], dtype=float),
            tp_lats   = tp_lats,
            tp_lons   = tp_lons,
            tang_km   = np.asarray(cl["tangent_km"], dtype=float),
            conid     = str(leo_id),
            prn_id    = str(prn_id),
        ))

    print(f"  [EKF] Running EKF_Param on {len(arc_truth_list)} arc(s) …")
    ekf_result = EKF_Param(
        arc_truth_list, model_state, grid_lats, grid_lons, alt_grid,
        sigma_obs=sigma_obs, max_update_rays=max_update_rays,
        alpha=alpha, tol=tol, max_iter=max_iter, tec_rmse_tol=tec_rmse_tol,
        adapt_alpha=adapt_alpha, alpha_max=alpha_max, step_clip=step_clip,
        jacobian_analytical = True,
        prior_mean=mean_state, return_diagnostics=return_diagnostics,
        free_params=free_params, param_stages=param_stages,
    )
    _plot_ekf_convergence(
        residual_history=ekf_result["residual_history"],
        update_norm_history=ekf_result["update_norm_history"],
        converged=ekf_result["converged"],
        tol=tol, save_dir=save_dir, group_key=group_key,
    )

    # ── Translate EKF_Param's tec_slices schema to the {measured, prior_tec,
    #    post_tec, tangent_km} shape _arc_stats_from_tec_slices / plot_kf_enkf_
    #    comparison expect ──────────────────────────────────────────────────
    tec_slices_ekf = [
        dict(
            measured   = sl["tec_truth"],
            prior_tec  = sl["prior_tec"],
            post_tec   = sl["post_tec"],
            tangent_km = sl["tang_km"],
        )
        for sl in ekf_result["tec_slices"]
    ]

    res_ekf = dict(res_kf)   # copy all shared fields (eds_occ, clean_list, ...)
    res_ekf["prior_edp_3d"]        = ekf_result["prior_edp"]
    res_ekf["post_edp_3d"]         = ekf_result["posterior_edp"]
    res_ekf["joint_post_edp_3d"]   = ekf_result["posterior_edp"]
    res_ekf["tec_slices"]          = tec_slices_ekf
    # Overwrite the gridded KF's inherited (Ne-space) prior_P/post_P with the
    # EKF's own analytical (parametric-space) covariance — see EKF_Param's
    # "6b" section for the derivation and state ordering.
    res_ekf["prior_P"]             = ekf_result["prior_P"]
    res_ekf["post_P"]              = ekf_result["post_P"]
    res_ekf["prior_tec_rmse"]      = ekf_result["prior_rmse"]
    res_ekf["post_tec_rmse"]       = ekf_result["post_rmse"]
    res_ekf["joint_post_tec_rmse"] = ekf_result["post_rmse"]
    res_ekf["ekf_converged"]       = ekf_result["converged"]
    res_ekf["ekf_n_iterations"]    = ekf_result["n_iterations"]
    # Parametric state vectors (N_STATE, n_geo), needed by downstream plots
    # (plotIonosphereTomography.py) to render the 8-parameter readout box.
    res_ekf["prior_mean_state"]     = ekf_result["prior_mean_state"]
    res_ekf["posterior_mean_state"] = ekf_result["posterior_mean_state"]
    # Observability diagnostics (only present when return_diagnostics=True)
    if return_diagnostics:
        res_ekf["prior_jacobian"]  = ekf_result.get("prior_jacobian")
        res_ekf["prior_y_hat"]     = ekf_result.get("prior_y_hat")
        res_ekf["prior_obs_sigma"] = ekf_result.get("prior_obs_sigma")
        res_ekf["ekf_n_grid"]      = ekf_result.get("n_grid_points")

    return res_ekf


def _ro_extrema_points(group_meta: pd.DataFrame) -> list[tuple[float, float]]:
    """
    Ray-path corner points (lat, lon) for every occultation in a group, used
    to size the ROI.  Computed directly from the group's raw podTc2 files
    (not from a post-hoc trimmed/shrunk mesh) via the same
    EDPSamples.get_occultation_extrema approach as demo_group.py.
    """
    pts: list[tuple[float, float]] = []
    for _, row in group_meta.iterrows():
        try:
            data = parse_podTc2_nc_file(row["full_path"])
            if data is None:
                continue
            pt1, pt2, pt3 = EDPSamples.get_occultation_extrema(
                data["LEO"], data["GNSS"], alt_limit=700.0
            )
            found = False
            for pt in (pt1, pt2, pt3):
                _lat, _lon = float(pt[0]), float(pt[1])
                if np.isfinite(_lat) and np.isfinite(_lon):
                    pts.append((_lat, _lon))
                    found = True
            if not found:
                raise ValueError("non-finite extrema")
        except Exception:
            pts.append((float(row["lat"]), float(row["lon"])))
    return pts


def _safe_group_key(group_key: str) -> str:
    return group_key.replace("/", "_").replace(" ", "_").replace(":", "")


# Module-level (not nested in run_all_filters) so print_progress_status()'s
# window x bin_count completion grid can check the exact same DA_CACHE paths
# _run_or_load() resumes from, without duplicating the naming convention.
def _bin_label(bin_count: "int | None") -> str:
    return "all" if bin_count is None else str(bin_count)


def _manifest_key(group_key: str, bin_count: "int | None") -> str:
    return f"{group_key}_bin{_bin_label(bin_count)}"


def _cache_path(group_key: str, bin_count: "int | None",
                 obs_mode: str, filter_type: str) -> Path:
    return DA_CACHE / f"{group_key}_bin{_bin_label(bin_count)}_{obs_mode}_{filter_type}.pkl"


def _parse_bin_count_value(raw: str) -> "int | None":
    """Parse one --bin-count/--occ-bins token: "all"/"none" (any case) -> None
    (use every available arc), else int(raw)."""
    s = raw.strip().lower()
    if s in ("all", "none"):
        return None
    return int(s)


def run_all_filters(day_info: dict, windows: list[dict], igs_arcs: list,
                     edps: list[dict], force: bool = False,
                     no_plot: bool = False) -> tuple[dict, list[dict]]:
    """
    Run gridded-KF and parametric-EKF filters across all observation modes
    (ro_only, ro_igs, igs_only) AND across the full OCC_COUNT_BINS
    occultation-count sensitivity sweep, for every minima window on one ISR
    day.

    *windows* is the list of dicts returned by build_minima_windows_for_day()
    (each: {"window_key", "lo", "hi", "t_centre", "group_meta", "n_occ"}).
    For every window, every bin_count in OCC_COUNT_BINS is run in turn --
    bin_count=None (all available RO occultations) first, then decreasing
    counts -- via _process_window_bin(), which subsamples that window's RO
    occultations to bin_count via select_arcs_by_count_bin() (IGS arcs are
    never subsampled).

    Results are cached to DA_CACHE as
    "{group_key}_bin{bin_label}_{obs_mode}_{filter_type}.pkl" and pickled
    immediately after each is computed, so a re-run with force=False resumes
    where it left off, independently per (window, bin_count).

    Metrics and figures for each (window, bin_count) unit are produced right
    after that unit's 6 filter runs finish (rather than waiting for every
    unit in the day to finish), so plots start appearing incrementally as
    the day progresses.

    Returns
    -------
    (results, metrics) where results is keyed by
    (group_key, bin_count, obs_mode, filter_type) -> result dict or None,
    and metrics is the flat list of per-(window, bin_count) ISR comparison
    rows (see compute_isr_metrics).
    """
    DA_CACHE.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    all_metrics: list[dict] = []
    global_edp_cache: dict = {}
    # Memo of the EKF Fibonacci grid + its IRI Monte-Carlo prior, keyed by
    # group_key (window identity). The grid is built from the FULL window's
    # occultations (bin-independent), so it is identical across the whole
    # OCC_COUNT_BINS sweep -- computing the ~3.3M-eval prior once per window
    # instead of once per bin (a ~7x saving), and keeping the EKF state grid
    # fixed so the occultation-count study varies only measurement density.
    _ekf_grid_memo: dict = {}
    progress_manifest = _load_progress_manifest()

    _tro = INSTRUMENTS["TRO"]
    _igs_region = assign_region(_tro["lat"], _tro["lon"])

    def _result_status(value) -> str:
        if value is None:
            return "None"
        if isinstance(value, dict):
            return str(value.get("status", "ok (no 'status' key)"))
        return type(value).__name__

    def _run_or_load(group_key: str, bin_count: "int | None", obs_mode: str,
                      filter_type: str, compute_fn):
        path = _cache_path(group_key, bin_count, obs_mode, filter_type)
        tag = f"{group_key} | bin={_bin_label(bin_count):<3} | {obs_mode:<9} | {filter_type:<14}"
        print(f"▶▶▶ _run_or_load START | {tag}")
        if path.exists() and not force:
            t0 = time.time()
            with open(path, "rb") as fh:
                value = pickle.load(fh)
            print(f"◀◀◀ _run_or_load END   | {tag} : CACHE HIT  "
                  f"({path.name}, {time.time()-t0:.1f}s load) "
                  f"-> status={_result_status(value)}")
        else:
            print(f"    [diag] {tag} : RUNNING    (force={force}, cached={path.exists()}) ...")
            t0 = time.time()
            try:
                value = compute_fn()
            except Exception:
                dt = time.time() - t0
                print(f"◀◀◀ _run_or_load END   | {tag} : FAILED     ({dt:.1f}s) -- "
                      f"exception below; result treated as None for this run and NOT "
                      f"cached, so the next non-force run retries it instead of "
                      f"reusing a crash.")
                traceback.print_exc()
                value = None
            else:
                dt = time.time() - t0
                print(f"◀◀◀ _run_or_load END   | {tag} : DONE       "
                      f"({dt:.1f}s) -> status={_result_status(value)}")
                with open(path, "wb") as fh:
                    pickle.dump(value, fh, protocol=4)
        results[(group_key, bin_count, obs_mode, filter_type)] = value
        return value

    def _process_window_bin(window: dict, bin_count: "int | None") -> None:
        """
        Run the full 3-obs-mode x 2-filter suite for one occultation-count
        bin of one minima window.

        *window* is one entry from build_minima_windows_for_day() -- a
        {"window_key", "lo", "hi", "t_centre", "group_meta", "n_occ"} dict.
        *bin_count* subsamples window["group_meta"]'s RO occultations down to
        (up to) that many rows via select_arcs_by_count_bin() (None = use
        every occultation in the window). IGS ground-station arcs are never
        subsampled -- the bin only controls RO measurement density, which is
        the whole point of comparing ro_only/ro_igs/igs_only across bins.
        """
        bin_label = _bin_label(bin_count)
        window_key = window["window_key"]
        t_centre   = window["t_centre"]
        lo, hi     = window["lo"], window["hi"]
        full_group_meta = window["group_meta"]

        # group_key keeps the "YYYY-MM-DD_HHMM" convention _parse_time_window()
        # / _filter_igs_cmp() expect; the bin dimension lives in manifest_key
        # (and the DA_CACHE pickle filenames via _cache_path), not here, so a
        # window's identity stays stable across its whole OCC_COUNT_BINS sweep.
        group_key = t_centre.strftime("%Y-%m-%d_%H%M")

        manifest_key = _manifest_key(group_key, bin_count)
        manifest_entry = progress_manifest.get(manifest_key)
        if manifest_entry is not None and manifest_entry.get("status") == "complete" and not force:
            print(f"  [resume] {manifest_key} already marked complete in the progress "
                  f"manifest (metrics+plots written on a previous run) -- skipping. "
                  f"Use --force to redo it.")
            return

        # ── Subsample RO occultations to this bin (seeded on window_key, so
        #    re-running the same window/bin reproduces the same subset --
        #    mirrors select_arcs_by_count_bin()'s own reproducibility
        #    contract in test_param_iono.py). select_arcs_by_count_bin()
        #    expects a list, so pass positional indices and use its returned
        #    selected_indices to subset the DataFrame's rows directly.
        _dummy_arcs = list(range(len(full_group_meta)))
        _, _sel_meta = select_arcs_by_count_bin(_dummy_arcs, bin_count, window_key)
        group_meta = full_group_meta.iloc[_sel_meta["selected_indices"]].reset_index(drop=True)

        # IGS arcs are filtered to the window's actual (minima-derived, not
        # fixed-30-min) span -- window_minutes is chosen so _filter_igs_cmp's
        # t_centre +/- window/2 reproduces exactly [lo, hi).
        window_width_min = (hi - lo).total_seconds() / 60.0
        igs_window_arcs = _filter_igs_cmp(igs_arcs, group_key, window_minutes=window_width_min)
        # Use a single raypath per arc (its central-time epoch) instead of the
        # full ~30-minute arc as multiple observations spread across the
        # window -- see _collapse_igs_arc_to_central_epoch() docstring.
        igs_window_arcs = [_collapse_igs_arc_to_central_epoch(a) for a in igs_window_arcs]

        if t_centre.hour not in global_edp_cache:
            global_edp_cache.update(_build_hour_edp_cache(t_centre))

        # ── Per-bin/per-window/per-obs-mode figure output directories ───────
        # Canonical layout: SAVE_DIR/bin_<label>/<window>/<obs_mode>/. This is
        # the SAME ordering used by _plot_group_all_modes (save_dir/safe_key/
        # mode), plot_isr_truth_comparison, and the occ-diagnostics below (all
        # keyed off bin_save_dir = SAVE_DIR/bin_<label>), so the KF/EnKF/EKF
        # run figures land ALONGSIDE the comparison figures for the same
        # window+bin instead of under a divergent SAVE_DIR/<window>/bin_.../
        # tree (the old bug: bin and window folders at different levels).
        safe_key   = _safe_group_key(group_key)
        group_dirs = {mode: SAVE_DIR / f"bin_{bin_label}" / safe_key / mode for mode in OBS_MODES}
        for _d in group_dirs.values():
            _d.mkdir(parents=True, exist_ok=True)

        # ── ROI: data-driven from RO tangent extrema + IGS pierce points ───
        # Same point set (RO + IGS) drives the ROI for every obs_mode, so the
        # region is identical whether or not IGS TEC is actually assimilated.
        igs_roi_points = [
            (float(a["lat_tecmax_tangent"]), float(a["lon_tecmax_tangent"]))
            for a in igs_window_arcs
            if np.isfinite(a.get("lat_tecmax_tangent", np.nan))
            and np.isfinite(a.get("lon_tecmax_tangent", np.nan))
        ]
        ro_roi_points = _ro_extrema_points(group_meta)

        _roi_lats = [p[0] for p in ro_roi_points + igs_roi_points]
        _roi_lons = [p[1] for p in ro_roi_points + igs_roi_points]
        # 200 km margin, lat-aware (see _tight_bbox_from_points docstring in
        # demo_group.py) so points near the edge of the RO+IGS footprint are
        # reliably included even for near-polar/near-meridional groups where
        # a flat lat/lon-degree margin badly under-covers physical distance
        # in longitude.
        _igs_tbbox = _tight_bbox_from_points(_roi_lats, _roi_lons, margin_km=200.0)
        # Near-pole groups can have RO/IGS points scattered across most of the
        # longitude circle even in a small physical area (1° longitude is a
        # tiny distance near the pole), which would otherwise blow up the
        # regular-grid vertex count.  Coarsen dlat/dlon until the grid stays
        # at or under the same vertex budget used for the RO union mesh.
        _dlat = _dlon = 2.0
        while True:
            _igs_grid_lats, _igs_grid_lons = build_grid_from_bounds(
                *_igs_tbbox, dlat=_dlat, dlon=_dlon,
            )
            if len(_igs_grid_lats) <= MAX_MESH_VERTICES or (_dlat > 20.0 and _dlon > 20.0):
                break
            _dlat *= 1.5
            _dlon *= 1.5

        # ── Gridded KF: RO only ───────────────────────────────────────────────
        res_ro = _run_or_load(group_key, bin_count, "ro_only", "gridded_kf", lambda: process_group(
            group_key, group_meta, ALT_GRID,
            global_edp_cache=global_edp_cache,
            run_sequential=False, save_dir=str(group_dirs["ro_only"]),
            podtc_max_rays=200, extra_clean_list=None,
            roi_extra_points=igs_roi_points,
            filter_label="kf",
            skip_joint=(not RUN_GRIDDED_KF),
        ))
        # With run_sequential=False, process_group() never runs the step-by-step
        # KF loop, so result["post_edp_3d"] is left equal to the prior (see
        # demo_group.py's `else` branch under "Sequential Kalman Filter
        # updates"). The real posterior is the joint/batch update, stored
        # separately as "joint_post_edp_3d". Alias it in so downstream EDP
        # plots don't draw an identical prior/posterior pair (mirrors the same
        # fix already applied in demo_compare_kf_enkf.py).
        if res_ro is not None:
            res_ro["post_edp_3d"] = res_ro.get("joint_post_edp_3d", res_ro.get("post_edp_3d"))

        # ── Gridded KF: RO + IGS ──────────────────────────────────────────────
        res_roigs = _run_or_load(group_key, bin_count, "ro_igs", "gridded_kf", lambda: process_group(
            group_key, group_meta, ALT_GRID,
            global_edp_cache=global_edp_cache,
            run_sequential=False, save_dir=str(group_dirs["ro_igs"]),
            podtc_max_rays=200, extra_clean_list=igs_window_arcs,
            roi_extra_points=igs_roi_points,
            filter_label="kf",
            skip_joint=(not RUN_GRIDDED_KF),
        ))
        if res_roigs is not None:
            res_roigs["post_edp_3d"] = res_roigs.get("joint_post_edp_3d", res_roigs.get("post_edp_3d"))

        # ── Gridded KF: IGS only ──────────────────────────────────────────────
        # C_v/C_s are shared between the run_info_window() call and the
        # post-hoc plotting adapter below (both need the exact same prior
        # correlation inputs — see Q1 in project_isr_da_comparison_plan.md).
        _igs_n_grid = len(_igs_grid_lats)
        _igs_C_v = np.eye(len(ALT_GRID))
        _igs_C_s = np.eye(_igs_n_grid)

        def _run_igs_only_kf(t_centre=t_centre, igs_window_arcs=igs_window_arcs):
            ne_prior, _feat = _iri_at_instrument(t_centre, "TRO")
            ne_prior_2d = np.repeat(ne_prior[:, None], _igs_n_grid, axis=1)
            return run_info_window(
                clean_window=igs_window_arcs, t_centre=t_centre,
                grid_lats=_igs_grid_lats, grid_lons=_igs_grid_lons, alt_grid=ALT_GRID,
                ne_prior=ne_prior_2d, sigma_v=0.5 * np.ones(len(ALT_GRID)),
                C_v=_igs_C_v, C_s=_igs_C_s,
                sigma_obs=10.0, max_rays_per_arc=200,
            )

        res_igs = _run_or_load(group_key, bin_count, "igs_only", "gridded_kf", _run_igs_only_kf)

        # Augment res_igs (in place, so the same object cached under
        # results[...] picks up the extra fields too) with the _plot_group/
        # _plot_altitude_slices/plot_kf_enkf_comparison-compatible schema —
        # see _adapt_igs_kf_result_for_plotting's docstring for why this is
        # needed (run_info_window's return schema doesn't match what those
        # functions expect). Only the pickled cache holds the raw
        # run_info_window dict; this runs fresh every invocation regardless
        # of cache hit/miss.
        if res_igs is not None and "eds_occ" not in res_igs:
            res_igs.update(_adapt_igs_kf_result_for_plotting(
                res_igs,
                grid_lats=_igs_grid_lats, grid_lons=_igs_grid_lons,
                alt_grid=ALT_GRID, t_centre=t_centre, region=_igs_region,
                group_key=group_key, C_v=_igs_C_v, C_s=_igs_C_s,
                max_rays_per_arc=200,
            ))

        # ── Parametric EKF ────────────────────────────────────────────────────
        kf_results = {"ro_only": res_ro, "ro_igs": res_roigs, "igs_only": res_igs}

        # EKF modes: either the named-mode set from --ekf-modes (each its own
        # filter_type label, e.g. ekf_allfree / ekf_nmf2) or the legacy single
        # "parametric_ekf" run honouring --ekf-free.
        ekf_modes = (EKF_MODES if EKF_MODES is not None
                     else [("parametric_ekf", EKF_FREE_PARAMS)])

        # ── Optional dense Fibonacci-sphere grid for the EKF ─────────────────
        # Built once per window/bin (the grid is obs-mode independent). The EKF
        # runs on this grid instead of the RO union mesh. prior_edp_3d is the
        # IRI mean on the same points (used only by _run_parametric_ekf's
        # fallback path; the primary mean_state is rebuilt from IRI internally).
        _fib_eds_occ = None
        _fib_prior   = None
        if EKF_GRID_MODE == "fibonacci":
            _memo = _ekf_grid_memo.get(group_key)
            if _memo is None:
                # Build the EKF grid from the FULL window's occultations
                # (full_group_meta), NOT the bin-subsampled group_meta, so the
                # state canvas is fixed across the whole OCC_COUNT_BINS sweep --
                # only measurement density varies per bin, not the grid itself.
                # IGS pierce points (igs_roi_points) are already bin-independent.
                _ro_roi_full   = _ro_extrema_points(full_group_meta)
                # Fix 1: gate RO anchors to the IGS/ISR footprint before they
                # become grid anchors, so time-window strays (Siberia/Bering/
                # Mongolia when targeting ESR) can't drag the centroid/radius.
                _ro_roi_gated, _n_gated = _gate_ro_anchors_to_footprint(
                    _ro_roi_full, igs_roi_points, gate_km=EKF_ROI_GATE_KM)
                if _n_gated:
                    print(f"  [EKF] ROI gate @ {EKF_ROI_GATE_KM:.0f} km: dropped "
                          f"{_n_gated} stray RO anchor(s) far from the IGS/ISR "
                          f"footprint ({len(_ro_roi_gated)} RO anchor(s) kept)")
                _roi_lats_full = [p[0] for p in _ro_roi_gated + igs_roi_points]
                _roi_lons_full = [p[1] for p in _ro_roi_gated + igs_roi_points]
                _fib_lats, _fib_lons = _fibonacci_roi_grid(
                    _roi_lats_full, _roi_lons_full, EKF_GRID_KM,
                    margin_km=200.0, max_pts=EKF_GRID_MAX_PTS)
                print(f"  [EKF] Fibonacci ROI grid @ {EKF_GRID_KM:.0f} km: "
                      f"{len(_fib_lats)} points "
                      f"(great-circle around robust footprint centroid, "
                      f"cap={EKF_GRID_MAX_PTS}, {len(_roi_lats_full)} "
                      f"full-window ROI anchor pts; built once per window)")
                _memo_eds   = None
                _memo_prior = None
                if len(_fib_lats) >= 2:
                    _memo_eds = _build_igs_eds_occ(
                        t_centre, _fib_lats, _fib_lons, ALT_GRID)
                    try:
                        _ne_all, _ = _get_iri_edp_and_features_batch(
                            t_centre, _fib_lats.astype(float), _fib_lons.astype(float),
                            ALT_GRID, _solar_sampling_df(t_centre))
                        _memo_prior = _ne_all
                    except Exception as _exc:  # noqa: BLE001
                        print(f"  [EKF] Fibonacci prior IRI batch failed ({_exc}); "
                              f"fallback path will use the mesh prior.")
                else:
                    print("  [EKF] Fibonacci ROI grid too small (<2 pts) — using "
                          "the union-mesh grid for this window instead.")
                _memo = {"eds_occ": _memo_eds, "prior": _memo_prior,
                         "n_pts": len(_fib_lats)}
                _ekf_grid_memo[group_key] = _memo
            else:
                print(f"  [EKF] Fibonacci ROI grid @ {EKF_GRID_KM:.0f} km: "
                      f"{_memo['n_pts']} points (memoized for window "
                      f"{group_key}; reused across occultation-count bins)")
            _fib_eds_occ = _memo["eds_occ"]
            _fib_prior   = _memo["prior"]

        def _apply_grid_override(res_kf_local):
            """Swap eds_occ/prior to the dense Fibonacci grid when available."""
            if _fib_eds_occ is None:
                return res_kf_local
            out = dict(res_kf_local)
            out["eds_occ"] = _fib_eds_occ
            if _fib_prior is not None:
                out["prior_edp_3d"] = _fib_prior
            return out

        for obs_mode, kf_result in kf_results.items():
            for _ekf_label, _free in ekf_modes:
                if obs_mode == "igs_only":
                    def _run_igs_ekf(kf_result=kf_result, t_centre=t_centre,
                                     group_key=group_key, _free=_free,
                                     _ekf_label=_ekf_label):
                        if kf_result is None:
                            print(f"  [diag] {group_key} | igs_only  | {_ekf_label} : "
                                  f"SKIPPED (upstream igs_only data-prep is None)")
                            return None
                        clean_window = kf_result["clean_window"]
                        # Use kf_result's OWN grid_lats/grid_lons (the grid that
                        # produced kf_result["prior_edp"]); a cache hit from an
                        # older run may have a different vertex count, and using
                        # the current grid would desync eds_occ.geolocation from
                        # prior_edp_3d. (Grid override below handles fibonacci.)
                        _kf_grid_lats = kf_result.get("grid_lats", _igs_grid_lats)
                        _kf_grid_lons = kf_result.get("grid_lons", _igs_grid_lons)
                        eds_occ = _build_igs_eds_occ(
                            t_centre, _kf_grid_lats, _kf_grid_lons, ALT_GRID)
                        sat_ids = [
                            (arc.get("leo_id", "IGS"), arc.get("prn_id", "?"))
                            for arc in clean_window
                        ]
                        res_kf_adapted = dict(kf_result)
                        res_kf_adapted["eds_occ"]       = eds_occ
                        res_kf_adapted["clean_list"]    = clean_window
                        res_kf_adapted["prior_edp_3d"]  = kf_result["prior_edp"]
                        res_kf_adapted["sat_ids"]       = sat_ids
                        res_kf_adapted["time_window"]   = group_key
                        res_kf_adapted["region"]        = _igs_region
                        res_kf_adapted["alt_grid"]      = ALT_GRID
                        res_kf_adapted["file_labels"]   = [
                            f"{leo}/{prn}" for leo, prn in sat_ids
                        ]
                        res_kf_adapted["lats"] = [
                            float(arc.get("lat_tecmax_tangent", np.nan))
                            for arc in clean_window
                        ]
                        res_kf_adapted["lons"] = [
                            float(arc.get("lon_tecmax_tangent", np.nan))
                            for arc in clean_window
                        ]
                        res_kf_adapted = _apply_grid_override(res_kf_adapted)
                        return _run_parametric_ekf(
                            res_kf=res_kf_adapted, alt_grid=ALT_GRID,
                            save_dir=str(group_dirs["igs_only"]),
                            group_key=f"{group_key}_igs_only_{_ekf_label}",
                            n_members=EKF_N_MEMBERS, sigma_obs=EKF_SIGMA_OBS,
                            max_update_rays=EKF_MAX_RAYS, alpha=EKF_ALPHA,
                            tol=EKF_TOL, max_iter=EKF_MAX_ITER,
                            prior_scale=EKF_PRIOR_SCALE,
                            free_params=_free,
                            adapt_alpha=EKF_ADAPT_ALPHA, alpha_max=EKF_ALPHA_MAX,
                            tec_rmse_tol=EKF_TEC_RMSE_TOL,
                        )

                    _run_or_load(group_key, bin_count, obs_mode, _ekf_label, _run_igs_ekf)
                    continue

                def _run_ekf(kf_result=kf_result, obs_mode=obs_mode,
                             _free=_free, _ekf_label=_ekf_label):
                    if kf_result is None or kf_result.get("status") != "Success":
                        reason = "upstream result is None" if kf_result is None else \
                            f"upstream data-prep status={kf_result.get('status')!r} (need 'Success')"
                        print(f"  [diag] {group_key} | {obs_mode:<9} | {_ekf_label} : "
                              f"SKIPPED ({reason})")
                        return None
                    res_kf_local = _apply_grid_override(kf_result)
                    return _run_parametric_ekf(
                        res_kf=res_kf_local, alt_grid=ALT_GRID,
                        save_dir=str(group_dirs[obs_mode]),
                        group_key=f"{group_key}_{obs_mode}_{_ekf_label}",
                        n_members=EKF_N_MEMBERS, sigma_obs=EKF_SIGMA_OBS,
                        max_update_rays=EKF_MAX_RAYS, alpha=EKF_ALPHA,
                        tol=EKF_TOL, max_iter=EKF_MAX_ITER,
                        prior_scale=EKF_PRIOR_SCALE,
                        free_params=_free,
                        adapt_alpha=EKF_ADAPT_ALPHA, alpha_max=EKF_ALPHA_MAX,
                        tec_rmse_tol=EKF_TEC_RMSE_TOL,
                    )
                _run_or_load(group_key, bin_count, obs_mode, _ekf_label, _run_ekf)

        # ── Per-group summary: which of the (obs_mode, filter_type) combos
        #    actually produced a usable result vs. were skipped/failed ────────
        _n_combos = len(OBS_MODES) * len(FILTER_TYPES)
        print(f"  [diag] {group_key} | bin={bin_label} : summary of {_n_combos} "
              f"obs_mode/filter_type combos")
        _missing = object()
        for obs_mode in OBS_MODES:
            for filter_type in FILTER_TYPES:
                value = results.get((group_key, bin_count, obs_mode, filter_type), _missing)
                status = "MISSING (never ran)" if value is _missing else _result_status(value)
                print(f"           {obs_mode:<9} | {filter_type:<14} -> {status}")

        # ── Metrics + figures for this group, right away ─────────────────────
        # Previously this ran in a second pass over all groups in main(), only
        # after every group in the day had finished its 6 filter runs -- so no
        # plots appeared until hours into a day's processing. Doing it here
        # means each group's plots land as soon as that group is done.
        group_filter_results = {
            obs_mode: {
                filter_type: results.get((group_key, bin_count, obs_mode, filter_type))
                for filter_type in FILTER_TYPES
            }
            for obs_mode in OBS_MODES
        }

        group_day_info = dict(
            day_info, group_key=group_key,
            bin_count=bin_count, bin_label=bin_label,
            n_ro_occultations=len(group_meta),
            n_igs_arcs=len(igs_window_arcs),
        )
        group_metrics = compute_isr_metrics(group_day_info, group_filter_results, edps)
        all_metrics.extend(group_metrics)
        # Persist immediately rather than waiting for the whole run to finish
        # (main() previously only wrote the CSV once, after every day had
        # completed -- a crash partway through lost every already-computed
        # group's metrics). Safe to call repeatedly: dedup keeps the latest
        # row per (group_key, bin_count, obs_mode, filter_type, instrument) --
        # bin_count is part of the dedup key so different OCC_COUNT_BINS
        # sweep points for the same window don't overwrite each other.
        _append_metrics_csv(group_metrics)

        def _mark_group_complete(plots_written: bool) -> None:
            progress_manifest[manifest_key] = {
                "date":              str(day_info.get("date")),
                "group_key":         group_key,
                "bin_count":         bin_count,
                "bin_label":         _bin_label(bin_count),
                "n_ro_occultations": len(group_meta),
                "n_igs_arcs":        len(igs_window_arcs),
                "has_isr_truth":     bool(_isr_profiles_for_window(edps, t_centre)),
                "obs_mode_status": {
                    obs_mode: {
                        filter_type: _result_status(group_filter_results[obs_mode][filter_type])
                        for filter_type in FILTER_TYPES
                    }
                    for obs_mode in OBS_MODES
                },
                "n_metrics_rows": len(group_metrics),
                "plots_written":  plots_written,
                "status":         "complete",
                "completed_at":   pd.Timestamp.now().isoformat(),
            }
            _save_progress_manifest(progress_manifest)

        if no_plot:
            _mark_group_complete(plots_written=False)
            return

        window_edps = _isr_profiles_for_window(edps, t_centre)
        window_isr_profiles = [_isr_edp_to_profile(e) for e in window_edps]

        # Bin-tag the save_dir root passed to these two plotting helpers (they
        # each build their own group_key-keyed subfolder beneath it) so
        # different OCC_COUNT_BINS sweep points for the same window land in
        # separate figure directories instead of overwriting one another.
        bin_save_dir = SAVE_DIR / f"bin_{bin_label}"

        _plot_group_all_modes(
            group_key, group_filter_results, igs_window_arcs,
            window_isr_profiles, bin_save_dir, window_edps,
        )

        if window_edps:
            solar = get_solar_conditions(t_centre)
            group_save_dir = bin_save_dir / _safe_group_key(group_key)
            # plot_isr_truth_comparison's output filename is keyed only by
            # (group_key, inst_name) -- when a window contains several scans
            # from the same site, looping over every scan silently overwrote
            # the earlier scans' figures (and re-ran the full plot each time
            # for nothing). Keep just the scan closest to the window centre
            # per site instead.
            best_by_site: dict[str, tuple[float, dict]] = {}
            for edp in window_edps:
                inst_name = _identify_instrument(float(edp["lat"]))
                edp_time = pd.Timestamp(edp["time"])
                if edp_time.tzinfo is not None:
                    edp_time = edp_time.tz_localize(None)
                dt = abs((edp_time - t_centre).total_seconds())
                if inst_name not in best_by_site or dt < best_by_site[inst_name][0]:
                    best_by_site[inst_name] = (dt, edp)
            for _dt, edp in best_by_site.values():
                plot_isr_truth_comparison(
                    edp, group_filter_results, group_key, solar, group_save_dir,
                )

        # ── Per-(obs_mode, filter_type) prior/posterior/truth occultation
        #    diagnostics -- one representative real RO ray per combo (never
        #    a collapsed single-epoch IGS arc), so we can actually see what
        #    the joint update did to a ray's electron-density curtain instead
        #    of only the aggregate RMSE/foF2 numbers.
        #
        #    Selection: a single group can hold several occultations of the
        #    *same* PRN (different LEOs tracking the same GNSS satellite),
        #    and picking purely by "closest tangent point to the ISR site"
        #    can silently prefer a short, high-altitude-only arc fragment
        #    (e.g. a ray that only got tracked down to ~500 km, well above
        #    the F2 peak, but happens to sit geographically near the site)
        #    over a geographically-farther ray that actually descends
        #    through the whole ionosphere. So: first restrict to rays with a
        #    genuine full profile (minimum tangent altitude below
        #    OCC_DIAG_FULL_PROFILE_MAX_ALT_KM), then, among those, pick the
        #    one whose TEC-max tangent point (the deepest/closest-approach
        #    point, used elsewhere in the codebase -- see
        #    demo_compare_kf_enkf._arc_representative_tangent -- as a proxy
        #    for the point of maximum columnar electron content along the
        #    ray) is nearest a co-located ISR scan, so the truth row is both
        #    populated and geographically meaningful. Falls back to the
        #    deepest-reaching ray available if nothing reaches below the
        #    full-profile threshold this window.
        occdiag_save_dir = bin_save_dir / _safe_group_key(group_key) / "occ_diagnostics"
        diag_edp = None
        if window_edps:
            def _edp_dt(e):
                t = pd.Timestamp(e["time"])
                if t.tzinfo is not None:
                    t = t.tz_localize(None)
                return abs((t - t_centre).total_seconds())
            diag_edp = min(window_edps, key=_edp_dt)
        diag_isr_profile = _isr_edp_to_profile(diag_edp) if diag_edp is not None else None
        diag_isr_site = (
            (float(diag_edp["lon"]), float(diag_edp["lat"]))
            if diag_edp is not None else None
        )

        for obs_mode in OBS_MODES:
            for filter_type in FILTER_TYPES:
                result = group_filter_results.get(obs_mode, {}).get(filter_type)
                if result is None or result.get("status", "Success") != "Success":
                    continue
                clean_list = result.get("clean_list") or []
                ro_indices = [
                    i for i, occ in enumerate(clean_list)
                    if len(occ.get("tangent_km", [])) > 1
                ]
                if not ro_indices:
                    continue

                def _min_tangent_alt(i, _clean_list=clean_list):
                    return float(np.min(_clean_list[i]["tangent_km"]))

                full_profile_indices = [
                    i for i in ro_indices
                    if _min_tangent_alt(i) < OCC_DIAG_FULL_PROFILE_MAX_ALT_KM
                ]
                if full_profile_indices:
                    candidate_indices = full_profile_indices
                else:
                    print(f"  [occ-diag] {group_key} {obs_mode}/{filter_type}: "
                          f"no occultation reaches below "
                          f"{OCC_DIAG_FULL_PROFILE_MAX_ALT_KM:.0f}km tangent alt; "
                          f"falling back to deepest-reaching ray available.")
                    candidate_indices = ro_indices

                if diag_isr_site is not None:
                    site_lon, site_lat = diag_isr_site
                    from demo_compare_kf_enkf import _arc_representative_tangent

                    def _tec_max_dist(i, _clean_list=clean_list):
                        occ = _clean_list[i]
                        la, lo = _arc_representative_tangent(
                            np.asarray(occ["LEO"]), np.asarray(occ["GNSS"])
                        )
                        return _haversine_km(la, lo, site_lat, site_lon)

                    occ_idx = min(candidate_indices, key=_tec_max_dist)
                else:
                    occ_idx = min(candidate_indices, key=_min_tangent_alt)

                try:
                    plot_occultation_prior_post_truth(
                        result, occ_idx, result.get("alt_grid", ALT_GRID), group_key,
                        occdiag_save_dir, label=f"{obs_mode}_{filter_type}",
                        isr_profile=diag_isr_profile, isr_site=diag_isr_site,
                    )
                except Exception:
                    print(f"  [occ-diag] plot failed for {group_key} "
                          f"{obs_mode}/{filter_type}; continuing.")
                    traceback.print_exc()

        _mark_group_complete(plots_written=True)

    # ── Keep only windows with co-located ISR truth ──────────────────────────
    # A window with no co-located ISR profile (within ISR_WINDOW_HALF_MINUTES
    # of a known ISR site) can never produce an ISR comparison metric/plot
    # (see compute_isr_metrics / window_edps above). By default we DROP those
    # ISR-blind windows entirely so the pipeline only runs where truth
    # verification data exists. Pass --all-windows to keep them (they are then
    # sorted last, most-occultations-first, as before).
    def _isr_edp_count(window: dict) -> int:
        return len(_isr_profiles_for_window(edps, window["t_centre"]))

    _n_total = len(windows)
    _isr_count = {id(w): _isr_edp_count(w) for w in windows}
    _isr_windows  = [w for w in windows if _isr_count[id(w)] > 0]
    _blind_windows = [w for w in windows if _isr_count[id(w)] == 0]
    # "Best" == most RO occultations, tie-broken by most ISR truth EDPs.
    _isr_windows.sort(key=lambda w: (w["n_occ"], _isr_count[id(w)]), reverse=True)

    if BEST_WINDOW_ONLY:
        # Investigate only the single best (most occultations + ISR truth)
        # window for this day. The OCC_COUNT_BINS sweep still runs on that one
        # window, so the occultation-count sensitivity study is preserved.
        if _isr_windows:
            _best = _isr_windows[0]
            windows = [_best]
            print(f"  [diag] BEST-WINDOW-ONLY: {day_info.get('date')} -> "
                  f"window {_best['window_key']} "
                  f"(n_occ={_best['n_occ']}, "
                  f"n_isr_edp={_isr_count[id(_best)]}) selected from "
                  f"{len(_isr_windows)} ISR-truth window(s).")
        else:
            windows = []
            print(f"  [diag] BEST-WINDOW-ONLY: no ISR-truth window for "
                  f"{day_info.get('date')}; nothing to run.")
    elif REQUIRE_ISR_TRUTH:
        windows = _isr_windows
        print(f"  [diag] {len(windows)}/{_n_total} window(s) have co-located ISR "
              f"truth; dropped {len(_blind_windows)} ISR-blind window(s) "
              f"(--all-windows to keep them).")
        if not windows:
            print(f"  [diag] No ISR-truth windows for {day_info.get('date')}; "
                  f"nothing to run.")
    else:
        _blind_windows.sort(key=lambda w: -w["n_occ"])
        windows = _isr_windows + _blind_windows
        print(f"  [diag] Reordered {len(windows)} window(s): "
              f"{len(_isr_windows)} ISR-aligned (most occultations first), "
              f"{len(_blind_windows)} without ISR truth (run last).")

    # Each (window, bin_count) unit is processed in its own try/except so an
    # unexpected exception in one unit (ROI setup, adaptation code, etc. --
    # anything outside the per-filter try/except in _run_or_load) can't
    # discard already-cached results/plots for other units in the same day,
    # which previously propagated all the way out of run_all_filters() and
    # skipped the entire day's plotting pass in main(), even for units that
    # had fully succeeded. bin_count=None (all occultations) runs first for
    # each window, then OCC_COUNT_BINS' decreasing counts, matching
    # test_param_iono.py's sweep ordering.
    for window in windows:
        for bin_count in OCC_COUNT_BINS:
            try:
                _process_window_bin(window, bin_count)
            except Exception:
                print(f"  [error] window {window['window_key']} bin={_bin_label(bin_count)} "
                      f"failed with an unexpected exception; skipping to the next "
                      f"unit so already-computed results can still be plotted.")
                traceback.print_exc()

    return results, all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _pct_improvement(prior_val: float, post_val: float) -> float:
    """
    % improvement of post_val over prior_val for an error metric where lower
    is better (e.g. RMSE): positive means post is smaller (improved).
    NaN if prior_val is 0/NaN or post_val is NaN (can't form a ratio).
    """
    if not np.isfinite(prior_val) or prior_val == 0 or not np.isfinite(post_val):
        return np.nan
    return 100.0 * (prior_val - post_val) / prior_val


def _masked_mae(est: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute error of *est* vs *ref* over the boolean *mask*; NaN if the
    mask selects nothing. Used for the whole-profile / below-peak posterior-vs-
    ISR error metrics in compute_isr_metrics."""
    return float(np.mean(np.abs(est[mask] - ref[mask]))) if np.any(mask) else np.nan


def compute_isr_metrics(day_info: dict, filter_results: dict, edps: list[dict]) -> list[dict]:
    """
    Compute posterior-vs-ISR-truth error metrics for one RO group's filter
    results.

    Parameters
    ----------
    day_info       : dict with at least "date" and "group_key".
    filter_results : nested dict, filter_results[obs_mode][filter_type] -> result
                      dict (from process_group / run_info_window / _run_parametric_ekf)
                      or None.
    edps           : full list of ISR EDP dicts (from load_edps()).

    Returns
    -------
    List of metric-row dicts, one per (obs_mode, filter_type, ISR instrument)
    combination that had co-located ISR truth within the group's time window.

    New posterior/prior-vs-ISR columns (ISR profile is the reference throughout;
    Ne in m⁻³, plasma frequency in MHz):
      - {prior,post}_profile_mae_ne / _mhz     : whole-profile mean-abs error
        over all valid ISR gates.
      - {prior,post}_below_peak_mae_ne / _mhz  : mean-abs error restricted to
        below-F2-peak ISR gates.
      - {prior,post}_hf_refl_err_km_{f}mhz     : signed HF reflection-height
        error [km] (est − ISR) for f in HF_REFLECTION_FREQS_MHZ, i.e. the max
        altitude where f_p(z) ≥ f in the model column vs the ISR profile.
    foF2/foE errors ({prior,post}_foF2_err_mhz, {prior,post}_foE_err_mhz) are
    already emitted above and unchanged.
    """
    rows: list[dict] = []
    date       = day_info.get("date")
    group_key  = day_info.get("group_key")
    bin_count  = day_info.get("bin_count")
    bin_label  = day_info.get("bin_label", "all")
    print(f"\n▶▶▶ compute_isr_metrics START | group={group_key} bin={bin_label}")

    for obs_mode, per_filter in filter_results.items():
        for filter_type, result in per_filter.items():
            if result is None:
                continue
            if result.get("status", "Success") != "Success":
                continue
            if any(k not in result for k in
                   ("prior_edp_3d", "post_edp_3d", "alt_grid", "eds_occ")):
                # e.g. igs_only/gridded_kf uses a regular lat/lon grid
                # ("prior_edp"/"post_edp") rather than an eds_occ mesh.
                continue

            prior_edp_3d = np.asarray(result["prior_edp_3d"])
            post_edp_3d  = np.asarray(result["post_edp_3d"])
            alt_grid     = np.asarray(result["alt_grid"])
            geoloc       = np.asarray(result["eds_occ"].geolocation)  # (n_geo,2): lon, lat

            # Defensive guard: eds_occ.geolocation and prior_edp_3d/post_edp_3d
            # should always share the same vertex count (they're built from
            # the same grid), but a DA_CACHE pickle produced before a
            # ROI/grid-construction code change can have them out of sync
            # (see _adapt_igs_kf_result_for_plotting's docstring). Skip this
            # combo rather than crashing the whole run's metrics computation
            # with an out-of-bounds cKDTree column index below.
            if prior_edp_3d.ndim < 2 or geoloc.shape[0] != prior_edp_3d.shape[1]:
                print(f"  [warn] {group_key} bin={bin_label} {obs_mode}/{filter_type}: "
                      f"eds_occ has {geoloc.shape[0]} vertices but prior_edp_3d has "
                      f"{prior_edp_3d.shape[1] if prior_edp_3d.ndim >= 2 else '?'} columns "
                      f"-- stale/mismatched cache, skipping ISR metrics for this combo "
                      f"(re-run with --force to regenerate).")
                continue

            # ── Group/filter-level diagnostics (same for every ISR site in
            #    this window -- attached to each row below for easy
            #    histogramming/filtering without a join back to the caches).
            region              = result.get("region", np.nan)
            n_grid_points       = int(geoloc.shape[0])
            n_ro_occultations   = day_info.get("n_ro_occultations", result.get("n_occultations", np.nan))
            n_igs_arcs          = day_info.get("n_igs_arcs", np.nan)
            prior_tec_rmse      = result.get("prior_tec_rmse", np.nan)
            post_tec_rmse       = result.get("joint_post_tec_rmse", result.get("post_tec_rmse", np.nan))
            tec_rmse_pct_improvement = _pct_improvement(prior_tec_rmse, post_tec_rmse)
            ekf_converged       = result.get("ekf_converged", np.nan)
            ekf_n_iterations    = result.get("ekf_n_iterations", np.nan)

            t_centre = _parse_time_window(result.get("time_window", group_key))
            t_lo = t_centre - pd.Timedelta(minutes=ISR_WINDOW_HALF_MINUTES)
            t_hi = t_centre + pd.Timedelta(minutes=ISR_WINDOW_HALF_MINUTES)

            # ── ISR profiles co-located with this window/site ────────────────
            window_edps: list[tuple[str, dict]] = []
            for e in edps:
                e_time = pd.Timestamp(e["time"])
                if e_time.tzinfo is not None:
                    e_time = e_time.tz_localize(None)
                if not (t_lo <= e_time <= t_hi):
                    continue
                for inst_name in ISR_SITES:
                    inst = INSTRUMENTS[inst_name]
                    if (abs(e["lat"] - inst["lat"]) <= ISR_SITE_MATCH_DEG
                            and abs(e["lon"] - inst["lon"]) <= ISR_SITE_MATCH_DEG):
                        window_edps.append((inst_name, e))
                        break

            if not window_edps:
                continue

            mesh_pts = np.column_stack([geoloc[:, 1], geoloc[:, 0]])  # (lat, lon)
            tree = cKDTree(mesh_pts)

            for inst_name, edp in window_edps:
                inst = INSTRUMENTS[inst_name]
                _dist, nearest_idx = tree.query([inst["lat"], inst["lon"]])

                prior_col = prior_edp_3d[:, nearest_idx]
                post_col  = post_edp_3d[:, nearest_idx]

                isr_alt = np.asarray(edp["alt_km"])
                isr_ne  = np.asarray(edp["ne_m3"])

                # F2 peak altitude from the prior column; restrict RMSE to below-peak altitudes.
                _, prior_hmF2 = extract_robust_f2_peak(prior_col, alt_grid)
                below_peak = (isr_alt <= prior_hmF2) if np.isfinite(prior_hmF2) \
                             else np.ones(len(isr_alt), dtype=bool)

                prior_at_isr = np.interp(isr_alt, alt_grid, prior_col)
                post_at_isr  = np.interp(isr_alt, alt_grid, post_col)

                valid = (isr_ne > 1e8) & np.isfinite(isr_ne) & below_peak
                if valid.sum() < ISR_MIN_VALID_GATES:
                    continue

                prior_rmse = float(np.sqrt(np.mean(
                    (prior_at_isr[valid] - isr_ne[valid]) ** 2)))
                post_rmse = float(np.sqrt(np.mean(
                    (post_at_isr[valid] - isr_ne[valid]) ** 2)))

                pr_nm, pr_hm = extract_robust_f2_peak(prior_col, alt_grid)
                po_nm, po_hm = extract_robust_f2_peak(post_col, alt_grid)
                isr_nm, isr_hm = extract_robust_f2_peak(isr_ne, isr_alt)

                if np.isfinite(isr_nm) and isr_nm != 0:
                    prior_NmF2_err_pct = 100.0 * (pr_nm - isr_nm) / isr_nm
                    post_NmF2_err_pct  = 100.0 * (po_nm - isr_nm) / isr_nm
                else:
                    prior_NmF2_err_pct = np.nan
                    post_NmF2_err_pct  = np.nan

                if np.isfinite(isr_hm):
                    prior_hmF2_err_km = pr_hm - isr_hm
                    post_hmF2_err_km  = po_hm - isr_hm
                else:
                    prior_hmF2_err_km = np.nan
                    post_hmF2_err_km  = np.nan

                # ── Frequency-domain metrics ─────────────────────────────────
                # foF2 = critical frequency of the F2 layer as seen from the ground [MHz]
                # foE  = blanketing frequency of the E layer [MHz]
                # These are the HF propagation quantities that operators care about.

                prior_foF2 = float(ne_to_mhz(pr_nm)) if np.isfinite(pr_nm) else np.nan
                post_foF2  = float(ne_to_mhz(po_nm)) if np.isfinite(po_nm) else np.nan
                isr_foF2   = float(ne_to_mhz(isr_nm)) if np.isfinite(isr_nm) else np.nan

                prior_foF2_err = prior_foF2 - isr_foF2   # signed [MHz]
                post_foF2_err  = post_foF2  - isr_foF2

                # E-layer peak from ISR profile (interpolated filter profile at ISR altitudes)
                isr_nme, isr_hme     = extract_e_layer_peak(isr_ne,         isr_alt)
                prior_nme, prior_hme = extract_e_layer_peak(prior_at_isr,   isr_alt)
                post_nme,  post_hme  = extract_e_layer_peak(post_at_isr,    isr_alt)

                prior_foE = float(ne_to_mhz(prior_nme)) if np.isfinite(prior_nme) else np.nan
                post_foE  = float(ne_to_mhz(post_nme))  if np.isfinite(post_nme)  else np.nan
                isr_foE   = float(ne_to_mhz(isr_nme))   if np.isfinite(isr_nme)   else np.nan

                prior_foE_err = prior_foE - isr_foE
                post_foE_err  = post_foE  - isr_foE

                # Full-profile RMSE in plasma frequency [MHz] — the "entire profile" metric
                valid_fp = valid  # reuse existing gate (ne > 1e8, finite, below hmF2)
                isr_fp_arr   = ne_to_mhz(isr_ne)
                prior_fp_arr = ne_to_mhz(prior_at_isr)
                post_fp_arr  = ne_to_mhz(post_at_isr)
                prior_profile_fp_rmse = float(np.sqrt(np.mean((prior_fp_arr[valid_fp] - isr_fp_arr[valid_fp])**2))) \
                                         if valid_fp.any() else np.nan
                post_profile_fp_rmse  = float(np.sqrt(np.mean((post_fp_arr[valid_fp]  - isr_fp_arr[valid_fp])**2))) \
                                         if valid_fp.any() else np.nan

                # Threshold flags
                MHZ_THRESHOLDS = [0.5, 0.2, 0.1]
                threshold_fields: dict = {}
                for thr in MHZ_THRESHOLDS:
                    ts = str(thr).replace(".", "")
                    threshold_fields[f"prior_foF2_within_{ts}mhz"] = bool(np.isfinite(prior_foF2_err) and abs(prior_foF2_err) <= thr)
                    threshold_fields[f"post_foF2_within_{ts}mhz"]  = bool(np.isfinite(post_foF2_err)  and abs(post_foF2_err)  <= thr)
                    threshold_fields[f"prior_foE_within_{ts}mhz"]  = bool(np.isfinite(prior_foE_err)  and abs(prior_foE_err)  <= thr)
                    threshold_fields[f"post_foE_within_{ts}mhz"]   = bool(np.isfinite(post_foE_err)   and abs(post_foE_err)   <= thr)
                    threshold_fields[f"prior_profile_within_{ts}mhz"] = bool(np.isfinite(prior_profile_fp_rmse) and prior_profile_fp_rmse <= thr)
                    threshold_fields[f"post_profile_within_{ts}mhz"]  = bool(np.isfinite(post_profile_fp_rmse)  and post_profile_fp_rmse  <= thr)

                # ── Whole-profile & below-peak absolute error vs ISR truth ───
                # ISR profile (isr_ne / isr_alt) is the reference throughout.
                # `valid_all` gates the whole profile (finite, Ne>1e8); `valid`
                # (reused from the RMSE block above) additionally restricts to
                # below the F2 peak. Each computed in Ne [m⁻³] and, via the
                # already-built fp arrays, plasma frequency [MHz].
                valid_all = ((isr_ne > 1e8) & np.isfinite(isr_ne)
                             & np.isfinite(post_at_isr) & np.isfinite(prior_at_isr))

                prior_profile_mae_ne  = _masked_mae(prior_at_isr, isr_ne,     valid_all)
                post_profile_mae_ne   = _masked_mae(post_at_isr,  isr_ne,     valid_all)
                prior_profile_mae_mhz = _masked_mae(prior_fp_arr, isr_fp_arr, valid_all)
                post_profile_mae_mhz  = _masked_mae(post_fp_arr,  isr_fp_arr, valid_all)

                prior_below_peak_mae_ne  = _masked_mae(prior_at_isr, isr_ne,     valid)
                post_below_peak_mae_ne   = _masked_mae(post_at_isr,  isr_ne,     valid)
                prior_below_peak_mae_mhz = _masked_mae(prior_fp_arr, isr_fp_arr, valid)
                post_below_peak_mae_mhz  = _masked_mae(post_fp_arr,  isr_fp_arr, valid)

                # ── HF reflection-height error [km] per frequency ─────────────
                # Max altitude where f_p(z) ≥ freq (bottom-up) in the posterior/
                # prior model column (on alt_grid) vs the ISR profile (on
                # isr_alt); recorded as a signed height difference (est − ISR).
                prior_fp_col = ne_to_mhz(prior_col)
                post_fp_col  = ne_to_mhz(post_col)
                isr_fp_prof  = ne_to_mhz(isr_ne)
                hf_reflection_fields: dict = {}
                for _f in HF_REFLECTION_FREQS_MHZ:
                    _fs = f"{int(_f)}" if float(_f).is_integer() else str(_f).replace(".", "")
                    _isr_h   = _get_reflection_height(isr_fp_prof,  isr_alt,  _f)
                    _prior_h = _get_reflection_height(prior_fp_col, alt_grid, _f)
                    _post_h  = _get_reflection_height(post_fp_col,  alt_grid, _f)
                    hf_reflection_fields[f"prior_hf_refl_err_km_{_fs}mhz"] = (
                        float(_prior_h - _isr_h)
                        if np.isfinite(_prior_h) and np.isfinite(_isr_h) else np.nan)
                    hf_reflection_fields[f"post_hf_refl_err_km_{_fs}mhz"] = (
                        float(_post_h - _isr_h)
                        if np.isfinite(_post_h) and np.isfinite(_isr_h) else np.nan)

                rows.append({
                    "date":                date,
                    "group_key":           group_key,
                    "bin_count":           bin_count,
                    "bin_label":           bin_label,
                    "obs_mode":            obs_mode,
                    "filter_type":         filter_type,
                    "instrument":          inst_name,
                    "t_centre":            t_centre,
                    "region":              region,
                    "n_ro_occultations":   n_ro_occultations,
                    "n_occ_assimilated":   n_ro_occultations,
                    "n_igs_arcs":          n_igs_arcs,
                    "n_grid_points":       n_grid_points,
                    # TEC-space fit (RO/IGS observations vs. filter), same
                    # value for every ISR site in this window/obs_mode/filter.
                    "prior_tec_rmse":            prior_tec_rmse,
                    "post_tec_rmse":             post_tec_rmse,
                    "tec_rmse_pct_improvement":  tec_rmse_pct_improvement,
                    # EDP (absolute Ne, m⁻³) fit vs. ISR truth at this specific site.
                    "prior_edp_rmse":      prior_rmse,
                    "post_edp_rmse":       post_rmse,
                    "delta_rmse":          post_rmse - prior_rmse,
                    "edp_rmse_pct_improvement": _pct_improvement(prior_rmse, post_rmse),
                    "prior_NmF2_err_pct":  prior_NmF2_err_pct,
                    "post_NmF2_err_pct":   post_NmF2_err_pct,
                    "prior_hmF2_err_km":   prior_hmF2_err_km,
                    "post_hmF2_err_km":    post_hmF2_err_km,
                    # Frequency-domain metrics (HF propagation quantities) [MHz].
                    "isr_foF2":               isr_foF2,
                    "prior_foF2":             prior_foF2,
                    "post_foF2":              post_foF2,
                    "prior_foF2_err_mhz":     prior_foF2_err,
                    "post_foF2_err_mhz":      post_foF2_err,
                    "isr_foE":                isr_foE,
                    "prior_foE":              prior_foE,
                    "post_foE":               post_foE,
                    "prior_foE_err_mhz":      prior_foE_err,
                    "post_foE_err_mhz":       post_foE_err,
                    "prior_profile_fp_rmse_mhz": prior_profile_fp_rmse,
                    "post_profile_fp_rmse_mhz":  post_profile_fp_rmse,
                    # Whole-profile absolute error vs ISR (all valid gates),
                    # in Ne [m⁻³] and plasma frequency [MHz].
                    "prior_profile_mae_ne":    prior_profile_mae_ne,
                    "post_profile_mae_ne":     post_profile_mae_ne,
                    "prior_profile_mae_mhz":   prior_profile_mae_mhz,
                    "post_profile_mae_mhz":    post_profile_mae_mhz,
                    # Below-F2-peak absolute error vs ISR (below-peak gates).
                    "prior_below_peak_mae_ne":   prior_below_peak_mae_ne,
                    "post_below_peak_mae_ne":    post_below_peak_mae_ne,
                    "prior_below_peak_mae_mhz":  prior_below_peak_mae_mhz,
                    "post_below_peak_mae_mhz":   post_below_peak_mae_mhz,
                    # HF reflection-height error [km] per frequency (signed,
                    # est − ISR): prior_/post_hf_refl_err_km_{f}mhz.
                    **hf_reflection_fields,
                    **threshold_fields,
                    "n_isr_gates_valid":   int(valid.sum()),
                    "ekf_converged":       ekf_converged,
                    "ekf_n_iterations":    ekf_n_iterations,
                })

    print(f"◀◀◀ compute_isr_metrics END   | group={group_key} bin={bin_label} -> {len(rows)} metric row(s)")
    return rows


_METRICS_DEDUP_COLS = ["group_key", "bin_count", "obs_mode", "filter_type", "instrument"]


def _append_metrics_csv(metrics_rows: list[dict]) -> pd.DataFrame:
    """
    Append *metrics_rows* to the on-disk ISR-metrics CSV (ISR_METRICS_CSV)
    immediately, deduplicated on group_key+bin_count+obs_mode+filter_type+
    instrument (keeping the latest row for any repeat), and return the full
    accumulated DataFrame. bin_count is part of the key so the OCC_COUNT_BINS
    sweep's per-bin rows for the same window coexist instead of overwriting
    each other.

    Called right after each group's metrics are computed (inside
    run_all_filters) so results survive a crash/interrupt instead of only
    being written once at the very end of a full multi-day run. Safe to call
    with an empty list (no-op) or repeatedly with the same rows (dedup keeps
    the run idempotent).
    """
    DA_CACHE.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(metrics_rows)

    if ISR_METRICS_CSV.exists():
        existing_df = pd.read_csv(ISR_METRICS_CSV, parse_dates=["t_centre"])
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    if combined.empty:
        return combined

    combined = (combined
                .drop_duplicates(subset=_METRICS_DEDUP_COLS, keep="last")
                .sort_values(["date", "t_centre", "obs_mode", "filter_type", "instrument"])
                .reset_index(drop=True))
    combined.to_csv(ISR_METRICS_CSV, index=False)
    return combined


def summarize_statistics(metrics_rows: list[dict] | None = None) -> pd.DataFrame:
    """
    Aggregate per-group ISR metrics into summary statistics across days/modes.

    If *metrics_rows* is given, appends them to the on-disk CSV first (see
    _append_metrics_csv); otherwise reads whatever is already accumulated on
    disk (this is what lets a resumed/partial run still print full-run
    statistics from previously-completed days). Prints a per-(obs_mode,
    filter_type) summary table, saves that table to SAVE_DIR, and returns the
    full accumulated DataFrame.
    """
    DA_CACHE.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if metrics_rows:
        combined = _append_metrics_csv(metrics_rows)
    elif ISR_METRICS_CSV.exists():
        combined = pd.read_csv(ISR_METRICS_CSV, parse_dates=["t_centre"])
    else:
        combined = pd.DataFrame()

    if combined.empty:
        print("[ISR-DA] No metrics available to summarize.")
        return combined

    # ── Per-(obs_mode, filter_type) summary table ─────────────────────────────
    # delta_rmse is in absolute Ne (m⁻³), so it's printed in scientific
    # notation rather than the fixed-point format that suited the old
    # log10(Ne) RMSE (which lived in a small ~0-1 range).
    lines = [
        f"{'obs_mode':<10} {'filter_type':<14} {'n':>4}  "
        f"{'delta_rmse m⁻³ (mean±std)':>26}  {'frac_improved':>13}  "
        f"{'mean NmF2 err %':>16}  {'mean hmF2 err km':>17}",
    ]
    lines.append("-" * len(lines[0]))

    for (obs_mode, filter_type), grp in combined.groupby(["obs_mode", "filter_type"]):
        n = len(grp)
        mean_delta = grp["delta_rmse"].mean()
        std_delta  = grp["delta_rmse"].std()
        frac_improved = float((grp["delta_rmse"] < 0).mean())
        mean_nmf2_err = grp["post_NmF2_err_pct"].mean()
        mean_hmf2_err = grp["post_hmF2_err_km"].mean()
        lines.append(
            f"{obs_mode:<10} {filter_type:<14} {n:>4}  "
            f"{mean_delta:>10.3e} ± {std_delta:<9.3e}  {frac_improved:>12.1%}  "
            f"{mean_nmf2_err:>16.2f}  {mean_hmf2_err:>17.2f}"
        )

    # ── Threshold-based performance tables ────────────────────────────────────
    for metric_label, prior_col, post_col in [
        ("foF2 (critical freq)",    "prior_foF2_within_{}mhz", "post_foF2_within_{}mhz"),
        ("foE  (blanketing freq)",  "prior_foE_within_{}mhz",  "post_foE_within_{}mhz"),
        ("Profile fp RMSE",         "prior_profile_within_{}mhz", "post_profile_within_{}mhz"),
    ]:
        lines.append(f"\n  {metric_label} — fraction of cases within threshold:")
        lines.append(f"  {'obs_mode':<10} {'filter_type':<14} {'n':>4}  "
                     + "  ".join(f"{'prior|post @'+str(t)+'MHz':>20}" for t in [0.5, 0.2, 0.1]))
        for (obs_mode, filter_type), grp in combined.groupby(["obs_mode", "filter_type"]):
            n = len(grp)
            thr_parts = []
            for thr in [0.5, 0.2, 0.1]:
                ts = str(thr).replace(".", "")
                pc = prior_col.format(ts)
                po = post_col.format(ts)
                if pc in grp.columns and po in grp.columns:
                    prior_frac = grp[pc].mean()
                    post_frac  = grp[po].mean()
                    thr_parts.append(f"{prior_frac:>8.1%}|{post_frac:<8.1%}")
                else:
                    thr_parts.append("        N/A      ")
            lines.append(f"  {obs_mode:<10} {filter_type:<14} {n:>4}  " + "  ".join(thr_parts))

    summary_text = "\n".join(lines)
    print(summary_text)

    stats_path = SAVE_DIR / "statistics_summary.txt"
    stats_path.write_text(summary_text + "\n")

    if SAVE_DIR.exists():
        try:
            plot_isr_freq_metrics(ISR_METRICS_CSV, SAVE_DIR)
        except Exception:
            print("[ISR-DA] Frequency-domain plotting failed; continuing without it.")
            traceback.print_exc()

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _isr_edp_to_profile(edp: dict) -> dict:
    """
    Adapt a load_edps() ISR dict (time/lat/lon/alt_km/ne_m3) to the
    isr_profiles schema expected by _plot_group / plot_kf_enkf_comparison
    (hour_utc/alt_km/ne/nm_f2/hm_f2) — see demo_verification.load_isr_profiles.
    """
    t = pd.Timestamp(edp["time"])
    alt_km = np.asarray(edp["alt_km"], dtype=float)
    ne     = np.asarray(edp["ne_m3"],  dtype=float)
    nm_f2, hm_f2 = extract_robust_f2_peak(ne, alt_km)
    return {
        "hour_utc": t.hour + t.minute / 60.0 + t.second / 3600.0,
        "alt_km":   alt_km,
        "ne":       ne,
        "nm_f2":    nm_f2,
        "hm_f2":    hm_f2,
    }


def _isr_profiles_for_window(edps: list[dict], t_centre: pd.Timestamp) -> list[dict]:
    """
    ISR EDPs from *edps* co-located with a known ISR site (ISR_SITES) and
    within ISR_WINDOW_HALF_MINUTES of *t_centre*.  Mirrors the windowing
    logic in compute_isr_metrics.
    """
    t_lo = t_centre - pd.Timedelta(minutes=ISR_WINDOW_HALF_MINUTES)
    t_hi = t_centre + pd.Timedelta(minutes=ISR_WINDOW_HALF_MINUTES)
    matched: list[dict] = []
    for e in edps:
        e_time = pd.Timestamp(e["time"])
        if e_time.tzinfo is not None:
            e_time = e_time.tz_localize(None)
        if not (t_lo <= e_time <= t_hi):
            continue
        for inst_name in ISR_SITES:
            inst = INSTRUMENTS[inst_name]
            if (abs(e["lat"] - inst["lat"]) <= ISR_SITE_MATCH_DEG
                    and abs(e["lon"] - inst["lon"]) <= ISR_SITE_MATCH_DEG):
                matched.append(e)
                break
    return matched


def plot_isr_freq_metrics(metrics_csv: str | Path, save_dir: str | Path) -> None:
    """
    Read the accumulated ISR-metrics CSV (see compute_isr_metrics /
    _append_metrics_csv) and render the HF-propagation (foF2/foE) summary
    figures into *save_dir*:

      1. isr_foF2_improvement_boxplot.png
      2. isr_foE_improvement_boxplot.png
      3. isr_threshold_fractions.png
      4. isr_foF2_scatter_by_mode.png
      5. isr_hf_propagation_timeseries.png

    No-ops (with a printed message) if the CSV is missing/empty or predates
    the frequency-domain columns.
    """
    metrics_csv = Path(metrics_csv)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_csv.exists():
        print(f"[ISR-DA] {metrics_csv} not found; skipping frequency-metric plots.")
        return

    df = pd.read_csv(metrics_csv, parse_dates=["t_centre"])
    if df.empty or "post_foF2_err_mhz" not in df.columns:
        print("[ISR-DA] No frequency-domain metrics available; skipping frequency-metric plots.")
        return

    thresholds = [0.5, 0.2, 0.1]
    combos = [(om, ft) for om in OBS_MODES for ft in FILTER_TYPES]
    obs_mode_colors = dict(zip(OBS_MODES, plt.cm.tab10.colors))

    # ── Figures 1 & 2: foF2 / foE improvement boxplots ─────────────────────
    for metric, fname in [("foF2", "isr_foF2_improvement_boxplot.png"),
                           ("foE",  "isr_foE_improvement_boxplot.png")]:
        prior_col = f"prior_{metric}_err_mhz"
        post_col  = f"post_{metric}_err_mhz"
        if prior_col not in df.columns or post_col not in df.columns:
            continue

        fig, axes = plt.subplots(1, len(ISR_SITES), figsize=(6 * len(ISR_SITES), 5), squeeze=False)
        axes = axes[0]
        for ax, site in zip(axes, ISR_SITES):
            site_df = df[df["instrument"] == site]
            positions, box_data, box_colors, tick_labels = [], [], [], []
            pos = 0
            for obs_mode, filter_type in combos:
                grp = site_df[(site_df["obs_mode"] == obs_mode) & (site_df["filter_type"] == filter_type)]
                box_data.append(grp[prior_col].abs().dropna().values)
                positions.append(pos)
                box_colors.append("lightgrey")
                box_data.append(grp[post_col].abs().dropna().values)
                positions.append(pos + 1)
                box_colors.append("tab:blue")
                tick_labels.append((pos + 0.5, f"{obs_mode}\n{filter_type}"))
                pos += 3

            bp = ax.boxplot(box_data, positions=positions, widths=0.8, patch_artist=True)
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)

            for thr in thresholds:
                ax.axhline(thr, color="tab:red", ls="--", lw=0.8, alpha=0.6)
                ax.text(positions[-1] + 1.2, thr, f"{thr} MHz", fontsize=7, color="tab:red", va="center")

            ax.set_xticks([p for p, _ in tick_labels])
            ax.set_xticklabels([lbl for _, lbl in tick_labels], fontsize=8)
            ax.set_ylabel(f"|{metric} error| [MHz]")
            ax.set_title(f"{site}: {metric} error (grey=prior, blue=post)")

        fig.suptitle(f"ISR {metric} retrieval error by obs_mode / filter_type")
        fig.tight_layout()
        fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 3: fraction-within-threshold bar chart ──────────────────────
    metric_bar_specs = ["foF2", "foE", "profile"]
    bar_labels = [f"{m} {phase}" for m in metric_bar_specs for phase in ("prior", "post")]
    n_bars = len(bar_labels)
    thr_cols_exist = all(
        f"{phase}_{m}_within_{str(t).replace('.', '')}mhz" in df.columns
        for m in metric_bar_specs for phase in ("prior", "post") for t in thresholds)

    if thr_cols_exist:
        fig, axes = plt.subplots(1, len(thresholds), figsize=(6 * len(thresholds), 5), sharey=True)
        width = 0.8 / n_bars
        x0 = np.arange(len(combos)) * (n_bars * width + 1.0)

        for ax, thr in zip(axes, thresholds):
            ts = str(thr).replace(".", "")
            bar_idx = 0
            for m in metric_bar_specs:
                for phase in ("prior", "post"):
                    col = f"{phase}_{m}_within_{ts}mhz"
                    vals = [df[(df["obs_mode"] == om) & (df["filter_type"] == ft)][col].mean()
                            for om, ft in combos]
                    ax.bar(x0 + bar_idx * width, vals, width=width,
                           label=bar_labels[bar_idx] if ax is axes[0] else None,
                           color=plt.cm.tab20(bar_idx / n_bars))
                    bar_idx += 1
            ax.set_xticks(x0 + (n_bars * width) / 2 - width / 2)
            ax.set_xticklabels([f"{om}\n{ft}" for om, ft in combos], fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_title(f"within {thr} MHz")
            ax.set_ylabel("fraction of cases")

        axes[0].legend(fontsize=7, loc="upper right")
        fig.suptitle("Fraction of ISR comparisons within frequency threshold")
        fig.tight_layout()
        fig.savefig(save_dir / "isr_threshold_fractions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 4: foF2 scatter, truth vs. posterior, per EKF filter mode ───
    # One figure per active EKF filter_type (parametric_ekf / ekf_allfree /
    # ekf_nmf2 / …); filename suffixed by the label so runs with several EKF
    # modes don't clobber each other.
    _ekf_labels = [ft for ft in df["filter_type"].unique() if ft != "gridded_kf"]
    for _ekf_label in _ekf_labels:
        ekf_df = df[df["filter_type"] == _ekf_label]
        if ekf_df.empty or not {"isr_foF2", "post_foF2"}.issubset(ekf_df.columns):
            continue
        fig, axes = plt.subplots(1, len(OBS_MODES), figsize=(6 * len(OBS_MODES), 5.5), squeeze=False)
        axes = axes[0]
        sm = None
        for ax, obs_mode in zip(axes, OBS_MODES):
            grp = ekf_df[ekf_df["obs_mode"] == obs_mode].dropna(subset=["isr_foF2", "post_foF2"])
            if grp.empty:
                ax.set_title(f"{obs_mode} (no data)")
                continue
            c = grp["n_ro_occultations"] if "n_ro_occultations" in grp.columns else None
            sc = ax.scatter(grp["isr_foF2"], grp["post_foF2"], c=c, cmap="viridis",
                             s=30, edgecolor="k", linewidth=0.3)
            lo = min(grp["isr_foF2"].min(), grp["post_foF2"].min())
            hi = max(grp["isr_foF2"].max(), grp["post_foF2"].max())
            ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1.0, label="perfect retrieval")
            ax.set_xlabel("ISR truth foF2 [MHz]")
            ax.set_ylabel("Posterior foF2 [MHz]")
            ax.set_title(obs_mode)
            ax.legend(fontsize=7)
            if c is not None:
                sm = sc
        if sm is not None:
            fig.colorbar(sm, ax=axes.tolist(), label="n_ro_occultations", fraction=0.03, pad=0.02)
        fig.suptitle(f"{_ekf_label}: ISR truth vs. posterior foF2")
        fig.savefig(save_dir / f"isr_foF2_scatter_by_mode_{_ekf_label}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 5: HF propagation perspective (time series), per EKF mode ────
    if {"isr_foF2", "post_foF2", "isr_foE", "post_foE"}.issubset(df.columns):
        for _ekf_label in _ekf_labels:
            ekf_ts_df = df[df["filter_type"] == _ekf_label].sort_values("t_centre")
            if ekf_ts_df.empty:
                continue
            fig, axes = plt.subplots(len(ISR_SITES), 1, figsize=(11, 4.5 * len(ISR_SITES)), squeeze=False)
            axes = axes[:, 0]
            for ax, site in zip(axes, ISR_SITES):
                site_df = ekf_ts_df[ekf_ts_df["instrument"] == site]
                if site_df.empty:
                    ax.set_title(f"{site} (no data)")
                    continue
                truth = site_df.drop_duplicates(subset=["t_centre"]).sort_values("t_centre")
                ax.plot(truth["t_centre"], truth["isr_foF2"], color="black", lw=1.6,
                         marker="o", ms=3, label="truth foF2")
                ax.plot(truth["t_centre"], truth["isr_foE"], color="black", lw=1.2, ls="--",
                         marker="o", ms=3, label="truth foE")
                for obs_mode in OBS_MODES:
                    mode_df = site_df[site_df["obs_mode"] == obs_mode].sort_values("t_centre")
                    if mode_df.empty:
                        continue
                    color = obs_mode_colors[obs_mode]
                    ax.plot(mode_df["t_centre"], mode_df["post_foF2"], color=color, lw=1.3,
                             marker="s", ms=3, label=f"{obs_mode} post foF2")
                    ax.plot(mode_df["t_centre"], mode_df["post_foE"], color=color, lw=1.0, ls="--",
                             marker="s", ms=3, label=f"{obs_mode} post foE")
                ax.set_ylabel("Frequency [MHz]")
                ax.set_title(f"{site}: HF propagation frequencies ({_ekf_label})")
                ax.legend(fontsize=6, ncol=3)
            axes[-1].set_xlabel("time (t_centre)")
            fig.tight_layout()
            fig.savefig(save_dir / f"isr_hf_propagation_timeseries_{_ekf_label}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"[ISR-DA] Frequency-domain figures saved to {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Occultation-count sensitivity study (real ISR ground truth)
# ─────────────────────────────────────────────────────────────────────────────

_OCC_CONVERGENCE_FILTER_STYLE = {
    "gridded_kf":     dict(color="steelblue", ls="-",  marker="s"),
    "parametric_ekf": dict(color="crimson",   ls="--", marker="^"),
    "ekf_allfree":    dict(color="crimson",   ls="--", marker="^"),
    "ekf_nmf2":       dict(color="darkorange", ls=":", marker="D"),
}

# Fallback cycle for any EKF filter_type label not explicitly styled above.
_EKF_STYLE_CYCLE = [
    dict(color="crimson",    ls="--", marker="^"),
    dict(color="darkorange", ls=":",  marker="D"),
    dict(color="purple",     ls="-.", marker="v"),
    dict(color="green",      ls="--", marker="P"),
]


def _ekf_filter_labels():
    """EKF filter_type labels currently active (everything except gridded_kf)."""
    return [ft for ft in FILTER_TYPES if ft != "gridded_kf"]


def _filter_style(filter_type, _idx=0):
    """Plot style for a filter_type, falling back to an EKF cycle entry."""
    if filter_type in _OCC_CONVERGENCE_FILTER_STYLE:
        return _OCC_CONVERGENCE_FILTER_STYLE[filter_type]
    return _EKF_STYLE_CYCLE[_idx % len(_EKF_STYLE_CYCLE)]

_OCC_CONVERGENCE_METRICS = {
    "below_peak_ne_mae": dict(
        title="Below-F2-peak Ne error vs. ISR truth",
        ylabel="Below-peak Ne MAE  [m$^{-3}$]",
    ),
    "foF2_error_mhz": dict(
        title="foF2 error vs. ISR truth",
        ylabel="|foF2 error|  [MHz]",
    ),
    "hf_reflection_height_error_km": dict(
        title="HF reflection-height error vs. ISR truth",
        ylabel="|HF reflection-height error|  [km]  "
               f"(mean over {HF_REFLECTION_FREQS_MHZ} MHz)",
    ),
}


def _add_occ_convergence_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the three posterior-error columns plotted by
    plot_isr_convergence_vs_occ_count() from the raw ISR_METRICS_CSV fields
    written by compute_isr_metrics(). All three are non-negative "how far
    from ISR truth" magnitudes so they can be aggregated (median, std) the
    same way regardless of sign convention in the source column(s).
    """
    df = df.copy()
    df["below_peak_ne_mae"] = df["post_below_peak_mae_ne"]
    df["foF2_error_mhz"] = df["post_foF2_err_mhz"].abs()

    hf_cols = [f"post_hf_refl_err_km_{int(f) if float(f).is_integer() else str(f).replace('.', '')}mhz"
               for f in HF_REFLECTION_FREQS_MHZ]
    hf_cols = [c for c in hf_cols if c in df.columns]
    if hf_cols:
        df["hf_reflection_height_error_km"] = df[hf_cols].abs().mean(axis=1, skipna=True)
    else:
        df["hf_reflection_height_error_km"] = np.nan
    return df


def _occ_convergence_series(
    df: pd.DataFrame, metric_col: str, obs_mode: str, filter_type: str,
) -> "dict | None":
    """
    Group *df* (already filtered to one metric's non-null rows) by bin_label
    for a given (obs_mode, filter_type), pooling across windows/instruments.
    One point per bin_label: x = mean n_occ_assimilated in that bin, y =
    median of the metric, with the sample std as a +/-1 sigma band.  The
    "all" bin (bin_count is NaN, i.e. every available arc used) is placed at
    the largest x among the other bins, since it otherwise wouldn't sort
    correctly against the fixed OCC_COUNT_BINS values.
    """
    sub = df[(df["obs_mode"] == obs_mode) & (df["filter_type"] == filter_type)]
    sub = sub.dropna(subset=[metric_col, "n_occ_assimilated"])
    if sub.empty:
        return None

    rows = []
    for bin_label, g in sub.groupby("bin_label"):
        vals = g[metric_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        n_x = float(g["n_occ_assimilated"].mean())
        rows.append([bin_label, n_x, float(np.median(vals)),
                     float(np.std(vals)) if vals.size > 1 else 0.0])
    if not rows:
        return None

    max_x = max(r[1] for r in rows)
    for r in rows:
        if r[0] == "all":
            r[1] = max_x
    rows.sort(key=lambda r: r[1])

    return dict(
        bin_labels=[r[0] for r in rows],
        n=np.array([r[1] for r in rows], dtype=float),
        median=np.array([r[2] for r in rows], dtype=float),
        std=np.array([r[3] for r in rows], dtype=float),
    )


def _plot_occ_convergence_panel(ax, series: "dict | None", filter_type: str, fit: bool) -> "dict | None":
    """
    Draw one (obs_mode, filter_type) convergence line + +/-1 sigma band onto
    *ax*. If *fit* is True, also fits RMSE(n) = a * n**(-b) (via
    test_param_iono._fit_power_law) and appends "(b=.., R²=..)" to the legend
    label. Returns the fit dict (or None) so callers can print/annotate it
    elsewhere too.
    """
    style = _OCC_CONVERGENCE_FILTER_STYLE[filter_type]
    if series is None:
        return None

    n, median, std = series["n"], series["median"], series["std"]
    label = filter_type
    fit_result = _fit_power_law(n, median) if fit else None
    if fit_result is not None:
        label += f"  (b={fit_result['b']:.2f}, R²={fit_result['r2']:.2f})"

    ax.plot(n, median, color=style["color"], ls=style["ls"], marker=style["marker"],
             lw=1.8, markersize=5, label=label)
    ax.fill_between(n, median - std, median + std, color=style["color"], alpha=0.15,
                     linewidth=0)
    return fit_result


def plot_isr_convergence_vs_occ_count(
    metrics_csv: "str | Path" = ISR_METRICS_CSV,
    save_dir: "str | Path" = SAVE_DIR,
) -> None:
    """
    Occultation-count sensitivity study against REAL ISR ground truth
    (compute_isr_metrics / load_edps) -- the ISR-DA-comparison analogue of
    test_param_iono.plot_convergence_vs_measurement_count(), which instead
    compares against synthetic IRI truth.

    Reads the accumulated ISR_METRICS_CSV (bin_count/bin_label/
    n_occ_assimilated columns written by compute_isr_metrics per the
    OCC_COUNT_BINS sweep) and, for each metric of interest, draws a 1x3
    panel row (one panel per OBS_MODES entry). Each panel shows one line per
    FILTER_TYPES entry (gridded_kf solid, parametric_ekf dashed), median +/-
    1 sigma across windows/instruments, vs. n_occ_assimilated.

    ro_only / ro_igs panels get a RMSE(n) = a*n**(-b) power-law fit (exponent
    b and R² annotated in the legend) since those modes actually assimilate a
    varying number of occultations. igs_only doesn't depend on RO count at
    all -- its n_occ_assimilated column merely echoes the RO bin size of the
    window it was computed alongside (see compute_isr_metrics), so it is
    drawn as a flat horizontal reference line (overall median) rather than a
    fitted series, spanning the same x-range as the other two panels for
    visual comparison.

    The scientific question this makes visible: does adding IGS TEC
    (ro_igs vs. ro_only) let the filters reach the same ISR-truth accuracy
    with fewer occultations?

    Saved to {save_dir}/isr_convergence_vs_occ_count_{metric}.png, one file
    per entry in _OCC_CONVERGENCE_METRICS.
    """
    metrics_csv = Path(metrics_csv)
    save_dir = Path(save_dir)
    if not metrics_csv.exists():
        print(f"  [plot_isr_convergence_vs_occ_count] {metrics_csv} not found -- skipping.")
        return

    df = pd.read_csv(metrics_csv, parse_dates=["t_centre"])
    if df.empty:
        print("  [plot_isr_convergence_vs_occ_count] ISR metrics CSV is empty -- skipping.")
        return
    if "bin_label" not in df.columns or "n_occ_assimilated" not in df.columns:
        print("  [plot_isr_convergence_vs_occ_count] bin_label/n_occ_assimilated "
              "columns not found in ISR metrics CSV -- skipping.")
        return

    df = _add_occ_convergence_metric_columns(df)
    save_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, meta in _OCC_CONVERGENCE_METRICS.items():
        if metric_key not in df.columns:
            continue

        fig, axes = plt.subplots(1, len(OBS_MODES), figsize=(6 * len(OBS_MODES), 5), squeeze=False)
        axes = axes[0]

        # Non-igs_only panels drive the shared x-range that the igs_only
        # reference lines are drawn across.
        shared_xlim = [np.inf, -np.inf]

        for ax, obs_mode in zip(axes, OBS_MODES):
            has_data = False
            for filter_type in FILTER_TYPES:
                series = _occ_convergence_series(df, metric_key, obs_mode, filter_type)
                if series is None:
                    continue
                has_data = True
                if obs_mode == "igs_only":
                    ref_val = float(np.median(series["median"]))
                    style = _OCC_CONVERGENCE_FILTER_STYLE[filter_type]
                    ax.axhline(ref_val, color=style["color"], ls=style["ls"], lw=1.8,
                                label=f"{filter_type}  (flat ref., median={ref_val:.3g})")
                else:
                    _plot_occ_convergence_panel(ax, series, filter_type, fit=True)
                    shared_xlim[0] = min(shared_xlim[0], series["n"].min())
                    shared_xlim[1] = max(shared_xlim[1], series["n"].max())

            ax.set_title(obs_mode)
            ax.set_xlabel("Occultations assimilated (n_occ_assimilated)")
            ax.set_ylabel(meta["ylabel"])
            ax.grid(True, lw=0.3, alpha=0.4)
            if has_data:
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

        if np.isfinite(shared_xlim[0]) and np.isfinite(shared_xlim[1]) and shared_xlim[0] < shared_xlim[1]:
            igs_ax = axes[OBS_MODES.index("igs_only")] if "igs_only" in OBS_MODES else None
            if igs_ax is not None:
                igs_ax.set_xlim(shared_xlim[0], shared_xlim[1])

        fig.suptitle(f"{meta['title']} — vs. occultation count")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        save_path = save_dir / f"isr_convergence_vs_occ_count_{metric_key}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

class _Tee:
    """File-like object that writes to both an underlying stream (terminal)
    and a log file, so `print()`/traceback output is visible live *and*
    captured for later troubleshooting (see run log at LOG_DIR)."""

    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log_fh = log_fh

    def write(self, data):
        self._stream.write(data)
        self._log_fh.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log_fh.flush()

    def isatty(self):
        # argparse / some libraries check this; forward to the real stream
        # so behaviour (e.g. colour output) is unaffected.
        return getattr(self._stream, "isatty", lambda: False)()


def _setup_file_logging() -> tuple[Path, "object"]:
    """Tee stdout+stderr to a timestamped log file under LOG_DIR so a run's
    full output can be inspected after the fact (terminal scrollback alone
    is not enough for long/backgrounded runs). Returns (log_path, log_fh);
    caller is responsible for closing log_fh when done."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"isr_da_comparison_{stamp}.log"
    log_fh = open(log_path, "a", buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.stdout, log_fh)
    sys.stderr = _Tee(sys.stderr, log_fh)
    print(f"[ISR-DA] Logging this run to: {log_path}")
    return log_path, log_fh


def main() -> None:
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    log_path, log_fh = _setup_file_logging()
    try:
        _main_impl()
    except Exception:
        # Make sure the traceback for an unhandled exception lands in the
        # log file too (KeyboardInterrupt/SystemExit -- e.g. --help or the
        # existing Ctrl-C handler -- are intentional exits, not logged as
        # errors), then re-raise so normal process-exit behaviour is unchanged.
        traceback.print_exc()
        raise
    finally:
        print(f"[ISR-DA] Log written to: {log_path}")
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        log_fh.close()


def _main_impl() -> None:
    global EKF_ALPHA, EKF_SIGMA_OBS, EKF_MAX_RAYS, EKF_TOL, EKF_MAX_ITER
    global EKF_N_MEMBERS, EKF_PRIOR_SCALE, OCC_COUNT_BINS
    global EKF_FREE_PARAMS, EKF_ADAPT_ALPHA, EKF_ALPHA_MAX, EKF_TEC_RMSE_TOL
    global RUN_GRIDDED_KF, EKF_GRID_MODE, EKF_GRID_KM, EKF_GRID_MAX_PTS, EKF_ROI_GATE_KM
    global EKF_MODES, FILTER_TYPES
    global REQUIRE_ISR_TRUTH, BEST_WINDOW_ONLY
    global SAVE_DIR, ISR_METRICS_CSV, PROGRESS_MANIFEST
    parser = argparse.ArgumentParser(
        description="Compare KF/EKF data assimilation against ISR ground truth.")
    parser.add_argument("--force", action="store_true",
                        help="Ignore caches and recompute everything.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip figure generation.")
    parser.add_argument("--days", type=int, default=None,
                        help="Limit processing to the first N priority days "
                             "(after --start-date filtering, if given).")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=None,
                        help="Only process days of this priority tier "
                             "(1=TRO+ESR, 2=either, 3=JRO only). Default: all tiers.")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Begin processing at this date, YYYY-MM-DD (e.g. "
                             "2025-08-27), in chronological order -- overrides "
                             "the default tier-then-date priority ordering so "
                             "earlier/lower-tier days aren't skipped ahead of it.")
    parser.add_argument("--list-days", action="store_true",
                        help="Print priority-day summary and exit.")
    parser.add_argument("--status", action="store_true",
                        help="Print resume/progress status (groups completed, "
                             "filter-run outcomes, ISR metrics CSV size, and a "
                             "window x bin_count completion grid) and exit "
                             "without running anything.")
    parser.add_argument("--bin-count", type=str, default=None, metavar="COUNT",
                        help='Run only this OCC_COUNT_BINS value: an int (e.g. '
                             '30) or "all" (bin_count=None, every available RO '
                             'occultation in the window). Overrides --occ-bins.')
    parser.add_argument("--window", type=str, default=None, metavar="HHMM",
                        help='Run only this minima-window, by window_key '
                             '(e.g. "1430").')
    parser.add_argument("--occ-bins", type=str, default=None, metavar="LIST",
                        help='Comma-separated override of OCC_COUNT_BINS, e.g. '
                             '"all,30,20,10" (each token an int or "all").')
    # ── Parametric-EKF tuning overrides (see the EKF_* config block) ─────────
    parser.add_argument("--ekf-alpha", type=float, default=None, metavar="A",
                        help="Override EKF Gauss-Newton step-size damping in "
                             f"(0,1] (default {EKF_ALPHA}).")
    parser.add_argument("--ekf-sigma-obs", type=float, default=None, metavar="TECU",
                        help="Override EKF observation-noise std-dev in TECU "
                             f"(default {EKF_SIGMA_OBS}).")
    parser.add_argument("--ekf-max-rays", type=int, default=None, metavar="N",
                        help="Override EKF representative update rays per arc "
                             f"(default {EKF_MAX_RAYS}).")
    parser.add_argument("--ekf-tol", type=float, default=None, metavar="TOL",
                        help="Override EKF relative convergence tolerance "
                             f"(default {EKF_TOL}).")
    parser.add_argument("--ekf-max-iter", type=int, default=None, metavar="N",
                        help=f"Override EKF max iterations (default {EKF_MAX_ITER}).")
    parser.add_argument("--ekf-n-members", type=int, default=None, metavar="N",
                        help="Override EKF prior ensemble size factoring the "
                             f"covariance (default {EKF_N_MEMBERS}).")
    parser.add_argument("--ekf-prior-scale", type=float, default=None, metavar="S",
                        help="Multiplier on the EKF prior parameter variances; "
                             ">1 loosens, <1 tightens the background "
                             f"(default {EKF_PRIOR_SCALE}).")
    parser.add_argument("--ekf-free", type=str, default=None, metavar="SPEC",
                        help='Free-parameter set for the EKF update: "all" '
                             '(default, every param free) or a comma-separated '
                             'list to freeze the rest, e.g. "log10(NmF2)" '
                             '(tuned config: only the observable amplitude moves).')
    parser.add_argument("--ekf-adapt-alpha", action="store_true",
                        help="Enable residual-merit adaptive step size (grow "
                             "alpha on descent, damp+rollback on rise). Tuned "
                             "config; pairs with --ekf-alpha as the start value.")
    parser.add_argument("--ekf-alpha-max", type=float, default=None, metavar="A",
                        help="Cap on adaptive-alpha growth "
                             f"(default {EKF_ALPHA_MAX}; only used with "
                             "--ekf-adapt-alpha).")
    parser.add_argument("--ekf-tec-rmse-tol", type=float, default=None, metavar="TECU",
                        help="Compound convergence gate: also require the TEC-"
                             "innovation RMSE below this (TECU) before declaring "
                             "convergence (else run to --ekf-max-iter). Tuned "
                             "config ~45.")
    # ── EKF-only dense-grid experiment flags ─────────────────────────────────
    parser.add_argument("--no-gridded-kf", dest="run_gridded_kf",
                        action="store_false", default=True,
                        help="Do NOT run the gridded KF: skip its joint batch "
                             "solve (process_group skip_joint=True) and drop "
                             "gridded_kf from scoring/plots. The EKF still uses "
                             "process_group's RO data-prep (clean_list/prior).")
    parser.add_argument("--ekf-grid", type=str, default=None,
                        choices=["mesh", "fibonacci"], metavar="MODE",
                        help='EKF horizontal grid: "mesh" (RO union mesh, '
                             'default) or "fibonacci" (dense ROI-restricted '
                             "Fibonacci sphere at --ekf-grid-km spacing).")
    parser.add_argument("--ekf-grid-km", type=float, default=None, metavar="KM",
                        help="Fibonacci-grid point spacing in km (only used with "
                             f"--ekf-grid fibonacci; default {EKF_GRID_KM}).")
    parser.add_argument("--ekf-grid-max-pts", type=int, default=None, metavar="N",
                        help="Hard cap on Fibonacci grid points (EKF state = "
                             "N_STATE x n_geo, covariance is n^2). If the "
                             "footprint disk at --ekf-grid-km exceeds this, the "
                             "points closest to the footprint centroid are kept. "
                             "0 removes the cap (rely on the ROI gate + spacing "
                             f"to bound n_geo). Default {EKF_GRID_MAX_PTS}.")
    parser.add_argument("--ekf-roi-gate-km", type=float, default=None, metavar="KM",
                        help="Great-circle gate (km): drop RO tangent anchors "
                             "farther than this from the IGS/ISR footprint "
                             "centroid before they seed the Fibonacci grid, so "
                             "time-window strays can't drag the centre/radius. "
                             f"Default {EKF_ROI_GATE_KM:.0f}; 0 disables.")
    parser.add_argument("--ekf-modes", type=str, default=None, metavar="LIST",
                        help='Comma list of named EKF modes to run, each as its '
                             'own filter_type: "allfree" (all 8 params free) '
                             'and/or "nmf2" (only log10(NmF2) free). '
                             'E.g. "allfree,nmf2". Omit for a single '
                             '"parametric_ekf" run honouring --ekf-free.')
    parser.add_argument("--dates", type=str, default=None, metavar="LIST",
                        help="Comma-separated YYYY-MM-DD dates to process "
                             "(e.g. 2025-08-27,2025-09-22,2025-10-17). Selects "
                             "exactly these ISR days, overriding "
                             "--start-date/--days/--tier ordering.")
    parser.add_argument("--all-windows", dest="require_isr_truth",
                        action="store_false", default=True,
                        help="Also process availability-minima windows that have "
                             "NO co-located ISR truth profile (sorted last). "
                             "Default: only run windows with ISR truth "
                             "verification data.")
    parser.add_argument("--best-window-only", action="store_true", default=False,
                        help="Investigate ONLY the single best window per day: "
                             "the one with the most RO occultations, tie-broken "
                             "by most co-located ISR truth EDPs. The "
                             "occultation-count sweep still runs on that window.")
    parser.add_argument("--output-name", type=str, default=None, metavar="NAME",
                        help="Figure/metrics output folder name under "
                             "Figures/ISR_DA_Comparison/ (default OUTPUT). A "
                             "non-default name also routes the metrics CSV and "
                             "progress manifest to dedicated files so stats "
                             "aren't mixed with other runs.")
    parser.add_argument("--final", action="store_true", default=False,
                        help="Convenience: --best-window-only + "
                             "--output-name OUTPUT_FINAL. Investigates only the "
                             "best (most occultations + ISR truth) window per "
                             "day and writes figures/metrics/statistics to "
                             "OUTPUT_FINAL.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="Skip (window, bin, obs_mode, filter) units already complete in "
             "the progress manifest / DA_CACHE (default).")
    resume_group.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Force recompute every (window, bin, obs_mode, filter) unit, "
             "ignoring the progress manifest/DA_CACHE -- alias of --force for "
             "the occultation-count sweep axis.")
    args = parser.parse_args()

    force = args.force or not args.resume

    if args.ekf_alpha is not None:        EKF_ALPHA       = args.ekf_alpha
    if args.ekf_sigma_obs is not None:    EKF_SIGMA_OBS   = args.ekf_sigma_obs
    if args.ekf_max_rays is not None:     EKF_MAX_RAYS    = args.ekf_max_rays
    if args.ekf_tol is not None:          EKF_TOL         = args.ekf_tol
    if args.ekf_max_iter is not None:     EKF_MAX_ITER    = args.ekf_max_iter
    if args.ekf_n_members is not None:    EKF_N_MEMBERS   = args.ekf_n_members
    if args.ekf_prior_scale is not None:  EKF_PRIOR_SCALE = args.ekf_prior_scale
    if args.ekf_free is not None:         EKF_FREE_PARAMS  = _parse_ekf_free_spec(args.ekf_free)
    if args.ekf_adapt_alpha:              EKF_ADAPT_ALPHA  = True
    if args.ekf_alpha_max is not None:    EKF_ALPHA_MAX    = args.ekf_alpha_max
    if args.ekf_tec_rmse_tol is not None: EKF_TEC_RMSE_TOL = args.ekf_tec_rmse_tol
    if any(v is not None for v in (
            args.ekf_alpha, args.ekf_sigma_obs, args.ekf_max_rays, args.ekf_tol,
            args.ekf_max_iter, args.ekf_n_members, args.ekf_prior_scale,
            args.ekf_free, args.ekf_alpha_max, args.ekf_tec_rmse_tol)) \
            or args.ekf_adapt_alpha:
        print(f"[ISR-DA] EKF config: alpha={EKF_ALPHA} sigma_obs={EKF_SIGMA_OBS} "
              f"max_rays={EKF_MAX_RAYS} tol={EKF_TOL} max_iter={EKF_MAX_ITER} "
              f"n_members={EKF_N_MEMBERS} prior_scale={EKF_PRIOR_SCALE} "
              f"free={EKF_FREE_PARAMS} adapt_alpha={EKF_ADAPT_ALPHA} "
              f"alpha_max={EKF_ALPHA_MAX} tec_rmse_tol={EKF_TEC_RMSE_TOL}")

    # ── EKF-only dense-grid experiment configuration ─────────────────────────
    RUN_GRIDDED_KF = bool(args.run_gridded_kf)
    REQUIRE_ISR_TRUTH = bool(args.require_isr_truth)

    # ── Best-window-only + output-folder routing (incl. --final bundle) ──────
    BEST_WINDOW_ONLY = bool(args.best_window_only or args.final)
    _out_name = args.output_name or ("OUTPUT_FINAL" if args.final else None)
    if _out_name:
        SAVE_DIR = ROOT / "Figures" / "ISR_DA_Comparison" / _out_name
        ISR_METRICS_CSV = DA_CACHE / f"isr_metrics_{_out_name}.csv"
        PROGRESS_MANIFEST = DA_CACHE / f"progress_manifest_{_out_name}.json"
        print(f"[ISR-DA] Output routed to {SAVE_DIR} "
              f"(metrics={ISR_METRICS_CSV.name}, manifest={PROGRESS_MANIFEST.name})")
    if BEST_WINDOW_ONLY:
        print("[ISR-DA] BEST-WINDOW-ONLY: one window per day (max occultations "
              "+ ISR truth); occultation-count sweep runs on that window.")
    if args.ekf_grid is not None:         EKF_GRID_MODE    = args.ekf_grid
    if args.ekf_grid_km is not None:      EKF_GRID_KM      = args.ekf_grid_km
    if args.ekf_grid_max_pts is not None:
        EKF_GRID_MAX_PTS = None if args.ekf_grid_max_pts <= 0 else args.ekf_grid_max_pts
    if args.ekf_roi_gate_km is not None:
        EKF_ROI_GATE_KM = None if args.ekf_roi_gate_km <= 0 else args.ekf_roi_gate_km
    if args.ekf_modes is not None:
        _tokens = [t.strip().lower() for t in args.ekf_modes.split(",") if t.strip()]
        _bad = [t for t in _tokens if t not in _EKF_MODE_REGISTRY]
        if _bad:
            parser.error(f"--ekf-modes: unknown mode(s) {_bad}; "
                         f"choose from {sorted(_EKF_MODE_REGISTRY)}")
        EKF_MODES = [_EKF_MODE_REGISTRY[t] for t in _tokens]

    # Assemble the active filter set: gridded_kf (unless suppressed) + the EKF
    # filter label(s). Everything downstream (results dict, metrics CSV,
    # summarize_statistics, occ-count plots, per-group summary) iterates
    # FILTER_TYPES, so reassigning it here reconfigures the whole pipeline.
    _ekf_labels = ([lbl for lbl, _ in EKF_MODES] if EKF_MODES is not None
                   else ["parametric_ekf"])
    FILTER_TYPES = (["gridded_kf"] if RUN_GRIDDED_KF else []) + _ekf_labels
    if (not RUN_GRIDDED_KF) or (EKF_MODES is not None) or (EKF_GRID_MODE != "mesh"):
        _cap = "off" if EKF_GRID_MAX_PTS is None else str(EKF_GRID_MAX_PTS)
        _gate = "off" if EKF_ROI_GATE_KM is None else f"{EKF_ROI_GATE_KM:.0f}km"
        print(f"[ISR-DA] Experiment config: run_gridded_kf={RUN_GRIDDED_KF} "
              f"ekf_grid={EKF_GRID_MODE} ekf_grid_km={EKF_GRID_KM} "
              f"max_pts={_cap} roi_gate={_gate} filters={FILTER_TYPES}")

    if args.bin_count is not None:
        OCC_COUNT_BINS = [_parse_bin_count_value(args.bin_count)]
    elif args.occ_bins is not None:
        OCC_COUNT_BINS = [_parse_bin_count_value(tok) for tok in args.occ_bins.split(",")]

    priority_days = select_priority_days(force=force)

    if args.start_date is not None:
        start = pd.Timestamp(args.start_date).date()
        priority_days = [d for d in priority_days if d["date"] >= start]
        priority_days.sort(key=lambda e: e["date"])  # chronological, ignore tier ordering

    if args.dates is not None:
        _want = {pd.Timestamp(t.strip()).date()
                 for t in args.dates.split(",") if t.strip()}
        priority_days = [d for d in priority_days if d["date"] in _want]
        priority_days.sort(key=lambda e: e["date"])
        _missing = sorted(_want - {d["date"] for d in priority_days})
        if _missing:
            print(f"[ISR-DA] WARNING: requested --dates not found as ISR days "
                  f"(no ISR EDP cache entry): {[str(m) for m in _missing]}")

    if args.list_days:
        tier_counts = {1: 0, 2: 0, 3: 0}
        podtc_count = 0
        for e in priority_days:
            tier_counts[e["tier"]] += 1
            if e["podtc_dir"] is not None:
                podtc_count += 1

        print(f"[ISR-DA] {len(priority_days)} priority day(s) found")
        print(f"  Tier 1 (TRO+ESR): {tier_counts[1]}")
        print(f"  Tier 2 (TRO|ESR): {tier_counts[2]}")
        print(f"  Tier 3 (JRO only): {tier_counts[3]}")
        print(f"  Days with podTc2 data: {podtc_count}/{len(priority_days)}")
        return

    if args.tier is not None:
        priority_days = [d for d in priority_days if d["tier"] == args.tier]
    if args.days is not None:
        priority_days = priority_days[:args.days]

    if args.status:
        print_progress_status(priority_days, window_key_filter=args.window)
        return

    DA_CACHE.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ISR-DA] Processing {len(priority_days)} day(s)")
    for d in priority_days:
        print(f"  Tier {d['tier']}  {d['date']}  instruments={d['instruments']}  "
              f"podtc={'yes' if d['podtc_dir'] else 'no'}")

    print("\n[ISR-DA] Loading ISR EDP cache ...")
    edps = load_edps()
    n_all = len(edps)
    edps = [e for e in edps if e.get("kindat") == "6400"]
    print(f"         {n_all} profiles loaded, {len(edps)} fitted (kindat 6400) "
          f"used as ground truth")

    all_metrics: list[dict] = []

    # Wrapping the whole day/window/bin sweep in one try/except means a
    # Ctrl-C at any point -- mid-window, mid-bin, mid-filter -- lands here
    # rather than propagating a bare traceback: every unit that finished
    # before the interrupt already has its DA_CACHE pickle + progress-manifest
    # entry on disk (each is written immediately in run_all_filters /
    # _run_or_load, not batched at the end), so re-running the same command
    # resumes exactly where it left off.
    try:
        for day_info in priority_days:
            date = day_info["date"]
            print(f"\n{'=' * 70}")
            print(f"[Day] {date}  tier={day_info['tier']}  instruments={day_info['instruments']}")

            windows  = build_minima_windows_for_day(day_info["podtc_dir"], date)
            if args.window is not None:
                windows = [w for w in windows if w["window_key"] == args.window]
            igs_arcs = load_igs_for_day(pd.Timestamp(date))
            print(f"  {len(windows)} minima window(s), {len(igs_arcs)} IGS arc(s)")

            if not windows:
                # ro_only/ro_igs configs require real occultation metadata that
                # process_group cannot fabricate; skip days with no RO windows
                # rather than run only 2 of 6 filter configurations.
                note = f" matching --window {args.window}" if args.window else ""
                print(f"  [skip] No GNSS-RO minima windows for this day{note}.")
                continue

            print(f"  Running {len(windows)} window(s) x {len(OCC_COUNT_BINS)} occultation-count "
                  f"bins x {len(OBS_MODES)} obs modes x {len(FILTER_TYPES)} filters ...")
            _, day_metrics = run_all_filters(
                day_info, windows, igs_arcs, edps,
                force=force, no_plot=args.no_plot,
            )
            all_metrics.extend(day_metrics)
    except KeyboardInterrupt:
        print("\n[ISR-DA] Checkpoints saved — re-run to resume.")
        sys.exit(130)

    # Metrics from this run were already appended to ISR_METRICS_CSV
    # incrementally (per group, inside run_all_filters), so this final
    # summary always reflects everything accumulated on disk so far --
    # including from previous runs -- not just what happened this invocation
    # (which may be zero rows if every group was already complete/skipped).
    print(f"\n[done] {len(all_metrics)} new ISR comparison row(s) this run "
          f"across {len(priority_days)} day(s)")
    summarize_statistics(all_metrics)

    if not args.no_plot:
        try:
            plot_isr_convergence_vs_occ_count(ISR_METRICS_CSV, SAVE_DIR)
        except Exception:
            print("[ISR-DA] Occultation-count convergence plotting failed; continuing without it.")
            traceback.print_exc()


if __name__ == "__main__":
    main()
