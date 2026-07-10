#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
demo_isr_da_comparison.py

Compares data-assimilation filter outputs (gridded KF vs. parametric EKF)
against ISR ground-truth electron density profiles, across GNSS-RO-only,
IGS-only, and combined observation modes.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
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
)
from test_param_iono import EKF_Param
from Ionosphere_Tomography_Inverter.ionospheric_state import IonosphericState, N_STATE, PARAM_NAMES
from Ionosphere_Tomography_Inverter.observation_operator import _ne_profile_ensemble
from Ionosphere_Tomography_Inverter.enkf_update import _haversine_km
from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
from demo import _build_hourly_global_edp, extract_robust_f2_peak
from EDPSamples.edp_samples import EDPSamples, get_IRI2020_EDP

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

IGS_STATIONS_NORDIC = ["TRO1", "WUTH", "NYA1"]
OBS_MODES    = ["ro_only", "ro_igs", "igs_only"]
FILTER_TYPES = ["gridded_kf", "parametric_ekf"]
POLAR_LAT_THRESHOLD = 60.0   # matches demo_group.py

ISR_SITES              = ("ESR", "TRO")
ISR_SITE_MATCH_DEG     = 0.5
ISR_WINDOW_HALF_MINUTES = 15
ISR_ROI_MAX_KM         = 2500.0   # RO peak-tangent-point → ISR site gate (great-circle)


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


def print_progress_status() -> None:
    """
    Print a human-readable summary of how much of the analysis has already
    been completed, from the progress manifest and the accumulated ISR
    metrics CSV -- without running anything. Used by `--status`.
    """
    manifest = _load_progress_manifest()
    print(f"[ISR-DA][status] Progress manifest: {PROGRESS_MANIFEST}")
    if not manifest:
        print("  No groups recorded as complete yet.")
    else:
        dates = sorted({e.get("date") for e in manifest.values() if e.get("date")})
        date_range = f"{dates[0]} .. {dates[-1]}" if dates else "unknown"
        print(f"  {len(manifest)} group(s) marked complete, spanning {len(dates)} "
              f"day(s) ({date_range})")

        from collections import Counter
        status_counts: Counter = Counter()
        for entry in manifest.values():
            for obs_mode, per_filter in entry.get("obs_mode_status", {}).items():
                for filter_type, status in per_filter.items():
                    status_counts[(obs_mode, filter_type, status)] += 1
        print("  Filter-run outcomes across completed groups:")
        for (obs_mode, filter_type, status), n in sorted(status_counts.items()):
            print(f"    {obs_mode:<9} {filter_type:<14} {status:<45} : {n}")

    if ISR_METRICS_CSV.exists():
        df = pd.read_csv(ISR_METRICS_CSV, parse_dates=["t_centre"])
        print(f"\n  ISR metrics CSV: {ISR_METRICS_CSV}")
        print(f"    {len(df)} row(s), {df['group_key'].nunique()} group(s), "
              f"date range {df['date'].min()} .. {df['date'].max()}")
    else:
        print(f"\n  No ISR metrics CSV yet at {ISR_METRICS_CSV}")


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
    "arc_time_sec", "time_s", "time_utc_h",
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
                        n_mc: int = 50) -> "SimpleNamespace":
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
    """
    clean_window = result["clean_window"]
    eds_occ      = _build_igs_eds_occ(t_centre, grid_lats, grid_lons, alt_grid)

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
        alpha=alpha, tol=tol, max_iter=max_iter, jacobian_analytical = True,
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


def run_all_filters(day_info: dict, ro_groups: list, igs_arcs: list,
                     edps: list[dict], force: bool = False,
                     no_plot: bool = False) -> tuple[dict, list[dict]]:
    """
    Run gridded-KF and parametric-EKF filters across all observation modes
    (ro_only, ro_igs, igs_only) for every 30-min RO group on one ISR day.

    Results are cached to DA_CACHE as
    "{group_key}_{obs_mode}_{filter_type}.pkl" and pickled immediately after
    each is computed, so a re-run with force=False resumes where it left off.

    Metrics and figures for each group are produced right after that group's
    6 filter runs finish (rather than waiting for every group in the day to
    finish), so plots start appearing incrementally as the day progresses.

    Returns
    -------
    (results, metrics) where results is keyed by
    (group_key, obs_mode, filter_type) -> result dict or None, and metrics
    is the flat list of per-group ISR comparison rows (see compute_isr_metrics).
    """
    DA_CACHE.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    all_metrics: list[dict] = []
    global_edp_cache: dict = {}
    progress_manifest = _load_progress_manifest()

    _tro = INSTRUMENTS["TRO"]
    _igs_region = assign_region(_tro["lat"], _tro["lon"])

    def _cache_path(group_key: str, obs_mode: str, filter_type: str) -> Path:
        return DA_CACHE / f"{group_key}_{obs_mode}_{filter_type}.pkl"

    def _result_status(value) -> str:
        if value is None:
            return "None"
        if isinstance(value, dict):
            return str(value.get("status", "ok (no 'status' key)"))
        return type(value).__name__

    def _run_or_load(group_key: str, obs_mode: str, filter_type: str, compute_fn):
        path = _cache_path(group_key, obs_mode, filter_type)
        tag = f"{group_key} | {obs_mode:<9} | {filter_type:<14}"
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
        results[(group_key, obs_mode, filter_type)] = value
        return value

    def _process_group_all_filters(group_key, group_meta):
        manifest_entry = progress_manifest.get(group_key)
        if manifest_entry is not None and manifest_entry.get("status") == "complete" and not force:
            print(f"  [resume] {group_key} already marked complete in the progress "
                  f"manifest (metrics+plots written on a previous run) -- skipping. "
                  f"Use --force to redo it.")
            return

        t_centre = _parse_time_window(group_key)
        win_key  = group_meta["time_window"].iloc[0]

        igs_window_arcs = _filter_igs_cmp(igs_arcs, win_key, window_minutes=30)
        # Use a single raypath per arc (its central-time epoch) instead of the
        # full ~30-minute arc as multiple observations spread across the
        # window -- see _collapse_igs_arc_to_central_epoch() docstring.
        igs_window_arcs = [_collapse_igs_arc_to_central_epoch(a) for a in igs_window_arcs]

        if t_centre.hour not in global_edp_cache:
            global_edp_cache.update(_build_hour_edp_cache(t_centre))

        # ── Per-window/per-obs-mode figure output directories ───────────────
        safe_key   = _safe_group_key(group_key)
        group_dirs = {mode: SAVE_DIR / safe_key / mode for mode in OBS_MODES}
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
        _igs_tbbox = _tight_bbox_from_points(_roi_lats, _roi_lons, margin_deg=100.0 / 111.32)
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
        res_ro = _run_or_load(group_key, "ro_only", "gridded_kf", lambda: process_group(
            group_key, group_meta, ALT_GRID,
            global_edp_cache=global_edp_cache,
            run_sequential=False, save_dir=str(group_dirs["ro_only"]),
            podtc_max_rays=200, extra_clean_list=None,
            roi_extra_points=igs_roi_points,
            filter_label="kf",
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
        res_roigs = _run_or_load(group_key, "ro_igs", "gridded_kf", lambda: process_group(
            group_key, group_meta, ALT_GRID,
            global_edp_cache=global_edp_cache,
            run_sequential=False, save_dir=str(group_dirs["ro_igs"]),
            podtc_max_rays=200, extra_clean_list=igs_window_arcs,
            roi_extra_points=igs_roi_points,
            filter_label="kf",
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

        res_igs = _run_or_load(group_key, "igs_only", "gridded_kf", _run_igs_only_kf)

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

        for obs_mode, kf_result in kf_results.items():
            if obs_mode == "igs_only":
                def _run_igs_ekf(kf_result=kf_result, t_centre=t_centre,
                                  group_key=group_key):
                    if kf_result is None:
                        print(f"  [diag] {group_key} | igs_only  | parametric_ekf : "
                              f"SKIPPED (upstream igs_only gridded_kf result is None)")
                        return None
                    clean_window = kf_result["clean_window"]
                    eds_occ = _build_igs_eds_occ(
                        t_centre, _igs_grid_lats, _igs_grid_lons, ALT_GRID,
                    )
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
                    return _run_parametric_ekf(
                        res_kf=res_kf_adapted, alt_grid=ALT_GRID,
                        save_dir=str(group_dirs["igs_only"]),
                        group_key=f"{group_key}_igs_only", n_members=200,
                        sigma_obs=10.0, max_update_rays=100,
                    )

                _run_or_load(group_key, obs_mode, "parametric_ekf", _run_igs_ekf)
                continue

            def _run_ekf(kf_result=kf_result, obs_mode=obs_mode):
                if kf_result is None or kf_result.get("status") != "Success":
                    reason = "upstream result is None" if kf_result is None else \
                        f"upstream gridded_kf status={kf_result.get('status')!r} (need 'Success')"
                    print(f"  [diag] {group_key} | {obs_mode:<9} | parametric_ekf : "
                          f"SKIPPED ({reason})")
                    return None
                return _run_parametric_ekf(
                    res_kf=kf_result, alt_grid=ALT_GRID,
                    save_dir=str(group_dirs[obs_mode]),
                    group_key=f"{group_key}_{obs_mode}", n_members=200,
                    sigma_obs=10.0, max_update_rays=100,
                )
            _run_or_load(group_key, obs_mode, "parametric_ekf", _run_ekf)

        # ── Per-group summary: which of the 6 (obs_mode, filter_type) combos
        #    actually produced a usable result vs. were skipped/failed ────────
        print(f"  [diag] {group_key} : summary of 6 obs_mode/filter_type combos")
        _missing = object()
        for obs_mode in OBS_MODES:
            for filter_type in FILTER_TYPES:
                value = results.get((group_key, obs_mode, filter_type), _missing)
                status = "MISSING (never ran)" if value is _missing else _result_status(value)
                print(f"           {obs_mode:<9} | {filter_type:<14} -> {status}")

        # ── Metrics + figures for this group, right away ─────────────────────
        # Previously this ran in a second pass over all groups in main(), only
        # after every group in the day had finished its 6 filter runs -- so no
        # plots appeared until hours into a day's processing. Doing it here
        # means each group's plots land as soon as that group is done.
        group_filter_results = {
            obs_mode: {
                filter_type: results.get((group_key, obs_mode, filter_type))
                for filter_type in FILTER_TYPES
            }
            for obs_mode in OBS_MODES
        }

        group_day_info = dict(
            day_info, group_key=group_key,
            n_ro_occultations=len(group_meta),
            n_igs_arcs=len(igs_window_arcs),
        )
        group_metrics = compute_isr_metrics(group_day_info, group_filter_results, edps)
        all_metrics.extend(group_metrics)
        # Persist immediately rather than waiting for the whole run to finish
        # (main() previously only wrote the CSV once, after every day had
        # completed -- a crash partway through lost every already-computed
        # group's metrics). Safe to call repeatedly: dedup keeps the latest
        # row per (group_key, obs_mode, filter_type, instrument).
        _append_metrics_csv(group_metrics)

        def _mark_group_complete(plots_written: bool) -> None:
            progress_manifest[group_key] = {
                "date":              str(day_info.get("date")),
                "group_key":         group_key,
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

        _plot_group_all_modes(
            group_key, group_filter_results, igs_window_arcs,
            window_isr_profiles, SAVE_DIR, window_edps,
        )

        if window_edps:
            solar = get_solar_conditions(t_centre)
            group_save_dir = SAVE_DIR / _safe_group_key(group_key)
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

        _mark_group_complete(plots_written=True)

    # ── Run ISR-aligned groups first, most occultations first ────────────────
    # A group whose time window has no co-located ISR profile can never
    # produce an ISR comparison metric/plot (see compute_isr_metrics /
    # window_edps above), so running those last means the runs that actually
    # yield useful output land first instead of waiting behind a full day of
    # ISR-blind groups.
    def _group_priority(item: tuple) -> tuple:
        group_key, group_meta = item
        n_occ = len(group_meta)
        try:
            t_centre = _parse_time_window(group_key)
            has_isr = bool(_isr_profiles_for_window(edps, t_centre))
        except ValueError:
            has_isr = False
        return (0 if has_isr else 1, -n_occ)

    keyed_groups = sorted(((_group_priority(g), g) for g in ro_groups), key=lambda p: p[0])
    ro_groups = [item for _, item in keyed_groups]
    n_isr_aligned = sum(1 for key, _ in keyed_groups if key[0] == 0)
    print(f"  [diag] Reordered {len(ro_groups)} group(s): "
          f"{n_isr_aligned} ISR-aligned (most occultations first), "
          f"{len(ro_groups) - n_isr_aligned} without ISR truth (run last)")

    # Each group is processed in its own try/except so an unexpected exception
    # in one group (ROI setup, adaptation code, etc. -- anything outside the
    # per-filter try/except in _run_or_load) can't discard already-cached
    # results/plots for other groups in the same day, which previously
    # propagated all the way out of run_all_filters() and skipped the entire
    # day's plotting pass in main(), even for groups that had fully succeeded.
    for group_key, group_meta in ro_groups:
        try:
            _process_group_all_filters(group_key, group_meta)
        except Exception:
            print(f"  [error] group {group_key} failed with an unexpected "
                  f"exception; skipping to the next group so already-computed "
                  f"results can still be plotted.")
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
    """
    rows: list[dict] = []
    date       = day_info.get("date")
    group_key  = day_info.get("group_key")
    print(f"\n▶▶▶ compute_isr_metrics START | group={group_key}")

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

                prior_at_isr = np.interp(isr_alt, alt_grid, prior_col)
                post_at_isr  = np.interp(isr_alt, alt_grid, post_col)

                valid = (isr_ne > 1e8) & np.isfinite(isr_ne)
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

                rows.append({
                    "date":                date,
                    "group_key":           group_key,
                    "obs_mode":            obs_mode,
                    "filter_type":         filter_type,
                    "instrument":          inst_name,
                    "t_centre":            t_centre,
                    "region":              region,
                    "n_ro_occultations":   n_ro_occultations,
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
                    "n_isr_gates_valid":   int(valid.sum()),
                    "ekf_converged":       ekf_converged,
                    "ekf_n_iterations":    ekf_n_iterations,
                })

    print(f"◀◀◀ compute_isr_metrics END   | group={group_key} -> {len(rows)} metric row(s)")
    return rows


_METRICS_DEDUP_COLS = ["group_key", "obs_mode", "filter_type", "instrument"]


def _append_metrics_csv(metrics_rows: list[dict]) -> pd.DataFrame:
    """
    Append *metrics_rows* to the on-disk ISR-metrics CSV (ISR_METRICS_CSV)
    immediately, deduplicated on group_key+obs_mode+filter_type+instrument
    (keeping the latest row for any repeat), and return the full accumulated
    DataFrame.

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

    summary_text = "\n".join(lines)
    print(summary_text)

    stats_path = SAVE_DIR / "statistics_summary.txt"
    stats_path.write_text(summary_text + "\n")

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




# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare KF/EKF data assimilation against ISR ground truth.")
    parser.add_argument("--force", action="store_true",
                        help="Ignore caches and recompute everything.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip figure generation.")
    parser.add_argument("--days", type=int, default=None,
                        help="Limit processing to the first N priority days.")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=None,
                        help="Only process days of this priority tier "
                             "(1=TRO+ESR, 2=either, 3=JRO only). Default: all tiers.")
    parser.add_argument("--list-days", action="store_true",
                        help="Print priority-day summary and exit.")
    parser.add_argument("--status", action="store_true",
                        help="Print resume/progress status (groups completed, "
                             "filter-run outcomes, ISR metrics CSV size) and exit "
                             "without running anything.")
    args = parser.parse_args()

    if args.status:
        print_progress_status()
        return

    priority_days = select_priority_days(force=args.force)

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

    for day_info in priority_days:
        date = day_info["date"]
        print(f"\n{'=' * 70}")
        print(f"[Day] {date}  tier={day_info['tier']}  instruments={day_info['instruments']}")

        ro_groups = load_ro_group_for_day(day_info["podtc_dir"])
        igs_arcs  = load_igs_for_day(pd.Timestamp(date))
        print(f"  {len(ro_groups)} POLAR_N group(s), {len(igs_arcs)} IGS arc(s)")

        if not ro_groups:
            # ro_only/ro_igs configs require real occultation metadata that
            # process_group cannot fabricate; skip days with no RO groups
            # rather than run only 2 of 6 filter configurations.
            print("  [skip] No GNSS-RO groups for this day.")
            continue

        print(f"  Running {len(ro_groups)} group(s) x {len(OBS_MODES)} obs modes "
              f"x {len(FILTER_TYPES)} filters ...")
        _, day_metrics = run_all_filters(
            day_info, ro_groups, igs_arcs, edps,
            force=args.force, no_plot=args.no_plot,
        )
        all_metrics.extend(day_metrics)

    # Metrics from this run were already appended to ISR_METRICS_CSV
    # incrementally (per group, inside run_all_filters), so this final
    # summary always reflects everything accumulated on disk so far --
    # including from previous runs -- not just what happened this invocation
    # (which may be zero rows if every group was already complete/skipped).
    print(f"\n[done] {len(all_metrics)} new ISR comparison row(s) this run "
          f"across {len(priority_days)} day(s)")
    summarize_statistics(all_metrics)


if __name__ == "__main__":
    main()
