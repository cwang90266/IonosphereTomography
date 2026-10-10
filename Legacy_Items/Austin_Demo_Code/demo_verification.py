#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
demo_verification.py — ISR-focused verification of the grouped KF retrieval.

Restricts processing to a 30°-lat × 60°-lon region centred on the
Millstone Hill ISR (42.62°N, 288.51°E / 71.49°W), runs the same joint
Kalman-Filter assimilation as demo_group.py, then compares the retrieved
EDP against ground-truth profiles measured by the ISR.

ISR comparison figures:
  • Profile overlay  — prior / posterior / ISR Ne profiles at the mesh
                       vertex nearest to Millstone Hill, colour-coded by
                       UTC hour of the ISR sweep.
  • Scatter plot     — NmF2 and hmF2 from the retrieval vs. ISR truth.
  • Bias / RMSE vs. altitude — profile-by-profile absolute error.

All other diagnostic plots (trim, TEC panels, sequential-step centre plot)
are inherited unchanged from demo_group.py.

Run from the project root:
    python demo_verification.py
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
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import time
import gc
import datetime

import netCDF4
from scipy.optimize import curve_fit
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj

from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
from EDPSamples.edp_samples import EDPSamples
from Ionosphere_Tomography_Inverter.Ionophy_Tomography_Inverter import (
    Ionosphere_Tomography_Inverter,
)
from demo import build_daily_global_edps, extract_robust_f2_peak

# Re-use helpers from demo_group
import demo_group as _demo_group
from demo_group import (
    scan_metadata,
    process_group,
    _make_tec_slices,
    _ECEF_TO_LL,
    WINDOW_MINUTES,
    MAX_MESH_VERTICES,
    CONSTELLATION_CONFIG,
    _CONST_FALLBACK_CMAP,
    _save_stats_csv,
    GAUSSIAN_COV_SIGMA,  # noqa: F401 — re-exported so callers see the active value
)

# ─────────────────────────────────────────────────────────────────────────────
# Millstone Hill ISR location and verification region
# ─────────────────────────────────────────────────────────────────────────────

ISR_LAT   =  42.62          # °N
ISR_LON   = 288.51          # °E  (= -71.49°W)
ISR_LON_W = ISR_LON - 360.0 # °E in -180…+180 convention

HALF_LAT  = 15.0            # ±15° → 30° tall
HALF_LON  = 30.0            # ±30° → 60° wide

VERIF_LAT_MIN = ISR_LAT - HALF_LAT   #  27.62°N
VERIF_LAT_MAX = ISR_LAT + HALF_LAT   #  57.62°N
VERIF_LON_MIN = ISR_LON_W - HALF_LON # -101.49°E
VERIF_LON_MAX = ISR_LON_W + HALF_LON #  -41.49°E


# ─────────────────────────────────────────────────────────────────────────────
# Patch demo_group.region_bounding_box to handle "VERIF_MH".
# process_group, _group_bounding_box, _roi_centre_idx, and _draw_roi_boundary
# all call the version bound inside demo_group's own module namespace, so we
# must replace it there rather than just defining a local override.
# ─────────────────────────────────────────────────────────────────────────────

_orig_region_bounding_box = _demo_group.region_bounding_box


def _patched_region_bounding_box(region_key: str):
    if region_key == "VERIF_MH":
        return (VERIF_LAT_MIN, VERIF_LAT_MAX, VERIF_LON_MIN, VERIF_LON_MAX)
    return _orig_region_bounding_box(region_key)


_demo_group.region_bounding_box = _patched_region_bounding_box


# ── _plot_group patch: inject ISR truth into joint plots ─────────────────────
# _isr_profiles_for_patch is populated in demo_verification_main() after loading
# ISR data, before processing begins.  The patch selects sweeps that fall within
# the group's 30-minute time window and passes them to _plot_group.
_isr_profiles_for_patch: list[dict] = []

_orig_plot_group = _demo_group._plot_group


def _patched_plot_group(
    result, save_dir, group_key, *, suffix="", mode_label="Sequential KF", **kwargs
):
    isr_arg = None
    if suffix == "_joint" and _isr_profiles_for_patch:
        win = result.get("time_window", "")
        try:
            hhmm  = win.split("_")[-1]
            h_mid = int(hhmm[:2]) + int(hhmm[2:]) / 60.0
            half  = WINDOW_MINUTES / 120.0   # half-window in hours
            isr_arg = [
                p for p in _isr_profiles_for_patch
                if abs(p["hour_utc"] - h_mid) < half
            ]
            if not isr_arg:
                isr_arg = [min(
                    _isr_profiles_for_patch,
                    key=lambda p: min(
                        abs(p["hour_utc"] - h_mid),
                        24 - abs(p["hour_utc"] - h_mid),
                    ),
                )]
        except Exception:
            isr_arg = _isr_profiles_for_patch[:1]
    kwargs.setdefault("isr_profiles", isr_arg)
    kwargs.setdefault("isr_site", (ISR_LON_W, ISR_LAT) if isr_arg else None)
    return _orig_plot_group(
        result, save_dir, group_key,
        suffix=suffix, mode_label=mode_label,
        **kwargs,
    )


_demo_group._plot_group = _patched_plot_group


# ─────────────────────────────────────────────────────────────────────────────
# §A  Filter occultations to the Millstone Hill verification region
# ─────────────────────────────────────────────────────────────────────────────

def filter_to_verif_region(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only occultations whose TEC-max tangent point falls within the
    30°-lat × 60°-lon box centred on Millstone Hill.

    Longitudes in the metadata are in −180…+180; VERIF_LON_* are also in
    that convention.
    """
    lat_ok = (meta["lat"] >= VERIF_LAT_MIN) & (meta["lat"] <= VERIF_LAT_MAX)
    lon_ok = (meta["lon"] >= VERIF_LON_MIN) & (meta["lon"] <= VERIF_LON_MAX)
    filtered = meta[lat_ok & lon_ok].copy()
    print(
        f"  Verification region: lat [{VERIF_LAT_MIN:.1f}, {VERIF_LAT_MAX:.1f}]  "
        f"lon [{VERIF_LON_MIN:.1f}, {VERIF_LON_MAX:.1f}]\n"
        f"  Occultations in region: {len(filtered)} / {len(meta)}"
    )
    return filtered.reset_index(drop=True)


def time_window_key(dt: pd.Timestamp, window_minutes: int = WINDOW_MINUTES) -> str:
    total_min = dt.hour * 60 + dt.minute
    floored   = (total_min // window_minutes) * window_minutes
    h, m      = divmod(floored, 60)
    return f"{dt.strftime('%Y-%m-%d')}_{h:02d}{m:02d}"


def assign_orbit_groups(
    meta: pd.DataFrame,
    gap_minutes: float = 20.0,
) -> pd.DataFrame:
    """
    Group occultations by LEO orbital pass over the verification region.

    Within each LEO spacecraft, consecutive occultations separated by less
    than `gap_minutes` are placed on the same orbit group.  Groups from
    different spacecraft that overlap in time are merged into a single KF
    group so they inform the same posterior EDP field.

    The `time_window` for each group is the mean UTC time of its members,
    formatted as "YYYY-MM-DD_HHMM".  `process_group` already derives the IRI
    background-state hour from the median observation timestamp, so this
    mean-time label is used only for display and file-naming; the actual EDP
    selection uses the real timestamps loaded from each file.

    Parameters
    ----------
    gap_minutes : float
        Maximum gap (minutes) between consecutive occultations from the same
        spacecraft that are still considered part of the same orbital pass.
    """
    meta = meta.copy().sort_values("date").reset_index(drop=True)

    # ── Step 1: label each row with a per-spacecraft orbit index ─────────────
    meta["_orbit_idx"] = -1
    orbit_counter = 0
    for sc, grp in meta.groupby("spacecraft", sort=False):
        grp = grp.sort_values("date")
        idxs = grp.index.tolist()
        dates = grp["date"].tolist()
        current_orbit = orbit_counter
        meta.at[idxs[0], "_orbit_idx"] = current_orbit
        for k in range(1, len(idxs)):
            gap = (dates[k] - dates[k - 1]).total_seconds() / 60.0
            if gap > gap_minutes:
                orbit_counter += 1
                current_orbit = orbit_counter
            meta.at[idxs[k], "_orbit_idx"] = current_orbit
        orbit_counter += 1

    # ── Step 2: merge spacecraft groups that overlap in time ──────────────────
    # Build per-orbit-index time intervals, then union overlapping ones.
    orbit_intervals: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for oi, grp in meta.groupby("_orbit_idx"):
        orbit_intervals[oi] = (grp["date"].min(), grp["date"].max())

    # Sort orbit indices by start time and greedily merge overlapping windows
    sorted_orbits = sorted(orbit_intervals.keys(), key=lambda o: orbit_intervals[o][0])
    merged: list[tuple[list[int], pd.Timestamp, pd.Timestamp]] = []
    for oi in sorted_orbits:
        t0, t1 = orbit_intervals[oi]
        if merged:
            prev_ids, prev_t0, prev_t1 = merged[-1]
            gap_to_prev = (t0 - prev_t1).total_seconds() / 60.0
            if gap_to_prev <= gap_minutes:
                merged[-1] = (prev_ids + [oi], prev_t0, max(t1, prev_t1))
                continue
        merged.append(([oi], t0, t1))

    # Assign a final merged group index to each row
    orbit_to_merged: dict[int, int] = {}
    for mg_idx, (orbit_ids, _, _) in enumerate(merged):
        for oi in orbit_ids:
            orbit_to_merged[oi] = mg_idx

    meta["_merged_idx"] = meta["_orbit_idx"].map(orbit_to_merged)
    meta.drop(columns=["_orbit_idx"], inplace=True)

    # ── Step 3: compute mean time per merged group and build keys ─────────────
    date_str = meta["date"].iloc[0].strftime("%Y-%m-%d")

    def _group_key(grp_df: pd.DataFrame) -> pd.Series:
        mg_idx   = grp_df.name   # the groupby key value, always available
        mean_ts  = pd.Timestamp(int(grp_df["date"].apply(lambda x: x.value).mean()))
        win_key  = mean_ts.strftime("%Y-%m-%d_%H%M")
        gk       = f"{win_key}__orbit{mg_idx:02d}__VERIF_MH"
        return pd.Series({
            "time_window": win_key,
            "group_key":   gk,
            "region":      "VERIF_MH",
        })

    keys_df = meta.groupby("_merged_idx", sort=True).apply(_group_key)
    meta.drop(columns=["time_window", "group_key", "region"], inplace=True)
    meta = meta.join(keys_df[["time_window", "group_key", "region"]],
                     on="_merged_idx")
    meta.drop(columns=["_merged_idx"], inplace=True)

    n_groups = meta["group_key"].nunique()
    print(
        f"  Orbit grouping (gap={gap_minutes:.0f} min): "
        f"{len(meta)} occultations → {n_groups} orbit group(s)"
    )
    for gk, sub in meta.groupby("group_key", sort=True):
        t0 = sub["date"].min().strftime("%H:%M")
        t1 = sub["date"].max().strftime("%H:%M")
        print(f"    {gk}  ({len(sub)} occ, {t0}–{t1} UTC)")

    return meta.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# §B  Load ISR electron density data from Millstone Hill netCDF files
# ─────────────────────────────────────────────────────────────────────────────

def load_isr_profiles(isr_files: list[str]) -> list[dict]:
    """
    Load ISR electron density profiles from a list of Millstone Hill netCDF
    files (same format as used by run_ISR_data.py).

    Returns a list of dicts, one per time sweep, each containing:
        'hour_utc' : float  — UTC hour of the sweep
        'alt_km'   : ndarray — altitude grid (km)
        'ne'       : ndarray — electron density (m⁻³)
        'nm_f2'    : float  — peak Ne (m⁻³)
        'hm_f2'    : float  — peak altitude (km)
    Only sweeps with ≥ 10 valid range gates are included.
    """
    profiles = []
    for fpath in isr_files:
        if not os.path.exists(fpath):
            print(f"  [ISR] File not found, skipping: {fpath}")
            continue
        try:
            ds   = netCDF4.Dataset(fpath, "r")
            alt  = np.array(ds.variables["gdalt"][:])           # (n_alt,) km
            nel  = np.ma.filled(
                np.array(ds.variables["nel"][:]), np.nan
            )                                                    # log10(Ne), (n_t, n_alt) or (n_alt, n_t)
            times = np.array(ds.variables["timestamps"][:])     # Unix seconds
            ds.close()

            # Ensure nel is (n_time, n_alt)
            if nel.ndim == 2:
                if nel.shape[1] != alt.shape[0]:
                    nel = nel.T

            n_time = nel.shape[0]
            for i in range(n_time):
                row = nel[i, :]
                valid = ~np.isnan(row)
                if valid.sum() < 10:
                    continue
                alt_v = alt[valid]
                ne_v  = 10.0 ** row[valid]          # m⁻³
                t_sec = float(times[i]) if times.ndim == 1 else float(np.nanmean(times[i, :]))
                hour_utc = (t_sec % 86400) / 3600.0
                nm, hm = extract_robust_f2_peak(ne_v, alt_v)
                profiles.append({
                    "hour_utc": hour_utc,
                    "alt_km":   alt_v,
                    "ne":       ne_v,
                    "nm_f2":    nm,
                    "hm_f2":    hm,
                    "tec_tecu": compute_isr_tec(alt_v, ne_v),
                })
        except Exception as exc:
            print(f"  [ISR] Error reading {fpath}: {exc}")

    print(f"  Loaded {len(profiles)} ISR sweeps from {len(isr_files)} file(s).")
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# §C  Find the mesh vertex nearest to Millstone Hill
# ─────────────────────────────────────────────────────────────────────────────

def millstone_vertex_idx(verts_geo: np.ndarray) -> int:
    """
    Return the index of the EDP mesh vertex nearest to Millstone Hill.

    verts_geo : (n_geo, 2)  col-0 = longitude (°E, −180…180), col-1 = latitude
    """
    dlat = verts_geo[:, 1] - ISR_LAT
    dlon = verts_geo[:, 0] - ISR_LON_W
    return int(np.argmin(dlat ** 2 + dlon ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# §D  Interpolate a retrieval EDP onto the ISR altitude grid
# ─────────────────────────────────────────────────────────────────────────────

def interp_to_isr_alt(ne_ret: np.ndarray, alt_ret: np.ndarray,
                      alt_isr: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate a retrieval profile (ne_ret on alt_ret) onto the
    ISR altitude grid alt_isr.  Points outside the retrieval range are NaN.
    """
    return np.interp(alt_isr, alt_ret, ne_ret,
                     left=np.nan, right=np.nan)


def compute_isr_tec(
    alt_km: np.ndarray,
    ne: np.ndarray,
    topside_scale_height_m: float = 150000.0,
    topside_H_H_m: float = 1000000.0,
    topside_alpha: float = 0.05,
) -> float:
    """
    Compute vertical TEC from an ISR EDP using the same topside extension
    as the point-geometry case in EDPSamples/Ionophy_Tomography_Inverter.

    Numerically integrates Ne from the ISR profile's minimum altitude to its
    maximum altitude (trapezoid rule), then appends the analytic exponential
    tail above:  N_e(h) = (ne_top / H_eff) * exp(-(h - h_top) / H_eff)
    whose vertical integral is simply  ne_top * H_eff / 1e16  TECU.

    The effective topside scale height matches the inverter default:
        H_eff = (1 - alpha) * H_scale + alpha * H_H

    Returns
    -------
    float : vertical TEC in TECU, or np.nan if the profile is too short.
    """
    order  = np.argsort(alt_km)
    alt    = alt_km[order]
    ne_v   = ne[order]

    valid  = np.isfinite(ne_v)
    alt    = alt[valid]
    ne_v   = ne_v[valid]

    if len(alt) < 2:
        return np.nan

    # Numeric integral over the ISR altitude range (km → m conversion)
    dl_m      = np.diff(alt) * 1000.0
    ne_mid    = (ne_v[:-1] + ne_v[1:]) / 2.0
    vtec_grid = float(np.sum(ne_mid * dl_m)) / 1e16  # TECU

    # Analytic exponential tail above the top of the ISR profile
    H_eff_m   = (1.0 - topside_alpha) * topside_scale_height_m + topside_alpha * topside_H_H_m
    ne_top    = float(ne_v[-1])
    vtec_top  = ne_top * H_eff_m / 1e16              # TECU

    return vtec_grid + vtec_top


# ─────────────────────────────────────────────────────────────────────────────
# §E  Verification plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_isr_profile_comparison(
    isr_profiles:    list[dict],
    prior_edp_at_mh: np.ndarray,
    post_edp_at_mh:  np.ndarray,
    alt_grid:        np.ndarray,
    group_key:       str,
    save_dir:        str,
) -> str:
    """
    Three-panel figure comparing the retrieval to ISR truth at the Millstone
    Hill grid vertex.

    Panel 1 (left)   — EDP profiles: prior (black dashed), posterior (blue),
                        each ISR sweep coloured by UTC hour.
    Panel 2 (centre) — NmF2 scatter: retrieval prior & posterior vs. ISR.
    Panel 3 (right)  — Absolute bias |Ne_ret − Ne_ISR| vs. altitude, averaged
                        across all ISR sweeps that overlap the retrieval.
    """
    os.makedirs(save_dir, exist_ok=True)

    ne_fmt   = ScalarFormatter(useMathText=True)
    ne_fmt.set_powerlimits((-2, 2))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    ax_prof, ax_nm, ax_bias = axes
    fig.suptitle(
        f"Millstone Hill ISR Verification\n{group_key}",
        fontsize=13, y=1.01,
    )

    # ── Panel 1: EDP profile overlay ─────────────────────────────────────────
    ISR_COLOR = "mediumseagreen"
    for prof in isr_profiles:
        time_str = time.strftime('%H:%M', time.gmtime(prof["hour_utc"] * 3600))
        ax_prof.plot(prof["ne"], prof["alt_km"], color=ISR_COLOR, lw=1.2, alpha=1,
                     label=f'MH ISR: {time_str} UTC')


    ax_prof.plot(prior_edp_at_mh, alt_grid,
                 color="black", lw=2.2, ls="--", label="Prior (IRI)", zorder=5)
    ax_prof.plot(post_edp_at_mh,  alt_grid,
                 color="royalblue", lw=2.5, ls="-", label="Posterior (KF)", zorder=6)

    # Peak markers for prior and posterior
    nm_pr, hm_pr = extract_robust_f2_peak(prior_edp_at_mh, alt_grid)
    nm_po, hm_po = extract_robust_f2_peak(post_edp_at_mh,  alt_grid)
    if not np.isnan(nm_pr):
        ax_prof.plot(nm_pr, hm_pr, marker="D", ms=9,
                     color="black", mec="black", zorder=7)
    if not np.isnan(nm_po):
        ax_prof.plot(nm_po, hm_po, marker="D", ms=9,
                     color="royalblue", mec="black", zorder=7)

    ax_prof.set_xlabel("Electron Density (m⁻³)", fontsize=10)
    ax_prof.set_ylabel("Altitude (km)", fontsize=10)
    ax_prof.set_title("Ne Profiles at Millstone Hill Vertex", fontsize=10)
    ax_prof.xaxis.set_major_formatter(ne_fmt)
    ax_prof.grid(True, alpha=0.3, ls=":")
    ax_prof.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax_prof.set_ylim(bottom=0)

    # ── Panel 2: NmF2 scatter ─────────────────────────────────────────────────
    isr_nm = np.array([p["nm_f2"] for p in isr_profiles if not np.isnan(p["nm_f2"])])
    isr_hm = np.array([p["hm_f2"] for p in isr_profiles if not np.isnan(p["hm_f2"])])

    if len(isr_nm) > 0:
        nm_mean = float(np.nanmean(isr_nm))
        hm_mean = float(np.nanmean(isr_hm))

        # ISR truth cloud
        ax_nm.scatter(isr_nm, isr_hm,
                      color=ISR_COLOR,
                      s=40, edgecolors="gray", linewidths=0.4,
                      zorder=4, label="ISR sweeps", alpha=0.8)

        # ISR mean
        ax_nm.axvline(nm_mean, color="gray", lw=1.0, ls=":", alpha=0.7)
        ax_nm.axhline(hm_mean, color="gray", lw=1.0, ls=":", alpha=0.7)
        ax_nm.plot(nm_mean, hm_mean, marker="s", ms=11,
                   color="gray", mec="black", mew=1.0, zorder=6,
                   label=f"ISR mean\n({nm_mean:.2e} m⁻³, {hm_mean:.0f} km)")

    # Prior and posterior peaks
    if not np.isnan(nm_pr):
        ax_nm.plot(nm_pr, hm_pr, marker="D", ms=12,
                   color="black", mec="black", zorder=7,
                   label=f"Prior\n({nm_pr:.2e} m⁻³, {hm_pr:.0f} km)")
    if not np.isnan(nm_po):
        ax_nm.plot(nm_po, hm_po, marker="*", ms=16,
                   color="royalblue", mec="black", zorder=8,
                   label=f"Posterior\n({nm_po:.2e} m⁻³, {hm_po:.0f} km)")

    ax_nm.set_xlabel("NmF2 (m⁻³)", fontsize=10)
    ax_nm.set_ylabel("hmF2 (km)",   fontsize=10)
    ax_nm.set_title("F2-Peak: Retrieval vs. ISR", fontsize=10)
    ax_nm.xaxis.set_major_formatter(ne_fmt)
    ax_nm.grid(True, alpha=0.3, ls=":")
    ax_nm.legend(fontsize=8, loc="upper right", framealpha=0.85)

    # ── Panel 3: altitude-resolved absolute bias ──────────────────────────────
    # For each ISR sweep, interpolate prior and posterior onto the ISR alt grid,
    # then accumulate |Ne_prior − Ne_ISR| and |Ne_post − Ne_ISR|.
    # Bias panel: interpolate everything onto the common retrieval alt_grid so
    # that sweeps with different numbers of valid range gates can be stacked.
    bias_pr_on_grid: list[np.ndarray] = []
    bias_po_on_grid: list[np.ndarray] = []
    for prof in isr_profiles:
        # Interpolate the ISR sweep onto the retrieval altitude grid
        ne_isr_on_grid = np.interp(alt_grid, prof["alt_km"], prof["ne"],
                                   left=np.nan, right=np.nan)
        valid = ~np.isnan(ne_isr_on_grid)
        if valid.sum() < 5:
            continue
        abs_err_pr = np.where(valid, np.abs(prior_edp_at_mh - ne_isr_on_grid), np.nan)
        abs_err_po = np.where(valid, np.abs(post_edp_at_mh  - ne_isr_on_grid), np.nan)
        bias_pr_on_grid.append(abs_err_pr)
        bias_po_on_grid.append(abs_err_po)
        # Individual sweep signed error (light blue lines)
        signed_err = np.where(valid, post_edp_at_mh - ne_isr_on_grid, np.nan)
        ax_bias.plot(signed_err, alt_grid, color="royalblue", lw=0.7, alpha=0.25)

    if bias_pr_on_grid and bias_po_on_grid:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_bias_pr = np.nanmean(np.vstack(bias_pr_on_grid), axis=0)
            mean_bias_po = np.nanmean(np.vstack(bias_po_on_grid), axis=0)
        valid_alt = np.isfinite(mean_bias_pr) & np.isfinite(mean_bias_po)
        ax_bias.plot(mean_bias_pr[valid_alt], alt_grid[valid_alt],
                     color="black", lw=2.2, ls="--",
                     label="Mean |error| — Prior")
        ax_bias.plot(mean_bias_po[valid_alt], alt_grid[valid_alt],
                     color="royalblue", lw=2.5, ls="-",
                     label="Mean |error| — Posterior")
        ax_bias.axvline(0, color="gray", lw=0.8, ls=":")

    ax_bias.set_xlabel("|Ne_retrieval − Ne_ISR| (m⁻³)", fontsize=10)
    ax_bias.set_ylabel("Altitude (km)", fontsize=10)
    ax_bias.set_title("Profile Absolute Error vs. ISR", fontsize=10)
    ax_bias.xaxis.set_major_formatter(ne_fmt)
    ax_bias.grid(True, alpha=0.3, ls=":")
    ax_bias.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax_bias.set_ylim(bottom=0)

    plt.tight_layout()
    safe_key  = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    plot_path = os.path.join(save_dir, f"verif_isr_{safe_key}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ISR verification plot saved → {plot_path}")
    return plot_path


def plot_isr_summary(
    all_results:  list[dict],
    isr_profiles: list[dict],
    alt_grid:     np.ndarray,
    save_dir:     str,
) -> None:
    """
    Summary figure across all processed time windows:
      Left  — NmF2 time series (UTC hour): ISR truth, prior, posterior.
      Right — hmF2 time series (UTC hour): ISR truth, prior, posterior.
    """
    os.makedirs(save_dir, exist_ok=True)

    prior_nm, prior_hm = [], []
    post_nm,  post_hm  = [], []
    ret_hours           = []

    for res in all_results:
        if res.get("status") != "Success":
            continue
        eds    = res.get("eds_occ")
        if eds is None:
            continue
        verts  = eds.geolocation
        idx_mh = millstone_vertex_idx(verts)
        n_geo  = verts.shape[0]
        n_h    = len(alt_grid)

        pr_3d  = res.get("prior_edp_3d")
        # Use joint posterior — sequential is disabled in demo_verification.py
        # so post_edp_3d holds the prior fallback, not a real posterior.
        _jnt = res.get("joint_post_edp_3d")
        po_3d  = _jnt if _jnt is not None else res.get("post_edp_3d")
        if pr_3d is None or po_3d is None:
            continue

        pr_mh = np.asarray(pr_3d).reshape(n_h, n_geo)[:, idx_mh]
        po_mh = np.asarray(po_3d).reshape(n_h, n_geo)[:, idx_mh]
        nm_pr, hm_pr = extract_robust_f2_peak(pr_mh, alt_grid)
        nm_po, hm_po = extract_robust_f2_peak(po_mh, alt_grid)

        # Derive hour from time_window string "YYYY-MM-DD_HHMM"
        win = res["time_window"]
        try:
            hhmm = win.split("_")[-1]
            h    = int(hhmm[:2]) + int(hhmm[2:]) / 60.0
        except Exception:
            h = np.nan

        prior_nm.append(nm_pr); prior_hm.append(hm_pr)
        post_nm.append(nm_po);  post_hm.append(hm_po)
        ret_hours.append(h)

    isr_nm_arr  = np.array([p["nm_f2"]    for p in isr_profiles])
    isr_hm_arr  = np.array([p["hm_f2"]    for p in isr_profiles])
    isr_hr_arr  = np.array([p["hour_utc"] for p in isr_profiles])

    fig, (ax_nm, ax_hm) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Millstone Hill ISR F2-Peak Comparison — All Time Windows",
                 fontsize=13)

    ne_fmt = ScalarFormatter(useMathText=True)
    ne_fmt.set_powerlimits((-2, 2))

    # NmF2
    ax_nm.scatter(isr_hr_arr, isr_nm_arr,
                  s=30, color="gray", alpha=0.6, label="ISR sweeps", zorder=3)
    if ret_hours:
        ax_nm.plot(ret_hours, prior_nm, marker="D", ms=7, lw=1.5,
                   color="black", ls="--", label="Prior (IRI)", zorder=5)
        ax_nm.plot(ret_hours, post_nm,  marker="o", ms=7, lw=1.8,
                   color="royalblue", ls="-", label="Posterior (KF)", zorder=6)
    ax_nm.set_xlabel("UTC Hour", fontsize=10)
    ax_nm.set_ylabel("NmF2 (m⁻³)", fontsize=10)
    ax_nm.set_title("NmF2 vs. UTC Hour", fontsize=10)
    ax_nm.yaxis.set_major_formatter(ne_fmt)
    ax_nm.grid(True, alpha=0.3, ls=":")
    ax_nm.legend(fontsize=9, framealpha=0.85)

    # hmF2
    ax_hm.scatter(isr_hr_arr, isr_hm_arr,
                  s=30, color="gray", alpha=0.6, label="ISR sweeps", zorder=3)
    if ret_hours:
        ax_hm.plot(ret_hours, prior_hm, marker="D", ms=7, lw=1.5,
                   color="black", ls="--", label="Prior (IRI)", zorder=5)
        ax_hm.plot(ret_hours, post_hm,  marker="o", ms=7, lw=1.8,
                   color="royalblue", ls="-", label="Posterior (KF)", zorder=6)
    ax_hm.set_xlabel("UTC Hour", fontsize=10)
    ax_hm.set_ylabel("hmF2 (km)", fontsize=10)
    ax_hm.set_title("hmF2 vs. UTC Hour", fontsize=10)
    ax_hm.grid(True, alpha=0.3, ls=":")
    ax_hm.legend(fontsize=9, framealpha=0.85)

    plt.tight_layout()
    path = os.path.join(save_dir, "verif_summary_f2peak.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ISR summary figure saved → {path}")


def plot_verif_region_map(
    meta:     pd.DataFrame,
    save_dir: str,
    doy:      int = None,
    year:     int = None,
) -> None:
    """
    Regional map showing the 30°×60° verification patch and all occultation
    tangent points within it, colour-coded by orbit group.

    Each orbit gets one of 3 cycling colours from the Paired colormap.
    GN04 → circle, GN05 → square.  Legend is placed outside the axes to the right.
    """
    os.makedirs(save_dir, exist_ok=True)

    # One distinct colour per orbit group
    sorted_gkeys      = sorted(meta["group_key"].unique())
    n_orbits     = len(sorted_gkeys)
    _base_cmap   = cm.get_cmap("tab10" if n_orbits <= 10 else "tab20",
                               max(n_orbits, 1))
    ORBIT_COLORS = [_base_cmap(i) for i in range(n_orbits)]

    SC_MARKER = {"GN04": "o", "GN05": "s"}
    gkey_to_orbit_idx = {gk: i for i, gk in enumerate(sorted_gkeys)}

    proj = ccrs.LambertConformal(
        central_longitude=ISR_LON_W,
        central_latitude=ISR_LAT,
    )
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": proj})
    ax.set_extent(
        [VERIF_LON_MIN - 5, VERIF_LON_MAX + 5,
         VERIF_LAT_MIN - 3, VERIF_LAT_MAX + 3],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.7, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),   lw=0.4, edgecolor="lightgray")
    ax.add_feature(cfeature.STATES.with_scale("50m"),    lw=0.3, edgecolor="lightgray")
    ax.gridlines(draw_labels=True, lw=0.4, alpha=0.5)

    # Verification region boundary
    lon_box = [VERIF_LON_MIN, VERIF_LON_MAX, VERIF_LON_MAX,
               VERIF_LON_MIN, VERIF_LON_MIN]
    lat_box = [VERIF_LAT_MIN, VERIF_LAT_MIN, VERIF_LAT_MAX,
               VERIF_LAT_MAX, VERIF_LAT_MIN]
    ax.plot(lon_box, lat_box, transform=ccrs.Geodetic(),
            color="limegreen", lw=2.2, ls="-", zorder=4)

    # ── Scatter occultations per orbit × spacecraft ───────────────────────────
    legend_handles = [
        Line2D([0], [0], color="limegreen", lw=2.2, label="Verification region"),
    ]
    for gk in sorted_gkeys:
        orbit_i   = gkey_to_orbit_idx[gk]
        col       = ORBIT_COLORS[orbit_i % len(ORBIT_COLORS)]
        orbit_sub = meta[meta["group_key"] == gk]

        try:
            win_part = gk.split("__")[0]
            hhmm     = win_part.split("_")[-1]
            time_lbl = f"{hhmm[:2]}:{hhmm[2:]} UTC"
        except Exception:
            time_lbl = gk

        for sc in ["GN04", "GN05"]:
            sc_sub = orbit_sub[orbit_sub["spacecraft"] == sc]
            if sc_sub.empty:
                continue
            marker = SC_MARKER[sc]
            ax.scatter(
                sc_sub["lon"].values, sc_sub["lat"].values,
                transform=ccrs.Geodetic(),
                s=55, color=col, marker=marker,
                edgecolors="black", linewidths=0.5, zorder=5,
            )
            legend_handles.append(
                Line2D([0], [0], marker=marker, color="w",
                       markerfacecolor=col, markeredgecolor="black",
                       markersize=8,
                       label=f"Orbit {orbit_i} · {sc} ({time_lbl})")
            )

    # Millstone Hill ISR
    ax.plot(ISR_LON_W, ISR_LAT, transform=ccrs.Geodetic(),
            marker="^", ms=14, color="crimson", mec="black", mew=1.2,
            zorder=7)
    legend_handles.append(
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor="crimson", markeredgecolor="black",
               markersize=11, label="Millstone Hill ISR")
    )

    doy_str  = f"DOY {doy}"  if doy  is not None else ""
    year_str = f"{year}"     if year is not None else ""
    date_lbl = f"{year_str}  {doy_str}".strip() if (year_str or doy_str) else ""
    ax.set_title(
        f"Verification Region Centred on Millstone Hill ISR\n"
        f"30°lat × 60°lon  |  {len(meta)} occultations"
        + (f"  |  {date_lbl}" if date_lbl else ""),
        fontsize=12,
    )
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1), borderaxespad=0,
              fontsize=8.5, framealpha=0.85)

    path = os.path.join(save_dir, f"verif_region_map{year}.{doy}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Region map saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# §F  Incremental KF — sort occultations by distance from MH, add one-by-one
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (km) between two points in degrees."""
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _run_incremental_kf(
    res_full: dict,
    alt_grid: np.ndarray,
    measurement_err: float = 10.0,
    relaxation: float = 0.99,
) -> list[dict]:
    """
    Given the result of process_group (which already holds the parsed clean_list,
    eds_occ, H matrices, etc.), sort the occultations by great-circle distance
    from the Millstone Hill ISR and run cumulative joint-KF updates with
    1, 2, 3, … N occultations.

    Returns a list of step dicts (one per subset size), each containing:
        n_occ      : int   — number of occultations included
        dist_km    : float — great-circle distance of the last-added occultation
        time_delta : float — hours from mean group time to last-added occultation
        edp_mh     : (n_height,) ndarray — EDP at MH vertex after this update
        label      : str   — PRN / LEO label of the last-added occultation
    """
    from Ionosphere_Tomography_Inverter.Ionophy_Tomography_Inverter import (
        Ionosphere_Tomography_Inverter,
    )
    from demo_group import GAUSSIAN_COV_SIGMA

    clean_list  = res_full["clean_list"]
    eds_occ     = res_full["eds_occ"]
    sat_ids     = res_full.get("sat_ids", [])
    n_occ       = len(clean_list)

    if n_occ == 0:
        return []

    verts_geo  = eds_occ.geolocation
    n_geo      = verts_geo.shape[0]
    n_height   = len(alt_grid)
    idx_mh     = millstone_vertex_idx(verts_geo)

    # ── Compute TEC-max tangent lat/lon from ECEF geometry ───────────────────
    # clean_list entries contain LEO/GNSS arrays but not pre-computed lat/lon,
    # so derive them from the ray closest-approach point at the TEC-max sample.
    def _tangent_latlon(cl: dict) -> tuple[float, float]:
        try:
            i_tm   = int(np.argmax(cl["tec"]))
            leo_pt = cl["LEO"][:, i_tm]
            gns_pt = cl["GNSS"][:, i_tm]
            d      = gns_pt - leo_pt
            denom  = float(np.dot(d, d))
            if denom == 0:
                raise ValueError
            t_tp  = -float(np.dot(leo_pt, d)) / denom
            tp    = leo_pt + t_tp * d        # ECEF km
            r     = float(np.linalg.norm(tp))
            lat   = float(np.degrees(np.arcsin(tp[2] / r)))
            lon   = float(np.degrees(np.arctan2(tp[1], tp[0])))
            return lat, lon
        except Exception:
            return float(verts_geo[idx_mh, 1]), float(verts_geo[idx_mh, 0])

    occ_meta = []
    for i, cl in enumerate(clean_list):
        tec_lat, tec_lon = _tangent_latlon(cl)
        dist  = _haversine_km(ISR_LAT, ISR_LON_W, tec_lat, tec_lon)
        label = sat_ids[i][1] if i < len(sat_ids) else f"occ{i}"
        occ_meta.append({"idx": i, "dist_km": dist,
                         "label": label})

    # Sort by distance to MH ISR (closest first)
    occ_meta.sort(key=lambda m: m["dist_km"])

    # ── Build H matrices once (same prior mesh for all subsets) ──────────────
    # Re-use the pre-computed H blocks already computed by process_group.
    # They are stored in clean_list as cl["H"] if available, otherwise recompute.
    inverter_ref = Ionosphere_Tomography_Inverter(
        EDPSam=eds_occ, meanscale=1, topside_prior_floor_tecu=1.0,
        topside_alpha=0.0, gaussian_cov_sigma=GAUSSIAN_COV_SIGMA,
        altitude_taper_km=100.0, altitude_taper_min_scale=0.0001,
    )
    num_ray_segments = 60
    H_all = inverter_ref.get_observation_operator_batch(
        clean_list, num_segments=num_ray_segments
    )
    _n_sv     = inverter_ref.attrs["n_state_vars"]
    _n_sv_aug = inverter_ref.attrs["n_state_vars_aug"]
    prior_mean = inverter_ref.attrs["initial_edps_mean"]   # (n_sv, 1)
    x_top_pr   = inverter_ref.attrs["x_top_prior"]

    prior_edp_mh = prior_mean.reshape(n_height, n_geo)[:, idx_mh].copy()

    steps = []
    for step, meta in enumerate(occ_meta):
        # Subset of sorted occultation indices up to and including this step
        subset_orig_idxs = [m["idx"] for m in occ_meta[:step + 1]]

        # Build H and obs for this subset
        H_sub  = np.vstack([H_all[i] for i in subset_orig_idxs]).astype(np.float32)
        obs_sub = np.concatenate([
            clean_list[i]["tec"] for i in subset_orig_idxs
        ]).astype(np.float64)

        # Fresh inverter from same prior
        inv = Ionosphere_Tomography_Inverter(
            EDPSam=eds_occ, meanscale=1, topside_prior_floor_tecu=1.0,
            topside_alpha=0.0, gaussian_cov_sigma=GAUSSIAN_COV_SIGMA,
            altitude_taper_km=100.0, altitude_taper_min_scale=0.0001,
        )

        n_rel = sum(
            1 for i in subset_orig_idxs
            if clean_list[i].get("tec_type") == "relative"
        )
        if n_rel > 0:
            # Augment state for biases (re-build inverter with bias states)
            inv = Ionosphere_Tomography_Inverter(
                EDPSam=eds_occ, meanscale=1, topside_prior_floor_tecu=1.0,
                topside_alpha=0.0, gaussian_cov_sigma=GAUSSIAN_COV_SIGMA,
                altitude_taper_km=100.0, altitude_taper_min_scale=0.0001,
                n_rel_arcs=n_rel,
            )
            H_sub = np.hstack([
                H_sub,
                np.zeros((H_sub.shape[0], n_rel), dtype=np.float32),
            ])

        post_flat = inv.assimilate(
            obs=obs_sub, obs_operator=H_sub,
            relaxation=relaxation, measurement_err=measurement_err,
        )
        edp_mh = np.asarray(post_flat).reshape(n_height, n_geo)[:, idx_mh]

        steps.append({
            "n_occ":      step + 1,
            "dist_km":    meta["dist_km"],
            "edp_mh":     edp_mh.copy(),
            "label":      meta["label"],
            "orig_idx":   meta["idx"],
        })
        print(f"    [incr KF] step {step+1}/{n_occ}: +{meta['label']} "
              f"({meta['dist_km']:.0f} km from MH)  → MH NmF2 = "
              f"{float(np.max(edp_mh)):.2e} m⁻³")

    return prior_edp_mh, steps


def _plot_incremental_convergence(
    prior_edp_mh: np.ndarray,
    steps:        list[dict],
    isr_profiles: list[dict],
    alt_grid:     np.ndarray,
    group_key:    str,
    save_dir:     str,
) -> str:
    """
    Three-panel figure showing how the MH-vertex EDP converges as occultations
    sorted by distance are added one by one.

    Panel 1 (left)  — EDP profiles: prior (dashed black), ISR sweeps (coloured
                       by UTC hour), and each incremental posterior (blue→red
                       colour gradient by step number, labelled with PRN + dist).
    Panel 2 (centre) — NmF2 and hmF2 vs. cumulative occultation count (x-axis),
                        with ISR truth band shown as shaded region.
    Panel 3 (right) — Distance from MH (km) of each added occultation on the
                       x-axis vs. NmF2 error relative to ISR mean truth.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_steps  = len(steps)
    cmap_inc = cm.coolwarm
    norm_inc = mcolors.Normalize(vmin=0, vmax=max(n_steps - 1, 1))
    cmap_isr = cm.viridis
    norm_isr = mcolors.Normalize(vmin=0, vmax=24)
    ne_fmt   = ScalarFormatter(useMathText=True)
    ne_fmt.set_powerlimits((-2, 2))

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    ax_edp, ax_conv, ax_dist = axes
    fig.suptitle(
        f"Incremental KF Convergence at Millstone Hill  —  {group_key}\n"
        "Occultations added in order of increasing distance from ISR",
        fontsize=12,
    )

    # ISR truth values for reference
    isr_nm_vals = np.array([p["nm_f2"] for p in isr_profiles
                             if not np.isnan(p.get("nm_f2", np.nan))])
    isr_hm_vals = np.array([p["hm_f2"] for p in isr_profiles
                             if not np.isnan(p.get("hm_f2", np.nan))])
    isr_nm_mean = float(np.nanmean(isr_nm_vals)) if len(isr_nm_vals) else np.nan
    isr_hm_mean = float(np.nanmean(isr_hm_vals)) if len(isr_hm_vals) else np.nan
    isr_nm_std  = float(np.nanstd(isr_nm_vals))  if len(isr_nm_vals) else 0.0
    isr_hm_std  = float(np.nanstd(isr_hm_vals))  if len(isr_hm_vals) else 0.0

    # ── Panel 1: EDP profiles ─────────────────────────────────────────────────
    # ISR sweeps
    for prof in isr_profiles:
        col = cmap_isr(norm_isr(prof["hour_utc"]))
        ax_edp.plot(prof["ne"], prof["alt_km"], color=col, lw=1.0,
                    alpha=0.7, zorder=2)

    # Prior
    ax_edp.plot(prior_edp_mh, alt_grid, color="black", lw=2.2,
                ls="--", label="Prior (IRI)", zorder=4)

    # Incremental posteriors
    for k, step in enumerate(steps):
        col = cmap_inc(norm_inc(k))
        lw  = 1.2 + 0.5 * (k / max(n_steps - 1, 1))
        ax_edp.plot(step["edp_mh"], alt_grid, color=col, lw=lw,
                    alpha=0.85, zorder=3,
                    label=f"N={step['n_occ']} +{step['label']} "
                          f"({step['dist_km']:.0f} km)")

    ax_edp.set_xlabel("Electron Density (m⁻³)", fontsize=10)
    ax_edp.set_ylabel("Altitude (km)", fontsize=10)
    ax_edp.set_title("EDP at Millstone Hill Vertex", fontsize=10)
    ax_edp.xaxis.set_major_formatter(ne_fmt)
    ax_edp.grid(True, alpha=0.3, ls=":")
    ax_edp.legend(fontsize=7, loc="upper right", framealpha=0.85,
                  ncol=max(1, n_steps // 12))
    ax_edp.set_ylim(bottom=0)

    # ── Panel 2: NmF2 and hmF2 vs. step count ────────────────────────────────
    ns    = [s["n_occ"]  for s in steps]
    nm_st = [float(np.max(s["edp_mh"])) for s in steps]
    hm_st = [float(extract_robust_f2_peak(s["edp_mh"], alt_grid)[1])
              for s in steps]
    pr_nm, pr_hm = float(np.max(prior_edp_mh)), \
                   float(extract_robust_f2_peak(prior_edp_mh, alt_grid)[1])

    ax2 = ax_conv.twinx()

    # ISR truth bands
    if not np.isnan(isr_nm_mean):
        ax_conv.axhline(isr_nm_mean, color="steelblue", lw=1.4, ls=":",
                        alpha=0.8, label="ISR NmF2 mean")
        ax_conv.axhspan(isr_nm_mean - isr_nm_std, isr_nm_mean + isr_nm_std,
                        color="steelblue", alpha=0.10)
    if not np.isnan(isr_hm_mean):
        ax2.axhline(isr_hm_mean, color="darkorange", lw=1.4, ls=":",
                    alpha=0.8, label="ISR hmF2 mean")
        ax2.axhspan(isr_hm_mean - isr_hm_std, isr_hm_mean + isr_hm_std,
                    color="darkorange", alpha=0.10)

    ax_conv.plot([0] + ns, [pr_nm] + nm_st,
                 marker="o", ms=6, color="steelblue", lw=1.8,
                 label="NmF2 posterior")
    ax2.plot([0] + ns, [pr_hm] + hm_st,
             marker="s", ms=6, color="darkorange", lw=1.8, ls="--",
             label="hmF2 posterior")

    # x-axis tick labels: step number + PRN
    ax_conv.set_xticks([0] + ns)
    ax_conv.set_xticklabels(
        ["prior"] + [f"N={s['n_occ']}\n{s['label']}" for s in steps],
        fontsize=7, rotation=45, ha="right",
    )
    ax_conv.set_xlabel("Cumulative occultations added", fontsize=10)
    ax_conv.set_ylabel("NmF2 (m⁻³)", fontsize=10, color="steelblue")
    ax2.set_ylabel("hmF2 (km)", fontsize=10, color="darkorange")
    ax_conv.set_title("F2-Peak Convergence vs. Occultation Count", fontsize=10)
    ax_conv.yaxis.set_major_formatter(ne_fmt)
    ax_conv.grid(True, alpha=0.3, ls=":")
    lines1, labs1 = ax_conv.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax_conv.legend(lines1 + lines2, labs1 + labs2, fontsize=8, framealpha=0.85)

    # ── Panel 3: NmF2 error vs. distance ─────────────────────────────────────
    dists = [s["dist_km"] for s in steps]
    if not np.isnan(isr_nm_mean):
        nm_err = [float(np.max(s["edp_mh"])) - isr_nm_mean for s in steps]
        pr_err = pr_nm - isr_nm_mean
        colours = [cmap_inc(norm_inc(k)) for k in range(n_steps)]
        ax_dist.axhline(0, color="gray", lw=1.0, ls="--", alpha=0.7)
        ax_dist.axhline(pr_err, color="black", lw=1.2, ls=":",
                        alpha=0.7, label=f"Prior error")
        ax_dist.scatter(dists, nm_err, c=colours, s=60, zorder=4,
                        edgecolors="black", linewidths=0.5)
        # Connect the dots in order
        for k in range(n_steps - 1):
            ax_dist.plot([dists[k], dists[k + 1]],
                         [nm_err[k], nm_err[k + 1]],
                         color="gray", lw=0.8, alpha=0.5, zorder=3)
        for k, s in enumerate(steps):
            ax_dist.annotate(f"N={s['n_occ']}", (dists[k], nm_err[k]),
                             fontsize=6, xytext=(3, 3),
                             textcoords="offset points", alpha=0.8)
        ax_dist.set_xlabel("Distance from MH ISR (km)", fontsize=10)
        ax_dist.set_ylabel("NmF2 error vs. ISR mean (m⁻³)", fontsize=10)
        ax_dist.yaxis.set_major_formatter(ne_fmt)
        ax_dist.set_title("NmF2 Error vs. Occultation Distance", fontsize=10)
        ax_dist.grid(True, alpha=0.3, ls=":")
        ax_dist.legend(fontsize=9)
        # Colour bar for step number
        sm = cm.ScalarMappable(cmap=cmap_inc, norm=norm_inc)
        sm.set_array([])
        fig.colorbar(sm, ax=ax_dist, label="Step (0=closest)", fraction=0.046, pad=0.04)
    else:
        ax_dist.text(0.5, 0.5, "No ISR NmF2 data", transform=ax_dist.transAxes,
                     ha="center", va="center", fontsize=12, color="gray")

    plt.tight_layout()
    safe_key  = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    plot_path = os.path.join(save_dir, f"verif_{safe_key}_incremental.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved incremental convergence plot → {plot_path}")
    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# §G  Thin wrapper around process_group that enforces the fixed bounding box
# ─────────────────────────────────────────────────────────────────────────────

def _process_verif_group(
    group_key:              str,
    group_meta:             pd.DataFrame,
    alt_grid:               np.ndarray,
    global_edp_cache:       dict,
    measurement_err:        float = 10.0,
    relaxation:             float = 0.99,
    generate_plots:         bool  = True,
    save_dir:               str   = "./Figures/Verification/",
    conphs_base_dir:        str   = None,
    conphs_max_rays:        int   = 200,
    altitude_taper_km:        float = 100.0,
    altitude_taper_min_scale: float = 0.05,
    topside_follow_f2:        bool  = True,
    extra_clean_list:         list  = None,
) -> dict:
    """
    Thin wrapper around process_group() for the Millstone Hill verification
    region.  Sequential KF and its associated plots are disabled — only the
    joint (batch) update and the joint summary figure are produced.
    """
    return process_group(
        group_key                = group_key,
        group_meta               = group_meta,
        alt_grid                 = alt_grid,
        global_edp_cache         = global_edp_cache,
        measurement_err          = measurement_err,
        relaxation               = relaxation,
        generate_plots           = generate_plots,
        save_dir                 = save_dir,
        run_sequential           = False,
        conphs_base_dir          = conphs_base_dir,
        conphs_max_rays          = conphs_max_rays,
        altitude_taper_km        = altitude_taper_km,
        altitude_taper_min_scale = altitude_taper_min_scale,
        topside_follow_f2        = topside_follow_f2,
        extra_clean_list         = extra_clean_list,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §G  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def demo_verification_main() -> None:
    """
    Run the Millstone Hill ISR verification pipeline.

    1.  Scan all podTc2 metadata for the day.
    2.  Filter to the 30°×60° region centred on Millstone Hill.
    3.  Re-label all occultations with a single 'VERIF_MH' region key so
        they are grouped only by 30-minute time window (not by geographic bin).
    4.  Build the global EDP prior cache (shared with demo_group).
    5.  Process each time-window group with a joint KF update.
    6.  Compare the posterior at the Millstone Hill vertex to ISR truth.
    """

    # ── User-configurable settings ─────────────────────────────────────────────
    DOY  = 154
    YYYY = 2025
    base_path = (
        f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/"
        f"{YYYY}.{DOY}/"
    )
    alt_grid    = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)
    TYPE        = "log"
    save_dir    = "./Figures/Verification/"
    num_workers = 12
    kf_config   = {"measurement_err": 1.0, "relaxation": 0.99, "topside_follow_f2": True}

    # Optional: root directory containing conPhs files for the same day.
    # Set to None to assimilate only absolute podTc2 TEC (original behaviour).
    # When set, each podTc2 occultation will also try to load a matching
    # conPhs file; successfully matched arcs contribute relative TEC with
    # their carrier-phase bias jointly estimated by the Kalman Filter.
    conphs_base_dir = None
    # conphs_base_dir = (
    #     f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/conPhs/"
    #     f"{YYYY}.{DOY}/"
    # )

    # Maximum rays to retain per conPhs arc after decimation.
    # conPhs is recorded at 100 Hz; a typical 1–2 min arc contains thousands
    # of points.  Uniform-stride decimation brings this down to a manageable
    # size for the H-matrix computation.  Raise if you want finer arc sampling;
    # lower if memory or runtime is a concern.
    conphs_max_rays = 200

    # ISR netCDF files (Millstone Hill) — list as many days as needed.
    # Files should cover the same UTC day as DOY above.
    # Example: downloaded from Madrigal (http://millstonehill.haystack.mit.edu)
    isr_files = [
        # "./DataFiles/EDPS/mlh250602m.002.nc",
        "./DataFiles/EDPS/mlh250603m.002.nc",
    ]

    # Set to True to run the incremental KF (adds occultations one by one in
    # order of distance from MH and plots convergence).  Can be slow for groups
    # with many occultations.
    RUN_INCREMENTAL_KF = False
    # ──────────────────────────────────────────────────────────────────────────

    if not os.path.exists(base_path):
        print(f"ERROR: base_path not found: {base_path}")
        return

    print("=" * 65)
    print("  demo_verification.py — Millstone Hill ISR Verification")
    print("=" * 65)
    print(f"  ISR location : {ISR_LAT:.2f}°N, {ISR_LON:.2f}°E ({ISR_LON_W:.2f}°)")
    print(f"  Patch        : lat [{VERIF_LAT_MIN:.1f}, {VERIF_LAT_MAX:.1f}]  "
          f"lon [{VERIF_LON_MIN:.1f}, {VERIF_LON_MAX:.1f}]")

    # ── Step 1: Scan and filter metadata ──────────────────────────────────────
    meta = scan_metadata(base_path)
    if meta.empty:
        print("No valid podTc2 files found.  Exiting.")
        return

    meta_verif = filter_to_verif_region(meta)
    if meta_verif.empty:
        print("No occultations found in the verification region.  Exiting.")
        return

    meta_verif = assign_orbit_groups(meta_verif)

    # ── Step 2: Region map ────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    plot_verif_region_map(meta_verif, save_dir, doy=DOY, year=YYYY)

    # ── Step 3: Build global EDP prior ────────────────────────────────────────
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

    # ── Step 4: Load ISR truth ─────────────────────────────────────────────────
    isr_profiles: list[dict] = []
    if isr_files:
        isr_profiles = load_isr_profiles(isr_files)
    else:
        print("  [ISR] No ISR files configured — skipping ISR comparison plots.")

    # Make ISR profiles available to the _plot_group patch
    global _isr_profiles_for_patch
    _isr_profiles_for_patch = isr_profiles

    # ── Step 5: Process each orbit group ─────────────────────────────────────
    groups     = meta_verif.groupby("group_key", sort=True)
    group_keys = list(groups.groups.keys())
    print(f"\nProcessing {len(group_keys)} time-window group(s) …")

    all_results: list[dict] = []
    for g_idx, gk in enumerate(group_keys[:2]):
        print(f"\n[{g_idx + 1}/{len(group_keys)}]", end="")
        gm  = groups.get_group(gk)
        if conphs_base_dir is not None:
            gk = f"{gk}_conPhs"
        res = _process_verif_group(
            group_key        = gk,
            group_meta       = gm,
            alt_grid         = alt_grid,
            global_edp_cache = global_edp_cache,
            generate_plots   = True,
            save_dir         = save_dir,
            conphs_base_dir  = conphs_base_dir,
            conphs_max_rays  = conphs_max_rays,
            **kf_config,
        )
        all_results.append(res)

        # Per-window ISR comparison (only if ISR data available)
        if isr_profiles and res.get("status") == "Success":
            eds  = res.get("eds_occ")
            if eds is not None:
                verts  = eds.geolocation
                idx_mh = millstone_vertex_idx(verts)
                n_geo  = verts.shape[0]
                n_h    = len(alt_grid)
                pr_mh  = np.asarray(res["prior_edp_3d"]).reshape(n_h, n_geo)[:, idx_mh]
                # Always use the joint posterior — sequential is disabled in
                # demo_verification.py so post_edp_3d holds the prior fallback.
                po_mh  = np.asarray(res["joint_post_edp_3d"]).reshape(n_h, n_geo)[:, idx_mh]

                # Select ISR sweeps within ±WINDOW_MINUTES/2 of the orbit mean
                # time; fall back to the single sweep closest in time.
                win = res["time_window"]          # "YYYY-MM-DD_HHMM" (mean orbit time)
                try:
                    hhmm   = win.split("_")[-1]
                    h_mid  = int(hhmm[:2]) + int(hhmm[2:]) / 60.0
                    half   = WINDOW_MINUTES / 120.0
                    isr_win = [
                        p for p in isr_profiles
                        if abs(p["hour_utc"] - h_mid) < half
                    ]
                    if not isr_win:
                        # Pick the single sweep whose hour_utc is nearest to the
                        # orbit mean time (wrapping midnight via modular distance).
                        isr_win = [min(
                            isr_profiles,
                            key=lambda p: min(
                                abs(p["hour_utc"] - h_mid),
                                24 - abs(p["hour_utc"] - h_mid),
                            ),
                        )]
                except Exception:
                    isr_win = isr_profiles[:1]  # last-resort: first sweep only

                try:
                    plot_isr_profile_comparison(
                        isr_profiles    = isr_win,
                        prior_edp_at_mh = pr_mh,
                        post_edp_at_mh  = po_mh,
                        alt_grid        = alt_grid,
                        group_key       = gk,
                        save_dir        = save_dir,
                    )
                except Exception as exc:
                    print(f"  [warn] ISR comparison plot failed: {exc}")

                # ── Incremental KF: sort by distance to MH, add one-by-one ──
                if RUN_INCREMENTAL_KF and res.get("clean_list") and len(res["clean_list"]) > 1:
                    print(f"  Running incremental KF ({len(res['clean_list'])} occ) …")
                    try:
                        prior_mh_incr, incr_steps = _run_incremental_kf(
                            res_full        = res,
                            alt_grid        = alt_grid,
                            measurement_err = kf_config.get("measurement_err", 10.0),
                            relaxation      = kf_config.get("relaxation", 0.99),
                        )
                        _plot_incremental_convergence(
                            prior_edp_mh = prior_mh_incr,
                            steps        = incr_steps,
                            isr_profiles = isr_win,
                            alt_grid     = alt_grid,
                            group_key    = gk,
                            save_dir     = save_dir,
                        )
                    except Exception as exc_incr:
                        print(f"  [warn] Incremental KF failed: {exc_incr}")

    # ── Step 6: Statistics CSV ────────────────────────────────────────────────
    stats_csv = _save_stats_csv(all_results, YYYY, DOY)
    print(f"\nStats CSV saved → {stats_csv}")

    # ── Step 7: Summary ISR plot (all time windows together) ─────────────────
    if isr_profiles:
        try:
            plot_isr_summary(all_results, isr_profiles, alt_grid, save_dir)
        except Exception as exc:
            print(f"  [warn] ISR summary plot failed: {exc}")

    # ── Step 8: Console statistics ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Verification complete.  Statistics:")
    print("=" * 65)
    success = [r for r in all_results if r["status"] == "Success"]
    print(f"  Groups processed   : {len(success)} / {len(all_results)}")
    if success:
        rmse_pr = np.nanmean([r["prior_tec_rmse"] for r in success])
        rmse_po = np.nanmean([r["post_tec_rmse"]  for r in success])
        imprv   = (rmse_pr - rmse_po) / rmse_pr * 100.0 if rmse_pr > 0 else 0.0
        print(f"  Mean prior  TEC RMSE : {rmse_pr:.3f} TECU")
        print(f"  Mean post   TEC RMSE : {rmse_po:.3f} TECU")
        print(f"  Mean TEC improvement : {imprv:.1f} %")

    if isr_profiles:
        # Global NmF2/hmF2 bias summary across all successful windows
        nm_bias_list, hm_bias_list = [], []
        for res in success:
            eds = res.get("eds_occ")
            if eds is None:
                continue
            verts  = eds.geolocation
            idx_mh = millstone_vertex_idx(verts)
            n_geo  = verts.shape[0]
            n_h    = len(alt_grid)
            _jnt_arr = res.get("joint_post_edp_3d")
            po_mh  = np.asarray(
                _jnt_arr if _jnt_arr is not None else res["post_edp_3d"]
            ).reshape(n_h, n_geo)[:, idx_mh]
            nm_po, hm_po = extract_robust_f2_peak(po_mh, alt_grid)
            nm_isr = float(np.nanmean([p["nm_f2"] for p in isr_profiles]))
            hm_isr = float(np.nanmean([p["hm_f2"] for p in isr_profiles]))
            if not np.isnan(nm_po):
                nm_bias_list.append(nm_po - nm_isr)
            if not np.isnan(hm_po):
                hm_bias_list.append(hm_po - hm_isr)

        if nm_bias_list:
            print(f"  NmF2 bias (post−ISR): {np.nanmean(nm_bias_list):.3e} m⁻³")
        if hm_bias_list:
            print(f"  hmF2 bias (post−ISR): {np.nanmean(hm_bias_list):.1f} km")

    print("\nAll figures written.  Done.")


# ─────────────────────────────────────────────────────────────────────────────
# §H  Nearest-occultation TEC vs. ISR forward-modeled TEC
# ─────────────────────────────────────────────────────────────────────────────

def _iri_ne_at_mh(unix_sec: float, alt_grid_km: np.ndarray) -> np.ndarray:
    """
    Run IRI-2020 at the Millstone Hill ISR location for the UTC time
    corresponding to `unix_sec` and return the Ne profile (m⁻³) on `alt_grid_km`.

    Uses the iri2020 Python package (already on sys.path via ROOT/iri2020_new/src).
    Returns an array of NaN if IRI fails.
    """
    try:
        from iri2020 import IRI
        import datetime as _dt

        utc_dt  = _dt.datetime.utcfromtimestamp(unix_sec)
        # IRI expects altitude in km as a 3-element list [start, stop, step]
        alt_arr = np.asarray(alt_grid_km, dtype=float)
        alt_range = [float(alt_arr[0]), float(alt_arr[-1]),
                     float(alt_arr[1] - alt_arr[0])]

        iono = IRI(utc_dt, alt_range, ISR_LAT, ISR_LON)   # lon in °E 0–360
        ne   = np.asarray(iono["ne"].values).squeeze()     # m⁻³, on IRI alt grid

        # IRI alt grid may differ from alt_grid_km; interpolate onto our grid
        iri_alt = np.asarray(iono.coords["alt_km"].values).squeeze()
        ne_interp = np.interp(alt_arr, iri_alt, ne, left=0.0, right=0.0)
        return ne_interp

    except Exception as exc:
        print(f"    [IRI] failed for unix={unix_sec:.0f}: {exc}")
        return np.full(len(alt_grid_km), np.nan)

def _load_isr_sweeps_full(isr_files: list[str]) -> list[dict]:
    """
    Load ISR sweeps with absolute UTC timestamps and direct TEC from the
    Millstone Hill netCDF files.  Returns one dict per sweep containing:
        'unix_sec'  : float  — Unix timestamp of the sweep
        'hour_utc'  : float  — UTC hour (fractional)
        'alt_km'    : ndarray — altitude grid (km)
        'ne'        : ndarray — electron density (m⁻³)
        'nm_f2'     : float  — peak Ne
        'hm_f2'     : float  — peak altitude (km)
        'tec_tecu'  : float  — ISR vertical TEC (TECU) from file or integrated
    """
    sweeps = []
    for fpath in isr_files:
        if not os.path.exists(fpath):
            print(f"  [ISR] Not found: {fpath}")
            continue
        try:
            ds = netCDF4.Dataset(fpath, "r")
            alt   = np.array(ds.variables["gdalt"][:])               # (n_alt,) km
            nel   = np.ma.filled(np.array(ds.variables["nel"][:]), np.nan)  # (n_t, n_alt) log10 m⁻³
            times = np.array(ds.variables["timestamps"][:])          # Unix seconds

            # ISR TEC variable if present
            isr_tec_var = None
            if "tec" in ds.variables:
                isr_tec_var = np.array(ds.variables["tec"][:])       # (n_t,) TECU

            ds.close()

            if nel.ndim == 2 and nel.shape[1] != alt.shape[0]:
                nel = nel.T   # ensure (n_time, n_alt)

            n_time = nel.shape[0]
            for i in range(n_time):
                row   = nel[i, :]
                valid = ~np.isnan(row)
                if valid.sum() < 10:
                    continue
                alt_v    = alt[valid]
                ne_v     = 10.0 ** row[valid]
                t_unix   = float(times[i]) if times.ndim == 1 else float(np.nanmean(times[i]))
                hour_utc = (t_unix % 86400) / 3600.0
                nm, hm   = extract_robust_f2_peak(ne_v, alt_v)

                # Prefer direct ISR TEC variable; fall back to integration
                if isr_tec_var is not None and np.isfinite(isr_tec_var[i]):
                    tec_val = float(isr_tec_var[i])
                else:
                    tec_val = compute_isr_tec(alt_v, ne_v)

                sweeps.append({
                    "unix_sec":  t_unix,
                    "hour_utc":  hour_utc,
                    "alt_km":    alt_v,
                    "ne":        ne_v,
                    "nm_f2":     nm,
                    "hm_f2":     hm,
                    "tec_tecu":  tec_val,
                })
        except Exception as exc:
            print(f"  [ISR] Error reading {fpath}: {exc}")

    sweeps.sort(key=lambda s: s["unix_sec"])
    print(f"  Loaded {len(sweeps)} ISR sweeps from {len(isr_files)} file(s).")
    return sweeps



# Seconds between the Unix epoch (1970-01-01) and the GPS/podTc2 epoch (1980-01-06).
# podTc2 'start_time' is seconds elapsed since 1980-01-06T00:00 UTC;
# 'time' is seconds elapsed since 'start_time'.
_GPS_EPOCH_UNIX_OFFSET: float = pd.Timestamp("1980-01-06").timestamp()


def _gps_to_unix(gps_sec: float) -> float:
    """Convert podTc2 GPS-epoch seconds to Unix seconds."""
    return gps_sec + _GPS_EPOCH_UNIX_OFFSET


def _scan_metadata_with_exact_time(base_paths: dict[int, str]) -> pd.DataFrame:
    """
    Scan podTc2 files from multiple DOY directories and extract metadata
    including the exact TEC-max time derived from 'start_time' + 'time[i]'.

    'start_time' is the file epoch in seconds since 1980-01-06T00:00 UTC.
    'time' is relative to 'start_time' in seconds.

    Parameters
    ----------
    base_paths : dict mapping DOY (int) to directory path (str)

    Returns
    -------
    DataFrame with columns:
        full_path, date (Timestamp of file start), unix_tecmax (Unix s at TEC max),
        lat, lon (TEC-max tangent), spacecraft, filename, doy
    """
    rows = []
    for doy, base_path in base_paths.items():
        if not os.path.exists(base_path):
            print(f"  [scan] DOY {doy} path not found: {base_path}")
            continue
        files = sorted(f for f in os.listdir(base_path) if f.endswith(".0001_nc"))
        print(f"  DOY {doy}: scanning {len(files)} files …")
        for fname in files:
            fpath = os.path.join(base_path, fname)
            try:
                with netCDF4.Dataset(fpath, "r") as nc:
                    lat  = float(nc.getncattr("lat_tecmax_tangent"))
                    lon  = float(nc.getncattr("lon_tecmax_tangent"))
                    if abs(lat) > 90:
                        continue
                    yr, mo, dy = int(nc.getncattr("year")), int(nc.getncattr("month")), int(nc.getncattr("day"))
                    hh, mm, ss = int(nc.getncattr("hour")), int(nc.getncattr("minute")), int(nc.getncattr("second"))
                    time_arr   = np.array(nc.variables["time"][:])   # absolute GPS-epoch seconds
                    tec_arr    = np.array(nc.variables["TEC"][:])
            except Exception:
                continue

            dt_start   = pd.Timestamp(yr, mo, dy, hh, mm, ss)
            spacecraft = fname.split(".")[0].replace("podTc2_", "")

            # time[i] is absolute GPS-epoch seconds; convert to Unix for comparison
            # with ISR timestamps (which are Unix seconds).
            i_tm        = int(np.argmax(np.abs(tec_arr)))
            unix_tecmax = _gps_to_unix(float(time_arr[i_tm]))

            rows.append({
                "full_path":   fpath,
                "filename":    fname,
                "date":        dt_start,
                "unix_tecmax": unix_tecmax,
                "lat":         lat,
                "lon":         lon,          # −180…+180
                "spacecraft":  spacecraft,
                "doy":         doy,
            })

    df = pd.DataFrame(rows).sort_values("unix_tecmax").reset_index(drop=True)
    print(f"  Total occultations found: {len(df)}")
    return df


def _fit_epstein_topside(
    alt_km: np.ndarray,
    ne:     np.ndarray,
    hmF2:   float,
    NmF2:   float,
) -> tuple[float, float]:
    """
    Fit H0 and gamma of the NeQuick/IRI restricted-scale-height Epstein layer
    to the topside of a measured Ne profile.

    Epstein model (Bilitza et al. Eq. 6):
        dh    = h - hmF2
        H_top = H0 * (1 + r*gamma*dh / (r*H0 + gamma*dh))   [r = 100]
        z     = dh / H_top
        Ne    = 4*NmF2 * exp(z) / (1 + exp(z))^2

    Returns (H0_km, gamma) or fallback (70.0, 0.15) if fit fails.
    """
    mask   = alt_km >= hmF2
    h_top  = alt_km[mask]
    ne_top = ne[mask]

    if len(h_top) < 2:
        return 70.0, 0.15

    def epstein_model(h, H0, gamma):
        r  = 100.0
        dh = h - hmF2
        H_top = H0 * (1.0 + (r * gamma * dh) / (r * H0 + gamma * dh + 1e-9))
        z = np.clip(dh / H_top, -100, 100)
        return 4.0 * NmF2 * np.exp(z) / (1.0 + np.exp(z)) ** 2

    try:
        popt, _ = curve_fit(
            epstein_model, h_top, ne_top,
            p0=[70.0, 0.15],
            bounds=([20.0, 0.001], [300.0, 1.5]),
            maxfev=2000,
        )
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return 70.0, 0.15


def _extend_isr_profile_topside(
    alt_km:          np.ndarray,
    ne:              np.ndarray,
    topside_ceil_km: float           = 2000.0,
    topside_step_km: float           = 2.0,
    n_fit_pts:       int             = 5,     # kept for API compatibility; unused
    iri_alt_km:      np.ndarray|None = None,  # IRI altitude grid (km)
    iri_ne:          np.ndarray|None = None,  # IRI Ne (m⁻³) on that grid
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extend an ISR Ne profile above its top measured point using an Epstein
    layer.

    When IRI data are supplied (iri_alt_km / iri_ne), H0 and gamma are fitted
    to the *IRI* topside — which has a full, smooth column — and then applied
    using the ISR's own hmF2/NmF2 as the anchor.  This gives the physically
    motivated scale-height shape from IRI while respecting the ISR's measured
    peak density and height.

    Without IRI data the parameters are fitted directly to the ISR topside
    (original behaviour).

    The topmost ISR measured point is always discarded before fitting (it is
    often noisy at the radar's range limit).
    """
    # Drop the noisy topmost point
    alt_trunc = alt_km[:-1]
    ne_trunc  = ne[:-1]

    if len(alt_trunc) < 2:
        return alt_km, ne

    h_top_meas = float(alt_trunc[-1])

    # F2 peak from the ISR truncated profile (used as anchor)
    i_peak = int(np.argmax(ne_trunc))
    hmF2   = float(alt_trunc[i_peak])
    NmF2   = float(ne_trunc[i_peak])

    h_tail = np.arange(h_top_meas + topside_step_km,
                       topside_ceil_km + topside_step_km,
                       topside_step_km)

    isr_peak_resolved = (hmF2 < h_top_meas) and (NmF2 > 0)

    if isr_peak_resolved:
        # Decide source for Epstein shape parameters
        if iri_alt_km is not None and iri_ne is not None:
            iri_alt_arr = np.asarray(iri_alt_km)
            iri_ne_arr  = np.asarray(iri_ne)
            valid_iri   = np.isfinite(iri_ne_arr) & (iri_ne_arr > 0)
            # Locate IRI F2 peak to use as reference for the shape fit
            if valid_iri.sum() >= 4:
                i_iri_peak  = int(np.argmax(iri_ne_arr[valid_iri]))
                hmF2_iri    = float(iri_alt_arr[valid_iri][i_iri_peak])
                NmF2_iri    = float(iri_ne_arr[valid_iri][i_iri_peak])
                H0, gamma   = _fit_epstein_topside(
                    iri_alt_arr[valid_iri], iri_ne_arr[valid_iri],
                    hmF2_iri, NmF2_iri,
                )
            else:
                H0, gamma = _fit_epstein_topside(alt_trunc, ne_trunc, hmF2, NmF2)
        else:
            H0, gamma = _fit_epstein_topside(alt_trunc, ne_trunc, hmF2, NmF2)

        # Evaluate the Epstein shape at the junction altitude so we can
        # scale the whole tail to be continuous with the last ISR point.
        def _epstein_ne(h_arr):
            r     = 100.0
            dh    = h_arr - hmF2
            H_top = H0 * (1.0 + (r * gamma * dh) / (r * H0 + gamma * dh + 1e-9))
            z     = np.clip(dh / H_top, -100, 100)
            return 4.0 * NmF2 * np.exp(z) / (1.0 + np.exp(z)) ** 2

        ne_epstein_at_junction = _epstein_ne(np.array([h_top_meas]))[0]
        ne_isr_at_junction     = float(ne_trunc[-1])

        # Scale factor forces continuity at h_top_meas; guard against near-zero
        if ne_epstein_at_junction > 0:
            scale = ne_isr_at_junction / ne_epstein_at_junction
        else:
            scale = 1.0

        ne_tail = scale * _epstein_ne(h_tail)

    else:
        # F2 peak above ISR range — exponential fallback from the top point
        ne_top = float(ne_trunc[-1])
        pos    = (ne_trunc > 0) & (alt_trunc >= h_top_meas - 100.0)
        if pos.sum() >= 2:
            coeffs = np.polyfit(alt_trunc[pos], np.log(ne_trunc[pos]), 1)
            slope  = float(coeffs[0])
            H_km   = -1.0 / slope if slope < 0 else 50.0
        else:
            H_km = 50.0
        ne_tail = ne_top * np.exp(-(h_tail - h_top_meas) / H_km)

    return np.concatenate([alt_trunc, h_tail]), np.concatenate([ne_trunc, ne_tail])


def _forward_slant_tec_from_isr(
    leo_ecef_km:     np.ndarray,        # shape (3,)
    gnss_ecef_km:    np.ndarray,        # shape (3,)
    isr_alt_km:      np.ndarray,
    isr_ne:          np.ndarray,        # m⁻³
    n_segments:      int            = 300,
    extend_topside:  bool           = True,
    topside_ceil_km: float          = 2000.0,
    n_fit_pts:       int            = 5,
    iri_alt_km:      np.ndarray|None = None,  # IRI altitude grid for topside shape
    iri_ne:          np.ndarray|None = None,  # IRI Ne for topside shape
) -> float:
    """
    Forward-model the slant TEC along the LEO–GNSS ray using the ISR vertical
    Ne profile.  The ionosphere is assumed horizontally uniform (the ISR profile
    applies everywhere along the ray).

    When `extend_topside` is True (default) the profile is first passed through
    `_extend_isr_profile_topside`, which discards the noisy topmost point and
    appends an exponential decay fit to the upper portion of the remaining
    profile up to `topside_ceil_km`.

    Integration is a trapezoid sum along the ray; units conversion:
        TEC [TECU] = Σ Ne [m⁻³] × dl [m] / 1e16

    Returns np.nan if the ray does not pass through the profile altitude range.
    """
    # Sort and strip non-finite / non-positive values
    order  = np.argsort(isr_alt_km)
    alt    = isr_alt_km[order]
    ne     = isr_ne[order]
    valid  = np.isfinite(ne) & (ne > 0)
    alt    = alt[valid]
    ne     = ne[valid]

    if len(alt) < 2:
        return np.nan

    if extend_topside:
        alt, ne = _extend_isr_profile_topside(
            alt, ne,
            topside_ceil_km = topside_ceil_km,
            n_fit_pts       = n_fit_pts,
            iri_alt_km      = iri_alt_km,
            iri_ne          = iri_ne,
        )

    t_vals  = np.linspace(0.0, 1.0, n_segments)
    ray_pts = leo_ecef_km[:, None] + t_vals[None, :] * (gnss_ecef_km - leo_ecef_km)[:, None]
    r_km    = np.linalg.norm(ray_pts, axis=0)
    alt_ray = r_km - 6371.0

    alt_lo, alt_hi = float(alt.min()), float(alt.max())
    if not np.any((alt_ray >= alt_lo) & (alt_ray <= alt_hi)):
        return np.nan

    ne_interp = np.interp(alt_ray, alt, ne, left=0.0, right=0.0)

    dl_km  = np.linalg.norm(np.diff(ray_pts, axis=1), axis=0)
    ne_mid = (ne_interp[:-1] + ne_interp[1:]) / 2.0
    tec    = float(np.sum(ne_mid * dl_km * 1e3)) / 1e16
    return tec


def parse_ionosonde_file(filepath: str) -> pd.DataFrame:
    """
    Parse a standard GIRO DIDBase text file into a DataFrame.

    The header line beginning with '# Time' supplies column names; 'QD'
    columns are renamed '<param>_QD' to avoid duplicates.  Missing values
    ('---') become NaN and the Time column is cast to datetime64.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Ionosonde file not found: {filepath}")

    header_cols: list[str] = []
    skip_lines = 0
    with open(filepath, "r") as fh:
        for i, line in enumerate(fh):
            if line.startswith("# Time"):
                raw = line.replace("#", "").strip().split()
                for j, col in enumerate(raw):
                    if col == "QD" and j > 0:
                        header_cols.append(f"{raw[j-1]}_QD")
                    else:
                        header_cols.append(col)
                skip_lines = i + 1
                break

    if not header_cols:
        raise ValueError("Could not find header line starting with '# Time'")

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=skip_lines,
        names=header_cols,
        na_values=["---"],
        engine="python",
    )
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])

    return df


def plot_occ_tec_max_vs_isr_forward_tec(
    isr_files:    list[str],
    base_paths:   dict,
    yyyy:         int,
    save_dir:     str   = "./Figures/Verification/",
    spatial_weight_km:   float = 500.0,
    temporal_weight_min: float = 60.0,
    max_match_dist_km:   float = 1500.0,
    kf_profiles:  list[dict] | None = None,
    ionosonde_file:      str | None = None,
    alt_grid:            np.ndarray | None = None,
    global_edp_cache:    dict | None = None,
    kf_config:           dict | None = None,
    num_workers:         int  = 8,
    global_edp_data_dir: str  = "./Data/Global_EDPS/",
) -> str:
    """
    For each ISR EDP sweep (DOY 153 & 154, Millstone Hill), find the nearest
    podTc2 occultation in combined space-time, then compare:

      • Occultation truth TEC max  — peak of TEC_podTc2 array, plotted vs.
                                     the *exact* time of the TEC-max sample
                                     (from the 'time' Unix-seconds variable),
                                     colour-coded by great-circle distance
                                     from the TEC-max tangent point to MH ISR.

      • ISR forward-modeled TEC   — ISR Ne(h) profile at the matched sweep
                                     time, forward-modeled along the matched
                                     occultation ray geometry via a slant-path
                                     integral (horizontally uniform assumption).

      • Assimilation (KF) EDP     — joint KF posterior at the MH mesh vertex,
                                     computed via _process_verif_group for the
                                     single occultation closest to each ISR sweep
                                     (mirrors demo_verification_single approach).
                                     The global EDP prior cache is built
                                     automatically per DOY if not supplied.

    The combined space-time metric used for matching is:
        score = (dist_km / spatial_weight_km) + (|Δt_min| / temporal_weight_min)
    The best match (lowest score) within max_match_dist_km is selected.

    Parameters
    ----------
    isr_files            : list of ISR netCDF file paths (both days)
    base_paths           : dict {doy: directory_path} for podTc2 files
    yyyy                 : calendar year (e.g. 2025); used to convert DOY to date
                           when building the global EDP prior cache.
    save_dir             : output directory for the figure
    spatial_weight_km    : normalization weight for spatial distance (km)
    temporal_weight_min  : normalization weight for temporal difference (min)
    max_match_dist_km    : maximum allowed spatial distance for a valid match (km)
    ionosonde_file       : path to a GIRO DIDBase text file; if given, foF2 is
                           overlaid on the overview panel and NmF2 is marked on
                           the EDP panel for each matched occultation.
    alt_grid             : altitude grid (km) for the assimilation KF; defaults
                           to 55 log-spaced levels from 60 to 800 km.
    global_edp_cache     : pre-built EDP prior cache ({hour: EDPSamples}) shared
                           across all DOYs, or a {doy: {hour: EDPSamples}} dict
                           for per-DOY caches.  If None the cache is built
                           automatically from IRI-2020 via build_daily_global_edps.
    kf_config            : dict of KF kwargs forwarded to _process_verif_group
                           (e.g. {"measurement_err": 1.0, "relaxation": 0.99}).
    num_workers          : parallel workers for build_daily_global_edps.
    global_edp_data_dir  : directory for caching EDP NetCDF files built by
                           build_daily_global_edps.

    Returns
    -------
    str : path to the saved figure
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Parse ionosonde data (optional) ──────────────────────────────────────
    ionosonde_df: pd.DataFrame | None = None
    if ionosonde_file:
        try:
            ionosonde_df = parse_ionosonde_file(ionosonde_file)
            print(f"  [ionosonde] Loaded {len(ionosonde_df)} records from {ionosonde_file}")
        except Exception as exc:
            print(f"  [ionosonde] Warning — could not load '{ionosonde_file}': {exc}")

    # ── Load ISR sweeps ───────────────────────────────────────────────────────
    isr_sweeps = _load_isr_sweeps_full(isr_files)
    if not isr_sweeps:
        print("  [plot_occ_tec_max] No ISR sweeps loaded.  Aborting.")
        return ""

    # ── Scan all podTc2 metadata (both DOYs) ─────────────────────────────────
    occ_meta = _scan_metadata_with_exact_time(base_paths)
    if occ_meta.empty:
        print("  [plot_occ_tec_max] No occultation files found.  Aborting.")
        return ""

    # ── Match each ISR sweep to the nearest occultation ───────────────────────
    # Pre-compute distance from every occultation TEC-max tangent to MH ISR
    occ_dist_km = np.array([
        _haversine_km(ISR_LAT, ISR_LON_W, row["lat"], row["lon"])
        for _, row in occ_meta.iterrows()
    ])

    matched: list[dict] = []
    print(f"\n  Matching {len(isr_sweeps)} ISR sweeps to "
          f"{len(occ_meta)} occultations …")

    for sweep in isr_sweeps:
        dt_isr_min = sweep["unix_sec"] / 60.0   # Unix time in minutes

        # Combined space-time score for every occultation
        dt_occ_min = occ_meta["unix_tecmax"].values / 60.0
        time_diff_min = np.abs(dt_occ_min - dt_isr_min)
        score = (occ_dist_km / spatial_weight_km) + (time_diff_min / temporal_weight_min)

        # Mask out occultations beyond the spatial cutoff
        score[occ_dist_km > max_match_dist_km] = np.inf

        best_idx = int(np.argmin(score))
        if not np.isfinite(score[best_idx]):
            # No match within spatial cutoff
            continue

        best_row  = occ_meta.iloc[best_idx]
        dist_km   = float(occ_dist_km[best_idx])
        dt_min    = float(time_diff_min[best_idx])

        # ── Parse the matched occultation file ────────────────────────────────
        try:
            pdata = parse_podTc2_nc_file(best_row["full_path"])
        except Exception as exc:
            print(f"    [warn] parse failed for {best_row['filename']}: {exc}")
            continue
        if pdata is None:
            continue

        tec_arr  = np.asarray(pdata["TEC_podTc2"])
        leo_arr  = np.asarray(pdata["LEO"])    # (3, n_t) km
        gnss_arr = np.asarray(pdata["GNSS"])   # (3, n_t) km
        time_arr_abs = np.asarray(pdata["time"])  # Unix seconds (before relative shift)

        i_tm = int(np.argmax(np.abs(tec_arr)))
        tec_max_truth = float(tec_arr[i_tm])

        # Exact UTC time at TEC max: re-read raw start_time + time[i] and
        # convert from GPS-epoch seconds to Unix seconds.
        try:
            with netCDF4.Dataset(best_row["full_path"], "r") as nc_raw:
                time_raw = np.array(nc_raw.variables["time"][:])
                tec_raw  = np.array(nc_raw.variables["TEC"][:])
            i_tm_raw    = int(np.argmax(np.abs(tec_raw)))
            unix_tecmax = _gps_to_unix(float(time_raw[i_tm_raw]))
        except Exception:
            unix_tecmax = float(best_row["unix_tecmax"])

        # ── Forward-model TEC from ISR EDP along the matched ray ──────────────
        leo_tm  = leo_arr[:, i_tm]
        gnss_tm = gnss_arr[:, i_tm]
        fwd_tec = _forward_slant_tec_from_isr(
            leo_ecef_km  = leo_tm,
            gnss_ecef_km = gnss_tm,
            isr_alt_km   = sweep["alt_km"],
            isr_ne       = sweep["ne"],
        )

        _con_id = str(pdata.get("conid",  "?")).strip()
        _prn_id = str(pdata.get("prn_id", "?")).strip()
        matched.append({
            "isr_unix":        sweep["unix_sec"],
            "isr_hour_utc":    sweep["hour_utc"],
            "isr_tec_tecu":    sweep["tec_tecu"],
            "isr_alt_km":      sweep["alt_km"],
            "isr_ne":          sweep["ne"],
            "occ_unix_tecmax": unix_tecmax,
            "occ_hour_utc":    (unix_tecmax % 86400) / 3600.0,
            "occ_tec_max":     tec_max_truth,
            "occ_fwd_tec":     fwd_tec,
            "dist_km":         dist_km,
            "dt_min":          dt_min,
            "filename":        best_row["filename"],
            "full_path":       best_row["full_path"],
            "spacecraft":      best_row["spacecraft"],
            "con_id":          _con_id,
            "prn_id":          _prn_id,
            "prn_label":       f"{_con_id}{_prn_id}",
            "doy":             int(best_row["doy"]),
            # ray geometry at TEC-max needed for IRI forward model (computed after dedup)
            "_leo_tm":         leo_tm.tolist(),
            "_gnss_tm":        gnss_tm.tolist(),
        })
        print(
            f"    ISR {sweep['hour_utc']:.2f} h UTC  →  {best_row['filename'][:50]}  "
            f"dist={dist_km:.0f} km  Δt={dt_min:.1f} min  "
            f"TEC_occ={tec_max_truth:.2f} TECU  TEC_fwd={fwd_tec:.2f} TECU"
        )

    if not matched:
        print("  [plot_occ_tec_max] No matched pairs found.  Aborting.")
        return ""

    # ── Deduplicate: keep the best ISR-sweep match per unique occultation file ─
    # Multiple ISR sweeps can point to the same occultation (the closest one in
    # space-time), which would produce repeated forward-TEC points at the same
    # occ_hour_utc.  For each unique file keep only the sweep with the smallest
    # combined space-time score (dist/w_s + dt/w_t).
    best_per_occ: dict[str, dict] = {}
    for m in matched:
        fname = m["filename"]
        score = m["dist_km"] / spatial_weight_km + m["dt_min"] / temporal_weight_min
        if fname not in best_per_occ or score < best_per_occ[fname]["_score"]:
            m["_score"] = score
            best_per_occ[fname] = m
    matched_dedup = sorted(best_per_occ.values(), key=lambda m: m["occ_unix_tecmax"])

    print(f"\n  Matched {len(matched)} ISR sweep–occultation pairs "
          f"→ {len(matched_dedup)} unique occultations after deduplication.")

    matched = matched_dedup

    # ── Assimilation (KF) posterior EDPs — demo_verification_single approach ──
    # For each matched occultation run _process_verif_group with that single
    # file to obtain a joint KF posterior EDP at the Millstone Hill mesh vertex.
    #
    # Global EDP prior cache resolution (per DOY):
    #   1. Caller supplied global_edp_cache as {doy: {hour: EDPSamples}} — use
    #      the sub-dict for each DOY directly.
    #   2. Caller supplied global_edp_cache as {hour: EDPSamples} (single DOY,
    #      e.g. pre-built externally) — share it across all DOYs.
    #   3. global_edp_cache is None — build one cache per DOY automatically
    #      via build_daily_global_edps using yyyy + the DOY from base_paths.

    # Resolve altitude grid — default to 55 log-spaced levels 60–800 km
    if alt_grid is None:
        alt_grid = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)

    # Apply module-level patches (same as demo_verification_single)
    from EDPSamples.edp_samples import EDPSamples as _EDPSamples
    _demo_group.region_bounding_box    = _patched_region_bounding_box
    _demo_group._plot_group            = _patched_plot_group
    _EDPSamples.subset_union_triangles = lambda self, *a, **kw: self

    _kf_cfg = kf_config or {}

    # Determine whether the caller gave a flat {hour: cache} or a {doy: {hour: cache}}
    _first_val   = next(iter(global_edp_cache.values())) if global_edp_cache else None
    _cache_by_doy = isinstance(_first_val, dict)   # True → already per-DOY

    # Build per-DOY lookup: {doy: {hour: EDPSamples}}
    _doy_cache: dict[int, dict] = {}
    for _doy, _bpath in base_paths.items():
        if not os.path.exists(_bpath):
            continue
        if global_edp_cache is not None:
            # Use supplied cache (flat or per-DOY)
            _doy_cache[_doy] = global_edp_cache[_doy] if _cache_by_doy else global_edp_cache
        else:
            # Build automatically for this DOY
            _batch_date = pd.Timestamp(
                datetime.date(yyyy, 1, 1) + datetime.timedelta(days=int(_doy) - 1)
            )
            _data_dir = os.path.join(global_edp_data_dir, f"DOY_{_doy}/")
            print(f"\n  Building global EDP cache for DOY {_doy} ({_batch_date.date()}) …")
            _doy_cache[_doy] = build_daily_global_edps(
                _batch_date, alt_grid,
                dLat=5.0, dLon=5.0,
                num_workers=num_workers,
                data_dir=_data_dir,
            )
            print(f"  Global EDP cache for DOY {_doy} ready.")

    # Build full scan_metadata for each DOY once (needed by _process_verif_group)
    _doy_meta: dict[int, pd.DataFrame] = {}
    for _doy, _bpath in base_paths.items():
        if os.path.exists(_bpath):
            _doy_meta[_doy] = scan_metadata(_bpath)

    print(f"\n  Running KF assimilation for {len(matched)} occultations …")
    for m in matched:
        _doy      = int(m["doy"])
        _cache_m  = _doy_cache.get(_doy)
        _meta_all = _doy_meta.get(_doy)

        if _cache_m is None or _meta_all is None:
            m["kf_alt"] = None
            m["kf_ne"]  = None
            continue

        _meta_single = _meta_all[_meta_all["filename"] == m["filename"]].copy()
        if _meta_single.empty:
            m["kf_alt"] = None
            m["kf_ne"]  = None
            continue

        _meta_single["region"]    = "VERIF_MH"
        _meta_single["group_key"] = _meta_single["time_window"] + "__VERIF_MH"
        _gk = _meta_single["group_key"].iloc[0]

        try:
            _res = _process_verif_group(
                group_key        = _gk,
                group_meta       = _meta_single,
                alt_grid         = alt_grid,
                global_edp_cache = _cache_m,
                generate_plots   = False,
                save_dir         = save_dir,
                **_kf_cfg,
            )
            if _res.get("status") == "Success":
                _eds   = _res.get("eds_occ")
                _verts = _eds.geolocation if _eds is not None else None
                if _verts is not None:
                    _idx_mh = millstone_vertex_idx(_verts)
                    _n_h    = len(alt_grid)
                    _n_geo  = _verts.shape[0]
                    _po     = np.asarray(_res["joint_post_edp_3d"]).reshape(_n_h, _n_geo)
                    m["kf_alt"] = alt_grid
                    m["kf_ne"]  = _po[:, _idx_mh]
                    _kf_nm, _kf_hm = extract_robust_f2_peak(m["kf_ne"], m["kf_alt"])
                    print(f"    KF ok  {m['spacecraft']} {m['prn_label']}: "
                          f"NmF2={_kf_nm:.2e} m⁻³  hmF2={_kf_hm:.1f} km")
                else:
                    m["kf_alt"] = None
                    m["kf_ne"]  = None
            else:
                print(f"    [KF] {m['filename'][:45]}  status={_res.get('status')}")
                m["kf_alt"] = None
                m["kf_ne"]  = None
        except Exception as _exc:
            print(f"    [KF warn] {m['filename'][:45]}: {_exc}")
            m["kf_alt"] = None
            m["kf_ne"]  = None

    # ── Match ionosonde measurement to each occultation ───────────────────────
    # foF2 (MHz) → NmF2 (m⁻³): NmF2 = (foF2 × 10⁶)² / 80.6
    # hmF2 (km) used as the altitude of the NmF2 marker if available.
    _IONO_FO_TO_NM = lambda fo_mhz: (fo_mhz * 1e6) ** 2 / 80.6
    _iono_unix: np.ndarray | None = None
    if ionosonde_df is not None and "foF2" in ionosonde_df.columns and "Time" in ionosonde_df.columns:
        _valid_iono = ionosonde_df.dropna(subset=["foF2", "Time"])
        if not _valid_iono.empty:
            _iono_unix = _valid_iono["Time"].astype("int64").values / 1e9  # → Unix s

    for m in matched:
        m["iono_foF2"]   = np.nan
        m["iono_hmF2"]   = np.nan
        m["iono_NmF2"]   = np.nan
        m["iono_unix"]   = np.nan
        if _iono_unix is not None and len(_iono_unix) > 0:
            _dt = np.abs(_iono_unix - m["occ_unix_tecmax"])
            _best_i = int(np.argmin(_dt))
            _row = _valid_iono.iloc[_best_i]
            m["iono_foF2"] = float(_row["foF2"]) if pd.notna(_row["foF2"]) else np.nan
            m["iono_NmF2"] = _IONO_FO_TO_NM(m["iono_foF2"]) if np.isfinite(m["iono_foF2"]) else np.nan
            m["iono_unix"] = float(_iono_unix[_best_i])
            if "hmF2" in _valid_iono.columns and pd.notna(_row.get("hmF2", np.nan)):
                m["iono_hmF2"] = float(_row["hmF2"])

    # ── IRI forward TEC at MH for each unique occultation ────────────────────
    # Use a uniform 1-km altitude grid from 60 to 2000 km so IRI can be called
    # with a simple [start, stop, step] specification and the full topside is
    # captured.  The Ne profile is then passed to the same slant-path integrator.
    print(f"  Running IRI-2020 at MH for {len(matched)} occultation times …")
    iri_alt_uniform = np.arange(60.0, 2001.0, 1.0)   # 1-km steps, 60–2000 km
    for m in matched:
        ne_iri = _iri_ne_at_mh(m["occ_unix_tecmax"], iri_alt_uniform)
        m["iri_ne"]  = ne_iri
        m["iri_alt"] = iri_alt_uniform
        if np.all(np.isnan(ne_iri)):
            m["iri_fwd_tec"] = np.nan
        else:
            m["iri_fwd_tec"] = _forward_slant_tec_from_isr(
                leo_ecef_km  = np.asarray(m["_leo_tm"]),
                gnss_ecef_km = np.asarray(m["_gnss_tm"]),
                isr_alt_km   = iri_alt_uniform,
                isr_ne       = ne_iri,
                n_segments   = 500,
            )
        print(f"    {m['filename'][:50]}  IRI fwd TEC = {m['iri_fwd_tec']:.2f} TECU")

    # ── Re-compute ISR forward TEC using IRI-informed Epstein topside ─────────
    # IRI data is now available for each match; recompute occ_fwd_tec so the
    # topside shape comes from IRI rather than the ISR profile alone.
    print("  Recomputing ISR forward TEC with IRI-informed topside …")
    for m in matched:
        iri_ne_m  = m.get("iri_ne")
        iri_alt_m = m.get("iri_alt")
        if iri_ne_m is None or np.all(np.isnan(iri_ne_m)):
            continue   # no IRI data; keep the original value
        m["occ_fwd_tec"] = _forward_slant_tec_from_isr(
            leo_ecef_km  = np.asarray(m["_leo_tm"]),
            gnss_ecef_km = np.asarray(m["_gnss_tm"]),
            isr_alt_km   = np.asarray(m["isr_alt_km"]),
            isr_ne       = np.asarray(m["isr_ne"]),
            iri_alt_km   = iri_alt_m,
            iri_ne       = iri_ne_m,
        )
        print(f"    {m['filename'][:50]}  ISR fwd TEC (IRI topside) = {m['occ_fwd_tec']:.2f} TECU")

    # ── Pre-compute arc data (needed by both Row 1 bounds and Row 2 plots) ─────
    import matplotlib.gridspec as gridspec

    from Abel_Inverter import run_abel_inversion

    n_occ = len(matched)
    print(f"\n  Computing full TEC arcs for {n_occ} occultations …")

    for occ_idx, m in enumerate(matched):
        m["_arc_data"] = None   # sentinel; filled below on success
        m["abel"]      = None   # Abel inversion result; filled below on success
        try:
            pdata_full = parse_podTc2_nc_file(m["full_path"])
        except Exception as exc:
            print(f"    [warn] arc parse failed: {exc}")
            continue

        tec_full  = np.asarray(pdata_full["TEC_podTc2"])
        leo_full  = np.asarray(pdata_full["LEO"])    # (3, n_t)
        gnss_full = np.asarray(pdata_full["GNSS"])   # (3, n_t)

        # Absolute unix time for each QC'd sample
        try:
            with netCDF4.Dataset(m["full_path"], "r") as nc_raw2:
                time_raw0 = float(nc_raw2.variables["time"][0])
            unix_full = _gps_to_unix(time_raw0) + np.asarray(pdata_full["time"])
        except Exception:
            unix_full = np.arange(len(tec_full), dtype=float)

        # Tangent-point altitude for every sample via single batch call
        # rayTangent expects (3, n) arrays and returns (tangent_point, p, alt_m)
        # where alt_m is altitude above the WGS84 ellipsoid in metres.
        n_samples = len(tec_full)
        try:
            _, _, tang_alt_m = rayTangent(leo_full, gnss_full, units="km")
            tang_alt = np.asarray(tang_alt_m) / 1000.0   # m → km
        except Exception as exc:
            print(f"    [warn] rayTangent batch failed: {exc}")
            tang_alt = np.full(n_samples, np.nan)

        # Forward TEC at subsampled points along the arc
        stride    = max(1, n_samples // 60)
        idx_sub   = np.arange(0, n_samples, stride)
        tang_sub  = tang_alt[idx_sub]
        isr_alt_m = np.asarray(m["isr_alt_km"])
        isr_ne_m  = np.asarray(m["isr_ne"])
        iri_ne_m  = m.get("iri_ne")
        iri_alt_m = m.get("iri_alt")

        isr_fwd_arc = np.full(len(idx_sub), np.nan)
        iri_fwd_arc = np.full(len(idx_sub), np.nan)

        use_iri_topside = iri_ne_m is not None and not np.all(np.isnan(iri_ne_m))
        for j, k in enumerate(idx_sub):
            leo_k  = leo_full[:, k]
            gnss_k = gnss_full[:, k]
            isr_fwd_arc[j] = _forward_slant_tec_from_isr(
                leo_k, gnss_k, isr_alt_m, isr_ne_m, n_segments=150,
                iri_alt_km=iri_alt_m if use_iri_topside else None,
                iri_ne=iri_ne_m if use_iri_topside else None,
            )
            if use_iri_topside:
                iri_fwd_arc[j] = _forward_slant_tec_from_isr(
                    leo_k, gnss_k, iri_alt_m, iri_ne_m, n_segments=150,
                )

        # Store everything for Row 2 plotting and Row 1 bounds
        fin_isr = np.isfinite(isr_fwd_arc)
        fin_iri = np.isfinite(iri_fwd_arc)
        m["_arc_data"] = {
            "tec_full":     tec_full,
            "tang_alt":     tang_alt,
            "unix_full":    unix_full,
            "tec_sub":      tec_full[idx_sub],
            "tang_sub":     tang_sub,
            "isr_fwd_arc":  isr_fwd_arc,
            "iri_fwd_arc":  iri_fwd_arc,
            "isr_fwd_min":  float(np.nanmin(isr_fwd_arc[fin_isr])) if fin_isr.any() else np.nan,
            "isr_fwd_max":  float(np.nanmax(isr_fwd_arc[fin_isr])) if fin_isr.any() else np.nan,
            "iri_fwd_min":  float(np.nanmin(iri_fwd_arc[fin_iri])) if fin_iri.any() else np.nan,
            "iri_fwd_max":  float(np.nanmax(iri_fwd_arc[fin_iri])) if fin_iri.any() else np.nan,
        }
        # Abel inversion on the full (unmasked) arc
        try:
            abel = run_abel_inversion(pdata_full)
            if abel is not None and len(abel.get("Ne", [])) > 0:
                m["abel"] = abel
                _abel_nm, _abel_hm = extract_robust_f2_peak(abel["Ne"], abel["alt_km"])
                print(f"    [Abel] {m['spacecraft']} {m['prn_label']}: "
                      f"NmF2={_abel_nm:.2e} m⁻³  hmF2={_abel_hm:.1f} km")
        except Exception as _exc_a:
            print(f"    [Abel warn] {m['spacecraft']} {m['prn_label']}: {_exc_a}")

        print(f"    Arc {occ_idx+1}/{n_occ}: {m['spacecraft']}  "
              f"{len(idx_sub)} pts  ISR [{m['_arc_data']['isr_fwd_min']:.1f}–"
              f"{m['_arc_data']['isr_fwd_max']:.1f}] TECU")

    # ── Build figure ──────────────────────────────────────────────────────────
    dist_vals  = np.array([m["dist_km"] for m in matched])
    dist_norm  = mcolors.Normalize(vmin=0.0, vmax=min(max_match_dist_km, dist_vals.max() * 1.05))
    dist_cmap  = cm.plasma_r

    fig = plt.figure(figsize=(max(22, n_occ * 2.8), 22))
    fig.suptitle(
        "Occultation TEC vs. ISR / IRI Forward-Modeled TEC\n"
        "Millstone Hill ISR  |  DOY 153–154, 2025",
        fontsize=18, y=0.995,
    )

    outer_gs = gridspec.GridSpec(
        3, 1, figure=fig,
        hspace=0.38,
        height_ratios=[1.0, 1.0, 1.2],
        top=0.975, bottom=0.04, left=0.06, right=0.97,
    )

    # ── Row 1: TEC-max overview with min/max bounds ────────────────────────────
    ax1 = fig.add_subplot(outer_gs[0])

    sc = ax1.scatter(
        [m["occ_hour_utc"] + (m["doy"] - 153) * 24 for m in matched],
        [m["occ_tec_max"]  for m in matched],
        c=[m["dist_km"] for m in matched],
        cmap=dist_cmap, norm=dist_norm,
        s=160, zorder=5, edgecolors="black", linewidths=1.0,
        label="Occultation truth TEC max",
    )
    for m in matched:
        x_ann = m["occ_hour_utc"] + (m["doy"] - 153) * 24
        ax1.annotate(
            f"{m['spacecraft']}\n{m['prn_label']}\n{m['dist_km']:.0f} km",
            xy=(x_ann, m["occ_tec_max"]),
            fontsize=9, ha="center", va="bottom",
            xytext=(0, 5), textcoords="offset points", alpha=0.8,
        )

    fwd_valid = [m for m in matched if np.isfinite(m["occ_fwd_tec"])]
    if fwd_valid:
        ax1.plot(
            [m["occ_hour_utc"] + (m["doy"] - 153) * 24 for m in fwd_valid],
            [m["occ_fwd_tec"] for m in fwd_valid],
            marker="D", ms=13, lw=0, ls="none",
            color="crimson", zorder=4,
            label="ISR EDP forward-modeled TEC (slant)",
        )

    iri_valid = [m for m in matched if np.isfinite(m.get("iri_fwd_tec", np.nan))]
    if iri_valid:
        ax1.plot(
            [m["occ_hour_utc"] + (m["doy"] - 153) * 24 for m in iri_valid],
            [m["iri_fwd_tec"] for m in iri_valid],
            marker="s", ms=13, lw=0, ls="none",
            color="forestgreen", zorder=4,
            label="IRI-2020 @ MH forward-modeled TEC (slant)",
        )

    # x-axis limits tight around the actual data with 1-hour padding each side
    all_x = [m["occ_hour_utc"] + (m["doy"] - 153) * 24 for m in matched]
    x_lo  = max(0.0, min(all_x) - 1.0)
    x_hi  = min(48.0, max(all_x) + 1.0)
    # Snap to 3-hour grid boundaries
    x_lo = np.floor(x_lo / 3) * 3
    x_hi = np.ceil(x_hi  / 3) * 3
    tick_hours = [h for h in range(0, 49, 3) if x_lo - 0.1 <= h <= x_hi + 0.1]
    tick_labels = []
    for h in tick_hours:
        doy_off = 153 + h // 24
        hh      = h % 24
        tick_labels.append(f"{hh:02d}:00\nDOY {doy_off}")
    ax1.set_xticks(tick_hours)
    ax1.set_xticklabels(tick_labels, fontsize=11)
    ax1.set_xlim(x_lo, x_hi)
    ax1.set_xlabel("UTC Time", fontsize=13)
    ax1.set_ylabel("TEC (TECU)", fontsize=13)
    ax1.grid(True, alpha=0.3, ls=":")

    # ── Ionosonde foF2 on right y-axis ────────────────────────────────────────
    _iono_valid = [m for m in matched if np.isfinite(m["iono_foF2"])]
    if _iono_valid and ionosonde_df is not None:
        ax1r = ax1.twinx()
        # Full ionosonde time series clipped to the x-axis window
        _iono_plot = ionosonde_df.dropna(subset=["foF2", "Time"]).copy()
        _iono_plot["hour_plot"] = (
            _iono_plot["Time"].astype("int64") / 1e9 % 86400
        ) / 3600.0
        # Shift DOY-154 records by +24 h (same convention as matched x-axis)
        _ref_unix0 = pd.Timestamp("2025-06-02").timestamp()   # DOY 153
        _iono_plot["hour_plot"] = (
            (_iono_plot["Time"].astype("int64") / 1e9 - _ref_unix0) / 3600.0
        )
        _mask = (_iono_plot["hour_plot"] >= x_lo) & (_iono_plot["hour_plot"] <= x_hi)
        _ip = _iono_plot[_mask]
        if not _ip.empty:
            ax1r.plot(
                _ip["hour_plot"], _ip["foF2"],
                color="darkorchid", lw=2.2, ls="-", alpha=0.75,
                label="Ionosonde foF2 (MHz)",
            )
        # Also mark the per-occultation matched value
        ax1r.scatter(
            [m["occ_hour_utc"] + (m["doy"] - 153) * 24 for m in _iono_valid],
            [m["iono_foF2"] for m in _iono_valid],
            marker="P", s=130, color="darkorchid", zorder=6,
            edgecolors="black", linewidths=1.0,
            label="Ionosonde foF2 @ match time",
        )
        ax1r.set_ylabel("foF2 (MHz)", fontsize=13, color="darkorchid")
        ax1r.tick_params(axis="y", labelcolor="darkorchid", labelsize=11)
        ax1r.legend(fontsize=10, loc="upper left", framealpha=0.85)
        ax1r.set_xlim(x_lo, x_hi)

    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=11, framealpha=0.9, loc="upper right")
    _title_extra = " + ionosonde foF2" if _iono_valid else ""
    ax1.set_title(
        f"TEC-max vs. time  (scatter = truth, diamonds = ISR fwd, squares = IRI fwd{_title_extra})",
        fontsize=12,
    )
    cbar1 = fig.colorbar(sc, ax=ax1, pad=0.01, fraction=0.018)
    cbar1.set_label("Dist TEC-max → MH (km)", fontsize=11)
    cbar1.ax.tick_params(labelsize=10)

    # ── Row 2: sTEC vs. tangent-point altitude per occultation ────────────────
    inner_gs2 = gridspec.GridSpecFromSubplotSpec(
        1, n_occ, subplot_spec=outer_gs[1], wspace=0.35,
    )

    for col_idx, m in enumerate(matched):
        ax2 = fig.add_subplot(inner_gs2[col_idx])
        ax2.set_title(
            f"{m['spacecraft']} · {m['prn_label']}\n{(m['occ_unix_tecmax'] % 86400)/3600:.2f} UTC",
            fontsize=10,
        )

        arc = m.get("_arc_data")
        if arc is None:
            ax2.text(0.5, 0.5, "no data", ha="center", va="center",
                     fontsize=10, transform=ax2.transAxes)
        else:
            tec_full = arc["tec_full"]
            tang_alt = arc["tang_alt"]
            tang_sub = arc["tang_sub"]
            isr_fwd  = arc["isr_fwd_arc"]
            iri_fwd  = arc["iri_fwd_arc"]

            # Measured sTEC vs tangent altitude
            fin_meas = np.isfinite(tang_alt) & np.isfinite(tec_full)
            ax2.plot(tec_full[fin_meas], tang_alt[fin_meas],
                     color="steelblue", lw=2.0, label="Meas.")

            fin_isr = np.isfinite(isr_fwd) & np.isfinite(tang_sub)
            if fin_isr.any():
                ax2.plot(isr_fwd[fin_isr], tang_sub[fin_isr],
                         color="crimson", lw=2.0, ls="--", label="ISR fwd")

                # Mark where tangent altitude crosses the ISR measurement bounds
                isr_alt_min = float(np.asarray(m["isr_alt_km"]).min())
                isr_alt_max = float(np.asarray(m["isr_alt_km"]).max())
                t_sub_fin   = tang_sub[fin_isr]
                f_sub_fin   = isr_fwd[fin_isr]
                for bound_alt, marker, label in [
                    (isr_alt_min, "v", f"ISR bot {isr_alt_min:.0f} km"),
                    (isr_alt_max, "^", f"ISR top {isr_alt_max:.0f} km"),
                ]:
                    idx_near = int(np.argmin(np.abs(t_sub_fin - bound_alt)))
                    ax2.plot(f_sub_fin[idx_near], t_sub_fin[idx_near],
                             marker=marker, ms=10, color="crimson",
                             mec="black", mew=1.0, zorder=6, label=label)

            fin_iri = np.isfinite(iri_fwd) & np.isfinite(tang_sub)
            if fin_iri.any():
                ax2.plot(iri_fwd[fin_iri], tang_sub[fin_iri],
                         color="forestgreen", lw=2.0, ls="-.", label="IRI fwd")

        ax2.set_xlabel("Slant TEC (TECU)", fontsize=10)
        if col_idx == 0:
            ax2.set_ylabel("Tangent alt. (km)", fontsize=11)
            ax2.legend(fontsize=9, loc="upper right")
        ax2.xaxis.set_tick_params(labelsize=9)
        ax2.yaxis.set_tick_params(labelsize=9)
        ax2.grid(True, alpha=0.25, ls=":")

    # ── Row 3: EDP profiles per occultation ───────────────────────────────────
    inner_gs3 = gridspec.GridSpecFromSubplotSpec(
        1, n_occ, subplot_spec=outer_gs[2], wspace=0.40,
    )

    # KF profile lookup by nearest time within 30 min
    kf_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if kf_profiles:
        kf_times = np.array([kp["unix_sec"] for kp in kf_profiles])
        for m in matched:
            dt_kf = np.abs(kf_times - m["occ_unix_tecmax"])
            best_kf = int(np.argmin(dt_kf))
            if dt_kf[best_kf] < 1800.0:
                kf_lookup[m["filename"]] = (
                    kf_profiles[best_kf]["alt_km"],
                    kf_profiles[best_kf]["ne"],
                )

    for col_idx, m in enumerate(matched):
        ax3 = fig.add_subplot(inner_gs3[col_idx])
        ax3.set_title(f"{m['spacecraft']} · {m['prn_label']}\nDOY {m['doy']}", fontsize=10)

        isr_alt = np.asarray(m["isr_alt_km"])
        isr_ne  = np.asarray(m["isr_ne"])

        ax3.plot(isr_ne / 1e10, isr_alt, color="crimson", lw=2.2, label="ISR meas.")

        if len(isr_alt) >= 2:
            alt_ext, ne_ext = _extend_isr_profile_topside(
                isr_alt, isr_ne, topside_ceil_km=800.0, topside_step_km=2.0,
                iri_alt_km=m.get("iri_alt"), iri_ne=m.get("iri_ne"),
            )
            tail_start = float(isr_alt[-2])
            mask_tail  = alt_ext > tail_start
            if mask_tail.any():
                ax3.plot(ne_ext[mask_tail] / 1e10, alt_ext[mask_tail],
                         color="crimson", lw=2.0, ls=":", label="ISR topside (Epstein/IRI)")

        iri_ne_m  = m.get("iri_ne")
        iri_alt_m = m.get("iri_alt")
        if iri_ne_m is not None:
            iri_ne_arr  = np.asarray(iri_ne_m)
            iri_alt_arr = np.asarray(iri_alt_m)
            show_iri = (iri_alt_arr >= 80.0) & (iri_alt_arr <= 800.0) & (iri_ne_arr > 0)
            if show_iri.any():
                ax3.plot(iri_ne_arr[show_iri] / 1e10, iri_alt_arr[show_iri],
                         color="forestgreen", lw=2.0, ls="--", label="IRI-2020")

        # Abel inversion Ne profile
        _abel = m.get("abel")
        if _abel is not None:
            _a_alt = np.asarray(_abel["alt_km"])
            _a_ne  = np.asarray(_abel["Ne"])
            _show_a = np.isfinite(_a_alt) & np.isfinite(_a_ne) & (_a_alt >= 80.0) & (_a_alt <= 800.0)
            if _show_a.any():
                ax3.plot(_a_ne[_show_a] / 1e10, _a_alt[_show_a],
                         color="darkorange", lw=2.0, ls="-", label="Abel inv.")
            # Also plot multi-layer result if available
            _a_alt_m = np.asarray(_abel.get("alt_km_m", []))
            _a_ne_m  = np.asarray(_abel.get("Ne_m", []))
            if len(_a_alt_m) > 0 and len(_a_ne_m) > 0:
                _show_m = (np.isfinite(_a_alt_m) & np.isfinite(_a_ne_m)
                           & (_a_alt_m >= 80.0) & (_a_alt_m <= 800.0))
                if _show_m.any():
                    ax3.plot(_a_ne_m[_show_m] / 1e10, _a_alt_m[_show_m],
                             color="darkorange", lw=1.8, ls=":", label="Abel inv. (multi-layer)")

        # KF posterior: prefer internally-computed profile (from _process_verif_group),
        # fall back to externally-supplied kf_profiles lookup.
        _kf_alt_m = m.get("kf_alt")
        _kf_ne_m  = m.get("kf_ne")
        if _kf_alt_m is not None and _kf_ne_m is not None:
            ax3.plot(np.asarray(_kf_ne_m) / 1e10, np.asarray(_kf_alt_m),
                     color="steelblue", lw=2.2, ls="-", label="KF posterior (single occ)")
        else:
            kf_pair = kf_lookup.get(m["filename"])
            if kf_pair is not None:
                kf_alt, kf_ne = kf_pair
                ax3.plot(np.asarray(kf_ne) / 1e10, np.asarray(kf_alt),
                         color="steelblue", lw=2.0, ls="-.", label="KF posterior")

        # Ionosonde NmF2 marker: horizontal line at hmF2 (or 300 km) with the
        # NmF2 value from the closest-in-time ionosonde record.
        if np.isfinite(m.get("iono_NmF2", np.nan)):
            _nm_val = m["iono_NmF2"] / 1e10
            _hm_val = m["iono_hmF2"] if np.isfinite(m.get("iono_hmF2", np.nan)) else 300.0
            ax3.axhline(_hm_val, color="darkorchid", lw=1.4, ls=":", alpha=0.6)
            ax3.plot(_nm_val, _hm_val,
                     marker="*", ms=14, color="darkorchid", zorder=7,
                     mec="black", mew=1.0, label=f"Iono NmF2 ({m['iono_foF2']:.1f} MHz)")

        ax3.set_ylim(0, 800)
        ax3.set_xlabel("Ne (×10¹⁰ m⁻³)", fontsize=10)
        if col_idx == 0:
            ax3.set_ylabel("Altitude (km)", fontsize=11)
            ax3.legend(fontsize=9, loc="upper right")
        ax3.xaxis.set_tick_params(labelsize=9)
        ax3.yaxis.set_tick_params(labelsize=9)
        ax3.grid(True, alpha=0.25, ls=":")

    plot_path = os.path.join(save_dir, "occ_tec_max_vs_isr_forward_tec.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Multi-panel plot saved → {plot_path}")

    # ── Scatter plot: forward-modeled TEC vs measured RO TEC ─────────────────
    # ISR fwd TEC (x) vs RO TEC max (y)
    _sc_pairs_isr = [(m["occ_fwd_tec"], m["occ_tec_max"]) for m in matched
                     if np.isfinite(m["occ_tec_max"]) and np.isfinite(m["occ_fwd_tec"])]
    # ISR fwd TEC (x) vs IRI fwd TEC (y)
    _sc_pairs_iri = [(m["occ_fwd_tec"], m["iri_fwd_tec"]) for m in matched
                     if np.isfinite(m["occ_fwd_tec"]) and np.isfinite(m.get("iri_fwd_tec", np.nan))]

    if _sc_pairs_isr or _sc_pairs_iri:
        fig_sc, ax_sc = plt.subplots(figsize=(7, 6))

        all_vals: list[float] = []
        if _sc_pairs_isr:
            x_isr, y_isr = np.array(_sc_pairs_isr).T
            ax_sc.scatter(x_isr, y_isr, color="crimson", s=80, zorder=5,
                          edgecolors="black", linewidths=0.8,
                          label="ISR EDP forward TEC vs RO TEC")
            all_vals.extend(x_isr.tolist() + y_isr.tolist())
            # Fit line through zero: slope = Σ(x·y) / Σ(x²)
            _slope_isr = float(np.dot(x_isr, y_isr) / np.dot(x_isr, x_isr))
        if _sc_pairs_iri:
            x_iri, y_iri = np.array(_sc_pairs_iri).T
            ax_sc.scatter(x_iri, y_iri, color="forestgreen", s=80, zorder=5,
                          marker="s", edgecolors="black", linewidths=0.8,
                          label="IRI-2020 forward TEC vs ISR forward TEC")
            all_vals.extend(x_iri.tolist() + y_iri.tolist())

        # 1:1 reference line and axis limits
        _v_lo = 0.0
        _v_hi = max(all_vals) * 1.1 if all_vals else 1.0
        _fit_x = np.array([_v_lo, _v_hi])
        ax_sc.plot(_fit_x, _fit_x,
                   color="black", lw=1.5, ls="--", zorder=3, label="1:1")

        # Fit line through zero for ISR fwd vs RO TEC
        if _sc_pairs_isr:
            ax_sc.plot(_fit_x, _slope_isr * _fit_x,
                       color="crimson", lw=1.5, ls=":", zorder=4,
                       label=f"Fit (slope={_slope_isr:.2f})")

        ax_sc.set_xlim(_v_lo, _v_hi)
        ax_sc.set_ylim(_v_lo, _v_hi)
        ax_sc.set_aspect("equal", adjustable="box")
        ax_sc.set_xlabel("ISR EDP Forward-Modeled TEC (TECU)", fontsize=13)
        ax_sc.set_ylabel("TEC (TECU)", fontsize=13)
        ax_sc.set_title(
            "Correlation: ISR Forward-Modeled TEC vs. RO / IRI TEC\n"
            "Millstone Hill ISR  |  DOY 153–154, 2025",
            fontsize=12,
        )
        ax_sc.legend(fontsize=11, framealpha=0.9)
        ax_sc.grid(True, alpha=0.3, ls=":")

        scatter_path = os.path.join(save_dir, "occ_tec_max_vs_forward_tec_scatter.png")
        fig_sc.tight_layout()
        fig_sc.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig_sc)
        print(f"  Scatter correlation plot saved → {scatter_path}")

    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_occ_tec_max_vs_isr_forward_tec(
        isr_files=[
            "./DataFiles/EDPS/mlh250602m.002.nc",
            "./DataFiles/EDPS/mlh250603m.002.nc",
        ],
        base_paths={
            153: "/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/2025.153/",
            154: "/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/2025.154/",
        },
        yyyy=2025,
        save_dir="./Figures/Verification/",
        ionosonde_file="./Data/IONOSONDE_short.txt",    # set to None to disable
        # alt_grid defaults to 55 log-spaced levels 60–800 km.
        # global_edp_cache defaults to None → built automatically per DOY.
        # Pass a pre-built cache to skip the IRI build step, e.g.:
        #   global_edp_cache={153: cache153, 154: cache154}
        kf_config={"measurement_err": 1.0, "relaxation": 0.99, "topside_follow_f2": True},
        num_workers=12,
        global_edp_data_dir="./Data/Global_EDPS/",
    )
    # demo_verification_main()
