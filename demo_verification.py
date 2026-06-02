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
    result, save_dir, group_key, *, suffix="", mode_label="Sequential KF"
):
    isr_arg = None
    if suffix == "_joint" and _isr_profiles_for_patch:
        win = result.get("time_window", "")
        try:
            hhmm  = win.split("_")[-1]
            h_win = int(hhmm[:2]) + int(hhmm[2:]) / 60.0
            h_mid = h_win + WINDOW_MINUTES / 120.0
            isr_arg = [
                p for p in _isr_profiles_for_patch
                if h_win <= p["hour_utc"] < h_win + WINDOW_MINUTES / 60.0
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
    return _orig_plot_group(
        result, save_dir, group_key,
        suffix=suffix, mode_label=mode_label,
        isr_profiles=isr_arg,
        isr_site=(ISR_LON_W, ISR_LAT) if isr_arg else None,
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


def assign_single_region(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Override the grouping so that every occultation in the filtered metadata
    lands in ONE region key — the verification patch — regardless of the
    standard lat/lon bin grid.  This forces all occultations in the window
    to be processed together in a single KF update covering the patch.
    """
    meta = meta.copy()
    meta["region"]     = "VERIF_MH"
    meta["time_window"] = meta["date"].apply(time_window_key)
    meta["group_key"]   = meta["time_window"] + "__VERIF_MH"
    return meta


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

    cmap_isr = cm.viridis
    norm_isr = mcolors.Normalize(vmin=0, vmax=24)
    ne_fmt   = ScalarFormatter(useMathText=True)
    ne_fmt.set_powerlimits((-2, 2))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    ax_prof, ax_nm, ax_bias = axes
    fig.suptitle(
        f"Millstone Hill ISR Verification\n{group_key}",
        fontsize=13, y=1.01,
    )

    # ── Panel 1: EDP profile overlay ─────────────────────────────────────────
    for prof in isr_profiles:
        col = cmap_isr(norm_isr(prof["hour_utc"]))
        # Convert float hours to seconds, then format as HH:MM
        time_str = time.strftime('%H:%M', time.gmtime(prof["hour_utc"] * 3600))
        
        # Use the formatted string in your label
        ax_prof.plot(prof["ne"], prof["alt_km"], color=col, lw=1.2, alpha=1, 
                     label=f'MH ISR: {time_str}')


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

    # ── Panel 2: NmF2 scatter ─────────────────────────────────────────────────
    isr_nm = np.array([p["nm_f2"] for p in isr_profiles if not np.isnan(p["nm_f2"])])
    isr_hm = np.array([p["hm_f2"] for p in isr_profiles if not np.isnan(p["hm_f2"])])

    if len(isr_nm) > 0:
        nm_mean = float(np.nanmean(isr_nm))
        hm_mean = float(np.nanmean(isr_hm))

        # ISR truth cloud
        ax_nm.scatter(isr_nm, isr_hm,
                      c=[p["hour_utc"] for p in isr_profiles if not np.isnan(p["nm_f2"])],
                      cmap=cmap_isr, norm=norm_isr,
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

    if bias_pr_on_grid:
        mean_bias_pr = np.nanmean(np.vstack(bias_pr_on_grid), axis=0)
        mean_bias_po = np.nanmean(np.vstack(bias_po_on_grid), axis=0)
        valid_alt = ~np.isnan(mean_bias_pr) & ~np.isnan(mean_bias_po)
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
        po_3d  = res.get("joint_post_edp_3d") or res.get("post_edp_3d")
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
) -> None:
    """
    Regional map showing the 30°×60° verification patch and all occultation
    tangent points within it.
    """
    os.makedirs(save_dir, exist_ok=True)
    proj = ccrs.LambertConformal(
        central_longitude=ISR_LON_W,
        central_latitude=ISR_LAT,
    )
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": proj})
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
            color="limegreen", lw=2.2, ls="-", zorder=4,
            label="Verification region")

    # Occultation tangent points
    sc = ax.scatter(
        meta["lon"].values, meta["lat"].values,
        transform=ccrs.Geodetic(),
        s=50, c="steelblue", edgecolors="black", linewidths=0.5,
        zorder=5, label=f"Occultations ({len(meta)})",
    )

    # Millstone Hill ISR
    ax.plot(ISR_LON_W, ISR_LAT, transform=ccrs.Geodetic(),
            marker="^", ms=14, color="crimson", mec="black", mew=1.2,
            zorder=7, label="Millstone Hill ISR")

    ax.set_title(
        f"Verification Region Centred on Millstone Hill ISR\n"
        f"30°lat × 60°lon  |  {len(meta)} occultations",
        fontsize=12,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)

    path = os.path.join(save_dir, "verif_region_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Region map saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# §F  Thin wrapper around process_group that enforces the fixed bounding box
# ─────────────────────────────────────────────────────────────────────────────

def _process_verif_group(
    group_key:        str,
    group_meta:       pd.DataFrame,
    alt_grid:         np.ndarray,
    global_edp_cache: dict,
    measurement_err:  float = 10.0,
    relaxation:       float = 0.99,
    generate_plots:   bool  = True,
    save_dir:         str   = "./Figures/Verification/",
) -> dict:
    """
    Thin wrapper around process_group() for the Millstone Hill verification
    region.  Sequential KF and its associated plots are disabled — only the
    joint (batch) update and the joint summary figure are produced.
    """
    return process_group(
        group_key        = group_key,
        group_meta       = group_meta,
        alt_grid         = alt_grid,
        global_edp_cache = global_edp_cache,
        measurement_err  = measurement_err,
        relaxation       = relaxation,
        generate_plots   = generate_plots,
        save_dir         = save_dir,
        run_sequential   = False,
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
    DOY  = 153
    YYYY = 2025
    base_path = (
        f"/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/"
        f"{YYYY}.{DOY}/"
    )
    alt_grid    = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)
    TYPE        = "log"
    save_dir    = "./Figures/Verification/"
    num_workers = 12
    kf_config   = {"measurement_err": 1.0, "relaxation": 0.99}

    # ISR netCDF files (Millstone Hill) — list as many days as needed.
    # Files should cover the same UTC day as DOY above.
    # Example: downloaded from Madrigal (http://millstonehill.haystack.mit.edu)
    isr_files = [
        "./DataFiles/EDPS/mlh250602m.002.nc",
        # "./DataFiles/EDPS/mlh250603m.002.nc",
    ]
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

    meta_verif = assign_single_region(meta_verif)

    # ── Step 2: Region map ────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    plot_verif_region_map(meta_verif, save_dir)

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

    # ── Step 5: Process each 30-minute window ────────────────────────────────
    groups     = meta_verif.groupby("group_key", sort=True)
    group_keys = list(groups.groups.keys())
    print(f"\nProcessing {len(group_keys)} time-window group(s) …")

    all_results: list[dict] = []
    for g_idx, gk in enumerate(group_keys):
        print(f"\n[{g_idx + 1}/{len(group_keys)}]", end="")
        gm  = groups.get_group(gk)
        res = _process_verif_group(
            group_key        = gk,
            group_meta       = gm,
            alt_grid         = alt_grid,
            global_edp_cache = global_edp_cache,
            generate_plots   = True,
            save_dir         = save_dir,
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

                # Select ISR sweeps: prefer those inside the 30-min window;
                # if none exist, fall back to the single sweep closest in time.
                win = res["time_window"]          # "YYYY-MM-DD_HHMM"
                try:
                    hhmm   = win.split("_")[-1]
                    h_win  = int(hhmm[:2]) + int(hhmm[2:]) / 60.0
                    h_mid  = h_win + WINDOW_MINUTES / 120.0   # window centre
                    isr_win = [
                        p for p in isr_profiles
                        if h_win <= p["hour_utc"] < h_win + WINDOW_MINUTES / 60.0
                    ]
                    if not isr_win:
                        # Pick the single sweep whose hour_utc is nearest to the
                        # window centre (wrapping midnight via modular distance).
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
            po_mh  = np.asarray(
                res.get("joint_post_edp_3d") or res["post_edp_3d"]
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
if __name__ == "__main__":
    demo_verification_main()
