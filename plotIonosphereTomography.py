#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
"""
plotIonosphereTomography.py

Consolidated plotting functions used by demo_isr_da_comparison.py, relocated
here (verbatim, save for two disambiguating renames) from demo_group.py and
demo_compare_kf_enkf.py so demo_isr_da_comparison.py depends on fewer files.

This module must NEVER import demo_group.py, demo_compare_kf_enkf.py,
demo_verification.py, demo_isr_da_comparison.py, or demo_isr_initial_conditions.py
at module top level — those files import back from here (directly or
transitively), and a top-level cycle would break on a fresh interpreter.
Every name needed from those five files is imported locally inside the
specific function body that uses it.
"""
from __future__ import annotations

import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree

from demo import extract_robust_f2_peak

# ─────────────────────────────────────────────────────────────────────────────
# Constants (moved from demo_isr_da_comparison.py)
# ─────────────────────────────────────────────────────────────────────────────

ISR_MIN_VALID_GATES = 5

# Arc-innovation diagnostic (Panel A / the per-arc bar chart) gets cluttered
# for ro_igs/igs_only once IGS ground-station arcs are mixed in with (or
# stand in for) the RO occultations -- IGS stations contribute many more
# arcs per window than RO. Cap how many IGS arcs are drawn, keeping ALL RO
# arcs and only a representative subset of IGS arcs (see
# _select_representative_arc_indices below).
ARC_INNOV_MAX_IGS_ARCS = 20

# Per-group processing can write dozens of figures (6 obs_mode/filter_type
# configs x several plot types x however many ISR scans fall in the window),
# each of which used to print its own "<figure> saved -> path" line -- set
# PLOT_VERBOSE_SAVE=1 in the environment to restore that per-figure logging;
# by default only the per-group start/end markers and [warn]/[error]
# diagnostics are printed.
VERBOSE_SAVE_PRINTS = os.environ.get("PLOT_VERBOSE_SAVE", "0") == "1"


def _print_saved(msg: str) -> None:
    if VERBOSE_SAVE_PRINTS:
        print(msg)

# ISR "kindat" -> (legend label, linestyle) for EDP dicts produced by
# demo_esr_isr.py's load_edps(). MAD6400 ("GUISDAP params") is the full
# spectral fit (corrected ne/ti/tr); MAD6300 ("GUISDAP pp resolution 0") is
# the quick-look power profile (Ne assuming Te/Ti=1, no ti/tr). Both can be
# co-located at the same site/time window, so spaghetti/overlay plots must
# not blend them into one undifferentiated "ISR truth" line style.
_ISR_KINDAT_STYLE = {
    "6400":      ("ISR fitted (6400)",        "-"),
    "6300":      ("ISR power profile (6300)", "--"),
    "jro":       ("ISR fitted (JRO)",         "-"),
    "simulated": ("Simulated truth EDP",      "-"),
}
_ISR_KINDAT_FALLBACK = ("ISR truth (kindat unknown — reload cache)", ":")


def _isr_kindat_style(kindat: str | None) -> tuple[str, str]:
    return _ISR_KINDAT_STYLE.get(kindat, _ISR_KINDAT_FALLBACK)


# ── Plasma-frequency / Ne conversion ─────────────────────────────────────────
# Ne [m⁻³] = _NE_TO_FP_SCALE × fₚ² [MHz²]   (standard ionospheric relation)
_NE_TO_FP_SCALE = 1.24e10
_FP_BAND_MHZ    = 0.5          # ±0.5 MHz truth-band half-width


def _ne_to_fp(ne: np.ndarray) -> np.ndarray:
    """Electron density (m⁻³) → plasma frequency (MHz)."""
    return np.sqrt(np.maximum(np.asarray(ne, float), 0.0) / _NE_TO_FP_SCALE)


def _fp_to_ne(fp: np.ndarray) -> np.ndarray:
    """Plasma frequency (MHz) → electron density (m⁻³)."""
    return _NE_TO_FP_SCALE * np.asarray(fp, float) ** 2


def _truth_fp_band(ne_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Ne_lo, Ne_hi) bounding a ±_FP_BAND_MHZ band around ne_truth."""
    fp = _ne_to_fp(ne_truth)
    return _fp_to_ne(np.maximum(fp - _FP_BAND_MHZ, 0.0)), _fp_to_ne(fp + _FP_BAND_MHZ)


_FILTER_LABELS = {"gridded_kf": "KF", "parametric_ekf": "EKF"}
_CONFIG_STYLES = {
    ("ro_only",  "gridded_kf"):     ("tab:blue",   "-"),
    ("ro_only",  "parametric_ekf"): ("tab:blue",   "--"),
    ("igs_only", "gridded_kf"):     ("tab:orange", "-"),
    ("igs_only", "parametric_ekf"): ("tab:orange", "--"),
    ("ro_igs",   "gridded_kf"):     ("tab:green",  "-"),
    ("ro_igs",   "parametric_ekf"): ("tab:green",  "--"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Moved from demo_group.py
# ─────────────────────────────────────────────────────────────────────────────

def _draw_roi_boundary(ax, region_key: str) -> None:
    """
    Draw the geographic region of interest on a cartopy axes.

    Mid-latitude bins  : rectangle formed by four PlateCarree line segments.
    Polar caps         : latitude boundary circle at ±POLAR_LAT_THRESHOLD.
    """
    from demo_group import POLAR_LAT_THRESHOLD, region_bounding_box

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
    from demo_group import region_bounding_box

    lat_min, lat_max, lon_min, lon_max = region_bounding_box(region_key)
    roi_clat = (lat_min + lat_max) / 2.0
    roi_clon = (lon_min + lon_max) / 2.0
    dlat = verts_geo[:, 1] - roi_clat
    dlon = verts_geo[:, 0] - roi_clon
    # if region_key == "POLAR_N":

    # if region_key == "POLAR_S":

    return int(np.argmin(dlat ** 2 + dlon ** 2))


def _plot_altitude_slices(
    result: dict,
    save_dir: str,
    group_key: str,
    *,
    suffix: str = "",
    altitudes_km: list[float] | None = None,
) -> str:
    """
    3×3 grid of globe plots showing ΔNe (posterior − prior) at fixed altitude
    slices across the EDP mesh.  Each panel is an orthographic globe centred on
    the group ROI, coloured with a diverging coolwarm map.

    Parameters
    ----------
    altitudes_km : list of 9 altitude values (km).  Defaults to 100:50:500.
    """
    from demo_group import _parse_time_window, _draw_terminator, _NOISE_SUFFIX

    if altitudes_km is None:
        altitudes_km = list(range(100, 501, 50))   # 9 values: 100…500 km

    alt_grid  = result["alt_grid"]
    prior_edp = result["prior_edp_3d"]
    post_edp  = result["post_edp_3d"]
    eds_occ   = result["eds_occ"]
    region    = result["region"]

    verts_geo = eds_occ.geolocation    # (n_geo, 2): col0=lon, col1=lat
    tris_geo  = eds_occ.mesh

    lats_c = result["lats"]
    lons_c = result["lons"]
    clon   = float(np.nanmean(lons_c)) if lons_c else 0.0
    clat   = float(np.nanmean(lats_c)) if lats_c else 0.0
    proj   = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)

    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(18, 14),
        subplot_kw={"projection": proj},
    )
    fig.suptitle(
        f"ΔNe (posterior − prior) at altitude slices — {result['time_window']}  |  {region}",
        fontsize=13,
    )

    # Pre-compute mean time for terminator — parse the time_window string directly
    try:
        mean_ts_sl = _parse_time_window(result["time_window"])
    except Exception:
        mean_ts_sl = None

    # Compute a symmetric colour limit shared across all panels for comparability.
    all_deltas = []
    for alt_km in altitudes_km:
        alt_idx = int(np.argmin(np.abs(alt_grid - alt_km)))
        all_deltas.append(post_edp[alt_idx, :] - prior_edp[alt_idx, :])
    global_max = float(np.nanpercentile(np.abs(np.concatenate(all_deltas)), 95))
    if global_max == 0:
        global_max = 1.0

    for panel_idx, (ax, alt_km) in enumerate(zip(axes.flat, altitudes_km)):
        alt_idx   = int(np.argmin(np.abs(alt_grid - alt_km)))
        true_alt  = float(alt_grid[alt_idx])
        delta     = post_edp[alt_idx, :] - prior_edp[alt_idx, :]

        ax.set_global()
        ax.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
        ax.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.4, edgecolor="gray")
        ax.gridlines(lw=0.2, alpha=0.3)

        try:
            tc = ax.tripcolor(
                verts_geo[:, 0], verts_geo[:, 1], tris_geo,
                delta,
                transform=ccrs.Geodetic(),
                cmap="coolwarm", shading="flat",
                vmin=-global_max, vmax=global_max,
                zorder=1,
            )
            cbar = fig.colorbar(tc, ax=ax, orientation="horizontal",
                                shrink=0.7, pad=0.03, fraction=0.04)
            cbar.set_label("ΔNe [m⁻³]", fontsize=7)
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.update_ticks()
        except Exception:
            pass

        if mean_ts_sl is not None:
            try:
                _draw_terminator(ax, mean_ts_sl, zorder=5)
            except Exception:
                pass
        _draw_roi_boundary(ax, region)
        ax.set_title(f"{true_alt:.0f} km", fontsize=10)

    os.makedirs(save_dir, exist_ok=True)
    safe_key  = group_key.replace("/", "_").replace(" ", "_")
    plot_path = os.path.join(save_dir, f"group_{safe_key}{_NOISE_SUFFIX}_alt_slices{suffix}.png")
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    _print_saved(f"  Saved altitude-slice plot → {plot_path}")
    return plot_path


def _plot_covariance_panels_tagged(
    result: dict,
    save_dir: str,
    group_key: str,
    *,
    hmF2_ref_km: float | None = None,
    tag: str = "",
) -> str:
    """
    Four-panel figure showing the EDP prior and posterior covariance structure.

    Layout (2 rows × 2 cols):
      Row 0 — Prior:     Alt-Alt correlation  |  Horizontal correlation at hmF2
      Row 1 — Posterior: Alt-Alt correlation  |  Horizontal correlation at hmF2

    The altitude-altitude panels average the grid block of P over all geo-point
    pairs and normalise to a Pearson correlation matrix.

    The horizontal panels fix one reference vertex (centre of the ROI) at the
    altitude nearest to hmF2 (or the supplied hmF2_ref_km) and plot the
    correlation of that state element with every other geo vertex at the same
    altitude, mapped onto an orthographic globe.

    Parameters
    ----------
    hmF2_ref_km : float or None
        Altitude (km) for the horizontal panel.  Defaults to the prior F2-peak
        altitude at the centre vertex.
    tag : str
        Optional short label (e.g. "kf") inserted into the output filename as
        "..._{tag}_covariance.png" instead of the bare "..._covariance.png".
        Default "" preserves the original filename for existing callers.
    """
    from demo_group import _NOISE_SUFFIX

    alt_grid  = result["alt_grid"]
    prior_P   = result["prior_P"]
    post_P    = result["post_P"]
    eds_occ   = result["eds_occ"]
    region    = result["region"]
    prior_edp = result["prior_edp_3d"]

    n_height  = len(alt_grid)
    verts_geo = eds_occ.geolocation      # (n_geo, 2): col0=lon, col1=lat
    n_geo     = verts_geo.shape[0]
    n_sv      = n_height * n_geo

    # Centre vertex for the horizontal slice reference
    centre_idx = _roi_centre_idx(verts_geo, region)

    # Choose hmF2 reference altitude
    if hmF2_ref_km is None:
        _, hmF2_ref_km = extract_robust_f2_peak(prior_edp[:, centre_idx], alt_grid)
        if np.isnan(hmF2_ref_km):
            hmF2_ref_km = float(alt_grid[n_height // 2])
    alt_ref_idx = int(np.argmin(np.abs(alt_grid - hmF2_ref_km)))
    true_alt_ref = float(alt_grid[alt_ref_idx])

    # ── Helper: altitude-altitude Pearson correlation from augmented P ────────
    def _alt_corr(P_aug):
        P_grid = P_aug[:n_sv, :n_sv]
        P_4d   = P_grid.reshape(n_height, n_geo, n_height, n_geo)
        cov    = P_4d.mean(axis=(1, 3))                   # (n_height, n_height)
        std    = np.sqrt(np.maximum(np.diag(cov), 0.0))
        outer  = np.outer(std, std)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return cov / np.where(outer == 0, 1e-10, outer)

    # ── Helper: horizontal correlation at alt_ref_idx for centre vertex ───────
    def _horiz_corr(P_aug):
        P_grid  = P_aug[:n_sv, :n_sv]
        P_4d    = P_grid.reshape(n_height, n_geo, n_height, n_geo)
        # Covariance between (alt_ref, centre) and (alt_ref, every geo vertex)
        cov_row = P_4d[alt_ref_idx, centre_idx, alt_ref_idx, :]  # (n_geo,)
        var_ctr = float(P_4d[alt_ref_idx, centre_idx, alt_ref_idx, centre_idx])
        var_all = P_4d[alt_ref_idx, :, alt_ref_idx, :]            # (n_geo, n_geo)
        std_all = np.sqrt(np.maximum(np.diag(var_all), 0.0))
        std_ctr = float(np.sqrt(max(var_ctr, 0.0)))
        denom   = std_ctr * std_all
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return cov_row / np.where(denom == 0, 1e-10, denom)

    prior_alt_corr  = _alt_corr(prior_P)
    post_alt_corr   = _alt_corr(post_P)
    prior_horiz     = _horiz_corr(prior_P)
    post_horiz      = _horiz_corr(post_P)

    # ── Globe projection centred on the group ─────────────────────────────────
    lats_c = result["lats"]
    lons_c = result["lons"]
    clon   = float(np.nanmean(lons_c)) if lons_c else 0.0
    clat   = float(np.nanmean(lats_c)) if lats_c else 0.0
    proj   = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)

    alt_extent = [float(alt_grid[0]), float(alt_grid[-1]),
                  float(alt_grid[0]), float(alt_grid[-1])]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"EDP Covariance Structure — {result['time_window']}  |  {region}\n"
        f"Horizontal slice at {true_alt_ref:.0f} km  ·  ★ = centre vertex",
        fontsize=12,
    )
    gs = GridSpec(2, 2, figure=fig,
                  left=0.06, right=0.97, top=0.90, bottom=0.07,
                  wspace=0.30, hspace=0.35)

    row_labels = ["Prior", "Posterior"]
    corr_pairs = [(prior_alt_corr, prior_horiz), (post_alt_corr, post_horiz)]

    for row, (row_lbl, (alt_corr, horiz_corr)) in enumerate(
        zip(row_labels, corr_pairs)
    ):
        # ── Left: altitude-altitude correlation ──────────────────────────────
        ax_aa = fig.add_subplot(gs[row, 0])
        pcm = ax_aa.imshow(
            alt_corr, cmap="coolwarm", vmin=-1, vmax=1,
            extent=alt_extent, origin="lower", aspect="auto",
        )
        ax_aa.axhline(true_alt_ref, color="gold", lw=1.0, ls="--", alpha=0.8)
        ax_aa.axvline(true_alt_ref, color="gold", lw=1.0, ls="--", alpha=0.8)
        ax_aa.set_xlabel("Altitude (km)", fontsize=9)
        ax_aa.set_ylabel("Altitude (km)", fontsize=9)
        ax_aa.set_title(f"{row_lbl} — Alt-Alt Correlation", fontsize=10)
        fig.colorbar(pcm, ax=ax_aa, label="Pearson r", fraction=0.046, pad=0.04)

        # ── Right: horizontal correlation globe ───────────────────────────────
        ax_gl = fig.add_subplot(gs[row, 1], projection=proj)
        ax_gl.set_global()
        ax_gl.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
        ax_gl.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
        ax_gl.add_feature(cfeature.COASTLINE.with_scale("110m"),
                          lw=0.4, edgecolor="gray")
        ax_gl.gridlines(lw=0.2, alpha=0.3)

        try:
            tc = ax_gl.tripcolor(
                verts_geo[:, 0], verts_geo[:, 1], eds_occ.mesh,
                horiz_corr,
                transform=ccrs.Geodetic(),
                cmap="coolwarm", shading="flat",
                vmin=-1, vmax=1, zorder=1,
            )
            cb = fig.colorbar(tc, ax=ax_gl, orientation="horizontal",
                              shrink=0.75, pad=0.04, fraction=0.04)
            cb.set_label("Pearson r", fontsize=8)
        except Exception:
            pass

        # Mark the centre (reference) vertex
        ctr_lon = float(verts_geo[centre_idx, 0])
        ctr_lat = float(verts_geo[centre_idx, 1])
        ax_gl.plot(ctr_lon, ctr_lat, transform=ccrs.Geodetic(),
                   marker="*", color="gold", ms=12, mec="black", mew=0.8, zorder=8)
        _draw_roi_boundary(ax_gl, region)
        ax_gl.set_title(
            f"{row_lbl} — Horizontal Correlation at {true_alt_ref:.0f} km",
            fontsize=10,
        )

    os.makedirs(save_dir, exist_ok=True)
    safe_key  = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    _tag_part = f"_{tag}" if tag else ""
    plot_path = os.path.join(save_dir, f"group_{safe_key}{_NOISE_SUFFIX}{_tag_part}_covariance.png")
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    _print_saved(f"  Saved covariance plot → {plot_path}")
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
    igs_entries: list | None = None,
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

    Layout — GridSpec(2, 4) when *igs_entries* is empty / None:
        Cols 0–1, rows 0–1 : 2×2 TEC profile panels, one per GNSS constellation.
                             GPS (top-left / Blues), GLONASS (bottom-left / Purples),
                             Galileo (top-right / Oranges), BeiDou (bottom-right / Greens).
                             Within each panel the shade deepens with occultation index.
                             Legend entries use the GNSS PRN code (G03, E22, R13, C36).
        Col 2, rows 0–1    : Globe — ΔNe coolwarm map at posterior hmF2, ROI boundary,
                             per-occultation raypaths (top/TEC-max/bottom), centre ★.
        Col 3, rows 0–1    : EDP — prior/posterior spaghetti, centre-column profiles,
                             F2-peak markers, Abel Ne profiles in constellation colours.

    When *igs_entries* is provided the GridSpec is extended to (4, 4), adding
    two extra rows below:
        Cols 0–1, rows 2–3 : 2×2 constellation grid — ground-station sTEC vs
                             time (minutes from arc start), one panel per GNSS
                             constellation.  Each arc is a separate line coloured
                             by its station/PRN; IRI baseline shown dashed when
                             available.
        Col 2, rows 2–3    : Globe — F2-layer ionospheric pierce-point (IPP)
                             ground tracks for every IGS arc (dotted lines);
                             station locations (■); pierce point at TEC max (●).
        Col 3, rows 2–3    : IGS arc legend — station, PRN, arc time, TECU range.
    """
    from demo_group import (
        CONSTELLATION_CONFIG, _CONST_FALLBACK_CMAP, _NOISE_SUFFIX,
        _parse_time_window, _draw_terminator, _draw_leo_path, _draw_raypath,
        _isr_limb_tec, _plot_igs_stec_section,
    )

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
    #
    # ┌─ No IGS (2 rows) ──────────────────────────────────────────────────────┐
    # │  Row 0 │ GPS TEC  │ Galileo TEC │ Globe (rows 0-1) │ EDP  (row 0)    │
    # │  Row 1 │ GLO TEC  │ BeiDou TEC  │ Globe (rows 0-1) │ Abel (row 1)    │
    # │        │          │             │                   │ [same height]   │
    # └────────────────────────────────────────────────────────────────────────┘
    #
    # ┌─ With IGS (4 rows) ────────────────────────────────────────────────────┐
    # │  Row 0 │ GPS TEC  │ Galileo TEC │ Upper Globe  │ EDP (rows 0-1)      │
    # │  Row 1 │ GLO TEC  │ BeiDou TEC  │ (rows 0-1)   │ EDP (rows 0-1)      │
    # │  Row 2 │ IGS GPS  │ IGS Galileo │ Lower Globe  │ Abel Ne (row 2)     │
    # │  Row 3 │ IGS GLO  │ IGS BeiDou  │ (rows 2-3)   │ Arc legend (row 3)  │
    # └────────────────────────────────────────────────────────────────────────┘
    _has_igs = bool(igs_entries)
    _n_rows  = 4 if _has_igs else 2
    _fig_h   = 17 if _has_igs else 9

    fig = plt.figure(figsize=(26, _fig_h))
    fig.suptitle(
        f"Group KF Update ({mode_label}): {result['time_window']}  |  "
        f"Region: {region}  |  GN: {leo_str}\n"
        f"{n_occ} occultation(s)  —  "
        f"Prior RMSE {result['prior_tec_rmse']:.2f} → "
        f"Post RMSE {result['post_tec_rmse']:.2f} TECU",
        fontsize=12,
    )
    gs = GridSpec(_n_rows, 4, figure=fig,
                  width_ratios=[1, 1, 1.5, 1.2],
                  wspace=0.40, hspace=0.45)

    # 2×2 TEC panels — always rows 0–1, cols 0–1
    _CONST_POS = {"G": (0, 0), "R": (1, 0), "E": (0, 1), "C": (1, 1)}
    ax_tec      = {}
    first_tec   = None
    for const, (row, col) in _CONST_POS.items():
        cfg = CONSTELLATION_CONFIG.get(const, {"name": const, "title_color": "black"})
        ax  = fig.add_subplot(gs[row, col],
                              sharey=first_tec if first_tec is not None else None)
        ax.set_title(cfg["name"], fontsize=9, color=cfg["title_color"], fontweight="bold")
        ax.grid(True, alpha=0.3, ls=":")
        ax_tec[const] = ax
        if first_tec is None:
            first_tec = ax

    # Globe (col 2):
    #   No IGS  → spans both rows (rows 0–1)
    #   With IGS → upper half only (rows 0–1); lower globe added in _plot_igs_stec_section
    ax2 = fig.add_subplot(gs[0:2, 2], projection=proj)

    # EDP + Abel (col 3):
    #   No IGS  → EDP row 0, Abel row 1 (equal heights, both beside the globe)
    #   With IGS → EDP spans rows 0–1 (same height as 2×2 TEC + upper globe),
    #              Abel at row 2 (half that height, beside the lower globe)
    if _has_igs:
        ax3_kf   = fig.add_subplot(gs[0:2, 3], sharey=first_tec)
        ax3_abel = fig.add_subplot(gs[2,   3], sharey=first_tec, sharex=ax3_kf)
    else:
        ax3_kf   = fig.add_subplot(gs[0, 3], sharey=first_tec)
        ax3_abel = fig.add_subplot(gs[1, 3], sharey=first_tec, sharex=ax3_kf)

    # ── Separate all-TEC figure (every occultation, 2×2 constellation layout) ──
    # IGS ground-station arcs (if any) are excluded from the RO altitude-based
    # panels below and instead plotted vs time in their own 2×2 section.
    _igs_slice_map: dict[tuple, int] = {}
    if igs_entries:
        for _i, _cl in enumerate(clean_list):
            if _cl.get("obs_source") == "IGS_ground":
                _key = (str(_cl.get("leo_id", "")), str(_cl.get("prn_id", "")),
                        str(_cl.get("date", "")))
                _igs_slice_map[_key] = _i
    _igs_slice_idxs = set(_igs_slice_map.values())
    _ro_idxs = [i for i in range(n_occ) if i not in _igs_slice_idxs]

    all_alts = (np.concatenate([tec_slices[i]["tangent_km"] for i in _ro_idxs])
                if _ro_idxs else np.array([0.0]))
    alt_ylim = (0, max(float(np.nanmax(all_alts)) + 50, float(alt_grid[-1])))

    _has_igs_tec = bool(igs_entries)
    fig_tec = plt.figure(figsize=(14, 10 if not _has_igs_tec else 17))
    fig_tec.suptitle(
        f"All TEC Profiles — {result['time_window']}  |  Region: {region}  |  "
        f"GN: {leo_str}\n{n_occ} occultation(s)",
        fontsize=11,
    )
    gs_tec = GridSpec(4 if _has_igs_tec else 2, 2, figure=fig_tec,
                       wspace=0.35, hspace=0.5)
    _ax_tec_all = {}
    _first_tec_all = None
    for const, (row, col) in _CONST_POS.items():
        cfg = CONSTELLATION_CONFIG.get(const, {"name": const, "title_color": "black"})
        ax  = fig_tec.add_subplot(
            gs_tec[row, col],
            sharey=_first_tec_all if _first_tec_all is not None else None,
        )
        ax.set_title(cfg["name"], fontsize=9, color=cfg["title_color"], fontweight="bold")
        ax.grid(True, alpha=0.3, ls=":")
        _ax_tec_all[const] = ax
        if _first_tec_all is None:
            _first_tec_all = ax

    _all_tec_style = [
        Line2D([0], [0], color="gray", lw=2.2,          label="Measured TEC"),
        Line2D([0], [0], color="gray", lw=1.3, ls="--", label="Prior TEC"),
        Line2D([0], [0], color="gray", lw=1.5, ls=":",  label="KF Posterior"),
    ]
    _all_const_legend: dict = defaultdict(list)
    _all_style_placed = False

    for i in _ro_idxs:
        sl, col = tec_slices[i], occ_colours[i]
        const = occ_const[i]
        ax_a  = _ax_tec_all.get(const) or _ax_tec_all.get("G") or next(iter(_ax_tec_all.values()))
        ax_a.plot(sl["measured"],  sl["tangent_km"], color=col, lw=2.2)
        ax_a.plot(sl["prior_tec"], sl["tangent_km"], color=col, lw=1.3, ls="--", alpha=0.6)
        ax_a.plot(sl["post_tec"],  sl["tangent_km"], color=col, lw=1.5, ls=":",  alpha=0.9)
        prn_code = sat_ids[i][1] if i < len(sat_ids) else f"Occ {i + 1}"
        time_str = file_labels[i].split()[-1] if i < len(file_labels) else ""
        lbl      = f"{prn_code}  ({time_str})" if time_str else prn_code
        _all_const_legend[const].append(Line2D([0], [0], color=col, lw=2.2, label=lbl))

    for const, ax_a in _ax_tec_all.items():
        entries = _all_const_legend.get(const, [])
        if entries:
            leg_h = entries + (_all_tec_style if not _all_style_placed else [])
            ax_a.legend(handles=leg_h, fontsize=7, loc="upper right", framealpha=0.85)
            _all_style_placed = True
        else:
            ax_a.text(0.5, 0.5, "No data", transform=ax_a.transAxes,
                      ha="center", va="center", color="lightgray", fontsize=11,
                      style="italic")
        ax_a.set_ylim(*alt_ylim)
        if const in ("G", "R"):
            ax_a.set_ylabel("Tangent Altitude (km)")
        else:
            ax_a.tick_params(labelleft=False)
        if const in ("R", "C"):
            ax_a.set_xlabel("TEC (TECU)")

    # ── IGS ground-station TEC vs elevation — rows 2–3, same 2×2 constellation
    #    layout as the RO panels above, but as clustered bar charts (measured /
    #    prior / posterior, arc-mean) positioned at each arc's mean elevation
    #    angle rather than a continuous sTEC-vs-time curve. See
    #    demo_group._plot_igs_stec_section for the matching group-summary panel.
    if _has_igs_tec:
        import matplotlib.patches as _mpatch

        _ax_igs_all = {}
        for const, (row, col) in _CONST_POS.items():
            cfg = CONSTELLATION_CONFIG.get(const, {"name": const, "title_color": "black"})
            ax  = fig_tec.add_subplot(gs_tec[row + 2, col])
            ax.set_title(f"IGS sTEC — {cfg['name']}", fontsize=9,
                         color=cfg["title_color"], fontweight="bold")
            ax.set_xlabel("Elevation angle (deg)", fontsize=8)
            ax.grid(True, alpha=0.3, ls=":", axis="y")
            _ax_igs_all[const] = ax

        _bar_w      = 0.6   # degrees
        _bar_offset = {"measured": -0.7, "prior": 0.0, "post": 0.7}
        _bar_color  = {"measured": "dimgray", "prior": "tab:blue", "post": "tab:orange"}
        _bar_label  = {"measured": "Measured sTEC", "prior": "Prior sTEC (KF)",
                       "post": "Posterior sTEC (KF)"}
        _has_bars: dict[str, bool] = defaultdict(bool)

        for ce in igs_entries:
            prn   = ce.get("prn_id", "")
            const = prn[0].upper() if prn else "?"
            ax_i  = _ax_igs_all.get(const) or _ax_igs_all.get("G") or next(iter(_ax_igs_all.values()))

            elev = np.asarray(ce.get("elev_deg", []), dtype=float)
            tec  = np.asarray(ce.get("tec", []), dtype=float)
            valid = np.isfinite(elev) & np.isfinite(tec)
            if valid.sum() == 0:
                continue

            elev_mean = float(np.mean(elev[valid]))
            meas_mean = float(np.mean(tec[valid]))

            _ce_key = (str(ce.get("leo_id", "")), str(ce.get("prn_id", "")), str(ce.get("date", "")))
            _sl = tec_slices[_igs_slice_map[_ce_key]] if _ce_key in _igs_slice_map else None
            prior_mean = post_mean = np.nan
            if _sl is not None:
                prior_arr = np.asarray(_sl.get("prior_tec", []), dtype=float)
                post_arr  = np.asarray(_sl.get("post_tec",  []), dtype=float)
                if len(prior_arr) == len(tec):
                    prior_mean = float(np.mean(prior_arr[valid]))
                if len(post_arr) == len(tec):
                    post_mean = float(np.mean(post_arr[valid]))

            ax_i.bar(elev_mean + _bar_offset["measured"], meas_mean,
                     width=_bar_w, color=_bar_color["measured"], zorder=3)
            _has_bars[const] = True
            if np.isfinite(prior_mean):
                ax_i.bar(elev_mean + _bar_offset["prior"], prior_mean,
                         width=_bar_w, color=_bar_color["prior"], alpha=0.85, zorder=3)
            if np.isfinite(post_mean):
                ax_i.bar(elev_mean + _bar_offset["post"], post_mean,
                         width=_bar_w, color=_bar_color["post"], alpha=0.85, zorder=3)

        _igs_all_style = [
            _mpatch.Patch(facecolor=_bar_color["measured"], label=_bar_label["measured"]),
            _mpatch.Patch(facecolor=_bar_color["prior"], alpha=0.85, label=_bar_label["prior"]),
            _mpatch.Patch(facecolor=_bar_color["post"],  alpha=0.85, label=_bar_label["post"]),
        ]
        for const, ax_i in _ax_igs_all.items():
            if _has_bars[const]:
                ax_i.legend(handles=_igs_all_style, fontsize=7,
                            loc="upper right", framealpha=0.85)
            else:
                ax_i.text(0.5, 0.5, "No data", transform=ax_i.transAxes,
                          ha="center", va="center", color="lightgray", fontsize=11,
                          style="italic")
            if const in ("G", "R"):
                ax_i.set_ylabel("sTEC (TECU)")

    safe_key_tec  = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    all_tec_path  = os.path.join(save_dir, f"group_{safe_key_tec}{_NOISE_SUFFIX}{suffix}_all_tec.png")
    fig_tec.savefig(all_tec_path, dpi=100, bbox_inches="tight")
    plt.close(fig_tec)
    _print_saved(f"  Saved all-TEC plot → {all_tec_path}")

    # ── Identify max/min TECU-error occultations for the main constellation panels ─
    # Only consider podTc2 (absolute TEC) occultations; conPhs arcs are excluded.
    # Error metric: mean absolute residual (measured − posterior) per occultation.
    # Selection is per constellation so up to 2 × len(constellations) = 8 are shown.
    occ_errors = np.array([
        float(np.mean(np.abs(sl["measured"] - sl["post_tec"])))
        for sl in tec_slices
    ])

    highlight_idxs: dict[int, str] = {}   # idx → "max" | "min"
    # Group podTc indices by constellation
    const_podtc_idxs: dict[str, list[int]] = defaultdict(list)
    for i in range(n_occ):
        tec_type = clean_list[i].get("tec_type", "absolute") if i < len(clean_list) else "absolute"
        if tec_type != "absolute":
            continue
        const = occ_const[i]
        const_podtc_idxs[const].append(i)

    for const, idxs in const_podtc_idxs.items():
        if not idxs:
            continue
        errs = occ_errors[idxs]
        i_max = idxs[int(np.argmax(errs))]
        i_min = idxs[int(np.argmin(errs))]
        highlight_idxs[i_max] = "max"
        if i_min != i_max:
            highlight_idxs[i_min] = "min"

    # ISR-based TEC profile (limb integral from a single representative ISR sweep)
    isr_tec_cache: dict = {}  # occultation index → predicted TEC array
    if isr_profiles:
        # Pick the single profile whose hour_utc is closest to the median of all
        # profiles — this avoids smearing over profiles from different ionospheric
        # conditions while still selecting a representative sweep.
        _hours    = np.array([p["hour_utc"] for p in isr_profiles])
        _med_hour = float(np.median(_hours))
        _best_idx = int(np.argmin(np.abs(_hours - _med_hour)))
        _isr_ref_prof = isr_profiles[_best_idx]
        for idx in highlight_idxs:
            isr_tec_cache[idx] = _isr_limb_tec(
                _isr_ref_prof, tec_slices[idx]["tangent_km"]
            )

    # ── TEC panel drawing (main figure — only max/min error PRNs) ────────────
    style_entries = [
        Line2D([0], [0], color="gray", lw=2.2,          label="Measured TEC"),
        Line2D([0], [0], color="gray", lw=1.3, ls="--", label="Prior TEC"),
        Line2D([0], [0], color="gray", lw=1.5, ls=":",  label="KF Posterior"),
    ]
    if isr_profiles:
        style_entries.append(
            Line2D([0], [0], color="limegreen", lw=1.8, ls=(0, (3, 1, 1, 1)),
                   label="ISR-based TEC")
        )

    const_legend = defaultdict(list)

    for i, (sl, col) in enumerate(zip(tec_slices, occ_colours)):
        if i not in highlight_idxs:
            continue
        err_tag = highlight_idxs[i]
        const = occ_const[i]
        ax_t  = ax_tec.get(const) or ax_tec.get("G") or next(iter(ax_tec.values()))

        ax_t.plot(sl["measured"],  sl["tangent_km"], color=col, lw=2.2)
        ax_t.plot(sl["prior_tec"], sl["tangent_km"], color=col, lw=1.3,
                  ls="--", alpha=0.6)
        ax_t.plot(sl["post_tec"],  sl["tangent_km"], color=col, lw=1.5,
                  ls=":",  alpha=0.9)

        if i in isr_tec_cache:
            ax_t.plot(isr_tec_cache[i], sl["tangent_km"],
                      color="limegreen", lw=1.8, ls=(0, (3, 1, 1, 1)), alpha=0.9)

        prn_code = sat_ids[i][1] if i < len(sat_ids) else f"Occ {i + 1}"
        time_str = file_labels[i].split()[-1] if i < len(file_labels) else ""
        err_label = f"  [{err_tag} err: {occ_errors[i]:.2f} TECU]"
        lbl = f"{prn_code}  ({time_str}){err_label}" if time_str else f"{prn_code}{err_label}"
        const_legend[const].append(Line2D([0], [0], color=col, lw=2.2, label=lbl))

    style_placed = False

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
        if const in ("G", "R"):
            ax_t.set_ylabel("Tangent Altitude (km)")
        else:
            ax_t.tick_params(labelleft=False)
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

    # ── Terminator + night-side shade (drawn early so foreground is on top) ─
    try:
        mean_ts = _parse_time_window(result["time_window"])
        _draw_terminator(ax2, mean_ts, zorder=5)
    except Exception:
        pass

    # ── LEO ground-tracks ────────────────────────────────────────────────────
    _draw_leo_path(ax2, clean_list, occ_colours, zorder=5)

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
        ray_col  = col if i in highlight_idxs else (0.65, 0.65, 0.65, 0.55)

        for (rtype, ls, lw), ridx in zip(
            ray_defs, [idx_top, idx_tecmax, idx_bottom]
        ):
            lbl = f"{rtype}" if i == 0 else None
            _draw_raypath(ax2, LEO, GNSS, ridx,
                          color=ray_col, ls=ls, lw=lw, label=lbl, zorder=6,
                          TP=(ridx == idx_tecmax))

    ax2.set_title(
        f"ΔNe at hmF2 ({hmF2_label}) + Raypaths + ROI\n"
        "(solid=top, dashed=TEC-max, dotted=bottom  ★=centre)"
    )
    ax2.legend(loc="lower left", fontsize=7, framealpha=0.75)

    # ── Panel 3 (top): KF prior / posterior EDPs ─────────────────────────────
    # log-x, capped at 1e13; ax3_abel shares this xaxis (sharex=ax3_kf) so
    # both panels get the same scale/limits from this one call.
    ax3_kf.set_xscale("log")
    ax3_kf.set_xlim(1e9, 1e13)

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
        ax3_kf.plot(_isr_ref_prof["ne"], _isr_ref_prof["alt_km"],
                    color="limegreen", lw=1.2, alpha=0.9, zorder=3)
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
    ax3_kf.tick_params(labelbottom=False)
    ax3_kf.grid(True, alpha=0.3, ls=":")

    # ── Panel 3 (bottom): Abel Ne profiles ───────────────────────────────────
    abel_legend_lines = []
    for i, (col, abel) in enumerate(zip(occ_colours, abel_list)):
        if abel is None or len(abel.get("Ne", [])) == 0:
            continue
        prn_code = sat_ids[i][1] if i < len(sat_ids) else f"Occ {i + 1}"
        abel_col = col if i in highlight_idxs else (0.65, 0.65, 0.65, 0.55)
        ax3_abel.plot(abel["Ne"], abel["alt_km"], color=abel_col, lw=1.8, ls="-.",
                      alpha=0.9)
        # abel_legend_lines.append(
        #     Line2D([0], [0], color=col, lw=1.8, ls="-.", label=f"Abel {prn_code}")
        # )
        abel_nm, abel_hm = extract_robust_f2_peak(abel["Ne"], abel["alt_km"])
        if not np.isnan(abel_nm):
            ax3_abel.plot(abel_nm, abel_hm, marker="^", ms=7,
                          color=abel_col, mec="black", mew=0.6, zorder=5)

    # abel_legend_lines.append(
    #     Line2D([0], [0], marker="^", color="w", mfc="gray", mec="black", ms=7,
    #            label="F2 Peak (Abel)")
    # )

    # ISR truth overlay on Abel panel (same colour coding as EDP panel)
    # ── ISR truth overlay on EDP panel ───────────────────────────────────────
    if isr_profiles:
        ax3_abel.plot(_isr_ref_prof["ne"], _isr_ref_prof["alt_km"],
                    color="limegreen", lw=1.2, alpha=0.9, zorder=3)
        _isr_nm2, _isr_hm2 = extract_robust_f2_peak(_isr_ref_prof["ne"], _isr_ref_prof["alt_km"])
        if not np.isnan(_isr_nm2):
            ax3_abel.plot(_isr_nm2, _isr_hm2,
                          marker="^", ms=6, color="limegreen",
                          mec="black", mew=0.6, zorder=6)
        abel_legend_lines += [
            Line2D([0], [0], color="limegreen", lw=1.5, alpha=0.7,
                   label=f"ISR truth ({len(isr_profiles)} sweeps)"),
        ]

    ax3_abel.legend(handles=abel_legend_lines, fontsize=7, loc="upper right")
    ax3_abel.set_xlabel("Electron Density (m⁻³)")
    ax3_abel.set_title("Abel Ne Profiles  (▲ = ISR truth)" if isr_profiles
                       else "Abel Ne Profiles")
    ax3_abel.grid(True, alpha=0.3, ls=":")

    # ── IGS ground-station sTEC section (rows 2–3) ───────────────────────────
    if _has_igs:
        _plot_igs_stec_section(
            fig        = fig,
            gs         = gs,
            igs_entries= igs_entries,
            region     = region,
            proj       = proj,
            row_start  = 2,
            tec_slices = result.get("tec_slices"),
            clean_list = result.get("clean_list"),
            verts_geo  = verts_geo,
            tris_geo   = tris_geo,
        )

    safe_key  = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    _igs_tag  = "_igs" if _has_igs else ""
    plot_path = os.path.join(save_dir, f"group_{safe_key}{_NOISE_SUFFIX}{suffix}{_igs_tag}.png")
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    _print_saved(f"  Saved group plot ({mode_label}) → {plot_path}")
    return plot_path




# ─────────────────────────────────────────────────────────────────────────────
# Moved from demo_isr_da_comparison.py
# ─────────────────────────────────────────────────────────────────────────────

def _plot_igs_covariance_panels(
    result:    dict,
    save_dir:  str,
    group_key: str,
    *,
    hmF2_ref_km: float | None = None,
) -> str:
    """
    Covariance-structure figure for igs_only's gridded-KF result, analogous to
    _plot_covariance_panels but for a Kronecker-structured prior (C_v/C_s,
    never densified) plus a variance-only posterior (no cross-correlation —
    see Q1 in memory/project_isr_da_comparison_plan.md).

    Layout (2x2):
      [0,0] Prior vertical (alt-alt) correlation C_v — plotted as supplied,
            not recomputed from a dense matrix.
      [0,1] Prior horizontal correlation C_s at the ROI-centre vertex.
      [1,0] Posterior std-dev (not correlation) map at the hmF2 reference
            altitude — variance-only, no off-diagonal structure.
      [1,1] Blank panel with an explicit note that posterior correlation was
            not computed (truthful about what step Q1 ruled out).
    """
    alt_grid   = result["alt_grid"]
    C_v        = result["prior_C_v"]
    C_s        = result["prior_C_s"]
    post_sigma = result["post_sigma"]
    eds_occ    = result["eds_occ"]
    region     = result["region"]
    prior_edp  = result["prior_edp_3d"]

    verts_geo  = eds_occ.geolocation      # (n_geo, 2): col0=lon, col1=lat
    centre_idx = _roi_centre_idx(verts_geo, region)

    if hmF2_ref_km is None:
        _, hmF2_ref_km = extract_robust_f2_peak(prior_edp[:, centre_idx], alt_grid)
        if np.isnan(hmF2_ref_km):
            hmF2_ref_km = float(alt_grid[len(alt_grid) // 2])
    alt_ref_idx  = int(np.argmin(np.abs(alt_grid - hmF2_ref_km)))
    true_alt_ref = float(alt_grid[alt_ref_idx])

    horiz_corr_row = C_s[centre_idx, :]

    lats_c = result["lats"]
    lons_c = result["lons"]
    clon = float(np.nanmean(lons_c)) if lons_c else 0.0
    clat = float(np.nanmean(lats_c)) if lats_c else 0.0
    proj = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"IGS-only Gridded-KF Covariance Structure — "
        f"{result.get('time_window', group_key)}  |  {region}\n"
        f"Prior shown analytically (Kronecker C_v ⊗ C_s, not densified)  ·  "
        f"Posterior shown as variance only at {true_alt_ref:.0f} km  ·  ★ = centre vertex",
        fontsize=11,
    )
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.97, top=0.86, bottom=0.07,
                  wspace=0.30, hspace=0.40)

    # [0,0] Prior vertical correlation C_v, as supplied
    ax_v = fig.add_subplot(gs[0, 0])
    pcm = ax_v.imshow(
        C_v, cmap="coolwarm", vmin=-1, vmax=1,
        extent=[float(alt_grid[0]), float(alt_grid[-1]),
                float(alt_grid[0]), float(alt_grid[-1])],
        origin="lower", aspect="auto",
    )
    ax_v.axhline(true_alt_ref, color="gold", lw=1.0, ls="--", alpha=0.8)
    ax_v.axvline(true_alt_ref, color="gold", lw=1.0, ls="--", alpha=0.8)
    ax_v.set_xlabel("Altitude (km)", fontsize=9)
    ax_v.set_ylabel("Altitude (km)", fontsize=9)
    ax_v.set_title("Prior Vertical Correlation C_v (analytical input)", fontsize=10)
    fig.colorbar(pcm, ax=ax_v, label="Correlation", fraction=0.046, pad=0.04)

    # [0,1] Prior horizontal correlation C_s at centre vertex
    ax_h = fig.add_subplot(gs[0, 1], projection=proj)
    ax_h.set_global()
    ax_h.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
    ax_h.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
    ax_h.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.4, edgecolor="gray")
    ax_h.gridlines(lw=0.2, alpha=0.3)
    try:
        tc = ax_h.tripcolor(
            verts_geo[:, 0], verts_geo[:, 1], eds_occ.mesh,
            horiz_corr_row,
            transform=ccrs.Geodetic(), cmap="coolwarm", shading="flat",
            vmin=-1, vmax=1, zorder=1,
        )
        cb = fig.colorbar(tc, ax=ax_h, orientation="horizontal",
                           shrink=0.75, pad=0.04, fraction=0.04)
        cb.set_label("Correlation", fontsize=8)
    except Exception:
        pass
    ctr_lon = float(verts_geo[centre_idx, 0])
    ctr_lat = float(verts_geo[centre_idx, 1])
    ax_h.plot(ctr_lon, ctr_lat, transform=ccrs.Geodetic(),
              marker="*", color="gold", ms=12, mec="black", mew=0.8, zorder=8)
    _draw_roi_boundary(ax_h, region)
    ax_h.set_title("Prior Horizontal Correlation C_s (analytical input)", fontsize=10)

    # [1,0] Posterior std-dev map (variance only — no cross-correlation)
    ax_p = fig.add_subplot(gs[1, 0], projection=proj)
    ax_p.set_global()
    ax_p.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
    ax_p.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
    ax_p.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.4, edgecolor="gray")
    ax_p.gridlines(lw=0.2, alpha=0.3)
    sigma_map = post_sigma[alt_ref_idx, :]
    try:
        tc2 = ax_p.tripcolor(
            verts_geo[:, 0], verts_geo[:, 1], eds_occ.mesh,
            sigma_map,
            transform=ccrs.Geodetic(), cmap="viridis", shading="flat", zorder=1,
        )
        cb2 = fig.colorbar(tc2, ax=ax_p, orientation="horizontal",
                            shrink=0.75, pad=0.04, fraction=0.04)
        cb2.set_label("Posterior σ(Ne) [m⁻³]", fontsize=8)
    except Exception:
        pass
    _draw_roi_boundary(ax_p, region)
    ax_p.set_title(f"Posterior Std-Dev at {true_alt_ref:.0f} km (variance only)", fontsize=10)

    # [1,1] Blank note panel — truthful about what wasn't computed
    ax_note = fig.add_subplot(gs[1, 1])
    ax_note.axis("off")
    ax_note.text(
        0.5, 0.5,
        "Posterior correlation not computed for igs_only\n"
        "(variance-only update by design — see project_isr_da_comparison_\n"
        "plan.md, Q1). Densifying the full posterior covariance here would\n"
        "risk the same O(n³) memory blowup the SRIF batch-update fix was\n"
        "written to avoid.",
        ha="center", va="center", fontsize=9.5,
        bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"),
    )

    os.makedirs(save_dir, exist_ok=True)
    safe_key  = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    # "_kf" makes the filter explicit in the filename — this path is always
    # the gridded KF (igs_only's Kronecker-structured prior has no EKF
    # equivalent plot), but naming it explicitly keeps every saved figure
    # self-describing rather than relying on the reader to know that.
    plot_path = os.path.join(save_dir, f"group_{safe_key}_kf_igs_covariance.png")
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    _print_saved(f"  Saved IGS-only covariance plot → {plot_path}")
    return plot_path


def _select_representative_arc_indices(
    idx:            np.ndarray,
    arc_prior_rmse: np.ndarray,
    arc_post_rmse:  np.ndarray,
    n_keep:         int,
) -> np.ndarray:
    """
    Pick up to *n_keep* of *idx* (indices into the full arc arrays) that are
    representative of the full prior/posterior TEC-error distribution, for
    thinning out crowded arc-innovation panels (see ARC_INNOV_MAX_IGS_ARCS).

    Sorts the candidate arcs by a combined prior+post RMSE score and takes
    evenly spaced picks along that ranking (quantile sampling), rather than
    a random subset or a top-N-by-error cut -- this keeps the best-fit,
    worst-fit, and everything-in-between arcs all represented in proportion,
    instead of biasing the displayed subset toward only the largest errors.

    Returns *idx* unchanged (as an array) if len(idx) <= n_keep.
    """
    idx = np.asarray(idx)
    if len(idx) <= n_keep:
        return idx
    score = np.asarray(arc_prior_rmse)[idx] + np.asarray(arc_post_rmse)[idx]
    order = np.argsort(score)                       # ascending combined error
    pick_pos = np.unique(np.round(np.linspace(0, len(idx) - 1, n_keep)).astype(int))
    return idx[order[pick_pos]]


def _plot_group_all_modes(
    group_key: str,
    filter_results: dict,
    igs_window_arcs: list,
    window_isr_profiles: list | None,
    save_dir: Path,
    window_edps: list | None = None,
) -> None:
    """
    Plot a single occultation/station group across all observation modes.

    Parameters
    ----------
    group_key            : RO group identifier (also used in output filenames).
    filter_results        : nested dict, filter_results[obs_mode][filter_type] -> result
                             dict (from process_group / run_info_window / _run_parametric_ekf)
                             or None.
    igs_window_arcs       : IGS ground-station arcs co-located with this group's window.
    window_isr_profiles   : ISR profiles co-located with this group's window (or None) —
                             adapted schema (hour_utc/alt_km/ne/nm_f2/hm_f2), used by the
                             existing joint/alt_slices plots.
    save_dir              : output directory for all figures.
    window_edps           : raw ISR profiles co-located with this group's window (or None)
                             — load_edps() schema (time/lat/lon/alt_km/ne_m3), used by the
                             new per-config ISR-vs-EDP plot which needs lat/lon/time.
    """
    from demo_isr_da_comparison import OBS_MODES, FILTER_TYPES, _safe_group_key
    from demo_compare_kf_enkf import _arc_stats_from_tec_slices
    from demo_isr_initial_conditions import ALT_GRID, _identify_instrument

    # ── Group window_edps by ISR site up front so the spaghetti plot below
    #    draws all of a site's scans on one figure instead of one call (and
    #    one overwritten output file) per individual profile ────────────────
    edps_by_site: dict[str, list] = {}
    for isr_edp in (window_edps or []):
        edps_by_site.setdefault(_identify_instrument(float(isr_edp["lat"])), []).append(isr_edp)

    # Note: gridded_kf's group/joint/alt_slices/all_tec/covariance figures are
    # already produced correctly (using the true joint posterior) inside
    # process_group() itself, in each obs_mode's own save_dir — no additional
    # _plot_group call is needed here.

    # Per-obs_mode figure dirs, matching run_all_filters' SAVE_DIR/{safe_key}/
    # {obs_mode}/ layout from step 1 — figures land alongside the KF ones
    # instead of the old flat save_dir.
    print(f"\n{'▶' * 3} _plot_group_all_modes START | group={group_key}")

    safe_key   = _safe_group_key(group_key)
    group_dirs = {mode: Path(save_dir) / safe_key / mode for mode in OBS_MODES}
    for _d in group_dirs.values():
        _d.mkdir(parents=True, exist_ok=True)

    # ── Per-(obs_mode, filter_type) arc-innovation and covariance diagnostics ──
    for obs_mode in OBS_MODES:
        for filter_type in FILTER_TYPES:
            result = filter_results.get(obs_mode, {}).get(filter_type)
            if result is None:
                continue

            tec_slices = result.get("joint_tec_slices", result.get("tec_slices"))
            clean_list = result.get("clean_list")
            if tec_slices is not None and clean_list is not None:
                try:
                    arc_stats = _arc_stats_from_tec_slices(
                        tec_slices=tec_slices,
                        clean_list=clean_list,
                        sat_ids=result.get("sat_ids", []),
                    )

                    # ── Thin the per-arc panels for ro_igs/igs_only ─────────
                    # IGS ground stations contribute far more arcs per window
                    # than the RO occultations, so mixing them into (or, for
                    # igs_only, entirely filling) the per-arc bar/scatter/map
                    # panels drowns out the RO arcs and overcrowds the chart.
                    # Keep every RO arc, but thin IGS arcs down to a subset
                    # that's representative of the full prior/posterior TEC
                    # error spread (see _select_representative_arc_indices).
                    if obs_mode in ("ro_igs", "igs_only"):
                        _is_igs = np.array(
                            [cl.get("obs_source") == "IGS_ground" for cl in clean_list]
                        )
                        _ro_idx  = np.where(~_is_igs)[0]
                        _igs_idx = np.where(_is_igs)[0]
                        _igs_keep = _select_representative_arc_indices(
                            _igs_idx,
                            arc_stats["arc_prior_rmse"],
                            arc_stats["arc_post_rmse"],
                            ARC_INNOV_MAX_IGS_ARCS,
                        )
                        if len(_igs_keep) < len(_igs_idx):
                            _keep = np.sort(np.concatenate([_ro_idx, _igs_keep]))
                            print(f"  [diag] {group_key}/{obs_mode}/{filter_type}: "
                                  f"arc-innovation panels thinned to "
                                  f"{len(_ro_idx)} RO + {len(_igs_keep)}/"
                                  f"{len(_igs_idx)} representative IGS arcs "
                                  f"(of {len(clean_list)} total)")
                            arc_stats = dict(arc_stats)  # don't mutate the shared dict
                            for _key in ("arc_labels", "arc_prior_mean", "arc_post_mean",
                                         "arc_prior_rmse", "arc_post_rmse",
                                         "arc_lats", "arc_lons"):
                                _val = arc_stats[_key]
                                arc_stats[_key] = ([_val[k] for k in _keep]
                                                    if isinstance(_val, list)
                                                    else np.asarray(_val)[_keep])

                    _plot_arc_innovation_diagnostic(
                        arc_labels     = arc_stats["arc_labels"],
                        arc_prior_mean = arc_stats["arc_prior_mean"],
                        arc_post_mean  = arc_stats["arc_post_mean"],
                        arc_prior_rmse = arc_stats["arc_prior_rmse"],
                        arc_post_rmse  = arc_stats["arc_post_rmse"],
                        arc_lats       = arc_stats["arc_lats"],
                        arc_lons       = arc_stats["arc_lons"],
                        all_prior      = arc_stats["all_prior"],
                        all_post_main  = arc_stats["all_post"],
                        group_key      = f"{group_key}_{obs_mode}",
                        save_dir       = str(group_dirs[obs_mode]),
                        filter_name    = _FILTER_LABELS.get(filter_type, filter_type),
                        prior_rmse     = float(result.get("prior_tec_rmse", np.nan)),
                        post_rmse      = float(result.get("joint_post_tec_rmse",
                                                           result.get("post_tec_rmse", np.nan))),
                    )
                except Exception as exc:
                    print(f"  [warn] arc innovation diagnostic failed for "
                          f"{group_key}/{obs_mode}/{filter_type}: {exc}")

            if "prior_P" in result and "post_P" in result:
                try:
                    # EKF's prior_P/post_P live in the parametric (N_STATE, n_geo)
                    # state space, not the gridded KF's (n_alt, n_geo) Ne-space —
                    # they need a dedicated plot function, not _plot_covariance_panels.
                    if filter_type == "parametric_ekf":
                        _plot_ekf_param_covariance_panels(
                            result, str(group_dirs[obs_mode]), f"{group_key}_{obs_mode}",
                        )
                    else:
                        _plot_covariance_panels_labeled(
                            result, str(group_dirs[obs_mode]), f"{group_key}_{obs_mode}",
                            label=_FILTER_LABELS.get(filter_type, filter_type),
                        )
                except Exception as exc:
                    print(f"  [warn] covariance panels failed for "
                          f"{group_key}/{obs_mode}/{filter_type}: {exc}")
            elif (obs_mode == "igs_only" and filter_type == "gridded_kf"
                  and "prior_C_v" in result):
                # igs_only's gridded-KF prior is Kronecker-structured (never
                # densified) and its posterior is variance-only — needs its
                # own dedicated plot function, not _plot_covariance_panels.
                try:
                    _plot_igs_covariance_panels(
                        result, str(group_dirs[obs_mode]), f"{group_key}_{obs_mode}",
                    )
                except Exception as exc:
                    print(f"  [warn] igs-only covariance panels failed for "
                          f"{group_key}/{obs_mode}/{filter_type}: {exc}")

            # ── Per-config ISR-vs-EDP spaghetti plot (one per present ISR site) ──
            for site_profiles in edps_by_site.values():
                try:
                    _plot_isr_edp_spaghetti(
                        site_profiles, result, obs_mode, filter_type,
                        group_key, group_dirs[obs_mode],
                    )
                except Exception as exc:
                    print(f"  [warn] ISR-vs-EDP spaghetti plot failed for "
                          f"{group_key}/{obs_mode}/{filter_type}: {exc}")

    # ── igs_only gridded-KF joint/alt_slices/all_tec figures ──────────────────
    # Unlike ro_only/ro_igs (whose gridded-KF group/joint/alt_slices/all_tec
    # figures come from process_group()'s own internal _plot_group calls),
    # igs_only's gridded-KF result comes from run_info_window(), which has no
    # such internal plotting call. run_all_filters() merges the
    # _adapt_igs_kf_result_for_plotting() fields onto this result (guarded by
    # the "eds_occ" key below, since older cached results won't have them).
    res_igs_kf = filter_results.get("igs_only", {}).get("gridded_kf")
    if res_igs_kf is not None and "eds_occ" in res_igs_kf:
        igs_entries_kf = [
            cl for cl in res_igs_kf.get("clean_list", [])
            if cl.get("obs_source") == "IGS_ground"
        ] or None
        try:
            _plot_group(
                res_igs_kf, save_dir=str(group_dirs["igs_only"]),
                group_key=f"{group_key}_igs_only", suffix="_kf_joint",
                mode_label="Gridded KF", isr_profiles=window_isr_profiles or None,
                igs_entries=igs_entries_kf,
            )
        except Exception as exc:
            print(f"  [warn] igs_only gridded-KF _plot_group failed for {group_key}: {exc}")
        try:
            _plot_altitude_slices(
                res_igs_kf, save_dir=str(group_dirs["igs_only"]),
                group_key=f"{group_key}_igs_only", suffix="_kf_joint",
            )
        except Exception as exc:
            print(f"  [warn] igs_only gridded-KF altitude-slice plot failed for {group_key}: {exc}")

    # ── Per-obs-mode EKF joint/alt_slices/all_tec figures ─────────────────────
    # gridded_kf's equivalents are produced inside process_group() itself (see
    # note above); process_group only runs the KF, so EKF has no such internal
    # call and needs it here instead.
    for obs_mode in OBS_MODES:
        res_ekf = filter_results.get(obs_mode, {}).get("parametric_ekf")
        if res_ekf is None:
            continue
        igs_entries_ekf = [
            cl for cl in res_ekf.get("clean_list", [])
            if cl.get("obs_source") == "IGS_ground"
        ] or None
        try:
            _plot_group(
                res_ekf, save_dir=str(group_dirs[obs_mode]), group_key=f"{group_key}_{obs_mode}",
                suffix="_ekf_joint", mode_label="Parametric EKF",
                isr_profiles=window_isr_profiles or None,
                igs_entries=igs_entries_ekf,
            )
        except Exception as exc:
            print(f"  [warn] EKF _plot_group failed for {group_key}/{obs_mode}: {exc}")
        try:
            _plot_altitude_slices(
                res_ekf, save_dir=str(group_dirs[obs_mode]), group_key=f"{group_key}_{obs_mode}",
                suffix="_ekf_joint",
            )
        except Exception as exc:
            print(f"  [warn] EKF altitude-slice plot failed for {group_key}/{obs_mode}: {exc}")

    # ── Per-obs-mode KF-vs-EKF comparison figures ─────────────────────────────
    for obs_mode in OBS_MODES:
        res_kf  = filter_results.get(obs_mode, {}).get("gridded_kf")
        res_ekf = filter_results.get(obs_mode, {}).get("parametric_ekf")
        if res_kf is None or res_ekf is None:
            continue
        try:
            plot_kf_enkf_comparison(
                res_kf       = filter_results[obs_mode]["gridded_kf"],
                res_enkf     = filter_results[obs_mode]["parametric_ekf"],
                isr_profiles = window_isr_profiles or [],
                alt_grid     = ALT_GRID,
                group_key    = f"{group_key}_{obs_mode}",
                save_dir     = str(group_dirs[obs_mode]),
            )
        except Exception as exc:
            print(f"  [warn] plot_kf_enkf_comparison failed for {group_key}/{obs_mode}: {exc}")

    # ── Group summary: 6-config EDP-vs-ISR + TEC RMSE bar chart, once per group ──
    try:
        _plot_group_summary_metrics(
            group_key, filter_results, window_edps, Path(save_dir) / safe_key,
        )
    except Exception as exc:
        print(f"  [warn] group summary metrics plot failed for {group_key}: {exc}")

    print(f"{'◀' * 3} _plot_group_all_modes END   | group={group_key}")


def _plot_isr_tec_vs_obs(
    tec_panels: dict,
    group_key:  str,
    inst_name:  str,
    t_utc:      pd.Timestamp,
    solar:      dict,
    isr_lat:    float,
    isr_lon:    float,
    save_dir:   Path,
) -> str | None:
    """
    Companion figure to plot_isr_truth_comparison: retrieved TEC (prior +
    posterior) vs. MEASURED TEC (RO occultation / IGS station sTEC) for the
    arc nearest the ISR site, one panel per (obs_mode, filter_type).

    Unlike the EDP panels in plot_isr_truth_comparison, there is no ISR
    "truth" here — ISR measures electron density profiles, not slant TEC —
    so the reference curve plotted is the measured observation itself, not
    ISR. Panel/figure titles say "vs measured obs" explicitly to avoid this
    being confused with the ISR-truth EDP comparison.

    Parameters
    ----------
    tec_panels : dict, (obs_mode, filter_type) -> dict with keys
        'x', 'xlabel', 'measured', 'prior', 'post', 'label',
        'prior_rmse', 'post_rmse', 'dist_deg' (see plot_isr_truth_comparison).
    """
    from demo_isr_da_comparison import OBS_MODES, FILTER_TYPES
    from demo_isr_initial_conditions import INSTRUMENTS

    if not tec_panels:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    for row, filter_type in enumerate(FILTER_TYPES):
        for col, obs_mode in enumerate(OBS_MODES):
            ax     = axes[row][col]
            key    = (obs_mode, filter_type)
            panel  = tec_panels.get(key)
            colour, _ls = _CONFIG_STYLES.get(key, ("gray", "-"))
            title_prefix = f"{_FILTER_LABELS.get(filter_type, filter_type)} {obs_mode}"

            if panel is None:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="lightgray",
                        fontsize=12, style="italic")
                ax.set_title(title_prefix, fontsize=10)
                continue

            # RO arcs (x = tangent altitude) plot altitude on the y-axis and
            # TEC on the x-axis, matching the EDP panels' altitude-vertical
            # convention; IGS ground arcs (x = time from arc start) keep time
            # on the x-axis since it's a time series, not a vertical profile.
            is_altitude = "altitude" in panel["xlabel"].lower()
            if is_altitude:
                ax.plot(panel["measured"], panel["x"], color="black", lw=2.0, label="Measured")
                ax.plot(panel["prior"], panel["x"], color=colour, lw=1.4, ls="--",
                        alpha=0.75, label="Prior")
                ax.plot(panel["post"],  panel["x"], color=colour, lw=1.6, ls=":",
                        alpha=0.95, label="Posterior")
                ax.set_xlabel("sTEC (TECU)", fontsize=8)
                ax.set_ylabel(panel["xlabel"], fontsize=8)
            else:
                ax.plot(panel["x"], panel["measured"], color="black", lw=2.0, label="Measured")
                ax.plot(panel["x"], panel["prior"], color=colour, lw=1.4, ls="--",
                        alpha=0.75, label="Prior")
                ax.plot(panel["x"], panel["post"],  color=colour, lw=1.6, ls=":",
                        alpha=0.95, label="Posterior")
                ax.set_xlabel(panel["xlabel"], fontsize=8)
                ax.set_ylabel("sTEC (TECU)", fontsize=8)
            ax.grid(True, alpha=0.3, ls=":")
            ax.legend(fontsize=7, loc="best")
            ax.set_title(
                f"{title_prefix} vs measured obs — {panel['label']}\n"
                f"RMSE prior={panel['prior_rmse']:.3f}  post={panel['post_rmse']:.3f} TECU  "
                f"(arc {panel['dist_deg']:.2f}° from site)",
                fontsize=8.5,
            )

    fig.suptitle(
        f"{INSTRUMENTS[inst_name]['label']}  ·  {t_utc}  ·  "
        f"TEC vs. MEASURED obs (nearest arc to site {isr_lat:.2f}°, {isr_lon:.2f}°) "
        f"— NOT ISR truth (ISR measures Ne, not slant TEC)\n"
        f"F10.7={solar['f107']:.0f}  Ap={solar['ap']}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"isr_tec_vs_obs_{group_key}_{inst_name}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _print_saved(f"  ISR TEC-vs-measured-obs comparison saved → {out_path}")
    return str(out_path)


def _plot_isr_edp_spaghetti(
    isr_profiles: list[dict],
    result: dict,
    obs_mode: str,
    filter_type: str,
    group_key: str,
    save_dir: Path,
) -> str | None:
    """
    Spaghetti ISR-vs-EDP retrieval plot: every ISR profile co-located with
    this group's time window plotted as one thin line each (all from the
    same site, so they share a single nearest mesh vertex), with the one
    (obs_mode, filter_type) config's prior/posterior EDP retrieval overlaid
    on top. For the per-run stage of the plotting protocol (as opposed to
    plot_isr_truth_comparison's all-6-configs combined comparison).

    Parameters
    ----------
    isr_profiles : list of dicts with "time", "lat", "lon", "alt_km", "ne_m3",
                   "kindat" (from load_edps()), all from the same ISR site.
                   Individual scan lines are styled by kindat (solid = 6400
                   fitted params, dashed = 6300 power profile) since both
                   can be co-located at the same site/time — see
                   _ISR_KINDAT_STYLE.
    result       : single filter_results[obs_mode][filter_type] result dict.
    save_dir     : this (obs_mode)'s own output directory
                   (SAVE_DIR/{safe_key}/{obs_mode}/).
    """
    from demo_isr_initial_conditions import INSTRUMENTS, _identify_instrument

    if any(k not in result for k in
           ("prior_edp_3d", "post_edp_3d", "alt_grid", "eds_occ")):
        print(f"  [skip] ISR-vs-EDP spaghetti plot for {group_key}/{obs_mode}/"
              f"{filter_type}: result missing EDP output keys")
        return None

    valid_profiles = []
    for p in isr_profiles:
        isr_alt = np.asarray(p["alt_km"], dtype=float)
        isr_ne  = np.asarray(p["ne_m3"],  dtype=float)
        valid = (isr_ne > 1e8) & np.isfinite(isr_ne)
        if valid.sum() < ISR_MIN_VALID_GATES:
            continue
        valid_profiles.append((p, isr_alt, isr_ne, valid))

    if not valid_profiles:
        print(f"  [skip] ISR-vs-EDP spaghetti plot for {group_key}/{obs_mode}/"
              f"{filter_type}: no ISR profile with enough valid gates")
        return None

    isr_lat   = float(valid_profiles[0][0]["lat"])
    isr_lon   = float(valid_profiles[0][0]["lon"])
    inst_name = _identify_instrument(isr_lat)

    prior_edp_3d = np.asarray(result["prior_edp_3d"])
    post_edp_3d  = np.asarray(result["post_edp_3d"])
    alt_grid     = np.asarray(result["alt_grid"])
    geoloc       = np.asarray(result["eds_occ"].geolocation)  # (n_geo,2): lon, lat

    mesh_pts = np.column_stack([geoloc[:, 1], geoloc[:, 0]])  # (lat, lon)
    tree = cKDTree(mesh_pts)
    _dist, nearest_idx = tree.query([isr_lat, isr_lon])

    prior_col = prior_edp_3d[:, nearest_idx]
    post_col  = post_edp_3d[:, nearest_idx]

    # ── Per-profile RMSE vs the group's single prior/posterior retrieval,
    #    averaged across the window for the title ────────────────────────────
    prior_rmses, post_rmses, times = [], [], []
    for p, isr_alt, isr_ne, valid in valid_profiles:
        prior_at_isr = np.interp(isr_alt, alt_grid, prior_col)
        post_at_isr  = np.interp(isr_alt, alt_grid, post_col)
        prior_rmses.append(float(np.sqrt(np.mean(
            (prior_at_isr[valid] - isr_ne[valid]) ** 2))))
        post_rmses.append(float(np.sqrt(np.mean(
            (post_at_isr[valid] - isr_ne[valid]) ** 2))))
        times.append(pd.Timestamp(p["time"]))

    prior_rmse = float(np.mean(prior_rmses))
    post_rmse  = float(np.mean(post_rmses))
    t_lo, t_hi = min(times), max(times)

    colour, _ls = _CONFIG_STYLES.get((obs_mode, filter_type), ("tab:blue", "-"))

    fig, ax = plt.subplots(figsize=(7, 8))

    # ── Spaghetti: one thin line per ISR scan, shaded by time-in-window and
    #    styled by kindat (solid = fitted params 6400, dashed = power
    #    profile 6300, dotted = unlabeled cache — see _ISR_KINDAT_STYLE) ──────
    span_s = max((t_hi - t_lo).total_seconds(), 1.0)
    cmap = mpl.colormaps.get_cmap("Greys")
    t_nums = mdates.date2num(times)
    norm = mpl.colors.Normalize(vmin=t_nums.min(), vmax=t_nums.max())
    kindats_seen: set[str] = set()
    for (p, isr_alt, isr_ne, _valid), t in zip(valid_profiles, times):
        frac = 0.35 + 0.55 * ((t - t_lo).total_seconds() / span_s)
        kindat = p.get("kindat")
        _, kd_ls = _isr_kindat_style(kindat)
        kindats_seen.add(kindat)
        ax.plot(isr_ne, isr_alt, color=cmap(frac), ls=kd_ls, lw=0.9,
                 alpha=0.8, zorder=2)

    prior_line, = ax.plot(prior_col, alt_grid, color=colour, ls="--", lw=2.0,
                           label="Prior EDP", zorder=3)
    post_line,  = ax.plot(post_col,  alt_grid, color=colour, ls="-",  lw=2.2,
                           label="Posterior EDP", zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(1e9, 1e13)
    ax.set_ylim(0, 800)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Ne  (m⁻³)")
    ax.set_ylabel("Altitude  (km)")

    if filter_type == "parametric_ekf":
        from demo_compare_kf_enkf import _draw_param_boxes
        param_entries = []
        prior_state = result.get("prior_mean_state")
        post_state  = result.get("posterior_mean_state", result.get("post_mean_state"))
        if prior_state is not None:
            param_entries.append(("Prior EKF", colour, np.asarray(prior_state)[:, nearest_idx]))
        if post_state is not None:
            param_entries.append(("Posterior EKF", colour, np.asarray(post_state)[:, nearest_idx]))
        _draw_param_boxes(ax, param_entries, loc="lower right")

    kindat_legend = [Line2D([0], [0], color="dimgray", ls=_isr_kindat_style(kd)[1],
                             lw=1.4, label=_isr_kindat_style(kd)[0])
                      for kd in sorted(kindats_seen, key=str)]
    ax.legend(handles=[prior_line, post_line] + kindat_legend, fontsize=9, loc="upper left")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(f"ISR scan time UTC ({len(valid_profiles)} scans)")
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    filter_label = _FILTER_LABELS.get(filter_type, filter_type)
    ax.set_title(
        f"{INSTRUMENTS[inst_name]['label']}  ·  {t_lo:%H:%M}–{t_hi:%H:%M} UTC"
        f"  ·  {filter_label} {obs_mode}\n"
        f"Mean RMSE (Ne, m⁻³) prior={prior_rmse:.2e}  post={post_rmse:.2e}",
        fontsize=10,
    )
    fig.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"isr_truth_{group_key}_{obs_mode}_{filter_type}_{inst_name}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _print_saved(f"  ISR-vs-EDP spaghetti plot saved → {out_path}")
    return str(out_path)


def _group_edp_rmse_vs_isr(
    result: dict,
    edps_by_site: dict[str, list],
) -> tuple[float, float, float, float] | None:
    """
    Mean (prior_ne_rmse, post_ne_rmse, prior_fp_rmse, post_fp_rmse) for one
    filter result, averaged across every ISR profile in *edps_by_site* (all
    sites combined).  Ne values are in m⁻³; fp values are in MHz.

    Only altitudes below the prior F2 peak are included (same gate mask used
    by compute_isr_metrics).  Returns None when no valid gates remain.
    """
    if any(k not in result for k in
           ("prior_edp_3d", "post_edp_3d", "alt_grid", "eds_occ")):
        return None

    prior_edp_3d = np.asarray(result["prior_edp_3d"])
    post_edp_3d  = np.asarray(result["post_edp_3d"])
    alt_grid     = np.asarray(result["alt_grid"])
    geoloc       = np.asarray(result["eds_occ"].geolocation)  # (n_geo,2): lon, lat
    mesh_pts = np.column_stack([geoloc[:, 1], geoloc[:, 0]])  # (lat, lon)
    tree = cKDTree(mesh_pts)

    prior_vals, post_vals = [], []
    prior_fp_vals, post_fp_vals = [], []
    for profiles in edps_by_site.values():
        if not profiles:
            continue
        lat = float(profiles[0]["lat"])
        lon = float(profiles[0]["lon"])
        _dist, idx = tree.query([lat, lon])
        prior_col = prior_edp_3d[:, idx]
        post_col  = post_edp_3d[:, idx]

        # F2 peak altitude from the prior column; restrict RMSE to below-peak altitudes.
        _, prior_hmF2 = extract_robust_f2_peak(prior_col, alt_grid)

        for p in profiles:
            isr_alt = np.asarray(p["alt_km"], dtype=float)
            isr_ne  = np.asarray(p["ne_m3"],  dtype=float)
            below_peak = (isr_alt <= prior_hmF2) if np.isfinite(prior_hmF2) \
                         else np.ones(len(isr_alt), dtype=bool)
            valid = (isr_ne > 1e8) & np.isfinite(isr_ne) & below_peak
            if valid.sum() < ISR_MIN_VALID_GATES:
                continue
            prior_at_isr = np.interp(isr_alt, alt_grid, prior_col)
            post_at_isr  = np.interp(isr_alt, alt_grid, post_col)
            prior_vals.append(float(np.sqrt(np.mean(
                (prior_at_isr[valid] - isr_ne[valid]) ** 2))))
            post_vals.append(float(np.sqrt(np.mean(
                (post_at_isr[valid] - isr_ne[valid]) ** 2))))
            # MHz (plasma-frequency) RMSE
            fp_truth = _ne_to_fp(isr_ne[valid])
            prior_fp_vals.append(float(np.sqrt(np.mean(
                (_ne_to_fp(prior_at_isr[valid]) - fp_truth) ** 2))))
            post_fp_vals.append(float(np.sqrt(np.mean(
                (_ne_to_fp(post_at_isr[valid])  - fp_truth) ** 2))))

    if not prior_vals:
        return None
    return (float(np.mean(prior_vals)),    float(np.mean(post_vals)),
            float(np.mean(prior_fp_vals)), float(np.mean(post_fp_vals)))


def _plot_group_summary_metrics(
    group_key: str,
    filter_results: dict,
    window_edps: list[dict] | None,
    save_dir: Path,
) -> str | None:
    """
    One summary figure per group covering all six (obs_mode, filter_type)
    configs at once, laid out as: one altitude-vs-EDP panel per ISR truth
    station on the left, and the RMSE-of-EDP / RMSE-of-TEC panels stacked
    2 rows x 1 column on the right.

    A (left, one subplot per ISR site)
                 : altitude vs. EDP -- every config's prior (faint) /
                   posterior (bold) retrieval at that site's nearest mesh
                   vertex, overlaid with that site's co-located ISR truth
                   profiles as thin black spaghetti lines. If no ISR truth
                   station is present in the window, a single placeholder
                   panel is shown instead.
    B (top-right)    : mean EDP-vs-ISR-truth RMSE (absolute Ne, m⁻³), prior vs.
                 posterior, aggregated across every ISR profile co-located
                 with this group's time window (all sites combined) -- one
                 bar pair per config, not per individual scan/site (see
                 _plot_isr_edp_spaghetti for that per-site, per-scan detail
                 view).
    C (bottom-right) : prior/posterior TEC RMSE (TECU) for the same six
                 configs, read directly off each filter result (no ISR
                 truth required).

    Parameters
    ----------
    filter_results : nested dict, filter_results[obs_mode][filter_type] -> result
                      dict or None.
    window_edps    : ISR profiles (load_edps() schema) co-located with this
                     group's window, already time+site matched (see
                     _isr_profiles_for_window). May span multiple ISR sites.
    save_dir       : this group's own output directory
                     (SAVE_DIR/{safe_key}/).
    """
    from demo_isr_da_comparison import OBS_MODES, FILTER_TYPES
    from demo_isr_initial_conditions import _identify_instrument

    edps_by_site: dict[str, list] = {}
    for isr_edp in (window_edps or []):
        edps_by_site.setdefault(_identify_instrument(float(isr_edp["lat"])), []).append(isr_edp)

    configs = [(om, ft) for om in OBS_MODES for ft in FILTER_TYPES]
    labels  = [f"{om}\n{_FILTER_LABELS.get(ft, ft)}" for om, ft in configs]
    colours = [_CONFIG_STYLES.get(cfg, ("tab:blue", "-"))[0] for cfg in configs]

    edp_prior, edp_post, edp_prior_mhz, edp_post_mhz = [], [], [], []
    tec_prior, tec_post = [], []
    for obs_mode, filter_type in configs:
        result = filter_results.get(obs_mode, {}).get(filter_type)
        if result is None:
            edp_prior.append(np.nan);     edp_post.append(np.nan)
            edp_prior_mhz.append(np.nan); edp_post_mhz.append(np.nan)
            tec_prior.append(np.nan);     tec_post.append(np.nan)
            continue

        edp_metric = _group_edp_rmse_vs_isr(result, edps_by_site)
        if edp_metric is None:
            edp_prior.append(np.nan);     edp_post.append(np.nan)
            edp_prior_mhz.append(np.nan); edp_post_mhz.append(np.nan)
        else:
            edp_prior.append(edp_metric[0]);     edp_post.append(edp_metric[1])
            edp_prior_mhz.append(edp_metric[2]); edp_post_mhz.append(edp_metric[3])

        tec_prior.append(float(result.get("prior_tec_rmse", np.nan)))
        tec_post.append(float(result.get(
            "joint_post_tec_rmse", result.get("post_tec_rmse", np.nan))))

    if all(np.isnan(v) for v in edp_prior + edp_post + tec_prior + tec_post):
        print(f"  [skip] group summary metrics plot for {group_key}: no data in any config")
        return None

    x = np.arange(len(configs))
    width = 0.35

    # ── Layout: one altitude-vs-EDP panel per ISR site (left) + the
    # RMSE-of-EDP / RMSE-of-TEC panels stacked 2 rows x 1 col (right) ─────────
    sites = sorted(edps_by_site.keys())
    n_curve_panels = max(len(sites), 1)

    fig = plt.figure(figsize=(6.5 * n_curve_panels + 7.5, 11))
    gs = fig.add_gridspec(nrows=2, ncols=n_curve_panels + 1,
                          width_ratios=[1.0] * n_curve_panels + [1.15])
    curve_axes = [fig.add_subplot(gs[:, i]) for i in range(n_curve_panels)]
    ax_edp = fig.add_subplot(gs[0, n_curve_panels])
    ax_tec = fig.add_subplot(gs[1, n_curve_panels])

    # ── Panel A: one altitude-vs-EDP subplot per ISR truth station ────────────
    valid_isr_profiles = []  # aggregated across sites, for the panel-B title

    if not sites:
        ax_curve = curve_axes[0]
        ax_curve.set_xscale("log")
        ax_curve.set_xlim(1e9, 1e13)
        ax_curve.set_ylim(0, 800)
        ax_curve.grid(True, which="both", alpha=0.3)
        ax_curve.set_xlabel("Ne  (m⁻³)")
        ax_curve.set_ylabel("Altitude  (km)")
        ax_curve.text(0.5, 0.5, "No ISR truth in window", transform=ax_curve.transAxes,
                      ha="center", va="center", color="lightgray",
                      fontsize=12, style="italic")
        ax_curve.set_title("EDP vs. ISR truth")

    for panel_idx, site in enumerate(sites):
        ax_curve = curve_axes[panel_idx]
        site_profiles = edps_by_site[site]

        ax_curve.set_xscale("log")
        ax_curve.set_xlim(1e9, 1e13)
        ax_curve.set_ylim(0, 800)
        ax_curve.grid(True, which="both", alpha=0.3)
        ax_curve.set_xlabel("Ne  (m⁻³)")
        if panel_idx == 0:
            ax_curve.set_ylabel("Altitude  (km)")

        valid_site_profiles = []
        for p in site_profiles:
            isr_alt = np.asarray(p["alt_km"], dtype=float)
            isr_ne  = np.asarray(p["ne_m3"],  dtype=float)
            valid = (isr_ne > 1e8) & np.isfinite(isr_ne)
            if valid.sum() < ISR_MIN_VALID_GATES:
                continue
            valid_site_profiles.append((isr_alt[valid], isr_ne[valid], p.get("kindat")))
        valid_isr_profiles.extend(valid_site_profiles)

        # Styled by kindat (solid = 6400 fitted params, dashed = 6300 power
        # profile) since both can be co-located at the same site/window —
        # see _ISR_KINDAT_STYLE.
        kindat_counts: dict[str, int] = {}
        _band_label_added = False
        for isr_alt, isr_ne, kindat in valid_site_profiles:
            _, kd_ls = _isr_kindat_style(kindat)
            # ±0.5 MHz shaded band around truth (converted to Ne)
            ne_lo, ne_hi = _truth_fp_band(isr_ne)
            ax_curve.fill_betweenx(isr_alt, ne_lo, ne_hi,
                                   alpha=0.18, color="black", zorder=0,
                                   label="±0.5 MHz band" if not _band_label_added else None)
            _band_label_added = True
            ax_curve.plot(isr_ne, isr_alt, color="black", ls=kd_ls,
                          lw=0.8, alpha=0.35, zorder=1)
            kindat_counts[kindat] = kindat_counts.get(kindat, 0) + 1

        curve_legend = []
        if _band_label_added:
            curve_legend.append(Line2D([0], [0], color="black", lw=6, alpha=0.18,
                                        label="±0.5 MHz band"))
        for kd in sorted(kindat_counts, key=str):
            kd_label, kd_ls = _isr_kindat_style(kd)
            curve_legend.append(Line2D([0], [0], color="black", ls=kd_ls, lw=1.4, alpha=0.6,
                                        label=f"{kd_label} ({kindat_counts[kd]} scans)"))

        lat = float(site_profiles[0]["lat"]) if site_profiles else None
        lon = float(site_profiles[0]["lon"]) if site_profiles else None
        param_entries = []
        # Collect per-config EDP columns for the MHz duplicate figure.
        _site_config_curves: list[tuple] = []

        for obs_mode, filter_type in configs:
            result = filter_results.get(obs_mode, {}).get(filter_type)
            if result is None or any(k not in result for k in
                                      ("prior_edp_3d", "post_edp_3d", "alt_grid", "eds_occ")):
                continue
            if lat is None:
                continue

            colour, ls = _CONFIG_STYLES.get((obs_mode, filter_type), ("tab:blue", "-"))
            prior_edp_3d = np.asarray(result["prior_edp_3d"])
            post_edp_3d  = np.asarray(result["post_edp_3d"])
            alt_grid_cfg = np.asarray(result["alt_grid"])
            geoloc       = np.asarray(result["eds_occ"].geolocation)  # (n_geo,2): lon, lat
            mesh_pts = np.column_stack([geoloc[:, 1], geoloc[:, 0]])  # (lat, lon)
            tree = cKDTree(mesh_pts)
            _dist, idx = tree.query([lat, lon])

            prior_col = prior_edp_3d[:, idx]
            post_col  = post_edp_3d[:, idx]

            ax_curve.plot(prior_col, alt_grid_cfg, color=colour, ls=ls,
                          lw=1.3, alpha=0.45, zorder=2)
            ax_curve.plot(post_col,  alt_grid_cfg, color=colour, ls=ls,
                          lw=2.0, alpha=0.95, zorder=3)
            curve_legend.append(Line2D(
                [0], [0], color=colour, ls=ls, lw=2.0,
                label=f"{_FILTER_LABELS.get(filter_type, filter_type)} {obs_mode}"))

            if filter_type == "parametric_ekf":
                post_state = result.get("posterior_mean_state", result.get("post_mean_state"))
                if post_state is not None:
                    param_entries.append((
                        f"EKF {obs_mode}", colour, np.asarray(post_state)[:, idx],
                    ))

            _site_config_curves.append((
                colour, ls, prior_col, post_col, alt_grid_cfg,
                f"{_FILTER_LABELS.get(filter_type, filter_type)} {obs_mode}",
            ))

        ax_curve.set_title(f"{site}: EDP vs. ISR truth ({len(valid_site_profiles)} scan(s))")
        if curve_legend:
            ax_curve.legend(handles=curve_legend, fontsize=7, loc="upper left")
        if param_entries:
            from demo_compare_kf_enkf import _draw_param_boxes
            _draw_param_boxes(ax_curve, param_entries, loc="lower right", fontsize=6.0)

        # Stash per-site data for the MHz duplicate figure below.
        if panel_idx == 0:
            _mhz_site_data: list[tuple] = []
        _mhz_site_data.append((site, valid_site_profiles, _site_config_curves, lat, lon))

    ax_edp.bar(x - width / 2, edp_prior, width, color=colours, alpha=0.45, label="Prior")
    ax_edp.bar(x + width / 2, edp_post,  width, color=colours, alpha=0.95, label="Posterior")
    ax_edp.set_xticks(x)
    ax_edp.set_xticklabels(labels, fontsize=8)
    ax_edp.set_ylabel("EDP RMSE vs. ISR truth  (Ne, m⁻³)")
    ax_edp.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    n_profiles = sum(len(v) for v in edps_by_site.values())
    ax_edp.set_title(f"EDP vs. ISR truth ({n_profiles} scan(s), all sites)")
    ax_edp.grid(True, axis="y", alpha=0.3)
    ax_edp.legend(fontsize=9)

    ax_tec.bar(x - width / 2, tec_prior, width, color=colours, alpha=0.45, label="Prior")
    ax_tec.bar(x + width / 2, tec_post,  width, color=colours, alpha=0.95, label="Posterior")
    ax_tec.set_xticks(x)
    ax_tec.set_xticklabels(labels, fontsize=8)
    ax_tec.set_ylabel("TEC RMSE  (TECU)")
    ax_tec.set_title("TEC prior/posterior RMSE")
    ax_tec.grid(True, axis="y", alpha=0.3)
    ax_tec.legend(fontsize=9)

    fig.suptitle(f"Group summary — {group_key}", fontsize=12)
    fig.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"group_summary_{group_key}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _print_saved(f"  Group summary metrics plot saved → {out_path}")

    # ── Duplicate figure in plasma-frequency (MHz) units ──────────────────────
    _mhz_data = locals().get("_mhz_site_data", [])
    fig_mhz = plt.figure(figsize=(6.5 * n_curve_panels + 7.5, 11))
    gs_mhz = fig_mhz.add_gridspec(nrows=2, ncols=n_curve_panels + 1,
                                   width_ratios=[1.0] * n_curve_panels + [1.15])
    curve_axes_mhz = [fig_mhz.add_subplot(gs_mhz[:, i]) for i in range(n_curve_panels)]
    ax_edp_mhz = fig_mhz.add_subplot(gs_mhz[0, n_curve_panels])
    ax_tec_mhz = fig_mhz.add_subplot(gs_mhz[1, n_curve_panels])

    if not _mhz_data:
        ax_c = curve_axes_mhz[0]
        ax_c.set_xlim(0, 20);  ax_c.set_ylim(0, 800)
        ax_c.grid(True, alpha=0.3)
        ax_c.set_xlabel("fₚ  (MHz)");  ax_c.set_ylabel("Altitude  (km)")
        ax_c.text(0.5, 0.5, "No ISR truth in window", transform=ax_c.transAxes,
                  ha="center", va="center", color="lightgray", fontsize=12, style="italic")
        ax_c.set_title("EDP vs. ISR truth  (MHz)")

    for _pi, (site, _vsp, _cc, _lat, _lon) in enumerate(_mhz_data):
        ax_c = curve_axes_mhz[_pi]
        ax_c.set_xlim(0, 20);  ax_c.set_ylim(0, 800)
        ax_c.grid(True, alpha=0.3)
        ax_c.set_xlabel("fₚ  (MHz)")
        if _pi == 0:
            ax_c.set_ylabel("Altitude  (km)")

        _band_added = False
        _kd_counts: dict = {}
        for _alt, _ne, _kd in _vsp:
            _, _kd_ls = _isr_kindat_style(_kd)
            _fp = _ne_to_fp(_ne)
            ax_c.fill_betweenx(_alt, _fp - _FP_BAND_MHZ, _fp + _FP_BAND_MHZ,
                               alpha=0.18, color="black", zorder=0)
            ax_c.plot(_fp, _alt, color="black", ls=_kd_ls, lw=0.8, alpha=0.35, zorder=1)
            _kd_counts[_kd] = _kd_counts.get(_kd, 0) + 1
            _band_added = True

        _leg_mhz = []
        if _band_added:
            _leg_mhz.append(Line2D([0], [0], color="black", lw=6, alpha=0.18,
                                    label="±0.5 MHz band"))
        for _kd in sorted(_kd_counts, key=str):
            _kd_lbl, _kd_ls = _isr_kindat_style(_kd)
            _leg_mhz.append(Line2D([0], [0], color="black", ls=_kd_ls, lw=1.4, alpha=0.6,
                                    label=f"{_kd_lbl} ({_kd_counts[_kd]} scans)"))
        for _colour, _ls, _prior_col, _post_col, _alt_g, _lbl in _cc:
            ax_c.plot(_ne_to_fp(_prior_col), _alt_g, color=_colour, ls=_ls,
                      lw=1.3, alpha=0.45, zorder=2)
            ax_c.plot(_ne_to_fp(_post_col),  _alt_g, color=_colour, ls=_ls,
                      lw=2.0, alpha=0.95, zorder=3)
            _leg_mhz.append(Line2D([0], [0], color=_colour, ls=_ls, lw=2.0, label=_lbl))

        ax_c.set_title(f"{site}: EDP vs. ISR truth  ({len(_vsp)} scan(s))  [MHz]")
        if _leg_mhz:
            ax_c.legend(handles=_leg_mhz, fontsize=7, loc="upper left")

    ax_edp_mhz.bar(x - width / 2, edp_prior_mhz, width, color=colours, alpha=0.45, label="Prior")
    ax_edp_mhz.bar(x + width / 2, edp_post_mhz,  width, color=colours, alpha=0.95, label="Posterior")
    ax_edp_mhz.set_xticks(x)
    ax_edp_mhz.set_xticklabels(labels, fontsize=8)
    ax_edp_mhz.set_ylabel("EDP RMSE vs. ISR truth  (MHz)")
    ax_edp_mhz.set_title(f"EDP vs. ISR truth  ({n_profiles} scan(s), all sites)  [MHz]")
    ax_edp_mhz.grid(True, axis="y", alpha=0.3)
    ax_edp_mhz.legend(fontsize=9)

    ax_tec_mhz.bar(x - width / 2, tec_prior, width, color=colours, alpha=0.45, label="Prior")
    ax_tec_mhz.bar(x + width / 2, tec_post,  width, color=colours, alpha=0.95, label="Posterior")
    ax_tec_mhz.set_xticks(x)
    ax_tec_mhz.set_xticklabels(labels, fontsize=8)
    ax_tec_mhz.set_ylabel("TEC RMSE  (TECU)")
    ax_tec_mhz.set_title("TEC prior/posterior RMSE")
    ax_tec_mhz.grid(True, axis="y", alpha=0.3)
    ax_tec_mhz.legend(fontsize=9)

    fig_mhz.suptitle(f"Group summary (MHz) — {group_key}", fontsize=12)
    fig_mhz.tight_layout()
    out_path_mhz = save_dir / f"group_summary_{group_key}_MHz.png"
    fig_mhz.savefig(out_path_mhz, dpi=130, bbox_inches="tight")
    plt.close(fig_mhz)
    _print_saved(f"  Group summary (MHz) plot saved → {out_path_mhz}")

    return str(out_path)


def plot_isr_truth_comparison(
    isr_profile: dict,
    filter_results: dict,
    group_key: str,
    solar: dict,
    save_dir: Path,
) -> str | None:
    """
    Compare filter prior/posterior EDPs against one ISR ground-truth profile
    across all (obs_mode, filter_type) configurations.

    Also writes a companion "isr_tec_vs_obs_{group_key}_{inst_name}.png" figure
    (via _plot_isr_tec_vs_obs) comparing retrieved TEC against the MEASURED
    RO/IGS TEC observations for the arc nearest the ISR site — ISR has no
    direct slant-TEC truth, so that figure uses measured obs as its reference
    rather than ISR.

    Parameters
    ----------
    isr_profile    : dict with "time", "lat", "lon", "alt_km", "ne_m3", "kindat"
                     (from load_edps()). The caller picks a single scan closest
                     to the window centre, which may be kindat 6400 (fitted
                     params) or 6300 (power profile) depending on what's
                     available -- the "ISR truth" legend/title label reflects
                     which one was used (see _ISR_KINDAT_STYLE).
    filter_results : nested dict, filter_results[obs_mode][filter_type] -> result dict.
    group_key      : RO group identifier (used in the output filename).
    solar          : solar-condition dict from get_solar_conditions().
    save_dir       : output directory for the figure.

    Returns
    -------
    str | None : path to the saved PNG, or None if no configuration had usable
                 EDP output to compare against the ISR profile.
    """
    from demo_isr_da_comparison import OBS_MODES, FILTER_TYPES
    from demo_compare_kf_enkf import _arc_stats_from_tec_slices
    from demo_isr_initial_conditions import INSTRUMENTS, _identify_instrument

    isr_alt = np.asarray(isr_profile["alt_km"], dtype=float)
    isr_ne  = np.asarray(isr_profile["ne_m3"],  dtype=float)
    isr_lat = float(isr_profile["lat"])
    isr_lon = float(isr_profile["lon"])
    t_utc   = pd.Timestamp(isr_profile["time"])
    inst_name = _identify_instrument(isr_lat)

    valid = (isr_ne > 1e8) & np.isfinite(isr_ne)
    if valid.sum() < ISR_MIN_VALID_GATES:
        return None

    prior_curves:     dict = {}   # (obs_mode, filter_type) -> (ne_at_isr_alt, ne_rmse)
    post_curves:      dict = {}
    prior_curves_mhz: dict = {}  # (obs_mode, filter_type) -> (fp_at_isr_alt, fp_rmse)
    post_curves_mhz:  dict = {}
    peak_errs:        dict = {}   # (obs_mode, filter_type) -> dict of NmF2/hmF2 errors
    peak_errs_mhz:    dict = {}   # same but NmF2 error in MHz (Δfₚ)
    tec_panels:       dict = {}   # (obs_mode, filter_type) -> nearest-arc TEC comparison data
    param_states:     dict = {}   # (obs_mode, filter_type) -> (prior_pvec, post_pvec), parametric_ekf only

    for obs_mode in OBS_MODES:
        for filter_type in FILTER_TYPES:
            result = filter_results.get(obs_mode, {}).get(filter_type)
            if result is None:
                continue

            # ── TEC-vs-measured-obs panel data (independent of EDP keys below,
            #    so a config missing EDP output can still contribute a TEC panel
            #    and vice versa) ──────────────────────────────────────────────
            tec_slices = result.get("joint_tec_slices", result.get("tec_slices"))
            clean_list = result.get("clean_list")
            sat_ids    = result.get("sat_ids", [])
            if tec_slices and clean_list:
                try:
                    arc_stats = _arc_stats_from_tec_slices(
                        tec_slices=tec_slices, clean_list=clean_list, sat_ids=sat_ids,
                    )
                    arc_lats = arc_stats["arc_lats"]
                    arc_lons = arc_stats["arc_lons"]
                    valid_arc = np.isfinite(arc_lats) & np.isfinite(arc_lons)
                    if valid_arc.any():
                        arc_tree = cKDTree(
                            np.column_stack([arc_lats[valid_arc], arc_lons[valid_arc]])
                        )
                        _arc_dist, _nearest_local = arc_tree.query([isr_lat, isr_lon])
                        idx = int(np.flatnonzero(valid_arc)[_nearest_local])

                        sl = tec_slices[idx]
                        cl = clean_list[idx]
                        measured = np.asarray(sl["measured"],  dtype=float)
                        prior    = np.asarray(sl["prior_tec"], dtype=float)
                        post     = np.asarray(sl["post_tec"],  dtype=float)

                        if cl.get("obs_source") == "IGS_ground":
                            x = np.asarray(
                                cl.get("arc_time_sec", np.arange(len(measured))),
                                dtype=float,
                            ) / 60.0
                            xlabel = "Time from arc start (min)"
                        else:
                            x = np.asarray(sl["tangent_km"], dtype=float)
                            xlabel = "Tangent altitude (km)"

                        tec_panels[(obs_mode, filter_type)] = dict(
                            x=x, xlabel=xlabel,
                            measured=measured, prior=prior, post=post,
                            label=arc_stats["arc_labels"][idx],
                            prior_rmse=float(arc_stats["arc_prior_rmse"][idx]),
                            post_rmse=float(arc_stats["arc_post_rmse"][idx]),
                            dist_deg=float(_arc_dist),
                        )
                except Exception as exc:
                    print(f"  [warn] TEC-vs-obs panel failed for "
                          f"{group_key}/{obs_mode}/{filter_type}: {exc}")

            if any(k not in result for k in
                   ("prior_edp_3d", "post_edp_3d", "alt_grid", "eds_occ")):
                continue

            prior_edp_3d = np.asarray(result["prior_edp_3d"])
            post_edp_3d  = np.asarray(result["post_edp_3d"])
            alt_grid     = np.asarray(result["alt_grid"])
            geoloc       = np.asarray(result["eds_occ"].geolocation)  # (n_geo,2): lon, lat

            mesh_pts = np.column_stack([geoloc[:, 1], geoloc[:, 0]])  # (lat, lon)
            tree = cKDTree(mesh_pts)
            _dist, nearest_idx = tree.query([isr_lat, isr_lon])

            prior_col = prior_edp_3d[:, nearest_idx]
            post_col  = post_edp_3d[:, nearest_idx]

            prior_at_isr = np.interp(isr_alt, alt_grid, prior_col)
            post_at_isr  = np.interp(isr_alt, alt_grid, post_col)

            prior_rmse = float(np.sqrt(np.mean(
                (prior_at_isr[valid] - isr_ne[valid]) ** 2)))
            post_rmse = float(np.sqrt(np.mean(
                (post_at_isr[valid] - isr_ne[valid]) ** 2)))

            # MHz (plasma-frequency) equivalents for the duplicate figure.
            fp_truth      = _ne_to_fp(isr_ne[valid])
            prior_fp_rmse = float(np.sqrt(np.mean(
                (_ne_to_fp(prior_at_isr[valid]) - fp_truth) ** 2)))
            post_fp_rmse  = float(np.sqrt(np.mean(
                (_ne_to_fp(post_at_isr[valid])  - fp_truth) ** 2)))

            key = (obs_mode, filter_type)
            prior_curves[key] = (prior_at_isr, prior_rmse)
            post_curves[key]  = (post_at_isr,  post_rmse)
            prior_curves_mhz[key] = (_ne_to_fp(prior_at_isr), prior_fp_rmse)
            post_curves_mhz[key]  = (_ne_to_fp(post_at_isr),  post_fp_rmse)

            if filter_type == "parametric_ekf":
                prior_state = result.get("prior_mean_state")
                post_state  = result.get("posterior_mean_state", result.get("post_mean_state"))
                if prior_state is not None and post_state is not None:
                    param_states[key] = (
                        np.asarray(prior_state)[:, nearest_idx],
                        np.asarray(post_state)[:, nearest_idx],
                    )

            pr_nm, pr_hm   = extract_robust_f2_peak(prior_col, alt_grid)
            po_nm, po_hm   = extract_robust_f2_peak(post_col,  alt_grid)
            isr_nm, isr_hm = extract_robust_f2_peak(isr_ne, isr_alt)

            if np.isfinite(isr_nm) and isr_nm != 0:
                prior_nm_err = 100.0 * (pr_nm - isr_nm) / isr_nm
                post_nm_err  = 100.0 * (po_nm - isr_nm) / isr_nm
            else:
                prior_nm_err = np.nan
                post_nm_err  = np.nan

            # NmF2 error in MHz: Δfₚ = fₚ(est) − fₚ(truth)
            isr_fp_nm = float(_ne_to_fp(np.asarray([isr_nm]))) if np.isfinite(isr_nm) else np.nan
            prior_nm_err_mhz = (float(_ne_to_fp(np.asarray([pr_nm]))) - isr_fp_nm) \
                if np.isfinite(isr_fp_nm) else np.nan
            post_nm_err_mhz  = (float(_ne_to_fp(np.asarray([po_nm]))) - isr_fp_nm) \
                if np.isfinite(isr_fp_nm) else np.nan

            if np.isfinite(isr_hm):
                prior_hm_err = pr_hm - isr_hm
                post_hm_err  = po_hm - isr_hm
            else:
                prior_hm_err = np.nan
                post_hm_err  = np.nan

            peak_errs[key] = dict(
                prior_nm_err=prior_nm_err, post_nm_err=post_nm_err,
                prior_hm_err=prior_hm_err, post_hm_err=post_hm_err,
            )
            peak_errs_mhz[key] = dict(
                prior_nm_err=prior_nm_err_mhz, post_nm_err=post_nm_err_mhz,
                prior_hm_err=prior_hm_err,     post_hm_err=post_hm_err,
            )

    _plot_isr_tec_vs_obs(
        tec_panels, group_key, inst_name, t_utc, solar, isr_lat, isr_lon, save_dir,
    )

    if not prior_curves:
        return None

    isr_kd_label, _ = _isr_kindat_style(isr_profile.get("kindat"))

    # Pre-compute truth band (Ne units) once for the existing Ne figure.
    ne_lo_band, ne_hi_band = _truth_fp_band(isr_ne[valid])
    isr_fp = _ne_to_fp(isr_ne)   # full-altitude MHz truth for the MHz figure

    def _draw_edp_panels(ax_pr, ax_po, *, in_mhz: bool) -> None:
        """Populate the prior and posterior EDP panels in either Ne or MHz units."""
        src_prior = prior_curves_mhz if in_mhz else prior_curves
        src_post  = post_curves_mhz  if in_mhz else post_curves
        xlabel    = "fₚ  (MHz)" if in_mhz else "Ne  (m⁻³)"

        for key, (curve, rmse) in src_prior.items():
            obs_mode, filter_type = key
            colour, ls = _CONFIG_STYLES.get(key, ("gray", "-"))
            lbl = f"{_FILTER_LABELS.get(filter_type, filter_type)} {obs_mode}  RMSE={rmse:.3f}" \
                  if in_mhz else \
                  f"{_FILTER_LABELS.get(filter_type, filter_type)} {obs_mode}  RMSE={rmse:.2e}"
            ax_pr.plot(curve, isr_alt, color=colour, ls=ls, lw=1.6, label=lbl)

        for key, (curve, rmse) in src_post.items():
            obs_mode, filter_type = key
            colour, ls = _CONFIG_STYLES.get(key, ("gray", "-"))
            lbl = f"{_FILTER_LABELS.get(filter_type, filter_type)} {obs_mode}  RMSE={rmse:.3f}" \
                  if in_mhz else \
                  f"{_FILTER_LABELS.get(filter_type, filter_type)} {obs_mode}  RMSE={rmse:.2e}"
            ax_po.plot(curve, isr_alt, color=colour, ls=ls, lw=1.6, label=lbl)

        if param_states:
            from demo_compare_kf_enkf import _draw_param_boxes
            prior_entries, post_entries = [], []
            for (obs_mode, filter_type), (prior_pvec, post_pvec) in param_states.items():
                colour = _CONFIG_STYLES.get((obs_mode, filter_type), ("gray", "-"))[0]
                prior_entries.append((f"EKF {obs_mode}", colour, prior_pvec))
                post_entries.append((f"EKF {obs_mode}",  colour, post_pvec))
            _draw_param_boxes(ax_pr, prior_entries, loc="lower right", fontsize=6.0)
            _draw_param_boxes(ax_po, post_entries,  loc="lower right", fontsize=6.0)

        for ax, title in ((ax_pr, "Prior"), (ax_po, "Posterior")):
            if in_mhz:
                # ±0.5 MHz band is trivial in MHz space.
                fp_valid = isr_fp[valid]
                ax.fill_betweenx(isr_alt[valid], fp_valid - _FP_BAND_MHZ,
                                 fp_valid + _FP_BAND_MHZ,
                                 alpha=0.25, color="black", label="±0.5 MHz band")
                ax.plot(isr_fp, isr_alt, color="black", lw=2.8, label=isr_kd_label)
                ax.set_xlim(0, max(float(np.nanmax(isr_fp)) * 1.25, 2.0))
            else:
                # ±0.5 MHz band converted to Ne (asymmetric in log space).
                ax.fill_betweenx(isr_alt[valid], ne_lo_band, ne_hi_band,
                                 alpha=0.25, color="black", label="±0.5 MHz band")
                ax.plot(isr_ne, isr_alt, color="black", lw=2.8, label=isr_kd_label)
                ax.set_xscale("log")
                ax.set_xlim(1e9, 1e13)
            ax.set_ylim(0, 800)
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Altitude  (km)")
            ax.set_title(f"{title} EDP vs. ISR truth")
            ax.legend(fontsize=7, loc="upper left")

    def _draw_rmse_scatter(ax, *, in_mhz: bool) -> None:
        src_prior = prior_curves_mhz if in_mhz else prior_curves
        src_post  = post_curves_mhz  if in_mhz else post_curves
        bar_keys   = list(src_prior.keys())
        prior_r    = [src_prior[k][1] for k in bar_keys]
        post_r     = [src_post[k][1]  for k in bar_keys]
        unit_lbl   = "MHz" if in_mhz else "Ne, m⁻³"
        _marker_for = {"gridded_kf": "o", "parametric_ekf": "s"}

        lim_hi = max(prior_r + post_r) * 1.15 if (prior_r or post_r) else 1.0
        lims = [0.0, lim_hi]
        ax.fill_between(lims, lims, [0, 0], color="green", alpha=0.06, zorder=0)
        ax.fill_between(lims, lims, [lim_hi, lim_hi], color="red", alpha=0.06, zorder=0)
        ax.plot(lims, lims, color="black", lw=1.0, ls="--", alpha=0.6, label="No change", zorder=1)
        for key, pr, po in zip(bar_keys, prior_r, post_r):
            obs_mode, filter_type = key
            colour, _ls = _CONFIG_STYLES.get(key, ("gray", "-"))
            marker = _marker_for.get(filter_type, "o")
            label  = f"{_FILTER_LABELS.get(filter_type, filter_type)} {obs_mode}"
            ax.scatter(pr, po, color=colour, marker=marker, s=90,
                       edgecolors="black", linewidths=0.8, label=label, zorder=3)
        ax.set_xlim(lims);  ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"RMSE of EDP — prior  ({unit_lbl})")
        ax.set_ylabel(f"RMSE of EDP — posterior  ({unit_lbl})")
        ax.set_title("RMSE of EDP vs. ISR truth: prior vs. posterior")
        if not in_mhz:
            ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    def _draw_peak_scatter(ax, errs: dict) -> None:
        for key, pe in errs.items():
            obs_mode, filter_type = key
            colour, _ls = _CONFIG_STYLES.get(key, ("gray", "-"))
            label = f"{_FILTER_LABELS.get(filter_type, filter_type)} {obs_mode}"
            ax.scatter(pe["prior_nm_err"], pe["prior_hm_err"],
                       facecolors="none", edgecolors=colour, s=70, lw=1.6)
            ax.scatter(pe["post_nm_err"],  pe["post_hm_err"],
                       facecolors=colour,  edgecolors=colour, s=70, label=label)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    # ── Ne figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax_prior, ax_post = axes[0, 0], axes[0, 1]
    ax_bar, ax_scatter = axes[1, 0], axes[1, 1]

    _draw_edp_panels(ax_prior, ax_post, in_mhz=False)
    _draw_rmse_scatter(ax_bar, in_mhz=False)
    _draw_peak_scatter(ax_scatter, peak_errs)
    ax_scatter.set_xlabel("NmF2 error (%)")
    ax_scatter.set_ylabel("hmF2 error (km)")
    ax_scatter.set_title("F2-peak error  (○ prior, ● posterior)")

    _suptitle = (
        f"{INSTRUMENTS[inst_name]['label']}  ·  {t_utc}  ·  {isr_kd_label}  ·  "
        f"F10.7={solar['f107']:.0f}  Ap={solar['ap']}"
    )
    fig.suptitle(_suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"isr_truth_{group_key}_{inst_name}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _print_saved(f"  ISR truth comparison saved → {out_path}")

    # ── MHz duplicate figure ──────────────────────────────────────────────────
    fig_mhz, axes_mhz = plt.subplots(2, 2, figsize=(15, 12))
    ax_pr_m, ax_po_m = axes_mhz[0, 0], axes_mhz[0, 1]
    ax_bar_m, ax_sc_m = axes_mhz[1, 0], axes_mhz[1, 1]

    _draw_edp_panels(ax_pr_m, ax_po_m, in_mhz=True)
    _draw_rmse_scatter(ax_bar_m, in_mhz=True)
    _draw_peak_scatter(ax_sc_m, peak_errs_mhz)
    ax_sc_m.set_xlabel("NmF2 error  (MHz, Δfₚ)")
    ax_sc_m.set_ylabel("hmF2 error (km)")
    ax_sc_m.set_title("F2-peak error  (○ prior, ● posterior)  [MHz]")

    fig_mhz.suptitle(_suptitle + "  [MHz]", fontsize=13, fontweight="bold")
    fig_mhz.tight_layout(rect=[0, 0, 1, 0.95])
    out_path_mhz = save_dir / f"isr_truth_{group_key}_{inst_name}_MHz.png"
    fig_mhz.savefig(out_path_mhz, dpi=130, bbox_inches="tight")
    plt.close(fig_mhz)
    _print_saved(f"  ISR truth comparison (MHz) saved → {out_path_mhz}")

    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Moved from demo_compare_kf_enkf.py
# ─────────────────────────────────────────────────────────────────────────────

def _plot_arc_innovation_diagnostic(
    arc_labels:         list,           # (n_arcs,)  PRN strings, e.g. "G15", "E03"
    arc_prior_mean:     np.ndarray,     # (n_arcs,)  mean (obs−model) prior per arc
    arc_post_mean:      np.ndarray,     # (n_arcs,)  mean (obs−model) post  per arc
    arc_prior_rmse:     np.ndarray,     # (n_arcs,)  RMSE prior per arc
    arc_post_rmse:      np.ndarray,     # (n_arcs,)  RMSE post  per arc
    arc_lats:           np.ndarray,     # (n_arcs,)  tangent-pt centroid lat
    arc_lons:           np.ndarray,     # (n_arcs,)  tangent-pt centroid lon
    all_prior:          np.ndarray,     # (n_total,) flat prior residuals → histogram
    all_post_main:      np.ndarray,     # (n_total,) flat post  residuals → histogram
    group_key:          str,
    save_dir:           str,
    filter_name:        str,            # "KF" or "EnKF" — used in titles and filename
    prior_rmse:         float,          # global prior RMSE (for title)
    post_rmse:          float,          # global post  RMSE (for title)
    all_post_raw:       np.ndarray | None = None,  # optional 2nd post (histogram only)
    post_raw_label:     str = "Post (raw)",
    mda_arc_means_list: list | None = None,  # per-step (n_arcs,) arrays (Panel A)
    mda_flat_list:      list | None = None,  # per-step flat innovation arrays (Panel D)
) -> None:
    """
    Four-panel figure summarising per-arc TEC residual statistics.

    Shared by both the voxel KF and the parametric EnKF.

    Panel A (left, tall)
        Horizontal dual-bar chart of signed mean residual (obs − model).
        Arcs sorted by |prior mean|.  Blue = prior; green = improved post
        (|post mean| < |prior mean|); red = degraded.

    Panel B (top-right)
        Prior RMSE vs posterior RMSE scatter.  Points below the y = x
        diagonal improved.  Colour = ΔRMSE = post − prior.  PRN labels.

    Panel C (middle-right)
        Geographic map showing *both* prior and posterior at each arc's
        tangent-point centroid.  A hollow circle (sized by prior RMSE,
        grey edge) represents the prior state; a filled circle (sized by
        post RMSE) is green where the posterior improved and red where it
        degraded.  PRN code annotated once per arc.

    Panel D (bottom-right)
        KDE + histogram of all sample residuals before and after the update.
    """
    import matplotlib.pyplot as _plt
    import matplotlib.patches as _mpatch
    from matplotlib.colors import Normalize as _Norm
    from matplotlib.cm import ScalarMappable as _SM
    from scipy.stats import gaussian_kde as _kde

    n_arcs = len(arc_labels)

    # Improvement based on |mean residual| (for bar) and RMSE (for map/scatter)
    imp_mean = np.abs(arc_post_mean) < np.abs(arc_prior_mean)
    imp_rmse = arc_post_rmse         < arc_prior_rmse

    # Sort bar chart by |prior mean| descending (largest bias at top)
    sort_idx = np.argsort(np.abs(arc_prior_mean))[::-1]

    # ── figure layout ──────────────────────────────────────────────────────────
    fig = _plt.figure(figsize=(18, max(10, 0.38 * n_arcs + 2)))
    gs  = fig.add_gridspec(
        3, 2,
        width_ratios  = [1.5, 1],
        height_ratios = [1, 1, 1],
        hspace=0.52, wspace=0.42,
    )
    ax_bar  = fig.add_subplot(gs[:, 0])   # left column, all 3 rows
    ax_scat = fig.add_subplot(gs[0, 1])   # top-right
    ax_map  = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())  # middle-right
    ax_hist = fig.add_subplot(gs[2, 1])   # bottom-right

    # ── Panel A: signed mean residual bar chart ────────────────────────────────
    bh    = 0.28
    y_pos = np.arange(n_arcs, dtype=float)

    _has_mda = mda_arc_means_list is not None and len(mda_arc_means_list) > 1

    if _has_mda:
        # Overlapping bars: prior → MDA steps → final posterior
        # All drawn at the same y position with partial alpha.
        # Color gradient: dark blue (prior) → light blue/teal (steps) → green/red (final).
        _n_steps   = len(mda_arc_means_list)
        _bar_h     = bh * 2.6
        # Blues palette for prior + intermediate steps
        _step_cols = _plt.cm.Blues_r(np.linspace(0.15, 0.65, _n_steps))

        for k, si in enumerate(sort_idx):
            y   = y_pos[k]
            imp = bool(imp_mean[si])

            # Draw MDA steps from bottom (prior) to top — earlier steps show through
            for _si, (_smeans, _scol) in enumerate(
                    zip(mda_arc_means_list, _step_cols)):
                _alpha = 0.38 + 0.18 * (_si / max(_n_steps - 1, 1))
                _lbl = ("Initial (prior)" if _si == 0
                        else f"MDA step {_si}")
                ax_bar.barh(y, _smeans[si], height=_bar_h,
                            color=_scol, alpha=_alpha, zorder=_si + 2,
                            label=_lbl if k == 0 else "")

            # Final posterior bar (top layer, narrower so earlier bars bleed through)
            _post_col = "#1a9641" if imp else "#d7191c"
            ax_bar.barh(y, arc_post_mean[si], height=_bar_h * 0.55,
                        color=_post_col, alpha=0.88, zorder=_n_steps + 3,
                        label=("Final (post ↓)" if (k == 0 and imp)
                               else ("Final (post ↑)" if (k == 0 and not imp)
                                     else "")))
    else:
        # Original two-bar layout (no MDA data)
        for k, si in enumerate(sort_idx):
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
    ax_bar.set_yticklabels(
        [arc_labels[sort_idx[k]] for k in range(n_arcs)],
        fontsize=8, fontfamily="monospace",
    )
    ax_bar.set_xlabel("Mean residual  obs − model  (TECU)", fontsize=9)
    ax_bar.set_title(
        f"Per-occultation mean TEC error — {filter_name}  ·  group {group_key}\n"
        f"Global RMSE: Prior {prior_rmse:.2f} TECU  →  Post {post_rmse:.2f} TECU",
        fontsize=9, fontweight="bold",
    )
    if _has_mda:
        _n_steps = len(mda_arc_means_list)
        _step_cols_leg = _plt.cm.Blues_r(np.linspace(0.15, 0.65, _n_steps))
        handles = (
            [_mpatch.Patch(color=_step_cols_leg[0], alpha=0.56,
                           label="Initial (prior)")]
            + [_mpatch.Patch(color=_step_cols_leg[_si],
                             alpha=0.38 + 0.18 * (_si / max(_n_steps - 1, 1)),
                             label=f"MDA step {_si}")
               for _si in range(1, _n_steps)]
            + [_mpatch.Patch(color="#1a9641", alpha=0.88, label="Final (post ↓ improved)"),
               _mpatch.Patch(color="#d7191c", alpha=0.88, label="Final (post ↑ degraded)")]
        )
    else:
        handles = [
            _mpatch.Patch(color="#2166ac", alpha=0.88, label="Prior  mean(obs−model)"),
            _mpatch.Patch(color="#1a9641", alpha=0.84, label="Post  ↓ |bias| reduced"),
            _mpatch.Patch(color="#d7191c", alpha=0.84, label="Post  ↑ |bias| increased"),
        ]
    ax_bar.legend(handles=handles, fontsize=8, loc="lower right")
    ax_bar.grid(axis="x", lw=0.4, alpha=0.5)

    # ── Panel B: prior RMSE vs posterior RMSE scatter ─────────────────────────
    delta_rmse = arc_post_rmse - arc_prior_rmse    # <0 = improved
    v_sc   = max(np.percentile(np.abs(delta_rmse), 95), 2.0)
    norm_sc = _Norm(-v_sc, v_sc)

    sc = ax_scat.scatter(arc_prior_rmse, arc_post_rmse,
                         c=delta_rmse, cmap="RdYlGn_r", norm=norm_sc,
                         s=60, edgecolors="k", linewidths=0.4, zorder=4)
    lim = max(np.concatenate([arc_prior_rmse, arc_post_rmse]).max() * 1.08, 5.0)
    ax_scat.plot([0, lim], [0, lim], "--", color="0.5", lw=0.9, label="no change")
    ax_scat.set_xlim(0, lim); ax_scat.set_ylim(0, lim)
    ax_scat.set_xlabel("Prior RMSE (TECU)", fontsize=8)
    ax_scat.set_ylabel("Post RMSE (TECU)",  fontsize=8)
    ax_scat.set_title(f"{filter_name}  Prior → Posterior RMSE per arc", fontsize=8)
    ax_scat.legend(fontsize=7)
    cb_sc = fig.colorbar(sc, ax=ax_scat, fraction=0.05, pad=0.02)
    cb_sc.set_label("ΔRMSE  post−prior (TECU)", fontsize=7)
    for k in range(n_arcs):
        ax_scat.annotate(arc_labels[k], (arc_prior_rmse[k], arc_post_rmse[k]),
                         fontsize=5, ha="center", va="bottom",
                         xytext=(0, 3), textcoords="offset points",
                         color="k", zorder=5)

    # ── Panel C: geographic map — prior (ring) + posterior (filled + colorbar) ──
    # Marker sizes scale with RMSE so ring vs dot diameters are comparable.
    _sz_scale = 5.0
    sz_prior = 20 + _sz_scale * arc_prior_rmse   # hollow ring
    sz_post  = 20 + _sz_scale * arc_post_rmse    # filled dot

    # ΔRMSE drives the diverging colormap (green = improved, red = degraded).
    # Symmetric limits: clip at 95th percentile of |ΔRMSE| so a few large
    # outliers don't wash out the colour range for the rest.
    delta_rmse_map = arc_post_rmse - arc_prior_rmse        # (n_arcs,)
    v_map = max(float(np.percentile(np.abs(delta_rmse_map), 95)), 2.0)
    norm_map = _Norm(-v_map, v_map)
    cmap_map = _plt.get_cmap("RdYlGn_r")   # red = worse, green = better

    # Basemap: coastlines/land/ocean under the occultation points, extent
    # padded around the arc centroids so sparse/clustered geometries both
    # render sensibly.
    _lon_pad = max((np.nanmax(arc_lons) - np.nanmin(arc_lons)) * 0.15, 5.0)
    _lat_pad = max((np.nanmax(arc_lats) - np.nanmin(arc_lats)) * 0.15, 5.0)
    ax_map.set_extent(
        [np.nanmin(arc_lons) - _lon_pad, np.nanmax(arc_lons) + _lon_pad,
         np.nanmin(arc_lats) - _lat_pad, np.nanmax(arc_lats) + _lat_pad],
        crs=ccrs.PlateCarree(),
    )
    ax_map.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
    ax_map.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
    ax_map.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.4, edgecolor="gray")
    ax_map.add_feature(cfeature.BORDERS.with_scale("110m"),   lw=0.3, edgecolor="lightgray")
    gl_map = ax_map.gridlines(draw_labels=True, lw=0.3, alpha=0.4)
    gl_map.top_labels = False
    gl_map.right_labels = False
    gl_map.xlabel_style = {"fontsize": 7}
    gl_map.ylabel_style = {"fontsize": 7}

    # Prior: hollow grey ring (no fill) — size encodes prior RMSE
    ax_map.scatter(arc_lons, arc_lats,
                   s=sz_prior, facecolors="none",
                   edgecolors="#555555", linewidths=1.6,
                   transform=ccrs.PlateCarree(), zorder=3)

    # Posterior: filled dot coloured by ΔRMSE, sized by post RMSE
    sc_map = ax_map.scatter(arc_lons, arc_lats,
                             s=sz_post,
                             c=delta_rmse_map, cmap=cmap_map, norm=norm_map,
                             alpha=0.82, edgecolors="k", linewidths=0.35,
                             transform=ccrs.PlateCarree(), zorder=4)

    # PRN labels above each arc marker
    for k in range(n_arcs):
        ax_map.annotate(arc_labels[k], (arc_lons[k], arc_lats[k]),
                        fontsize=5, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points",
                        color="k", zorder=5)

    # Colorbar for ΔRMSE
    cb_map = fig.colorbar(sc_map, ax=ax_map, fraction=0.05, pad=0.02)
    cb_map.set_label("ΔRMSE  post − prior (TECU)\n← improved   degraded →",
                     fontsize=7)

    # Small legend to explain the ring vs dot encoding
    map_handles = [
        _mpatch.Patch(facecolor="none", edgecolor="#555555",
                      linewidth=1.6, label="Prior RMSE (ring size)"),
        _mpatch.Patch(facecolor="grey", edgecolor="k",
                      linewidth=0.5, label="Post RMSE (dot size, coloured by ΔRMSE)"),
    ]
    ax_map.legend(handles=map_handles, fontsize=6, loc="best")
    ax_map.set_title(
        f"{filter_name}  Prior ○ vs Posterior ● RMSE per arc\n"
        f"Dot colour: ΔRMSE (green = improved, red = degraded)",
        fontsize=8,
    )

    # ── Panel D: residual histograms ──────────────────────────────────────────
    _has_mda_flat = mda_flat_list is not None and len(mda_flat_list) > 1

    all_arrs = [all_prior, all_post_main]
    if all_post_raw is not None:
        all_arrs.append(all_post_raw)
    if _has_mda_flat:
        all_arrs += [f[np.isfinite(f)] for f in mda_flat_list]
    finite_vals = np.concatenate([a[np.isfinite(a)] for a in all_arrs])
    lo = np.percentile(finite_vals,  1) - 5
    hi = np.percentile(finite_vals, 99) + 5
    bins = np.linspace(lo, hi, 45)
    x_k  = np.linspace(bins[0], bins[-1], 300)

    if _has_mda_flat:
        # Prior: filled histogram + KDE (reference anchor)
        _arr_pr = all_prior[np.isfinite(all_prior)]
        ax_hist.hist(_arr_pr, bins=bins, density=True, alpha=0.30,
                     color="#2166ac",
                     label=f"Prior  μ={np.nanmean(all_prior):+.1f}  σ={np.nanstd(all_prior):.1f}")
        try:
            ax_hist.plot(x_k, _kde(_arr_pr)(x_k), color="#2166ac", lw=2.0)
        except Exception:
            pass

        # MDA intermediate steps: KDE curves only (avoid filled-histogram clutter)
        _n_flat = len(mda_flat_list)
        _flat_cols = _plt.cm.Blues_r(np.linspace(0.15, 0.65, _n_flat))
        for _fi, (_farr, _fcol) in enumerate(zip(mda_flat_list, _flat_cols)):
            _a = _farr[np.isfinite(_farr)]
            _mu, _sg = float(np.nanmean(_farr)), float(np.nanstd(_farr))
            _lbl = (f"Initial (prior, rep)  μ={_mu:+.1f} σ={_sg:.1f}"
                    if _fi == 0
                    else f"MDA step {_fi}  μ={_mu:+.1f} σ={_sg:.1f}")
            # Light-fill histogram + curve
            ax_hist.hist(_a, bins=bins, density=True,
                         alpha=0.18 + 0.06 * (_fi / max(_n_flat - 1, 1)),
                         color=_fcol, label=_lbl)
            try:
                ax_hist.plot(x_k, _kde(_a)(x_k), color=_fcol,
                             lw=1.3 + 0.4 * (_fi / max(_n_flat - 1, 1)),
                             alpha=0.75 + 0.20 * (_fi / max(_n_flat - 1, 1)))
            except Exception:
                pass

        # Final posterior: filled + KDE (prominent anchor)
        _arr_po = all_post_main[np.isfinite(all_post_main)]
        ax_hist.hist(_arr_po, bins=bins, density=True, alpha=0.35,
                     color="#1a9641",
                     label=f"Final post  μ={np.nanmean(all_post_main):+.1f}  σ={np.nanstd(all_post_main):.1f}")
        try:
            ax_hist.plot(x_k, _kde(_arr_po)(x_k), color="#1a9641", lw=2.2)
        except Exception:
            pass

        if all_post_raw is not None:
            _arr_rw = all_post_raw[np.isfinite(all_post_raw)]
            ax_hist.hist(_arr_rw, bins=bins, density=True, alpha=0.28,
                         color="#fdae61",
                         label=f"{post_raw_label}  μ={np.nanmean(all_post_raw):+.1f}  σ={np.nanstd(all_post_raw):.1f}")
            try:
                ax_hist.plot(x_k, _kde(_arr_rw)(x_k), color="#fdae61", lw=1.6)
            except Exception:
                pass

        ax_hist.set_title(
            f"{filter_name}  residual distribution per MDA iteration", fontsize=8)
    else:
        hist_series = [
            (all_prior,     "#2166ac",
             f"Prior      μ={np.nanmean(all_prior):+.1f}  σ={np.nanstd(all_prior):.1f}"),
            (all_post_main, "#1a9641",
             f"Post {filter_name}   μ={np.nanmean(all_post_main):+.1f}  σ={np.nanstd(all_post_main):.1f}"),
        ]
        if all_post_raw is not None:
            hist_series.append(
                (all_post_raw, "#fdae61",
                 f"{post_raw_label}  μ={np.nanmean(all_post_raw):+.1f}  σ={np.nanstd(all_post_raw):.1f}")
            )

        for arr, col, lbl in hist_series:
            ax_hist.hist(arr[np.isfinite(arr)], bins=bins,
                         density=True, alpha=0.42, color=col, label=lbl)
            try:
                kde_fn = _kde(arr[np.isfinite(arr)])
                ax_hist.plot(x_k, kde_fn(x_k), color=col, lw=1.6)
            except Exception:
                pass

        ax_hist.set_title(
            f"{filter_name}  residual distribution (all samples)", fontsize=8)

    ax_hist.axvline(0, color="k", lw=0.8, linestyle="--")
    ax_hist.set_xlabel("Residual  obs − model  (TECU)", fontsize=8)
    ax_hist.set_ylabel("Density", fontsize=8)
    ax_hist.legend(fontsize=7)
    ax_hist.grid(lw=0.3, alpha=0.4)

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    tag      = filter_name.lower().replace(" ", "_")
    out_path = os.path.join(save_dir, f"{tag}_arc_innovations_{group_key}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    _plt.close(fig)
    _print_saved(f"  [{filter_name}] Arc innovation diagnostic saved → {out_path}")


def _plot_covariance_panels_labeled(
    result: dict,
    save_dir: str,
    group_key: str,
    label: str,
    *,
    hmF2_ref_km: float | None = None,
) -> str:
    """
    Four-panel figure showing the EDP prior and posterior covariance structure.

    Layout (2 rows × 2 cols):
      Row 0 — Prior:     Alt-Alt correlation  |  Horizontal correlation at hmF2
      Row 1 — Posterior: Alt-Alt correlation  |  Horizontal correlation at hmF2

    Parameters
    ----------
    result      : result dict with keys prior_P, post_P, alt_grid, eds_occ,
                  region, prior_edp_3d, lats, lons, time_window.
    label       : short label for the title, e.g. "KF" or "EnKF".
    hmF2_ref_km : altitude for the horizontal slice; defaults to prior F2 peak.
    """
    import warnings

    alt_grid  = result["alt_grid"]
    prior_P   = result["prior_P"]
    post_P    = result["post_P"]
    eds_occ   = result["eds_occ"]
    region    = result["region"]
    prior_edp = result["prior_edp_3d"]

    n_height  = len(alt_grid)
    verts_geo = eds_occ.geolocation      # (n_geo, 2): col0=lon, col1=lat
    n_geo     = verts_geo.shape[0]
    n_sv      = n_height * n_geo

    centre_idx = _roi_centre_idx(verts_geo, region)

    if hmF2_ref_km is None:
        _, hmF2_ref_km = extract_robust_f2_peak(prior_edp[:, centre_idx], alt_grid)
        if np.isnan(hmF2_ref_km):
            hmF2_ref_km = float(alt_grid[n_height // 2])
    alt_ref_idx  = int(np.argmin(np.abs(alt_grid - hmF2_ref_km)))
    true_alt_ref = float(alt_grid[alt_ref_idx])

    def _alt_corr(P_aug):
        P_grid = P_aug[:n_sv, :n_sv]
        P_4d   = P_grid.reshape(n_height, n_geo, n_height, n_geo)
        cov    = P_4d.mean(axis=(1, 3))
        std    = np.sqrt(np.maximum(np.diag(cov), 0.0))
        outer  = np.outer(std, std)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return cov / np.where(outer == 0, 1e-10, outer)

    def _horiz_corr(P_aug):
        P_grid  = P_aug[:n_sv, :n_sv]
        P_4d    = P_grid.reshape(n_height, n_geo, n_height, n_geo)
        cov_row = P_4d[alt_ref_idx, centre_idx, alt_ref_idx, :]
        var_ctr = float(P_4d[alt_ref_idx, centre_idx, alt_ref_idx, centre_idx])
        var_all = P_4d[alt_ref_idx, :, alt_ref_idx, :]
        std_all = np.sqrt(np.maximum(np.diag(var_all), 0.0))
        std_ctr = float(np.sqrt(max(var_ctr, 0.0)))
        denom   = std_ctr * std_all
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return cov_row / np.where(denom == 0, 1e-10, denom)

    prior_alt_corr = _alt_corr(prior_P)
    post_alt_corr  = _alt_corr(post_P)
    prior_horiz    = _horiz_corr(prior_P)
    post_horiz     = _horiz_corr(post_P)

    lats_c = result["lats"]
    lons_c = result["lons"]
    clon   = float(np.nanmean(lons_c)) if lons_c else 0.0
    clat   = float(np.nanmean(lats_c)) if lats_c else 0.0
    proj   = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)

    alt_extent = [float(alt_grid[0]), float(alt_grid[-1]),
                  float(alt_grid[0]), float(alt_grid[-1])]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"{label} — EDP Covariance Structure\n"
        f"{result.get('time_window', group_key)}  |  {region}\n"
        f"Horizontal slice at {true_alt_ref:.0f} km  ·  ★ = centre vertex",
        fontsize=12,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.06, right=0.97, top=0.88, bottom=0.07,
                           wspace=0.30, hspace=0.35)

    row_labels = ["Prior", "Posterior"]
    corr_pairs = [(prior_alt_corr, prior_horiz), (post_alt_corr, post_horiz)]

    for row, (row_lbl, (alt_corr, horiz_corr)) in enumerate(
        zip(row_labels, corr_pairs)
    ):
        ax_aa = fig.add_subplot(gs[row, 0])
        pcm = ax_aa.imshow(
            alt_corr, cmap="coolwarm", vmin=-1, vmax=1,
            extent=alt_extent, origin="lower", aspect="auto",
        )
        ax_aa.axhline(true_alt_ref, color="gold", lw=1.0, ls="--", alpha=0.8)
        ax_aa.axvline(true_alt_ref, color="gold", lw=1.0, ls="--", alpha=0.8)
        ax_aa.set_xlabel("Altitude (km)", fontsize=9)
        ax_aa.set_ylabel("Altitude (km)", fontsize=9)
        ax_aa.set_title(f"{row_lbl} — Alt-Alt Correlation", fontsize=10)
        fig.colorbar(pcm, ax=ax_aa, label="Pearson r", fraction=0.046, pad=0.04)

        ax_gl = fig.add_subplot(gs[row, 1], projection=proj)
        ax_gl.set_global()
        ax_gl.add_feature(cfeature.LAND,  facecolor="whitesmoke", zorder=0)
        ax_gl.add_feature(cfeature.OCEAN, facecolor="aliceblue",  zorder=0)
        ax_gl.add_feature(cfeature.COASTLINE.with_scale("110m"),
                          lw=0.4, edgecolor="gray")
        ax_gl.gridlines(lw=0.2, alpha=0.3)

        try:
            tc = ax_gl.tripcolor(
                verts_geo[:, 0], verts_geo[:, 1], eds_occ.mesh,
                horiz_corr,
                transform=ccrs.Geodetic(),
                cmap="coolwarm", shading="flat",
                vmin=-1, vmax=1, zorder=1,
            )
            cb = fig.colorbar(tc, ax=ax_gl, orientation="horizontal",
                              shrink=0.75, pad=0.04, fraction=0.04)
            cb.set_label("Pearson r", fontsize=8)
        except Exception:
            pass

        ctr_lon = float(verts_geo[centre_idx, 0])
        ctr_lat = float(verts_geo[centre_idx, 1])
        ax_gl.plot(ctr_lon, ctr_lat, transform=ccrs.Geodetic(),
                   marker="*", color="gold", ms=12, mec="black", mew=0.8, zorder=8)
        _draw_roi_boundary(ax_gl, region)
        ax_gl.set_title(
            f"{row_lbl} — Horizontal Correlation at {true_alt_ref:.0f} km",
            fontsize=10,
        )

    os.makedirs(save_dir, exist_ok=True)
    safe_key  = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    safe_label = label.lower().replace(" ", "_")
    plot_path = os.path.join(save_dir, f"compare_{safe_key}_{safe_label}_covariance.png")
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    _print_saved(f"  Covariance plot ({label}) saved → {plot_path}")
    return plot_path


def _plot_ekf_param_covariance_panels(
    result: dict,
    save_dir: str,
    group_key: str,
    label: str = "EKF",
) -> str:
    """
    Four-panel figure showing EKF_Param's prior/posterior covariance structure
    in its native parametric state space (N_STATE Chapman/IRI parameters ×
    n_geo grid points).  Analogous to _plot_covariance_panels but NOT
    interchangeable with it — the gridded KF's prior_P/post_P are indexed
    (n_alt, n_geo) Ne-space, while EKF_Param's are indexed (N_STATE, n_geo)
    parameter space, so the "altitude" axis there is replaced here by a
    "parameter" axis, and the horizontal-correlation reference is
    log10(NmF2) (the density-like parameter) instead of a chosen altitude.

    Layout (2 rows × 2 cols):
      Row 0 — Prior:     Param-Param correlation | Horizontal corr. of log10(NmF2)
      Row 1 — Posterior: Param-Param correlation | Horizontal corr. of log10(NmF2)
    """
    from Ionosphere_Tomography_Inverter.ionospheric_state import (
        N_STATE, PARAM_NAMES, I_LOG_NMF2,
    )

    prior_P = result["prior_P"]
    post_P  = result["post_P"]
    eds_occ = result["eds_occ"]
    region  = result["region"]

    verts_geo  = eds_occ.geolocation      # (n_geo, 2): col0=lon, col1=lat
    n_geo      = verts_geo.shape[0]
    centre_idx = _roi_centre_idx(verts_geo, region)
    ref_idx    = I_LOG_NMF2

    def _param_corr(P_aug):
        P_4d  = P_aug.reshape(N_STATE, n_geo, N_STATE, n_geo)
        cov   = P_4d.mean(axis=(1, 3))                    # (N_STATE, N_STATE)
        std   = np.sqrt(np.maximum(np.diag(cov), 0.0))
        outer = np.outer(std, std)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return cov / np.where(outer == 0, 1e-10, outer)

    def _horiz_corr(P_aug):
        P_4d    = P_aug.reshape(N_STATE, n_geo, N_STATE, n_geo)
        cov_row = P_4d[ref_idx, centre_idx, ref_idx, :]           # (n_geo,)
        var_ctr = float(P_4d[ref_idx, centre_idx, ref_idx, centre_idx])
        var_all = P_4d[ref_idx, :, ref_idx, :]                    # (n_geo, n_geo)
        std_all = np.sqrt(np.maximum(np.diag(var_all), 0.0))
        std_ctr = float(np.sqrt(max(var_ctr, 0.0)))
        denom   = std_ctr * std_all
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return cov_row / np.where(denom == 0, 1e-10, denom)

    prior_pp = _param_corr(prior_P)
    post_pp  = _param_corr(post_P)
    prior_hz = _horiz_corr(prior_P)
    post_hz  = _horiz_corr(post_P)

    lats_c = result["lats"]
    lons_c = result["lons"]
    clon   = float(np.nanmean(lons_c)) if lons_c else 0.0
    clat   = float(np.nanmean(lats_c)) if lats_c else 0.0
    proj   = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"{label} — Parametric-State Covariance Structure\n"
        f"{result.get('time_window', group_key)}  |  {region}\n"
        f"Horizontal slice at reference param log10(NmF2)  ·  ★ = centre vertex",
        fontsize=12,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig,
                            left=0.06, right=0.97, top=0.88, bottom=0.07,
                            wspace=0.30, hspace=0.35)

    row_labels = ["Prior", "Posterior"]
    corr_pairs = [(prior_pp, prior_hz), (post_pp, post_hz)]

    for row, (row_lbl, (pp_corr, hz_corr)) in enumerate(zip(row_labels, corr_pairs)):
        # ── Left: parameter-parameter correlation ────────────────────────────
        ax_pp = fig.add_subplot(gs[row, 0])
        pcm = ax_pp.imshow(pp_corr, cmap="coolwarm", vmin=-1, vmax=1, origin="lower")
        ax_pp.set_xticks(range(N_STATE))
        ax_pp.set_xticklabels(PARAM_NAMES, rotation=45, ha="right", fontsize=7)
        ax_pp.set_yticks(range(N_STATE))
        ax_pp.set_yticklabels(PARAM_NAMES, fontsize=7)
        ax_pp.set_title(f"{row_lbl} — Param-Param Correlation", fontsize=10)
        fig.colorbar(pcm, ax=ax_pp, label="Pearson r", fraction=0.046, pad=0.04)

        # ── Right: horizontal correlation globe ───────────────────────────────
        ax_gl = fig.add_subplot(gs[row, 1], projection=proj)
        ax_gl.set_global()
        ax_gl.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
        ax_gl.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
        ax_gl.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.4, edgecolor="gray")
        ax_gl.gridlines(lw=0.2, alpha=0.3)

        try:
            tc = ax_gl.tripcolor(
                verts_geo[:, 0], verts_geo[:, 1], eds_occ.mesh,
                hz_corr,
                transform=ccrs.Geodetic(),
                cmap="coolwarm", shading="flat",
                vmin=-1, vmax=1, zorder=1,
            )
            cb = fig.colorbar(tc, ax=ax_gl, orientation="horizontal",
                               shrink=0.75, pad=0.04, fraction=0.04)
            cb.set_label("Pearson r", fontsize=8)
        except Exception:
            pass

        ctr_lon = float(verts_geo[centre_idx, 0])
        ctr_lat = float(verts_geo[centre_idx, 1])
        ax_gl.plot(ctr_lon, ctr_lat, transform=ccrs.Geodetic(),
                   marker="*", color="gold", ms=12, mec="black", mew=0.8, zorder=8)
        _draw_roi_boundary(ax_gl, region)
        ax_gl.set_title(
            f"{row_lbl} — Horizontal Correlation of log10(NmF2)", fontsize=10,
        )

    os.makedirs(save_dir, exist_ok=True)
    safe_key   = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    safe_label = label.lower().replace(" ", "_")
    plot_path  = os.path.join(save_dir, f"compare_{safe_key}_{safe_label}_covariance.png")
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    _print_saved(f"  EKF parametric covariance plot ({label}) saved → {plot_path}")
    return plot_path



def _render_globe_ax(
    ax,
    fig,
    res: dict,
    alt_grid: np.ndarray,
    label: str,
    isr_site: tuple[float, float] | None = None,
    n_occ_shown: int = 3,
    shown_indices: list[int] | None = None,
    occ_colours_override: list | None = None,
) -> None:
    """
    Render a globe panel (ΔNe map + raypaths + ROI) onto an existing cartopy axes.

    Parameters
    ----------
    ax                   : cartopy GeoAxes to draw on.
    fig                  : parent Figure (for colorbar).
    res                  : result dict (from process_group or _run_parametric_enkf).
    alt_grid             : altitude grid in km.
    label                : title label string (e.g. "KF" or "EnKF").
    isr_site             : (lon_deg, lat_deg) of the ISR site, or None.
    n_occ_shown          : maximum number of occultation raypaths to draw (used
                           only when shown_indices is None).
    shown_indices        : explicit list of occultation indices to draw.  When
                           provided, overrides n_occ_shown so the globe shows
                           exactly the same occultations as the top panels.
    occ_colours_override : per-occultation colours aligned with the full
                           occultation list.  When provided, the colours on the
                           globe match the top-row TEC / EDP panels exactly.
    """
    from collections import defaultdict

    from demo_group import (
        CONSTELLATION_CONFIG, _CONST_FALLBACK_CMAP, _parse_time_window,
        _draw_terminator, _draw_leo_path, _draw_raypath,
    )

    eds_occ    = res["eds_occ"]
    clean_list = res.get("clean_list", [])
    prior_edp  = res["prior_edp_3d"]
    _jnt = res.get("joint_post_edp_3d")
    post_edp   = _jnt if _jnt is not None else res["post_edp_3d"]
    sat_ids    = res.get("sat_ids", [])
    region     = res["region"]

    verts_geo = eds_occ.geolocation
    tris_geo  = eds_occ.mesh
    n_geo     = verts_geo.shape[0]

    centre_idx = _roi_centre_idx(verts_geo, region)
    prior_centre = prior_edp[:, centre_idx]
    post_centre  = post_edp[:, centre_idx]
    pr_nm, pr_hm = extract_robust_f2_peak(prior_centre, alt_grid)
    po_nm, po_hm = extract_robust_f2_peak(post_centre, alt_grid)

    if not np.isnan(pr_hm):
        alt_idx     = int(np.argmin(np.abs(alt_grid - pr_hm)))
        delta_slice = post_edp[alt_idx, :] - prior_edp[alt_idx, :]
        hmF2_label  = f"~{alt_grid[alt_idx]:.0f} km"
    else:
        delta_slice = np.zeros(n_geo)
        hmF2_label  = "F2 unavailable"

    # Per-occultation colours — prefer the override so globe matches top panels
    n_occ = len(clean_list)
    if occ_colours_override is not None:
        occ_colours = list(occ_colours_override)
    else:
        # Fallback: build per-constellation colours independently
        const_counts = defaultdict(int)
        occ_const    = []
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
            cmap_c    = mpl.colormaps.get_cmap(cmap_name)
            n_in      = const_counts[const]
            idx_in    = const_counter[const]
            t = (0.40 + 0.50 * (idx_in / max(n_in - 1, 1))) if n_in > 1 else 0.70
            occ_colours.append(cmap_c(t))
            const_counter[const] += 1

    ax.set_global()
    ax.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="aliceblue",  zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.5, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS.with_scale("110m"),   lw=0.3, edgecolor="lightgray")
    ax.gridlines(lw=0.3, alpha=0.4)

    try:
        max_delta = float(np.nanmax(np.abs(delta_slice)))
        if max_delta > 0:
            tc = ax.tripcolor(
                verts_geo[:, 0], verts_geo[:, 1], tris_geo,
                delta_slice,
                transform=ccrs.Geodetic(),
                cmap="coolwarm", shading="flat",
                vmin=-max_delta, vmax=max_delta,
                zorder=1,
            )
            cbar = fig.colorbar(tc, ax=ax, orientation="horizontal",
                                shrink=0.75, pad=0.04, fraction=0.04)
            cbar.set_label(f"ΔNe at hmF2 ({hmF2_label}) [m⁻³]", fontsize=7)
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.update_ticks()
    except Exception:
        pass

    try:
        mean_ts = _parse_time_window(res["time_window"])
        _draw_terminator(ax, mean_ts, zorder=5)
    except Exception:
        pass

    # Show only a subset of occultations to keep the plot readable.
    # Use the caller-supplied indices when available so the globe shows exactly
    # the same occultations (same colours) as the top TEC / EDP panels.
    if shown_indices is None:
        shown_indices = list(range(min(n_occ_shown, n_occ)))
    shown_clean   = [clean_list[i] for i in shown_indices if i < len(clean_list)]
    shown_colours = [occ_colours[i] for i in shown_indices if i < len(occ_colours)]
    _draw_leo_path(ax, shown_clean, shown_colours, zorder=5)

    ctr_lon = float(verts_geo[centre_idx, 0])
    ctr_lat = float(verts_geo[centre_idx, 1])
    ax.plot(ctr_lon, ctr_lat, transform=ccrs.Geodetic(),
            marker="*", color="yellow", ms=12, mec="black", mew=0.8,
            zorder=8, label="Centre EDP vertex")

    _draw_roi_boundary(ax, region)

    ray_defs = [
        ("top",     "solid",  2.0),
        ("tec-max", "dashed", 1.8),
        ("bottom",  "dotted", 1.5),
    ]
    for i, (cl, col) in enumerate(zip(shown_clean, shown_colours)):
        LEO  = cl["LEO"]
        GNSS = cl["GNSS"]
        tec  = cl["tec"]
        tang = cl["tangent_km"]
        idx_top    = int(np.argmax(tang))
        idx_bottom = int(np.argmin(tang))
        idx_tecmax = int(np.argmax(tec))
        for (rtype, ls, lw), ridx in zip(
            ray_defs, [idx_top, idx_tecmax, idx_bottom]
        ):
            lbl = rtype if i == 0 else None
            _draw_raypath(ax, LEO, GNSS, ridx,
                          color=col, ls=ls, lw=lw, label=lbl, zorder=6,
                          TP=(ridx == idx_tecmax))

    if isr_site is not None:
        ax.plot(isr_site[0], isr_site[1], transform=ccrs.Geodetic(),
                marker="^", ms=10, color="limegreen",
                mec="black", mew=1.0, zorder=9, label="MH ISR")

    ax.set_title(
        f"{label} Globe — ΔNe at hmF2 ({hmF2_label})\n"
        f"RMSE {res['prior_tec_rmse']:.2f}→{res['post_tec_rmse']:.2f} TECU",
        fontsize=9,
    )
    ax.legend(loc="lower left", fontsize=6, framealpha=0.75)



def plot_occultation_prior_post_truth(
    res: dict,
    occ_idx: int,
    alt_grid: np.ndarray,
    group_key: str,
    save_dir: str,
    label: str = "",
    isr_profile: dict | None = None,
    isr_site: tuple[float, float] | None = None,
) -> str | None:
    """
    Single-occultation "what did assimilation do to this ray?" diagnostic.

    ┌───────────────────────┬─────────────────────────┐
    │ [0,0] TEC prior/post  │ [0,1] Orthographic       │
    │ vs. measured, vs.     │ geometry for this        │
    │ tangent altitude      │ occultation only         │
    ├───────────────────────┼─────────────────────────┤
    │ [1,0] Prior Ne curtain│ [1,1] EDP-vs-alt: ISR    │
    │ (top) / Posterior Ne  │ truth, prior & post @    │
    │ curtain (bottom),     │ TEC-max tangent point    │
    │ along the raypath     │                          │
    └───────────────────────┴─────────────────────────┘

    [0,0] mirrors plot_kf_enkf_comparison's TEC-vs-tangent-altitude
    styling. [0,1] reuses _render_globe_ax with shown_indices=[occ_idx] so
    only this ray's geometry is drawn on the ΔNe globe.

    [1,0] is 2 stacked pcolormesh curtains (prior on top, posterior on
    bottom) -- altitude (y) vs along-occultation distance (x), cividis
    colormap, shared LogNorm scale and one shared colorbar -- built by
    nearest-neighbour (cKDTree) lookup of each along-track tangent-point
    (lat, lon) into eds_occ.geolocation and pulling the full altitude
    column out of prior_edp_3d / post_edp_3d at the nearest mesh vertex,
    i.e. genuine mesh-resolved horizontal structure along the ray (the
    "EDP along the raypath, before + after"). A white dashed vertical line
    marks the along-track sample closest to isr_site, if given; a magenta
    dotted vertical line marks the ray's TEC-max tangent-point sample.

    [1,1] is a plain EDP-vs-altitude line plot (log-x electron density,
    linear-y altitude) comparing the ISR truth sounding against the
    prior/posterior EDP columns pulled from the ray's "TEC-max" tangent
    point -- the same deepest-tangent-altitude sample used as the TEC-max
    proxy elsewhere in the codebase (see
    demo_compare_kf_enkf._arc_representative_tangent), i.e. the
    along-track column most representative of the ray's peak columnar
    electron content.

    Parameters
    ----------
    res         : result dict from process_group() / EKF equivalent -- must
                  contain clean_list, eds_occ, prior_edp_3d, post_edp_3d,
                  tec_slices (or joint_tec_slices), lats, lons, sat_ids.
    occ_idx     : index into res["clean_list"] of the occultation to plot
                  (must be a genuine RO ray, i.e. len(tangent_km) > 1 --
                  collapsed single-epoch IGS arcs can't be curtain-plotted).
    alt_grid    : shared altitude grid (km), length n_alt.
    group_key   : used for figure title / filename.
    save_dir    : output directory (created if missing).
    label       : short tag (e.g. "ro_igs_gridded_kf") folded into the title
                  and filename to disambiguate obs_mode/filter combos.
    isr_profile : optional dict with "alt_km"/"ne" (see _isr_edp_to_profile
                  in demo_isr_da_comparison.py) for the ISR truth row; if
                  None, the truth row is left blank with an annotation.
    isr_site    : optional (lon, lat) tuple for the vertical marker line.

    Returns
    -------
    str | None : path to the saved PNG, or None if occ_idx isn't usable
                 (too few samples / missing required result keys).
    """
    from matplotlib.colors import LogNorm
    from demo_compare_kf_enkf import _tangent_latlon_single
    from demo_verification import _haversine_km

    required = ("clean_list", "eds_occ", "prior_edp_3d", "post_edp_3d")
    if any(res.get(k) is None for k in required):
        print(f"  [occ-diag] skip {label or group_key}: missing required result keys")
        return None

    clean_list = res["clean_list"]
    if occ_idx < 0 or occ_idx >= len(clean_list):
        print(f"  [occ-diag] skip {label or group_key}: occ_idx {occ_idx} out of range")
        return None

    occ = clean_list[occ_idx]
    gnss_ecef  = np.asarray(occ["GNSS"])
    leo_ecef   = np.asarray(occ["LEO"])
    tangent_km = np.asarray(occ["tangent_km"])
    n_samp = tangent_km.shape[0]
    if n_samp < 2:
        print(f"  [occ-diag] skip {label or group_key} occ {occ_idx}: "
              f"only {n_samp} sample(s) -- not a genuine RO ray "
              f"(likely a collapsed IGS arc)")
        return None

    tec_slices = res.get("tec_slices") or res.get("joint_tec_slices")
    if not tec_slices or occ_idx >= len(tec_slices):
        print(f"  [occ-diag] skip {label or group_key}: no tec_slices for occ {occ_idx}")
        return None
    sl = tec_slices[occ_idx]

    alt_grid = np.asarray(alt_grid)
    n_alt = len(alt_grid)

    # ── Along-track tangent-point geometry ─────────────────────────────────
    track_lat = np.empty(n_samp)
    track_lon = np.empty(n_samp)
    for i in range(n_samp):
        la, lo = _tangent_latlon_single(gnss_ecef[:, i], leo_ecef[:, i])
        track_lat[i] = la
        track_lon[i] = lo

    along_km = np.zeros(n_samp)
    for i in range(1, n_samp):
        along_km[i] = along_km[i - 1] + _haversine_km(
            track_lat[i - 1], track_lon[i - 1], track_lat[i], track_lon[i]
        )

    # ── Nearest-mesh-vertex lookup for the curtain fields ───────────────────
    eds_occ  = res["eds_occ"]
    geoloc   = np.asarray(eds_occ.geolocation)                  # (n_geo,2) = lon, lat
    mesh_pts = np.column_stack([geoloc[:, 1], geoloc[:, 0]])    # (lat, lon)
    tree     = cKDTree(mesh_pts)
    _dist, nearest_idx = tree.query(np.column_stack([track_lat, track_lon]))

    n_geo = geoloc.shape[0]
    prior_edp_3d = np.asarray(res["prior_edp_3d"]).reshape(n_alt, n_geo)
    post_edp_3d  = np.asarray(res["post_edp_3d"]).reshape(n_alt, n_geo)

    prior_curtain = prior_edp_3d[:, nearest_idx]   # (n_alt, n_samp)
    post_curtain  = post_edp_3d[:, nearest_idx]    # (n_alt, n_samp)

    # ── ISR truth sounding, interpolated onto alt_grid (line-plot only now) ──
    truth_ne = np.full(n_alt, np.nan)
    have_truth = False
    if isr_profile is not None:
        t_alt = np.asarray(isr_profile["alt_km"], dtype=float)
        t_ne  = np.asarray(isr_profile["ne"], dtype=float)
        order = np.argsort(t_alt)
        t_alt, t_ne = t_alt[order], t_ne[order]
        valid = np.isfinite(t_alt) & np.isfinite(t_ne)
        if valid.sum() >= 2:
            truth_ne = np.interp(alt_grid, t_alt[valid], t_ne[valid],
                                  left=np.nan, right=np.nan)
            have_truth = True

    # ── TEC-max tangent point (deepest-tangent-altitude sample) -- same
    #    proxy used by demo_compare_kf_enkf._arc_representative_tangent --
    #    and the prior/posterior EDP columns pulled at that along-track
    #    sample, for the EDP-vs-altitude comparison panel below.
    tec_max_idx      = int(np.argmin(tangent_km))
    prior_ne_tecmax  = prior_curtain[:, tec_max_idx]
    post_ne_tecmax   = post_curtain[:, tec_max_idx]
    tecmax_along_km  = along_km[tec_max_idx]

    # along-track sample closest to isr_site, for a vertical marker line
    site_along_km = None
    if isr_site is not None:
        site_lon, site_lat = isr_site
        d_site = np.array([
            _haversine_km(track_lat[i], track_lon[i], site_lat, site_lon)
            for i in range(n_samp)
        ])
        site_along_km = along_km[int(np.argmin(d_site))]

    # ── Shared colour scale across the 2 curtains ────────────────────────────
    stacked = [prior_curtain, post_curtain]
    finite_pos = np.concatenate([
        c[np.isfinite(c) & (c > 0)].ravel() for c in stacked
    ])
    if finite_pos.size == 0:
        finite_pos = np.array([1e9, 1e12])
    vmin = max(float(np.percentile(finite_pos, 1)), 1e7)
    vmax = float(np.percentile(finite_pos, 99))
    if vmax <= vmin:
        vmax = vmin * 10.0
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # ── Figure layout ─────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    sat_ids = res.get("sat_ids", [])
    prn = sat_ids[occ_idx][1] if occ_idx < len(sat_ids) else f"Occ{occ_idx + 1}"

    lats_c = res.get("lats")
    lons_c = res.get("lons")
    clon = float(np.nanmean(lons_c)) if lons_c else float(np.nanmean(track_lon))
    clat = float(np.nanmean(lats_c)) if lats_c else float(np.nanmean(track_lat))
    proj = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)

    fig = plt.figure(figsize=(14, 12))
    title_label = f" ({label})" if label else ""
    fig.suptitle(
        f"Occultation Diagnostic — {prn}{title_label}\n{group_key}",
        fontsize=13, y=1.02,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.32)
    ax_tec = fig.add_subplot(gs[0, 0])
    ax_geo = fig.add_subplot(gs[0, 1], projection=proj)
    gs_raypath = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1, 0], hspace=0.15
    )
    ax_pr  = fig.add_subplot(gs_raypath[0, 0])
    ax_po  = fig.add_subplot(gs_raypath[1, 0], sharex=ax_pr, sharey=ax_pr)
    ax_edp = fig.add_subplot(gs[1, 1])

    # ── Left: TEC prior/posterior vs. measured ───────────────────────────────
    ax_tec.plot(sl["measured"],  sl["tangent_km"], color="black",     lw=2.0,               label="Measured")
    ax_tec.plot(sl["prior_tec"], sl["tangent_km"], color="royalblue", lw=1.8, ls="--",       label="Prior")
    ax_tec.plot(sl["post_tec"],  sl["tangent_km"], color="firebrick", lw=1.8, ls="-.",       label="Posterior")
    ax_tec.set_xlabel("TEC (TECU)", fontsize=10)
    ax_tec.set_ylabel("Tangent Altitude (km)", fontsize=10)
    ax_tec.set_title(
        f"TEC: prior {res.get('prior_tec_rmse', float('nan')):.2f} → "
        f"post {res.get('post_tec_rmse', float('nan')):.2f} TECU",
        fontsize=10,
    )
    ax_tec.legend(fontsize=8, loc="best", framealpha=0.85)
    ax_tec.grid(True, alpha=0.3, ls=":")

    # ── Middle: single-occultation geometry ───────────────────────────────────
    try:
        _render_globe_ax(
            ax_geo, fig, res, alt_grid, label or "Occ",
            isr_site=isr_site, shown_indices=[occ_idx],
        )
    except Exception:
        ax_geo.set_title("Geometry unavailable", fontsize=9)

    # ── [1,0]: prior/posterior Ne curtains along the raypath (before/after) ──
    mesh = None
    for ax, field, title in (
        (ax_pr, prior_curtain, "Prior"),
        (ax_po, post_curtain,  "Posterior"),
    ):
        mesh = ax.pcolormesh(along_km, alt_grid, field, shading="nearest",
                              cmap="cividis", norm=norm)
        ax.set_ylabel("Alt (km)", fontsize=8.5)
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=7.5)
        if site_along_km is not None:
            ax.axvline(site_along_km, color="white", lw=1.2, ls="--", alpha=0.85)
        ax.axvline(tecmax_along_km, color="magenta", lw=1.2, ls=":", alpha=0.9)

    ax_po.set_xlabel("Along-occultation distance (km)", fontsize=9)
    plt.setp(ax_pr.get_xticklabels(), visible=False)
    cbar = fig.colorbar(mesh, ax=[ax_pr, ax_po], fraction=0.046, pad=0.02)
    cbar.set_label("Electron Density (m⁻³)", fontsize=9)

    # ── [1,1]: EDP vs altitude -- ISR truth vs prior/posterior @
    #    the ray's TEC-max tangent point ──────────────────────────────────
    if have_truth:
        ax_edp.plot(truth_ne, alt_grid, color="mediumseagreen", lw=2.0,
                    label="ISR truth")
    else:
        ax_edp.text(0.5, 0.5, "no co-located ISR profile in window",
                    transform=ax_edp.transAxes, ha="center", va="center",
                    fontsize=8, color="dimgray")
    ax_edp.plot(prior_ne_tecmax, alt_grid, color="royalblue", lw=1.8, ls="--",
                label="Prior @ TEC-max pt")
    ax_edp.plot(post_ne_tecmax,  alt_grid, color="firebrick", lw=1.8, ls="-.",
                label="Posterior @ TEC-max pt")
    ax_edp.set_xscale("log")
    ax_edp.set_xlabel("Electron Density (m⁻³)", fontsize=9)
    ax_edp.set_ylabel("Alt (km)", fontsize=8.5)
    ax_edp.set_ylim(alt_grid.min(), alt_grid.max())
    ax_edp.set_title("EDP @ TEC-max tangent point", fontsize=9, loc="left")
    ax_edp.tick_params(labelsize=7.5)
    ax_edp.legend(fontsize=7.5, loc="best", framealpha=0.85)
    ax_edp.grid(True, alpha=0.3, ls=":")

    safe_key   = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    safe_label = (label or "occ").replace("/", "_").replace(" ", "_")
    plot_path = os.path.join(save_dir, f"occdiag_{safe_key}_{safe_label}_{prn}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print_saved(f"  Occultation diagnostic plot saved → {plot_path}")
    return plot_path


def plot_kf_enkf_comparison(
    res_kf:       dict,
    res_enkf:     dict,
    isr_profiles: list[dict],
    alt_grid:     np.ndarray,
    group_key:    str,
    save_dir:     str,
    n_tec_shown:  int = 3,
) -> str:
    """
    3×2 direct comparison figure.

    ┌──────────────────────┬──────────────────────┐
    │  [0,0] Prior TEC     │  [0,1] Posterior TEC  │
    │  measured, KF, EKF   │  measured, KF, EKF    │
    ├──────────────────────┼──────────────────────┤
    │  [1,0] Prior EDP     │  [1,1] Posterior EDP  │
    │  KF, EKF, ISR        │  KF, EKF, ISR         │
    ├──────────────────────┼──────────────────────┤
    │  [2,0] KF Globe      │  [2,1] EKF Globe      │
    │  ΔNe + raypaths      │  ΔNe + raypaths       │
    └──────────────────────┴──────────────────────┘

    TEC panels show only `n_tec_shown` occultations (default 3).

    Parameters
    ----------
    res_kf, res_enkf  : result dicts from process_group / _run_parametric_enkf.
    isr_profiles      : ISR sweeps (may be empty list).
    alt_grid          : shared altitude grid (km).
    group_key         : used for figure title and filename.
    save_dir          : output directory.
    n_tec_shown       : number of occultations to show in TEC panels.

    Returns
    -------
    str : path to the saved PNG.
    """
    from demo_verification import millstone_vertex_idx, ISR_LAT, ISR_LON_W

    os.makedirs(save_dir, exist_ok=True)

    verts_geo = res_kf["eds_occ"].geolocation
    n_geo     = verts_geo.shape[0]
    n_alt     = len(alt_grid)
    idx_mh    = millstone_vertex_idx(verts_geo)

    # ── Extract EDP at MH vertex ──────────────────────────────────────────────
    kf_prior_mh  = np.asarray(res_kf["prior_edp_3d"]).reshape(n_alt, n_geo)[:, idx_mh]
    _kf_jnt      = res_kf.get("joint_post_edp_3d")
    kf_post_mh   = np.asarray(
        _kf_jnt if _kf_jnt is not None else res_kf["post_edp_3d"]
    ).reshape(n_alt, n_geo)[:, idx_mh]

    enkf_prior_mh = np.asarray(res_enkf["prior_edp_3d"]).reshape(n_alt, n_geo)[:, idx_mh]
    enkf_post_mh  = np.asarray(res_enkf["post_edp_3d"]).reshape(n_alt, n_geo)[:, idx_mh]

    # ── TEC slices — limit to n_tec_shown occultations ────────────────────────
    kf_slices   = res_kf["tec_slices"]
    enkf_slices = res_enkf["tec_slices"]
    n_occ       = len(kf_slices)
    shown_idx   = list(range(min(n_tec_shown, n_occ)))

    sat_ids  = res_kf.get("sat_ids", [])
    cmap_occ = mpl.colormaps.get_cmap("tab10")
    occ_cols = [cmap_occ(i % 10) for i in range(n_occ)]

    # ── Globe projections ─────────────────────────────────────────────────────
    lats_c = res_kf["lats"]
    lons_c = res_kf["lons"]
    clon   = float(np.nanmean(lons_c)) if lons_c else 0.0
    clat   = float(np.nanmean(lats_c)) if lats_c else 0.0
    proj_kf   = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)
    proj_enkf = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 21))
    fig.suptitle(
        f"KF vs. Parametric EKF Comparison\n{group_key}",
        fontsize=13, y=0.99,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.35, hspace=0.45,
                           height_ratios=[1, 1, 1.4])

    ax_tec_pr = fig.add_subplot(gs[0, 0])
    ax_tec_po = fig.add_subplot(gs[0, 1], sharey=ax_tec_pr, sharex=ax_tec_pr)
    ax_edp_pr = fig.add_subplot(gs[1, 0])
    ax_edp_po = fig.add_subplot(gs[1, 1], sharey=ax_edp_pr, sharex=ax_edp_pr)
    ax_globe_kf   = fig.add_subplot(gs[2, 0], projection=proj_kf)
    ax_globe_enkf = fig.add_subplot(gs[2, 1], projection=proj_enkf)

    # ── [0,0] Prior TEC (n_tec_shown occultations) ────────────────────────────
    ax = ax_tec_pr
    for i in shown_idx:
        sl_kf = kf_slices[i]
        sl_en = enkf_slices[i]
        col   = occ_cols[i]
        prn   = sat_ids[i][1] if i < len(sat_ids) else f"Occ {i+1}"
        ax.plot(sl_kf["measured"],  sl_kf["tangent_km"],
                color=col, lw=2.0, label=prn)
        ax.plot(sl_kf["prior_tec"], sl_kf["tangent_km"],
                color=col, lw=1.4, ls="--", alpha=0.8)
        ax.plot(sl_en["prior_tec"], sl_en["tangent_km"],
                color=col, lw=1.4, ls=":",  alpha=0.8)

    _style_legend = [
        Line2D([0], [0], color="gray", lw=2.0,          label="Measured TEC"),
        Line2D([0], [0], color="gray", lw=1.4, ls="--", label="KF prior"),
        Line2D([0], [0], color="gray", lw=1.4, ls=":",  label="EKF prior"),
    ] + [Line2D([0], [0], color=occ_cols[i], lw=2.0,
                label=sat_ids[i][1] if i < len(sat_ids) else f"Occ {i+1}")
         for i in shown_idx]
    ax.legend(handles=_style_legend, fontsize=8, loc="upper right", framealpha=0.85)
    ax.set_xlabel("TEC (TECU)", fontsize=10)
    ax.set_ylabel("Tangent Altitude (km)", fontsize=10)
    ax.set_title(
        f"Prior TEC — KF RMSE {res_kf['prior_tec_rmse']:.2f} TECU"
        f" | EKF {res_enkf['prior_tec_rmse']:.2f} TECU",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, ls=":")

    # ── [0,1] Posterior TEC ───────────────────────────────────────────────────
    ax = ax_tec_po
    for i in shown_idx:
        sl_kf = kf_slices[i]
        sl_en = enkf_slices[i]
        col   = occ_cols[i]
        ax.plot(sl_kf["measured"],  sl_kf["tangent_km"], color=col, lw=2.0)
        ax.plot(sl_kf["post_tec"],  sl_kf["tangent_km"], color=col, lw=1.4, ls="--", alpha=0.8)
        ax.plot(sl_en["post_tec"],  sl_en["tangent_km"], color=col, lw=1.4, ls=":",  alpha=0.9)

    _style_legend_po = [
        Line2D([0], [0], color="gray", lw=2.0,          label="Measured TEC"),
        Line2D([0], [0], color="gray", lw=1.4, ls="--", label="KF posterior"),
        Line2D([0], [0], color="gray", lw=1.4, ls=":",  label="EKF posterior"),
    ]
    ax.legend(handles=_style_legend_po, fontsize=8, loc="upper right", framealpha=0.85)
    ax.set_xlabel("TEC (TECU)", fontsize=10)
    ax.set_title(
        f"Posterior TEC — KF RMSE {res_kf['post_tec_rmse']:.2f} TECU"
        f" | EKF {res_enkf['post_tec_rmse']:.2f} TECU",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, ls=":")

    # ── [1,0] Prior EDP at MH ─────────────────────────────────────────────────
    ISR_COLOR = "mediumseagreen"
    ax = ax_edp_pr
    # log-x, capped at 1e13; ax_edp_po shares this xaxis (sharex=ax_edp_pr).
    ax.set_xscale("log")
    ax.set_xlim(1e9, 1e13)

    for prof in isr_profiles:
        ax.plot(prof["ne"], prof["alt_km"],
                color=ISR_COLOR, lw=1.0, alpha=0.7,
                label="ISR truth" if prof is isr_profiles[0] else "_")

    ax.plot(kf_prior_mh,   alt_grid, color="royalblue",  lw=2.2, ls="--",
            label="KF prior (IRI)")
    ax.plot(enkf_prior_mh, alt_grid, color="darkorange",  lw=2.2, ls="-.",
            label="EKF prior (parametric)")

    for (nm, hm), col, mrk in [
        (extract_robust_f2_peak(kf_prior_mh,   alt_grid), "royalblue",  "D"),
        (extract_robust_f2_peak(enkf_prior_mh, alt_grid), "darkorange", "s"),
    ]:
        if not np.isnan(nm):
            ax.plot(nm, hm, marker=mrk, ms=9, color=col, mec="black", zorder=7)

    prior_state_ekf = res_enkf.get("prior_mean_state")
    if prior_state_ekf is not None:
        from demo_compare_kf_enkf import _draw_param_boxes
        _draw_param_boxes(
            ax, [("EKF prior", "darkorange", np.asarray(prior_state_ekf)[:, idx_mh])],
            loc="lower right",
        )

    ax.set_xlabel("Electron Density (m⁻³)", fontsize=10)
    ax.set_ylabel("Altitude (km)", fontsize=10)
    ax.set_title("Prior EDP at Millstone Hill vertex", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.3, ls=":")
    ax.set_ylim(bottom=0)

    # ── [1,1] Posterior EDP at MH ─────────────────────────────────────────────
    ax = ax_edp_po

    for prof in isr_profiles:
        ax.plot(prof["ne"], prof["alt_km"],
                color=ISR_COLOR, lw=1.0, alpha=0.7,
                label="ISR truth" if prof is isr_profiles[0] else "_")

    ax.plot(kf_post_mh,   alt_grid, color="royalblue",  lw=2.5, ls="--",
            label="KF posterior")
    ax.plot(enkf_post_mh, alt_grid, color="darkorange",  lw=2.5, ls="-.",
            label="EKF posterior")

    for (nm, hm), col, mrk in [
        (extract_robust_f2_peak(kf_post_mh,   alt_grid), "royalblue",  "D"),
        (extract_robust_f2_peak(enkf_post_mh, alt_grid), "darkorange", "s"),
    ]:
        if not np.isnan(nm):
            ax.plot(nm, hm, marker=mrk, ms=9, color=col, mec="black", zorder=7)

    if isr_profiles:
        isr_nm_mean = np.nanmean([p["nm_f2"] for p in isr_profiles])
        isr_hm_mean = np.nanmean([p["hm_f2"] for p in isr_profiles])
        nm_kf,  hm_kf  = extract_robust_f2_peak(kf_post_mh,   alt_grid)
        nm_en,  hm_en  = extract_robust_f2_peak(enkf_post_mh, alt_grid)
        bias_nm_kf  = nm_kf  - isr_nm_mean if not np.isnan(nm_kf)  else np.nan
        bias_nm_en  = nm_en  - isr_nm_mean if not np.isnan(nm_en)  else np.nan
        bias_hm_kf  = hm_kf  - isr_hm_mean if not np.isnan(hm_kf)  else np.nan
        bias_hm_en  = hm_en  - isr_hm_mean if not np.isnan(hm_en)  else np.nan
        bias_text = (
            f"NmF2 bias — KF: {bias_nm_kf:.2e}  EKF: {bias_nm_en:.2e} m⁻³\n"
            f"hmF2 bias — KF: {bias_hm_kf:.1f}  EKF: {bias_hm_en:.1f} km"
        )
        ax.text(0.03, 0.03, bias_text, transform=ax.transAxes,
                fontsize=7.5, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))

    post_state_ekf = res_enkf.get("posterior_mean_state", res_enkf.get("post_mean_state"))
    if post_state_ekf is not None:
        from demo_compare_kf_enkf import _draw_param_boxes
        _draw_param_boxes(
            ax, [("EKF posterior", "darkorange", np.asarray(post_state_ekf)[:, idx_mh])],
            loc="lower right",
        )

    ax.set_xlabel("Electron Density (m⁻³)", fontsize=10)
    ax.set_title("Posterior EDP at Millstone Hill vertex", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.3, ls=":")

    # ── [2,0] KF Globe ────────────────────────────────────────────────────────
    isr_site_arg = (ISR_LON_W, ISR_LAT) if isr_profiles else None
    _render_globe_ax(ax_globe_kf,   fig, res_kf,   alt_grid, "KF",
                     isr_site=isr_site_arg, n_occ_shown=n_tec_shown,
                     shown_indices=shown_idx, occ_colours_override=occ_cols)

    # ── [2,1] EKF Globe ───────────────────────────────────────────────────────
    _render_globe_ax(ax_globe_enkf, fig, res_enkf, alt_grid, "EKF",
                     isr_site=isr_site_arg, n_occ_shown=n_tec_shown,
                     shown_indices=shown_idx, occ_colours_override=occ_cols)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    safe_key  = group_key.replace("/", "_").replace(" ", "_").replace(":", "")
    plot_path = os.path.join(save_dir, f"compare_{safe_key}_3x2.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print_saved(f"  Comparison 3×2 plot saved → {plot_path}")

    # ── Separate covariance plots for KF and EKF ──────────────────────────────
    if "prior_P" in res_kf and "post_P" in res_kf:
        _plot_covariance_panels_labeled(res_kf,   save_dir, group_key, "KF")
    if "prior_P" in res_enkf and "post_P" in res_enkf:
        _plot_ekf_param_covariance_panels(res_enkf, save_dir, group_key, "EKF")

    return plot_path
