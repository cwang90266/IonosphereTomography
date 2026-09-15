#!/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
# -*- coding: utf-8 -*-
"""
diagnose_forward_tec.py
=======================
Standalone diagnostic for understanding why the forward-modeled (prior) TEC
returns near-zero values for the top portion of some occultations.

No assimilation is performed — this script only builds the IRI background,
constructs the observation operator H, and decomposes the forward-model TEC
into its in-grid and topside contributions for each ray.

Usage
-----
    python diagnose_forward_tec.py

Adjust the FILES list and ALT_GRID near the top to match your setup.
"""

import sys
import os
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyproj
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for SSH/headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from joblib import Parallel, delayed

# ── Locate the project root and wire up imports ─────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EDPSamples" / "Locate in mesh" / "outputs"))

from TEC_model.podTc_file_processing import parse_podTc2_nc_file, rayTangent
from EDPSamples.edp_samples import EDPSamples, interp_heights, find_containing_triangles
from Ionosphere_Tomography_Inverter.Ionophy_Tomography_Inverter import (
    Ionosphere_Tomography_Inverter,
    _process_single_ray,
)
from demo import build_daily_global_edps, extract_robust_f2_peak

# ── Configuration ────────────────────────────────────────────────────────────

FILES = [
    (
        "/home/pin/Desktop/tomography_project/piq_data/podTc2/"
        "2025.153/podTc2_GN05.2025.153.01.31.0025.C35.00_0000.0001_nc",
        "C35 — high-alt only, proper TEC",
    ),
    (
        "/home/pin/Desktop/tomography_project/piq_data/podTc2/"
        "2025.153/podTc2_GN05.2025.153.01.07.0027.E12.01_0000.0001_nc",
        "E12 — full occ, bad topside TEC",
    ),
    (
        "/home/pin/Desktop/tomography_project/piq_data/podTc2/2025.153/podTc2_GN04.2025.153.00.42.0025.G18.01_0000.0001_nc",
        "G18 — full occ, bad topside TEC",
    ),
    (
         "/home/pin/Desktop/tomography_project/piq_data/podTc2/2025.153/podTc2_GN04.2025.153.18.48.0025.R03.00_0000.0001_nc",
         "R03 — full occ, good topside TEC",
    ),
    # (
    #     "/home/austinhunter/Downloads/PlanetiQ_Code/BC_Processing/podTc2/"
    #     "2025.153/podTc2_GN05.2025.153.06.15.0034.R18.00_0000.0001_nc",
    #     "R18 — full occ, good topside TEC",
    # ),
]

# Altitude grid — must match the grid used in demo_group.py
ALT_GRID = np.logspace(np.log10(60.0), np.log10(800.0), num=55, dtype=float)

# Topside model parameters — must match Ionosphere_Tomography_Inverter defaults
TOPSIDE_SCALE_HEIGHT_M   = 150_000.0   # metres   (H_O+)
TOPSIDE_H_H_M            = 1_000_000.0  # metres   (H_H+/He+)
TOPSIDE_ALPHA            = 0.05         # mixing fraction for hydrogen/helium ion layer
# Plasmasphere prior floor.  IRI-2020 gives floor-level density (1e8 m^-3) above
# ~450 km at low latitudes / night-side, collapsing x_top_prior to ~0.002 TECU and
# making the forward prior TEC essentially zero for high-tangent-altitude rays.
# Set to 0.0 to replicate the old (broken) behavior, or 1.0 for the corrected floor.
TOPSIDE_PRIOR_FLOOR_TECU = 1.0         # TECU  (set 0.0 to disable)

# Directory for cached global EDP NetCDFs (re-used if they already exist)
GLOBAL_EDP_DIR = "./Data/Global_EDPS_153_log/"

# Number of workers for building H (parallel joblib) and global EDP (mp.Pool)
NUM_WORKERS = 8
NUM_SEGMENTS = 1000     # ray midpoints (1000 → 999 segments)

# Margin added around the occultation footprint when subsetting the EDP mesh
BBOX_MARGIN = 10.0        # degrees

# Save diagnostic figure to this path
OUTPUT_FIG = "./Figures/diagnose_forward_tec_153.png"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_bbox(parsed: dict, alt_grid: np.ndarray,
              margin: float = BBOX_MARGIN) -> tuple[float, float, float, float]:
    """Bounding box (lat_min, lat_max, lon_min, lon_max) for one occultation."""
    try:
        pt1, pt2, pt3 = EDPSamples.get_occultation_extrema(
            parsed["LEO"], parsed["GNSS"], alt_limit=float(alt_grid[-1]) + 50.0
        )
        lats = [pt1[0], pt2[0], pt3[0]]
        lons = [pt1[1], pt2[1], pt3[1]]
    except Exception:
        # Fallback: use tangent-point range
        _, _, tang_m = rayTangent(parsed["LEO"], parsed["GNSS"])
        tang_km = tang_m * 1e-3
        valid = tang_km > 0
        xfm = pyproj.Transformer.from_crs(
            pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
            pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
            always_xy=True,
        )
        lons_raw, lats_raw, _ = xfm.transform(
            parsed["LEO"][0, valid] * 1e3,
            parsed["LEO"][1, valid] * 1e3,
            parsed["LEO"][2, valid] * 1e3,
        )
        lats, lons = list(lats_raw), list(lons_raw)

    return (
        float(min(lats)) - margin,
        float(max(lats)) + margin,
        float(min(lons)) - margin,
        float(max(lons)) + margin,
    )


def _count_midpoints_per_ray(
    parsed: dict,
    alt_grid: np.ndarray,
    num_segments: int = NUM_SEGMENTS,
) -> dict:
    """
    For every ray in *parsed*, trace midpoints and count how many fall:
      - below the grid bottom  (below alt_grid[0])
      - inside the grid        (alt_grid[0] … alt_grid[-1])
      - above the grid top     (topside, > alt_grid[-1])

    Returns a dict with arrays of shape (n_rays,):
        n_below, n_ingrid, n_topside, tang_km, leo_alt_km
    """
    LEO  = parsed["LEO"]
    GNSS = parsed["GNSS"]
    n_rays = LEO.shape[1]
    t      = np.linspace(0, 1, num_segments)

    xfm = pyproj.Transformer.from_crs(
        pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
        pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
        always_xy=True,
    )

    n_below   = np.zeros(n_rays, dtype=int)
    n_ingrid  = np.zeros(n_rays, dtype=int)
    n_topside = np.zeros(n_rays, dtype=int)

    for i in range(n_rays):
        ray = GNSS[:, i:i+1] + (LEO[:, i:i+1] - GNSS[:, i:i+1]) * t
        mid = (ray[:, :-1] + ray[:, 1:]) / 2.0
        _, _, alts_m = xfm.transform(
            mid[0, :] * 1e3, mid[1, :] * 1e3, mid[2, :] * 1e3
        )
        alts_km = alts_m / 1e3
        n_below[i]   = int(np.sum(alts_km < alt_grid[0]))
        n_ingrid[i]  = int(np.sum((alts_km >= alt_grid[0]) & (alts_km <= alt_grid[-1])))
        n_topside[i] = int(np.sum(alts_km > alt_grid[-1]))

    # Tangent altitudes and LEO altitudes
    _, _, tang_m_arr = rayTangent(LEO, GNSS)
    tang_km = tang_m_arr * 1e-3

    _, _, leo_m = xfm.transform(
        LEO[0, :] * 1e3, LEO[1, :] * 1e3, LEO[2, :] * 1e3
    )
    leo_alt_km = leo_m / 1e3

    return dict(
        n_below=n_below,
        n_ingrid=n_ingrid,
        n_topside=n_topside,
        tang_km=tang_km,
        leo_alt_km=leo_alt_km,
    )


def _decompose_forward_tec(
    H: np.ndarray,
    n_sv: int,
    prior_state_flat: np.ndarray,
    x_top_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the prior forward TEC into in-grid and topside contributions.

    Returns
    -------
    tec_ingrid   : (n_rays,) — TEC from the in-grid state
    tec_topside  : (n_rays,) — TEC from the topside state
    tec_total    : (n_rays,) — total prior forward TEC
    """
    tec_ingrid  = (H[:, :n_sv] @ prior_state_flat).flatten()
    tec_topside = (H[:, n_sv:] @ x_top_prior[:, None]).flatten()
    return tec_ingrid, tec_topside, tec_ingrid + tec_topside


# ── Per-file processing ──────────────────────────────────────────────────────

def process_one_file(
    filepath: str,
    label: str,
    global_edp_cache: dict,
    alt_grid: np.ndarray,
) -> Optional[dict]:
    """
    Parse one podTc2 file, build the forward model, and return a results dict.
    """
    print(f"\n{'='*65}")
    print(f"  Processing: {label}")
    print(f"  File: {os.path.basename(filepath)}")
    print(f"{'='*65}")

    # 1. Parse file
    parsed = parse_podTc2_nc_file(filepath)
    if parsed is None:
        print("  !! parse_podTc2_nc_file returned None — skipping.")
        return None

    n_rays = parsed["LEO"].shape[1]
    tang   = parsed["tangent_alt_km"]
    print(f"  Rays (after QC): {n_rays}")
    print(f"  Tangent alt range: {tang.min():.1f} – {tang.max():.1f} km")

    # LEO altitude
    xfm = pyproj.Transformer.from_crs(
        pyproj.CRS.from_proj4("+proj=geocent +ellps=WGS84 +datum=WGS84"),
        pyproj.CRS.from_proj4("+proj=longlat +ellps=WGS84 +datum=WGS84"),
        always_xy=True,
    )
    _, _, leo_m = xfm.transform(
        parsed["LEO"][0, :] * 1e3,
        parsed["LEO"][1, :] * 1e3,
        parsed["LEO"][2, :] * 1e3,
    )
    leo_alt_km = leo_m / 1e3
    print(f"  LEO alt range:    {leo_alt_km.min():.1f} – {leo_alt_km.max():.1f} km")
    print(f"  alt_grid top:     {alt_grid[-1]:.1f} km")

    # 2. Determine hour for EDP cache
    profile_hour = int(parsed["date"].hour)
    print(f"  Profile hour:     {profile_hour:02d}:xx UTC")

    # 3. Subset EDP mesh to occultation footprint
    lat_min, lat_max, lon_min, lon_max = _get_bbox(parsed, alt_grid)
    print(f"  BBox: lat [{lat_min:.1f}, {lat_max:.1f}]  "
          f"lon [{lon_min:.1f}, {lon_max:.1f}]")

    eds_occ = global_edp_cache[profile_hour].subset_region(
        lat_min, lat_max, lon_min, lon_max
    )
    n_geo    = eds_occ.geolocation.shape[0]
    n_height = len(alt_grid)
    n_sv     = n_height * n_geo
    print(f"  EDP mesh vertices: {n_geo}")

    # 4. Build inverter (no assimilation — just for H and prior access)
    inverter = Ionosphere_Tomography_Inverter(
        EDPSam=eds_occ,
        meanscale=1,
        topside_scale_height_m=TOPSIDE_SCALE_HEIGHT_M,
        topside_H_H_m=TOPSIDE_H_H_M,
        topside_alpha=TOPSIDE_ALPHA,
        topside_prior_floor_tecu=TOPSIDE_PRIOR_FLOOR_TECU,
    )
    prior_state_flat = inverter.attrs["initial_edps_mean"]   # (n_sv, 1)  m^-3
    x_top_prior      = inverter.attrs["x_top_prior"]         # (n_geo,)   TECU
    H_eff_m          = inverter.attrs["topside_H_eff_m"]

    print(f"\n  Topside model: H_eff = {H_eff_m/1e3:.0f} km  "
          f"| floor = {TOPSIDE_PRIOR_FLOOR_TECU:.2f} TECU")
    print(f"  x_top_prior (geo-mean): {x_top_prior.mean():.4f} TECU  "
          f"(min {x_top_prior.min():.4f}, max {x_top_prior.max():.4f})")

    # ne at the grid top from the IRI background
    ne_top_arr = prior_state_flat.reshape(n_height, n_geo)[-1, :]
    print(f"  IRI ne @ {alt_grid[-1]:.0f} km (geo-mean): "
          f"{ne_top_arr.mean():.3e} m^-3  "
          f"(min {ne_top_arr.min():.3e}, max {ne_top_arr.max():.3e})")

    # 5. Build H matrix
    print(f"\n  Building H matrix …")
    H = inverter.get_observation_operator(
        {"LEO": parsed["LEO"], "GNSS": parsed["GNSS"]},
        num_segments=NUM_SEGMENTS,
    )

    # 6. Decompose forward TEC
    tec_ingrid, tec_topside, tec_prior = _decompose_forward_tec(
        H, n_sv, prior_state_flat, x_top_prior
    )
    tec_obs  = parsed["TEC_podTc2"]

    # 7. Count midpoints per ray
    print("  Counting in-grid / topside midpoints per ray …")
    mp_counts = _count_midpoints_per_ray(parsed, alt_grid)

    # 8. Diagnostic table for rays in the top portion (> 300 km tangent alt)
    print(f"\n  {'tang_km':>9} {'n_below':>8} {'n_ingrid':>9} "
          f"{'n_top':>8} {'TEC_ingrid':>11} {'TEC_topside':>12} "
          f"{'TEC_prior':>10} {'TEC_obs':>10}")
    print("  " + "-" * 88)
    mask_top = tang > 300.0
    idx_top  = np.where(mask_top)[0]
    # Show every ~10th ray in the top section to keep output manageable
    step = max(1, len(idx_top) // 20)
    for ii in idx_top[::step]:
        print(f"  {tang[ii]:9.1f} {mp_counts['n_below'][ii]:8d} "
              f"{mp_counts['n_ingrid'][ii]:9d} {mp_counts['n_topside'][ii]:8d}  "
              f"{tec_ingrid[ii]:11.4f} {tec_topside[ii]:12.4f}  "
              f"{tec_prior[ii]:10.4f} {tec_obs[ii]:10.4f}")

    # 9. Altitude sensitivity profile of H (column sums by altitude bin)
    H_grid_2d = H[:, :n_sv].reshape(n_rays, n_height, n_geo)
    alt_sensitivity = np.abs(H_grid_2d).sum(axis=(0, 2))   # (n_height,)

    # 10. Background EDP at the occultation centre (geo-mean across vertices)
    prior_edp_centre = prior_state_flat.reshape(n_height, n_geo).mean(axis=1)  # (n_height,)

    return dict(
        label         = label,
        filepath      = filepath,
        parsed        = parsed,
        tang          = tang,
        leo_alt_km    = leo_alt_km,
        tec_obs       = tec_obs,
        tec_prior     = tec_prior,
        tec_ingrid    = tec_ingrid,
        tec_topside   = tec_topside,
        mp_counts     = mp_counts,
        alt_sensitivity = alt_sensitivity,
        prior_edp_centre = prior_edp_centre,
        x_top_prior   = x_top_prior,
        H_eff_m       = H_eff_m,
        alt_grid      = alt_grid,
        n_geo         = n_geo,
        n_sv          = n_sv,
    )


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_diagnostics(results: list[dict], output_path: str):
    """
    Multi-panel figure:  one column per occultation file.

    Row 1 — TEC vs tangent altitude: observed (blue) vs prior total (red)
             with in-grid (green) and topside (orange) sub-contributions.
    Row 2 — Midpoint distribution: fraction of 999 midpoints that are
             in-grid (green) vs topside (orange) per ray.
    Row 3 — IRI background EDP profile (ne vs altitude) at the geo-mean vertex.
    Row 4 — Altitude sensitivity of H (how much each altitude bin is sampled).
    """
    n_files = len(results)
    n_rows  = 4
    fig = plt.figure(figsize=(7 * n_files, 5 * n_rows))
    gs  = gridspec.GridSpec(
        n_rows, n_files,
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97, top=0.93, bottom=0.06,
    )

    fig.suptitle(
        "Forward-Model TEC Diagnostic  —  No Assimilation\n"
        f"alt_grid: {results[0]['alt_grid'][0]:.0f}–{results[0]['alt_grid'][-1]:.0f} km  "
        f"({len(results[0]['alt_grid'])} levels)",
        fontsize=13, fontweight="bold",
    )

    ROW_LABELS = [
        "Observed vs Prior TEC  [TECU]",
        "Midpoints per ray  (of 999)",
        "IRI background EDP  [m⁻³]",
        "H column sensitivity (Σ|H_grid|)",
    ]

    for col, res in enumerate(results):
        alt_grid  = res["alt_grid"]
        tang      = res["tang"]
        tec_obs   = res["tec_obs"]
        tec_prior = res["tec_prior"]
        tec_ig    = res["tec_ingrid"]
        tec_ts    = res["tec_topside"]
        mp        = res["mp_counts"]
        edp       = res["prior_edp_centre"]
        sens      = res["alt_sensitivity"]
        label     = res["label"]
        leo_alt   = res["leo_alt_km"]

        # ── Row 0: TEC comparison ─────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        ax0.plot(tec_obs,   tang, color="steelblue",  lw=1.5, label="Observed")
        ax0.plot(tec_prior, tang, color="firebrick",   lw=1.5, label="Prior total",
                 ls="--")
        ax0.plot(tec_ig,    tang, color="seagreen",   lw=1.0, label="In-grid part",
                 ls=":", alpha=0.85)
        ax0.plot(tec_ts,    tang, color="darkorange",  lw=1.0, label="Topside part",
                 ls=":", alpha=0.85)

        # Mark alt_grid top and mean LEO altitude
        ax0.axhline(alt_grid[-1], color="gray", ls="--", lw=0.8, alpha=0.7,
                    label=f"Grid top ({alt_grid[-1]:.0f} km)")
        ax0.axhline(leo_alt.mean(), color="mediumpurple", ls="-.", lw=0.8, alpha=0.7,
                    label=f"LEO mean ({leo_alt.mean():.0f} km)")

        ax0.set_xlabel("TEC (TECU)")
        ax0.set_ylabel("Tangent altitude (km)")
        ax0.set_ylim(0, min(tang.max() + 30, alt_grid[-1] + 50))
        ax0.legend(fontsize=6.5, loc="lower right")
        ax0.grid(True, alpha=0.3)
        ax0.set_title(label, fontsize=8, fontweight="bold")
        if col == 0:
            ax0.set_ylabel(ROW_LABELS[0])

        # ── Row 1: Midpoint distribution ──────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        n_tot = NUM_SEGMENTS - 1
        frac_ig = mp["n_ingrid"] / n_tot * 100.0
        frac_ts = mp["n_topside"] / n_tot * 100.0
        frac_bl = mp["n_below"] / n_tot * 100.0

        ax1.barh(tang, frac_ig, height=np.diff(tang, prepend=tang[0]-2),
                 color="seagreen", alpha=0.65, label="In-grid %")
        ax1.barh(tang, frac_ts, left=frac_ig,
                 height=np.diff(tang, prepend=tang[0]-2),
                 color="darkorange", alpha=0.65, label="Topside %")
        ax1.barh(tang, frac_bl, left=frac_ig + frac_ts,
                 height=np.diff(tang, prepend=tang[0]-2),
                 color="gray", alpha=0.4, label="Below grid %")

        ax1.axhline(alt_grid[-1], color="gray", ls="--", lw=0.8)
        ax1.axhline(leo_alt.mean(), color="mediumpurple", ls="-.", lw=0.8)
        ax1.set_xlabel("% of midpoints")
        ax1.set_ylim(0, min(tang.max() + 30, alt_grid[-1] + 50))
        ax1.set_xlim(0, 100)
        ax1.legend(fontsize=6.5, loc="lower right")
        ax1.grid(True, alpha=0.3)
        if col == 0:
            ax1.set_ylabel(ROW_LABELS[1])

        # Annotate: how many midpoints are in-grid for the highest-tangent ray
        idx_top = int(np.argmax(tang))
        ax1.annotate(
            f"Top ray: {mp['n_ingrid'][idx_top]} in-grid,\n"
            f"{mp['n_topside'][idx_top]} topside",
            xy=(50, tang[idx_top]),
            fontsize=7, color="black",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

        # ── Row 2: IRI EDP ────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[2, col])
        ax2.semilogx(np.maximum(edp, 1e6), alt_grid, color="navy", lw=1.5)
        ax2.axhline(alt_grid[-1], color="gray", ls="--", lw=0.8,
                    label=f"Grid top {alt_grid[-1]:.0f} km")
        ax2.axhline(leo_alt.mean(), color="mediumpurple", ls="-.", lw=0.8,
                    label=f"LEO {leo_alt.mean():.0f} km")

        # Mark ne at grid top
        ne_top_mean = edp[-1]
        ax2.plot(max(ne_top_mean, 1e6), alt_grid[-1], "ro", ms=6,
                 label=f"ne@top={ne_top_mean:.2e}")
        ax2.set_xlabel("ne (m⁻³)")
        ax2.set_ylim(alt_grid[0], alt_grid[-1] + 50)
        ax2.legend(fontsize=6.5)
        ax2.grid(True, alpha=0.3, which="both")
        if col == 0:
            ax2.set_ylabel(ROW_LABELS[2])

        # Annotate x_top_prior
        x_top_mean = float(res["x_top_prior"].mean())
        H_eff_km   = res["H_eff_m"] / 1e3
        ax2.text(
            0.97, 0.05,
            f"x_top_prior\n(mean) = {x_top_mean:.4f} TECU\nH_eff = {H_eff_km:.0f} km",
            transform=ax2.transAxes, fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
        )

        # ── Row 3: H altitude sensitivity ────────────────────────────────────
        ax3 = fig.add_subplot(gs[3, col])
        ax3.plot(sens, alt_grid, color="darkorange", lw=1.5)
        ax3.axhline(alt_grid[-1], color="gray", ls="--", lw=0.8)
        ax3.axhline(leo_alt.mean(), color="mediumpurple", ls="-.", lw=0.8)
        ax3.set_xlabel("Σ|H_grid| across rays")
        ax3.set_ylim(alt_grid[0], alt_grid[-1] + 50)
        ax3.grid(True, alpha=0.3)
        if col == 0:
            ax3.set_ylabel(ROW_LABELS[3])

        # If the top of the grid has almost zero sensitivity, annotate it
        top_sens_frac = sens[-5:].sum() / (sens.sum() + 1e-30)
        ax3.text(
            0.97, 0.95,
            f"Top-5 levels:\n{top_sens_frac*100:.1f}% of total",
            transform=ax3.transAxes, fontsize=7, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Diagnostic figure saved → {output_path}")
    plt.close(fig)


# ── Second figure: direct head-to-head TEC comparison ───────────────────────

def plot_tec_comparison(results: list[dict], output_path: str):
    """
    Single large side-by-side plot showing observed vs prior TEC for all files,
    and the topside-fraction per ray.  Good for a quick at-a-glance comparison.
    """
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 10), squeeze=False)

    for col, res in enumerate(results):
        tang  = res["tang"]
        tec_o = res["tec_obs"]
        tec_p = res["tec_prior"]
        tec_ts = res["tec_topside"]
        mp    = res["mp_counts"]
        label = res["label"]
        ag    = res["alt_grid"]
        leo   = res["leo_alt_km"]

        ax_tec = axes[0, col]
        ax_tec.plot(tec_o, tang, "o-", ms=2, lw=1.2, color="steelblue",
                    label="Observed TEC")
        ax_tec.plot(tec_p, tang, "^--", ms=2, lw=1.2, color="firebrick",
                    label="Prior (forward) TEC")
        ax_tec.axhline(ag[-1], ls="--", color="gray", lw=0.9,
                       label=f"Grid top {ag[-1]:.0f} km")
        ax_tec.axhline(leo.mean(), ls="-.", color="mediumpurple", lw=0.9,
                       label=f"LEO {leo.mean():.0f} km")
        ax_tec.set_xlabel("TEC (TECU)", fontsize=9)
        ax_tec.set_ylabel("Tangent altitude (km)", fontsize=9)
        ax_tec.set_ylim(0, min(tang.max() + 30, ag[-1] + 60))
        ax_tec.legend(fontsize=7)
        ax_tec.grid(True, alpha=0.3)
        ax_tec.set_title(label, fontsize=9, fontweight="bold")

        ax_frac = axes[1, col]
        frac_ts = mp["n_topside"] / (NUM_SEGMENTS - 1) * 100.0
        ax_frac.plot(frac_ts, tang, "-", lw=1.2, color="darkorange",
                     label="Topside midpoints (%)")
        ax_frac.axhline(ag[-1], ls="--", color="gray", lw=0.9)
        ax_frac.axhline(leo.mean(), ls="-.", color="mediumpurple", lw=0.9)
        ax_frac.set_xlabel("Topside midpoints (%)", fontsize=9)
        ax_frac.set_ylabel("Tangent altitude (km)", fontsize=9)
        ax_frac.set_ylim(0, min(tang.max() + 30, ag[-1] + 60))
        ax_frac.set_xlim(0, 105)
        ax_frac.grid(True, alpha=0.3)
        ax_frac.set_title(f"{label}\n(topside midpoint fraction)", fontsize=8)

        # Annotate residual
        common = np.minimum(len(tec_o), len(tec_p))
        rmse   = np.sqrt(np.mean((tec_o[:common] - tec_p[:common]) ** 2))
        ax_tec.text(
            0.02, 0.98,
            f"RMSE = {rmse:.3f} TECU",
            transform=ax_tec.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.75),
        )

    fig.suptitle(
        "TEC Comparison: Observed vs Forward-Model Prior\n"
        f"(alt_grid top = {results[0]['alt_grid'][-1]:.0f} km)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = output_path.replace(".png", "_comparison.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"  Comparison figure saved → {out2}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  diagnose_forward_tec.py")
    print("=" * 65)
    print(f"\nalt_grid: {ALT_GRID[0]:.1f} – {ALT_GRID[-1]:.1f} km  "
          f"({len(ALT_GRID)} levels)")
    print(f"Topside: H_eff = "
          f"{((1-TOPSIDE_ALPHA)*TOPSIDE_SCALE_HEIGHT_M + TOPSIDE_ALPHA*TOPSIDE_H_H_M)/1e3:.0f} km")

    # ── 1. Determine which dates we need global EDP caches for ───────────────
    # All three files are from 2025-153 per the paths; load that date's cache.
    # If files span multiple dates, extend this logic.
    year, doy = 2025, 153
    batch_date = pd.Timestamp(
        datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
    )
    print(f"\nLoading/building global EDP cache for {batch_date.date()} …")
    global_edp_cache = build_daily_global_edps(
        batch_date,
        ALT_GRID,
        dLat=5.0, dLon=5.0,
        data_dir=GLOBAL_EDP_DIR,
        num_workers=NUM_WORKERS,
    )
    print("Global EDP cache ready.\n")

    # ── 2. Process each file ─────────────────────────────────────────────────
    results = []
    for filepath, label in FILES:
        res = process_one_file(filepath, label, global_edp_cache, ALT_GRID)
        if res is not None:
            results.append(res)

    if not results:
        print("\nNo files successfully processed — exiting.")
        return

    # ── 3. Print summary table ───────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    hdr = f"{'Label':<40} {'n_rays':>7} {'tang_max':>10} {'leo_alt':>9} "
    hdr += f"{'RMSE_prior':>11} {'x_top_mean':>11}"
    print(hdr)
    print("-" * len(hdr))
    for res in results:
        common = min(len(res["tec_obs"]), len(res["tec_prior"]))
        rmse   = np.sqrt(np.mean((res["tec_obs"][:common]
                                  - res["tec_prior"][:common]) ** 2))
        print(
            f"{res['label']:<40} {len(res['tang']):>7} "
            f"{res['tang'].max():>10.1f} {res['leo_alt_km'].mean():>9.1f}  "
            f"{rmse:>11.4f} {res['x_top_prior'].mean():>11.6f}"
        )

    # ── 4. Produce figures ────────────────────────────────────────────────────
    print(f"\nGenerating diagnostic figures …")
    plot_diagnostics(results, OUTPUT_FIG)
    plot_tec_comparison(results, OUTPUT_FIG)

    print("\nDone.")


if __name__ == "__main__":
    main()
