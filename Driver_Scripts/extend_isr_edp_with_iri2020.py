#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extend Tromso ISR electron density profiles above their ~650 km ceiling
using an IRI2020 climatological library, fit to each profile's own shape.

Three steps (see the approved plan):
  1. Draw a wide-range, 2000-sample IRI2020 driving-parameter ensemble
     (IRI_Sample_Inputs.randomSamples) and evaluate it with the real
     IRI2020 driver (EDPSamples) at the ISR site, on the ISR altitude
     gates plus a 650-900 km / 10 km extension grid.
  2. For each ISR time sample, fit log10(measured Ne) as a linear
     combination of the leading principal components (EOFs) of the
     log10(IRI library), restricted to ISR altitudes >= FIT_ALT_MIN --
     not the full 2000-sample basis, and not the noisy/highly-variable
     E-region and bottomside (see the rationale on N_PCA_MODES/
     FIT_ALT_MIN below).
  3. Apply that same combination to the library's 650-900 km rows to
     get the extension, and splice it onto the as-measured profile.

Fit design (revised from the first version of this script after reviewing
its results): the raw 2000-sample library is almost entirely redundant --
its singular-value spectrum (see plot_iri_library_singular_values.png)
drops ~5 orders of magnitude within the first 5-8 modes, i.e. IRI2020's
actual profile-*shape* response to a wide driving-parameter sweep spans
only a handful of independent shapes, mostly scaling/shifting one common
curve rather than deforming it. Fitting against all 2000 raw samples (the
original approach) chases that noise floor, producing a numerically
unstable fit that reproduces the measured profile almost exactly in-sample
but extrapolates to wildly unphysical (and often non-monotonic) densities
above 650km. Fitting against only the leading N_PCA_MODES left singular
vectors (principal profile shapes) of the library, and restricting the fit
itself to ISR altitudes >= FIT_ALT_MIN (no need to chase the E-region/
bottomside, which isn't representative of topside behavior anyway), was
empirically scanned against the real dataset: N_PCA_MODES=3 with
MIN_VALID_FIT_POINTS=8 gives a 100% monotonically-decreasing extension
with no outliers, at a modest cost in in-sample fit accuracy (median log10
residual ~0.05, i.e. ~10% density error) -- see the scan in this script's
development history for the full table across N_PCA_MODES in {3,4,5,8,10,
15,20,30}.

Usage
-----
    source Driver_Scripts/init_iri2020_env.sh
    /opt/anaconda3/bin/python3 Driver_Scripts/extend_isr_edp_with_iri2020.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "IRI_Sample_Inputs"))
sys.path.insert(0, str(_REPO_ROOT / "EDPSamples"))

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from IRI_Sample_inputs import IRI_Sample_Inputs
from edp_samples import EDPSamples

STATION_LAT = 69.583
STATION_LON = 19.21
# Station-agnostic (ISR_Integration_Plan.md Section 2/4.1.0): main() reads
# the real station location from the input ISR file's own
# station_latitude/station_longitude attrs when present (every file this
# project's ISR preprocessing produces carries them) -- these two
# module-level constants are now only a fallback for a file that lacks
# those attrs, not the station this script is hardcoded to.

EXT_ALT_MIN = 650.0
EXT_ALT_MAX = 900.0
EXT_ALT_STEP = 10.0

# The PCA-fit curve generally doesn't pass exactly through the last measured
# point (it's a 3-mode approximation, not an interpolant), so splicing it in
# cold at EXT_ALT_MIN leaves a visible step at the join. Instead, the join
# altitude is pinned exactly to the last valid measurement (an additive
# log10 offset, see fit_and_extend), and that offset is smoothly tapered to
# zero by TRANSITION_END_KM -- above that, the extension is the untouched
# PCA fit, same as before. Below TRANSITION_END_KM, this *is* a slightly
# different curve than the raw fit (by construction, continuous with the
# data); it's still read off Uk_ext, so it stays a linear combination of
# the same library modes, just bias-corrected near the join.
TRANSITION_END_KM = 700.0

# Fit design, see the module docstring: below this altitude the ISR profile
# shape (E-region, bottomside) is noisy/highly variable and irrelevant to
# the topside extension, so it's excluded from the least-squares objective
# (though still kept as-is in the output -- only the *fit* ignores it).
FIT_ALT_MIN = 200.0

# Matches Parameterization.density_10ex's default floor (10**4.0 m^-3):
# IRI2020 can return sentinel/unreliable (near-zero or negative) densities
# below ~90-100 km, which would otherwise poison log10 with NaN/-inf and
# break the pseudo-inverse for every sample column that touches those rows.
DENSITY_FLOOR = 1.0e4

# dataviz categorical slots 1/2/3 (blue/orange/aqua) and chart chrome, see
# EDPSamples/../dataviz palette reference used throughout this script's plots.
COLOR_MEASURED = "#2a78d6"
COLOR_EXTENDED = "#eb6834"
COLOR_FIT = "#1baf7a"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"
# Sequential blue ramp (dataviz palette, step 150->650) for the n_modes
# sensitivity comparison -- an ordinal quantity (more modes), one series
# each, so a single-hue ramp rather than the categorical slots.
SEQUENTIAL_RAMP = ["#b7d3f6", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--isr-file",
        default="/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/TROISR2025_nonan.nc",
        help="Input ISR NetCDF file (any station -- station_latitude/station_longitude "
             "are read from the file's own attrs, see ISR_Integration_Plan.md Section 2).",
    )
    p.add_argument(
        "--out",
        default="/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Data_Assimilation_Cycle/ISR_PCA_Basis",
        help="Output directory for the extended NetCDF and diagnostic plots.",
    )
    p.add_argument("--nsample", type=int, default=2000, help="Number of IRI2020 driving-parameter samples.")
    p.add_argument("--hour-range", type=int, default=12, help="randomSamples hour_sample_range (covers the full 0-23h local-time cycle).")
    p.add_argument("--f107-range", type=int, default=10**6, help="randomSamples f107_sample_range; deliberately larger than apf107.dat's length so it saturates to the entire historical record.")
    p.add_argument("--ap-range", type=int, default=10**6, help="randomSamples ap_sample_range; saturates to the entire historical record.")
    p.add_argument("--ig-range", type=int, default=10**5, help="randomSamples ig_sample_range; saturates to the entire historical record.")
    p.add_argument("--rz-range", type=int, default=10**5, help="randomSamples rz_sample_range; saturates to the entire historical record.")
    p.add_argument("--n-modes", type=int, default=3, help="Number of leading principal components (left singular vectors) of the IRI2020 library used as the fit basis, instead of all nsample raw profiles. Empirically scanned against the real dataset (see module docstring): 3 gives a 100%% monotonically-decreasing, outlier-free extension; beyond ~5-8 modes the library is into its noise floor and the fit/extension destabilizes badly.")
    p.add_argument("--fit-alt-min", type=float, default=FIT_ALT_MIN, help="Only ISR altitude gates at or above this altitude (km) constrain the least-squares fit -- the E-region/bottomside shape doesn't need to be replicated to extend the topside.")
    p.add_argument("--min-valid", type=int, default=8, help="Minimum number of non-NaN ISR altitude points at or above --fit-alt-min required to fit/extend a profile.")
    p.add_argument("--transition-end", type=float, default=TRANSITION_END_KM, help="Altitude (km) by which the join correction (pinning the extension to the last measured point) has fully decayed to zero, leaving the untouched PCA fit above it. Must be > EXT_ALT_MIN (650).")
    return p.parse_args()


def build_iri_library(center_time: pd.Timestamp, altitude_grid: np.ndarray, args,
                       station_lat: float = STATION_LAT, station_lon: float = STATION_LON):
    """Draw the sample ensemble and run the real IRI2020 driver once for
    the whole batch, at ``(station_lat, station_lon)`` -- station-agnostic
    (ISR_Integration_Plan.md Section 2/4.1.0): ``main()`` resolves these
    from the input ISR file's own attrs, not the module-level Tromso
    defaults, whenever the file provides them. Returns (log10(density)
    floored, shape (n_height, nsample); sampling_parameters)."""
    # data_dir=args.out keeps the fetched apf107.dat/ig_rz.dat cache files
    # under this script's own output directory, not the source tree.
    iri_in = IRI_Sample_Inputs(center_time.isoformat(), data_dir=args.out)
    sampling_parameters = iri_in.randomSamples(
        hour_sample_range=args.hour_range,
        f107_sample_range=args.f107_range,
        ap_sample_range=args.ap_range,
        ig_sample_range=args.ig_range,
        rz_sample_range=args.rz_range,
        nSample=args.nsample,
    )
    os.makedirs(args.out, exist_ok=True)
    sampling_parameters.to_pickle(f"{args.out}/isr_extension_iri_sample_inputs_{args.nsample}.pkl")
    sampling_parameters.to_csv(f"{args.out}/isr_extension_iri_sample_inputs_{args.nsample}.csv", index=False)

    print(f"Running IRI2020 for {args.nsample} samples x {len(altitude_grid)} altitudes at "
          f"({station_lat}, {station_lon})...")
    t0 = time.time()
    edp = EDPSamples(
        DateTime=center_time.isoformat(),
        geo_type="Point",
        altitude=altitude_grid,
        sampling_parameters=sampling_parameters,
        evaluate_iri=1,
        Lon=station_lon, Lat=station_lat,
    )
    print(f"IRI2020 batch run complete in {time.time() - t0:.1f} s.")

    # float64, not the bare float32 the Fortran driver returns: the library's
    # condition number runs to ~1e8-1e9 (2000 similarly-shaped IRI profiles),
    # and solving at that condition number in float32 (~1e-7 eps) is garbage
    # -- residuals order-1 even at the fit altitudes. float64 (~1e-16 eps)
    # keeps the pinv solve meaningful.
    density = np.squeeze(edp.edps, axis=1).astype(np.float64)  # (n_height, nsample)
    density = np.maximum(density, DENSITY_FLOOR)
    return np.log10(density), sampling_parameters


def compute_svd_basis(log_iri: np.ndarray) -> np.ndarray:
    """Left singular vectors of the (n_height, nsample) library, full rank --
    the ordered principal profile *shapes* spanning the full altitude grid
    (fit altitudes and extension altitudes together), ranked by how much of
    the library's sample-to-sample variance they explain. A k-mode fit
    basis is just the leading k columns, U[:, :k]; fitting against those
    instead of the raw nsample columns is what keeps the extension
    well-posed (see module docstring)."""
    U, _, _ = np.linalg.svd(log_iri, full_matrices=False)
    return U


def pca_fit_single(col: np.ndarray, Uk_isr: np.ndarray, fit_alt_mask: np.ndarray, min_valid: int):
    """Least-squares fit of one profile's log10(Ne) against a k-mode PCA
    basis, restricted to altitudes where fit_alt_mask is True. Returns
    (c, nv, rms) with c=None if nv < min_valid."""
    usable = ~np.isnan(col) & fit_alt_mask
    nv = int(usable.sum())
    if nv < min_valid:
        return None, nv, np.nan
    y = np.log10(np.maximum(col[usable], DENSITY_FLOOR))
    A = Uk_isr[usable, :]
    c = np.linalg.pinv(A) @ y
    rms = float(np.sqrt(np.mean((A @ c - y) ** 2)))
    return c, nv, rms


def smooth_taper_weight(altitude: np.ndarray, alt_start: float, alt_end: float) -> np.ndarray:
    """1 at/below alt_start, 0 at/above alt_end, smoothstep (C1, zero slope
    at both ends) in between -- used to taper the join bias-correction from
    full strength at the last measured altitude down to zero by
    TRANSITION_END_KM."""
    if alt_end <= alt_start:
        return np.where(altitude <= alt_start, 1.0, 0.0)
    t = np.clip((altitude - alt_start) / (alt_end - alt_start), 0.0, 1.0)
    return 1.0 - (3.0 * t**2 - 2.0 * t**3)


def fit_and_extend(Ne_isr: np.ndarray, Uk_isr: np.ndarray, Uk_ext: np.ndarray,
                    fit_alt_mask: np.ndarray, min_valid: int,
                    altitude_isr: np.ndarray, altitude_ext: np.ndarray, transition_end_km: float):
    """Per ISR time column, least-squares fit of log10(Ne) against the
    leading-PCA-mode basis, restricted to altitudes where fit_alt_mask is
    True, applied to the extension altitudes -- with a join correction:
    the fit curve generally doesn't pass exactly through the last measured
    point, so the extension is pinned to it with an additive log10 offset,
    smoothly tapered to zero by transition_end_km (see TRANSITION_END_KM).
    Returns (Ne_ext, n_valid, rms_log10_residual, C, join_offset);
    n_valid counts only the fit-eligible altitudes; C (n_modes, n_time)
    holds the fitted mode coefficients, NaN columns where skipped -- kept
    so callers can reconstruct the (uncorrected) fit curve at any altitude
    for diagnostics; join_offset (n_time,) is the log10 correction applied
    at the join, NaN where skipped."""
    n_isr_alt, n_time = Ne_isr.shape
    n_ext_alt = Uk_ext.shape[0]
    n_modes = Uk_isr.shape[1]

    n_valid = np.zeros(n_time, dtype=np.int32)
    rms_resid = np.full(n_time, np.nan, dtype=np.float64)
    Ne_ext = np.full((n_ext_alt, n_time), np.nan, dtype=np.float32)
    C = np.full((n_modes, n_time), np.nan, dtype=np.float64)
    join_offset = np.full(n_time, np.nan, dtype=np.float64)

    for t in range(n_time):
        col = Ne_isr[:, t]
        c, nv, rms = pca_fit_single(col, Uk_isr, fit_alt_mask, min_valid)
        n_valid[t] = nv
        if c is None:
            continue
        rms_resid[t] = rms
        C[:, t] = c

        idx_last = np.flatnonzero(~np.isnan(col))[-1]
        offset = np.log10(max(col[idx_last], DENSITY_FLOOR)) - float(Uk_isr[idx_last, :] @ c)
        weight = smooth_taper_weight(altitude_ext, altitude_isr[idx_last], transition_end_km)
        join_offset[t] = offset

        Ne_ext[:, t] = 10.0 ** (Uk_ext @ c + offset * weight)

    return Ne_ext, n_valid, rms_resid, C, join_offset


def save_output(ds: xr.Dataset, altitude_grid: np.ndarray, Ne_full: np.ndarray,
                 n_valid_fit: np.ndarray, rms_resid: np.ndarray, join_offset: np.ndarray,
                 n_isr_alt: int, n_ext_alt: int, center_time: pd.Timestamp, args, out_path: str):
    n_time = Ne_full.shape[1]

    median_interval_full = np.concatenate(
        [ds["median_interval_minutes"].values, np.full((n_ext_alt, n_time), -1, dtype=np.int8)], axis=0)
    deviation_full = np.concatenate(
        [ds["deviation_percent"].values, np.full((n_ext_alt, n_time), np.nan, dtype=np.float32)], axis=0)

    out_ds = xr.Dataset(
        data_vars=dict(
            altitude=(("altitude_gate",), altitude_grid, {
                "long_name": "altitude",
                "units": "km",
                "description": f"first {n_isr_alt} gates: native ISR altitude gates (unchanged); "
                                f"last {n_ext_alt}: {EXT_ALT_MIN:.0f}-{EXT_ALT_MAX:.0f} km IRI2020 extension grid, {EXT_ALT_STEP:.0f} km step",
            }),
            Ne=(("altitude_gate", "time"), Ne_full, {
                "units": "m-3",
                "long_name": "electron density: ISR measurement below ~650 km, IRI2020 extension above",
                "description": "below the native ISR ceiling: unchanged ISR Ne (NaNs preserved as-is); "
                                "above: IRI2020 climatological-library shape (leading PCA modes) best matching "
                                f"this profile's own shape at altitudes >= {args.fit_alt_min:.0f} km",
            }),
            n_valid_fit=(("time",), n_valid_fit, {
                "long_name": f"number of non-NaN ISR altitude points at or above {args.fit_alt_min:.0f} km used in the fit",
            }),
            rms_log10_residual_fit=(("time",), rms_resid.astype(np.float32), {
                "long_name": f"RMS residual of the log10 fit at the >= {args.fit_alt_min:.0f} km ISR altitudes used",
                "description": "NaN where the profile was skipped (fewer than min_valid points)",
            }),
            join_offset_log10=(("time",), join_offset.astype(np.float32), {
                "long_name": f"log10 bias correction applied at {EXT_ALT_MIN:.0f} km to pin the extension to the "
                              "last measured point, tapered to 0 by extension_transition_end_km",
                "description": "NaN where the profile was skipped",
            }),
            median_interval_minutes=(("altitude_gate", "time"), median_interval_full, dict(ds["median_interval_minutes"].attrs, **{
                "description": ds["median_interval_minutes"].attrs.get("description", "") + "; -1 for the IRI2020 extension gates (not applicable)",
            })),
            deviation_percent=(("altitude_gate", "time"), deviation_full, dict(ds["deviation_percent"].attrs, **{
                "description": ds["deviation_percent"].attrs.get("description", "") + "; NaN for the IRI2020 extension gates (not applicable)",
            })),
            topside_jump_repaired=(("time",), ds["topside_jump_repaired"].values, ds["topside_jump_repaired"].attrs),
            time_utc=(("time",), ds["time_utc"].values),
            local_time=(("time",), ds["local_time"].values, ds["local_time"].attrs),
        ),
        attrs=dict(ds.attrs, **{
            "title": ds.attrs.get("title", "") + " -- extended to 900 km via IRI2020",
            "extension_method": "log10-space least-squares fit of each profile against the leading "
                                 "principal components (EOFs) of an IRI2020 sample library, restricted to "
                                 "ISR altitudes above extension_fit_alt_min_km; the extension is pinned to the "
                                 "last measured point at extension_altitude_min_km with a smoothstep-tapered "
                                 "log10 bias correction (see join_offset_log10), fully decayed by "
                                 "extension_transition_end_km",
            "extension_center_time_utc": center_time.isoformat(),
            "extension_n_sample": args.nsample,
            "extension_hour_sample_range": args.hour_range,
            "extension_f107_sample_range": args.f107_range,
            "extension_ap_sample_range": args.ap_range,
            "extension_ig_sample_range": args.ig_range,
            "extension_rz_sample_range": args.rz_range,
            "extension_n_pca_modes": args.n_modes,
            "extension_fit_alt_min_km": args.fit_alt_min,
            "extension_min_valid_points": args.min_valid,
            "extension_altitude_min_km": EXT_ALT_MIN,
            "extension_altitude_max_km": EXT_ALT_MAX,
            "extension_altitude_step_km": EXT_ALT_STEP,
            "extension_transition_end_km": args.transition_end,
        }),
    )
    out_ds.to_netcdf(out_path)


def make_plots(altitude_grid, n_isr_alt, n_fit_alt, Ne_full, time_utc, n_valid_fit, rms_resid,
               Uk, C, transition_end_km, out_dir):
    good = np.isfinite(rms_resid)
    good_idx = np.flatnonzero(good)
    example_idx = good_idx[np.linspace(0, len(good_idx) - 1, 6).astype(int)] if len(good_idx) >= 6 else good_idx

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    for ax, t in zip(axes.ravel(), example_idx):
        log_ne = np.log10(np.maximum(Ne_full[:, t], 1.0))
        # Full-altitude-range reconstruction from the same fitted PCA
        # coefficients used for the extension -- lets you see fit quality
        # against the measured curve everywhere, not just above 650km
        # (including the <200km band the fit was never asked to match).
        log_fit = Uk @ C[:, t]
        ax.plot(log_fit, altitude_grid, "-", color=COLOR_FIT, lw=1.25, alpha=0.85,
                label="Best-fit IRI reconstruction")
        ax.plot(log_ne[:n_isr_alt], altitude_grid[:n_isr_alt], "o-", color=COLOR_MEASURED,
                ms=3, lw=1.5, label="ISR measured")
        ax.plot(log_ne[n_isr_alt:], altitude_grid[n_isr_alt:], "o-", color=COLOR_EXTENDED,
                ms=3, lw=1.5, label="IRI2020 extension")
        ax.axhline(altitude_grid[n_isr_alt - 1], color=COLOR_MUTED, lw=0.75, ls="--")
        ax.axhline(transition_end_km, color=COLOR_MUTED, lw=0.75, ls=":")
        ax.set_title(str(np.datetime_as_string(time_utc[t], unit="m")), fontsize=9, color=COLOR_INK)
        ax.set_facecolor(COLOR_SURFACE)
        ax.grid(True, color=COLOR_GRID, lw=0.75)
        ax.set_xlabel("log10(Ne [m$^{-3}$])", fontsize=8, color=COLOR_INK)
    axes[0, 0].set_ylabel("Altitude (km)", color=COLOR_INK)
    axes[1, 0].set_ylabel("Altitude (km)", color=COLOR_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    fig.suptitle("Example extended ISR profiles", y=1.04)
    fig.savefig(f"{out_dir}/plot_example_extended_profiles.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return example_idx

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(time_utc, rms_resid, ".", color=COLOR_MEASURED, ms=2, alpha=0.5)
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color=COLOR_GRID, lw=0.75)
    ax.set_xlabel("Time (UTC)", color=COLOR_INK)
    ax.set_ylabel("RMS log10 fit residual", color=COLOR_INK)
    ax.set_title("PCA-mode fit quality across the year", color=COLOR_INK)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/plot_fit_residual_vs_time.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(n_valid_fit, bins=np.arange(0, n_fit_alt + 2) - 0.5, color=COLOR_MEASURED, edgecolor="white")
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color=COLOR_GRID, lw=0.75, axis="y")
    ax.set_xlabel(f"Valid ISR altitude points used in fit (of {n_fit_alt} eligible)", color=COLOR_INK)
    ax.set_ylabel("Number of time samples", color=COLOR_INK)
    ax.set_title("Fit input-point coverage", color=COLOR_INK)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/plot_n_valid_fit_histogram.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_n_modes_sensitivity_plot(U: np.ndarray, Ne_isr: np.ndarray, altitude_grid: np.ndarray,
                                   n_isr_alt: int, fit_alt_mask: np.ndarray, min_valid: int,
                                   example_idx, time_utc, modes_list, out_dir: str):
    """Same example profiles as plot_example_extended_profiles.png, but
    refit with each of modes_list instead of the production --n-modes, to
    show directly what using more principal components does: beyond
    ~5-8 modes the library is into its noise floor (see
    plot_iri_library_singular_values.png) and the extension stops being
    the smooth, physically-reasonable curve the default N_PCA_MODES=3
    gives -- see the module docstring's N_PCA_MODES scan for the dataset-
    wide version of this same finding."""
    ramp = SEQUENTIAL_RAMP[: len(modes_list)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    for ax, t in zip(axes.ravel(), example_idx):
        col = Ne_isr[:, t]
        log_ne = np.log10(np.maximum(col, 1.0))
        ax.plot(log_ne[:n_isr_alt], altitude_grid[:n_isr_alt], "o-", color=COLOR_INK,
                ms=3, lw=1.25, label="ISR measured", zorder=10)
        for k, color in zip(modes_list, ramp):
            c, nv, rms = pca_fit_single(col, U[:n_isr_alt, :k], fit_alt_mask, min_valid)
            if c is None:
                continue
            log_fit = U[:, :k] @ c
            ax.plot(log_fit, altitude_grid, "-", color=color, lw=1.5, label=f"n_modes={k}")
        ax.axhline(altitude_grid[n_isr_alt - 1], color=COLOR_MUTED, lw=0.75, ls="--")
        ax.set_xlim(8, 14)
        ax.set_title(str(np.datetime_as_string(time_utc[t], unit="m")), fontsize=9, color=COLOR_INK)
        ax.set_facecolor(COLOR_SURFACE)
        ax.grid(True, color=COLOR_GRID, lw=0.75)
        ax.set_xlabel("log10(Ne [m$^{-3}$])", fontsize=8, color=COLOR_INK)
    axes[0, 0].set_ylabel("Altitude (km)", color=COLOR_INK)
    axes[1, 0].set_ylabel("Altitude (km)", color=COLOR_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(modes_list) + 1, frameon=False)
    fig.suptitle("Effect of n_modes on the same example profiles (x-axis capped at log10=14)", y=1.05)
    fig.savefig(f"{out_dir}/plot_n_modes_sensitivity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_library_diagnostic_plots(sampling_parameters: pd.DataFrame, log_iri: np.ndarray,
                                   altitude_grid: np.ndarray, n_isr_alt: int, n_modes: int, out_dir: str):
    """Why the 2000 library profiles end up nearly degenerate, motivating
    the leading-PCA-modes fit (see --n-modes): (1) the input-parameter
    histograms below, and (2) the profile-shape overlay and singular-value
    spectrum, which show how much of that input spread survives as actual
    profile-*shape* diversity."""
    labels = {"hour": "Hour (local time)", "f107": "F10.7 (sfu)", "ap": "ap index",
              "ig12": "IG12", "rz12": "Rz12"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), ["hour", "f107", "ap", "ig12", "rz12"]):
        ax.hist(sampling_parameters[col].dropna(), bins=30, color=COLOR_MEASURED, edgecolor="white")
        ax.set_facecolor(COLOR_SURFACE)
        ax.grid(True, color=COLOR_GRID, lw=0.75, axis="y")
        ax.set_xlabel(labels[col], color=COLOR_INK)
        ax.set_ylabel("Count", color=COLOR_INK)
    axes.ravel()[-1].axis("off")
    fig.suptitle(f"IRI2020 input-parameter distribution (N={sampling_parameters.shape[0]})", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/plot_iri_input_histograms.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(0)
    n_show = min(150, log_iri.shape[1])
    show_idx = rng.choice(log_iri.shape[1], size=n_show, replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in show_idx:
        ax.plot(log_iri[:, i], altitude_grid, color=COLOR_MEASURED, lw=0.6, alpha=0.15)
    ax.axhline(altitude_grid[n_isr_alt - 1], color=COLOR_MUTED, lw=0.75, ls="--")
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color=COLOR_GRID, lw=0.75)
    ax.set_xlabel("log10(Ne [m$^{-3}$])", color=COLOR_INK)
    ax.set_ylabel("Altitude (km)", color=COLOR_INK)
    ax.set_title(f"{n_show} of {log_iri.shape[1]} IRI2020 library profiles overlaid", color=COLOR_INK)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/plot_iri_library_profile_overlay.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Singular-value spectrum of the full (fit + extension altitude) library:
    # how many effectively independent profile *shapes* the nsample-column
    # library actually spans. The sharp drop after the first few modes is
    # why the fit uses only the leading n_modes of these (see --n-modes).
    s = np.linalg.svd(log_iri, compute_uv=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(s) + 1), s, "o-", color=COLOR_MEASURED, ms=4, lw=1.2)
    ax.axvline(n_modes, color=COLOR_EXTENDED, lw=1.2, ls="--", label=f"n_modes={n_modes}")
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color=COLOR_GRID, lw=0.75, which="both")
    ax.set_xlabel("Singular-value index", color=COLOR_INK)
    ax.set_ylabel("Singular value (log scale)", color=COLOR_INK)
    ax.set_title(f"Library shape diversity across all {log_iri.shape[0]} altitudes\n"
                  f"(condition number = {s[0] / s[-1]:.1e})", color=COLOR_INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/plot_iri_library_singular_values.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not os.environ.get("IRI2020_PATH"):
        sys.exit(
            "IRI2020_PATH is not set. Run 'source init_iri2020_env.sh' from "
            f"{_THIS_DIR} before running this script."
        )

    ds = xr.open_dataset(args.isr_file, decode_timedelta=False)
    altitude_isr = ds["altitude"].values
    Ne_isr = ds["Ne"].values.astype(np.float64)
    # time_utc is stored as ISO strings in this project's ISR preprocessing
    # output (confirmed directly against TROISR2025_nonan.nc -- a real,
    # previously-untested case: the original TROISR2025.nc this script was
    # written against apparently stored a numeric/datetime64 time_utc,
    # since pd.Series.mean()/np.datetime_as_string both require a real
    # datetime dtype, not strings). Parse once here so every downstream use
    # (center_time, make_plots/make_n_modes_sensitivity_plot's
    # np.datetime_as_string calls) gets a real datetime64 array regardless
    # of which dtype the source file used.
    time_utc = pd.to_datetime(ds["time_utc"].values).values
    n_isr_alt = altitude_isr.shape[0]

    # Station-agnostic (ISR_Integration_Plan.md Section 2/4.1.0): read the
    # real station location from the file's own attrs when present (every
    # file this project's ISR preprocessing produces carries them) rather
    # than assuming this is always the Tromso instrument.
    station_lat = float(ds.attrs.get("station_latitude", STATION_LAT))
    station_lon = float(ds.attrs.get("station_longitude", STATION_LON))
    station_name = str(ds.attrs.get("station_name", "ISR station"))

    center_time = pd.Timestamp(pd.Series(time_utc).mean())
    print(f"ISR file: {args.isr_file}")
    print(f"Station: {station_name} ({station_lat}, {station_lon})")
    print(f"n_time={Ne_isr.shape[1]}, n_isr_alt={n_isr_alt}, center_time={center_time.isoformat()}")

    altitude_ext = np.arange(EXT_ALT_MIN, EXT_ALT_MAX + EXT_ALT_STEP / 2, EXT_ALT_STEP)
    altitude_grid = np.concatenate([altitude_isr, altitude_ext])
    n_ext_alt = altitude_ext.shape[0]

    log_iri, sampling_parameters = build_iri_library(
        center_time, altitude_grid, args, station_lat=station_lat, station_lon=station_lon,
    )

    make_library_diagnostic_plots(sampling_parameters, log_iri, altitude_grid, n_isr_alt, args.n_modes, args.out)

    U = compute_svd_basis(log_iri)
    Uk = U[:, : args.n_modes]
    Uk_isr = Uk[:n_isr_alt]
    Uk_ext = Uk[n_isr_alt:]
    fit_alt_mask = altitude_isr >= args.fit_alt_min
    print(f"Fit basis: {args.n_modes} leading PCA modes, restricted to the "
          f"{int(fit_alt_mask.sum())} ISR altitude gates >= {args.fit_alt_min:.0f} km.")

    Ne_ext, n_valid_fit, rms_resid, C, join_offset = fit_and_extend(
        Ne_isr, Uk_isr, Uk_ext, fit_alt_mask=fit_alt_mask, min_valid=args.min_valid,
        altitude_isr=altitude_isr, altitude_ext=altitude_ext, transition_end_km=args.transition_end)

    n_skipped = int(np.sum(n_valid_fit < args.min_valid))
    print(f"Skipped {n_skipped}/{len(n_valid_fit)} profiles (fewer than {args.min_valid} valid points "
          f">= {args.fit_alt_min:.0f} km).")
    finite_resid = rms_resid[np.isfinite(rms_resid)]
    if finite_resid.size:
        print(f"RMS log10 residual: median={np.median(finite_resid):.4f}, "
              f"95th pct={np.percentile(finite_resid, 95):.4f}")
    finite_offset = join_offset[np.isfinite(join_offset)]
    if finite_offset.size:
        print(f"Join correction (log10, at {EXT_ALT_MIN:.0f} km, tapered to 0 by "
              f"{args.transition_end:.0f} km): median={np.median(np.abs(finite_offset)):.4f}, "
              f"95th pct={np.percentile(np.abs(finite_offset), 95):.4f}")

    Ne_full = np.concatenate([Ne_isr.astype(np.float32), Ne_ext], axis=0)

    # Station-agnostic output name (derived from the input file's own
    # stem, not hardcoded to Tromso) -- ISR_Integration_Plan.md Section 2.
    input_stem = Path(args.isr_file).stem
    out_path = f"{args.out}/{input_stem}_extended.nc"
    save_output(ds, altitude_grid, Ne_full, n_valid_fit, rms_resid, join_offset,
                n_isr_alt, n_ext_alt, center_time, args, out_path)
    print(f"Saved extended ISR profiles to {out_path}")

    example_idx = make_plots(altitude_grid, n_isr_alt, int(fit_alt_mask.sum()), Ne_full,
                              time_utc, n_valid_fit, rms_resid, Uk, C, args.transition_end, args.out)
    print(f"Saved diagnostic plots to {args.out}")

    modes_list = sorted(set([args.n_modes, 5, 8, 15]))
    make_n_modes_sensitivity_plot(U, Ne_isr, altitude_grid, n_isr_alt, fit_alt_mask, args.min_valid,
                                   example_idx, time_utc, modes_list, args.out)
    print(f"Saved n_modes sensitivity plot to {args.out}")


if __name__ == "__main__":
    main()
