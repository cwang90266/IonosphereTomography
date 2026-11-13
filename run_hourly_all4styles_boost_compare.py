#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run one-hour RO-only assimilation windows over a requested UTC interval,
with a cached reference-run IRI ensemble when available (otherwise a fresh IRI2020 ensemble), preserving run_package's
normal outputs and automatically selecting ISR validation stations that lie inside the current ROI.

For each hour, ISR validation stations are selected automatically from
TRO/SVL/MIL based on whether the station lies inside the configured ROI.
Only stations with an in-hour ISR profile are used for that hour.

Hours are assimilated only when BOTH conditions are met:
  - at least one active ISR station has an in-hour profile;
  - the scaled minimum number of accepted RO occultation profiles is available.

Skipped hours leave no hourly tomography folder. The daily status CSV keeps
one cumulative row per requested hour, including RO counts, ISR availability,
permanent assimilation eligibility, rejection reason, and a separate tomography
execution status.

The script precomputes/filters RO observations for each hour BEFORE running
IRI2020. Hours with no accepted RO occultation are skipped, so expensive IRI
and KF work is not wasted.

Example:

    source Driver_Scripts/init_iri2020_env.sh
    python run_hourly_all4styles_boost_compare.py \
        --date 2025-10-16 \
        --start-hour 0 \
        --end-hour 24

By default, large/cache intermediate artifacts are suppressed. To write
all package artifacts, including the normally suppressed .dat/.pkl/.nc files:

    python run_hourly_all4styles_boost_compare.py \
        --date 2025-10-16 \
        --start-hour 0 \
        --end-hour 24 \
        --output all


To run only one of the PCA styles:
    python run_hourly_all4styles_boost_compare.py \
        --date 2025-10-16 \
        --start-hour 0 \
        --end-hour 24 \
        --styles PCA_3D_10ex

The default run compares the parameterization styles selected in DEFAULT_STYLES:
  - ANCHOR
  - PCA_3D
  - PCA_3D_10ex
  - PCA_1D
  - PCA_1D_10ex
  - PCA_1D_10ex_ISR


and, for every style, BOTH covariance conditions:
  - no_boost
  - boosted (diagonal boost amplitude set by BOOST_AMPLITUDE)

Therefore cases per successful hour = len(DEFAULT_STYLES) x 2 covariance conditions.

For PCA_1D_10ex_ISR the script automatically reuses/builds:
  TROISR2025_nonan_extended.nc
  TROISR2025_nonan_pca_basis.nc

under ISR_PCA/PCA_data. The expensive TRO extension is built only if missing.
Use --rebuild-tro-pca-basis to refit the PCA basis from the existing extended
TRO file. An explicit --isr-pca-basis can still be supplied as an override.
"""

from __future__ import annotations

# This hourly wrapper does not create/use an iri_debug directory. Current
# EDPSamples IRI execution uses tempfile.TemporaryDirectory() for scratch files.

import argparse
import csv
import gc
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# USER CONFIGURATION
# === EDIT HERE ONLY ===
# =============================================================================

# Project / data paths. I made it to match the server branch you have, please check if they're correct
REPO = Path("/home/knelson/src/IonosphereTomography")  # === Edit here ===
RO_ROOT = Path("/home/pyxis/Test/2026-10-13_ionosphere_tomography/podTc2")  # === Edit here ===
ISR_DIR = Path("/home/pyxis/Test/2026-10-13_ionosphere_tomography/ISR_data")  # === Edit here ===
OUTPUT_ROOT = Path("/mnt/tmp-sbem-ionosphere-tomography/tomography_output")  # === Edit here ===

# Tomography center: switch between "MIL", "TRO", and "SVL". center lat lon will auto update below yo!
CENTER_SITE = "MIL"  # === Edit here ===  switch between MIL/TRO/SVL

# Campaign date range used in LABEL_OF_EACH_RUN and RUN_MESSAGE.Example below becomes "oct1417_2025".
RUN_DATE_RANGE = ("2025-08-27", "2025-08-27")  # === Edit here ===

# Observation ROI radius, also dynamic to run label and message
ROI_RADIUS_KM = 2000.0  # === Edit here ===

# Diagonal covariance boost amplitude. Example 0.3 becomes "Op3Boost" in the label.
BOOST_AMPLITUDE = 0.3  # === Edit here ===

# Observation-error standard deviation for TEC assimilation, in TECU.
OBS_SIGMA_TECU = 5.0  # === Edit here ===

# Downsample the nominal IRI spread ranges by this factor.
# Base ranges are:
#   hour=3, f107=30, ap=30, ig=12, rz=12
# Example factor=3 -> hour=1, f107=10, ap=10, ig=4, rz=4.
IRI_DOWNSAMPLING_FACTOR = 1  # === Edit here ===

# Required fraction of the RO LOS that must lie inside the tomography ROI.
RO_AREA_FRACTION_REQUIRED = 0.65  # === Edit here ===

# Scale the fraction of PCA variance that is discarded.
# 1  -> original thresholds
# 10 -> discard 10x more variance:
#       0.9999 -> 0.999
#       0.99999 -> 0.9999
PCA_DISCARD_SCALE = 1.0  # === Edit here ===

# Choose the parameterization styles to run.
DEFAULT_STYLES = [
    #"ANCHOR",
    "PCA_3D",
    #"PCA_3D_10ex",
    "PCA_1D",
    #"PCA_1D_10ex",
    #"PCA_1D_10ex_ISR",
]  # === Edit here ===

# Optional reference-run label for reusing the exact hourly IRI ensemble.
# Set "" or None to disable reference-IRI reuse.
REFERENCE_RUN_LABEL = ""  # === Edit here ===

# Example label may look like, gross long
# MIL_aug27_2025_R2k_BOp3_Obs5_IRIScale3_LOSfracOp65_PCAdisScale10_3D_1D

# =============================================================================
# Derived configuration -- Don't need to edit for normal runs
# =============================================================================

for _p in (
    REPO,
    REPO / "Parameterization",
    REPO / "EDPSamples",
    REPO / "IRI_Sample_Inputs",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Assimilation_Cycle.cycle_config import CycleConfig, default_altitude_grid
from Assimilation_Cycle.package_run import run_package
from Assimilation_Cycle import ensemble_init, observation_stream, output, iri_selection
from Assimilation_Cycle import cycle_driver as cycle_driver_module
from Assimilation_Cycle import package_run as package_run_module
from Assimilation_Cycle.diagonal_boost import apply_diagonal_boost
from Parameterization import Parameterized_EDPSamples
from Assimilation_Cycle import isr_pca_basis as ipb
from Assimilation_Cycle.isr_comparison import load_isr_dataset, nearest_isr_profile
from Ensemble_Kalman_Engine import EnsembleState
from Ensemble_Kalman_Engine.driver import GeneralEnKFDriver


# Center coordinates are selected only from CENTER_SITE above.
CENTER_COORDS = {
    "MIL": (42.6195, -71.4917),  # Millstone Hill
    "TRO": (69.6000, 19.2000),   # Tromso
    "SVL": (78.1530, 16.0290),   # Svalbard
}

CENTER_SITE = str(CENTER_SITE).strip().upper()
if CENTER_SITE not in CENTER_COORDS:
    raise ValueError(
        f"CENTER_SITE must be one of {sorted(CENTER_COORDS)}, got {CENTER_SITE!r}"
    )

CENTER_LAT, CENTER_LON = CENTER_COORDS[CENTER_SITE]


def _run_range_tag(date_range: tuple[str, str]) -> str:
    """Convert ('2025-10-14','2025-10-17') -> 'oct1417_2025'."""
    if len(date_range) != 2:
        raise ValueError("RUN_DATE_RANGE must contain exactly (start_date, end_date)")

    start = datetime.strptime(str(date_range[0]), "%Y-%m-%d")
    end = datetime.strptime(str(date_range[1]), "%Y-%m-%d")

    if end < start:
        raise ValueError("RUN_DATE_RANGE end date must be >= start date")

    mon1 = start.strftime("%b").lower()
    mon2 = end.strftime("%b").lower()

    if start.year == end.year and start.month == end.month:
        if start.day == end.day:
            return f"{mon1}{start.day:02d}_{start.year}"
        return f"{mon1}{start.day:02d}{end.day:02d}_{start.year}"

    if start.year == end.year:
        return f"{mon1}{start.day:02d}{mon2}{end.day:02d}_{start.year}"

    return (
        f"{mon1}{start.day:02d}_{start.year}_"
        f"{mon2}{end.day:02d}_{end.year}"
    )


def _boost_tag(amplitude: float) -> str:
    """
    Compact label form requested for the run name:
      0.3  -> Op3
      0.35 -> Op35
      1.2  -> 1p2
    """
    value = f"{float(amplitude):g}"
    if value.startswith("0."):
        return "Op" + value[2:]
    return value.replace(".", "p").replace("-", "m")


def _compact_number_tag(value: float) -> str:
    """Compact numeric tag: 0.65 -> Op65, 10 -> 10, 2.5 -> 2p5."""
    s = f"{float(value):g}"
    if s.startswith("0."):
        return "0p" + s[2:]
    return s.replace(".", "p").replace("-", "m")


def _styles_tag(styles: list[str]) -> str:
    """Compact, stable style-set label."""
    abbreviations = {
        "ANCHOR": "A",
        "PCA_3D": "3D",
        "PCA_3D_10ex": "3DL",
        "PCA_1D": "1D",
        "PCA_1D_10ex": "1DL",
        "PCA_1D_10ex_ISR": "ISR",
    }
    unknown = [s for s in styles if s not in abbreviations]
    if unknown:
        raise ValueError(f"Unknown style(s) in DEFAULT_STYLES: {unknown}")
    return "_".join(abbreviations[s] for s in styles)


RUN_RANGE_TAG = _run_range_tag(RUN_DATE_RANGE)
BOOST_TAG = _boost_tag(BOOST_AMPLITUDE)
OBS_SIG_TAG = _compact_number_tag(OBS_SIGMA_TECU)
IRI_DOWNSAMPLE_TAG = _compact_number_tag(IRI_DOWNSAMPLING_FACTOR)
AREA_FRACTION_TAG = _compact_number_tag(RO_AREA_FRACTION_REQUIRED)
PCA_DISCARD_TAG = _compact_number_tag(PCA_DISCARD_SCALE)
STYLES_TAG = _styles_tag(DEFAULT_STYLES)

# Compact but readable fully parameterized run label.
# Examples:
#   ROI 2000 km -> R2k
#   ROI 2500 km -> R2p5k
#   LOS fraction 0.65 -> LOSfracOp65
#   LOS fraction 1.0  -> LOSfrac1
#
# Example full label:
#   MIL_aug2529_2025_R2k_BOp3_Obs5_IRIScale3_LOSfracOp65_PCAdisScale10_A_3D_3DL_1D_1DL_ISR
ROI_TAG = _compact_number_tag(ROI_RADIUS_KM / 1000.0)

LABEL_OF_EACH_RUN = (
    f"{CENTER_SITE}_{RUN_RANGE_TAG}_"
    f"R{ROI_TAG}k_"
    f"B{BOOST_TAG}_"
    f"Obs{OBS_SIG_TAG}_"
    f"IRIScale{IRI_DOWNSAMPLE_TAG}_"
    f"LOSfrac{AREA_FRACTION_TAG}_"
    f"PCAdisScale{PCA_DISCARD_TAG}_"
    f"{STYLES_TAG}"
)

# Keep the run message synchronized automatically with the generated label.
RUN_MESSAGE = LABEL_OF_EACH_RUN # ===Edit here (maybe not, Kevin)===
# The run messges will just be label by default, if nothing special, can just leave it alone


TRO_ISR_FILE = ISR_DIR / "TROISR2025_nonan.nc"
SVL_ISR_FILE = ISR_DIR / "SVLISR2025_nonan.nc"
MIL_ISR_FILE = ISR_DIR / "MILISR2025_nonan.nc"

# All supported ISR validation inputs. The driver automatically uses only
# stations whose site lies inside the current tomography ROI.
ISR_FILES = {
    "TRO": TRO_ISR_FILE,
    "SVL": SVL_ISR_FILE,
    "MIL": MIL_ISR_FILE,
}

# PCA_1D_10ex_ISR training artifacts.
# IMPORTANT: this remains TRO-trained regardless of CENTER_SITE.
# TRO/SVL/MIL selection above controls tomography center / validation only.
TRO_ISR_EXTENDED_FILE = ISR_DIR / "TROISR2025_nonan_extended.nc"
TRO_ISR_PCA_BASIS_FILE = ISR_DIR / "TROISR2025_nonan_pca_basis.nc"


ALL_STYLE_DISPLAY = {
    "ANCHOR": "ANCHOR",
    "PCA_3D": "IRI-trained PCA_3D",
    "PCA_3D_10ex": "IRI-trained PCA_3D_10ex",
    "PCA_1D": "IRI-trained PCA_1D",
    "PCA_1D_10ex": "IRI-trained PCA_1D_10ex",
    "PCA_1D_10ex_ISR": "TRO-ISR-trained PCA_1D_10ex_ISR",
}

# Base retaining thresholds before PCA_DISCARD_SCALE is applied.
# The ISR-trained basis historically uses 0.9999 (= 1 - 1e-4).
BASE_PCA_RETAINING_THRESHOLD = {
    "PCA_3D": 0.9999,
    "PCA_3D_10ex": 0.9999,
    "PCA_1D": 0.99999,
    "PCA_1D_10ex": 0.99999,
    "PCA_1D_10ex_ISR": 0.9999,
}


def _scaled_retaining_threshold(base_threshold: float) -> float:
    """Scale discarded variance while keeping the result in (0, 1)."""
    discarded = 1.0 - float(base_threshold)
    scaled_discarded = discarded * float(PCA_DISCARD_SCALE)
    threshold = 1.0 - scaled_discarded
    if not (0.0 < threshold < 1.0):
        raise ValueError(
            "PCA_DISCARD_SCALE produces an invalid retaining threshold: "
            f"base={base_threshold}, scale={PCA_DISCARD_SCALE}, result={threshold}"
        )
    return threshold


STYLE_DISPLAY = {
    style: ALL_STYLE_DISPLAY[style]
    for style in DEFAULT_STYLES
}

DEFAULT_HYPER_PARAMS = {
    style: {
        "retaining_threshold": _scaled_retaining_threshold(
            BASE_PCA_RETAINING_THRESHOLD[style]
        )
    }
    for style in DEFAULT_STYLES
    if style in BASE_PCA_RETAINING_THRESHOLD
}


RUN_CONDITIONS = [
    {
        "name": "no_boost",
        "label": "No boost",
        "diagonal_boost_amplitude": None,
        "diagonal_boost_log_space": False,
        "diagonal_boost_rng_seed": None,
    },
    {
        "name": "boosted",
        "label": f"Boost {BOOST_AMPLITUDE}",
        "diagonal_boost_amplitude": BOOST_AMPLITUDE,
        "diagonal_boost_log_space": True,
        "diagonal_boost_rng_seed": 888,
    },
]


GRID_RADIUS_DEG = 18.0 * (ROI_RADIUS_KM / 2000.0)

# Scale minimum RO count with ROI area.
# 2000 km -> 5
# 3000 km -> 11.25 -> floor -> 11
# 4000 km -> 20
MIN_RO_PROFILES = int(
    np.floor(5.0 * (ROI_RADIUS_KM / 2000.0) ** 2)
)

N_ENSEMBLE = 2000
MAX_TEC_PER_BATCH = 100_000
ANALYSIS_RNG_SEED = 777

# Diagonal covariance boost configuration.
BOOST_TAPER_START_KM = 400.0
BOOST_TAPER_END_KM = 700.0
BOOST_TAPER_FLOOR = 1.0/7.0

BASE_IRI_SPREAD_KWARGS = {
    "hour_sample_range": 3,
    "f107_sample_range": 30,
    "ap_sample_range": 30,
    "ig_sample_range": 12,
    "rz_sample_range": 12,
}


def _downsample_iri_range(value: int) -> int:
    """Divide a nominal integer IRI range by IRI_DOWNSAMPLING_FACTOR."""
    factor = float(IRI_DOWNSAMPLING_FACTOR)
    if factor <= 0:
        raise ValueError("IRI_DOWNSAMPLING_FACTOR must be > 0")
    scaled = int(round(float(value) / factor))
    return max(1, scaled)


IRI_SPREAD_KWARGS = {
    key: _downsample_iri_range(value)
    for key, value in BASE_IRI_SPREAD_KWARGS.items()
}

RO_KWARGS_TEMPLATE = {
    "roi_mode": "full_los",
    "alt_limit_km": None,
    "fraction_required": RO_AREA_FRACTION_REQUIRED,
    "min_valid_rays": 50,
    "max_rays_per_occultation": 200,
    "max_lowest_tangent_alt_km": 400.0,
    "require_dcb_flags": True,
    "duplicate_time_tolerance_s": 30.0,
    "include_abel": True,
}



# =============================================================================
# Helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="UTC date, YYYY-MM-DD")
    parser.add_argument(
        "--start-hour", type=int, default=0,
        help="First UTC hour to run, inclusive (default 0).",
    )
    parser.add_argument(
        "--end-hour", type=int, default=24,
        help="UTC hour boundary to stop at, exclusive (default 24).",
    )
    parser.add_argument(
        "--styles", nargs="+", default=DEFAULT_STYLES,
        help=f"Parameterization styles. Default: {' '.join(DEFAULT_STYLES)}",
    )
    parser.add_argument(
        "--isr-pca-basis", type=Path, default=None,
        help=(
            "Optional explicit PCA basis override for PCA_1D_10ex_ISR. "
            "Normally omit this: the script builds/reuses the TRO-only basis "
            "TROISR2025_nonan_pca_basis.nc."
        ),
    )
    parser.add_argument(
        "--rebuild-tro-pca-basis",
        action="store_true",
        help=(
            "Refit TROISR2025_nonan_pca_basis.nc from the existing "
            "TROISR2025_nonan_extended.nc. The extended TRO file itself is "
            "only regenerated when missing."
        ),
    )
    parser.add_argument(
        "--n-ensemble", type=int, default=N_ENSEMBLE,
        help=f"IRI/KF ensemble size (default {N_ENSEMBLE}).",
    )
    parser.add_argument(
        "--continue-on-error", action=argparse.BooleanOptionalAction, default=True,
        help="Continue to later hours after a failed hour (default true).",
    )
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=False,
        help="Skip an hour if its hourly_summary.csv already says success.",
    )
    parser.add_argument(
        "--output",
        choices=("minimal", "all"),
        default="minimal",
        help=(
            "Output policy for large/cache intermediate artifacts. "
            "'minimal' (default) suppresses apf107.dat, ig_rz.dat, "
            "*_iri_sample_inputs.pkl, *_edp_samples.nc, *_parameterized.nc, "
            "and *_ensembles/ensemble_batch_*.nc. "
            "'all' writes those artifacts too. Scientific figures, CSV summaries, "
            "processed RO observations, and validation products are written in both modes."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> datetime:
    day = datetime.strptime(args.date, "%Y-%m-%d")

    if OBS_SIGMA_TECU <= 0:
        raise ValueError("OBS_SIGMA_TECU must be > 0")
    if IRI_DOWNSAMPLING_FACTOR <= 0:
        raise ValueError("IRI_DOWNSAMPLING_FACTOR must be > 0")
    if not (0.0 < RO_AREA_FRACTION_REQUIRED <= 1.0):
        raise ValueError("RO_AREA_FRACTION_REQUIRED must be in (0, 1]")
    if PCA_DISCARD_SCALE <= 0:
        raise ValueError("PCA_DISCARD_SCALE must be > 0")

    unknown_styles = [
        style for style in args.styles
        if style not in ALL_STYLE_DISPLAY
    ]
    if unknown_styles:
        raise ValueError(f"Unsupported parameterization style(s): {unknown_styles}")
    if not (0 <= args.start_hour < 24):
        raise ValueError("--start-hour must be 0..23")
    if not (1 <= args.end_hour <= 24):
        raise ValueError("--end-hour must be 1..24")
    if args.end_hour <= args.start_hour:
        raise ValueError("--end-hour must be greater than --start-hour")
    if args.n_ensemble < 2:
        raise ValueError("--n-ensemble must be >= 2")
    return day


def ro_day_dir(day: datetime) -> Path:
    doy = day.timetuple().tm_yday
    return RO_ROOT / f"{day.year}.{doy:03d}"


def hour_folder_name(start: datetime, end: datetime) -> str:
    return f"{start:%Y-%m-%d}_{start:%H%M}{end:%H%M}"


def hour_label(start: datetime, end: datetime) -> str:
    return f"{start:%Y-%m-%d %H:%M}-{end:%H:%M} UTC"


def canonical_hour_edp_path(run_label: str, start: datetime, end: datetime) -> Path:
    """
    Canonical cache location for the unboosted hourly IRI EDPSamples.

    Example:
      /home/pin/Desktop/tomography_output/test_label/
      2025-10-14_09001000/edp_samples.nc
    """
    return (
        OUTPUT_ROOT
        / run_label
        / hour_folder_name(start, end)
        / "edp_samples.nc"
    )


def find_reference_edp_samples(
    start: datetime,
    end: datetime,
) -> Path | None:
    """
    Find a previously saved unboosted EDPSamples file for exactly this hour.

    Search order:
      1. Canonical cache:
           <OUTPUT_ROOT>/<REFERENCE_RUN_LABEL>/<hour>/edp_samples.nc

      2. Backward-compatible search under that hour's no_boost folder:
           <hour>/no_boost/**/*_edp_samples.nc

    Boosted EDP files are NEVER used as the reference prior.

    Returns None when reuse is disabled or no matching file exists.
    """
    if REFERENCE_RUN_LABEL in (None, ""):
        return None

    reference_root = OUTPUT_ROOT / str(REFERENCE_RUN_LABEL)
    hour_root = reference_root / hour_folder_name(start, end)

    canonical = hour_root / "edp_samples.nc"
    if canonical.is_file():
        return canonical.resolve()

    no_boost_dir = hour_root / "no_boost"
    if no_boost_dir.is_dir():
        candidates = sorted(no_boost_dir.rglob("*_edp_samples.nc"))

        if len(candidates) == 1:
            return candidates[0].resolve()

        if len(candidates) > 1:
            # Prefer a file whose name explicitly identifies no_boost.
            explicit = [
                p for p in candidates
                if "no_boost" in p.name.lower()
            ]
            if len(explicit) == 1:
                return explicit[0].resolve()

            raise RuntimeError(
                "Multiple candidate reference EDPSamples files found for "
                f"{hour_label(start, end)} under {no_boost_dir}:\n  "
                + "\n  ".join(str(p) for p in candidates)
                + "\nCreate/use the canonical hour-level edp_samples.nc "
                  "or remove the ambiguous duplicates."
            )

    return None


def save_canonical_hour_edp(
    edp_samples,
    run_label: str,
    start: datetime,
    end: datetime,
) -> Path:
    """
    Save the unboosted hourly IRI EDPSamples to the canonical cache path.

    This deliberately stores only the base/no-boost ensemble so later tuning
    runs can reuse the exact same initial IRI Ne ensemble.
    """
    path = canonical_hour_edp_path(run_label, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Directly refresh this label/hour's canonical cache with the exact
    # unboosted base ensemble used by this run. Reusing the same run label
    # after changing setup therefore cannot leave a stale cache behind.
    edp_samples.saveNetCDF(str(path))

    return path


def unique_ro_entries(batches) -> list:
    seen = set()
    entries = []
    for batch in batches:
        for entry, _sl in batch.entry_ray_ranges:
            if entry.obs_type != "RO":
                continue
            key = id(entry)
            if key not in seen:
                seen.add(key)
                entries.append(entry)
    return entries


def count_tec_rays(batches) -> int:
    return int(sum(len(batch.y_obs) for batch in batches))


def save_figure(fig, path: Path, dpi: int = 160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def interp_log10(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Model-density interpolation to ISR native gates; no extrapolation."""
    x_new = np.asarray(x_new, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    logy = np.log10(np.clip(y, 1.0, None))
    out = np.interp(x_new, x, logy, left=np.nan, right=np.nan)
    return 10.0 ** out


def great_circle_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two geographic points."""
    r_earth_km = 6371.0
    p1 = np.deg2rad(float(lat1))
    p2 = np.deg2rad(float(lat2))
    dlat = np.deg2rad(float(lat2) - float(lat1))
    dlon = np.deg2rad(float(lon2) - float(lon1))
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    )
    return float(2.0 * r_earth_km * np.arcsin(np.sqrt(a)))


def load_isr_registry() -> tuple[dict, dict]:
    """
    Load every available ISR input file, then select validation stations by ROI.

    Missing validation files are allowed. PCA_1D_10ex_ISR training remains
    a separate TRO-basis concern handled by ensure_tro_isr_pca_basis().
    """
    datasets = {}
    active = {}

    print("Loading ISR validation registry...")
    for code, path in ISR_FILES.items():
        if not path.is_file():
            print(f"  {code}: file not found -> validation unavailable ({path})")
            continue

        ds = load_isr_dataset(path)
        datasets[code] = ds
        distance_km = great_circle_distance_km(
            CENTER_LAT, CENTER_LON, ds.station.lat, ds.station.lon
        )
        in_roi = distance_km <= ROI_RADIUS_KM

        print(
            f"  {code}: {ds.station.name}, {len(ds.time)} profiles, "
            f"{ds.altitude.min():.1f}-{ds.altitude.max():.1f} km, "
            f"distance={distance_km:.0f} km -> "
            f"{'ACTIVE' if in_roi else 'outside ROI / ignored'}"
        )

        if in_roi:
            active[code] = ds

    if not active:
        raise RuntimeError(
            "No ISR validation station lies inside the current tomography ROI. "
            f"Center=({CENTER_LAT:.4f}, {CENTER_LON:.4f}), "
            f"ROI_RADIUS_KM={ROI_RADIUS_KM:g}."
        )

    return datasets, active


def nearest_profile_in_hour(isr, start: datetime, end: datetime):
    """
    Select the ISR profile nearest the one-hour midpoint, but require the
    selected ISR sample itself to lie within [start, end]. This avoids
    silently borrowing ISR truth from a different hour when coverage is absent.
    """
    midpoint = start + (end - start) / 2
    nearest = nearest_isr_profile(isr, midpoint)
    if nearest is None:
        return None
    ts, density = nearest
    ts_py = pd.Timestamp(ts)
    if not (pd.Timestamp(start) <= ts_py <= pd.Timestamp(end)):
        return None
    return ts_py, np.asarray(density, dtype=float)


def ensure_tro_isr_pca_basis(
    requested: Path | None,
    styles: list[str],
    rebuild_basis: bool = False,
) -> Path | None:
    """
    Resolve/build the PCA basis for PCA_1D_10ex_ISR.

    Training provenance is deliberately fixed:
        TROISR2025_nonan.nc -> TROISR2025_nonan_extended.nc
                            -> TROISR2025_nonan_pca_basis.nc

    SVLISR2025_nonan.nc is never passed to the extension or PCA-fitting code.
    It remains validation-only.
    """
    if "PCA_1D_10ex_ISR" not in styles:
        return None

    # Explicit override is allowed for reproducibility with an already-built
    # basis, but the normal/default path below is guaranteed TRO-only.
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ISR PCA basis not found: {path}")
        print("Using explicit ISR PCA basis override:", path)
        return path

    if not TRO_ISR_FILE.is_file():
        raise FileNotFoundError(
            f"Tromso ISR training file not found: {TRO_ISR_FILE}"
        )

    extension_script = REPO / "extend_isr_edp_with_iri2020.py"

    # Native TRO ISR stops below the production 90-900 km grid. Build the
    # project-standard IRI-assisted topside extension once, using TRO only.
    if not TRO_ISR_EXTENDED_FILE.is_file():
        if not extension_script.is_file():
            raise FileNotFoundError(
                "TRO ISR PCA requires the extended TRO file, but neither the "
                f"file nor extension script was found. Missing: {extension_script}"
            )

        print("\nBuilding TRO-only ISR extension for PCA training...")
        print("  input :", TRO_ISR_FILE)
        print("  output:", TRO_ISR_EXTENDED_FILE)
        subprocess.run(
            [
                sys.executable,
                str(extension_script),
                "--isr-file", str(TRO_ISR_FILE),
                "--out", str(ISR_DIR),
            ],
            check=True,
        )

        if not TRO_ISR_EXTENDED_FILE.is_file():
            raise RuntimeError(
                "TRO extension command completed but expected file was not created: "
                f"{TRO_ISR_EXTENDED_FILE}"
            )
    else:
        print("Reusing TRO-only extended ISR training file:", TRO_ISR_EXTENDED_FILE)

    # Build once and reuse for every hour. The ISR PCA basis is NOT re-fit
    # from the hourly IRI ensemble.
    if rebuild_basis or not TRO_ISR_PCA_BASIS_FILE.is_file():
        print("\nFitting TRO-only ISR PCA basis...")
        print("  training file:", TRO_ISR_EXTENDED_FILE)
        print("  SVL training  : DISABLED (validation only)")
        basis = ipb.build_isr_pca_basis(
            TRO_ISR_EXTENDED_FILE,
            default_altitude_grid(),
            retaining_threshold=DEFAULT_HYPER_PARAMS[
                "PCA_1D_10ex_ISR"
            ]["retaining_threshold"],
        )
        ipb.save_isr_pca_basis(basis, TRO_ISR_PCA_BASIS_FILE)
        print(
            "  saved:", TRO_ISR_PCA_BASIS_FILE,
            f"| retained={basis.diagnostics.n_retained}",
            f"| profiles={basis.diagnostics.n_profiles_used}",
        )
    else:
        print("Reusing TRO-only ISR PCA basis:", TRO_ISR_PCA_BASIS_FILE)

    return TRO_ISR_PCA_BASIS_FILE.resolve()


def build_hyper_params_by_style(styles: list[str], basis_path: Path | None) -> dict:
    """Build hyper-parameters once; inject the pre-fit ISR PCA explicitly."""
    out = {
        style: dict(DEFAULT_HYPER_PARAMS[style])
        for style in styles
        if style in DEFAULT_HYPER_PARAMS
    }
    if "PCA_1D_10ex_ISR" in styles:
        if basis_path is None:
            raise RuntimeError("ISR PCA basis path is required for PCA_1D_10ex_ISR")
        basis = ipb.load_isr_pca_basis(basis_path)
        out["PCA_1D_10ex_ISR"] = ipb.hyper_params_from_basis(basis)
        print(
            "Loaded ISR PCA basis:", basis_path,
            f"| retained={basis.diagnostics.n_retained}",
            f"| profiles={basis.diagnostics.n_profiles_used}",
        )
    return out



class _NoArtifactRunContext:
    """
    Hourly run context that always preserves the controlled A/B behavior
    (boosted reuses the exact no-boost base EDP in memory) while making
    heavy/cache artifacts optional.

    output_mode="minimal" (default) suppresses:
      - *_edp_samples.nc
      - *_parameterized.nc
      - *_ensembles/ensemble_batch_*.nc
      - *_iri_sample_inputs.pkl
      - apf107.dat
      - ig_rz.dat

    output_mode="all" writes those artifacts using the normal package paths.

    In both modes the exact in-memory Parameterized_EDPSamples objects are
    captured so downstream ISR validation does not depend on reopening a
    parameterized netCDF file.
    """

    def __init__(self, *, base_edp_samples=None, output_mode: str = "minimal"):
        if output_mode not in {"minimal", "all"}:
            raise ValueError(f"Unknown output_mode={output_mode!r}")
        self.base_edp_samples = base_edp_samples
        self.output_mode = output_mode
        self.parameterizations = {}

        self._orig_load_or_build = None
        self._orig_load_or_build_iri = None
        self._orig_save_parameterized = None
        self._orig_save_ensemble = None
        self._iri_tmpdir = None

    def __enter__(self):
        self._orig_load_or_build = ensemble_init.load_or_build_edp_samples
        self._orig_load_or_build_iri = iri_selection.load_or_build_iri_sample_inputs
        self._orig_save_parameterized = Parameterized_EDPSamples.saveNetCDF
        self._orig_save_ensemble = cycle_driver_module.save_ensemble_netcdf

        base_edp_samples = self.base_edp_samples
        output_mode = self.output_mode
        orig_load = self._orig_load_or_build
        orig_load_iri = self._orig_load_or_build_iri
        orig_save_parameterized = self._orig_save_parameterized
        captured = self.parameterizations

        if output_mode == "minimal":
            self._iri_tmpdir = tempfile.TemporaryDirectory(prefix="ionosphere_iri_")
            iri_tmpdir = self._iri_tmpdir.name

            def _load_or_build_iri_no_artifacts(cfg):
                tmp_cfg = replace(
                    cfg,
                    output_dir=iri_tmpdir,
                    iri_sample_inputs_output_path=None,
                )
                return orig_load_iri(tmp_cfg)

            iri_selection.load_or_build_iri_sample_inputs = _load_or_build_iri_no_artifacts

        def _load_or_build_controlled_ab(cfg):
            if base_edp_samples is not None:
                amplitude = cfg.diagonal_boost_amplitude
                if amplitude in (None, 0):
                    edp = base_edp_samples
                else:
                    edp = apply_diagonal_boost(
                        base_edp_samples,
                        amplitude=amplitude,
                        vertical_scale_km=cfg.diagonal_boost_vertical_scale_km,
                        horizontal_scale_km=cfg.diagonal_boost_horizontal_scale_km,
                        rng=np.random.default_rng(cfg.diagonal_boost_rng_seed),
                        log_space=cfg.diagonal_boost_log_space,
                        taper_start_km=cfg.diagonal_boost_taper_start_km,
                        taper_end_km=cfg.diagonal_boost_taper_end_km,
                        taper_floor=cfg.diagonal_boost_taper_floor,
                    )

                if output_mode == "all" and cfg.edp_samples_output_path is not None:
                    Path(cfg.edp_samples_output_path).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    edp.saveNetCDF(cfg.edp_samples_output_path)
                return edp

            if output_mode == "minimal":
                return orig_load(replace(cfg, edp_samples_output_path=None))
            return orig_load(cfg)

        def _capture_parameterized(pes, path):
            style = getattr(pes, "style", None)
            if style is None:
                name = Path(path).name
                for candidate in DEFAULT_STYLES:
                    if f"_{candidate}_parameterized.nc" in name:
                        style = candidate
                        break
            if style is not None:
                captured[str(style)] = pes

            if output_mode == "all":
                return orig_save_parameterized(pes, path)
            return None

        def _skip_ensemble_nc(*args, **kwargs):
            return None

        ensemble_init.load_or_build_edp_samples = _load_or_build_controlled_ab
        Parameterized_EDPSamples.saveNetCDF = _capture_parameterized

        if output_mode == "minimal":
            cycle_driver_module.save_ensemble_netcdf = _skip_ensemble_nc

        return self

    def __exit__(self, exc_type, exc, tb):
        ensemble_init.load_or_build_edp_samples = self._orig_load_or_build
        iri_selection.load_or_build_iri_sample_inputs = self._orig_load_or_build_iri
        Parameterized_EDPSamples.saveNetCDF = self._orig_save_parameterized
        cycle_driver_module.save_ensemble_netcdf = self._orig_save_ensemble
        if self._iri_tmpdir is not None:
            self._iri_tmpdir.cleanup()
            self._iri_tmpdir = None
        return False


def make_precheck_cfg(
    start: datetime,
    end: datetime,
    ro_dir: Path,
    ro_output_path: Path,
) -> CycleConfig:
    ro_kwargs = dict(RO_KWARGS_TEMPLATE)
    ro_kwargs["podtc_dir"] = str(ro_dir)
    return CycleConfig(
        start_time=start.isoformat(),
        end_time=end.isoformat(),
        center_lat=CENTER_LAT,
        center_lon=CENTER_LON,
        radius_km=ROI_RADIUS_KM,
        grid_radius_deg=GRID_RADIUS_DEG,
        grid_margin_km=0.0,
        max_tec_per_batch=MAX_TEC_PER_BATCH,
        obs_sigma=OBS_SIGMA_TECU,
        obs_sources="RO",
        ro_kwargs=ro_kwargs,
        ro_observations_output_path=str(ro_output_path),
    )


def make_run_cfg(
    start: datetime,
    end: datetime,
    ro_dir: Path,
    ro_observations_path: Path,
    output_dir: Path,
    label: str,
    styles: list[str],
    n_ensemble: int,
    hyper_params_by_style: dict,
    isr_pca_basis_path: Path | None,
    condition: dict,
    active_hour_isr_files: dict[str, Path],
    base_edp_samples_path: Path | None = None,
) -> CycleConfig:
    ro_kwargs = dict(RO_KWARGS_TEMPLATE)
    ro_kwargs["podtc_dir"] = str(ro_dir)

    # no_boost: build a fresh IRI2020 ensemble for this hour.
    # boosted : reuse the exact saved no-boost IRI ensemble for a controlled A/B
    #           comparison, then apply diagonal covariance boosting BEFORE
    #           parameterization.
    return CycleConfig(
        label=label,
        output_dir=str(output_dir),
        start_time=start.isoformat(),
        end_time=end.isoformat(),
        center_lat=CENTER_LAT,
        center_lon=CENTER_LON,
        radius_km=ROI_RADIUS_KM,
        grid_radius_deg=GRID_RADIUS_DEG,
        grid_margin_km=0.0,
        n_ensemble=n_ensemble,
        iri_spread_kwargs=dict(IRI_SPREAD_KWARGS),
        # If a matching reference-run EDP exists, load it directly and
        # skip the expensive fresh IRI2020 ensemble build for this hour.
        # Otherwise ensemble_init follows its normal fresh-build path.
        edp_samples_path=(
            str(base_edp_samples_path)
            if base_edp_samples_path is not None
            else None
        ),
        # The hourly wrapper saves one canonical base EDP itself at
        # <run label>/<hour>/edp_samples.nc, so package_run does not need
        # to emit another condition-specific EDP file.
        edp_samples_output_path=None,
        styles=list(styles),
        hyper_params_by_style=hyper_params_by_style,
        isr_pca_basis_path=(
            str(isr_pca_basis_path) if isr_pca_basis_path is not None else None
        ),
        max_tec_per_batch=MAX_TEC_PER_BATCH,
        obs_sigma=OBS_SIGMA_TECU,
        analysis_rng_seed=ANALYSIS_RNG_SEED,
        obs_sources="RO",
        ro_kwargs=ro_kwargs,
        # Reuse the exact accepted/QC'd RO set from the precheck.
        ro_observations_path=str(ro_observations_path),
        # Run package ISR comparison only for stations that are both inside
        # the current tomography ROI and have an Ne profile in this hour.
        isr_file_path=(
            str(next(iter(active_hour_isr_files.values())))
            if active_hour_isr_files else None
        ),
        additional_isr_file_paths=(
            {
                code.lower(): str(path)
                for code, path in list(active_hour_isr_files.items())[1:]
            }
            if len(active_hour_isr_files) > 1
            else {}
        ),
        isr_range_percentile=0.0,
        diagonal_boost_amplitude=condition["diagonal_boost_amplitude"],
        diagonal_boost_log_space=condition["diagonal_boost_log_space"],
        diagonal_boost_rng_seed=condition["diagonal_boost_rng_seed"],
        diagonal_boost_taper_start_km=BOOST_TAPER_START_KM,
        diagonal_boost_taper_end_km=BOOST_TAPER_END_KM,
        diagonal_boost_taper_floor=BOOST_TAPER_FLOOR,
        inter_batch_inflation_factor=None,
    )


def decode_profiles_at_station(
    cfg: CycleConfig,
    result,
    style: str,
    lat: float,
    lon: float,
    parameterizations: dict,
):
    """
    Decode using the exact in-memory parameterization captured during
    run_package. No *_parameterized.nc file is written or reopened.
    """
    if style not in parameterizations:
        raise KeyError(f"Missing captured in-memory parameterization for {style}")

    pes = parameterizations[style]
    style_hyper = cfg.hyper_params_by_style.get(style)
    ensemble_prior = EnsembleState.from_parameterized_edp_samples(pes)
    edp_reload = result.edp_samples

    cycle_result = result.results_by_style[style][0]
    if not result.batches:
        raise RuntimeError("No batches available for decode")

    driver = GeneralEnKFDriver(style=style, hyper_params=style_hyper)
    obs_op = driver.build_observation_operator(
        edp_reload,
        pes.Parameterization,
        ensemble_prior.param_shape,
        result.batches[0].podTc2_data,
    )

    decoded_forecast = obs_op.decode(ensemble_prior.to_param_shape())
    forecast_profile = output._interpolate_field_at_latlon(
        edp_reload, decoded_forecast, lat, lon
    )
    del decoded_forecast
    gc.collect()

    decoded_analysis = obs_op.decode(cycle_result.final_ensemble.to_param_shape())
    analysis_profile = output._interpolate_field_at_latlon(
        edp_reload, decoded_analysis, lat, lon
    )
    del decoded_analysis
    gc.collect()

    return (
        np.asarray(edp_reload.altitude, dtype=float),
        np.asarray(forecast_profile, dtype=float),
        np.asarray(analysis_profile, dtype=float),
    )


def plot_dual_isr_ne(
    style: str,
    start: datetime,
    end: datetime,
    n_ro: int,
    n_tec: int,
    model_altitude: np.ndarray,
    forecast_profile: np.ndarray,
    analysis_profile: np.ndarray,
    tro,
    tro_match,
    svl,
    svl_match,
    output_path: Path,
):
    f_mean = np.nanmean(forecast_profile, axis=-1)
    f_std = np.nanstd(forecast_profile, axis=-1)
    a_mean = np.nanmean(analysis_profile, axis=-1)
    a_std = np.nanstd(analysis_profile, axis=-1)

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(f_mean, model_altitude, color="C0", label="forecast mean")
    ax.fill_betweenx(
        model_altitude, f_mean - f_std, f_mean + f_std,
        color="C0", alpha=0.20, label="forecast +/-1 std",
    )
    ax.plot(a_mean, model_altitude, color="C1", label="analysis mean")
    ax.fill_betweenx(
        model_altitude, a_mean - a_std, a_mean + a_std,
        color="C1", alpha=0.20, label="analysis +/-1 std",
    )

    if tro_match is not None:
        tro_time, tro_ne = tro_match
        ax.plot(
            tro_ne, tro.altitude,
            linestyle="--", marker=".", color="black", markersize=5,
            label=f"TRO ISR ({tro_time:%H:%M})",
            zorder=5,
        )

    if svl_match is not None:
        svl_time, svl_ne = svl_match
        ax.plot(
            svl_ne, svl.altitude,
            linestyle="--", marker=".", color="green", markersize=5,
            label=f"SVL ISR ({svl_time:%H:%M})",
            zorder=5,
        )

    ax.set_xlabel(r"Electron density (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(
        f"{style} -- analysis EDP vs. TRO/SVL ISR\n"
        f"{hour_label(start, end)} | RO profiles={n_ro} | TEC rays={n_tec}"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    save_figure(fig, output_path)


def station_error_profile(
    model_altitude: np.ndarray,
    analysis_profile: np.ndarray,
    isr,
    match,
):
    if match is None:
        return None
    isr_time, isr_ne = match
    a_mean = np.nanmean(analysis_profile, axis=-1)
    model_at_isr = interp_log10(isr.altitude, model_altitude, a_mean)
    valid = np.isfinite(model_at_isr) & np.isfinite(isr_ne)
    error = np.full_like(isr.altitude, np.nan, dtype=float)
    # With one model/ISR pair per hour, sqrt((model-ISR)^2) = absolute error.
    # Across hourly windows these gate-wise errors are aggregated below.
    error[valid] = np.abs(model_at_isr[valid] - isr_ne[valid])
    return {
        "time": isr_time,
        "altitude_km": np.asarray(isr.altitude, dtype=float),
        "error_m3": error,
        "isr_ne_m3": np.asarray(isr_ne, dtype=float),
        "model_ne_m3": model_at_isr,
    }



def plot_dual_isr_error(
    style: str,
    start: datetime,
    end: datetime,
    n_ro: int,
    n_tec: int,
    tro_error,
    svl_error,
    output_path: Path,
):
    fig, ax = plt.subplots(figsize=(7, 9))

    if tro_error is not None:
        ax.plot(
            tro_error["error_m3"], tro_error["altitude_km"],
            ".-", color="red", markersize=4, label="TRO analysis RMSE/error",
        )
    if svl_error is not None:
        ax.plot(
            svl_error["error_m3"], svl_error["altitude_km"],
            ".-", color="green", markersize=4, label="SVL analysis RMSE/error",
        )

    ax.set_xlabel(r"Analysis RMSE/error vs. ISR (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(
        f"{style} -- analysis error vs. TRO/SVL ISR\n"
        f"{hour_label(start, end)} | RO profiles={n_ro} | TEC rays={n_tec}"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    save_figure(fig, output_path)


def append_error_rows(
    rows: list[dict],
    *,
    style: str,
    condition: str,
    station_code: str,
    start: datetime,
    end: datetime,
    n_ro: int,
    n_tec: int,
    error_profile,
):
    if error_profile is None:
        return
    for altitude, error in zip(
        error_profile["altitude_km"], error_profile["error_m3"]
    ):
        rows.append({
            "style": style,
            "condition": condition,
            "station": station_code,
            "window_start_utc": start.isoformat(),
            "window_end_utc": end.isoformat(),
            "n_ro_profiles": n_ro,
            "n_tec_rays": n_tec,
            "altitude_km": float(altitude),
            "analysis_error_m3": float(error) if np.isfinite(error) else np.nan,
        })



def _boost_amplitude_at_altitude(altitude_km: float) -> float:
    """Effective boosted-condition amplitude at a requested altitude."""
    if altitude_km <= BOOST_TAPER_START_KM:
        return float(BOOST_AMPLITUDE)
    if altitude_km >= BOOST_TAPER_END_KM:
        return float(BOOST_AMPLITUDE * BOOST_TAPER_FLOOR)
    frac = (altitude_km - BOOST_TAPER_START_KM) / (
        BOOST_TAPER_END_KM - BOOST_TAPER_START_KM
    )
    taper = 1.0 - (1.0 - BOOST_TAPER_FLOOR) * frac
    return float(BOOST_AMPLITUDE * taper)


def _run_settings_row(args: argparse.Namespace) -> dict:
    """Settings intentionally tracked for each labeled experiment."""
    now_mt = datetime.now(ZoneInfo("America/Denver"))
    return {
        "label_of_each_run": LABEL_OF_EACH_RUN,
        "reference_run_label": (
            REFERENCE_RUN_LABEL
            if REFERENCE_RUN_LABEL not in (None, "")
            else ""
        ),
        "run_time_mt": now_mt.strftime("%b%d_%Y_%H:%M_%Z"),
        "center_lat": CENTER_LAT,
        "center_lon": CENTER_LON,
        "roi_radius_km": ROI_RADIUS_KM,
        "grid_radius_deg": GRID_RADIUS_DEG,
        "min_ro_profiles": MIN_RO_PROFILES,
        "n_ensemble": int(args.n_ensemble),
        "obs_sigma_tecu": OBS_SIGMA_TECU,
        "iri_downsampling_factor": IRI_DOWNSAMPLING_FACTOR,
        "ro_area_fraction_required": RO_AREA_FRACTION_REQUIRED,
        "pca_discard_scale": PCA_DISCARD_SCALE,
        "styles": "|".join(DEFAULT_STYLES),
        "boost_le_400_km": round(_boost_amplitude_at_altitude(400.0), 2),
        "boost_500_km": round(_boost_amplitude_at_altitude(500.0), 2),
        "boost_600_km": round(_boost_amplitude_at_altitude(600.0), 2),
        "boost_ge_700_km": round(_boost_amplitude_at_altitude(700.0), 2),
        "roi_mode": RO_KWARGS_TEMPLATE["roi_mode"],
        "alt_limit_km": RO_KWARGS_TEMPLATE["alt_limit_km"],
        "fraction_required": RO_KWARGS_TEMPLATE["fraction_required"],
        "min_valid_rays": RO_KWARGS_TEMPLATE["min_valid_rays"],
        "max_rays_per_occultation": RO_KWARGS_TEMPLATE["max_rays_per_occultation"],
        "max_lowest_tangent_alt_km": RO_KWARGS_TEMPLATE["max_lowest_tangent_alt_km"],
        "require_dcb_flags": RO_KWARGS_TEMPLATE["require_dcb_flags"],
        "duplicate_time_tolerance_s": RO_KWARGS_TEMPLATE["duplicate_time_tolerance_s"],
        "run_message": RUN_MESSAGE,
    }


def _parse_run_time_mt(value) -> datetime | None:
    """Parse run_time_mt written by _run_settings_row."""
    try:
        s = str(value)
        base, _tz = s.rsplit("_", 1)
        naive = datetime.strptime(base, "%b%d_%Y_%H:%M")
        return naive.replace(tzinfo=ZoneInfo("America/Denver"))
    except Exception:
        return None


def _same_setting_value(a, b) -> bool:
    """Stable comparison for CSV-loaded setting values, including None/NaN."""
    if pd.isna(a) and pd.isna(b):
        return True
    if a is None and (b is None or pd.isna(b)):
        return True
    if b is None and (a is None or pd.isna(a)):
        return True
    try:
        af = float(a)
        bf = float(b)
        if np.isfinite(af) and np.isfinite(bf):
            return bool(np.isclose(af, bf, rtol=0.0, atol=1e-12))
    except Exception:
        pass
    return str(a) == str(b)


def update_each_run_setting_csv(args: argparse.Namespace) -> Path:
    """
    Append OUTPUT_ROOT/each_run_setting.csv without deleting older runs.

    A launcher is treated as the SAME experiment only when ALL are true:
      1. label_of_each_run is identical;
      2. every tracked setting is identical;
      3. launch time is within 12 hours of an existing record.

    In that case the existing row is kept unchanged. A different label,
    changed settings, or a >12-hour gap creates a new row.

    This script is intentionally serial and does not implement parallel-job
    coordination, file locks, temporary CSV files, or atomic replacement.
    """
    path = OUTPUT_ROOT / "each_run_setting.csv"
    row = _run_settings_row(args)

    setting_columns = [
        key for key in row
        if key not in {"label_of_each_run", "run_time_mt", "run_message"}
    ]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    now_dt = _parse_run_time_mt(row["run_time_mt"])
    matched_index = None

    if not df.empty and now_dt is not None:
        for idx, old in df.iterrows():
            # Different labels are different experiments.
            if str(old.get("label_of_each_run", "")) != str(
                row["label_of_each_run"]
            ):
                continue

            old_dt = _parse_run_time_mt(old.get("run_time_mt", ""))
            if old_dt is None:
                continue

            if abs((now_dt - old_dt).total_seconds()) > 12 * 3600:
                continue

            same_settings = all(
                key in df.columns
                and _same_setting_value(old.get(key), row[key])
                for key in setting_columns
            )
            if same_settings:
                matched_index = idx
                break

    if matched_index is None:
        # New experiment record. Preserve every older row.
        df = pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True,
            sort=False,
        )
    else:
        # Same label + same settings + within 12 h = same run.
        # Keep the original record exactly as it was.
        pass

    ordered = list(row.keys()) + [
        c for c in df.columns if c not in row
    ]
    df = df[ordered]
    df.to_csv(path, index=False)
    return path


def save_daily_ro_status(rows: list[dict], day: datetime, run_output_root: Path) -> Path:
    """
    Upsert hourly RO/ISR status rows into the persistent daily CSV.

    The file keeps one complete row per requested hour. Updating one hour
    replaces only that hour and preserves every other existing hour.

    This script is intentionally serial: no file locks, temporary files, or
    parallel-job coordination are used here.
    """
    path = run_output_root / f"{day:%Y-%m-%d}_ro_hourly_status.csv"

    new_df = pd.DataFrame(rows)
    if new_df.empty:
        return path

    if "window_start_utc" not in new_df.columns:
        raise ValueError("RO hourly status rows require window_start_utc")

    # If the same hour appears more than once in this update, newest wins.
    new_df = new_df.drop_duplicates(
        subset=["window_start_utc"],
        keep="last",
    )

    run_output_root.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            old_df = pd.read_csv(path)
        except Exception:
            old_df = pd.DataFrame()
    else:
        old_df = pd.DataFrame()

    if not old_df.empty and "window_start_utc" in old_df.columns:
        replace_hours = set(new_df["window_start_utc"].astype(str))
        old_df = old_df[
            ~old_df["window_start_utc"].astype(str).isin(replace_hours)
        ]

    combined = pd.concat(
        [old_df, new_df],
        ignore_index=True,
        sort=False,
    )

    combined = combined.drop_duplicates(
        subset=["window_start_utc"],
        keep="last",
    )

    combined = combined.sort_values(
        "window_start_utc",
        kind="stable",
    ).reset_index(drop=True)

    combined.to_csv(path, index=False)
    return path


def successful_summary_exists(hour_dir: Path) -> bool:
    path = hour_dir / "hourly_summary.csv"
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        if df.empty or "condition" not in df.columns:
            return False
        ok = df[df["status"] == "success"]
        required = {item["name"] for item in RUN_CONDITIONS}
        return required.issubset(set(ok["condition"].astype(str)))
    except Exception:
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    day = validate_args(args)

    # ISR validation files are loaded through the registry below.
    # Missing station files do not stop the run unless no available ISR lies
    # inside the selected ROI.
    ro_dir = ro_day_dir(day)
    if not ro_dir.is_dir():
        raise FileNotFoundError(
            f"RO day directory not found: {ro_dir}\n"
            f"For {day:%Y-%m-%d}, expected YYYY.DDD = {ro_dir.name}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT_ROOT = OUTPUT_ROOT / LABEL_OF_EACH_RUN
    RUN_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if REFERENCE_RUN_LABEL not in (None, ""):
        reference_root = OUTPUT_ROOT / str(REFERENCE_RUN_LABEL)
        if not reference_root.is_dir():
            print(
                "WARNING: reference run folder does not currently exist:",
                reference_root,
            )
            print(
                "         Matching hours will fall back to fresh IRI2020."
            )

    setting_csv_path = update_each_run_setting_csv(args)
    print("Run label            :", LABEL_OF_EACH_RUN)
    print(
        "Reference run label  :",
        REFERENCE_RUN_LABEL
        if REFERENCE_RUN_LABEL not in (None, "")
        else "DISABLED",
    )
    print("Labeled output root  :", RUN_OUTPUT_ROOT)
    print("Run settings registry:", setting_csv_path)
    print(f"ROI radius            : {ROI_RADIUS_KM:g} km")
    print(f"Grid radius           : {GRID_RADIUS_DEG:g} deg")
    print(f"Minimum accepted RO   : {MIN_RO_PROFILES}")

    isr_pca_basis_path = ensure_tro_isr_pca_basis(
        args.isr_pca_basis,
        args.styles,
        rebuild_basis=args.rebuild_tro_pca_basis,
    )
    hyper_params_by_style = build_hyper_params_by_style(
        args.styles, isr_pca_basis_path
    )

    if isr_pca_basis_path is not None:
        print("ISR-derived PCA basis:", isr_pca_basis_path)
        print("PCA training station : TRO only")
        print("SVL role             : validation only")

    print("Styles to run:")
    for style in args.styles:
        print(" ", style)
    print("Conditions:")
    for condition in RUN_CONDITIONS:
        print(
            f"  {condition['name']}: "
            f"amplitude={condition['diagonal_boost_amplitude']}, "
            f"log_space={condition['diagonal_boost_log_space']}"
        )
    print(f"Total cases per successful hour: {len(args.styles) * len(RUN_CONDITIONS)}")
    print(f"Output policy: {args.output}")
    if args.output == "minimal":
        print(
            "  Suppressed: apf107.dat, ig_rz.dat, *_iri_sample_inputs.pkl, "
            "*_edp_samples.nc, *_parameterized.nc, *_ensembles/ensemble_batch_*.nc"
        )
    else:
        print("  All package artifacts enabled.")

    isr_datasets, active_isr_datasets = load_isr_registry()
    print("Active ISR validation station(s):", ", ".join(active_isr_datasets))

    hourly_rows: list[dict] = []
    mode_timing_rows: list[dict] = []
    daily_ro_status_rows: list[dict] = []
    total_t0 = time.perf_counter()

    for hour in range(args.start_hour, args.end_hour):
        start = day + timedelta(hours=hour)
        end = start + timedelta(hours=1)
        folder = hour_folder_name(start, end)
        hour_dir = RUN_OUTPUT_ROOT / folder

        print("\n" + "=" * 96)
        print(f"HOUR: {hour_label(start, end)}")
        print(
            f"OUTPUT (only if RO >= {MIN_RO_PROFILES} "
            f"and ISR exists): {hour_dir}"
        )
        print("=" * 96)

        if args.skip_existing and successful_summary_exists(hour_dir):
            print("SKIP: both no_boost and boosted already successful")
            existing_summary = pd.read_csv(hour_dir / "hourly_summary.csv")
            for row in existing_summary.to_dict("records"):
                row["status"] = "skipped_existing"
                hourly_rows.append(row)
            continue

        hour_t0 = time.perf_counter()
        hour_error_rows: list[dict] = []
        hour_summary_rows: list[dict] = []
        status_row = None

        try:
            # Precheck in a temporary directory so skipped hours leave no folder/files.
            with tempfile.TemporaryDirectory(prefix="ro_precheck_") as pre_tmp:
                precomputed_ro_tmp = Path(pre_tmp) / "accepted_ro_observations.nc"
                pre_cfg = make_precheck_cfg(start, end, ro_dir, precomputed_ro_tmp)
                pre_batches = observation_stream.assemble(pre_cfg)
                ro_entries = unique_ro_entries(pre_batches)
                n_ro = len(ro_entries)
                n_tec = count_tec_rays(pre_batches)

                print(f"Accepted RO occultation profiles: {n_ro}")
                print(f"Accepted TEC rays:                {n_tec}")

                # Always count/QC RO first so the daily status CSV records RO
                # availability for every requested hour, even if no ISR exists.
                skipped_for_ro = (n_ro < MIN_RO_PROFILES or n_tec == 0)

                # Match only ISR stations that lie inside the current ROI.
                isr_matches = {
                    code: nearest_profile_in_hour(ds, start, end)
                    for code, ds in active_isr_datasets.items()
                }
                active_hour_isr = {
                    code: active_isr_datasets[code]
                    for code, match in isr_matches.items()
                    if match is not None
                }
                active_hour_isr_files = {
                    code: ISR_FILES[code]
                    for code in active_hour_isr
                }

                n_isr_stations = len(active_hour_isr)
                skipped_for_isr = (n_isr_stations == 0)

                print(
                    "ISR available in ROI this hour: "
                    + ", ".join(
                        f"{code}={'yes' if isr_matches[code] is not None else 'no'}"
                        for code in active_isr_datasets
                    )
                )

                # Rejection precedence follows the hourly processing order:
                # first insufficient RO, then no ISR.
                if skipped_for_ro:
                    status = "skipped_too_few_ro"
                    rejection_reason = "insufficient_ro"
                elif skipped_for_isr:
                    status = "skipped_no_isr"
                    rejection_reason = "no_isr"
                else:
                    status = "accepted_for_assimilation"
                    rejection_reason = ""

                skipped_hour = skipped_for_ro or skipped_for_isr

                status_row = {
                    "window_start_utc": start.isoformat(),
                    "window_end_utc": end.isoformat(),
                    "tro_in_roi": bool("TRO" in active_isr_datasets),
                    "svl_in_roi": bool("SVL" in active_isr_datasets),
                    "mil_in_roi": bool("MIL" in active_isr_datasets),
                    "tro_isr_available": bool(
                        "TRO" in isr_matches and isr_matches["TRO"] is not None
                    ),
                    "svl_isr_available": bool(
                        "SVL" in isr_matches and isr_matches["SVL"] is not None
                    ),
                    "mil_isr_available": bool(
                        "MIL" in isr_matches and isr_matches["MIL"] is not None
                    ),
                    "n_isr_stations": int(n_isr_stations),
                    "n_ro_profiles": int(n_ro),
                    "n_tec_rays": int(n_tec),
                    "skipped": bool(skipped_hour),

                    # Eligibility status is permanent:
                    # accepted_for_assimilation / skipped_too_few_ro / skipped_no_isr.
                    "status": status,
                    "rejection_reason": rejection_reason,

                    # Tomography execution status is tracked separately so an
                    # accepted hour never loses its accepted status.
                    "tomography_status": (
                        "not_run" if skipped_hour else "running"
                    ),
                    "tomography_message": "",
                }

                daily_ro_status_rows.append(status_row)

                # Save ONLY this current hour. The function merges it with the
                # complete latest daily CSV and preserves every other hour.
                daily_status_path = save_daily_ro_status(
                    [status_row], day, RUN_OUTPUT_ROOT
                )

                if skipped_for_ro:
                    print(
                        f"SKIP: only {n_ro} accepted RO profiles "
                        f"(minimum required = {MIN_RO_PROFILES}); "
                        "no hourly folder/files created."
                    )
                    print(f"Daily RO/ISR status updated: {daily_status_path}")
                    continue

                if skipped_for_isr:
                    print(
                        "SKIP: RO requirement passed, but no ISR station inside "
                        "the current ROI has an Ne profile in this 1-hour window; "
                        "no hourly tomography folder/files created."
                    )
                    print(f"Daily RO/ISR status updated: {daily_status_path}")
                    continue

                hour_dir.mkdir(parents=True, exist_ok=True)
                precomputed_ro = hour_dir / "accepted_ro_observations.nc"
                shutil.copy2(precomputed_ro_tmp, precomputed_ro)

            # -------------------------------------------------------------
            # 2. Resolve the base IRI ensemble for this hour.
            #
            #    Preferred:
            #      reuse matching <REFERENCE_RUN_LABEL>/<hour>/edp_samples.nc
            #
            #    Fallback:
            #      fresh IRI2020 ensemble build.
            #
            #    Both no_boost and boosted then use the exact same base
            #    EDPSamples object for a controlled A/B comparison.
            # -------------------------------------------------------------
            reference_edp_path = find_reference_edp_samples(start, end)

            if reference_edp_path is not None:
                print("Reference EDP cache HIT:")
                print(" ", reference_edp_path)
                initial_edp_source = "reference_run"
            else:
                if REFERENCE_RUN_LABEL not in (None, ""):
                    print(
                        "Reference EDP cache MISS for "
                        f"{hour_label(start, end)} under label "
                        f"{REFERENCE_RUN_LABEL!r}; building fresh IRI2020."
                    )
                else:
                    print("Reference EDP reuse disabled; building fresh IRI2020.")
                initial_edp_source = "fresh_iri2020"

            base_edp_samples = None
            canonical_edp_path = canonical_hour_edp_path(
                LABEL_OF_EACH_RUN, start, end
            )

            for condition in RUN_CONDITIONS:
                condition_name = condition["name"]
                condition_label = condition["label"]
                condition_dir = hour_dir / condition_name
                condition_dir.mkdir(parents=True, exist_ok=True)
                label = (
                    f"{CENTER_SITE.lower()}_allstyles_{condition_name}_"
                    f"{start:%Y%m%d_%H%M}_{end:%H%M}"
                )

                cfg = make_run_cfg(
                    start,
                    end,
                    ro_dir,
                    precomputed_ro,
                    condition_dir,
                    label,
                    args.styles,
                    args.n_ensemble,
                    hyper_params_by_style,
                    isr_pca_basis_path,
                    condition,
                    active_hour_isr_files,
                    base_edp_samples_path=(
                        reference_edp_path
                        if condition_name == "no_boost"
                        else None
                    ),
                )

                print("\n" + "-" * 96)
                print(f"CONDITION: {condition_name}")
                print("Styles:", ", ".join(cfg.styles))
                if condition_name == "no_boost":
                    if reference_edp_path is not None:
                        print(
                            "IRI ensemble: load matching reference-run "
                            "edp_samples.nc (skip fresh IRI2020 build)"
                        )
                    else:
                        print("IRI ensemble: fresh hourly IRI2020 build")
                else:
                    print("IRI ensemble: reuse exact no-boost hourly ensemble")
                    print(
                        f"Boost: nominal amplitude={BOOST_AMPLITUDE}, "
                        "log10-density space, seed=888; "
                        f"altitude taper <={BOOST_TAPER_START_KM:.0f} km: "
                        f"{BOOST_AMPLITUDE:.2f}, "
                        f"500 km: {BOOST_AMPLITUDE * (1.0 - 0.9 * (500.0 - BOOST_TAPER_START_KM) / (BOOST_TAPER_END_KM - BOOST_TAPER_START_KM)):.2f}, "
                        f"600 km: {BOOST_AMPLITUDE * (1.0 - 0.9 * (600.0 - BOOST_TAPER_START_KM) / (BOOST_TAPER_END_KM - BOOST_TAPER_START_KM)):.2f}, "
                        f">={BOOST_TAPER_END_KM:.0f} km: "
                        f"{BOOST_AMPLITUDE * BOOST_TAPER_FLOOR:.2f}"
                    )
                    if base_edp_samples is None:
                        raise RuntimeError(
                            "Boosted run requested before no-boost in-memory EDP exists"
                        )

                condition_t0 = time.perf_counter()
                with _NoArtifactRunContext(
                    base_edp_samples=(
                        base_edp_samples if condition_name == "boosted" else None
                    ),
                    output_mode=args.output,
                ) as no_artifacts:
                    result = run_package(cfg)
                    in_memory_parameterizations = dict(no_artifacts.parameterizations)

                condition_elapsed = time.perf_counter() - condition_t0

                n_ro_used = len(unique_ro_entries(result.batches))
                n_tec_used = count_tec_rays(result.batches)
                print(
                    f"{condition_name} used RO profiles={n_ro_used}, "
                    f"TEC rays={n_tec_used}"
                )

                # package_run stores the actual run_cycle wall time separately
                # for each parameterization style. Report/store all 8 modes:
                # 4 styles x (no_boost, boosted).
                print("Per-mode assimilation timing:")
                for style in cfg.styles:
                    _cycle_result, n_state, style_wall_time_s = result.results_by_style[style]
                    print(
                        f"  {condition_name:9s} | {style:16s} | "
                        f"{style_wall_time_s:9.2f} s "
                        f"({style_wall_time_s / 60.0:6.2f} min) | "
                        f"n_state={n_state}"
                    )
                    mode_timing_rows.append({
                        "window_start_utc": start.isoformat(),
                        "window_end_utc": end.isoformat(),
                        "condition": condition_name,
                        "style": style,
                        "n_state": int(n_state),
                        "n_ro_profiles": int(n_ro_used),
                        "n_tec_rays": int(n_tec_used),
                        "assimilation_wall_time_seconds": float(style_wall_time_s),
                        "assimilation_wall_time_minutes": float(style_wall_time_s) / 60.0,
                    })

                if condition_name == "no_boost":
                    base_edp_samples = result.edp_samples

                    # Always keep one canonical unboosted base EDP for this
                    # label/hour so this run can itself serve as the reference
                    # for future tuning experiments.
                    canonical_edp_path = save_canonical_hour_edp(
                        base_edp_samples,
                        LABEL_OF_EACH_RUN,
                        start,
                        end,
                    )

                    print("Base EDP retained in memory for boosted A/B run")
                    print("Canonical hourly EDP cache:", canonical_edp_path)

                # ---------------------------------------------------------
                # 3. Extra location-matched validation for every active ISR
                #    station inside the ROI.
                # ---------------------------------------------------------
                for style in cfg.styles:
                    display = (
                        f"{STYLE_DISPLAY.get(style, style)} "
                        f"[{condition_label}]"
                    )
                    style_dir = condition_dir / f"{label}_{style}"

                    for code, station in active_hour_isr.items():
                        match = isr_matches[code]
                        print(f"Creating {code} validation:", display)

                        model_altitude, f_prof, a_prof = decode_profiles_at_station(
                            cfg,
                            result,
                            style,
                            station.station.lat,
                            station.station.lon,
                            in_memory_parameterizations,
                        )

                        fm = np.nanmean(f_prof, axis=-1)
                        fs = np.nanstd(f_prof, axis=-1)
                        am = np.nanmean(a_prof, axis=-1)
                        ass = np.nanstd(a_prof, axis=-1)

                        fig, ax = plt.subplots(figsize=(7, 9))
                        ax.plot(fm, model_altitude, color="C0", label="forecast mean")
                        ax.fill_betweenx(
                            model_altitude, fm - fs, fm + fs,
                            color="C0", alpha=0.2
                        )
                        ax.plot(am, model_altitude, color="C1", label="analysis mean")
                        ax.fill_betweenx(
                            model_altitude, am - ass, am + ass,
                            color="C1", alpha=0.2
                        )

                        if match is not None:
                            mt, mne = match
                            ax.plot(
                                mne,
                                station.altitude,
                                linestyle="--",
                                marker=".",
                                color="black",
                                markersize=5,
                                label=f"{code} ISR ({mt:%H:%M})",
                            )

                        ax.set_xlabel(r"Electron density (m$^{-3}$)")
                        ax.set_ylabel("Altitude (km)")
                        ax.set_title(
                            f"{display} -- model at {code} ISR location\n"
                            f"{hour_label(start, end)} | "
                            f"RO profiles={n_ro_used} | TEC rays={n_tec_used}"
                        )
                        ax.grid(True, alpha=0.2)
                        ax.legend(loc="best", fontsize=8)
                        save_figure(
                            fig,
                            style_dir / f"{code.lower()}_isr_ne_comparison.png",
                        )

                        error_profile = station_error_profile(
                            model_altitude, a_prof, station, match
                        )

                        fig, ax = plt.subplots(figsize=(7, 9))
                        if error_profile is not None:
                            ax.plot(
                                error_profile["error_m3"],
                                error_profile["altitude_km"],
                                ".-",
                                color="red",
                                markersize=4,
                                label=f"{code} analysis RMSE/error",
                            )
                        ax.set_xlabel(
                            r"Analysis RMSE/error vs. ISR (m$^{-3}$)"
                        )
                        ax.set_ylabel("Altitude (km)")
                        ax.set_title(
                            f"{display} -- analysis error vs. {code} ISR\n"
                            f"{hour_label(start, end)} | "
                            f"RO profiles={n_ro_used} | TEC rays={n_tec_used}"
                        )
                        ax.grid(True, alpha=0.25)
                        ax.legend(loc="best", fontsize=8)
                        save_figure(
                            fig,
                            style_dir / f"{code.lower()}_isr_rmse_by_altitude.png",
                        )

                        append_error_rows(
                            hour_error_rows,
                            style=style,
                            condition=condition_name,
                            station_code=code,
                            start=start,
                            end=end,
                            n_ro=n_ro_used,
                            n_tec=n_tec_used,
                            error_profile=error_profile,
                        )

                        del f_prof, a_prof
                        gc.collect()

                condition_row = {
                    "window_start_utc": start.isoformat(),
                    "window_end_utc": end.isoformat(),
                    "condition": condition_name,
                    "status": "success",
                    "n_styles": len(cfg.styles),
                    "n_ro_profiles": n_ro_used,
                    "n_tec_rays": n_tec_used,
                    "elapsed_seconds": condition_elapsed,
                    "output_dir": str(condition_dir),
                    "initial_edp_source": initial_edp_source,
                    "reference_run_label": (
                        REFERENCE_RUN_LABEL
                        if reference_edp_path is not None
                        else ""
                    ),
                    "reference_edp_samples_path": (
                        str(reference_edp_path)
                        if reference_edp_path is not None
                        else ""
                    ),
                    "base_edp_samples_path": str(canonical_edp_path),
                    "boost_amplitude": condition[
                        "diagonal_boost_amplitude"
                    ],
                    "boost_nominal_amplitude": BOOST_AMPLITUDE,
                    "boost_log_space": condition[
                        "diagonal_boost_log_space"
                    ],
                    "boost_rng_seed": condition[
                        "diagonal_boost_rng_seed"
                    ],
                    "boost_taper_start_km": BOOST_TAPER_START_KM,
                    "boost_taper_end_km": BOOST_TAPER_END_KM,
                    "boost_taper_floor": BOOST_TAPER_FLOOR,
                    "message": "",
                }
                hour_summary_rows.append(condition_row)
                hourly_rows.append(condition_row)

                # Checkpoint after each condition because ANCHOR can be slow.
                pd.DataFrame(hour_summary_rows).to_csv(
                    hour_dir / "hourly_summary.csv", index=False
                )
                pd.DataFrame(hour_error_rows).to_csv(
                    hour_dir / "dual_isr_errors.csv", index=False
                )
                # Daily aggregate run/error CSVs intentionally disabled.
                # Per-hour hourly_summary.csv and dual_isr_errors.csv are retained.

            # Both conditions completed. Keep the permanent eligibility
            # status as accepted_for_assimilation and record execution outcome
            # separately.
            status_row["tomography_status"] = "success"
            status_row["tomography_message"] = ""
            save_daily_ro_status([status_row], day, RUN_OUTPUT_ROOT)

            hour_mode_timings = [
                row for row in mode_timing_rows
                if row["window_start_utc"] == start.isoformat()
            ]
            if hour_mode_timings:
                pd.DataFrame(hour_mode_timings).to_csv(
                    hour_dir / "mode_timing.csv", index=False
                )

            elapsed = time.perf_counter() - hour_t0
            print(f"Hour complete in {elapsed / 60.0:.1f} min")

        except Exception as exc:
            elapsed = time.perf_counter() - hour_t0

            if status_row is None:
                # Failure occurred before the normal RO/ISR status row could be
                # constructed. Record that explicitly instead of silently
                # losing the requested hour.
                status_row = {
                    "window_start_utc": start.isoformat(),
                    "window_end_utc": end.isoformat(),
                    "tro_in_roi": bool(
                        "TRO" in locals().get("active_isr_datasets", {})
                    ),
                    "svl_in_roi": bool(
                        "SVL" in locals().get("active_isr_datasets", {})
                    ),
                    "mil_in_roi": bool(
                        "MIL" in locals().get("active_isr_datasets", {})
                    ),
                    "tro_isr_available": bool(
                        locals().get("isr_matches", {}).get("TRO") is not None
                    ),
                    "svl_isr_available": bool(
                        locals().get("isr_matches", {}).get("SVL") is not None
                    ),
                    "mil_isr_available": bool(
                        locals().get("isr_matches", {}).get("MIL") is not None
                    ),
                    "n_isr_stations": int(
                        locals().get("n_isr_stations", 0)
                    ),
                    "n_ro_profiles": locals().get("n_ro", np.nan),
                    "n_tec_rays": locals().get("n_tec", np.nan),
                    "skipped": False,
                    "status": "precheck_failed",
                    "rejection_reason": "precheck_failure",
                    "tomography_status": "failed",
                    "tomography_message": f"{type(exc).__name__}: {exc}",
                }
            else:
                # Eligibility was already determined. Do not change accepted
                # RO/ISR status simply because the tomography execution failed.
                status_row["tomography_status"] = "failed"
                status_row["tomography_message"] = (
                    f"{type(exc).__name__}: {exc}"
                )

            save_daily_ro_status([status_row], day, RUN_OUTPUT_ROOT)
            print(
                f"FAILED hour {hour_label(start, end)}: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            row = {
                "window_start_utc": start.isoformat(),
                "window_end_utc": end.isoformat(),
                "condition": locals().get("condition_name", "unknown"),
                "status": "failed",
                "n_ro_profiles": locals().get("n_ro", np.nan),
                "n_tec_rays": locals().get("n_tec", np.nan),
                "elapsed_seconds": elapsed,
                "output_dir": str(hour_dir),
                "message": f"{type(exc).__name__}: {exc}",
            }
            hourly_rows.append(row)
            hour_summary_rows.append(row)
            pd.DataFrame(hour_summary_rows).to_csv(
                hour_dir / "hourly_summary.csv", index=False
            )
            if not args.continue_on_error:
                raise

    # Per-hour validation products are the only ISR error outputs; RO-bin aggregate
    # comparison products and their helper code have been removed.

    if mode_timing_rows:
        pd.DataFrame(mode_timing_rows).to_csv(
            RUN_OUTPUT_ROOT / f"{args.date}_mode_timing.csv", index=False
        )

    total_elapsed = time.perf_counter() - total_t0
    hh = int(total_elapsed // 3600)
    mm = int((total_elapsed % 3600) // 60)
    ss = total_elapsed % 60

    print("\n" + "=" * 96)
    print("ALL REQUESTED HOURS COMPLETE")
    print("=" * 96)
    print(
        f"Total elapsed: {hh:02d}:{mm:02d}:{ss:05.2f} (hh:mm:ss)"
    )
    print("Output root:", RUN_OUTPUT_ROOT)
    print("Run settings registry:", setting_csv_path)


if __name__ == "__main__":
    main()
