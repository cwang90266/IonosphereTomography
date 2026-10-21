#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run one-hour RO-only assimilation windows over a requested UTC interval,
with a fresh IRI2020 ensemble for every hour, preserving run_package's
normal outputs and adding dual-ISR (Tromso + Svalbard) validation products.

Main additions per successful one-hour slot, per style:
  1) dual_isr_ne_comparison.png
       forecast mean +/-1 std, analysis mean +/-1 std,
       nearest Tromso ISR profile, nearest Svalbard ISR profile,
       title includes number of RO occultation profiles and TEC rays used.
  2) dual_isr_rmse_by_altitude.png
       analysis error/RMSE-by-native-altitude-gate against Tromso (red)
       and Svalbard (green). No extrapolation beyond each ISR's native gates.

RO-count-bin comparison summaries are currently disabled; no aggregate\nRO-bin comparison CSVs or figures are generated.\n\nHours with fewer than 5 accepted RO occultation profiles are skipped.
RO-count bins for assimilated hours are non-overlapping:
    5, 6-10, 11-15, 16-20, >20

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

The default run compares ALL FOUR parameterization styles:
  - ANCHOR
  - PCA_3D_10ex
  - PCA_1D_10ex
  - PCA_1D_10ex_ISR


and, for every style, BOTH covariance conditions:
  - no_boost
  - boosted (diagonal boost amplitude=0.5 in log10-density space)

Therefore every successful one-hour slot runs 8 filter cases total.

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Repository / fixed project paths
# =============================================================================

REPO = Path("/home/pin/Desktop/IonosphereTomography-pin_dev")  # ===Edit here===
# Four "Edit here" in total, no trailing "/"

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


RO_ROOT = Path("/home/pin/Desktop/tomography_project/piq_data/podTc2")  # ===Edit here===
ISR_DIR = Path("/home/pin/Desktop/ISR_PCA/PCA_data")  # ===Edit here===
TRO_ISR_FILE = ISR_DIR / "TROISR2025_nonan.nc"
SVL_ISR_FILE = ISR_DIR / "SVLISR2025_nonan.nc"

# PCA_1D_10ex_ISR training artifacts.
# ONLY Tromso feeds these files. Svalbard is validation-only.
TRO_ISR_EXTENDED_FILE = ISR_DIR / "TROISR2025_nonan_extended.nc"
TRO_ISR_PCA_BASIS_FILE = ISR_DIR / "TROISR2025_nonan_pca_basis.nc"

OUTPUT_ROOT = Path("/home/pin/Desktop/tomography_output")  # ===Edit here===

DEFAULT_STYLES = [
    "ANCHOR",
    "PCA_3D_10ex",
    "PCA_1D_10ex",
    "PCA_1D_10ex_ISR",
]

STYLE_DISPLAY = {
    "ANCHOR": "ANCHOR",
    "PCA_3D_10ex": "IRI-trained PCA_3D_10ex",
    "PCA_1D_10ex": "IRI-trained PCA_1D_10ex",
    "PCA_1D_10ex_ISR": "TRO-ISR-trained PCA_1D_10ex_ISR",
}

DEFAULT_HYPER_PARAMS = {
    "PCA_3D_10ex": {"retaining_threshold": 0.9999},
    "PCA_1D_10ex": {"retaining_threshold": 0.99999},
}

BOOST_AMPLITUDE = 0.5

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

# Keep the same Tromso-centered tomography region used in the current run_test.
CENTER_LAT = 69.6
CENTER_LON = 19.2
ROI_RADIUS_KM = 2000.0
GRID_RADIUS_DEG = 18.0

N_ENSEMBLE = 2000
MAX_TEC_PER_BATCH = 100_000
OBS_SIGMA_TECU = 5.0
ANALYSIS_RNG_SEED = 777

# Diagonal covariance boost configuration.

# Altitude taper:
#   <= 400 km : 0.50
#      500 km : 0.35
#      600 km : 0.20
#   >= 700 km : 0.05
#
# Between 400 and 700 km the effective amplitude decreases linearly.
BOOST_TAPER_START_KM = 400.0
BOOST_TAPER_END_KM = 700.0
BOOST_TAPER_FLOOR = 0.1

IRI_SPREAD_KWARGS = {
    "hour_sample_range": 3,
    "f107_sample_range": 30,
    "ap_sample_range": 30,
    "ig_sample_range": 12,
    "rz_sample_range": 12,
}

RO_KWARGS_TEMPLATE = {
    # Updated develop behavior: retain individual RO rays only when at least
    # 65% of the LOS below that occultation's mean LEO altitude lies inside
    # the ROI.  alt_limit_km=None tells ro_source.py to derive the mean LEO
    # altitude separately for each occultation.
    "roi_mode": "full_los",
    "alt_limit_km": None,
    "fraction_required": 0.65,
    "min_valid_rays": 50,
    "max_rays_per_occultation": 200,
    "include_abel": True,
}

RO_COUNT_BINS = [
    ("05_05", "5", 5, 5),
    ("06_10", "6-10", 6, 10),
    ("11_15", "11-15", 11, 15),
    ("16_20", "16-20", 16, 20),
    ("gt20", ">20", 21, None),
]


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
        help="Parameterization styles. Default: ANCHOR PCA_3D_10ex PCA_1D_10ex PCA_1D_10ex_ISR",
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
            retaining_threshold=1.0 - 1e-4,
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
        # Hourly wrapper keeps the no-boost base EDP in memory; no EDP netCDF.
        edp_samples_path=None,
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
        # Keep original package ISR plots using Tromso, exactly as before.
        isr_file_path=str(TRO_ISR_FILE),
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


def ro_bin_mask(series: pd.Series, low: int, high: int | None) -> pd.Series:
    if high is None:
        return series >= low
    return (series >= low) & (series <= high)


def aggregate_station(group: pd.DataFrame):
    if group.empty:
        return pd.DataFrame(columns=["altitude_km", "median", "std", "n"])
    out = (
        group.groupby("altitude_km", sort=True)["analysis_error_m3"]
        .agg(
            median="median",
            std=lambda x: float(np.nanstd(np.asarray(x, dtype=float), ddof=1))
            if np.isfinite(np.asarray(x, dtype=float)).sum() > 1 else 0.0,
            n=lambda x: int(np.isfinite(np.asarray(x, dtype=float)).sum()),
        )
        .reset_index()
    )
    return out


def plot_ro_count_summary(
    df: pd.DataFrame,
    style: str,
    condition: str,
    bin_tag: str,
    bin_label: str,
    low: int,
    high: int | None,
    output_root: Path,
):
    sub = df[
        (df["style"] == style)
        & (df["condition"] == condition)
        & ro_bin_mask(df["n_ro_profiles"], low, high)
    ].copy()
    if sub.empty:
        return None

    slot_ids = sub["window_start_utc"].drop_duplicates()
    n_slots = len(slot_ids)

    tro_df = aggregate_station(sub[sub["station"] == "TRO"])
    svl_df = aggregate_station(sub[sub["station"] == "SVL"])
    n_tro_slots = sub.loc[sub["station"] == "TRO", "window_start_utc"].nunique()
    n_svl_slots = sub.loc[sub["station"] == "SVL", "window_start_utc"].nunique()

    fig, ax = plt.subplots(figsize=(7, 9))

    if not tro_df.empty:
        x = tro_df["median"].to_numpy(dtype=float)
        s = tro_df["std"].to_numpy(dtype=float)
        y = tro_df["altitude_km"].to_numpy(dtype=float)
        ax.plot(x, y, ".-", color="red", label=f"TRO median (n={n_tro_slots} slots)")
        ax.fill_betweenx(y, np.maximum(x - s, 0.0), x + s, color="red", alpha=0.16)

    if not svl_df.empty:
        x = svl_df["median"].to_numpy(dtype=float)
        s = svl_df["std"].to_numpy(dtype=float)
        y = svl_df["altitude_km"].to_numpy(dtype=float)
        ax.plot(x, y, ".-", color="green", label=f"SVL median (n={n_svl_slots} slots)")
        ax.fill_betweenx(y, np.maximum(x - s, 0.0), x + s, color="green", alpha=0.16)

    ax.set_xlabel(r"Median analysis RMSE/error vs. ISR (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(
        f"{STYLE_DISPLAY.get(style, style)} [{condition}] -- median Ne RMSE/error vs. altitude\n"
        f"RO profiles per 1-hour slot: {bin_label} | hourly slots used={n_slots} | shading=+/-1 sigma"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)

    path = output_root / f"summary_{condition}_{style}_RO_{bin_tag}_median_rmse.png"
    save_figure(fig, path)

    # Also save the summarized gate values used in the plot.
    pieces = []
    if not tro_df.empty:
        x = tro_df.copy()
        x.insert(0, "station", "TRO")
        pieces.append(x)
    if not svl_df.empty:
        x = svl_df.copy()
        x.insert(0, "station", "SVL")
        pieces.append(x)
    if pieces:
        pd.concat(pieces, ignore_index=True).to_csv(
            output_root / f"summary_{condition}_{style}_RO_{bin_tag}_median_rmse.csv",
            index=False,
        )
    return path


def _summary_for_bin(
    df: pd.DataFrame,
    style: str,
    condition: str,
    low: int,
    high: int | None,
):
    sub = df[
        (df["style"] == style)
        & (df["condition"] == condition)
        & ro_bin_mask(df["n_ro_profiles"], low, high)
    ].copy()
    if sub.empty:
        return {
            "sub": sub,
            "n_slots": 0,
            "n_tro_slots": 0,
            "n_svl_slots": 0,
            "tro": aggregate_station(sub),
            "svl": aggregate_station(sub),
        }
    return {
        "sub": sub,
        "n_slots": int(sub["window_start_utc"].nunique()),
        "n_tro_slots": int(sub.loc[sub["station"] == "TRO", "window_start_utc"].nunique()),
        "n_svl_slots": int(sub.loc[sub["station"] == "SVL", "window_start_utc"].nunique()),
        "tro": aggregate_station(sub[sub["station"] == "TRO"]),
        "svl": aggregate_station(sub[sub["station"] == "SVL"]),
    }


def _summary_global_xmax(summaries: list[dict]) -> float:
    values = []
    for summary in summaries:
        for station_key in ("tro", "svl"):
            frame = summary[station_key]
            if frame.empty:
                continue
            median = frame["median"].to_numpy(dtype=float)
            std = frame["std"].to_numpy(dtype=float)
            good = np.isfinite(median) & np.isfinite(std)
            if np.any(good):
                values.extend((median[good] + std[good]).tolist())
    if not values:
        return 1.0
    xmax = float(np.nanmax(values))
    return max(xmax * 1.05, 1.0)


def _draw_summary_panel(ax, summary: dict, bin_label: str, show_legend: bool = False):
    tro_df = summary["tro"]
    svl_df = summary["svl"]

    if not tro_df.empty:
        x = tro_df["median"].to_numpy(dtype=float)
        s = tro_df["std"].to_numpy(dtype=float)
        y = tro_df["altitude_km"].to_numpy(dtype=float)
        ax.plot(x, y, ".-", color="red", markersize=3,
                label=f"TRO (n={summary['n_tro_slots']})")
        ax.fill_betweenx(y, np.maximum(x - s, 0.0), x + s,
                         color="red", alpha=0.16)

    if not svl_df.empty:
        x = svl_df["median"].to_numpy(dtype=float)
        s = svl_df["std"].to_numpy(dtype=float)
        y = svl_df["altitude_km"].to_numpy(dtype=float)
        ax.plot(x, y, ".-", color="green", markersize=3,
                label=f"SVL (n={summary['n_svl_slots']})")
        ax.fill_betweenx(y, np.maximum(x - s, 0.0), x + s,
                         color="green", alpha=0.16)

    if summary["n_slots"] == 0:
        ax.text(0.5, 0.5, "No hourly slots", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)

    ax.set_title(f"{bin_label} RO\nn={summary['n_slots']} slots")
    ax.grid(True, alpha=0.25)
    if show_legend and (not tro_df.empty or not svl_df.empty):
        ax.legend(loc="best", fontsize=7)


def plot_ro_count_summary_row(
    df: pd.DataFrame,
    style: str,
    condition: str,
    output_root: Path,
):
    """One 1x5 row for one style/boost condition, common x-axis range."""
    summaries = [
        _summary_for_bin(df, style, condition, low, high)
        for _tag, _label, low, high in RO_COUNT_BINS
    ]
    if all(item["n_slots"] == 0 for item in summaries):
        return None

    xmax = _summary_global_xmax(summaries)
    fig, axes = plt.subplots(1, 5, figsize=(27, 7.5), sharey=True)

    for idx, (ax, summary, (_tag, label, _low, _high)) in enumerate(
        zip(axes, summaries, RO_COUNT_BINS)
    ):
        _draw_summary_panel(ax, summary, label, show_legend=(idx == 0))
        ax.set_xlim(0.0, xmax)
        ax.set_xlabel(r"Median $N_e$ error (m$^{-3}$)")
        if idx == 0:
            ax.set_ylabel("Altitude (km)")

    fig.suptitle(
        f"{STYLE_DISPLAY.get(style, style)} [{condition}] -- "
        "median analysis $N_e$ error vs. ISR by RO-count bin\n"
        "Red=TRO, green=SVL, shading=+/-1 sigma; identical x limits on all panels",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = output_root / f"summary_{condition}_{style}_RO_bins_1x5_median_rmse.png"
    save_figure(fig, path, dpi=180)
    return path


def plot_condition_summary_grid(
    df: pd.DataFrame,
    styles: list[str],
    condition: str,
    output_root: Path,
):
    """4x5 grid: four styles for one boost condition, common x limits."""
    all_summaries = {
        style: [
            _summary_for_bin(df, style, condition, low, high)
            for _tag, _label, low, high in RO_COUNT_BINS
        ]
        for style in styles
    }
    flat = [item for summaries in all_summaries.values() for item in summaries]
    if all(item["n_slots"] == 0 for item in flat):
        return None

    xmax = _summary_global_xmax(flat)
    fig, axes = plt.subplots(
        len(styles), 5, figsize=(28, 6.2 * len(styles)), sharey=True
    )
    axes = np.atleast_2d(axes)

    for row, style in enumerate(styles):
        for col, (summary, (_tag, label, _low, _high)) in enumerate(
            zip(all_summaries[style], RO_COUNT_BINS)
        ):
            ax = axes[row, col]
            _draw_summary_panel(ax, summary, label, show_legend=(col == 0))
            ax.set_xlim(0.0, xmax)
            ax.set_xlabel(r"Median $N_e$ error (m$^{-3}$)")
            if col == 0:
                ax.set_ylabel(
                    f"{STYLE_DISPLAY.get(style, style)}\nAltitude (km)"
                )

    fig.suptitle(
        f"All four styles [{condition}] -- median analysis $N_e$ error by RO-count bin\n"
        "SVL is validation-only; red=TRO, green=SVL, shading=+/-1 sigma; "
        "identical x limits across all panels",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = output_root / f"summary_{condition}_all4styles_RO_bins_4x5_median_rmse.png"
    save_figure(fig, path, dpi=180)
    return path


def plot_all_8runs_summary_grid(
    df: pd.DataFrame,
    styles: list[str],
    output_root: Path,
):
    """
    8x5 grid: each style appears twice (no_boost, boosted), with ONE common
    x-axis range across all 40 panels for direct visual comparison.
    """
    run_keys = [
        (style, condition["name"])
        for style in styles
        for condition in RUN_CONDITIONS
    ]

    all_summaries = {
        (style, condition): [
            _summary_for_bin(df, style, condition, low, high)
            for _tag, _label, low, high in RO_COUNT_BINS
        ]
        for style, condition in run_keys
    }
    flat = [item for summaries in all_summaries.values() for item in summaries]
    if all(item["n_slots"] == 0 for item in flat):
        return None

    xmax = _summary_global_xmax(flat)
    fig, axes = plt.subplots(
        len(run_keys), 5, figsize=(28, 5.3 * len(run_keys)), sharey=True
    )
    axes = np.atleast_2d(axes)

    for row, (style, condition) in enumerate(run_keys):
        for col, (summary, (_tag, label, _low, _high)) in enumerate(
            zip(all_summaries[(style, condition)], RO_COUNT_BINS)
        ):
            ax = axes[row, col]
            _draw_summary_panel(ax, summary, label, show_legend=(col == 0))
            ax.set_xlim(0.0, xmax)
            ax.set_xlabel(r"Median $N_e$ error (m$^{-3}$)")
            if col == 0:
                ax.set_ylabel(
                    f"{STYLE_DISPLAY.get(style, style)}\n{condition}\nAltitude (km)"
                )

    fig.suptitle(
        "All 8 filter cases: 4 parameterizations x no-boost/boosted\n"
        "Boosted = amplitude 0.5 in log10-density space; "
        "red=TRO, green=SVL; identical x limits across ALL panels",
        fontsize=17,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = output_root / "summary_all8runs_RO_bins_8x5_median_rmse.png"
    save_figure(fig, path, dpi=180)
    return path


def save_daily_ro_status(rows: list[dict], day: datetime) -> Path:
    """
    Merge newly checked hours into the persistent daily RO-status CSV.

    Behavior across separate script runs:
      - existing hours are preserved;
      - rerunning an hour replaces that hour's previous row;
      - rows are always sorted chronologically by window_start_utc.
    """
    path = OUTPUT_ROOT / f"{day:%Y-%m-%d}_ro_hourly_status.csv"

    new_df = pd.DataFrame(rows)
    if new_df.empty:
        return path

    if path.exists():
        try:
            old_df = pd.read_csv(path)
        except Exception:
            old_df = pd.DataFrame()
    else:
        old_df = pd.DataFrame()

    combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)

    # The newest result for a repeated hour wins.
    combined = combined.drop_duplicates(
        subset=["window_start_utc"],
        keep="last",
    )

    # ISO timestamps sort correctly in chronological order.
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

    if not TRO_ISR_FILE.exists():
        raise FileNotFoundError(TRO_ISR_FILE)
    if not SVL_ISR_FILE.exists():
        raise FileNotFoundError(SVL_ISR_FILE)

    ro_dir = ro_day_dir(day)
    if not ro_dir.is_dir():
        raise FileNotFoundError(
            f"RO day directory not found: {ro_dir}\n"
            f"For {day:%Y-%m-%d}, expected YYYY.DDD = {ro_dir.name}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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

    print("Loading ISR validation files once...")
    tro = load_isr_dataset(TRO_ISR_FILE)
    svl = load_isr_dataset(SVL_ISR_FILE)
    print(
        f"  TRO: {tro.station.name}, {len(tro.time)} profiles, "
        f"{tro.altitude.min():.1f}-{tro.altitude.max():.1f} km"
    )
    print(
        f"  SVL: {svl.station.name}, {len(svl.time)} profiles, "
        f"{svl.altitude.min():.1f}-{svl.altitude.max():.1f} km"
    )

    all_error_rows: list[dict] = []
    hourly_rows: list[dict] = []
    mode_timing_rows: list[dict] = []
    daily_ro_status_rows: list[dict] = []
    total_t0 = time.perf_counter()

    for hour in range(args.start_hour, args.end_hour):
        start = day + timedelta(hours=hour)
        end = start + timedelta(hours=1)
        folder = hour_folder_name(start, end)
        hour_dir = OUTPUT_ROOT / folder

        print("\n" + "=" * 96)
        print(f"HOUR: {hour_label(start, end)}")
        print(f"OUTPUT (only if RO >= 5): {hour_dir}")
        print("=" * 96)

        if args.skip_existing and successful_summary_exists(hour_dir):
            print("SKIP: both no_boost and boosted already successful")
            existing_summary = pd.read_csv(hour_dir / "hourly_summary.csv")
            for row in existing_summary.to_dict("records"):
                row["status"] = "skipped_existing"
                hourly_rows.append(row)
            existing_errors = hour_dir / "dual_isr_errors.csv"
            if existing_errors.exists():
                all_error_rows.extend(
                    pd.read_csv(existing_errors).to_dict("records")
                )
            continue

        hour_t0 = time.perf_counter()
        hour_error_rows: list[dict] = []
        hour_summary_rows: list[dict] = []

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

                skipped_for_ro = (n_ro < 5 or n_tec == 0)
                daily_ro_status_rows.append({
                    "window_start_utc": start.isoformat(),
                    "window_end_utc": end.isoformat(),
                    "n_ro_profiles": int(n_ro),
                    "n_tec_rays": int(n_tec),
                    "skipped": bool(skipped_for_ro),
                    "status": "skipped_too_few_ro" if skipped_for_ro else "accepted_for_assimilation",
                })
                daily_status_path = save_daily_ro_status(daily_ro_status_rows, day)

                if skipped_for_ro:
                    print(
                        f"SKIP: only {n_ro} accepted RO profiles "
                        "(minimum required = 5); no hourly folder/files created."
                    )
                    print(f"Daily RO status updated: {daily_status_path}")
                    continue

                hour_dir.mkdir(parents=True, exist_ok=True)
                precomputed_ro = hour_dir / "accepted_ro_observations.nc"
                shutil.copy2(precomputed_ro_tmp, precomputed_ro)

            tro_match = nearest_profile_in_hour(tro, start, end)
            svl_match = nearest_profile_in_hour(svl, start, end)

            # -------------------------------------------------------------
            # 2. Two package runs per hour:
            #      A) no_boost: fresh IRI ensemble
            #      B) boosted : SAME saved IRI ensemble + boost
            #    Each package run contains all four styles.
            # -------------------------------------------------------------
            base_edp_samples = None

            for condition in RUN_CONDITIONS:
                condition_name = condition["name"]
                condition_label = condition["label"]
                condition_dir = hour_dir / condition_name
                condition_dir.mkdir(parents=True, exist_ok=True)
                label = (
                    f"tromso_allstyles_{condition_name}_"
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
                )

                print("\n" + "-" * 96)
                print(f"CONDITION: {condition_name}")
                print("Styles:", ", ".join(cfg.styles))
                if condition_name == "no_boost":
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
                    print("Base EDP retained in memory for boosted A/B run")

                # ---------------------------------------------------------
                # 3. Extra TRO/SVL validation for each style in this
                #    boost condition.
                # ---------------------------------------------------------
                for style in cfg.styles:
                    display = (
                        f"{STYLE_DISPLAY.get(style, style)} "
                        f"[{condition_label}]"
                    )
                    print("Creating TRO+SVL validation:", display)

                    alt_tro, f_tro, a_tro = decode_profiles_at_station(
                        cfg, result, style, tro.station.lat, tro.station.lon,
                        in_memory_parameterizations,
                    )
                    alt_svl, f_svl, a_svl = decode_profiles_at_station(
                        cfg, result, style, svl.station.lat, svl.station.lon,
                        in_memory_parameterizations,
                    )

                    if not np.array_equal(alt_tro, alt_svl):
                        raise RuntimeError(
                            "TRO/SVL decoded model altitude grids differ"
                        )
                    model_altitude = alt_tro
                    style_dir = condition_dir / f"{label}_{style}"

                    plot_dual_isr_ne(
                        display,
                        start,
                        end,
                        n_ro_used,
                        n_tec_used,
                        model_altitude,
                        f_tro,
                        a_tro,
                        tro,
                        tro_match,
                        svl,
                        svl_match,
                        style_dir
                        / "dual_isr_ne_comparison_TRO_model_location.png",
                    )

                    # Physically location-matched TRO and SVL figure.
                    fig, axes = plt.subplots(
                        1, 2, figsize=(12, 9), sharey=True
                    )
                    for ax, code, station, match, f_prof, a_prof in (
                        (axes[0], "TRO", tro, tro_match, f_tro, a_tro),
                        (axes[1], "SVL", svl, svl_match, f_svl, a_svl),
                    ):
                        fm = np.nanmean(f_prof, axis=-1)
                        fs = np.nanstd(f_prof, axis=-1)
                        am = np.nanmean(a_prof, axis=-1)
                        ass = np.nanstd(a_prof, axis=-1)
                        ax.plot(
                            fm, model_altitude, color="C0",
                            label="forecast mean"
                        )
                        ax.fill_betweenx(
                            model_altitude, fm - fs, fm + fs,
                            color="C0", alpha=0.2
                        )
                        ax.plot(
                            am, model_altitude, color="C1",
                            label="analysis mean"
                        )
                        ax.fill_betweenx(
                            model_altitude, am - ass, am + ass,
                            color="C1", alpha=0.2
                        )
                        if match is not None:
                            mt, mne = match
                            truth_color = (
                                "black" if code == "TRO" else "green"
                            )
                            ax.plot(
                                mne,
                                station.altitude,
                                linestyle="--",
                                marker=".",
                                color=truth_color,
                                markersize=5,
                                label=f"{code} ISR ({mt:%H:%M})",
                            )
                        ax.set_title(f"Model at {code} location")
                        ax.set_xlabel(r"Electron density (m$^{-3}$)")
                        ax.grid(True, alpha=0.2)
                        ax.legend(loc="best", fontsize=8)

                    axes[0].set_ylabel("Altitude (km)")
                    fig.suptitle(
                        f"{display} -- location-matched TRO/SVL validation\n"
                        f"{hour_label(start, end)} | "
                        f"RO profiles={n_ro_used} | TEC rays={n_tec_used}"
                    )
                    fig.tight_layout()
                    save_figure(
                        fig,
                        style_dir
                        / "dual_isr_ne_comparison_location_matched.png",
                    )

                    tro_error = station_error_profile(
                        model_altitude, a_tro, tro, tro_match
                    )
                    svl_error = station_error_profile(
                        model_altitude, a_svl, svl, svl_match
                    )

                    plot_dual_isr_error(
                        display,
                        start,
                        end,
                        n_ro_used,
                        n_tec_used,
                        tro_error,
                        svl_error,
                        style_dir / "dual_isr_rmse_by_altitude.png",
                    )

                    append_error_rows(
                        hour_error_rows,
                        style=style,
                        condition=condition_name,
                        station_code="TRO",
                        start=start,
                        end=end,
                        n_ro=n_ro_used,
                        n_tec=n_tec_used,
                        error_profile=tro_error,
                    )
                    append_error_rows(
                        hour_error_rows,
                        style=style,
                        condition=condition_name,
                        station_code="SVL",
                        start=start,
                        end=end,
                        n_ro=n_ro_used,
                        n_tec=n_tec_used,
                        error_profile=svl_error,
                    )

                    del f_tro, a_tro, f_svl, a_svl
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
                    "base_edp_samples_path": "",
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

            # Both conditions completed.
            all_error_rows.extend(hour_error_rows)
            for _row in reversed(daily_ro_status_rows):
                if _row["window_start_utc"] == start.isoformat():
                    _row["status"] = "success"
                    _row["skipped"] = False
                    break
            save_daily_ro_status(daily_ro_status_rows, day)

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
            for _row in reversed(daily_ro_status_rows):
                if _row["window_start_utc"] == start.isoformat():
                    _row["status"] = "failed_after_ro_precheck"
                    _row["skipped"] = False
                    break
            save_daily_ro_status(daily_ro_status_rows, day)
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

    # ---------------------------------------------------------------------
    # 4. Optional aggregate RO-bin products are disabled below.
    #    No root-level hourly_run_summary_* or hourly_dual_isr_errors_* CSVs.
    # ---------------------------------------------------------------------
    # Root-level daily aggregate run/error CSVs intentionally disabled.
    # Keep only the per-hour CSVs plus the daily RO-status CSV and timing CSV.
    errors_df = pd.DataFrame(all_error_rows)

    # ================================================================
    # RO-count-bin comparison outputs DISABLED
    # ================================================================
    # Keep all per-hour assimilation and ISR validation outputs unchanged.
    #
    # The cross-hour comparison products grouped by RO count
    # (5 / 6-10 / 11-15 / 16-20 / >20) are intentionally not generated.
    # This disables only the aggregate RO-bin CSVs/figures:
    #
    #   summary_<condition>_<style>_RO_<bin>_median_rmse.csv
    #   summary_<condition>_<style>_RO_<bin>_median_rmse.png
    #   summary_<condition>_<style>_RO_bins_1x5_median_rmse.png
    #   summary_<condition>_all4styles_RO_bins_4x5_median_rmse.png
    #   summary_all8runs_RO_bins_8x5_median_rmse.png
    #
    # The helper functions are intentionally left in this file so these
    # summary products can be re-enabled later without changing the hourly
    # scientific workflow.
    #
    # if not errors_df.empty:
    #     print(
    #         "\nCreating RO-count-binned median +/-1 sigma "
    #         "summary figures for all 8 cases..."
    #     )
    #
    #     for style in args.styles:
    #         for condition in RUN_CONDITIONS:
    #             condition_name = condition["name"]
    #
    #             for bin_tag, bin_label, low, high in RO_COUNT_BINS:
    #                 path = plot_ro_count_summary(
    #                     errors_df,
    #                     style,
    #                     condition_name,
    #                     bin_tag,
    #                     bin_label,
    #                     low,
    #                     high,
    #                     OUTPUT_ROOT,
    #                 )
    #                 if path is not None:
    #                     print("  saved", path)
    #
    #             row_path = plot_ro_count_summary_row(
    #                 errors_df, style, condition_name, OUTPUT_ROOT
    #             )
    #             if row_path is not None:
    #                 print("  saved", row_path)
    #
    #     for condition in RUN_CONDITIONS:
    #         grid_path = plot_condition_summary_grid(
    #             errors_df,
    #             args.styles,
    #             condition["name"],
    #             OUTPUT_ROOT,
    #         )
    #         if grid_path is not None:
    #             print("  saved", grid_path)
    #
    #     grid8_path = plot_all_8runs_summary_grid(
    #         errors_df, args.styles, OUTPUT_ROOT
    #     )
    #     if grid8_path is not None:
    #         print("  saved", grid8_path)

    if mode_timing_rows:
        pd.DataFrame(mode_timing_rows).to_csv(
            OUTPUT_ROOT / f"{args.date}_mode_timing.csv", index=False
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
    print("Output root:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
