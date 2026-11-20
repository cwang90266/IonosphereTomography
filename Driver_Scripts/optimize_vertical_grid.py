#!/usr/bin/env python3
"""Full-resolution cubic-spline vertical-grid optimization (Documentation/
Spline_Basis_Knot_Optimization_Plan.md §4's named-but-until-now-unbuilt
driver), using the main assimilation package's own EDPSamples-generation
code (Spline_Basis/iri_collection.py) for the training/validation
collection, and no subsampling (members_per_column=None, metric_members=
None, geo_stride=1) when fitting.

Default scope (user, 2026-11-19 -- "one grid per location is sufficient
for now, I don't expect optimal knots is expected to change
substantially"): **one** Millstone Hill training condition + **one**
Tromso (May) validation condition, each still the full 2000-member
ensemble at full geo resolution -- just not pooled across many hours.
``--full`` instead pools all real hourly conditions of the existing
Millstone Hill MIL_aug2530_2025_R2k_..._3DL_3Dp_1Dp training run (34
hours, 2025-08-26 to 08-29) and every 4th hour of the existing
TRO_may1821_2025/TRO_oct1417_2025 R2k_..._3DL_3Dp_1Dp validation runs
(~14 of 56 hours) -- a ~10 hour real-IRI2020 job, only worth it if the
one-condition-per-location assumption above is revisited. Conditions are
enumerated via conditions_from_run_hours against the real run directories
on the shared drive, not hand-transcribed, in both cases.

Altitude grid: 60-900 km / 10 km, matching the committed grid file's domain
(h0=60 is intentional -- see the module/plan docstrings -- not something
this script changes).

Usage::

    python3 Driver_Scripts/optimize_vertical_grid.py            # one condition per location (default)
    python3 Driver_Scripts/optimize_vertical_grid.py --full     # full pooled collection (~10 hours)
"""
import argparse
import sys
from pathlib import Path

REPO = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography")
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs",
          REPO / "Spline_Basis"):
    sys.path.insert(0, str(p))

import numpy as np

import iri_collection as ic
from knot_optimization import KnotOptimizationConfig, optimize_from_config

GDRIVE = Path(
    "/Users/cwang/Library/CloudStorage/GoogleDrive-cwang@planetiq.com/Shared drives/"
    "science analysis/ionosphere/tomography_output"
)
MIL_RUN_DIR = GDRIVE / "MIL_aug2530_2025_R2k_BOp3_Obs5_IRIScale1_LOSfrac0p65_PCAdisScale1_3DL_3Dp_1Dp"
TRO_MAY_RUN_DIR = GDRIVE / "TRO_may1821_2025_R2k_BOp3_Obs5_IRIScale1_LOSfrac0p65_PCAdisScale1_3DL_3Dp_1Dp"
TRO_OCT_RUN_DIR = GDRIVE / "TRO_oct1417_2025_R2k_BOp3_Obs5_IRIScale1_LOSfrac0p65_PCAdisScale1_3DL_3Dp_1Dp"

MIL_LAT, MIL_LON = 42.6195, -71.4917
TRO_LAT, TRO_LON = 69.6, 19.2
GRID_RADIUS_DEG = 18.0

ALTITUDE_GRID = np.arange(60.0, 901.0, 10.0)

OUTPUT_ROOT = Path(
    "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Claude_Test/"
    "spline_knots/full_resolution"
)

# Same hyperparameters as the existing reduced-ensemble grid file
# (grid_h60_dvar_linear_n14.json) -- revisiting n_points/delta_min_km is a
# separate, later question (residual_vs_n), not this script's job.
N_POINTS = 14
DELTA_MIN_KM = [[60.0, 10.0], [150.0, 10.0], [250.0, 20.0]]
DEGREE = 3
FIT_SPACE = "linear"
REPRESENTATION = "ratio"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Use the full collection instead of one condition per location: all 34 "
                              "MIL training hours plus every-4th-hour TRO validation (~14 conditions). "
                              "Default is one MIL training condition + one TRO validation condition "
                              "(user, 2026-11-19: 'one grid per location is sufficient for now, I "
                              "don't expect optimal knots is expected to change substantially') -- a "
                              "full multi-hour pooled fit is a ~10 hour real-IRI2020 job, not needed "
                              "unless that assumption is revisited.")
    parser.add_argument("--n-ensemble", type=int, default=2000,
                         help="Members per condition (default: 2000, the full production ensemble size).")
    parser.add_argument("--output-name", default="grid_h60_dvar_linear_n14_full.json",
                         help="Grid file name under the output root (default leaves the existing "
                              "reduced-ensemble grid file untouched for comparison).")
    args = parser.parse_args()

    collection_dir = OUTPUT_ROOT / "collection"
    training_conditions = ic.conditions_from_run_hours(MIL_RUN_DIR, MIL_LAT, MIL_LON, GRID_RADIUS_DEG, "MIL")
    tro_may = ic.conditions_from_run_hours(TRO_MAY_RUN_DIR, TRO_LAT, TRO_LON, GRID_RADIUS_DEG, "TRO_may")
    tro_oct = ic.conditions_from_run_hours(TRO_OCT_RUN_DIR, TRO_LAT, TRO_LON, GRID_RADIUS_DEG, "TRO_oct")
    if args.full:
        # Full validation sets are large (29 May + 27 Oct = 56) relative to
        # the 34-condition training set; subsampled every 4th hour *within
        # each range* (user, 2026-11-19) so both date ranges stay
        # represented rather than one dominating -- ~7 + ~7 = ~14 conditions.
        validation_conditions = tro_may[::4] + tro_oct[::4]
    else:
        training_conditions = training_conditions[:1]
        validation_conditions = tro_may[:1]

    print(f"Training conditions: {len(training_conditions)}")
    print(f"Validation conditions: {len(validation_conditions)}")

    training_files = ic.build_collection(
        training_conditions, ALTITUDE_GRID, collection_dir / "training", n_ensemble=args.n_ensemble,
    )
    validation_files = ic.build_collection(
        validation_conditions, ALTITUDE_GRID, collection_dir / "validation", n_ensemble=args.n_ensemble,
    )

    cfg = KnotOptimizationConfig(
        h0_km=float(ALTITUDE_GRID[0]), h1_km=float(ALTITUDE_GRID[-1]), n_points=N_POINTS,
        delta_min_km=DELTA_MIN_KM, fit_space=FIT_SPACE, representation=REPRESENTATION, degree=DEGREE,
        training_files=[str(p) for p in training_files],
        validation_files=[str(p) for p in validation_files],
        output_path=str(OUTPUT_ROOT / args.output_name),
        members_per_column=None,    # full ensemble -- default is 200
        metric_members=None,        # full ensemble for metrics too -- default is 20
        geo_stride=1,               # full geo resolution (already the default)
    )

    print("Optimizing knot grid (full resolution, full ensemble)...")
    result = optimize_from_config(cfg, save=True, verbose=True)
    print(f"Done. train_objective={result.objective:.6g}  "
          f"validation_objective={result.validation_objective}")
    print(f"Grid file written to: {cfg.output_path}")


if __name__ == "__main__":
    main()
