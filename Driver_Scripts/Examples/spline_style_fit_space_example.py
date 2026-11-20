#!/usr/bin/env python3
"""Demonstrates the 'SPLINE' parameterization style end to end, and in
particular how its fit_space ('linear' / 'pow' / 'log10') is selected --
2026-11-19 user question: fit_space is a property of the *grid file* the
style is built from, not a separate style name, so switching it is just a
matter of pointing 'SPLINE' at a different grid file. See
``Documentation/Spline_Basis_README.md`` for the full explanation
(formulas, how to produce a grid file in each space, and how to wire a
chosen grid file into a ``CycleConfig``/``package_run`` run).

Reuses the real, full-resolution Millstone Hill ensemble already built and
validated for the knot optimizer (no fresh IRI2020 evaluation needed, so
this runs in seconds) -- the same file whose SPLINE-decoded reconstruction
RMSE was confirmed to match the grid file's own recorded training metric
exactly when the style was first implemented.

Usage::

    # Default: the existing, already-validated 'linear' grid file
    python3 Driver_Scripts/Examples/spline_style_fit_space_example.py

    # Any of the other committed grid files under Spline_Basis/grid_files/
    # (or one of your own, built per Documentation/Spline_Basis_README.md
    # Section 3) work the same way, with no other code change:
    python3 Driver_Scripts/Examples/spline_style_fit_space_example.py \\
        --grid-file Spline_Basis/grid_files/grid_h60_dvar_pow_alpha2_n14_full.json
"""
import argparse
import sys
from pathlib import Path

REPO = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography")
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs", REPO / "Spline_Basis"):
    sys.path.insert(0, str(p))

import numpy as np

import Parameterization as P
from edp_samples import EDPSamples
from knot_optimization import load_grid_file

DEFAULT_GRID_FILE = str(REPO / "Spline_Basis" / "grid_files" / "grid_h60_dvar_linear_n14_full.json")
DEFAULT_EDP_SAMPLES = (
    "/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Claude_Test/"
    "spline_knots/full_resolution/collection/training/MIL_2025-08-26_0000_edp_samples.nc"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-file", default=DEFAULT_GRID_FILE,
                         help="Grid file to build the SPLINE basis from -- this alone selects "
                              "fit_space ('linear'/'pow'/'log10'); default is the existing, "
                              "already-validated 'linear' grid file.")
    parser.add_argument("--edp-samples", default=DEFAULT_EDP_SAMPLES,
                         help="Real EDPSamples ensemble to fit/decode against (must span the "
                              "grid file's own h0-h1 range, e.g. 60-900 km).")
    args = parser.parse_args()

    grid = load_grid_file(args.grid_file)
    print(f"Grid file:  {args.grid_file}")
    print(f"  fit_space = {grid['fit_space']!r}, alpha = {grid.get('alpha')!r}, "
          f"grid_km = {grid['grid_km'][0]:.0f}-{grid['grid_km'][-1]:.0f} km, "
          f"degree = {grid['degree']}")

    edp = EDPSamples.fromNetCDF(args.edp_samples)
    print(f"\nEDPSamples: {args.edp_samples}")
    print(f"  altitude = {edp.altitude[0]:.0f}-{edp.altitude[-1]:.0f} km "
          f"({edp.altitude.size} levels), n_geo = {edp.edps.shape[1]}, "
          f"n_sample = {edp.edps.shape[2]}")

    # This one call is the entire style selection: 'SPLINE' + a grid file.
    # fit_space/alpha are read back from the grid file itself -- nothing
    # else about this call changes between 'linear', 'pow' and 'log10'.
    pe = P.Parameterized_EDPSamples(edp, style="SPLINE", hyper_params={"grid_file_path": args.grid_file})

    summary = pe.error_summary(in_log10=True)
    # rmse_by_altitude[0] is the structural Ne(h0)=0 boundary artifact, not
    # a real fit error -- see error_summary's own docstring.
    rmse_excl_h0 = float(np.sqrt(np.nanmean(summary["rmse_by_altitude"][1:] ** 2)))
    print(f"\nSPLINE reconstruction (log10 Ne), excluding the h0 boundary sample:")
    print(f"  RMSE = {rmse_excl_h0:.4f}")

    print(
        "\nTo use this same grid file in a full assimilation cycle, either:\n"
        "  cfg.spline_grid_file_path = " + repr(args.grid_file) + "\n"
        "or, per-style (takes priority over spline_grid_file_path):\n"
        "  cfg.hyper_params_by_style['SPLINE'] = {'grid_file_path': " + repr(args.grid_file) + "}\n"
        "with 'SPLINE' included in cfg.styles, and cfg.altitude_grid extended down to the grid "
        "file's own h0 (e.g. np.arange(60.0, 901.0, 10.0) -- *not* default_altitude_grid()'s "
        "90 km floor)."
    )


if __name__ == "__main__":
    main()
