#!/usr/bin/env python3
"""Regenerates every ISR-comparison plot under an existing Millstone Hill
run tree against the richer ISR file + ionosonde file (2026-11-19 user
request), with no EnKF/pipeline rerun at all. The default ionosonde file
is now the full vertical-profile schema (2026-11-25 user request --
``MILIonosonde_Ne_Aug2530_2025.nc``, not just the older ``nmF2``/``hmF2``
pair), so re-running this script against a previously completed run also
adds the ionosonde median-EDP curve + range band to its comparison plots
-- see ``isr_comparison.py``'s module docstring for how that's derived.

The run's forecast/analysis density at the ISR station is fully
reconstructed from files ``package_run.py`` already saved (each style's
``..._parameterized.nc`` + ``..._ensembles/ensemble_batch_*.nc``) --
see ``isr_comparison.py``'s module docstring for why that's possible
(``GenericObservationOperator.decode()`` needs no ray geometry). This
script is a thin wrapper around ``isr_comparison.replot_isr_comparison_tree``,
the actual batch entry point (mirrors ``impact_vectors.replot_impact_vectors_tree``'s
own precedent in this same package).

Usage::

    python3 Driver_Scripts/Examples/regenerate_mil_isr_comparison_plots.py
    python3 Driver_Scripts/Examples/regenerate_mil_isr_comparison_plots.py --root /path/to/other/run
"""
import argparse
import sys
from pathlib import Path

REPO = Path("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography")
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs", REPO / "Spline_Basis"):
    sys.path.insert(0, str(p))

from Assimilation_Cycle import isr_comparison

DEFAULT_ROOT = (
    "/Users/cwang/Library/CloudStorage/GoogleDrive-cwang@planetiq.com/Shared drives/"
    "science analysis/ionosphere/tomography_output/"
    "MIL_aug2530_2025_R2k_BOp3_Obs5_IRIScale1_LOSfrac0p65_PCAdisScale1_3DL_3Dp_1Dp"
)
DEFAULT_ISR_FILE = (
    "/Users/cwang/Documents/Consulting/PlanetIQ/Data/Tomography_data/ISR_Data/"
    "MILISR2025_nonan (1).nc"
)
DEFAULT_IONOSONDE_FILE = (
    "/Users/cwang/Documents/Consulting/PlanetIQ/Data/Tomography_data/Ionosonde_Data/"
    "MILIonosonde_Ne_Aug2530_2025.nc"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                         help="Run tree to walk for *_edp_samples.nc folders (default: the "
                              "2025-08-25/30 Millstone Hill R2k run on the shared drive).")
    parser.add_argument("--isr-file", default=DEFAULT_ISR_FILE,
                         help="Preprocessed ISR netCDF (Ne/Ne_min/Ne_max/profiles_used_per_gate).")
    parser.add_argument("--ionosonde-file", default=DEFAULT_IONOSONDE_FILE,
                         help="Preprocessed ionosonde netCDF -- either the nmF2/hmF2-only schema "
                              "or the full vertical-profile schema (altitude/Ne), which also adds "
                              "the median-EDP curve + range band. Pass '' to skip the ionosonde "
                              "overlay entirely.")
    parser.add_argument("--isr-range-percentile", type=float, default=0.0,
                         help="Forwarded to the cycle-wide range plot's old-schema fallback "
                              "(no effect when the ISR file has Ne_min/Ne_max -- see "
                              "isr_comparison._isr_window_bounds_and_confidence).")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"--root does not exist or is not a directory: {root}")

    ionosonde_file_path = args.ionosonde_file or None
    print(f"Regenerating ISR-comparison plots under: {root}")
    print(f"  ISR file:       {args.isr_file}")
    print(f"  Ionosonde file: {ionosonde_file_path or '(none)'}")

    written = isr_comparison.replot_isr_comparison_tree(
        root, args.isr_file,
        ionosonde_file_path=ionosonde_file_path,
        isr_range_percentile=args.isr_range_percentile,
    )

    print(f"\nWrote {len(written)} files:")
    for path in written:
        print(" ", path)


if __name__ == "__main__":
    main()
