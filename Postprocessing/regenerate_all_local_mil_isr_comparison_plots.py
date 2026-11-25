#!/usr/bin/env python3
"""Regenerates ISR-comparison plots for *every* locally stored Millstone
Hill-centered run, using only local ISR/ionosonde data files and local run
results -- no Google Drive access at all (2026-11-19 user request: a
coworker needs to run this on a server that only has local-disk copies,
since bulk plot regeneration against the shared Drive has proven
unreliable -- writes have been observed silently reverting after some
delay during an earlier bulk-regeneration job).

This is the multi-run-tree counterpart to
``regenerate_mil_isr_comparison_plots.py`` (which regenerates one already
-known run tree). Here, ``--results-root`` is a local directory that may
hold *several* run trees side by side (mirroring how the shared drive's
own ``tomography_output/`` folder is laid out, e.g.
``MIL_aug2530_2025_R2k_..., MIL_aug2727_2025_ROI2000kmRadius_..., ...``),
not necessarily all Millstone Hill-centered -- Tromso-centered run trees
(``TRO_...``) are routinely stored alongside them. Each immediate
subdirectory of ``--results-root`` is checked against its own first
``*_edp_samples.nc``'s ``Lat``/``Lon`` attrs (not its directory name --
robust to any naming convention) and kept only if that center is
Millstone Hill's (42.6195, -71.4917, same constants as
``Driver_Scripts/optimize_vertical_grid.py``'s ``MIL_LAT``/``MIL_LON``)
within ``--center-tolerance-deg``. Every kept run tree is then regenerated
in one call each via ``isr_comparison.replot_isr_comparison_tree`` (module
docstring there: reconstructs each style's density at the ISR station from
files ``package_run.py`` already saved -- no EnKF/pipeline rerun).

Usage::

    # List which local run trees would be touched, without writing anything
    python3 Postprocessing/regenerate_all_local_mil_isr_comparison_plots.py \\
        --results-root /data/tomography_output \\
        --isr-file /data/ISR_Data/MILISR2025_nonan.nc \\
        --ionosonde-file /data/Ionosonde_Data/MILIONO_Aug2530_2025.nc \\
        --dry-run

    # Actually regenerate
    python3 Postprocessing/regenerate_all_local_mil_isr_comparison_plots.py \\
        --results-root /data/tomography_output \\
        --isr-file /data/ISR_Data/MILISR2025_nonan.nc \\
        --ionosonde-file /data/Ionosonde_Data/MILIONO_Aug2530_2025.nc
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs", REPO / "Spline_Basis"):
    sys.path.insert(0, str(p))

import xarray as xr

from Assimilation_Cycle import isr_comparison

# Same constants as Driver_Scripts/optimize_vertical_grid.py's MIL_LAT/MIL_LON.
MIL_LAT, MIL_LON = 42.6195, -71.4917


def _edp_center(edp_path: Path) -> tuple[float, float] | None:
    """``(Lat, Lon)`` from one ``*_edp_samples.nc``'s own attrs, or ``None``
    if the file can't be opened or doesn't carry them (e.g. corrupted or
    not actually an EDPSamples file)."""
    try:
        with xr.open_dataset(edp_path, decode_timedelta=False) as ds:
            return float(ds.attrs["Lat"]), float(ds.attrs["Lon"])
    except Exception as exc:
        print(f"[regen]   warning: could not read Lat/Lon from {edp_path}: {exc}", flush=True)
        return None


def find_mil_run_dirs(
    results_root: Path, center_tolerance_deg: float, verbose: bool = True,
) -> list[Path]:
    """Every immediate subdirectory of ``results_root`` that holds at least
    one ``*_edp_samples.nc`` whose own ``Lat``/``Lon`` attrs put it within
    ``center_tolerance_deg`` of Millstone Hill. Centers are checked from a
    single representative file per candidate directory (its first
    ``*_edp_samples.nc`` in sorted order) -- every hour/condition folder in
    one run tree shares the same ROI center, so checking one is enough and
    avoids opening every file in every run tree just to filter."""
    kept: list[Path] = []
    for candidate in sorted(d for d in results_root.iterdir() if d.is_dir()):
        edp_paths = sorted(candidate.rglob("*_edp_samples.nc"))
        if not edp_paths:
            if verbose:
                print(f"[regen] {candidate.name}: skipped (no *_edp_samples.nc found)", flush=True)
            continue
        center = _edp_center(edp_paths[0])
        if center is None:
            continue
        lat, lon = center
        if abs(lat - MIL_LAT) <= center_tolerance_deg and abs(lon - MIL_LON) <= center_tolerance_deg:
            kept.append(candidate)
            if verbose:
                print(f"[regen] {candidate.name}: Millstone Hill-centered "
                      f"(lat={lat:.4f}, lon={lon:.4f}) -- {len(edp_paths)} hour/condition folders", flush=True)
        elif verbose:
            print(f"[regen] {candidate.name}: skipped (center lat={lat:.4f}, lon={lon:.4f}, "
                  f"not Millstone Hill)", flush=True)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", required=True, type=Path,
                         help="Local directory holding one or more run trees side by side "
                              "(e.g. a local copy of the shared drive's tomography_output/ folder).")
    parser.add_argument("--isr-file", required=True,
                         help="Local, preprocessed ISR netCDF (Ne/Ne_min/Ne_max/profiles_used_per_gate).")
    parser.add_argument("--ionosonde-file", default=None,
                         help="Local, preprocessed ionosonde netCDF -- either the nmF2/hmF2-only "
                              "schema or the full vertical-profile schema (altitude/Ne), which "
                              "also adds a median-EDP curve + range band. Omit to skip the "
                              "ionosonde overlay entirely.")
    parser.add_argument("--center-tolerance-deg", type=float, default=0.5,
                         help="Max |lat|/|lon| difference (degrees) from Millstone Hill's own "
                              "(42.6195, -71.4917) for a local run tree to be regenerated (default: 0.5).")
    parser.add_argument("--isr-range-percentile", type=float, default=0.0,
                         help="Forwarded to the cycle-wide range plot's old-schema fallback "
                              "(no effect when the ISR file has Ne_min/Ne_max -- see "
                              "isr_comparison._isr_window_bounds_and_confidence).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Only list which local run trees would be regenerated; write nothing.")
    args = parser.parse_args()

    results_root = args.results_root
    if not results_root.is_dir():
        raise SystemExit(f"--results-root does not exist or is not a directory: {results_root}")

    print(f"Scanning {results_root} for Millstone Hill-centered run trees "
          f"(tolerance={args.center_tolerance_deg} deg)...", flush=True)
    run_dirs = find_mil_run_dirs(results_root, args.center_tolerance_deg)

    if not run_dirs:
        raise SystemExit("No Millstone Hill-centered run trees found under --results-root.")

    print(f"\nFound {len(run_dirs)} Millstone Hill-centered run tree(s):")
    for run_dir in run_dirs:
        print(" ", run_dir)

    if args.dry_run:
        print("\n--dry-run: not regenerating anything.")
        return

    print(f"\nISR file:       {args.isr_file}")
    print(f"Ionosonde file: {args.ionosonde_file or '(none)'}")

    total_written = []
    for i, run_dir in enumerate(run_dirs, start=1):
        print(f"\n[{i}/{len(run_dirs)}] Regenerating: {run_dir}", flush=True)
        written = isr_comparison.replot_isr_comparison_tree(
            run_dir, args.isr_file,
            ionosonde_file_path=args.ionosonde_file,
            isr_range_percentile=args.isr_range_percentile,
        )
        print(f"  wrote {len(written)} file(s)")
        total_written.extend(written)

    print(f"\nDone. Wrote {len(total_written)} file(s) across {len(run_dirs)} run tree(s).")


if __name__ == "__main__":
    main()
