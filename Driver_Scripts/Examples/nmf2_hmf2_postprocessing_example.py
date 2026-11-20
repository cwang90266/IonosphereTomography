#!/usr/bin/env python3
"""Server-ready driver for the NmF2/hmF2 cross-run post-processing tools
(2026-11-20) -- runs ``Postprocessing/tabulate_nmf2_hmf2.py`` then
``Postprocessing/plot_nmf2_hmf2_comparison.py`` in one invocation, failing
loudly (nonzero exit, no silent partial output) if either step errors.
See ``Documentation/README_nmf2_hmf2_postprocessing.md`` for what each
step actually does and the full column/flag reference; this script is
just a thin, edit-in-place wrapper for unattended/headless use.

Unlike some other example scripts in this directory, this one hardcodes
no local-machine path -- every path is a CLI argument, so it runs as-is
on a server where the run trees and station files live somewhere other
than this repo's own checkout.

Usage::

    # One station, one run tree (or a directory of several run trees
    # side by side -- tabulate_nmf2_hmf2.py walks the whole subtree):
    python3 Driver_Scripts/Examples/nmf2_hmf2_postprocessing_example.py \\
        --results-root /data/tomography_output \\
        --isr MIL=/data/isr/MILISR2025_nonan.nc \\
        --ionosonde MIL=/data/ionosonde/MILIONO_Aug2530_2025.nc \\
        --out-dir /data/nmf2_hmf2_postprocessing

    # Several stations, filtered to one boost condition, run unattended:
    nohup python3 Driver_Scripts/Examples/nmf2_hmf2_postprocessing_example.py \\
        --results-root /data/tomography_output \\
        --isr MIL=/data/isr/MILISR2025_nonan.nc \\
        --isr TRO=/data/isr/TROISR2025_nonan.nc \\
        --out-dir /data/nmf2_hmf2_postprocessing \\
        --conditions boosted \\
        > nmf2_hmf2_postprocessing.log 2>&1 &

Output lands under ``--out-dir``: ``nmf2_hmf2_table.csv`` and a ``plots/``
subdirectory of PNGs, one per ``--group-by`` column.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABULATE_SCRIPT = REPO / "Postprocessing" / "tabulate_nmf2_hmf2.py"
PLOT_SCRIPT = REPO / "Postprocessing" / "plot_nmf2_hmf2_comparison.py"


def _run(cmd: list[str]) -> None:
    print(f"[driver] running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[driver] command failed (exit {result.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", required=True, type=Path,
                         help="Local directory holding one or more run trees (passed through "
                              "to tabulate_nmf2_hmf2.py's --results-root).")
    parser.add_argument("--isr", action="append", default=[], metavar="CODE=PATH",
                         help="Repeatable; passed through to tabulate_nmf2_hmf2.py's --isr.")
    parser.add_argument("--ionosonde", action="append", default=[], metavar="CODE=PATH",
                         help="Repeatable; passed through to tabulate_nmf2_hmf2.py's --ionosonde.")
    parser.add_argument("--out-dir", required=True, type=Path,
                         help="Where to write nmf2_hmf2_table.csv and plots/ (created if missing).")
    parser.add_argument("--e-top-km", type=float, default=150.0)
    parser.add_argument("--f-top-km", type=float, default=700.0)
    parser.add_argument("--center-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--hours", nargs="*", default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--styles", nargs="*", default=None)
    parser.add_argument("--group-by", nargs="*", default=["style", "condition", "roi_radius_km"],
                         help="Passed through to plot_nmf2_hmf2_comparison.py's --group-by.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_csv = args.out_dir / "nmf2_hmf2_table.csv"
    plots_dir = args.out_dir / "plots"

    tabulate_cmd = [
        sys.executable, str(TABULATE_SCRIPT),
        "--results-root", str(args.results_root),
        "--out", str(table_csv),
        "--e-top-km", str(args.e_top_km),
        "--f-top-km", str(args.f_top_km),
        "--center-tolerance-deg", str(args.center_tolerance_deg),
    ]
    for item in args.isr:
        tabulate_cmd += ["--isr", item]
    for item in args.ionosonde:
        tabulate_cmd += ["--ionosonde", item]
    if args.hours is not None:
        tabulate_cmd += ["--hours", *args.hours]
    if args.conditions is not None:
        tabulate_cmd += ["--conditions", *args.conditions]
    if args.styles is not None:
        tabulate_cmd += ["--styles", *args.styles]
    _run(tabulate_cmd)

    plot_cmd = [
        sys.executable, str(PLOT_SCRIPT),
        "--csv", str(table_csv),
        "--out-dir", str(plots_dir),
        "--group-by", *args.group_by,
    ]
    _run(plot_cmd)

    print(f"[driver] done -- table: {table_csv}, plots: {plots_dir}", flush=True)


if __name__ == "__main__":
    main()
