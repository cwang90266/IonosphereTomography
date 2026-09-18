#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end pipeline, stage 1: generate IRI2020 driving-parameter samples for
a given epoch, run the real IRI2020 Fortran driver over a regional
EDPSamples grid, and save the results.

This is the slow stage (dominated by the IRI2020 Fortran batch call --
roughly 1 minute per ~11,500 (geolocation, sample) profiles on this
machine/build; see the printed timing estimate before it starts). Stage 2
(`end_to_end_regional_test_analyze.py`) reads the NetCDF this script writes
and does the (fast) parameterization evaluation, so this only needs to be
rerun when the input ensemble or grid changes.

Requires the compiled IRI2020 driver: `source init_iri2020_env.sh` from this
directory before running this script (sets IRI2020_PATH).

Usage
-----
    source init_iri2020_env.sh
    python3 end_to_end_regional_test_generate.py \\
        --epoch 2025-07-01 --nsample 2000 \\
        --lat 50 --lon -178 --radius 30 --dlat 2.5 \\
        --alt-min 60 --alt-max 900 --alt-step 10 \\
        --out /Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Claude_Test
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "IRI_Sample_Inputs"))
sys.path.insert(0, str(_THIS_DIR / "EDPSamples"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from IRI_Sample_inputs import IRI_Sample_Inputs
from edp_samples import EDPSamples


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epoch", default="2025-07-01", help="Simulation epoch (date or datetime string).")
    p.add_argument("--nsample", type=int, default=2000, help="Number of IRI2020 driving-parameter samples.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sample generation.")
    p.add_argument("--hour-range", type=int, default=12, help="randomSamples hour_sample_range.")
    p.add_argument("--f107-range", type=int, default=90, help="randomSamples f107_sample_range (days).")
    p.add_argument("--ap-range", type=int, default=90, help="randomSamples ap_sample_range (days).")
    p.add_argument("--ig-range", type=int, default=6, help="randomSamples ig_sample_range (months).")
    p.add_argument("--rz-range", type=int, default=6, help="randomSamples rz_sample_range (months).")
    p.add_argument("--lat", type=float, default=50.0, help="Regional grid center latitude (deg).")
    p.add_argument("--lon", type=float, default=-178.0, help="Regional grid center longitude (deg).")
    p.add_argument("--radius", type=float, default=30.0, help="Regional grid angular radius (deg).")
    p.add_argument("--dlat", type=float, default=2.5, help="Regional grid target point spacing (deg).")
    p.add_argument("--alt-min", type=float, default=60.0, help="Altitude grid minimum (km).")
    p.add_argument("--alt-max", type=float, default=900.0, help="Altitude grid maximum (km).")
    p.add_argument("--alt-step", type=float, default=10.0, help="Altitude grid step (km).")
    p.add_argument("--out", required=True, help="Output directory for samples, NetCDF, and plots.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not os.environ.get("IRI2020_PATH"):
        sys.exit(
            "IRI2020_PATH is not set. Run 'source init_iri2020_env.sh' from "
            f"{_THIS_DIR} before running this script."
        )

    np.random.seed(args.seed)

    # -- Stage 1a: generate the driving-parameter ensemble ----------------
    iri_in = IRI_Sample_Inputs(args.epoch)
    samples = iri_in.randomSamples(
        hour_sample_range=args.hour_range,
        f107_sample_range=args.f107_range,
        ap_sample_range=args.ap_range,
        ig_sample_range=args.ig_range,
        rz_sample_range=args.rz_range,
        nSample=args.nsample,
    )
    with open(f"{args.out}/iri_sample_inputs_{args.nsample}.pkl", "wb") as f:
        pickle.dump(samples, f)
    samples.to_csv(f"{args.out}/iri_sample_inputs_{args.nsample}.csv", index=False)
    print(f"Saved {args.nsample} IRI2020 input samples (pkl + csv).")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    labels = {"hour": "Hour (UT)", "f107": "F10.7 (sfu)", "ap": "ap index",
              "ig12": "IG12", "rz12": "Rz12"}
    for ax, col in zip(axes, ["hour", "f107", "ap", "ig12", "rz12"]):
        ax.hist(samples[col].dropna(), bins=30, color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_xlabel(labels[col]); ax.set_ylabel("Count"); ax.set_title(labels[col])
        ax.grid(True, alpha=0.3)
    axes[-1].axis("off")
    fig.suptitle(f"IRI2020 Input Parameter Distributions (N={args.nsample}, epoch {args.epoch})", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{args.out}/plot_iri_input_histograms.png", dpi=140, bbox_inches="tight")
    plt.close("all")
    print("Saved input-parameter histogram plot.")

    # -- Stage 1b: build the regional grid and run the real IRI2020 driver -
    altitude = np.arange(args.alt_min, args.alt_max + args.alt_step / 2, args.alt_step)
    geo_probe, _ = EDPSamples.genRegionalArea(args.lat, args.lon, args.radius, args.dlat)
    n_profiles = geo_probe.shape[0] * args.nsample
    print(f"Regional grid: n_geo={geo_probe.shape[0]}, n_alt={len(altitude)}, "
          f"n_sample={args.nsample} -> {n_profiles} (geo, sample) profiles.")

    t0 = time.time()
    ds = EDPSamples(
        DateTime=f"{args.epoch}T12:00:00" if "T" not in args.epoch else args.epoch,
        geo_type="Regional",
        altitude=altitude,
        sampling_parameters=samples,
        Lat=args.lat, Lon=args.lon, radius=args.radius, dLat=args.dlat,
        evaluate_iri=1,
    )
    elapsed = time.time() - t0
    print(f"IRI2020 batch run complete in {elapsed / 60:.1f} minutes.")

    nan_frac = float(np.mean(np.isnan(ds.edps)))
    neg_frac = float(np.mean(ds.edps < 0))
    print(f"edps NaN fraction: {nan_frac:.4f}, negative-value fraction: {neg_frac:.4f} "
          "(IRI2020 can return sentinel/unreliable values below ~90-100 km; "
          "the parameterization code floors these before taking log10).")

    nc_path = f"{args.out}/regional_edps_{args.nsample}samples.nc"
    ds.saveNetCDF(nc_path)
    print(f"Saved EDPSamples to {nc_path}")

    ax = ds.plot_geolocation()
    ax.figure.savefig(f"{args.out}/plot_regional_grid.png", dpi=140, bbox_inches="tight")
    plt.close("all")

    for target_alt in (100.0, 200.0, 300.0, 400.0):
        ax = ds.plot_horizontal_field(scalar="mean", target_alt=target_alt)
        ax.figure.savefig(f"{args.out}/plot_mean_edp_{int(target_alt)}km.png", dpi=140, bbox_inches="tight")
        plt.close("all")
    print("Saved grid and mean-EDP plots.")
    print("STAGE 1 DONE")


if __name__ == "__main__":
    main()
