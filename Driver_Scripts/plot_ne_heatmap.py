#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Altitude-vs-time Ne heatmap for the extended ISR dataset, in the same style
as Runs/Tomography_Test/Claude_Test/first1000_NE.jpg (jet colormap, white
gaps, "Data Number" x-axis, Ne colorbar in units of 1e11 m^-3) -- a final
visual review of TROISR2025_extended.nc, covering the full 82-900 km grid
(native ISR gates + IRI2020 extension) instead of just the ISR's native
~650 km ceiling.

Usage
-----
    /opt/anaconda3/bin/python3 plot_ne_heatmap.py
"""
from __future__ import annotations

import argparse

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--extended-file",
        default="/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Claude_Test/TROISR2025_extended.nc",
    )
    p.add_argument("--out", default="/Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Claude_Test")
    p.add_argument("--n-samples", type=int, default=1000, help="Number of leading time samples to plot (Data Number 1..n).")
    p.add_argument("--start", type=int, default=0, help="Starting time index (0-based).")
    p.add_argument("--vmax", type=float, default=1.0e12, help="Colorbar max, Ne in m^-3.")
    return p.parse_args()


def main():
    args = parse_args()
    ds = xr.open_dataset(args.extended_file, decode_timedelta=False)
    altitude = ds["altitude"].values
    Ne = ds["Ne"].values

    sl = slice(args.start, args.start + args.n_samples)
    Ne_sub = np.ma.masked_invalid(Ne[:, sl].astype(np.float64) / 1.0e11)
    data_number = np.arange(sl.start + 1, sl.start + Ne_sub.shape[1] + 1)

    # Boundary between native ISR gates and the IRI2020 extension: the first
    # gate strictly above 650km (the ISR's native ceiling, see extend_isr_
    # edp_with_iri2020.py's EXT_ALT_MIN).
    ext_start_idx = int(np.searchsorted(altitude, 650.0, side="right") - 1)

    cmap = matplotlib.colormaps["jet"].copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(16, 6.5))
    mesh = ax.pcolormesh(data_number, altitude, Ne_sub, cmap=cmap, vmin=0.0, vmax=args.vmax / 1.0e11,
                          shading="nearest")
    ax.axhline(altitude[ext_start_idx], color="white", lw=1.0, ls="--")
    ax.text(data_number[-1], altitude[ext_start_idx], " IRI2020 extension above this line  ",
            color="black", fontsize=9, va="center", ha="right",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))
    ax.set_xlabel("Data Number")
    ax.set_ylabel("Altitude (km)")
    ax.set_xlim(data_number[0], data_number[-1])
    ax.set_ylim(altitude[0], altitude[-1])

    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label("Ne (m$^{-3}$)")
    cbar.ax.set_title(r"$\times 10^{11}$", fontsize=10, loc="left")

    fig.suptitle(f"Extended ISR Ne: Data Number {data_number[0]}-{data_number[-1]} "
                 f"({np.datetime_as_string(ds['time_utc'].values[sl][0], unit='m')} to "
                 f"{np.datetime_as_string(ds['time_utc'].values[sl][-1], unit='m')})", fontsize=11)
    fig.tight_layout()
    out_path = f"{args.out}/plot_ne_heatmap_first{args.n_samples}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
