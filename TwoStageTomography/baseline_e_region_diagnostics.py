#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline E-region diagnostics for the E-region retrieval plan
(TwoStageTomography/E_Region_Retrieval_Plan.md, Section 2).

Reads the per-hour ``.npz`` files written by
``Run_Scoring/extract_run_diagnostics.py`` and reports, binned by ray
tangent altitude:

- the ray count (how the observation weight is spread over altitude);
- the median observed TEC;
- the median part of the analysis TEC that comes from grid levels below
  the E top (the E-region "signal" a low ray carries);
- the RMS and mean of the analysis TEC residual.

It also reports the lag-1 autocorrelation of the analysis residuals inside
each occultation, which tells whether the misfit is white noise (near 0) or
a smooth, correlated error such as a per-occultation offset (near 1).

    python TwoStageTomography/baseline_e_region_diagnostics.py \
        --npz-dir "/path/to/run_scoring/extract" \
        --condition boosted --style PCA_3D_10ex
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

TANGENT_BINS_KM = (60, 90, 110, 130, 150, 200, 300, 400, 600, 900)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--condition", default="boosted")
    parser.add_argument("--style", default="PCA_3D_10ex")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key = f"{args.condition}|{args.style}|"
    files = sorted(glob.glob(str(args.npz_dir / "*.npz")))
    if not files:
        raise SystemExit(f"No .npz files in {args.npz_dir}")

    y_all, tang_all, e_all, res_all, lag1 = [], [], [], [], []
    for path in files:
        d = np.load(path)
        y = d["y"]
        res = y - d[key + "Ya"]
        y_all.append(y)
        tang_all.append(d["tang"])
        e_all.append(d[key + "YEa"])
        res_all.append(res)
        for occ in np.unique(d["occ"]):
            m = d["occ"] == occ
            if m.sum() > 10:
                x = res[m] - res[m].mean()
                lag1.append(np.corrcoef(x[:-1], x[1:])[0, 1])

    y = np.concatenate(y_all)
    tang = np.concatenate(tang_all)
    e_part = np.concatenate(e_all)
    res = np.concatenate(res_all)

    print(f"{len(files)} hours, {y.size} rays, {args.condition} / {args.style}")
    print(f"{'tangent km':>12} {'n':>6} {'TEC med':>8} {'E part':>7} {'res rms':>8} {'res mean':>9}")
    for lo, hi in zip(TANGENT_BINS_KM[:-1], TANGENT_BINS_KM[1:]):
        m = (tang >= lo) & (tang < hi)
        if not m.any():
            continue
        print(
            f"{lo:5d}-{hi:<6d} {m.sum():6d} {np.median(y[m]):8.1f} "
            f"{np.median(e_part[m]):7.2f} {np.sqrt(np.mean(res[m] ** 2)):8.2f} "
            f"{np.mean(res[m]):+9.2f}"
        )
    print(
        "Median lag-1 autocorrelation of analysis residuals within an "
        f"occultation: {np.nanmedian(lag1):.2f} ({len(lag1)} occultations)"
    )


if __name__ == "__main__":
    main()
