#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end pipeline, stage 2: evaluate PCA and ANCHOR parameterization
performance against the EDPSamples NetCDF written by
`end_to_end_regional_test_generate.py`, and visualize the difference between
the original IRI2020 EDP and each parameterization's reconstruction.

- PCA_1D (linear EDP) and PCA_1D_10ex (log10 EDP) are fit on the FULL
  ensemble (cheap: PCA is closed-form linear algebra on an (n_alt, n_geo *
  n_sample) matrix regardless of how large n_geo*n_sample is).
- ANCHOR is fit on a random SUBSAMPLE of profiles, since it's a per-profile
  nonlinear optimization (~40-50 profiles/sec on this altitude grid/machine)
  -- fitting the full n_geo*n_sample ensemble would take hours to days at
  realistic grid/sample sizes. Uses EDP_Parameterization directly (not
  Parameterized_EDPSamples, which assumes the full geo x sample tensor)
  since the subsample is an irregular flat list of (geo, sample) pairs.

"Truncation threshold" for PCA is interpreted as the fraction of ensemble
variance the truncation is allowed to DISCARD (retaining_threshold = 1 -
given_threshold), the usual meaning of an energy/error truncation threshold
in POD/EOF analysis.

Usage
-----
    python3 end_to_end_regional_test_analyze.py \\
        --nc /path/to/regional_edps_2000samples.nc \\
        --out /Users/cwang/Documents/Consulting/PlanetIQ/Runs/Tomography_Test/Claude_Test \\
        --pca-linear-threshold 1e-3 --pca-log-threshold 1e-2 \\
        --anchor-subsample 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "EDPSamples"))
sys.path.insert(0, str(_THIS_DIR / "Parameterization"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edp_samples import EDPSamples
import Parameterization as P


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nc", required=True, help="EDPSamples NetCDF from stage 1.")
    p.add_argument("--out", required=True, help="Output directory for plots and saved parameterizations.")
    p.add_argument("--pca-linear-threshold", type=float, default=1e-3,
                   help="Discarded-variance threshold for PCA_1D (linear EDP).")
    p.add_argument("--pca-log-threshold", type=float, default=1e-2,
                   help="Discarded-variance threshold for PCA_1D_10ex (log10 EDP).")
    p.add_argument("--anchor-subsample", type=int, default=3000,
                   help="Number of random (geo, sample) profiles to fit ANCHOR on.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _run_pca(ds, style, threshold, out_dir, rng):
    pe = P.Parameterized_EDPSamples(ds, style=style, hyper_params={"retaining_threshold": 1 - threshold})
    tag = "pca_linear" if style == "PCA_1D" else "pca_log"

    pe.saveNetCDF(f"{out_dir}/parameterization_{tag}.nc")

    diag = pe.Parameterization.hyper_params["pca_diagnostics"]
    summary = pe.error_summary()
    print(f"{style} (discard-threshold={threshold:g}): n_retained={diag.n_retained}, "
          f"cumulative_variance={diag.cumulative_variance_ratio[diag.n_retained - 1]:.6f}, "
          f"rmse_overall={summary['rmse_overall']:.4g}, max_abs_error={summary['max_abs_error']:.4g}")

    ax = pe.plot_reconstruction_error_statistics()
    ax[0].figure.savefig(f"{out_dir}/plot_{tag}_error_statistics.png", dpi=140, bbox_inches="tight")
    plt.close("all")

    ax = pe.plot_pca_spectrum()
    ax[0].figure.savefig(f"{out_dir}/plot_{tag}_spectrum.png", dpi=140, bbox_inches="tight")
    plt.close("all")

    n_geo, n_sample = ds.edps.shape[1], ds.edps.shape[2]
    g_idx, s_idx = rng.integers(0, n_geo), rng.integers(0, n_sample)
    ax = pe.plot_reconstruction_profile(sample_idx=s_idx, geo_idx=g_idx)
    ax[0].figure.savefig(f"{out_dir}/plot_{tag}_example_profile.png", dpi=140, bbox_inches="tight")
    plt.close("all")

    return pe


def _annotated_heatmap(fig, ax, matrix, labels, title, cmap, vmax=None, fmt=None):
    """Small helper: an imshow heatmap with per-cell numeric annotations and
    NaN cells (e.g. an undefined correlation for a zero-variance parameter)
    shown as gray 'n/a' rather than a misleading color or a bare 'nan'."""
    masked = np.ma.masked_invalid(matrix)
    if vmax is None:
        vmax = np.max(np.abs(masked)) if masked.count() else 1.0
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")
    im = ax.imshow(masked, cmap=cmap_obj, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "n/a", ha="center", va="center", color="dimgray", fontsize=7)
                continue
            text = fmt(val) if fmt else f"{val:.2f}"
            color = "white" if (vmax > 0 and abs(val) / vmax > 0.6) else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.85)


def _plot_anchor_param_covariance(param_vec, labels, out_dir):
    """
    Sample covariance and correlation of the 8 fitted ANCHOR parameters
    across the subsample. A parameter that never varies across the fit (a
    fixed fallback value rather than a genuine fit -- watch for this
    especially with hmE, whose E-region-peak detector can fail to find an
    interior local max for some ensembles) has undefined correlation
    (0/0), shown as 'n/a' rather than a spurious number.
    """
    n_sub = param_vec.shape[1]
    cov = np.cov(param_vec)
    std = np.sqrt(np.diag(cov))
    zero_var = std == 0
    if np.any(zero_var):
        print(f"NOTE: zero-variance ANCHOR parameter(s) across all {n_sub} fits: "
              f"{[labels[i] for i in np.where(zero_var)[0]]} -- correlation undefined for these.")
    safe_std = np.where(zero_var, 1.0, std)
    corr = cov / np.outer(safe_std, safe_std)
    corr[zero_var, :] = np.nan
    corr[:, zero_var] = np.nan

    def _cov_fmt(v):
        return f"{v:.1e}" if (abs(v) >= 1000 or (0 < abs(v) < 0.01)) else f"{v:.2f}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _annotated_heatmap(fig, axes[0], cov, labels, f"Sample covariance (n={n_sub})", "RdBu_r", fmt=_cov_fmt)
    _annotated_heatmap(fig, axes[1], corr, labels, f"Sample correlation (n={n_sub})", "RdBu_r", vmax=1.0)
    fig.suptitle("ANCHOR parameter covariance / correlation", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/plot_anchor_param_covariance.png", dpi=140, bbox_inches="tight")
    plt.close("all")


def _run_anchor(ds, n_sub, out_dir, rng):
    altitude = ds.altitude
    edps = ds.edps
    n_alt, n_geo, n_sample = edps.shape
    n_sub = min(n_sub, n_geo * n_sample)

    flat_idx = rng.choice(n_geo * n_sample, size=n_sub, replace=False)
    geo_idx_sub, sample_idx_sub = np.unravel_index(flat_idx, (n_geo, n_sample))
    edps_sub = edps[:, geo_idx_sub, sample_idx_sub]

    anchor = P.EDP_Parameterization(style="ANCHOR")
    param_vec = anchor.get_parameter(edps_sub, alt=altitude)
    density_hat = anchor.get_density(param_vec, alt=altitude)
    residual_log = (np.log10(np.maximum(edps_sub, 1.0)) - np.log10(np.maximum(density_hat, 1.0)))
    residual_lin = edps_sub - density_hat
    rmse = float(np.sqrt(np.nanmean(residual_log ** 2)))
    print(f"ANCHOR (n_sub={n_sub}): log10-rmse_overall={rmse:.4g}")

    np.savez(
        f"{out_dir}/parameterization_anchor_subsample.npz",
        altitude=altitude, geo_idx=geo_idx_sub, sample_idx=sample_idx_sub,
        edps_original=edps_sub, edps_reconstructed=density_hat,
        anchor_param_vec=param_vec, anchor_param_labels=np.array(P.ANCHOR_PARAM_LABELS),
        log10_residual=residual_log,
    )

    # log10-space error (large dynamic range, so this view stays informative
    # across the full altitude range).
    pct_levels = [1, 5, 16, 50, 84, 95, 99]
    p01, p05, p16, med, p84, p95, p99 = np.nanpercentile(residual_log, pct_levels, axis=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axvline(0, color="black", lw=1)
    ax.fill_betweenx(altitude, p01, p99, color="lightcoral", alpha=0.3, label="1st-99th %ile")
    ax.fill_betweenx(altitude, p05, p95, color="indianred", alpha=0.5, label="5th-95th %ile")
    ax.fill_betweenx(altitude, p16, p84, color="darkred", alpha=0.7, label="16th-84th %ile")
    ax.plot(med, altitude, color="black", lw=1.5, label="Median")
    ax.set_xlabel("log10(EDP) reconstruction error")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"ANCHOR reconstruction error (n={n_sub} random profiles)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.4, linestyle=":")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/plot_anchor_error_statistics_log10.png", dpi=140, bbox_inches="tight")
    plt.close("all")

    # Absolute + relative error, same layout/style as Parameterized_EDPSamples.
    # plot_reconstruction_error_statistics (used for the PCA styles), so
    # ANCHOR and PCA are directly comparable side by side.
    safe_edps = np.where(np.abs(edps_sub) > 0, np.abs(edps_sub), np.nan)
    rel_residual = residual_lin / safe_edps * 100.0
    ap01, ap05, ap16, amed, ap84, ap95, ap99 = np.nanpercentile(residual_lin, pct_levels, axis=1)
    rp01, rp05, rp16, rmed, rp84, rp95, rp99 = np.nanpercentile(rel_residual, pct_levels, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    ax0, ax1 = axes
    ax0.axvline(0, color="black", lw=1)
    ax0.fill_betweenx(altitude, ap01, ap99, color="lightcoral", alpha=0.3, label="1st-99th %ile")
    ax0.fill_betweenx(altitude, ap05, ap95, color="indianred", alpha=0.5, label="5th-95th %ile")
    ax0.fill_betweenx(altitude, ap16, ap84, color="darkred", alpha=0.7, label="16th-84th %ile")
    ax0.plot(amed, altitude, color="black", lw=1.5, label="Median")
    ax0.set_xlabel("Reconstruction error (m$^{-3}$)")
    ax0.set_ylabel("Altitude (km)")
    ax0.set_title(f"ANCHOR (n={n_sub}): absolute error")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.4, linestyle=":")

    ax1.axvline(0, color="black", lw=1)
    ax1.fill_betweenx(altitude, rp01, rp99, color="lightblue", alpha=0.3, label="1st-99th %ile")
    ax1.fill_betweenx(altitude, rp05, rp95, color="dodgerblue", alpha=0.5, label="5th-95th %ile")
    ax1.fill_betweenx(altitude, rp16, rp84, color="blue", alpha=0.7, label="16th-84th %ile")
    ax1.plot(rmed, altitude, color="black", lw=1.5, label="Median")
    ax1.set_xlabel("Reconstruction error (% of |EDP|)")
    ax1.set_title(f"ANCHOR (n={n_sub}): relative error")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.4, linestyle=":")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/plot_anchor_error_statistics.png", dpi=140, bbox_inches="tight")
    plt.close("all")

    _plot_anchor_param_covariance(param_vec, list(P.ANCHOR_PARAM_LABELS), out_dir)

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
    example_idx = rng.choice(n_sub, size=4, replace=False)
    for ax_i, k in zip(axes, example_idx):
        ax_i.plot(edps_sub[:, k], altitude, color="black", lw=1.5, label="Original (IRI2020)")
        ax_i.plot(density_hat[:, k], altitude, color="C1", lw=1.5, linestyle="--", label="ANCHOR reconstruction")
        ax_i.set_xlabel("Ne (m$^{-3}$)")
        ax_i.set_xscale("log")
        ax_i.grid(True, alpha=0.4, linestyle=":")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("ANCHOR: original vs. reconstructed EDP (4 random example profiles)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/plot_anchor_example_profiles.png", dpi=140, bbox_inches="tight")
    plt.close("all")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ds = EDPSamples.fromNetCDF(args.nc)
    n_alt, n_geo, n_sample = ds.edps.shape
    print(f"Loaded EDPSamples: n_alt={n_alt}, n_geo={n_geo}, n_sample={n_sample}")

    _run_pca(ds, "PCA_1D", args.pca_linear_threshold, args.out, rng)
    _run_pca(ds, "PCA_1D_10ex", args.pca_log_threshold, args.out, rng)
    _run_anchor(ds, args.anchor_subsample, args.out, rng)

    print("STAGE 2 DONE")


if __name__ == "__main__":
    main()
