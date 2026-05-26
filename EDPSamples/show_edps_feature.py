# -*- coding: utf-8 -*-
"""
show_edps_feature.py
Visualize the sampling-parameter distributions and EDP feature distributions
stored in an EDPSamples object.

Layout (3 rows × 2 columns):
  (a) f107          (b) ap
  (c) ig12          (d) rz12
  (E) log-density   (F) heights / shape params
"""

import numpy as np
import matplotlib.pyplot as plt

from edp_samples import EDPSamples


def show_edps_feature(edps: EDPSamples, title: str = "") -> plt.Figure:
    """
    6-panel overview of sampling parameters and EDP features.

    Parameters
    ----------
    edps : EDPSamples
        Dataset whose sampling_parameters and feathure_EDPS variables are plotted.
    title : str, optional
        Overall figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    sp   = edps.sampling_parameters          # DataFrame (n_sample, n_param)
    feat = edps.feature_edps                 # ndarray   (13, n_geo, n_sample)

    # Flatten geo × sample → single distribution axis
    feat_flat = feat.reshape(feat.shape[0], -1)   # (13, n_geo * n_sample)

    feat_labels = EDPSamples.FEATURE_LABEL

    # 1-based column groups as stated in the docstring, converted to 0-based
    log_cols = [0, 2, 4, 6, 10]          # cols 1,3,5,7,11  → nmf2,nmf1,nme,nmd,valley_base
    lin_cols = [1, 3, 5, 7, 8, 9, 11, 12] # cols 2,4,6,8,9,10,12,13 → hmf2,hmf1,hme,hmd,hhalf,b0,valley_top,b1

    # ------------------------------------------------------------------ layout
    fig = plt.figure(figsize=(14, 12))
    gs  = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_E = fig.add_subplot(gs[2, 0])
    ax_F = fig.add_subplot(gs[2, 1])

    # ------------------------------------------------ panels a–d: sampling params
    param_cfg = [
        (ax_a, "f107",  "F10.7 Solar Flux Index",       "(a)"),
        (ax_b, "ap",    "Ap Geomagnetic Index",          "(b)"),
        (ax_c, "ig12",  "IG12 Index",                    "(c)"),
        (ax_d, "rz12",  "Rz12 Smoothed Sunspot Number",  "(d)"),
    ]
    for ax, col, xlabel, panel in param_cfg:
        if col in sp.columns:
            data = sp[col].dropna().to_numpy()
            ax.hist(data, bins=30, color="steelblue", edgecolor="white", linewidth=0.4)
            ax.set_xlabel(xlabel, fontsize=9)
        else:
            ax.text(0.5, 0.5, f"Column '{col}' not found",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title(panel, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=8)

    # ------------------------------------------ panel E: log-scale density features
    colors_E = plt.cm.tab10(np.linspace(0.0, 0.49, len(log_cols)))
    for idx, color in zip(log_cols, colors_E):
        data  = feat_flat[idx]
        valid = data[np.isfinite(data) & (data > 0)]
        if valid.size == 0:
            continue
        ax_E.hist(np.log10(valid), bins=40, alpha=0.65, color=color,
                  edgecolor="none", label=feat_labels[idx])

    ax_E.set_title("(E)", loc="left", fontsize=10, fontweight="bold")
    ax_E.set_xlabel("log$_{10}$(Electron Density, m$^{-3}$)", fontsize=9)
    ax_E.set_ylabel("Count", fontsize=9)
    ax_E.legend(fontsize=7, ncol=2, framealpha=0.7)
    ax_E.grid(True, alpha=0.3, linestyle=":")
    ax_E.tick_params(labelsize=8)

    # --------------------------------------- panel F: linear height / shape features
    colors_F = plt.cm.tab10(np.linspace(0.0, 0.79, len(lin_cols)))
    for idx, color in zip(lin_cols, colors_F):
        data  = feat_flat[idx]
        valid = data[np.isfinite(data)]
        if valid.size == 0:
            continue
        ax_F.hist(valid, bins=40, alpha=0.65, color=color,
                  edgecolor="none", label=feat_labels[idx])

    ax_F.set_title("(F)", loc="left", fontsize=10, fontweight="bold")
    ax_F.set_xlabel("Value (km or dimensionless)", fontsize=9)
    ax_F.set_ylabel("Count", fontsize=9)
    ax_F.legend(fontsize=7, ncol=2, framealpha=0.7)
    ax_F.grid(True, alpha=0.3, linestyle=":")
    ax_F.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    return fig
