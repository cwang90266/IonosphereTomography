#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Plot the distribution of physical input parameters and the key
   charateristics of the resulting electron density profiles.

Created on Fri May 22 12:58:06 2026

@author: Chunming Wang
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Feature labels match EDPSamples.FEATURE_LABEL (0-based index → column name)
_FEATURE_LABEL = (
    "nmf2", "hmf2", "nmf1", "hmf1", "nme",
    "hme",  "nmd",  "hmd",  "hhalf", "b0",
    "valley_base", "valley_top", "b1"
)

# 1-based column groups converted to 0-based indices
# E: electron-density peaks  → log scale
#_LOG_IDX = [0, 2, 4, 6]          # cols 1,3,5,7
_LOG_IDX = [0, 2, 4, 6, 10, 11]          # cols 1,3,5,7
# F: heights and shape params → linear scale
#_LIN_IDX = [1, 3, 5, 7, 8, 9, 10, 11, 12] # cols 2,4,6,8,9,10,11,12,13
_LIN_IDX = [1, 3, 5, 7, 8, 9, 12] # cols 2,4,6,8,9,10,11,12,13


def show_edps_feature(iri_parameters: pd.DataFrame, feature_edps: np.ndarray):
    """
    6-panel figure showing distributions of IRI sampling parameters and EDP features.

    Parameters
    ----------
    iri_parameters : pd.DataFrame, shape (n_sample, n_param)
        Must contain columns 'f107', 'ap', 'ig12', 'rz12'.
    feature_edps : np.ndarray, shape (13, n_geo, n_sample)
        EDP feature array from EDPSamples.feature_edps.
        Row order follows EDPSamples.FEATURE_LABEL.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    layout = [
        ["A", "B", "C", "D"],
        ["E", "E", "F", "F"],
    ]
    fig, ax = plt.subplot_mosaic(layout, figsize=(14, 7))

    # ------------------------------------------------------------------ A–D
    param_cfg = [
        ("A", "f107",  "F10.7 Solar Flux Index"),
        ("B", "ap",    "Ap Geomagnetic Index"),
        ("C", "ig12",  "IG12 Index"),
        ("D", "rz12",  "Rz12 Smoothed Sunspot Number"),
    ]
    for key, col, xlabel in param_cfg:
        ax[key].set_title(col, fontsize=10, fontweight="bold")
        if col in iri_parameters.columns:
            data = iri_parameters[col].dropna().to_numpy()
            ax[key].hist(data, bins=30, color="steelblue",
                         edgecolor="white", linewidth=0.4)
            ax[key].set_xlabel(xlabel, fontsize=8)
        else:
            ax[key].text(0.5, 0.5, f"'{col}' not found",
                         ha="center", va="center",
                         transform=ax[key].transAxes, fontsize=9)
        ax[key].set_ylabel("Count", fontsize=8)
        ax[key].grid(True, alpha=0.3, linestyle=":")
        ax[key].tick_params(labelsize=8)

    # Flatten geo × sample so each row is a single distribution
    feat_flat = feature_edps.reshape(feature_edps.shape[0], -1)  # (13, n_geo*n_sample)

    # ------------------------------------------------------------------ E
    ax["E"].set_title("Densities at Key Heights", fontsize=10, fontweight="bold")
    colors_E = plt.cm.tab10(np.linspace(0.0, 0.49, len(_LOG_IDX)))
    for idx, color in zip(_LOG_IDX, colors_E):
        data  = feat_flat[idx]
        valid = data[np.isfinite(data) & (data > 0)]
        if valid.size == 0:
            continue
        
        counts, bin_edges, patches = ax["E"].hist(np.log10(valid), bins=40, alpha=0.65, color=color,
                     density=True,edgecolor="none", label=_FEATURE_LABEL[idx])
        if _FEATURE_LABEL[idx] == "nmf2":
            max_Height = max(counts)
            
    ax["E"].set_xlabel("log$_{10}$(Electron Density, m$^{-3}$)", fontsize=8)
    ax["E"].set_ylabel("Count", fontsize=8)
    ax["E"].set_ylim(0,1.2*max_Height)
    ax["E"].legend(fontsize=7, ncol=2, framealpha=0.7)
    ax["E"].grid(True, alpha=0.3, linestyle=":")
    ax["E"].tick_params(labelsize=8)

    # ------------------------------------------------------------------ F
    ax["F"].set_title("Key Heights", fontsize=10, fontweight="bold")
    colors_F = plt.cm.tab10(np.linspace(0.0, 0.79, len(_LIN_IDX)))
    for idx, color in zip(_LIN_IDX, colors_F):
        data  = feat_flat[idx]
        valid = data[np.isfinite(data)]
        if valid.size == 0:
            continue
        counts, bin_edges, patches = ax["F"].hist(valid, bins=40, alpha=0.65, color=color,
                     density=True,edgecolor="none", label=_FEATURE_LABEL[idx])
        if _FEATURE_LABEL[idx] == "hmf2":
            max_Height = max(counts)
            
    ax["F"].set_xlabel("Value (km or dimensionless)", fontsize=8)
    ax["F"].set_ylabel("Count", fontsize=8)
    ax["F"].set_ylim(0,1.2*max_Height)
    ax["F"].legend(fontsize=7, ncol=2, framealpha=0.7)
    ax["F"].grid(True, alpha=0.3, linestyle=":")
    ax["F"].tick_params(labelsize=8)

    fig.suptitle("Distribution of Input Parameters and Key EDP Characteristics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
