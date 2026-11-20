# -*- coding: utf-8 -*-
"""Diagnostic plots for the spline grid optimization (plan Phase 5)."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from spline_basis import ConstrainedSplineBasis
from spline_fit import BatchedSplineFit, ProfileTargets, _from_fit_space

__all__ = ["plot_basis", "plot_grid_over_profiles", "plot_error_by_altitude", "plot_residual_vs_n"]


def plot_basis(grid: np.ndarray, degree: int = 3, scale_height: float | None = 300.0, ax=None):
    """The constrained basis functions on ``grid`` (ratio mode, given H)."""
    ax = ax or plt.subplots(figsize=(5, 6))[1]
    cb = ConstrainedSplineBasis(grid, degree, lower_value=0.0, upper_scale_height=scale_height)
    h = np.linspace(grid[0], grid[-1], 800)
    ax.plot(cb.design_matrix(h), h, lw=1)
    for g in grid:
        ax.axhline(g, color="0.8", lw=0.5, zorder=0)
    ax.set_xlabel("basis value")
    ax.set_ylabel("altitude (km)")
    ax.set_title(f"constrained basis, n={len(grid)}, m={degree}")
    return ax


def _ratio(t: ProfileTargets, y):
    return _from_fit_space(y, t.mean[:, :, None], t.fit_space, t.alpha) / t.mean[:, :, None]


def plot_grid_over_profiles(t: ProfileTargets, grids: dict, ax=None):
    """Ratio Ne/mean percentiles vs altitude with each named grid marked."""
    ax = ax or plt.subplots(figsize=(6, 7))[1]
    r = (t.ne / t.mean[:, :, None]).transpose(1, 0, 2).reshape(t.h.size, -1)
    p = np.percentile(r, [5, 25, 50, 75, 95], axis=1)
    ax.fill_betweenx(t.h, p[0], p[4], color="C0", alpha=0.15, label="ratio 5-95%")
    ax.fill_betweenx(t.h, p[1], p[3], color="C0", alpha=0.3, label="ratio 25-75%")
    ax.plot(p[2], t.h, color="C0", lw=1.5, label="median")
    xmax = p[4].max() * 1.05
    for i, (name, g) in enumerate(grids.items()):
        x = xmax + 0.12 * (i + 1) * (xmax - p[0].min())
        ax.scatter(np.full(len(g), x), g, marker="_", s=200, color=f"C{i + 1}", label=name)
    ax.set_xlabel("Ne / column mean")
    ax.set_ylabel("altitude (km)")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_error_by_altitude(t: ProfileTargets, grids: dict, degree: int = 3, ax=None):
    """RMS ratio error vs altitude for each named grid."""
    ax = ax or plt.subplots(figsize=(5, 7))[1]
    f = BatchedSplineFit(t, degree)
    for i, (name, g) in enumerate(grids.items()):
        res = f.fit(g)
        err = _ratio(t, res.y_fit) - t.ne / t.mean[:, :, None]
        rms = np.sqrt((err ** 2).mean(axis=(0, 2)))
        ax.plot(rms, t.h, color=f"C{i + 1}", label=f"{name} (all {np.sqrt((err[:, 1:] ** 2).mean()):.4f})")
        ax.scatter(np.zeros(len(g)), g, marker="_", color=f"C{i + 1}", s=80)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("RMS error in Ne / mean")
    ax.set_ylabel("altitude (km)")
    ax.legend(fontsize=8)
    return ax


def plot_residual_vs_n(rows: Sequence[dict], key: str = "rms_ratio", ax=None):
    """Model-selection curve from knot_optimization.residual_vs_n."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    n = [r["n"] for r in rows]
    ax.plot(n, [r[f"train_{key}"] for r in rows], "o-", color="C1", label="train, optimized")
    ax.plot(n, [r[f"train_{key}_uniform"] for r in rows], "o--", color="C1", alpha=0.6, label="train, uniform")
    if "val_" + key in rows[0]:
        ax.plot(n, [r[f"val_{key}"] for r in rows], "s-", color="C2", label="validation, optimized")
        ax.plot(n, [r[f"val_{key}_uniform"] for r in rows], "s--", color="C2", alpha=0.6, label="validation, uniform")
    ax.set_yscale("log")
    ax.set_xlabel("number of grid points n")
    ax.set_ylabel(key.replace("_", " "))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax
