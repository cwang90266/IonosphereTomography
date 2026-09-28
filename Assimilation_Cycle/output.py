#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output and diagnostics (plan Section 4.7/4.14): decode the final ensemble,
save it in ``EDPSamples``'s own netCDF schema, and plot it -- horizontal
(reusing ``EDPSamples.plot_horizontal_field`` directly, which already
accepts a raw ``(n_geo,)`` array), vertical (new -- forecast vs. analysis
ensemble mean +/- spread by altitude, since nothing in ``EDPSamples``/
``Parameterization`` plots that comparison), and the three ``metrics.py``
plots (RMSE reduction, rank histogram, effective-rank time series).

All figures use a headless (``Agg``) matplotlib backend, matching the
convention already used by every other test/plotting module in this
repository (``EDPSamples``/``Parameterization``/``observation_preparation``
conftest.py files).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass
import matplotlib.pyplot as plt

from Ensemble_Kalman_Engine import EnsembleState
from Ensemble_Kalman_Engine.driver import GeneralEnKFDriver

from .cycle_driver import CycleResult

# Per-geo_type attrs EDPSamples.__init__ needs to regenerate the same
# geolocation/mesh -- matches the attrs it stores for each branch
# (see edp_samples.py EDPSamples.__init__'s final `match geo_type` block).
_GEO_TYPE_ATTRS: dict[str, tuple[str, ...]] = {
    "Point": ("Lon", "Lat"),
    "Rectangle": ("minLon", "maxLon", "dLon", "minLat", "maxLat", "dLat"),
    "Polar": ("minLat", "dLat"),
    "Global": ("equal_spaced", "dLat", "dLon"),
    "Regional": ("Lon", "Lat", "radius", "dLat"),
}


def decode_ensemble(driver: GeneralEnKFDriver, ensemble: EnsembleState, obs_operator) -> np.ndarray:
    """``(n_height, n_geo, n_members)`` physical electron density (plan
    Section 4.7) -- thin wrapper around ``GeneralEnKFDriver.decode_to_edps``."""
    return driver.decode_to_edps(ensemble, obs_operator)


def save_decoded_field_netcdf(edp_samples_template, decoded_density: np.ndarray, path: str | Path):
    """
    Save a decoded ``(n_height, n_geo, n_members)`` density field as an
    ``EDPSamples``-schema netCDF (plan Section 4.7), reusing
    ``edp_samples_template``'s own grid (geolocation/mesh/altitude,
    regenerated via the same ``geo_type``-specific attrs it was built
    with, not copied by reference -- so this is a real, independent
    ``EDPSamples`` object, round-trippable through ``fromNetCDF`` and
    ``plot_horizontal_field``/``plot_geolocation`` like any other).

    ``feature_edps`` (foF2/hmF2/... -- normally IRI2020's own Fortran
    output, not derivable in Python from an arbitrary density field) is
    not recomputed for the assimilated field; the saved file is zero-filled
    there with ``attrs['feature_edps_computed'] = 0`` so this is explicit
    rather than silently misleading.
    """
    E = type(edp_samples_template)
    attrs = edp_samples_template.attrs
    geo_type = attrs["geo_type"]
    if geo_type not in _GEO_TYPE_ATTRS:
        raise ValueError(
            f"save_decoded_field_netcdf: geo_type={geo_type!r} not supported "
            f"(supported: {sorted(_GEO_TYPE_ATTRS)})"
        )
    geo_kwargs = {k: attrs[k] for k in _GEO_TYPE_ATTRS[geo_type]}

    n_height, n_geo, n_members = decoded_density.shape
    sampling_parameters = _import_pandas().DataFrame({"member": np.arange(n_members)})
    feature_edps = np.zeros((len(E.FEATURE_LABEL), n_geo, n_members))

    field = E(
        DateTime=attrs["DateTime"], geo_type=geo_type,
        altitude=np.asarray(edp_samples_template.altitude),
        sampling_parameters=sampling_parameters,
        edps=decoded_density, feature_edps=feature_edps,
        **geo_kwargs,
    )
    field.attrs["feature_edps_computed"] = 0

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field.saveNetCDF(path)
    return field


def _import_pandas():
    import pandas as pd
    return pd


def plot_horizontal(edp_samples_template, density_slice: np.ndarray, *, target_alt: float,
                     ax=None, **kwargs):
    """Thin wrapper around ``EDPSamples.plot_horizontal_field`` (plan
    Section 4.14 -- horizontal reuses the existing method as-is), for a
    caller-computed ``(n_geo,)`` array (e.g. an ensemble mean at one
    altitude) rather than a sample already stored in ``edp_samples_template``."""
    label = kwargs.pop("scalar_label", f"Electron Density (m$^{{-3}}$) [Alt: {target_alt:.1f} km]")
    return edp_samples_template.plot_horizontal_field(scalar=density_slice, ax=ax, scalar_label=label, **kwargs)


def plot_vertical_profile(
    altitude: np.ndarray,
    forecast_density: np.ndarray,
    analysis_density: np.ndarray,
    geo_idx: int = 0,
    ax=None,
    figsize=(6, 7),
):
    """
    New (plan Section 4.14): forecast ensemble mean +/- spread vs. analysis
    ensemble mean +/- spread, by altitude, at one horizontal grid point --
    distinct from ``Parameterized_EDPSamples.plot_reconstruction_profile``
    (parameterization reconstruction error, not forecast-vs-analysis).

    Parameters
    ----------
    altitude : ndarray, shape (n_height,)
    forecast_density, analysis_density : ndarray, shape (n_height, n_geo, n_members)
    geo_idx : int
        Horizontal grid point to plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    f = forecast_density[:, geo_idx, :]
    a = analysis_density[:, geo_idx, :]
    f_mean, f_std = f.mean(axis=1), f.std(axis=1)
    a_mean, a_std = a.mean(axis=1), a.std(axis=1)

    ax.plot(f_mean, altitude, color="C0", label="forecast mean")
    ax.fill_betweenx(altitude, f_mean - f_std, f_mean + f_std, color="C0", alpha=0.2, label="forecast +/-1 std")
    ax.plot(a_mean, altitude, color="C1", label="analysis mean")
    ax.fill_betweenx(altitude, a_mean - a_std, a_mean + a_std, color="C1", alpha=0.2, label="analysis +/-1 std")

    ax.set_xlabel("Electron density (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"Vertical profile (geo index {geo_idx})")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_rmse_reduction(cycle_result: CycleResult, ax=None, figsize=(7, 4)):
    """RMSE-reduction bar/line chart by batch (plan Section 4.14)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    batch_idx = [o.batch_index for o in cycle_result.batch_outcomes]
    forecast = [o.rmse_reduction.rmse_forecast for o in cycle_result.batch_outcomes]
    analysis = [o.rmse_reduction.rmse_analysis for o in cycle_result.batch_outcomes]

    ax.plot(batch_idx, forecast, "o-", color="C0", label="forecast RMSE")
    ax.plot(batch_idx, analysis, "o-", color="C1", label="analysis RMSE")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("RMSE (TECU)")
    ax.set_title("RMSE reduction by batch")
    ax.legend(loc="best")
    return ax


def plot_rank_histogram(cycle_result: CycleResult, ax=None, figsize=(6, 4)):
    """Rank histogram pooled across all batches in the cycle (plan Section 4.13/4.14)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    pooled = None
    for outcome in cycle_result.batch_outcomes:
        pooled = outcome.rank_histogram.copy() if pooled is None else pooled + outcome.rank_histogram

    ranks = np.arange(len(pooled))
    ax.bar(ranks, pooled, color="C2")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Count")
    ax.set_title("Rank histogram (pooled across batches)")
    return ax


def plot_effective_rank_series(cycle_result: CycleResult, ax=None, figsize=(7, 4)):
    """Effective ensemble rank (forecast/analysis) across batches (plan
    Section 4.13/4.14) -- monitors rank collapse/degeneracy over the cycle."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    batch_idx = [o.batch_index for o in cycle_result.batch_outcomes]
    forecast_rank = [o.effective_rank_forecast["rank"] for o in cycle_result.batch_outcomes]
    analysis_rank = [o.effective_rank_analysis["rank"] for o in cycle_result.batch_outcomes]

    ax.plot(batch_idx, forecast_rank, "o-", color="C0", label="forecast rank")
    ax.plot(batch_idx, analysis_rank, "o-", color="C1", label="analysis rank")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Effective ensemble rank")
    ax.set_title("Effective ensemble rank by batch")
    ax.legend(loc="best")
    return ax


# ===========================================================================
# Final_Packaging.docx visualizations (2026-09-25). Functions above return a
# single ``ax`` (one natural panel); the ones below need several panels at
# once (one per driving index / altitude / style metric), so they take
# ``fig=None`` and return ``fig`` instead -- documented per-function.
# ===========================================================================

_IRI_INPUT_COLUMNS = ("hour", "f107", "ap", "ig12", "rz12")


def plot_iri_input_distributions(sampling_parameters, fig=None, figsize=(11, 6)):
    """Histograms of the drawn IRI2020 driving indices (plan/docx step 3)
    -- one panel per column of ``iri_selection.sampling_parameters_for_cycle``'s
    return value. A column that's all-NaN (no spread requested for that
    index, Section 8.1's `pd.isna` convention) gets an empty panel
    labelled accordingly rather than a matplotlib error on an all-NaN
    histogram."""
    if fig is None:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
    else:
        axes = np.asarray(fig.axes).reshape(2, 3) if fig.axes else fig.subplots(2, 3)
    axes = np.asarray(axes).ravel()

    for ax, col in zip(axes, _IRI_INPUT_COLUMNS):
        values = sampling_parameters[col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            ax.text(0.5, 0.5, "no spread requested", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(finite, bins=min(30, max(5, finite.size // 5)), color="C0", edgecolor="black", linewidth=0.3)
        ax.set_title(col)
        ax.set_ylabel("count")
    for ax in axes[len(_IRI_INPUT_COLUMNS):]:
        ax.axis("off")

    fig.suptitle(f"IRI2020 input distributions (n={len(sampling_parameters)})")
    fig.tight_layout()
    return fig


def plot_observation_operator_sum(
    edp_samples, entry, altitudes=(100, 200, 300, 400, 500, 600, 700, 800),
    num_segments: int = 1000, fig=None, figsize=(14, 7),
):
    """Sum of the observation operator's rows over every ray in one RO/IGS
    ``entry``, shown on the horizontal grid at several altitudes (plan/docx
    step 5) -- "where on the grid does this occultation/arc actually have
    sensitivity, and at what height." One panel per altitude in
    ``altitudes`` (nearest available grid level used)."""
    H_entry = edp_samples.get_observation_operator(entry.to_operator_dict(), num_segments=num_segments)
    n_height = len(edp_samples.altitude)
    n_geo = edp_samples.geolocation.shape[0]
    summed = np.asarray(H_entry).sum(axis=0).reshape(n_height, n_geo)

    n_alt = len(altitudes)
    ncols = min(4, n_alt)
    nrows = int(np.ceil(n_alt / ncols))
    if fig is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                  subplot_kw={"projection": _cartopy_projection()})
    axes = np.asarray(fig.axes).ravel()

    for ax, target_alt in zip(axes, altitudes):
        alt_idx = int(np.argmin(np.abs(edp_samples.altitude - target_alt)))
        actual_alt = float(edp_samples.altitude[alt_idx])
        plot_horizontal(edp_samples, summed[alt_idx, :], target_alt=actual_alt, ax=ax,
                         scalar_label=f"sum(H) [{actual_alt:.0f} km]")
    for ax in axes[n_alt:]:
        ax.axis("off")

    fig.suptitle(f"Observation operator sum: {entry.label or entry.obs_type} ({entry.n_rays} rays)")
    fig.tight_layout()
    return fig


def _cartopy_projection():
    import cartopy.crs as ccrs
    return ccrs.PlateCarree()


def plot_tec_profile_comparison(entry, y_forecast: np.ndarray, y_analysis: np.ndarray,
                                 y_measured: np.ndarray, ax=None, figsize=(6, 6)):
    """Forecast vs. analysis vs. measured TEC for one RO occultation (plan/
    docx step 7), by tangent altitude -- ``y_forecast``/``y_analysis``/
    ``y_measured`` are already sliced to this entry's rays (e.g. via
    ``CycleBatch.entry_ray_ranges``), all shape ``(entry.n_rays,)``. RO-
    specific: uses ``entry.tangent_alt_km`` as the y-axis, so this isn't
    meaningful for an IGS arc (no tangent point)."""
    if entry.tangent_alt_km is None:
        raise ValueError(
            "plot_tec_profile_comparison: entry has no tangent_alt_km -- "
            "this plot is RO-specific (occultations have a tangent-altitude "
            "profile; IGS arcs don't)."
        )
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    order = np.argsort(entry.tangent_alt_km)
    alt = np.asarray(entry.tangent_alt_km)[order]

    ax.plot(np.asarray(y_measured)[order], alt, "o-", color="k", label="measured", markersize=3)
    ax.plot(np.asarray(y_forecast)[order], alt, "o-", color="C0", label="forecast", markersize=3)
    ax.plot(np.asarray(y_analysis)[order], alt, "o-", color="C1", label="analysis", markersize=3)

    ax.set_xlabel("TEC (TECU)")
    ax.set_ylabel("Tangent altitude (km)")
    ax.set_title(f"TEC profile: {entry.label or entry.obs_type}")
    ax.legend(loc="best", fontsize=8)
    return ax


def _interpolate_field_at_latlon(edp_samples, field: np.ndarray, lat: float, lon: float) -> np.ndarray:
    """Horizontally interpolate a ``(n_height, n_geo, ...)`` field at
    ``(lat, lon)`` using ``EDPSamples.interp``'s own mesh-locate machinery
    (the same barycentric interpolation ``get_observation_operator`` uses
    internally) -- returns shape ``(n_height, ...)``. Falls back to the
    nearest grid vertex if the point lies outside every mesh triangle
    (same fallback ``get_observation_operator`` uses for out-of-mesh ray
    points)."""
    positions = np.array([[lat, lon, float(edp_samples.altitude[0])]])
    _, _, idx_mesh, weight_mesh = edp_samples.interp(positions, coordinate="lla")
    tri_idx = int(idx_mesh[0])

    if tri_idx == -1:
        from scipy.spatial import cKDTree
        tree = cKDTree(edp_samples.geolocation)
        _, nearest = tree.query([[lon, lat]])
        return field[:, int(nearest[0]), ...]

    verts = edp_samples.mesh[tri_idx]
    w = weight_mesh[0]
    return w[0] * field[:, verts[0], ...] + w[1] * field[:, verts[1], ...] + w[2] * field[:, verts[2], ...]


def plot_edp_profile_comparison(
    edp_samples, entry, decoded_forecast: np.ndarray, decoded_analysis: np.ndarray,
    *, lat: float | None = None, lon: float | None = None,
    tangent_alt_window: tuple[float, float] = (250.0, 350.0),
    ax=None, figsize=(6, 7),
):
    """Forecast/analysis EDP vertical profile at a query location, overlaid
    with the Abel-retrieved profile (plan/docx step 8). If ``lat``/``lon``
    are omitted, derived from ``entry`` itself: the tangent point of the
    ray with maximum TEC among those whose tangent altitude falls in
    ``tangent_alt_window`` (default 250-350km, straddling the typical F2
    peak). Passing ``lat``/``lon`` explicitly instead makes this the same
    function the docx's planned future external-validation use reuses, at
    an arbitrary specified location rather than one derived from an RO.

    ``decoded_forecast``/``decoded_analysis`` : ``(n_height, n_geo, n_members)``.
    """
    if lat is None or lon is None:
        if entry.tangent_alt_km is None:
            raise ValueError("plot_edp_profile_comparison: entry has no tangent_alt_km; pass lat/lon explicitly.")
        window_mask = (
            (np.asarray(entry.tangent_alt_km) >= tangent_alt_window[0])
            & (np.asarray(entry.tangent_alt_km) <= tangent_alt_window[1])
        )
        if not window_mask.any():
            raise ValueError(
                f"plot_edp_profile_comparison: no rays with tangent_alt_km in "
                f"{tangent_alt_window} for entry {entry.label!r}; pass lat/lon explicitly."
            )
        candidate_idx = np.where(window_mask)[0]
        best = candidate_idx[np.argmax(np.asarray(entry.tec)[candidate_idx])]

        from edp_samples import EDPSamples as _E, _ecef_to_geodetic
        tangent_xyz, _, _ = _E.rayTangent(
            np.asarray(entry.rec_ecef_km)[:, best:best + 1],
            np.asarray(entry.gnss_ecef_km)[:, best:best + 1],
            units="km",
        )
        lat_arr, lon_arr, _ = _ecef_to_geodetic(tangent_xyz.T * 1000.0)
        lat, lon = float(lat_arr[0]), float(lon_arr[0])

    forecast_profile = _interpolate_field_at_latlon(edp_samples, decoded_forecast, lat, lon)
    analysis_profile = _interpolate_field_at_latlon(edp_samples, decoded_analysis, lat, lon)
    f_mean, f_std = forecast_profile.mean(axis=-1), forecast_profile.std(axis=-1)
    a_mean, a_std = analysis_profile.mean(axis=-1), analysis_profile.std(axis=-1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    altitude = np.asarray(edp_samples.altitude)
    ax.plot(f_mean, altitude, color="C0", label="forecast mean")
    ax.fill_betweenx(altitude, f_mean - f_std, f_mean + f_std, color="C0", alpha=0.2)
    ax.plot(a_mean, altitude, color="C1", label="analysis mean")
    ax.fill_betweenx(altitude, a_mean - a_std, a_mean + a_std, color="C1", alpha=0.2)

    if entry.abel and "Ne" in entry.abel and "alt_km" in entry.abel:
        ax.plot(entry.abel["Ne"], entry.abel["alt_km"], color="k", linestyle="--", label="Abel-retrieved")

    ax.set_xlabel("Electron density (m$^{-3}$)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"EDP profile at ({lat:.2f}, {lon:.2f})" + (f" -- {entry.label}" if entry.label else ""))
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_style_comparison_summary(results_by_style: dict, fig=None, figsize=(11, 8)):
    """Cross-style performance summary (plan/docx step 9): final RMSE,
    state dimension, wall-clock time, and convergence rate, one bar chart
    each. ``results_by_style`` maps style name -> ``(CycleResult, n_state,
    wall_time_s)``."""
    styles = list(results_by_style.keys())
    final_rmse = [
        results_by_style[s][0].batch_outcomes[-1].rmse_reduction.rmse_analysis
        if results_by_style[s][0].batch_outcomes else float("nan")
        for s in styles
    ]
    n_state = [results_by_style[s][1] for s in styles]
    wall_time = [results_by_style[s][2] for s in styles]
    convergence_rate = [
        (sum(o.diagnostics.converged for o in results_by_style[s][0].batch_outcomes)
         / max(len(results_by_style[s][0].batch_outcomes), 1))
        for s in styles
    ]

    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = np.asarray(fig.axes).ravel()

    for ax, values, title, ylabel in zip(
        axes,
        (final_rmse, n_state, wall_time, convergence_rate),
        ("Final analysis RMSE", "State dimension", "Wall-clock time", "Convergence rate"),
        ("RMSE (TECU)", "n_state", "seconds", "fraction of batches converged"),
    ):
        ax.bar(styles, values, color="C0")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Cross-style comparison")
    fig.tight_layout()
    return fig
